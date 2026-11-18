#!/usr/bin/env python3
"""sim_app 的进程入口：起 Isaac Sim，跑主循环，和 bridge 交换数据。

这个进程不 import ROS。仿真数据（关节、图像、相机内参、物体真值）走一条
unix socket 发给 bridge，bridge 发来的关节/摆放指令走另一条收进来，两边
的字节格式全部定义在 protocol.py。

本文件必须由 AppLauncher 拉起后才 import 场景模块，所以 isaaclab 相关的
import 都放在函数内部，模块顶层只留标准库和 protocol。
"""

from __future__ import annotations

import os
import queue
import socket
import stat
import threading
import time

import protocol


class FrameServer:
    """两条 unix socket 的搬运工：收 cmd、发 state，断线自己重连。

    连接方向是 bridge 主动连，这里只负责 listen/accept，绝不主动 connect；
    bridge 侧每秒重试一次，所以 sim_app 晚启动也能接上，反过来也一样。

    两条通道各有一个监听 socket，各自接受一个客户端：
    - cmd 通道只读，收到的每条 64 字节帧解析后塞进 cmd_q，由主线程消费；
    - state 通道只写，帧先进 state_q，再由发送线程按序写出。
    分两个线程是为了收发互不阻塞：state 写不出去时不能连带把 cmd 卡死。
    """

    def __init__(self, cfg: dict):
        """建两条监听通道和两个队列，不启动任何线程。

        先 bind 再启动线程是有意的：bridge 可能比 sim_app 先起来，监听在
        构造阶段就绪以后，它随时连进来都不会吃 ECONNREFUSED。

        绑定前要删掉残留的 socket 文件——上次退出如果是被 kill 的，文件还在，
        直接 bind 会 EADDRINUSE；unix socket 文件不是普通文件，不能复用它。

        cmd_q 不限长（每秒最多几十条，主线程每步都抽干），state_q 限长，
        满了由 send 负责丢帧——不在这里丢，是因为「丢图像、保关节和真值」
        要看帧类型，__init__ 不该碰 payload。
        """
        sock_cfg = cfg["socket"]
        self.cmd_path = sock_cfg["cmd"]
        self.state_path = sock_cfg["state"]

        self.cmd_q: queue.Queue = queue.Queue()
        self.state_q: queue.Queue = queue.Queue(maxsize=int(cfg["sim"]["state_queue"]))

        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

        self._cmd_listener = self._listen(self.cmd_path)
        self._state_listener = self._listen(self.state_path)

    def _listen(self, path: str) -> socket.socket:
        """在 path 上建 AF_UNIX 监听 socket，返回值由调用方 close。

        listen(1) 就够：每条通道只允许一个 bridge 客户端。第二个连接排队等
        着没有意义，反而会在旧连接没断干净时让人误以为连上了。

        上次被 kill 或崩溃退出时会留下 socket 文件，直接 bind 会 EADDRINUSE，
        所以先删。只删确实属于 socket 的那个文件：路径上要是普通文件，说明
        配置指错了地方，宁可让 bind 报错，也不能把用户的数据删掉。
        """
        try:
            if stat.S_ISSOCK(os.stat(path).st_mode):
                os.unlink(path)
        except FileNotFoundError:
            pass

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.bind(path)
            sock.listen(1)
        except OSError:
            # bind 失败时 socket 还没注册到任何地方，但不关就是 fd 泄漏
            sock.close()
            raise
        return sock

    def start(self) -> None:
        """起收、发两个后台线程，重复调用直接返回。

        线程不下发任何回调，主线程只通过两个队列和它们交互：cmd 线程往里
        放命令，主线程取；主线程往里放 state 帧，state 线程取。这样 socket
        永远只被拥有它的那个线程读写，不需要加锁。

        daemon=True：关窗口时主循环直接返回，不用等 accept/recv 自己醒过来；
        这些线程就算被硬切掉也不会有未写完的状态要保存。
        """
        if self._threads:
            return

        for target, name in (
            (self._serve_cmd, "sim-cmd"),
            (self._serve_state, "sim-state"),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

    def stop(self) -> None:
        """让两个后台线程退出，并把 socket 文件从文件系统删掉。

        只置停止位不够：线程可能正卡在 accept/recv 上。监听 socket 用短
        timeout（见 _serve_* 的 accept 循环），所以最多等一个 timeout 就会
        自己检查 _stop 并退出；join 再给 1 秒兜底，万一对端不读数据、线程
        卡在 sendall 上，也不会把退出流程拖死——daemon 线程最终随进程结束。

        socket 文件必须手动删：close() 不删文件，留着下次 bind 就是
        EADDRINUSE（虽然 _listen 也会删，但退出时清掉更干净，别的进程看到
        文件消失就知道 sim_app 已经不在）。
        """
        self._stop.set()

        for listener in (self._cmd_listener, self._state_listener):
            try:
                listener.close()
            except OSError:
                pass

        for thread in self._threads:
            thread.join(timeout=1.0)
        self._threads.clear()

        for path in (self.cmd_path, self.state_path):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def send(self, frame: bytes) -> bool:
        """把一帧编码好的 state 交给发送线程，队列满时按类型决定丢谁。

        图像是 30Hz 的连续流，丢一帧只是掉帧，直接丢掉；关节、内参、物体
        真值和完成帧丢了会破坏语义——bridge 的服务要一直等到超时，而且
        type=4 必须排在 type=5 前面。所以这几类满了就先腾掉队首最旧的一帧
        再入队：队列里全是旧数据，保新不保旧。

        只在关键帧也塞不进去时打日志，图像掉帧逐帧打会把终端刷满。
        返回 True 表示已入队，不代表已经写进 socket。
        """
        try:
            self.state_q.put_nowait(frame)
            return True
        except queue.Full:
            pass

        ftype = protocol.HDR.unpack_from(frame)[1]
        if ftype == protocol.T_IMAGE:
            return False

        try:
            self.state_q.get_nowait()
            self.state_q.put_nowait(frame)
            return True
        except (queue.Empty, queue.Full):
            print(f"[sim_app] state 队列已满，丢弃 type={ftype} 帧", flush=True)
            return False

    def _serve_cmd(self) -> None:
        """cmd 通道：等 bridge 连进来，把后续字节流按帧解进 cmd_q。

        连接方向是 bridge 主动连，所以这里只 accept。连接断了或者帧流错位
        （parse_cmd 抛异常）都只能整条连接重来：定长帧没有重同步的办法，
        硬着头皮接着读只会把后面每一帧都解错。

        accept 用 0.5s 超时轮询 _stop。Linux 上 close() 不保证唤醒另一个
        线程里阻塞的 accept，靠超时才保证 stop() 之后线程一定能退出。

        命令只入队、不在这里执行：物理步进必须在主线程做，跨线程碰 PhysX
        和渲染会出问题。
        """
        self._cmd_listener.settimeout(0.5)

        while not self._stop.is_set():
            try:
                conn, _ = self._cmd_listener.accept()
            except TimeoutError:
                continue
            except OSError:
                # stop() 把监听 socket 关了，属于正常退出
                return

            print(f"[sim_app] cmd 通道已连接 {self.cmd_path}", flush=True)
            try:
                self._recv_cmd(conn)
            except (OSError, ValueError) as e:
                if not self._stop.is_set():
                    print(f"[sim_app] cmd 通道断开：{e}", flush=True)
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def _recv_cmd(self, conn: socket.socket) -> None:
        """从一条 cmd 连接里按 64 字节切帧，解析后放进 cmd_q。

        recv 可能只给半帧（流式 socket 不保证一次读满），所以用缓冲区攒够
        一帧才解析；解析成功立刻清空缓冲区，不留任何跨帧状态。

        recv 设 0.5s 超时时不时检查 _stop，行为跟 accept 一致。超时发生在
        半帧中间时缓冲区必须保留，否则下次读到的前几个字节会被当成帧头。

        对端正常关闭时 recv 返回 b""，直接返回让 _serve_cmd 去等重连；
        parse_cmd 抛的异常交给调用方，因为帧流已经错位，只能重连。
        """
        conn.settimeout(0.5)
        buf = b""

        while not self._stop.is_set():
            try:
                chunk = conn.recv(protocol.CMD_SIZE - len(buf))
            except TimeoutError:
                continue
            if not chunk:
                return

            buf += chunk
            if len(buf) < protocol.CMD_SIZE:
                continue

            kind, values = protocol.parse_cmd(buf)
            self.cmd_q.put((kind, values))
            buf = b""

    def _serve_state(self) -> None:
        """state 通道：等 bridge 连进来，把 state_q 里的帧按序写出去。

        这条和 cmd 相反，只写不读。accept 的 0.5s 超时同样是为了响应 stop。

        连接建立时先抽干 state_q：断线期间攒的关节角和图像都带着过去的
        时间戳，一次灌给 bridge 只会让下游处理一堆过期数据；全丢掉，从下一
        个新帧开始发。重连瞬间最多缺一个物理步，下游只看到 joint_age 短暂
        变大，比收到旧数据好诊断。

        sendall 保证一帧要么全写出、要么抛异常；对端关闭时抛 BrokenPipeError
        （OSError 子类），按断线处理回到 accept。
        """
        self._state_listener.settimeout(0.5)

        while not self._stop.is_set():
            try:
                conn, _ = self._state_listener.accept()
            except TimeoutError:
                continue
            except OSError:
                # stop() 关掉了监听 socket
                return

            while True:
                try:
                    self.state_q.get_nowait()
                except queue.Empty:
                    break

            print(f"[sim_app] state 通道已连接 {self.state_path}", flush=True)
            try:
                self._send_state(conn)
            except OSError as e:
                if not self._stop.is_set():
                    print(f"[sim_app] state 通道断开：{e}", flush=True)
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def _send_state(self, conn: socket.socket) -> None:
        """把 state_q 里的帧按入队顺序写进 conn，直到停止或对端断开。

        get 用 0.5s 超时而不是死等：没有新帧的时候也要定期回到循环顶部
        检查 _stop，不然 stop() 就只能靠 daemon 属性强杀。

        连接本身不设写超时（accepted socket 默认是阻塞的）：一帧最多约
        1MB，本机 unix socket 正常是微秒级。对端要是真的不读，宁可让
        sendall 一直阻塞，也不能把半帧写出去——半帧会直接毁掉对端的帧流，
        而阻塞至少还能靠进程退出收敛。
        """
        while not self._stop.is_set():
            try:
                frame = self.state_q.get(timeout=0.5)
            except queue.Empty:
                continue
            conn.sendall(frame)


def send_camera_info(server: FrameServer, cfg: dict) -> None:
    """把两路相机的内参编码成 type=3 帧，交给 state 通道。

    K 按针孔模型从配置算：fx = width * focal_length / horizontal_aperture，
    像素是正方形所以 fy = fx，主点取画面中心（Omniverse 相机的约定），
    不是常用的 (w-1)/2——两边差半个像素，下游查表会直接对不上。
    D 全零：仿真相机没有畸变。

    bridge 只是把内参缓存起来，随每帧图像一起转发；它要是比 sim_app 晚
    启动就会错过启动时发的那一次，所以主循环需要隔一段时间重发，成本是
    每帧 14 个 double。
    """
    stamp = time.time()

    for cam, name in ((protocol.CAM_FRONT, "front"), (protocol.CAM_WRIST, "wrist")):
        c = cfg["camera"][name]
        width = int(c["width"])
        height = int(c["height"])
        fx = width * float(c["focal_length"]) / float(c["horizontal_aperture"])

        k = (
            fx, 0.0, width / 2.0,
            0.0, fx, height / 2.0,
            0.0, 0.0, 1.0,
        )
        d = (0.0, 0.0, 0.0, 0.0, 0.0)
        server.send(protocol.frame_info(cam, k, d, stamp))
