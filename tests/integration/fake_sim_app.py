#!/usr/bin/env python3
"""fake_sim_app.py — 顶替 Isaac 的假仿真，只实现 unix socket 协议。

用来做 L2 集成测试：不需要 GPU、不需要 Isaac，就能把
orchestrator → sim_expert → sim_bridge → success_detector → recorder
整条仿真采集链路跑起来。

它骗过下游靠三件事：

  1. 关节不做动力学。指令到达即视为到位，expert 的闭环容差马上满足；
     真要模拟 PD 跟随，expert 会多等好几个周期，测试时间翻倍。

  2. 物体用运动学代替物理。夹爪闭合且指尖靠近方块时方块吸附到夹爪；
     松开后直接落到杯心——真物理里这一步要靠摩擦和重力，假仿真里
     只关心"最后能不能判成功"。

  3. 图像按真值画。杯子画成圆环、方块画成方块，位置严格跟着真值走，
     所以 vision 判定和 ground_truth 判定不会互相矛盾。

协议常量与编解码全部复用 sim_app/protocol.py，避免两处维护。

    python3 tests/integration/fake_sim_app.py \
        --cmd /tmp/fake_cmd.sock --state /tmp/fake_state.sock
"""

from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "sim_app"))

import protocol as P  # noqa: E402

# 与 scene.yaml 的机器人配置保持一致：全零位、夹爪张开
HOME_Q = np.zeros(6, dtype=float)


class FakePlant:
    """极简「世界模型」：关节指令、夹爪开合、方块是否被抓住。

    约定与真机一致：

      - gripper 是 0~1，1 表示张开
      - 夹爪闭合（< 0.2）且指尖到方块中心小于 grab_radius 时吸附
      - 松开后方块落到杯心，z 落在杯口高度以下，ground_truth 判定为成功
    """

    def __init__(
        self,
        cup_xy=(0.30, -0.26),
        cup_center_z: float = 0.04,
        cup_height: float = 0.08,
        cube_edge: float = 0.025,
    ):
        self.joints = HOME_Q.copy()
        self.gripper = 1.0
        self.cup_xy = np.asarray(cup_xy, dtype=float)
        self.cup_center_z = float(cup_center_z)
        self.cup_height = float(cup_height)
        self.cube_edge = float(cube_edge)
        # 方块初始位置只是占位，第一次 place 会覆盖
        self.cube = np.array([0.31, -0.08, cube_edge / 2.0])
        self.attached = False
        # 上一拍的夹爪状态：靠边沿判断「刚闭合」「刚张开」
        self._prev_gripper = 1.0

    def command_joints(self, values) -> None:
        """吃一条关节指令：关节直接到位，再按夹爪边沿处理抓取/释放。

        values 是 [q1..q6, gripper]，gripper 1=张开、0=闭合，与真机
        /joint_commands 的约定一致。不模拟 PD 跟随：expert 每帧都在闭环等
        容差，延迟到位只会让它白等好几个控制周期。

        抓取和释放只看夹爪开度的边沿，不看指尖位置——那需要真 FK，而 expert
        的轨迹顺序是固定的（靠近 → 闭合 → 抬起 → 到位 → 张开），边沿足够区分。

        松开时直接给终态（方块落在杯底板上方），因为下游只关心最终能不能判
        成功，中间过程由真物理负责，假仿真不必复现。
        """
        q = np.asarray(values, dtype=float)
        if q.size != 7:
            return
        self.joints = q[:6].copy()
        gripper = float(q[6])
        self.gripper = gripper

        if gripper < 0.2 and self._prev_gripper >= 0.2:
            self.attached = True
        elif gripper > 0.8 and self._prev_gripper <= 0.8 and self.attached:
            self.attached = False
            self.cube = np.array(
                [
                    self.cup_xy[0],
                    self.cup_xy[1],
                    self.cup_center_z - self.cup_height / 2.0 + self.cube_edge / 2.0,
                ]
            )
            print(
                f"[fake] release -> cube={np.round(self.cube, 4).tolist()}"
                f" (prev={self._prev_gripper:.2f} g={gripper:.2f})",
                flush=True,
            )

        self._prev_gripper = gripper

    # ------------------------------------------------------------ 场景操作
    def place(self, x: float, y: float, z: float = 0.0) -> None:
        """摆方块 + 机械臂回零位，对应真机 /sim/place 的原子语义。

        z 传 0 表示「贴在桌面上」，用边长的一半兜底：orchestrator 一般会
        把 task.yaml 里的 cube.center.z 原样传进来。
        """
        cz = float(z) if float(z) > 0.0 else self.cube_edge / 2.0
        self.cube = np.array([float(x), float(y), cz])
        self.attached = False
        self.joints = HOME_Q.copy()
        self.gripper = 1.0
        self._prev_gripper = 1.0
        print(
            f"[fake] place({float(x):.3f},{float(y):.3f},{float(z):.3f})"
            f" -> cube={np.round(self.cube, 4).tolist()}",
            flush=True,
        )

    def setup(self, cup_x: float, cup_y: float, cup_z: float) -> None:
        """同步杯子位置。cup_z 是杯体中心高度，不是杯口。"""
        self.cup_xy = np.array([float(cup_x), float(cup_y)])
        self.cup_center_z = float(cup_z)

    def truth(self) -> tuple[list[float], list[float]]:
        """按协议给真值：cup 的 z 是杯口高度，cube 的 z 是几何中心。"""
        cup = [
            self.cup_xy[0],
            self.cup_xy[1],
            self.cup_center_z + self.cup_height / 2.0,
            self.cup_height,
        ]
        cube = [self.cube[0], self.cube[1], self.cube[2], self.cube_edge]
        return cup, cube

    # ------------------------------------------------------------ 渲染
    def render(self, width: int, height: int) -> np.ndarray:
        """把杯子和方块按世界坐标画成一张俯视图。

        不是真投影：测试用 ground_truth 判定，图像只需要「位置对得上、
        看得出是什么」，让 recorder 有真实的像素可录、事后能肉眼核对。
        比例尺 900 px/m 接近 front 相机在 1280 宽下的实际覆盖。
        """
        img = np.empty((height, width, 3), dtype=np.uint8)
        img[:] = (188, 186, 180)

        scale = 900.0
        center = (0.33, -0.10)

        def to_px(x, y):
            return (
                width / 2.0 + (x - center[0]) * scale,
                height / 2.0 - (y - center[1]) * scale,
            )

        u, v = to_px(self.cup_xy[0], self.cup_xy[1])
        r_out = self.cup_height * 0.375 * scale
        r_in = r_out * 0.7
        _fill_disc(img, u, v, r_out, (200, 40, 70))
        _fill_disc(img, u, v, r_in, (150, 148, 143))

        u, v = to_px(self.cube[0], self.cube[1])
        half = self.cube_edge / 2.0 * scale
        _fill_rect(img, u, v, max(half, 1.5), (12, 12, 12))
        return img


def _fill_disc(img: np.ndarray, u: float, v: float, r: float, color) -> None:
    """在局部窗口里画实心圆，避免整幅图的网格运算。"""
    h, w = img.shape[:2]
    y0, y1 = max(0, int(v - r) - 1), min(h, int(v + r) + 2)
    x0, x1 = max(0, int(u - r) - 1), min(w, int(u + r) + 2)
    if y1 <= y0 or x1 <= x0:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - u) ** 2 + (yy - v) ** 2 <= r * r
    img[y0:y1, x0:x1][mask] = color


def _fill_rect(img: np.ndarray, u: float, v: float, half: float, color) -> None:
    h, w = img.shape[:2]
    y0, y1 = max(0, int(v - half)), min(h, int(v + half) + 1)
    x0, x1 = max(0, int(u - half)), min(w, int(u + half) + 1)
    if y1 > y0 and x1 > x0:
        img[y0:y1, x0:x1] = color


class FakeSimServer:
    """监听两条 unix socket，扮演跟 sim_app 一样的角色。

    bridge 是主动连接方：cmd 上只写，state 上只读，所以这里在 cmd 上收
    定长帧、在 state 上按 30Hz 推帧。setup/place 的确认必须走 state，
    且 type=4 物体帧要排在 type=5 之前（同 socket FIFO）。
    """

    def __init__(
        self,
        cmd_path: str,
        state_path: str,
        plant: FakePlant,
        width: int = 1280,
        height: int = 720,
        joints_hz: float = 30.0,
        images_hz: float = 15.0,
    ):
        self.cmd_path = cmd_path
        self.state_path = state_path
        self.plant = plant
        self.width, self.height = int(width), int(height)
        self.joints_period = 1.0 / float(joints_hz)
        self.images_period = 1.0 / float(images_hz)
        # 683 ≈ 1280 宽下 86° 水平视场，和 scene.yaml 的 front 一致
        self.K = [683.0, 0.0, self.width / 2, 0.0, 683.0, self.height / 2, 0.0, 0.0, 1.0]
        self.D = [0.0] * 5

        self.running = False
        self._stop = threading.Event()
        self._state_conn: socket.socket | None = None
        self._conn_lock = threading.Lock()
        self._listeners: list[socket.socket] = []
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------ 生命周期
    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self.running = True
        self._listeners = [
            _listen(self.cmd_path),
            _listen(self.state_path),
        ]
        self._threads = [
            _spawn(self._accept_cmd_loop, self._listeners[0], name="fake-accept-cmd"),
            _spawn(self._accept_state_loop, self._listeners[1], name="fake-accept-state"),
            _spawn(self._publish_loop, name="fake-publish"),
        ]

    def stop(self) -> None:
        if not self.running:
            return
        self.running = False
        self._stop.set()
        for sock in self._listeners:
            try:
                sock.close()
            except OSError:
                pass
        with self._conn_lock:
            conn, self._state_conn = self._state_conn, None
        if conn is not None:
            try:
                conn.close()
            except OSError:
                pass
        for t in self._threads:
            t.join(timeout=2.0)
        for path in (self.cmd_path, self.state_path):
            try:
                os.unlink(path)
            except OSError:
                pass

    # ------------------------------------------------------------ 连接处理
    def _accept_cmd_loop(self, listener: socket.socket) -> None:
        while not self._stop.is_set():
            conn = _accept(listener, self._stop)
            if conn is None:
                continue
            self._read_cmds(conn)

    def _accept_state_loop(self, listener: socket.socket) -> None:
        """每来一个 state 连接就单独起线程，accept 本身不被占用。

        bridge 断线重连时新连接必须能立刻接上；串行处理会让重连一直堵在
        backlog 里，表现为「连上了但永远没有数据」。
        """
        while not self._stop.is_set():
            conn = _accept(listener, self._stop)
            if conn is None:
                continue
            with self._conn_lock:
                old, self._state_conn = self._state_conn, conn
            if old is not None:
                try:
                    old.close()
                except OSError:
                    pass
            _spawn(self._hold_state, conn, name="fake-state-conn")

    def _read_cmds(self, conn: socket.socket) -> None:
        with conn:
            while not self._stop.is_set():
                try:
                    frame = _recv_exact(conn, P.CMD_SIZE)
                except OSError:
                    return
                if frame is None:
                    return
                try:
                    kind, values = P.parse_cmd(frame)
                except ValueError:
                    # 流已经错位，继续读只会拿到垃圾；丢连接让 bridge 重连
                    return
                self._dispatch(kind, values)

    def _hold_state(self, conn: socket.socket) -> None:
        """state 通道 bridge 只读不写，这里只是占住连接直到对端断开。

        不能给这条连接设 timeout：publish 线程在同一个 socket 上 sendall
        2.7MB 的图像帧，超时会被当成断线，连接随即被丢掉。退出靠 stop()
        关闭连接来唤醒阻塞的 recv。
        """
        try:
            while not self._stop.is_set():
                if not conn.recv(64):
                    return
        except OSError:
            return
        finally:
            with self._conn_lock:
                if self._state_conn is conn:
                    self._state_conn = None
            try:
                conn.close()
            except OSError:
                pass

    def _dispatch(self, kind: int, values) -> None:
        if kind == P.CMD_JOINTS:
            self.plant.command_joints(values)
            return
        if kind == P.CMD_SETUP:
            self.plant.setup(values[0], values[1], values[2])
            self._ack(int(values[3]), True)
            return
        if kind == P.CMD_PLACE:
            self.plant.place(values[0], values[1], values[2])
            self._ack(int(values[3]), True)

    def _ack(self, req_id: int, success: bool) -> None:
        """先发新真值再发完成帧：下游收到 success 时位姿一定已更新。

        两帧合并成一次 sendall：中途被图像帧插进来虽然不错位，但会把
        「type=4 紧接 type=5」这条约定变成概率事件，调试时很难看出来。
        """
        stamp = time.time()
        cup, cube = self.plant.truth()
        self._send(
            P.frame_object(cup, cube, stamp) + P.frame_done(req_id, success, stamp)
        )

    # ------------------------------------------------------------ 周期推送
    def _publish_loop(self) -> None:
        next_joints = time.monotonic()
        next_images = next_joints
        next_info = next_joints
        next_object = next_joints
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_joints:
                stamp = time.time()
                self._send(
                    P.frame_joints(
                        [*self.plant.joints.tolist(), self.plant.gripper], stamp
                    )
                )
                next_joints += self.joints_period
                if next_joints < now:
                    next_joints = now + self.joints_period

            if now >= next_images:
                stamp = time.time()
                # 两路共用一张图：假仿真里视角差异对判定没有影响，
                # 画两次只会让 1280x720 的字节流多传一倍
                rgb = self.plant.render(self.width, self.height)
                self._send(P.frame_image(P.CAM_FRONT, rgb, stamp))
                self._send(P.frame_image(P.CAM_WRIST, rgb, stamp))
                next_images += self.images_period
                if next_images < now:
                    next_images = now + self.images_period

            if now >= next_info:
                stamp = time.time()
                self._send(P.frame_info(P.CAM_FRONT, self.K, self.D, stamp))
                self._send(P.frame_info(P.CAM_WRIST, self.K, self.D, stamp))
                next_info = now + 1.0

            if now >= next_object:
                # 必须周期发：detector 的 ground_truth 判据读 /object/pose 的
                # 当前值，只在 place 时发一次的话，方块被抓起放进杯子之后
                # 它拿到的还是旧位置，永远判不出成功。
                stamp = time.time()
                cup, cube = self.plant.truth()
                self._send(P.frame_object(cup, cube, stamp))
                next_object = now + 0.1

            time.sleep(0.002)

    def _send(self, frame: bytes) -> None:
        # sendall 必须在锁内：publish 线程和 cmd 线程会同时写同一条 socket，
        # 只锁「取 conn」的话两次 sendall 的字节会交错，客户端按头切帧立刻错位
        with self._conn_lock:
            conn = self._state_conn
            if conn is None:
                return
            try:
                conn.sendall(frame)
            except OSError:
                self._state_conn = None


# ---------------------------------------------------------------- socket 工具
def _listen(path: str) -> socket.socket:
    """建监听 socket；残留的 socket 文件要先删，否则 bind 报 Address in use。"""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(path)
    sock.listen(4)
    return sock


def _accept(listener: socket.socket, stop: threading.Event) -> socket.socket | None:
    """带超时的 accept：不然 stop() 时线程会卡在 accept 上退不出来。"""
    listener.settimeout(0.2)
    while not stop.is_set():
        try:
            conn, _ = listener.accept()
            return conn
        except socket.timeout:
            continue
        except OSError:
            return None
    return None


def _recv_exact(conn: socket.socket, size: int) -> bytes | None:
    """读满 size 字节；对端关闭返回 None。

    不能假设一次 recv 就拿到一整帧：unix socket 是字节流，64 字节的 cmd
    也可能被拆成两段到达。
    """
    buf = b""
    while len(buf) < size:
        chunk = conn.recv(size - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def _spawn(target, *args, name: str) -> threading.Thread:
    t = threading.Thread(target=target, args=args, name=name, daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------- 入口
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="假仿真：顶替 Isaac 跑 L2 集成测试")
    parser.add_argument("--cmd", default="/tmp/fake_cmd.sock")
    parser.add_argument("--state", default="/tmp/fake_state.sock")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--joints-hz", type=float, default=30.0)
    parser.add_argument("--images-hz", type=float, default=15.0)
    args = parser.parse_args(argv)

    server = FakeSimServer(
        args.cmd,
        args.state,
        FakePlant(),
        width=args.width,
        height=args.height,
        joints_hz=args.joints_hz,
        images_hz=args.images_hz,
    )
    server.start()
    print(
        f"fake_sim_app 就绪：cmd={args.cmd} state={args.state} "
        f"{args.width}x{args.height}",
        flush=True,
    )
    try:
        while server.running:
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
