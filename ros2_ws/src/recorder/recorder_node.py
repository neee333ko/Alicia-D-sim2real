#!/usr/bin/env python3
"""recorder_node.py — 把一次试次的观测与动作写成 LeRobot 数据集。

录制不归 orchestrator 管：本节点订阅 /orch/trial 的 begin/end 自己开关窗口，
订阅 /orch/result 拿成功与否，决定这条 episode 进主数据集还是失败集。

数据流（两路异步 → 一帧）：

    /joint_states                    60Hz  7 维
    /camera/{front,wrist}/image_raw  30Hz  rgb8
                  ↓ FrameAligner（以图像为锚，state 取相邻对）
    frame{observation.state = q[t], action = q[t+1], observation.images.*}

action 不取 /joint_commands：真机拖动示教没有指令流，而且部署时执行器收到的
就是「关节位置」本身。统一取「下一时刻实际到达的位置」，仿真与真机同分布。

写盘分两段：begin 后先落 staging 目录（jpg + npy），end 拿到 result 再回灌进
LeRobotDataset。原因：lerobot 的 add_frame 没有回滚，而 success 要到 end 之后
才知道，必须先在 staging 里等结果。
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String

# 数据集 action/state 的列顺序。按名字取，不信消息里的顺序
JOINTS = ("Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6", "gripper")


class FrameAligner:
    """把两路异步消息拼成训练用的一帧，锚点是图像。

    图像 30Hz 决定采样节拍；每个采样点要两帧关节状态：

        observation.state = q[t]     与图像时刻最接近的那条
        action            = q[t+1]   紧随其后的一条

    「下一时刻」由 60Hz 的 /joint_states 提供，所以出帧总是滞后一条 state
    （约 17ms）。试次收尾时补不齐就丢掉最后半帧，不重复上一帧充数。

    图像只留最新一帧（30Hz 与采样节拍同频），state 留最近 0.3s 的滚动窗口：
    缓冲的意义是让迟到的图按时间戳找回自己的相邻对，而不是拿最新两条硬配，
    把「图像与 action 差了几百毫秒」的错配掩盖过去。
    """

    def __init__(self, image_keys=("front", "wrist"), tol_s: float = 0.05):
        """tol_s 只用来判断两路相机是否同步（是否有一路掉帧）。

        取 0.05s ≈ 1.5 个 30Hz 图像周期：正常相机之间的抖动远小于它，而真的
        掉了一帧就会超过。图像传输延迟不靠这里兜底——迟到几百毫秒的图由
        states 滚动缓冲按时间戳找回自己的相邻对，pair 不上才丢。
        """
        self.image_keys = tuple(image_keys)
        self.tol_s = float(tol_s)
        # 关节状态滚动缓冲（时间升序），长度按 time 裁剪：
        # 图像晚到时能回头找到真正夹住自己的那对 state，而不是拿最新的硬配。
        # 0.3s 是能容忍的图像延迟上限，也够覆盖真机 MJPG 的传输抖动。
        self.history_s = 0.3
        self.states: deque[tuple[float, np.ndarray]] = deque()
        # 相机 key -> (stamp, HxWx3 uint8)
        self.images: dict[str, tuple[float, np.ndarray]] = {}
        # 丢帧分类计数，挂到 /recorder/status 上，相机掉帧一眼能看出来
        self.dropped_missing = 0
        self.dropped_stale = 0
        # 已处理到的采样时刻（图像 stamp）：出帧、丢帧都会推进它。
        # 图像回调和 state 回调都会尝试出帧，靠它保证同一拍只处理一次。
        self.last_emitted: float | None = None
        # 等右邻 state 的图（stamp + 当时的 images 快照）。单槽足够：同一时刻
        # 只会有一张图卡在「右邻还没来」上，再多说明上游已经乱了。
        self.pending: tuple[float, dict[str, tuple[float, np.ndarray]]] | None = None

    def push_state(self, stamp: float, values) -> None:
        """记下一条关节状态，滚动保留最近 history_s 秒。

        为什么不是一个只有两条的窗口：图像要经 socket/DDS 传到本节点，晚到
        几十到上百毫秒是常态。只留两条时，能夹住迟到图像的那对 state 早被挤
        掉，只能拿最新两条硬配，动作就滞后了（之前把 tol_s 放宽到 150ms 就是
        在掩盖这个问题）。留 0.3s 后，迟到的图仍能配上它自己的相邻对。

        时间倒退（系统时钟被校时、上游乱序）时清空重来：否则缓冲里出现负的
        时间跨度，配对会算出错误的夹逼区间。
        """
        arr = np.asarray(values, dtype=np.float32)
        if self.states and stamp < self.states[-1][0]:
            self.states.clear()
        self.states.append((float(stamp), arr))

        # 至少留两条：只留一条时连「左邻 + 右邻」都凑不出来
        horizon = float(stamp) - self.history_s
        while len(self.states) > 2 and self.states[0][0] < horizon:
            self.states.popleft()

    def push_image(self, key: str, stamp: float, rgb) -> None:
        """记下某台相机的最新一帧。

        rgb 是 HxWx3 的 uint8，由调用方从 sensor_msgs/Image 的 data 重排好
        再传进来。这里只留最新一帧：图像 30Hz 与采样节拍同频，留多份没有
        收益，反而会在配对时选错时刻。
        """
        if key not in self.image_keys:
            return
        self.images[key] = (float(stamp), np.asarray(rgb, dtype=np.uint8))

    def try_frame(self) -> dict | None:
        """凑齐就吐一帧，否则 None。图像回调和 state 回调都会调用它。

        采样时刻优先取 pending（上一拍等右邻 state 没等到、被暂存下来的图），
        没有 pending 才取最新一张。然后在 state 缓冲里找真正夹住采样时刻的
        相邻一对（左邻 <= t_img <= 右邻）：左邻当 observation.state、右邻当
        action。不再拿最新两条硬配，迟到 200ms 的图也能配上它自己的那一对。

        右邻还没到（t_img 比最新 state 还新）时把这一拍存进 pending，等 state
        回调再配；图已经老过缓冲下界说明迟到超过 history_s，才计数丢弃。
        tol_s 只用来判断两路相机是否同步，不参与 state 配对。

        出帧后把水位推到「当前所有图里最新的 stamp」：两台相机各自触发时，稍
        晚的那台不会再把同一对图像出一遍，避免相隔几毫秒的重复样本。
        """
        if self.pending is not None:
            t_img, images = self.pending
        elif self.images:
            t_img = max(stamp for stamp, _ in self.images.values())
            images = self.images
        else:
            return None

        if self.last_emitted is not None and t_img <= self.last_emitted:
            # 这一拍已经处理过：pending 残留清掉，新图等下一拍再配
            self.pending = None
            return None

        if len(images) < len(self.image_keys):
            self.dropped_missing += 1
            self.last_emitted = t_img
            self.pending = None
            return None

        if any(t_img - stamp > self.tol_s for stamp, _ in images.values()):
            self.dropped_stale += 1
            self.last_emitted = t_img
            self.pending = None
            return None

        states = list(self.states)
        pair = None
        for left, right in zip(states, states[1:]):
            if left[0] <= t_img <= right[0]:
                pair = (left, right)
                break

        if pair is None:
            if not states or t_img > states[-1][0]:
                # 右邻还没来（或 state 才起头）：存起来，等 state 回调再配
                if self.pending is None:
                    self.pending = (t_img, dict(images))
                return None
            # 图比缓冲里最老的 state 还老：迟到超过 history_s，配不上了
            self.dropped_stale += 1
            self.last_emitted = t_img
            self.pending = None
            return None

        left, right = pair
        frame = {
            "stamp": t_img,
            "state": left[1],
            "action": right[1],
            "images": {k: img for k, (_, img) in images.items()},
        }
        self.last_emitted = max(
            [t_img] + [stamp for stamp, _ in self.images.values()]
        )
        self.pending = None
        return frame


class EpisodeStaging:
    """把一集的原始帧落到磁盘，等 result 到了再决定灌进哪个数据集。

    为什么不在内存里攒：1280x720 两路 15s 就是 2.4 GB，连跑几条必 OOM。
    lerobot 的 add_frame 又没有回滚，success 要等 end 之后才知道，所以只能
    先把原始帧写到磁盘，拿到结论再回灌。

    目录结构（每集一个）：

        <staging>/ep_000012/
            states.npy          (N, 7) float32
            actions.npy         (N, 7) float32
            stamps.npy          (N,)   float64
            front/000000.jpg    ...
            wrist/000000.jpg
    """

    def __init__(self, root: str, ep_id: int, image_keys=("front", "wrist")):
        self.dir = os.path.join(root, f"ep_{int(ep_id):06d}")
        self.ep_id = int(ep_id)
        self.image_keys = tuple(image_keys)
        # 上一轮异常退出可能留下同名目录，先清掉，别让两集的帧混在一起
        if os.path.isdir(self.dir):
            shutil.rmtree(self.dir)
        for key in self.image_keys:
            os.makedirs(os.path.join(self.dir, key), exist_ok=True)
        self.n_frames = 0
        # 7 维向量 × 每秒 30 帧 × 15 s 只有几 KB，留在内存最后一次性落盘
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._stamps: list[float] = []

    def write(self, frame: dict) -> None:
        """把一帧落盘：每台相机一张 jpg，state/action/stamp 留在内存。

        编码用 cv2.imencode 的同步路径：单张 4~8ms，两路也在 30Hz 的 33ms
        预算内，不需要另起写线程。质量 95 是给回灌 mp4 二次压缩留的余量。

        文件名用帧序号而不是 stamp：回灌按序号读，天然有序；stamp 只用于
        事后核对时间轴，不参与排序。
        """
        idx = self.n_frames
        for key in self.image_keys:
            # ROS 与 lerobot 都是 RGB，OpenCV 的编码器要 BGR。
            # 必须 ascontiguousarray：[:, :, ::-1] 是负步长视图，cv::Mat 表示
            # 不了负的通道 stride，直接喂给 imwrite 会报 layout 不兼容。
            bgr = np.ascontiguousarray(frame["images"][key][:, :, ::-1])
            path = os.path.join(self.dir, key, f"{idx:06d}.jpg")
            if not cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(
                    f"JPEG 写盘失败：ep={self.ep_id} key={key} idx={idx} path={path}"
                )

        self._states.append(np.asarray(frame["state"], dtype=np.float32))
        self._actions.append(np.asarray(frame["action"], dtype=np.float32))
        self._stamps.append(float(frame["stamp"]))
        self.n_frames = idx + 1

    def finish(self) -> int:
        """把内存里的 state/action/stamp 落盘，返回这一集的帧数。

        图像在 write() 时已经逐张写好了，这里只补三个 npy。一帧都没凑齐的
        空集也照常写空数组（形状 (0, 7)），调用方按返回值判断是否可用，
        不用在这里抛异常——丢掉一集不该影响整轮采集。
        """
        np.save(
            os.path.join(self.dir, "states.npy"),
            np.asarray(self._states, dtype=np.float32).reshape(-1, 7),
        )
        np.save(
            os.path.join(self.dir, "actions.npy"),
            np.asarray(self._actions, dtype=np.float32).reshape(-1, 7),
        )
        np.save(
            os.path.join(self.dir, "stamps.npy"),
            np.asarray(self._stamps, dtype=np.float64),
        )
        self._states.clear()
        self._actions.clear()
        self._stamps.clear()
        return self.n_frames


class RecorderNode(Node):
    """按 /orch/trial 的 begin/end 开关录制窗口，把数据写成 LeRobot 数据集。

    状态机（state 同时出现在 /recorder/status 里）：

        idle      空闲，能立刻接住下一条 begin
        recording 正在往 staging 写帧
        flushing  已 end，正在回灌 lerobot / 归档，readiness=false

    刻意不在 __init__ 里 import lerobot：它会把 torch 一起拉进来，启动要多等
    好几秒，而且录制开始前不需要它。数据集对象也是第一次 flush 时才创建。
    """

    def __init__(self):
        super().__init__("recorder")

        self.declare_parameter("task_file", "/ws/task.yaml")
        self.declare_parameter("staging_dir", "")
        self.declare_parameter("result_timeout_s", 10.0)

        task_path = str(self.get_parameter("task_file").value)
        with open(task_path, encoding="utf-8") as f:
            self.task = yaml.safe_load(f)

        data = self.task["data"]
        self.fps = int(self.task["episode"]["fps"])
        self.language = str(data.get("language", ""))
        self.main_root = os.path.join(str(data["root"]), str(data["demos_dir"]))
        self.failed_root = self.main_root + "_failed"
        self.staging_root = str(self.get_parameter("staging_dir").value) or os.path.join(
            str(data["root"]), ".staging"
        )
        os.makedirs(self.staging_root, exist_ok=True)

        # ---- 状态机 --------------------------------------------------------
        self.state = "idle"
        self.ep_id: int | None = None
        self.ep: EpisodeStaging | None = None
        self.ep_result: dict | None = None
        # end 之后等 /orch/result 用：判定是异步的，录制线程不能空转轮询
        self.ep_event = threading.Event()
        # 判定在 detector 里要凑 hold 时长，10s 比一轮试次（15s）短、比判定长
        self.result_timeout_s = float(self.get_parameter("result_timeout_s").value)
        self.aligner = FrameAligner()
        self.n_ok = 0
        self.n_fail = 0
        # lerobot 数据集不是线程安全的，回灌串行跑在这一个线程里，不占 ROS 回调
        self._flush_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flush")
        self._datasets: dict[str, object] = {}

        # ---- QoS -----------------------------------------------------------
        # 与 orchestrator 的发布端逐项对齐，否则不匹配、一条也收不到
        latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        events = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        qos_img = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.pub_status = self.create_publisher(String, "/recorder/status", latched)
        self.create_subscription(String, "/orch/trial", self.on_trial, events)
        self.create_subscription(String, "/orch/result", self.on_result, 10)
        self.create_subscription(JointState, "/joint_states", self.on_joint_state, 10)
        for key in self.aligner.image_keys:
            self.create_subscription(
                Image,
                f"/camera/{key}/image_raw",
                partial(self.on_image, key),
                qos_img,
            )

        self.create_timer(0.2, self.publish_status)
        self.get_logger().info(
            f"recorder 启动：task={task_path} main={self.main_root} "
            f"failed={self.failed_root} staging={self.staging_root}"
        )

    def publish_status(self) -> None:
        """5Hz 广播状态，格式遵循 5.8 的下游状态契约。

        只有 idle 才算 ready：recording 时在下一条 begin 到来前收不了尾，
        flushing 时数据集还在回灌。两者都不是故障，所以报 readiness=false
        让 orchestrator 跳过这一拍，而不是报 liveness=false 把整轮停掉。

        丢帧计数跟着状态一起发：相机掉帧不会让节点挂掉，但会让数据变差，
        这是唯一能观察到它的地方。
        """
        payload = {
            "stamp": time.time(),
            "liveness": True,
            "readiness": self.state == "idle",
            "state": self.state,
            "episode": self.ep_id,
            "n_ok": self.n_ok,
            "n_fail": self.n_fail,
            "dropped": {
                "missing": self.aligner.dropped_missing,
                "stale": self.aligner.dropped_stale,
            },
        }
        if self.ep is not None:
            payload["frames"] = self.ep.n_frames
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub_status.publish(msg)

    def on_joint_state(self, msg: JointState) -> None:
        """关节状态 → aligner，并按名字重排成 JOINTS 的顺序。

        缺任何一个关节就整条丢弃、不补 0：补齐会让策略学到不存在的姿态，
        而且这种错误在数据集里极难发现。宁可少几帧。

        header.stamp 为 0 时退回本机时刻（部分驱动不打时间戳）。对齐只用到
        时间差，同机时钟下够用。
        """
        if self.state != "recording":
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0.0:
            stamp = time.time()
        try:
            values = [msg.position[msg.name.index(n)] for n in JOINTS]
        except ValueError:
            self.get_logger().warn(
                "joint_states 缺少关节，丢弃这一条", throttle_duration_sec=5.0
            )
            return
        self.aligner.push_state(stamp, values)
        self._capture()

    def _capture(self) -> None:
        """尝试凑一帧并写进 staging，joint_state 和 image 两个回调都会调用。

        只有 recording 且 ep 已建好才写。凑不齐是常态（等右邻 state、相机
        还没出第一帧），不在这里打日志：aligner 的分类计数已经能说明问题。
        """
        if self.state != "recording" or self.ep is None:
            return
        frame = self.aligner.try_frame()
        if frame is not None:
            self.ep.write(frame)

    def on_image(self, key: str, msg: Image) -> None:
        """相机图像 → aligner，重排成 HxWx3 的 RGB。

        step 不一定等于 width*3（行尾可能有 padding），所以按 step 切行再裁到
        width*3，不能直接 frombuffer + reshape，那样宽高会错位。

        step 之外的两种编码都接：相机节点配的是 rgb8，但换成 bgr8 也在
        aligner 里被拒之前先翻过来，避免整路数据静默失效。
        """
        if self.state != "recording":
            return
        if msg.encoding not in ("rgb8", "bgr8"):
            self.get_logger().warn(
                f"{key} 图像编码是 {msg.encoding}，只支持 rgb8/bgr8",
                throttle_duration_sec=5.0,
            )
            return
        h, w = int(msg.height), int(msg.width)
        row = np.frombuffer(msg.data, dtype=np.uint8).reshape(h, int(msg.step))
        rgb = row[:, : w * 3].reshape(h, w, 3)
        if msg.encoding == "bgr8":
            # 同样避免把负步长视图带进 aligner：后面每一帧都要再反转一次
            rgb = np.ascontiguousarray(rgb[:, :, ::-1])
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if stamp <= 0.0:
            stamp = time.time()
        self.aligner.push_image(key, stamp, rgb)
        self._capture()

    def on_trial(self, msg: String) -> None:
        """begin/end 事件 → 开关录制窗口。

        begin：重建对齐器与 staging，切 recording。
        end  ：把 npy 落盘，切 flushing，回灌交给后台线程。

        不配对的事件只记日志、不动状态：orchestrator 是唯一事件源，真出问题
        会由它下一拍的 check_ready 暴露出来，recorder 自己猜反而更糟。
        """
        try:
            data = json.loads(msg.data)
            ep_id = int(data["id"])
            event = str(data["event"])
        except (ValueError, KeyError, TypeError):
            self.get_logger().warn(f"无法解析 /orch/trial：{msg.data!r}")
            return

        if event == "begin":
            if self.state != "idle":
                self.get_logger().warn(
                    f"忽略 begin：当前 {self.state}（ep={self.ep_id}）"
                )
                return
            self.ep_id = ep_id
            self.ep_result = None
            self.ep_event.clear()
            # 重建对齐器：上一集残留的 state/图像会让第一帧张冠李戴
            self.aligner = FrameAligner()
            self.ep = EpisodeStaging(self.staging_root, ep_id, self.aligner.image_keys)
            self.state = "recording"
            self.publish_status()
            self.get_logger().info(f"开始录制 ep={ep_id}")
            return

        if event == "end":
            if self.state != "recording" or self.ep is None or self.ep_id != ep_id:
                self.get_logger().warn(
                    f"忽略 end：state={self.state} ep={self.ep_id} 事件 id={ep_id}"
                )
                return
            ep, n = self.ep, self.ep.finish()
            self.state = "flushing"
            self.publish_status()
            self.get_logger().info(f"结束录制 ep={ep_id} frames={n}，等 result 回灌")
            self._flush_pool.submit(self._finalize, ep, ep_id, n)
            return

        self.get_logger().warn(f"未知事件 {event!r}，忽略")

    def on_result(self, msg: String) -> None:
        """记录判定结论并唤醒正在等它的 _finalize。

        只认当前这一集：detector 是异步判定的，上一个试次迟到的结论如果被
        收下，这一集会带着别人的 success 进数据集，而且事后无从发现。

        JSON 解析失败按失败处理而不是丢弃：宁可把一条好数据标成失败，
        也不要把一次失败的试次标成成功——后者会直接污染训练集。
        """
        try:
            data = json.loads(msg.data)
        except ValueError:
            # 没有 id 就没法归属，只能算到当前这一集上；方向取失败是安全的
            data = {"id": self.ep_id, "success": False, "reason": "bad_json"}
        if data.get("id") != self.ep_id:
            self.get_logger().warn(
                f"丢弃非当前试次的结论：id={data.get('id')}，当前 {self.ep_id}",
                throttle_duration_sec=2.0,
            )
            return
        self.ep_result = data
        self.ep_event.set()

    def _finalize(self, ep: EpisodeStaging, ep_id: int, n: int) -> None:
        """等 result，把 staging 回灌进对应数据集，然后清掉 staging。

        跑在 _flush_pool 的单线程里，所以这里改 self.state 不会被另一个回灌
        任务插队；ROS 侧看到的是 flushing → idle 的一次跳变。

        等不到 result 按失败归档（reason=result_timeout）：判定丢了说明这一
        试次本身可疑，不该进训练集，但帧还在，留给事后分析。
        """
        ok = False
        try:
            if not self.ep_event.wait(timeout=self.result_timeout_s):
                self.get_logger().warn(f"ep={ep_id} 等 result 超时，按失败归档")
                result = {"success": False, "reason": "result_timeout"}
            else:
                result = self.ep_result or {"success": False, "reason": "no_result"}
            ok = bool(result.get("success"))

            if n <= 0:
                # 带上对齐器的丢帧计数，否则只能看到 frames=0，没法判断是
                # 「图没来」还是「时间戳对不上」。
                self.get_logger().warn(
                    f"ep={ep_id} 没有有效帧，丢弃"
                    f"（missing={self.aligner.dropped_missing}"
                    f" stale={self.aligner.dropped_stale}）"
                )
            else:
                # 有帧也报一次丢帧数：录制期间的掉帧（图像丢包、配对失败）
                # 只能从这里看出来，否则 15Hz 和 30Hz 在日志里长得一样。
                self.get_logger().info(
                    f"ep={ep_id} 对齐统计：丢帧 missing={self.aligner.dropped_missing}"
                    f" stale={self.aligner.dropped_stale}"
                )
                root = self.main_root if ok else self.failed_root
                self._ingest(self._dataset(root), ep, n, result)

            if n > 0:
                if ok:
                    self.n_ok += 1
                else:
                    self.n_fail += 1
            self.get_logger().info(
                f"ep={ep_id} frames={n} success={ok} "
                f"reason={result.get('reason', '')} → "
                f"{'main' if ok else 'failed'}"
            )
        except Exception:
            self.n_fail += 1
            self.get_logger().error(f"ep={ep_id} 回灌失败：\n{traceback.format_exc()}")
        finally:
            # staging 只是中转：失败也删，残留几十个 2.4G 目录会把盘吃光
            shutil.rmtree(ep.dir, ignore_errors=True)
            self.ep = None
            self.state = "idle"
            self.publish_status()

    def _dataset(self, root: str):
        """按需创建或打开一个 LeRobotDataset，结果缓存在 _datasets 里。

        lerobot 只在第一次回灌时才 import：它会把 torch 一起拉起来，启动
        要多等好几秒，纯录制不跑策略时没必要付这个代价。

        目录里已有 meta/ 就 open 继续追加，没有才 create：进程重启不该把
        之前录的数据覆盖掉。

        features 的 shape 用 task.yaml 里登记的相机分辨率（读的是 [w, h]，
        而 lerobot 要 (h, w, 3)，这里换序）。
        """
        if root in self._datasets:
            return self._datasets[root]

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        repo_id = f"local/{os.path.basename(root)}"
        if os.path.isdir(os.path.join(root, "meta")):
            ds = LeRobotDataset(repo_id, root=root)
        else:
            features = {
                "observation.state": {
                    "dtype": "float32",
                    "shape": (len(JOINTS),),
                    "names": list(JOINTS),
                },
                "action": {
                    "dtype": "float32",
                    "shape": (len(JOINTS),),
                    "names": list(JOINTS),
                },
            }
            for key in self.aligner.image_keys:
                w, h = self.task["camera"][key]["resolution"]
                features[f"observation.images.{key}"] = {
                    "dtype": "video",
                    "shape": (int(h), int(w), 3),
                    "names": ["height", "width", "channels"],
                }
            ds = LeRobotDataset.create(
                repo_id=repo_id,
                root=root,
                fps=self.fps,
                features=features,
                robot_type="alicia_d",
                use_videos=True,
            )
        self._datasets[root] = ds
        return ds

    def _ingest(self, ds, ep: EpisodeStaging, n: int, result: dict) -> None:
        """把 staging 的 n 帧按序号灌进数据集，最后 save_episode。

        staging 是 jpg q95，这里解回 RGB 交给 lerobot 再压进 mp4：损失的
        只有这一次 jpg 解码，不额外放大。倒序成 RGB 后要 ascontiguousarray，
        lerobot 内部会直接把数组写进编码器，非连续内存会触发隐式拷贝甚至报错。
        """
        states = np.load(os.path.join(ep.dir, "states.npy"))
        actions = np.load(os.path.join(ep.dir, "actions.npy"))
        for i in range(n):
            frame = {
                "observation.state": states[i],
                "action": actions[i],
                "task": self.language,
            }
            for key in ep.image_keys:
                path = os.path.join(ep.dir, key, f"{i:06d}.jpg")
                bgr = cv2.imread(path, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f"读不到 staging 图像：{path}")
                frame[f"observation.images.{key}"] = np.ascontiguousarray(
                    bgr[:, :, ::-1]
                )
            ds.add_frame(frame)
        ds.save_episode()
        self.get_logger().debug(
            f"回灌完成 ep={ep.ep_id} frames={n} success={result.get('success')}"
        )

    def shutdown(self) -> None:
        """退出前收尾：丢掉没录完的，等回灌跑完，再关数据集。

        录制中的 staging 直接删——没有 result 就无法归档，留着只占盘。
        回灌中的必须等：一次回灌几十秒，Ctrl-C 砍在这里会留下半条 episode。

        最后的 finalize 不能省：lerobot v3 的 parquet writer 是流式的，只在
        finalize 时写 footer。少了它文件停在半截（没有结尾的 PAR1），pyarrow
        直接报 "Parquet magic bytes not found"，整份数据集都读不出来。
        """
        if self.ep is not None:
            self.get_logger().warn(f"退出时丢弃未完成的录制 ep={self.ep_id}")
            shutil.rmtree(self.ep.dir, ignore_errors=True)
            self.ep = None
        self._flush_pool.shutdown(wait=True)
        for root, ds in self._datasets.items():
            try:
                ds.finalize()
            except Exception:  # noqa: BLE001 - 一个数据集收尾失败不该影响其他
                self.get_logger().error(
                    f"数据集 finalize 失败：{root}\n{traceback.format_exc()}"
                )
        self._datasets.clear()


def main(argv=None) -> None:
    """起节点并 spin；退出前把正在回灌的 episod 等完再关池。

    shutdown(wait=True) 放在这里而不是 __del__：一次回灌要几十秒，Ctrl-C
    时直接杀进程会留下半条 episode 和一堆 staging 目录。

    SIGTERM（docker stop 默认发的信号）不走 KeyboardInterrupt，不接管的话
    Python 会直接退进程，finally 整段被跳过——长回灌正好会被砍在中间。
    这里把它转成同一个中断路径，两条退出方式行为一致。
    """
    rclpy.init(args=argv)
    node = RecorderNode()

    def _on_sigterm(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
