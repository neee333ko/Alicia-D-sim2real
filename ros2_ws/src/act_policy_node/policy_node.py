#!/usr/bin/env python3
"""policy_node.py — 把 lerobot 的策略包成 ROS 2 节点。

职责：收到 /policy/infer_request → 用缓存的最新观测推理一次 → 发一个 chunk。
时序、队列、平滑、安全全在 action_executor 里，这里一概不管。

它不知道下游怎么执行，也不知道 action_executor 存在；换 ACT → SmolVLA 只改
policy_path 和 policy_type。

话题
  订阅  /camera/front/image_raw, /camera/wrist/image_raw
        /joint_states, /task/instruction, /policy/infer_request
  发布  /policy/action_chunk   (std_msgs/Float64MultiArray)
        data[0] = 观测时刻（秒），data[1] = chunk_size，data[2:] = N×7

时间基准：全链路 30Hz。chunk 里的每一步代表 1/30 秒，下游按这个节奏消费；
相机、/joint_states、/joint_commands、训练数据集的 fps 必须都是 30。

观测对齐：三路各缓存 5 帧，时间戳一律用 header.stamp（采集时刻，不是收到时刻）。
推理时以最新图像为锚，找关节状态里离它最近的一帧，两者相差不超过一帧（33ms）
就算同一时刻。找不到就跳过这次请求，等 action_executor 重发。

前置：本文件不需要 colcon，直接 python3 跑。
      checkpoint 目录里必须有 lerobot 自己的 config.json 和 stats。
"""

from __future__ import annotations

import threading
import time
from collections import deque

import numpy as np
import rclpy
import torch
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Empty, Float64MultiArray, String

ACTION_DIM = 7

# 按类型拿策略类。加新模型只改这张表，不动别的代码。
_POLICY_CLASSES = {
    "act": ("lerobot.policies.act.modeling_act", "ACTPolicy"),
    "smolvla": ("lerobot.policies.smolvla.modeling_smolvla", "SmolVLAPolicy"),
    "pi0": ("lerobot.policies.pi0.modeling_pi0", "PI0Policy"),
}


def _policy_class(policy_type: str):
    import importlib

    mod, cls = _POLICY_CLASSES.get(policy_type, (None, None))
    if mod is None:
        raise ValueError(f"不支持的 policy_type={policy_type}，可选：{list(_POLICY_CLASSES)}")
    return getattr(importlib.import_module(mod), cls)


def _img_to_numpy(msg: Image) -> np.ndarray:
    """sensor_msgs/Image → HWC uint8。不用 cv_bridge，省掉一套 OpenCV。"""
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ("rgb8", "bgr8"):
        img = buf.reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            img = img[:, :, ::-1]
    elif msg.encoding in ("mono8", "8UC1"):
        img = np.repeat(buf.reshape(msg.height, msg.width, 1), 3, axis=2)
    else:
        raise ValueError(f"不支持的图像编码：{msg.encoding}")
    return np.ascontiguousarray(img)


class PolicyNode(Node):
    def __init__(self):
        super().__init__("policy_node")
        cb = ReentrantCallbackGroup()      # 推理期间不能把订阅堵死

        # ---- 参数 ----------------------------------------------------------
        self.declare_parameter("policy_path", "")
        self.declare_parameter("policy_type", "act")        # act | smolvla
        self.declare_parameter("device", "cuda")
        self.declare_parameter("front_key", "observation.images.front")
        self.declare_parameter("wrist_key", "observation.images.wrist")
        self.declare_parameter("state_key", "observation.state")
        self.declare_parameter("max_obs_age_s", 0.25)       # 对齐后整体太旧就不推理
        self.declare_parameter("align_tol_s", 1.0 / 30.0)   # 三路之间允许的时间差：一帧
        self.declare_parameter("buffer_len", 5)             # 每路缓存几帧
        self.declare_parameter("warmup", True)

        p = self.get_parameter
        self.policy_path = p("policy_path").value
        self.policy_type = p("policy_type").value
        self.device = p("device").value
        self.front_key = p("front_key").value
        self.wrist_key = p("wrist_key").value
        self.state_key = p("state_key").value
        self.max_obs_age = float(p("max_obs_age_s").value)
        self.align_tol = float(p("align_tol_s").value)
        self.buffer_len = int(p("buffer_len").value)
        if not self.policy_path:
            raise ValueError("必须指定 policy_path")

        # ---- 观测缓存 ------------------------------------------------------
        # 每路保留最近 buffer_len 帧，存 (stamp, 数据)。stamp 来自 header.stamp，
        # 是采集时刻；用收到时刻对齐会把 USB 传输和调度的抖动算进观测里。
        self.lock = threading.Lock()
        self.front_buf: deque[tuple[float, np.ndarray]] = deque(maxlen=self.buffer_len)
        self.wrist_buf: deque[tuple[float, np.ndarray]] = deque(maxlen=self.buffer_len)
        self.state_buf: deque[tuple[float, list[float]]] = deque(maxlen=self.buffer_len)
        self.instruction = ""
        self._stamp_warned = False

        # ---- 模型 ----------------------------------------------------------
        self._load_policy()
        if bool(p("warmup").value):
            self._warmup()

        # ---- ROS 接口 ------------------------------------------------------
        # 图像 best_effort：丢一帧无所谓，reliable 会在拥塞时把相机节点堵住
        qos_img = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=self.buffer_len,
        )
        # 关节状态 reliable：小消息，丢不起
        qos_state = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=self.buffer_len,
        )
        # 推理请求：只有"最新一次"有意义，depth 1 让积压的旧请求直接被顶掉
        qos_req = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # 任务指令：transient_local（相当于 latched），后启动的节点也能拿到最后一条。
        # 发布方也必须用同样的 durability，否则匹配不上。
        qos_task = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.sub_front = self.create_subscription(
            Image, "/camera/front/image_raw", lambda m: self._on_image(m, "front"), qos_img,
            callback_group=cb,
        )
        self.sub_wrist = self.create_subscription(
            Image, "/camera/wrist/image_raw", lambda m: self._on_image(m, "wrist"), qos_img,
            callback_group=cb,
        )
        self.sub_state = self.create_subscription(
            JointState, "/joint_states", self._on_state, qos_state, callback_group=cb
        )
        self.sub_task = self.create_subscription(
            String, "/task/instruction", self._on_task, qos_task, callback_group=cb
        )
        self.sub_req = self.create_subscription(
            Empty, "/policy/infer_request", self._on_request, qos_req, callback_group=cb
        )
        # chunk 用 reliable 保证不丢（一次丢整段等于瞎 2 秒）。
        # durability 保持 volatile：重启的 action_executor 就算拿到旧 chunk 也没用，
        # 它不知道这段计划是从第几步开始执行的上下文，等下一段（150ms 内）更干净。
        qos_chunk = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pub_chunk = self.create_publisher(
            Float64MultiArray, "/policy/action_chunk", qos_chunk
        )

        # ---- 推理线程 ------------------------------------------------------
        # 推理和订阅分离：一次 200ms 的推理不能把图像回调堵住
        self._req_event = threading.Event()
        self._stop = False
        self._worker = threading.Thread(target=self._infer_loop, daemon=True)
        self._worker.start()

        self.n_ok = 0
        self.n_skip = 0
        self.get_logger().info(
            f"policy_node 就绪：{self.policy_path}（{self.policy_type}，{self.device}），"
            f"chunk_size={self.chunk_size}"
        )

    # ------------------------------------------------------------------ 模型
    def _load_policy(self):
        """不走 lerobot.policies.factory：那个模块顶层 import gymnasium，
        而 gymnasium 是训练/环境依赖，推理镜像里不装。直接按类型取策略类和
        processor 管道，等价但少一堆依赖。
        """
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.converters import (
            batch_to_transition,
            policy_action_to_transition,
            transition_to_batch,
            transition_to_policy_action,
        )

        policy_cls = _policy_class(self.policy_type)
        self.policy = policy_cls.from_pretrained(self.policy_path)
        self.policy.to(self.device)
        self.policy.eval()

        # 归一化统计在 checkpoint 的 preprocessor.json / postprocessor.json 里。
        # 少一个文件就直接报错退出——不要静默跳过归一化，那会让策略输出整个错尺度。
        self.pre = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=self.policy_path,
            config_filename="preprocessor.json",
            to_transition=batch_to_transition,
            to_output=transition_to_batch,
        )
        self.post = PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=self.policy_path,
            config_filename="postprocessor.json",
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )
        self.chunk_size = int(self.policy.config.chunk_size)

    def _warmup(self):
        """第一次推理慢好几倍（CUDA 上下文、kernel 编译），先跑一次假的。"""
        try:
            with torch.no_grad():
                zeros = self._dummy_batch()
                _ = self.policy.predict_action_chunk(zeros)
            self.get_logger().info("warmup 完成")
        except Exception as e:                       # noqa: BLE001
            self.get_logger().warn(f"warmup 失败（不影响运行，首次推理会慢）：{e}")

    def _dummy_batch(self):
        h, w = self._image_hw()
        batch = {
            self.front_key: torch.zeros(1, 3, h, w, device=self.device),
            self.wrist_key: torch.zeros(1, 3, h, w, device=self.device),
            self.state_key: torch.zeros(1, ACTION_DIM, device=self.device),
        }
        return self.pre(batch)

    def _image_hw(self):
        feats = getattr(self.policy.config, "image_features", None) or {}
        for shape in feats.values():
            if len(shape) == 3:
                return int(shape[1]), int(shape[2])
        return 720, 1280

    # ------------------------------------------------------------------ 回调
    def _stamp_of(self, msg) -> float:
        """header.stamp → 秒。没填就退回到到达时刻，并告警一次。"""
        st = msg.header.stamp
        t = float(st.sec) + float(st.nanosec) * 1e-9
        if t <= 0.0:
            if not self._stamp_warned:
                self._stamp_warned = True
                self.get_logger().warn(
                    "消息没有 header.stamp，退回用到达时刻对齐；"
                    "请让 camera_node / driver_node 填上采集时刻"
                )
            return time.time()
        return t

    def _on_image(self, msg: Image, which: str):
        try:
            img = _img_to_numpy(msg)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=5.0)
            return
        t = self._stamp_of(msg)
        with self.lock:
            if which == "front":
                self.front_buf.append((t, img))
            else:
                self.wrist_buf.append((t, img))

    def _on_state(self, msg: JointState):
        if len(msg.position) < ACTION_DIM:
            return
        t = self._stamp_of(msg)
        with self.lock:
            self.state_buf.append((t, [float(v) for v in msg.position[:ACTION_DIM]]))

    def _on_task(self, msg: String):
        with self.lock:
            self.instruction = msg.data

    def _on_request(self, _msg: Empty):
        self._req_event.set()

    # ------------------------------------------------------------------ 推理
    def _infer_loop(self):
        while not self._stop:
            if not self._req_event.wait(timeout=0.1):
                continue
            self._req_event.clear()
            try:
                self._infer_once()
            except Exception as e:                   # noqa: BLE001
                self.get_logger().error(f"推理失败：{e}", throttle_duration_sec=2.0)

    def _infer_once(self):
        aligned = self._pick_aligned()
        if aligned is None:
            self._skip(
                f"{self.buffer_len} 帧里找不到相差 ≤ {self.align_tol * 1000:.0f}ms 的一组观测"
            )
            return
        obs_t, front, wrist, state = aligned

        h, w = self._image_hw()
        batch = {
            self.front_key: _to_tensor(front, h, w, self.device),
            self.wrist_key: _to_tensor(wrist, h, w, self.device),
            self.state_key: torch.tensor([state], dtype=torch.float32, device=self.device),
        }

        with torch.no_grad():
            batch = self.pre(batch)
            actions = self.policy.predict_action_chunk(batch)   # [1, chunk, 7]
            actions = self.post(actions)

        chunk = actions[0].detach().to("cpu").numpy()
        if chunk.shape[1] != ACTION_DIM:
            # SmolVLA / π0 会 pad 到 32 维，必须切掉；切错一位动作就全乱
            chunk = chunk[:, :ACTION_DIM]
        if chunk.shape != (self.chunk_size, ACTION_DIM):
            self._skip(f"输出形状异常 {chunk.shape}，期望 ({self.chunk_size}, {ACTION_DIM})")
            return

        msg = Float64MultiArray()
        # stamp 用图像时刻：下游 action_executor 靠它算"从第几步开始用"
        msg.data = [float(obs_t), float(self.chunk_size)] + chunk.astype(float).ravel().tolist()
        self.pub_chunk.publish(msg)
        self.n_ok += 1
        if self.n_ok % 20 == 0:
            self.get_logger().info(f"已发 {self.n_ok} 个 chunk，跳过 {self.n_skip} 次")

    def _pick_aligned(self):
        """从三路缓存里挑一组时间对齐的观测。

        以图像为锚（图像最贵、也不能插值），从最新的一帧往前试，找到第一组
        关节状态离锚点不超过 align_tol 的组合。整体太旧（超过 max_obs_age）也算失败，
        那说明某一路已经断流了。
        """
        with self.lock:
            fronts = list(self.front_buf)
            wrists = list(self.wrist_buf)
            states = list(self.state_buf)

        if not fronts or not wrists or not states:
            return None

        now = time.time()
        for t_img, img in reversed(fronts):
            if now - t_img > self.max_obs_age:
                return None                      # 连最新的都过期了，别往前找了
            wrist = _closest(wrists, t_img, self.align_tol)
            state = _closest(states, t_img, self.align_tol)
            if wrist is not None and state is not None:
                return t_img, img, wrist, state
        return None

    def _skip(self, why: str):
        self.n_skip += 1
        self.get_logger().warn(f"跳过推理：{why}", throttle_duration_sec=2.0)

    def shutdown(self):
        self._stop = True
        self._req_event.set()
        self._worker.join(timeout=2.0)


def _closest(buf, t_anchor: float, tol: float):
    """在 (stamp, 数据) 缓存里找离 t_anchor 最近的一条，超出 tol 返回 None。"""
    best, best_d = None, 1e9
    for t, data in buf:
        d = abs(t - t_anchor)
        if d < best_d:
            best, best_d = data, d
    return best if best_d <= tol else None


def _to_tensor(img: np.ndarray, h: int, w: int, device: str) -> torch.Tensor:
    """HWC uint8 → 1×3×H×W float32（0~1）。和 lerobot 训练时完全一致的格式。"""
    t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    if t.shape[-2] != h or t.shape[-1] != w:
        t = torch.nn.functional.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    return t.to(device)


def main():
    rclpy.init()
    node = PolicyNode()
    exe = MultiThreadedExecutor(num_threads=4)
    exe.add_node(node)
    try:
        exe.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
