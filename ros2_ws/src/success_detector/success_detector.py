#!/usr/bin/env python3
"""success_detector.py — 判定一次试次是否成功，不需要人看。

这是飞轮的闸门：判定靠人按 y/n，一晚上只能跑几十次；自动判定后一晚上几百次。

两种判定模式（配置里 judge 选）：
  vision        真机用。读图像，看方块的彩色质心是否落在杯口圆内。
                单目看不出"在杯口上方还是下方"，靠"停稳 + 松手"间接排除。
  ground_truth  仿真用。读 /object/pose 的三维坐标，直接比距离和高度，
                没有单目的歧义，判得最准。但只能用于仿真。

判据（三条同时成立，并保持 hold_s）：
  1. 方块的彩色质心落在杯口圆内（像素坐标，杯子固定所以只标定一次）
  2. 最近 settle_s 秒内质心抖动小于 settle_px（停稳了）
  3. 夹爪已经张开到 open_min 以上（松手了）

话题/服务
  订阅  /camera/front/image_raw    sensor_msgs/Image（rgb8）
        /joint_states             sensor_msgs/JointState（只取夹爪开度，第 7 维）
        /object/pose              geometry_msgs/PoseStamped（仿真真值，调试对拍用）
        /task/instruction         std_msgs/String（决定"判什么"，任务级）
        /orch/trial              std_msgs/String（决定"什么时候判"，试次级）
  发布  /detector/condition       std_msgs/Bool：**实时状态**，判定条件此刻成立与否。
                                条件变化才发，一个试次里可能变很多次。调试用。
        /orch/result              std_msgs/String：**试次结论**，每个试次恰好一条，带 id。
                                成功、超时、被手动结束都在这里，含原因和指标。
                                这是唯一的结论出口，不要再加一个"布尔版"——
                                两个话题表达同一件事，迟早有人给它们统计出两个数。
        /detector/mask            sensor_msgs/Image（mono8，调阈值时看这个）

语义约定：/orch/result 是结论，一个试次一条；实时状态看 /detector/condition。

本节点不提供任何服务：试次开始/结束是一对多事件（recorder 要录、detector 要判），
用话题广播，不用服务挨个调。

配置在 calib/detector.yaml，所有阈值都在那里，改完重启节点即可。

    python3 success_detector.py --ros-args -p config_file:=/calib/detector.yaml
"""

from __future__ import annotations

import array
import json
import math
import os
import time

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, String

JOINT_NAMES = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]


# --------------------------------------------------------------------- 工具
def rgb_to_hsv(img: np.ndarray) -> np.ndarray:
    """HWC uint8 RGB → H(0~180) S(0~255) V(0~255) 的 uint8。

    用整数近似而不是 cv2，只为了控制 HSV 区间，精度够用。
    """
    rgb = img.astype(np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=2)
    mn = np.min(rgb, axis=2)
    diff = (mx - mn).astype(np.int16)

    s = np.where(mx == 0, 0, (diff * 255) // np.maximum(mx, 1)).astype(np.int16)
    v = mx.astype(np.int16)

    # 色相按 60° 分段，最后压缩到 0~180
    h = np.zeros_like(mx, dtype=np.int16)
    nonzero = diff > 0
    rmax = nonzero & (mx == r)
    gmax = nonzero & (mx == g) & ~rmax
    bmax = nonzero & (mx == b) & ~rmax & ~gmax
    h[rmax] = (60 * ((g[rmax] - b[rmax]) / diff[rmax])) % 360
    h[gmax] = 60 * ((b[gmax] - r[gmax]) / diff[gmax]) + 120
    h[bmax] = 60 * ((r[bmax] - g[bmax]) / diff[bmax]) + 240
    h = (h // 2).astype(np.uint8)          # 0~180
    return np.stack([h, s.astype(np.uint8), v.astype(np.uint8)], axis=2)


def hsv_mask(hsv: np.ndarray, ranges: list) -> np.ndarray:
    """ranges 是 [(lower, upper), ...]，任一区间命中即为真（红色跨 0 需要用两段）。"""
    mask = np.zeros(hsv.shape[:2], dtype=bool)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    for lo, hi in ranges:
        mask |= (h >= lo[0]) & (h <= hi[0]) & (s >= lo[1]) & (s <= hi[1]) & (v >= lo[2]) & (v <= hi[2])
    return mask


# --------------------------------------------------------------------- 节点
class SuccessDetector(Node):
    def __init__(self):
        super().__init__("success_detector")

        self.declare_parameter("config_file", "/calib/detector.yaml")
        cfg_path = self.get_parameter("config_file").value
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"找不到判定配置：{cfg_path}")
        with open(cfg_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        ep = self.cfg.get("episode", {})
        self.settle_s = float(ep.get("settle_s", 0.5))
        self.settle_px = float(ep.get("settle_px", 2.0))
        self.hold_s = float(ep.get("hold_s", 2.0))
        self.timeout_s = float(ep.get("timeout_s", 15.0))

        self.judge = str(self.cfg.get("judge", "vision"))
        if self.judge not in ("vision", "ground_truth"):
            raise ValueError(f"judge 只能是 vision 或 ground_truth，收到 {self.judge}")

        cup = self.cfg["cup"]
        self.cup_c = (float(cup["center_px"][0]), float(cup["center_px"][1]))
        self.cup_r = float(cup["radius_px"]) - float(cup.get("margin_px", 0))

        # 真值模式要用的三维几何：杯心（世界坐标，米）、内径半径、杯口高度
        self.cup_xy_m = tuple(float(v) for v in cup.get("xy_m", [0.0, 0.0]))
        self.cup_r_m = float(cup.get("radius_m", 0.035))
        self.cup_h_m = float(cup.get("height_m", 0.09))

        cube = self.cfg["cube"]
        self.hsv_ranges = list(zip(cube["ranges_lower"], cube["ranges_upper"]))
        self.min_area = int(cube["min_area_px"])
        self.settle_m = float(cube.get("settle_m", 0.002))

        grip = self.cfg.get("gripper", {})
        self.grip_open_min = float(grip.get("open_min", 0.8))

        # ---- 状态 ----------------------------------------------------------
        self.watching = False
        self.t_start = 0.0
        self.hold_since = None
        self.settle_hist = []          # (t, x, y)
        self.joints = None             # 6 关节弧度
        self.gripper = None
        self.dist_px = float("inf")      # 视觉模式：到杯口圆心的像素距离
        self.cube_2d = None              # 视觉模式：(cx, cy, t, area)
        self.cube_3d = None              # 真值模式：(x, y, z, t)
        self.dist_m = float("inf")
        self.gt_pose = None
        self.success_latch = False
        self.episode_id = -1
        self.task = ""

        # ---- 内部时钟：判定用的是耗时，不跟 ROS 时钟绑定 -------------------
        self.qos_img = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.pub_condition = self.create_publisher(
            Bool,
            "/detector/condition",
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.VOLATILE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )
        self.pub_result = self.create_publisher(String, "/orch/result", 10)
        self.pub_mask = self.create_publisher(Image, "/detector/mask", self.qos_img)

        self.create_subscription(Image, "/camera/front/image_raw", self.on_image, self.qos_img)
        self.create_subscription(JointState, "/joint_states", self.on_state, 10)
        self.create_subscription(PoseStamped, "/object/pose", self.on_object, 10)
        self.create_subscription(
            String,
            "/task/instruction",
            self.on_instruction,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )
        self.create_subscription(
            String,
            "/orch/trial",
            self.on_episode,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )

        # 判定独立成定时器：仿真用真值时可能根本没有图像流
        self.create_timer(1.0 / 30.0, self._evaluate)

        self.get_logger().info(
            f"success_detector 就绪（{self.judge}）：杯口 ({self.cup_c[0]:.0f},{self.cup_c[1]:.0f}) r={self.cup_r:.0f}px，"
            f"保持 {self.hold_s}s，超时 {self.timeout_s}s"
        )

    # ---------------------------------------------------------------- 回调
    def on_state(self, msg: JointState):
        if len(msg.position) < 7:
            return
        self.joints = [float(v) for v in msg.position[:6]]
        self.gripper = float(msg.position[6])

    def on_instruction(self, msg: String):
        """任务级指令：决定用哪套判据。v1 只有一套，任务名对不上就告警。

        以后多任务时，这里按 task 切到不同的 cup/cube 参数组（读不同的配置段）。
        """
        self.task = msg.data.strip()
        expected = self.cfg.get("task", "cube_into_cup")
        if self.task and self.task != expected:
            self.get_logger().warn(
                f"当前任务指令「{self.task}」和配置里的「{expected}」不一致，"
                f"仍按配置的判据执行"
            )
        else:
            self.get_logger().info(f"任务：{self.task or expected}")

    def on_episode(self, msg: String):
        """试次生命周期由 orchestrator 广播：
        {"id": 7, "event": "begin"}  /  {"id": 7, "event": "end"}"""
        try:
            data = json.loads(msg.data)
            event = data.get("event")
            ep_id = int(data.get("id", -1))
        except (ValueError, TypeError):
            self.get_logger().error(f"/orch/trial 解析失败：{msg.data}", throttle_duration_sec=5.0)
            return

        if event == "begin":
            self.episode_id = ep_id
            self.watching = True
            self.t_start = time.monotonic()
            self.hold_since = None
            self.settle_hist.clear()
            self.success_latch = False
            self.cube_2d = None
            self.dist_px = float("inf")
            self.dist_m = float("inf")
            self._publish_condition(False)
            self.get_logger().info(f"开始判断试次 #{ep_id}")
        elif event == "end":
            if self.watching:
                self.watching = False
                self._publish_result(False, "ended_without_success")
        else:
            self.get_logger().warn(f"未知 event：{event}", throttle_duration_sec=5.0)

    # ---------------------------------------------------------------- 观测
    def on_image(self, msg: Image):
        """只负责"看到方块在哪"，不判定。判定统一在 _evaluate 里。"""
        img = self._to_rgb(msg)
        if img is None:
            return
        mask = hsv_mask(rgb_to_hsv(img), self.hsv_ranges)
        self._publish_mask(mask, msg.header.stamp)

        area = int(mask.sum())
        if area < self.min_area:
            self.cube_2d = None
            return
        ys, xs = np.nonzero(mask)
        self.cube_2d = (float(xs.mean()), float(ys.mean()), time.monotonic(), area)

    def on_object(self, msg: PoseStamped):
        """仿真真值。ground_truth 模式用它判定；vision 模式只拿来对拍。"""
        p = msg.pose.position
        self.gt_pose = (p.x, p.y, p.z)
        self.cube_3d = (p.x, p.y, p.z, time.monotonic())

    # ---------------------------------------------------------------- 判定
    def _evaluate(self):
        if not self.watching:
            return
        now = time.monotonic()
        if now - self.t_start > self.timeout_s:
            self.watching = False
            self._publish_result(False, "timeout")
            return

        if self.judge == "ground_truth":
            ok = self._check_ground_truth(now)
        else:
            ok = self._check_vision(now)

        self._publish_condition(ok)
        if ok and self.hold_since is not None and now - self.hold_since >= self.hold_s:
            self.watching = False
            self._publish_result(True, "success")

    def _check_vision(self, now: float) -> bool:
        if self.cube_2d is None:
            self.dist_px = float("inf")
            return self._fail()
        cx, cy, _t, _area = self.cube_2d
        self.dist_px = math.hypot(cx - self.cup_c[0], cy - self.cup_c[1])
        inside = self.dist_px <= self.cup_r
        settled = self._settled(now, (cx, cy), self.settle_px)
        released = self.gripper is not None and self.gripper >= self.grip_open_min
        return self._hold(inside and settled and released, now)

    def _check_ground_truth(self, now: float) -> bool:
        """真值模式：三维距离 + 高度。

        这是唯一能真正区分"在杯里"和"悬在杯口上方"的判据——单目做不到。
        """
        if self.cube_3d is None:
            self.dist_m = float("inf")
            return self._fail()
        x, y, z, _t = self.cube_3d
        self.dist_m = math.hypot(x - self.cup_xy_m[0], y - self.cup_xy_m[1])
        # 必须在杯口圈内，而且低于杯口（这一条只有真值模式能判）
        inside = self.dist_m <= self.cup_r_m and z <= self.cup_h_m
        settled = self._settled(now, (x, y), self.settle_m)
        released = self.gripper is not None and self.gripper >= self.grip_open_min
        return self._hold(inside and settled and released, now)

    def _settled(self, now: float, xy, tol: float) -> bool:
        """最近 settle_s 内的位置抖动是否小于 tol。"""
        self.settle_hist.append((now, xy[0], xy[1]))
        while self.settle_hist and now - self.settle_hist[0][0] > self.settle_s:
            self.settle_hist.pop(0)
        if len(self.settle_hist) < 3:
            return False
        xs = [p[1] for p in self.settle_hist]
        ys = [p[2] for p in self.settle_hist]
        return math.hypot(max(xs) - min(xs), max(ys) - min(ys)) <= tol

    def _hold(self, all_ok: bool, now: float) -> bool:
        if all_ok:
            if self.hold_since is None:
                self.hold_since = now
            return True
        self.hold_since = None
        return False

    def _fail(self) -> bool:
        self.hold_since = None
        self.settle_hist.clear()
        return False

    # ---------------------------------------------------------------- 发布
    def _publish_condition(self, ok: bool):
        """实时状态：条件此刻成立与否。只在变化时发，避免 30Hz 刷屏。"""
        if ok == self.success_latch:
            return
        self.success_latch = ok
        self.pub_condition.publish(Bool(data=bool(ok)))

    def _publish_result(self, success: bool, reason: str):
        payload = {
            "id": self.episode_id,
            "success": bool(success),
            "reason": reason,
            "hold_ms": int((time.monotonic() - self.hold_since) * 1000) if self.hold_since else 0,
            "judge": self.judge,
            "dist_px": round(self.dist_px, 1) if math.isfinite(self.dist_px) else None,
            "dist_m": round(self.dist_m, 4) if math.isfinite(self.dist_m) else None,
            "duration_s": round(time.monotonic() - self.t_start, 2),
            "gt_pose": self.gt_pose,
        }
        self.pub_result.publish(String(data=json.dumps(payload)))
        self.get_logger().info(f"试次结束：{payload}")

    def _publish_mask(self, mask: np.ndarray, stamp):
        # 纯诊断话题：没人订阅就不发，省掉 30Hz × 300KB 的拷贝
        if self.pub_mask.get_subscription_count() == 0:
            return
        msg = Image()
        msg.header.stamp = stamp
        msg.height, msg.width = mask.shape
        msg.encoding = "mono8"
        msg.step = msg.width
        # 和 sim_bridge 同一个坑：给 uint8[] 赋 bytes 会走逐元素校验，慢 500 倍
        msg.data = array.array("B", (mask.astype(np.uint8) * 255).tobytes())
        self.pub_mask.publish(msg)

    @staticmethod
    def _to_rgb(msg: Image):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        try:
            if msg.encoding in ("rgb8", "bgr8"):
                img = buf.reshape(msg.height, msg.width, 3)
                return img[:, :, ::-1] if msg.encoding == "bgr8" else img
            if msg.encoding in ("mono8", "8UC1"):
                return np.repeat(buf.reshape(msg.height, msg.width, 1), 3, axis=2)
        except ValueError:
            return None
        return None


def main():
    rclpy.init()
    node = SuccessDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
