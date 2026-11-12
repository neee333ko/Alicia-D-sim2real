#!/usr/bin/env python3
"""driver_node.py — 真机机械臂唯一的 ROS 2 出入口。

职责只有两件：
  1. 订阅 /joint_commands（7 维：6 关节弧度 + 夹爪 0~1），调 SDK 下发
  2. 定时读 SDK 的关节状态，发 /joint_states

整个项目里只有这一个文件允许 import alicia_d_sdk。上层（policy / recorder /
eval）一律通过话题通信，不许碰串口和厂商 SDK。

为什么读写都在一个定时器里：串口是独占的，读帧和写帧交错会串包。所以不做成
"回调里写、另一个线程里读"，而是同一个线程按固定频率先读后写。

单独跑（不需要 colcon）：
    python3 driver_node.py --ros-args -p port:=/dev/ttyUSB1 -p dry_run:=true

真机上第一次跑务必先 dry_run:=true，确认 /joint_states 有数、命令能限住，
再改 false。
"""

from __future__ import annotations

import json
import math
import os
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

try:
    import alicia_d_sdk
except ImportError:      # 允许 dry_run 在没有 SDK 的机器上跑
    alicia_d_sdk = None


JOINT_NAMES = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]
GRIPPER_NAME = "gripper"


class DriverNode(Node):
    def __init__(self):
        super().__init__("driver_node")

        # ---- 参数 ----------------------------------------------------------
        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baudrate", 1000000)
        self.declare_parameter("robot_version", "v5_6")
        self.declare_parameter("gripper_type", "50mm")
        self.declare_parameter("speed_deg_s", 90.0)      # 第一次试机建议降到 20
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("dry_run", False)
        # 复位姿态：{"joints": [6 个弧度], "gripper": 0~1}
        # 由 /arm/home 读取，每次调用都重新读文件，方便标完直接改
        self.declare_parameter("reset_pose_file", "poses/reset.json")
        # 认为"到位"的容差
        self.declare_parameter("home_tolerance_rad", 0.02)

        # 单个关节的最大速度（弧度/秒）。这是安全上限，不是调速器：
        # 正常轨迹应该远低于它，只有策略吐出跳变值时才被裁到。
        # 原来的写法是"每帧最大增量 0.05 rad"，物理含义会随 rate_hz 变——
        # 30Hz 下等于 86°/s，50Hz 下等于 143°/s，改频率就悄悄改了限速。
        # 现在换成速度定义，每帧限幅 = max_speed_rad_s / rate_hz。
        self.declare_parameter("max_speed_rad_s", 3.0)
        # 超过这么久没收到新命令就不再下发，臂保持当前位置
        self.declare_parameter("cmd_timeout_s", 0.2)
        # 6 个关节的绝对限位，单位度，按 min,max 顺序写 12 个数。
        # 默认值来自厂商 URDF：synriard/urdf/Alicia_D_v5_6/Alicia_D_v5_6_gripper_50mm.urdf
        #   J1 ±2.749  J2 ±1.570  J3 -0.780~3.142  J4 ±2.790  J5 ±1.570  J6 ±3.142 (rad)
        # 这是模型标称值，真实伺服可能略小。要收紧就传参覆盖。
        # 用字符串而不是数组：空数组在 ROS 2 里会被推成 BYTE_ARRAY，传浮点会直接报类型错。
        self.declare_parameter(
            "joint_limits_deg",
            "-157.5,157.5,-90,90,-44.7,180,-159.9,159.9,-90,90,-180,180",
        )

        p = self.get_parameter
        self.port = p("port").value
        self.dry_run = p("dry_run").value
        self.rate_hz = float(p("rate_hz").value)
        self.max_speed = float(p("max_speed_rad_s").value)
        self.max_delta = self.max_speed / self.rate_hz      # 每帧允许的增量
        self.cmd_timeout = float(p("cmd_timeout_s").value)
        self.limits = self._parse_limits(p("joint_limits_deg").value)
        self.reset_pose_file = p("reset_pose_file").value
        self.home_tol = float(p("home_tolerance_rad").value)

        # ---- ROS 接口 ------------------------------------------------------
        self.pub_state = self.create_publisher(JointState, "/joint_states", 10)
        self.sub_cmd = self.create_subscription(
            Float64MultiArray, "/joint_commands", self._on_command, 10
        )

        # ---- 硬件 ----------------------------------------------------------
        self.robot = None
        self._target = None          # 最近一次收到的目标（6 关节弧度 + 夹爪 0~1）
        self._target_gripper = 0.0
        self._homing = False        # 正在回复位位姿：看门狗不生效
        self._target_t = 0.0         # 收到时刻，用于看门狗
        self._last_sent = None       # 上一周期实际下发的关节角，用于增量限幅
        self._connect()

        self.srv_home = self.create_service(Trigger, "/arm/home", self._on_home)

        # 定时器：读状态 + 发命令，全在这一个线程里完成
        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)
        self.get_logger().info(
            f"driver_node 启动：port={self.port} rate={self.rate_hz}Hz "
            f"限速={math.degrees(self.max_speed):.0f}°/s"
            f"（每帧 {self.max_delta:.4f} rad） "
            f"dry_run={self.dry_run} 绝对限位={'有' if self.limits else '未配置'}"
        )

    # ------------------------------------------------------------------ 硬件
    def _connect(self):
        if self.dry_run:
            self.get_logger().warn("dry_run 模式：不打开串口，只回环假数据")
            return
        if alicia_d_sdk is None:
            raise RuntimeError("没有安装 alicia_d_sdk，真机模式起不来")

        p = self.get_parameter
        self.robot = alicia_d_sdk.create_robot(
            port=self.port,
            baudrate=int(p("baudrate").value),
            robot_version=p("robot_version").value,
            robot_type="follower",
            gripper_type=p("gripper_type").value,
            speed_deg_s=float(p("speed_deg_s").value),
        )
        if not self.robot.connect():
            # 连不上就退出，不要让它带着半死状态继续跑
            raise RuntimeError(f"串口连接失败：{self.port}")
        self.get_logger().info(f"已连接 {self.port}")

    def _parse_limits(self, raw):
        if not raw or not raw.strip():
            return None
        try:
            vals = [float(x) for x in raw.replace(";", ",").split(",") if x.strip()]
        except ValueError as e:
            raise ValueError(f"joint_limits_deg 解析失败：{raw!r}") from e
        if len(vals) != 12:
            raise ValueError(f"joint_limits_deg 需要 12 个数（6 组 min,max），收到 {len(vals)} 个")
        return [(math.radians(vals[2 * i]), math.radians(vals[2 * i + 1])) for i in range(6)]

    # ------------------------------------------------------------------ 回调
    def _on_command(self, msg: Float64MultiArray):
        if len(msg.data) != 7:
            self.get_logger().error(
                f"/joint_commands 需要 7 个值，收到 {len(msg.data)} 个，忽略",
                throttle_duration_sec=5.0,
            )
            return
        joints = [float(v) for v in msg.data[:6]]
        gripper = float(msg.data[6])

        # NaN/Inf 必须在这里挡掉。往下走的话 abs(nan) > max_delta 恒为 False，
        # 限幅和限位都不会生效，NaN 会一路写进串口帧发给舵机。
        if not all(math.isfinite(v) for v in joints) or not math.isfinite(gripper):
            self.get_logger().error(
                "命令里含 NaN/Inf，本帧丢弃", throttle_duration_sec=1.0
            )
            return

        self._target = joints
        self._target_gripper = gripper
        self._target_t = time.monotonic()
        if self._homing:
            # 策略开始下发了，回复位动作让位
            self._homing = False
            self.get_logger().info("收到关节命令，取消 /arm/home")

    # ------------------------------------------------------------------ 复位
    def _on_home(self, req, resp):
        """回到 poses/reset.json 里的姿态。

        不阻塞：把目标塞给常规的下发循环，让它按增量限幅慢慢走过去，服务立刻返回。
        阻塞的话会把 30Hz 控制循环卡住，而且调用方还可能超时。
        """
        try:
            pose = self._load_reset_pose()
        except Exception as e:                      # noqa: BLE001
            resp.success = False
            resp.message = str(e)
            self.get_logger().error(f"/arm/home 失败：{e}")
            return resp

        joints, gripper = pose

        # 限位检查：配错的复位姿态不能让机械臂自己撞过去
        if self.limits is not None:
            for i, (lo, hi) in enumerate(self.limits):
                if not lo <= joints[i] <= hi:
                    resp.success = False
                    resp.message = (
                        f"Joint{i + 1}={joints[i]:.3f} rad 超出限位 [{lo:.3f}, {hi:.3f}]"
                    )
                    self.get_logger().error(resp.message)
                    return resp

        self._target = list(joints)
        self._target_gripper = gripper
        self._target_t = time.monotonic()
        self._homing = True
        resp.success = True
        resp.message = "moving to reset pose"
        self.get_logger().info(
            "开始回复位：" + ", ".join(f"{v:.3f}" for v in joints) + f"，夹爪 {gripper:.2f}"
        )
        return resp

    def _load_reset_pose(self):
        path = self.reset_pose_file
        if not path or not os.path.isfile(path):
            raise FileNotFoundError(f"找不到复位姿态文件：{path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        joints = [float(v) for v in data["joints"]]
        gripper = float(data.get("gripper", 0.0))
        if len(joints) != 6:
            raise ValueError(f"joints 需要 6 个值，文件里是 {len(joints)} 个")
        if not all(math.isfinite(v) for v in joints) or not math.isfinite(gripper):
            raise ValueError("复位姿态里有 NaN/Inf")
        return joints, max(0.0, min(1.0, gripper))

    def _home_reached(self, current):
        return all(abs(a - b) <= self.home_tol for a, b in zip(current, self._target))

    # ------------------------------------------------------------------ 主循环
    def _tick(self):
        state = self._read_state()
        if state is None:
            return
        joints, gripper = state

        self._publish_state(joints, gripper)
        self._maybe_send(joints)

    def _read_state(self):
        if self.dry_run:
            base = self._last_sent or [0.0] * 6
            return list(base), 0.0

        try:
            joints = self.robot.get_joints()
            gripper = self.robot.get_gripper()
        except Exception as e:                      # noqa: BLE001
            self.get_logger().error(f"读状态失败：{e}", throttle_duration_sec=2.0)
            return None

        if joints is None or len(joints) != 6:
            self.get_logger().warn("读到空状态，跳过本周期", throttle_duration_sec=2.0)
            return None
        if not all(math.isfinite(v) for v in joints):
            self.get_logger().error(
                "读到 NaN/Inf 关节角，跳过本周期", throttle_duration_sec=2.0
            )
            return None
        return [float(v) for v in joints], float(gripper) if gripper is not None else 0.0

    def _publish_state(self, joints, gripper):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES + [GRIPPER_NAME]
        # 夹爪在 ROS 侧统一用 0~1，SDK 那边是 0~100，转换只在这里做
        msg.position = joints + [gripper / 100.0]
        self.pub_state.publish(msg)

    def _maybe_send(self, current):
        if self._target is None:
            return

        if self._homing:
            # 复位是一次性长动作（可能几秒），期间没有新命令是正常的，
            # 所以看门狗让路；到位后自动退出 homing。
            if self._home_reached(current):
                self._homing = False
                self.get_logger().info("已到达复位姿态")
            self._target_t = time.monotonic()

        if not self._homing and time.monotonic() - self._target_t > self.cmd_timeout:
            # 看门狗：命令断了就停手，臂停在上一个位置
            self.get_logger().warn(
                f"超过 {self.cmd_timeout}s 没有新命令，停止下发", throttle_duration_sec=2.0
            )
            return
        if self.dry_run:
            # dry_run 也走一遍限幅，这样没有硬件也能验证限位配得对不对
            self._last_sent, clamped = self._guard(current, self._target)
            if clamped:
                self.get_logger().warn("命令越界，已裁剪", throttle_duration_sec=1.0)
            return

        cmd, clamped = self._guard(current, self._target)
        if clamped:
            self.get_logger().warn("命令越界，已裁剪", throttle_duration_sec=1.0)

        try:
            self.robot.set_joint_target_no_wait(cmd, joint_format="rad")
            self.robot.set_gripper_target(
                value=max(0.0, min(1.0, self._target_gripper)) * 100.0,
                wait_for_completion=False,
            )
        except Exception as e:                      # noqa: BLE001
            self.get_logger().error(f"下发失败：{e}", throttle_duration_sec=2.0)
            return
        self._last_sent = cmd

    def _guard(self, current, target):
        """绝对限位 + 单周期增量限幅。返回 (可下发角度, 是否被裁剪)。"""
        out, clamped = [], False
        for i, (cur, tgt) in enumerate(zip(current, target)):
            if self.limits is not None:
                lo, hi = self.limits[i]
                tgt = min(max(tgt, lo), hi)
            delta = tgt - cur
            if abs(delta) > self.max_delta:
                tgt = cur + math.copysign(self.max_delta, delta)
                clamped = True
            out.append(tgt)
        return out, clamped

    # ------------------------------------------------------------------ 退出
    def shutdown(self):
        if self.robot is not None:
            try:
                self.robot.disconnect()
            except Exception:                       # noqa: BLE001
                pass
            self.get_logger().info("已断开串口")


def main():
    rclpy.init()
    node = DriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
