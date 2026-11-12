#!/usr/bin/env python3
"""sim_expert.py — 仿真的脚本专家：生成一条"抓起方块放进杯子"的示范。

它不是策略，也不碰 Isaac。只做一件事：读物体和关节状态，算出一条关节轨
迹，按 30Hz 发到 /joint_commands。剩下的（物理、渲染、成败判定）都由
sim_bridge / success_detector 负责。

为什么单独一个节点而不是塞进 sim_app：动作必须在 ROS 话题上出现，否则
recorder 录不到 action 字段。真机是"人 → /joint_commands → 臂"，仿真必须一致。

依赖（`alicia/driver:1.0` 镜像里都有）：
  rclpy、numpy、scipy、robocore（IK/FK）、synriard（URDF 模型）

话题/服务
  订阅  /joint_states    当前关节角（闭环等到位）
        /object/pose     方块真值（抓取点）
  发布  /joint_commands  30Hz 关节目标（7 维：6 弧度 + 夹爪 0~1）
  服务  /expert/run_once 生成一条（阻塞到结束，返回成败）
        /expert/home     让臂回到复位姿态（调用 /arm/home 之前先用）

    python3 sim_expert.py --ros-args -p cup_xy:="[0.52, 0.14]"
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

JOINT_NAMES = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]


def pose(x, y, z) -> np.ndarray:
    """工具朝下的抓取位姿：z 轴指向世界 -Z，x 轴指 +X。"""
    T = np.eye(4)
    T[:3, :3] = np.diag([1.0, -1.0, -1.0])
    T[:3, 3] = [x, y, z]
    return T


class SimExpert(Node):
    def __init__(self):
        super().__init__("sim_expert")

        self.declare_parameter("robot_version", "v5_6")
        self.declare_parameter("gripper_type", "50mm")
        self.declare_parameter("rate_hz", 30.0)
        # 杯子位置（底座坐标系），和 task.yaml 的 spawn.cup 对应
        self.declare_parameter("cup_xy", [0.52, 0.14])
        self.declare_parameter("cup_height", 0.09)      # 杯口高度
        # 抓取/放置时的抬升量
        self.declare_parameter("approach_m", 0.10)
        self.declare_parameter("grasp_lift_m", 0.15)
        self.declare_parameter("release_m", 0.10)
        # 运动参数：专家要慢一点，动作干净
        self.declare_parameter("joint_speed_rad_s", 0.8)
        self.declare_parameter("arrive_tol_rad", 0.03)
        self.declare_parameter("arrive_timeout_s", 4.0)
        # 扰动：让仿真数据别太"机械"
        self.declare_parameter("pos_noise_m", 0.003)
        self.declare_parameter("joint_noise_rad", 0.004)
        self.declare_parameter("timing_jitter_s", 0.03)

        p = self.get_parameter
        self.rate = float(p("rate_hz").value)
        self.dt = 1.0 / self.rate
        self.cup = [float(v) for v in p("cup_xy").value]
        self.cup_h = float(p("cup_height").value)
        self.approach = float(p("approach_m").value)
        self.grasp_lift = float(p("grasp_lift_m").value)
        self.release_h = float(p("release_m").value)
        self.speed = float(p("joint_speed_rad_s").value)
        self.tol = float(p("arrive_tol_rad").value)
        self.arrive_timeout = float(p("arrive_timeout_s").value)
        self.pos_noise = float(p("pos_noise_m").value)
        self.joint_noise = float(p("joint_noise_rad").value)
        self.jitter = float(p("timing_jitter_s").value)

        # ---- 运动学 --------------------------------------------------------
        from robocore.kinematics import forward_kinematics, inverse_kinematics
        from robocore.modeling import RobotModel
        from synriard import get_model_path

        urdf = get_model_path(
            "Alicia_D",
            version=p("robot_version").value,
            variant=f"gripper_{p('gripper_type').value}",
        )
        self.model = RobotModel(str(urdf), end_link="tool0")
        self.fk = forward_kinematics
        self.ik = inverse_kinematics
        self.rng = np.random.default_rng()

        # ---- 状态 ----------------------------------------------------------
        self.joints: list[float] | None = None
        self.cube: tuple[float, float, float] | None = None
        self.cmd = None                      # 当前要发的 7 维命令
        self.busy = False
        self.result = {"success": False, "reason": "idle"}

        # ---- 接口 ----------------------------------------------------------
        self.pub = self.create_publisher(Float64MultiArray, "/joint_commands", 10)
        self.create_subscription(JointState, "/joint_states", self.on_state, 10)
        self.create_subscription(PoseStamped, "/object/pose", self.on_object, 10)
        cb = ReentrantCallbackGroup()
        self.create_service(Trigger, "/expert/run_once", self.on_run, callback_group=cb)
        self.create_service(Trigger, "/expert/home", self.on_home, callback_group=cb)
        self.cli_home = self.create_client(Trigger, "/arm/home")

        self.create_timer(self.dt, self.publish_cmd)
        self.get_logger().info(
            f"sim_expert 就绪：杯心 ({self.cup[0]:.2f},{self.cup[1]:.2f})，"
            f"速度 {self.speed} rad/s，{self.rate:.0f}Hz"
        )

    # ---------------------------------------------------------------- 回调
    def on_state(self, msg: JointState):
        if len(msg.position) < 7:
            return
        self.joints = [float(v) for v in msg.position[:6]]

    def on_object(self, msg: PoseStamped):
        p = msg.pose.position
        self.cube = (p.x, p.y, p.z)

    def publish_cmd(self):
        if self.cmd is None:
            return
        msg = Float64MultiArray()
        msg.data = [float(v) for v in self.cmd]
        self.pub.publish(msg)

    def on_run(self, _req, resp):
        if self.busy:
            resp.success = False
            resp.message = "already running"
            return resp
        if self.joints is None or self.cube is None:
            resp.success = False
            resp.message = "还没有 /joint_states 或 /object/pose"
            return resp

        self.busy = True
        self.result = {"success": False, "reason": "running"}
        t = threading.Thread(target=self._run_wrapper, daemon=True)
        t.start()
        t.join(timeout=60.0)                  # 一条示范最多一分钟
        self.busy = False

        resp.success = bool(self.result.get("success"))
        resp.message = self.result.get("reason", "")
        return resp

    def on_home(self, _req, resp):
        """转到 /arm/home，由 driver/sim_bridge 走它自己的限位和限速逻辑。"""
        if not self.cli_home.service_is_ready():
            # 极端情况下退回直接发命令
            self.cmd = [0.0] * 6 + [1.0]
            resp.success = False
            resp.message = "/arm/home 不可用，已直接发零位命令"
            return resp
        fut = self.cli_home.call_async(Trigger.Request())
        deadline = time.monotonic() + 5.0
        while not fut.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not fut.done():
            resp.success = False
            resp.message = "/arm/home 超时"
            return resp
        resp.success = True
        resp.message = "homing"
        return resp

    # ---------------------------------------------------------------- 轨迹
    def _run_wrapper(self):
        try:
            self.result = self._run_sequence()
        except Exception as e:                       # noqa: BLE001
            self.result = {"success": False, "reason": f"exception: {e}"}
            self.get_logger().error(f"专家执行失败：{e}")
        self.get_logger().info(f"专家结束：{self.result}")

    def _run_sequence(self):
        assert self.cube is not None and self.joints is not None
        cx, cy, cz = self.cube
        cup_x, cup_y = self.cup

        # 抓取点稍微随机偏移，避免 200 条数据全是同一个位置
        cx += self.rng.normal(0, self.pos_noise)
        cy += self.rng.normal(0, self.pos_noise)

        # 一个 keyframe = (4x4 工具位姿 或 None, 夹爪开度, 说明)
        keys = [
            (pose(cx, cy, cz + self.approach), 1.0, "移动到抓取上方"),
            (pose(cx, cy, cz), 1.0, "下降到抓取位"),
            (pose(cx, cy, cz), 0.0, "闭合夹爪"),
            (pose(cx, cy, cz + self.grasp_lift), 0.0, "抬升"),
            (pose(cup_x, cup_y, self.cup_h + self.release_h), 0.0, "移到杯口上方"),
            (pose(cup_x, cup_y, self.cup_h + self.release_h), 1.0, "松开"),
            (pose(cup_x, cup_y, self.cup_h + self.release_h + 0.05), 1.0, "撤离"),
        ]

        for T, grip, label in keys:
            self.get_logger().info(label)
            if T is not None:
                q = self._solve_ik(T, self.joints)
                if q is None:
                    return {"success": False, "reason": f"IK 无解：{label}"}
                if not self._move_to(q, grip):
                    return {"success": False, "reason": f"未到位：{label}"}
            else:
                if not self._move_to(self.joints, grip):
                    return {"success": False, "reason": f"未到位：{label}"}
            time.sleep(max(0.0, self.rng.normal(0.15, self.jitter)))

        self.cmd = list(self.joints) + [1.0]
        return {"success": True, "reason": "done"}

    def _solve_ik(self, T_target, q0):
        res = self.ik(self.model, T_target, q0, method="dls", multi_start=4)
        if not res.get("success"):
            return None
        return [float(v) for v in res["q"]]

    def _move_to(self, q_target, grip) -> bool:
        """关节空间插值过去，30Hz 发；等到位（不是等固定时长）。"""
        q_start = list(self.joints) if self.joints else list(q_target)
        q_target = list(q_target) + self.rng.normal(0, self.joint_noise, 6).tolist()
        dist = max(abs(a - b) for a, b in zip(q_start, q_target))
        n = max(1, int(math.ceil(dist / (self.speed * self.dt))))

        for i in range(1, n + 1):
            k = i / n
            self.cmd = [a + (b - a) * k for a, b in zip(q_start, q_target)] + [grip]
            time.sleep(self.dt)

        # 等真的到位：仿真步进和真实时间不完全对齐，不能只按时间算
        deadline = time.monotonic() + self.arrive_timeout
        while time.monotonic() < deadline:
            if self.joints is None:
                return False
            err = max(abs(a - b) for a, b in zip(self.joints, q_target))
            if err <= self.tol:
                return True
            self.cmd = q_target + [grip]
            time.sleep(self.dt)
        return False


def main():
    rclpy.init()
    node = SimExpert()
    exe = MultiThreadedExecutor(num_threads=4)
    exe.add_node(node)
    try:
        exe.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
