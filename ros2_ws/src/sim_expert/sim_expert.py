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
        /object/pose     杯子与方块真值（alicia_interfaces/ObjectPose）
  发布  /joint_commands  30Hz 关节目标（7 维：6 弧度 + 夹爪 0~1）
        /expert/status   5Hz JSON（stamp/liveness/readiness/state/episode）
  动作  /expert/run_once 生成一条示范（RunOnce，可取消，带进度反馈）

    python3 sim_expert.py --ros-args -p rate_hz:=30.0
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import rclpy
from alicia_interfaces.action import RunOnce
from alicia_interfaces.msg import ObjectPose
from rclpy.action import ActionServer, CancelResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

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
        # 抓取/放置时的抬升量（相对方块顶面 / 杯口）
        self.declare_parameter("approach_m", 0.10)
        self.declare_parameter("grasp_lift_m", 0.15)
        self.declare_parameter("release_m", 0.10)
        # 运动参数：专家要慢一点，动作干净
        self.declare_parameter("joint_speed_rad_s", 0.8)
        self.declare_parameter("arrive_tol_rad", 0.03)
        self.declare_parameter("arrive_timeout_s", 4.0)
        # 夹爪开合是物理过程，指令变了要留时间让它夹/放到实物
        self.declare_parameter("grip_hold_s", 0.5)
        # IK 解与当前关节角的单轴最大偏差，超过就认为是解到了另一个分支
        self.declare_parameter("ik_max_jump_rad", 1.0)
        # 扰动：让仿真数据别太"机械"
        self.declare_parameter("pos_noise_m", 0.003)
        self.declare_parameter("joint_noise_rad", 0.004)
        self.declare_parameter("timing_jitter_s", 0.03)

        p = self.get_parameter
        self.rate = float(p("rate_hz").value)
        self.dt = 1.0 / self.rate
        self.approach = float(p("approach_m").value)
        self.grasp_lift = float(p("grasp_lift_m").value)
        self.release_h = float(p("release_m").value)
        self.speed = float(p("joint_speed_rad_s").value)
        self.tol = float(p("arrive_tol_rad").value)
        self.arrive_timeout = float(p("arrive_timeout_s").value)
        self.grip_hold = float(p("grip_hold_s").value)
        self.max_jump = float(p("ik_max_jump_rad").value)
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
        self.last_state_t = 0.0               # 上次收到 /joint_states 的时刻
        self.cube: tuple[float, float, float] | None = None   # 方块几何中心
        self.cube_edge: float | None = None                   # 方块边长
        self.cup: tuple[float, float, float] | None = None    # 杯口中心
        self.cmd = None                      # 当前要发的 7 维命令
        self.busy = False
        self.goal = None                     # 正在跑的 action goal，供取消检查

        # ---- 接口 ----------------------------------------------------------
        self.pub = self.create_publisher(Float64MultiArray, "/joint_commands", 10)
        # 状态用 latched：orchestrator 后启动也能立刻拿到最后一条，不必等下一拍
        self.pub_status = self.create_publisher(
            String,
            "/expert/status",
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            ),
        )
        self.create_subscription(JointState, "/joint_states", self.on_state, 10)
        self.create_subscription(ObjectPose, "/object/pose", self.on_object, 10)
        cb = ReentrantCallbackGroup()
        self.ac = ActionServer(
            self,
            RunOnce,
            "/expert/run_once",
            self.on_goal,
            callback_group=cb,
            # rclpy 默认回 CANCEL_REJECTED，不显式接受的话 /orch/abort 会静默失效
            cancel_callback=lambda _req: CancelResponse.ACCEPT,
        )

        self.create_timer(self.dt, self.publish_cmd)
        # orchestrator 的 status_timeout_s=3.0，5Hz 足够，也留足网络抖动余量
        self.create_timer(0.2, self._publish_status)
        self.get_logger().info(
            f"sim_expert 就绪：速度 {self.speed} rad/s，{self.rate:.0f}Hz，"
            "等 /object/pose 给出杯子与方块位姿"
        )

    # ---------------------------------------------------------------- 回调
    def on_state(self, msg: JointState):
        if len(msg.position) < 7:
            return
        self.joints = [float(v) for v in msg.position[:6]]
        self.last_state_t = time.monotonic()

    def on_object(self, msg: ObjectPose):
        """收下杯子与方块真值。只在生成轨迹时读一次，运动中不再跟随物体。"""
        self.cube = (msg.cube_pose.x, msg.cube_pose.y, msg.cube_pose.z)
        self.cube_edge = float(msg.cube_edge_m)
        self.cup = (msg.cup_pose.x, msg.cup_pose.y, msg.cup_pose.z)

    def publish_cmd(self):
        if self.cmd is None:
            return
        msg = Float64MultiArray()
        msg.data = [float(v) for v in self.cmd]
        self.pub.publish(msg)

    def _publish_status(self):
        """给 orchestrator 的就绪检查用：忙时 readiness=false，别让新 goal 排队。"""
        payload = {
            "stamp": time.time(),
            "liveness": True,
            "readiness": not self.busy,
            "state": "running" if self.busy else "idle",
            "episode": int(self.goal.request.id) if self.goal is not None else -1,
            "has_pose": None not in (self.cube, self.cup, self.cube_edge),
        }
        self.pub_status.publish(String(data=json.dumps(payload)))

    def on_goal(self, goal_handle):
        """执行一条示范。execute 回调，跑在 executor 的独立线程里。"""
        if self.busy:
            goal_handle.abort()
            return RunOnce.Result(success=False, message="正在执行上一条")
        if None in (self.joints, self.cube, self.cup, self.cube_edge):
            goal_handle.abort()
            return RunOnce.Result(
                success=False, message="还没有 /joint_states 或 /object/pose（需要杯子、方块位姿和边长）"
            )

        self.busy = True
        self.goal = goal_handle
        t0 = time.monotonic()
        try:
            res = self._run_sequence()
        except Exception as e:                       # noqa: BLE001
            res = {"success": False, "reason": f"exception: {e}"}
            self.get_logger().error(f"专家执行失败：{e}")

        if goal_handle.is_cancel_requested:
            res = {"success": False, "reason": "canceled"}
            goal_handle.canceled()
        elif res.get("success"):
            goal_handle.succeed()
        else:
            goal_handle.abort()

        self.busy = False
        self.goal = None
        self.get_logger().info(f"专家结束：{res}")
        return RunOnce.Result(
            success=bool(res.get("success")),
            message=res.get("reason", ""),
            duration_s=float(time.monotonic() - t0),
        )

    # ---------------------------------------------------------------- 轨迹

    def _run_sequence(self):
        assert self.cube is not None and self.joints is not None
        assert self.cup is not None and self.cube_edge is not None
        cx, cy, cz = self.cube
        cup_x, cup_y, cup_z = self.cup
        # 抬升量从方块顶面算，而不是中心：避免夹爪贴着桌面闭合
        top = cz + self.cube_edge / 2.0

        # 抓取点稍微随机偏移，避免 200 条数据全是同一个位置
        cx += self.rng.normal(0, self.pos_noise)
        cy += self.rng.normal(0, self.pos_noise)

        # 一个 keyframe = (4x4 工具位姿 或 None, 夹爪开度, 说明)
        keys = [
            (pose(cx, cy, top + self.approach), 1.0, "移动到抓取上方"),
            (pose(cx, cy, cz), 1.0, "下降到抓取位"),
            (pose(cx, cy, cz), 0.0, "闭合夹爪"),
            (pose(cx, cy, top + self.grasp_lift), 0.0, "抬升"),
            (pose(cup_x, cup_y, cup_z + self.release_h), 0.0, "移到杯口上方"),
            (pose(cup_x, cup_y, cup_z + self.release_h), 1.0, "松开"),
            (pose(cup_x, cup_y, cup_z + self.release_h + 0.05), 1.0, "撤离"),
        ]

        prev_grip = None
        total = len(keys)
        for i, (T, grip, label) in enumerate(keys):
            # keyframe 之间也查一次，取消后不再白算 IK
            if self.goal is not None and self.goal.is_cancel_requested:
                return {"success": False, "reason": "canceled"}
            self.get_logger().info(label)
            # 只在夹爪开度变化的那一步补保持时间，纯位移不拖慢
            hold = self.grip_hold if prev_grip is not None and grip != prev_grip else 0.0
            if T is not None:
                q = self._solve_ik(T, self.joints)
                if q is None:
                    return {"success": False, "reason": f"IK 无解：{label}"}
                ok = self._move_to(q, grip, hold)
            else:
                ok = self._move_to(self.joints, grip, hold)
            if not ok:
                return {"success": False, "reason": f"未到位：{label}"}
            prev_grip = grip

            if self.goal is not None:
                fb = RunOnce.Feedback()
                fb.phase = label
                fb.progress = float((i + 1) / total)
                try:
                    self.goal.publish_feedback(fb)
                except Exception:                    # noqa: BLE001
                    # cancel 与 feedback 有竞态，此时发布器可能已失效
                    pass

            time.sleep(max(0.0, self.rng.normal(0.15, self.jitter)))

        self.cmd = list(self.joints) + [1.0]
        return {"success": True, "reason": "done"}

    def _solve_ik(self, T_target, q0):
        # 固定种子：multi_start 的扰动可复现，同一个位姿不会这次左支下次右支
        res = self.ik(
            self.model, T_target, q0, method="dls", multi_start=4, random_seed=0
        )
        if not res.get("success"):
            return None
        q = [float(v) for v in res["q"]]
        # multi_start 只在基础解失败时按残差挑解，可能挑到另一个 IK 分支；
        # 种子是当前实测角度，正常解离得很近，偏这么多说明换支了，插值会扫出大弧线
        jump = max(abs(a - b) for a, b in zip(q, q0))
        if jump > self.max_jump:
            self.get_logger().warn(f"IK 跳解：单轴偏差 {jump:.2f} rad，放弃这一段")
            return None
        return q

    def _move_to(self, q_target, grip, hold: float = 0.0) -> bool:
        """关节空间插值过去，30Hz 发；等到位（不是等固定时长）。

        hold：到位后继续保持同一夹爪开度的时间。位姿不变时 dist=0、n=1，
        插值只发一帧，靠这个把"夹爪合拢所需的时间"补上。
        """
        q_start = list(self.joints) if self.joints else list(q_target)
        # 关节噪声要逐元素叠加。写成 list + list 会拼成 12 维，指令直接多出一截。
        noise = self.rng.normal(0, self.joint_noise, len(q_target))
        q_target = [float(b + d) for b, d in zip(q_target, noise)]
        dist = max(abs(a - b) for a, b in zip(q_start, q_target))
        n = max(1, int(math.ceil(dist / (self.speed * self.dt))))

        for i in range(1, n + 1):
            # 取消是协作式的：只有这里按帧查，/orch/abort 才能真正打断动作
            if self.goal is not None and self.goal.is_cancel_requested:
                return False
            k = i / n
            self.cmd = [a + (b - a) * k for a, b in zip(q_start, q_target)] + [grip]
            time.sleep(self.dt)

        # 等真的到位：仿真步进和真实时间不完全对齐，不能只按时间算
        deadline = time.monotonic() + self.arrive_timeout
        arrived = False
        while time.monotonic() < deadline:
            if self.goal is not None and self.goal.is_cancel_requested:
                return False
            if self.joints is None:
                return False
            # 断流时 self.joints 会冻在旧值，err 可能恰好很小而被误判到位，
            # 结果是录到一条"机械臂没动"的假示范。所以先确认反馈还在更新。
            if time.monotonic() - self.last_state_t > 0.2:
                self.get_logger().error("超过 0.2s 没收到 /joint_states，无法确认到位")
                return False
            err = max(abs(a - b) for a, b in zip(self.joints, q_target))
            if err <= self.tol:
                arrived = True
                break
            self.cmd = q_target + [grip]
            time.sleep(self.dt)
        if not arrived:
            return False

        # 夹爪指令是位置式的，物理合拢要时间；姿态已到位，原地保持即可
        if hold > 0.0:
            deadline = time.monotonic() + hold
            while time.monotonic() < deadline:
                if self.goal is not None and self.goal.is_cancel_requested:
                    return False
                self.cmd = list(q_target) + [grip]
                time.sleep(self.dt)
        return True


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
