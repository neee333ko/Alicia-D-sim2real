"""sim_expert 的单元测试。

SimExpert.__init__ 会加载 robocore / synriard 的 URDF 模型，代价高且要求 driver 镜像里的
私有包。这里同样用 object.__new__ 绕过 __init__，只注入被测函数用到的字段，测轨迹逻辑
本身：IK 守卫、插值、到位判定、取消、反馈。

时间被打了桩：`sim_expert.time.sleep` 换成记录 + 可选“理想跟踪”，否则一条轨迹要跑几十秒。
"""

from __future__ import annotations

import time
import types

import numpy as np
import pytest

sim_expert = pytest.importorskip("sim_expert", reason="需要 ROS 2 与 alicia_interfaces")


class FakeLogger:
    def __init__(self):
        self.records: list[tuple[str, str]] = []

    def info(self, msg, **_kw):
        self.records.append(("info", msg))

    def warn(self, msg, **_kw):
        self.records.append(("warn", msg))

    def error(self, msg, **_kw):
        self.records.append(("error", msg))

    def debug(self, msg, **_kw):
        self.records.append(("debug", msg))

    def messages(self, level: str) -> list[str]:
        return [m for lv, m in self.records if lv == level]


class FakeGoal:
    """够用的 goal handle：记录 succeed/abort/canceled 调用和反馈。"""

    def __init__(self, cancel: bool = False, feedback_raises: bool = False):
        self.is_cancel_requested = cancel
        self.request = types.SimpleNamespace(id=7)
        self.calls: list[str] = []
        self.fbs: list = []
        self.feedback_raises = feedback_raises

    def succeed(self):
        self.calls.append("succeed")

    def abort(self):
        self.calls.append("abort")

    def canceled(self):
        self.calls.append("canceled")

    def publish_feedback(self, fb):
        if self.feedback_raises:
            # 取消瞬间发布器会失效，真实 rclpy 会抛异常
            raise RuntimeError("feedback publisher is invalid")
        self.fbs.append(fb)


@pytest.fixture
def expert(monkeypatch):
    node = object.__new__(sim_expert.SimExpert)

    logger = FakeLogger()
    node.get_logger = lambda: logger
    node.logger = logger

    node.rate = 100.0
    node.dt = 0.01
    node.speed = 1.0
    node.tol = 0.03
    node.arrive_timeout = 0.05
    node.grip_hold = 0.05
    node.max_jump = 1.0
    node.approach = 0.10
    node.grasp_lift = 0.15
    node.release_h = 0.10
    node.pos_noise = 0.0
    node.joint_noise = 0.0
    node.jitter = 0.0
    node.rng = np.random.default_rng(0)

    node.joints = [0.0] * 6
    node.last_state_t = time.monotonic()
    node.cmd = None
    node.busy = False
    node.goal = None
    node.model = object()

    # 方块中心 z=0.03、边长 0.04 → 顶面 0.05；杯口 z=0.04
    node.cube = (0.50, 0.0, 0.03)
    node.cube_edge = 0.04
    node.cup = (0.60, 0.0, 0.04)

    node.sleep_calls: list[float] = []
    node.track = True          # True：命令发出后关节立刻跟上（理想伺服）

    def fake_sleep(seconds):
        node.sleep_calls.append(seconds)
        if node.track and node.cmd is not None:
            node.joints = [float(v) for v in node.cmd[:6]]
            node.last_state_t = time.monotonic()

    monkeypatch.setattr(sim_expert.time, "sleep", fake_sleep)

    yield node


def _ik_ok(q=None, captured: list | None = None, on_call=None):
    """造一个假的 inverse_kinematics：总是成功，返回 q。"""

    def fake(model, T, q0, **kw):
        if captured is not None:
            captured.append((T, dict(kw)))
        if on_call is not None:
            on_call()
        return {"success": True, "q": list(q if q is not None else [0.0] * 6)}

    return fake


# ---------------------------------------------------------------- 状态回调


def test_on_state_ignores_incomplete_message(expert):
    msg = types.SimpleNamespace(position=[0.1] * 6, name=[])
    before = expert.last_state_t
    expert.on_state(msg)
    assert expert.joints == [0.0] * 6
    assert expert.last_state_t == before


def test_on_state_takes_first_six_and_stamps_time(expert):
    expert.last_state_t = 0.0
    msg = types.SimpleNamespace(position=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.9], name=[])
    expert.on_state(msg)
    assert expert.joints == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert expert.last_state_t > 0.0


def test_on_object_stores_object_fields(expert):
    msg = types.SimpleNamespace(
        cube_pose=types.SimpleNamespace(x=0.41, y=-0.02, z=0.021),
        cup_pose=types.SimpleNamespace(x=0.62, y=0.03, z=0.045),
        cube_edge_m=0.038,
    )
    expert.on_object(msg)
    assert expert.cube == pytest.approx((0.41, -0.02, 0.021))
    assert expert.cup == pytest.approx((0.62, 0.03, 0.045))
    assert expert.cube_edge == pytest.approx(0.038)


# ---------------------------------------------------------------- IK


def test_solve_ik_returns_solution(expert):
    expert.ik = _ik_ok([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    q = expert._solve_ik(np.eye(4), [0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert q == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])


def test_solve_ik_uses_dls_multi_start_and_fixed_seed(expert):
    captured = []
    expert.ik = _ik_ok(captured=captured)
    expert._solve_ik(np.eye(4), [0.0] * 6)
    _, kwargs = captured[0]
    assert kwargs["method"] == "dls"
    assert kwargs["multi_start"] == 4
    # 固定种子：同一位姿每次失败回退都选同一批扰动，数据可复现
    assert kwargs["random_seed"] == 0


def test_solve_ik_returns_none_when_solver_fails(expert):
    expert.ik = lambda *a, **kw: {"success": False, "q": [0.0] * 6}
    assert expert._solve_ik(np.eye(4), [0.0] * 6) is None


def test_solve_ik_rejects_branch_jump(expert):
    # q0 全是 0，解出来第 3 轴偏 1.5 rad → 判定换支
    expert.ik = _ik_ok([0.0, 0.0, 1.5, 0.0, 0.0, 0.0])
    assert expert._solve_ik(np.eye(4), [0.0] * 6) is None
    assert any("跳解" in m for m in expert.logger.messages("warn"))


def test_solve_ik_accepts_jump_below_threshold(expert):
    expert.ik = _ik_ok([0.0, 0.0, 0.9, 0.0, 0.0, 0.0])
    assert expert._solve_ik(np.eye(4), [0.0] * 6) == pytest.approx(
        [0.0, 0.0, 0.9, 0.0, 0.0, 0.0]
    )


# ---------------------------------------------------------------- _move_to


def test_move_to_interpolates_to_target_with_grip(expert):
    target = [0.2] * 6
    assert expert._move_to(target, 0.0) is True
    assert expert.cmd == pytest.approx(target + [0.0])
    # 0.2 rad / (1.0 rad/s * 0.01s) = 20 帧
    assert len(expert.sleep_calls) >= 20
    # 指令必须是 7 维：6 关节 + 夹爪
    assert len(expert.cmd) == 7


def test_move_to_respects_speed_limit(expert):
    expert.speed = 0.5
    expert._move_to([0.1] * 6, 1.0)
    # 0.1 / (0.5 * 0.01) = 20 帧，至少这么多拍
    assert len(expert.sleep_calls) >= 20


def test_move_to_returns_false_on_timeout(expert):
    expert.track = False        # 关节永远不动
    expert.arrive_timeout = 0.02
    assert expert._move_to([0.5] * 6, 1.0) is False


def test_move_to_returns_false_when_state_is_stale(expert):
    expert.track = False
    expert.last_state_t = time.monotonic() - 1.0
    assert expert._move_to([0.0] * 6, 1.0) is False
    assert any("没收到 /joint_states" in m for m in expert.logger.messages("error"))


def test_move_to_returns_false_immediately_when_canceled(expert):
    expert.goal = FakeGoal(cancel=True)
    assert expert._move_to([0.3] * 6, 1.0) is False
    # 取消在插值第一帧就被发现，不该继续发指令
    assert expert.cmd is None


def test_move_to_hold_keeps_pose_and_grip(expert):
    expert.goal = FakeGoal()
    target = [0.1] * 6
    assert expert._move_to(target, 0.0, hold=0.05) is True
    # hold 期间持续发同一目标
    assert expert.cmd == pytest.approx(target + [0.0])
    assert expert.joints == pytest.approx(target)


def test_move_to_hold_aborts_on_cancel(expert, monkeypatch):
    goal = FakeGoal()
    expert.goal = goal
    state = {"n": 0}

    def fake_sleep(seconds):
        expert.sleep_calls.append(seconds)
        expert.joints = [float(v) for v in expert.cmd[:6]]
        expert.last_state_t = time.monotonic()
        state["n"] += 1
        if state["n"] > 15:
            # 插值 10 帧之后进入 hold，此时收到取消
            goal.is_cancel_requested = True

    monkeypatch.setattr(sim_expert.time, "sleep", fake_sleep)
    assert expert._move_to([0.1] * 6, 0.0, hold=0.5) is False


# ---------------------------------------------------------------- _run_sequence


def test_run_sequence_happy_path(expert):
    expert.goal = FakeGoal()
    expert.ik = _ik_ok([0.0] * 6)
    res = expert._run_sequence()
    assert res["success"] is True
    assert res["reason"] == "done"
    # 7 个 keyframe，每个发一次反馈，进度单调到 1.0
    assert len(expert.goal.fbs) == 7
    progress = [fb.progress for fb in expert.goal.fbs]
    assert progress == pytest.approx([1 / 7, 2 / 7, 3 / 7, 4 / 7, 5 / 7, 6 / 7, 1.0])
    # 结束时夹爪回到张开
    assert expert.cmd[-1] == pytest.approx(1.0)


def test_run_sequence_targets_are_computed_from_object_pose(expert):
    captured = []
    expert.goal = FakeGoal()
    expert.ik = _ik_ok([0.0] * 6, captured=captured)
    expert._run_sequence()

    # 0 接近点：方块顶面 + approach = 0.03 + 0.02 + 0.10
    assert captured[0][0][2, 3] == pytest.approx(0.15)
    # 1 抓取点：方块中心
    assert captured[1][0][2, 3] == pytest.approx(0.03)
    # 3 抬升：方块顶面 + grasp_lift = 0.03 + 0.02 + 0.15
    assert captured[3][0][2, 3] == pytest.approx(0.20)
    # 4 杯口上方：杯口 z + release = 0.04 + 0.10
    assert captured[4][0][2, 3] == pytest.approx(0.14)
    # x/y 来自方块与杯子的真值
    assert captured[0][0][0, 3] == pytest.approx(0.50)
    assert captured[4][0][0, 3] == pytest.approx(0.60)


def test_run_sequence_fails_when_ik_unsolvable(expert):
    expert.goal = FakeGoal()
    expert.ik = lambda *a, **kw: {"success": False, "q": [0.0] * 6}
    res = expert._run_sequence()
    assert res["success"] is False
    assert res["reason"].startswith("IK 无解")


def test_run_sequence_fails_when_move_times_out(expert):
    expert.goal = FakeGoal()
    expert.track = False
    expert.arrive_timeout = 0.01
    expert.ik = _ik_ok([0.5] * 6)
    res = expert._run_sequence()
    assert res["success"] is False
    assert res["reason"].startswith("未到位")


def test_run_sequence_stops_between_keyframes_when_canceled(expert):
    goal = FakeGoal()
    expert.goal = goal
    expert.ik = _ik_ok([0.0] * 6)
    moves = []

    def fake_move(q, grip, hold=0.0):
        moves.append(grip)
        if len(moves) == 1:
            # 第一段正常跑完，之后才收到取消：验证循环顶部的检查
            goal.is_cancel_requested = True
        return True

    expert._move_to = fake_move
    res = expert._run_sequence()
    assert res == {"success": False, "reason": "canceled"}
    assert len(moves) == 1


def test_run_sequence_survives_feedback_publish_error(expert):
    # cancel 与 feedback 有竞态：发布器失效不该让整条示范崩掉
    expert.goal = FakeGoal(feedback_raises=True)
    expert.ik = _ik_ok([0.0] * 6)
    res = expert._run_sequence()
    assert res["success"] is True


# ---------------------------------------------------------------- on_goal


def test_on_goal_rejects_when_busy(expert):
    expert.busy = True
    goal = FakeGoal()
    res = expert.on_goal(goal)
    assert goal.calls == ["abort"]
    assert res.success is False
    assert "正在执行" in res.message


def test_on_goal_rejects_when_pose_missing(expert):
    expert.cube = None
    goal = FakeGoal()
    res = expert.on_goal(goal)
    assert goal.calls == ["abort"]
    assert res.success is False
    assert "/object/pose" in res.message
    assert expert.busy is False


def test_on_goal_success_flow(expert):
    expert.ik = _ik_ok([0.0] * 6)
    goal = FakeGoal()
    res = expert.on_goal(goal)
    assert goal.calls == ["succeed"]
    assert res.success is True
    assert res.message == "done"
    assert res.duration_s >= 0.0
    # 运行结束后必须复位，否则节点再也接不了下一条
    assert expert.busy is False
    assert expert.goal is None


def test_on_goal_aborts_on_failure(expert):
    expert.ik = lambda *a, **kw: {"success": False, "q": [0.0] * 6}
    goal = FakeGoal()
    res = expert.on_goal(goal)
    assert goal.calls == ["abort"]
    assert res.success is False
    assert expert.busy is False


def test_on_goal_reports_canceled(expert):
    goal = FakeGoal()
    expert.ik = _ik_ok([0.0] * 6)

    def run_and_cancel():
        goal.is_cancel_requested = True
        return {"success": False, "reason": "被取消"}

    expert._run_sequence = run_and_cancel
    res = expert.on_goal(goal)
    assert goal.calls == ["canceled"]
    assert res.success is False
    assert res.message == "canceled"
    assert expert.busy is False


def test_on_goal_catches_exception(expert):
    goal = FakeGoal()

    def boom():
        raise RuntimeError("模型炸了")

    expert._run_sequence = boom
    res = expert.on_goal(goal)
    assert goal.calls == ["abort"]
    assert res.success is False
    assert res.message.startswith("exception:")
    assert any("模型炸了" in m for m in expert.logger.messages("error"))
    assert expert.busy is False
