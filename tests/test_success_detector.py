"""success_detector 的判定逻辑测试。

分三层：

  1. 配置解析（真节点）：ground_truth 模式下不需要视觉键；vision 模式仍能
     读旧键；judge 非法值、配置缺失都要在启动时就报错。

  2. 判定逻辑（object.__new__ + 假时钟）：settle → hold → 成功、超时、
     方块高于杯口、横向越界、夹爪没松开、begin 清缓存、end 兜底。

  3. 状态与模式开关：/detector/status 的两态；ground_truth 下 on_image 直接丢弃。

时间全部由假时钟驱动：_evaluate 只看传入的 now，不读墙钟，所以测试是确定的，
不靠 sleep。

必须在有 ROS 2 的环境里跑（容器内）：
    colcon build --packages-select alicia_interfaces
    source install/setup.bash
    pytest tests/test_success_detector.py -q
"""

from __future__ import annotations

import json
import math
import os
import sys

import pytest

from conftest import FakeLogger, FakePublisher, ParamStore

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DET_SRC = os.path.join(ROOT, "ros2_ws", "src", "success_detector")
if DET_SRC not in sys.path:
    sys.path.insert(0, DET_SRC)

pytest.importorskip("rclpy", reason="需要 ROS 2 环境，请在容器内运行")
det_mod = pytest.importorskip(
    "success_detector", reason="需要 rclpy 与 alicia_interfaces"
)
pytest.importorskip("alicia_interfaces", reason="需要先 colcon build alicia_interfaces 并 source")

from alicia_interfaces.msg import ObjectPose  # noqa: E402
from sensor_msgs.msg import Image  # noqa: E402
from std_msgs.msg import String  # noqa: E402


GT_YAML = """
judge: ground_truth
episode:
  settle_s: 0.2
  hold_s: 0.2
  timeout_s: 1.0
cup:
  radius_m: 0.035
cube:
  settle_m: 0.002
gripper:
  open_min: 0.8
"""

VISION_YAML = """
judge: vision
episode:
  settle_s: 0.2
  settle_px: 2.0
  hold_s: 0.2
  timeout_s: 1.0
cup:
  center_px: [320, 300]
  radius_px: 35.0
  margin_px: 5.0
  radius_m: 0.035
cube:
  ranges_lower: [[0, 120, 60]]
  ranges_upper: [[10, 255, 255]]
  min_area_px: 200
  settle_m: 0.002
gripper:
  open_min: 0.8
"""


class Clock:
    """可手动推进的时钟，替掉模块里的 time，让 _evaluate 的调用次数 = 时间。"""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def time(self):
        return 1_700_000_000.0 + self.t


class FakeMaskPublisher(FakePublisher):
    def __init__(self, subs: int = 0):
        super().__init__()
        self.subs = subs

    def get_subscription_count(self):
        return self.subs


def make_detector(**overrides):
    """绕开 __init__，只注入被测算子真正用到的字段。默认是一套真值模式阈值。"""
    d = object.__new__(det_mod.SuccessDetector)
    logger = FakeLogger()
    d.get_logger = lambda: logger
    d.logger = logger

    d.judge = "ground_truth"
    d.settle_s = 0.2
    d.settle_px = 2.0
    d.hold_s = 0.2
    d.timeout_s = 1.0
    d.cup_r_m = 0.035
    d.settle_m = 0.002
    d.cup_c = (320.0, 300.0)          # 视觉模式杯口圆心
    d.cup_r = 30.0                    # 视觉模式半径（已扣 margin）
    d.hsv_ranges = []
    d.min_area = 200
    d.grip_open_min = 0.8

    d.watching = False
    d.t_start = 0.0
    d.hold_since = None
    d.settle_hist = []
    d.joints = None
    d.gripper = None
    d.dist_px = float("inf")
    d.cube_2d = None
    d.cube_3d = None
    d.cube_edge = None
    d.cup_gt = None
    d.dist_m = float("inf")
    d.gt_pose = None
    d.episode_id = -1

    d.pub_result = FakePublisher()
    d.pub_mask = FakeMaskPublisher()
    d.pub_status = FakePublisher()

    for k, v in overrides.items():
        setattr(d, k, v)
    return d


def begin(d, ep_id: int = 1):
    d.on_episode(String(data=json.dumps({"id": ep_id, "event": "begin"})))


def end(d, ep_id: int = 1):
    d.on_episode(String(data=json.dumps({"id": ep_id, "event": "end"})))


def truth(cube=(0.60, 0.0, 0.10), edge=0.03, cup=(0.60, 0.0, 0.12)) -> ObjectPose:
    msg = ObjectPose()
    msg.cup_pose.x, msg.cup_pose.y, msg.cup_pose.z = cup
    msg.cup_height = 0.09
    msg.cube_pose.x, msg.cube_pose.y, msg.cube_pose.z = cube
    msg.cube_edge_m = edge
    return msg


def run(d, clock, seconds: float, feed=None, step=1.0 / 30.0):
    """按 step 推进时钟并调 _evaluate，直到试次结束或时间用尽。"""
    for _ in range(int(seconds / step) + 1):
        clock.t += step
        if feed is not None:
            feed()
        d._evaluate()
        if not d.watching:
            return


def last_result(d) -> dict:
    assert d.pub_result.sent, "没有发布 /orch/result"
    return json.loads(d.pub_result.sent[-1].data)


# ======================================================================
# 配置解析
# ======================================================================

@pytest.fixture
def detector_factory(tmp_path, monkeypatch, ros_context):
    """按 yaml 文本创建真节点；测试结束统一销毁。"""
    nodes = []

    def make(text: str):
        cfg = tmp_path / f"detector_{len(nodes)}.yaml"
        cfg.write_text(text, encoding="utf-8")
        store = ParamStore({"config_file": str(cfg)})
        monkeypatch.setattr(
            det_mod.SuccessDetector, "declare_parameter", staticmethod(store.declare)
        )
        monkeypatch.setattr(
            det_mod.SuccessDetector, "get_parameter", staticmethod(store.get)
        )
        node = det_mod.SuccessDetector()
        nodes.append(node)
        return node

    yield make

    for node in nodes:
        node.destroy_node()


def test_ground_truth_config_needs_no_vision_keys(detector_factory):
    d = detector_factory(GT_YAML)

    assert d.judge == "ground_truth"
    assert d.cup_r_m == pytest.approx(0.035)
    assert d.settle_m == pytest.approx(0.002)
    assert d.grip_open_min == pytest.approx(0.8)
    # 视觉参数没配也要能起来，占位成空值
    assert d.cup_c == (0.0, 0.0)
    assert d.cup_r == 0.0
    assert d.hsv_ranges == []
    assert d.min_area == 0


def test_vision_config_still_parses_legacy_keys(detector_factory):
    d = detector_factory(VISION_YAML)

    assert d.judge == "vision"
    assert d.cup_c == pytest.approx((320.0, 300.0))
    assert d.cup_r == pytest.approx(30.0)          # radius_px - margin_px
    assert d.hsv_ranges == [([0, 120, 60], [10, 255, 255])]
    assert d.min_area == 200


def test_unknown_judge_is_rejected(detector_factory):
    with pytest.raises(ValueError):
        detector_factory(GT_YAML.replace("ground_truth", "magic"))


def test_missing_config_file_is_rejected(tmp_path, monkeypatch, ros_context):
    store = ParamStore({"config_file": str(tmp_path / "nope.yaml")})
    monkeypatch.setattr(
        det_mod.SuccessDetector, "declare_parameter", staticmethod(store.declare)
    )
    monkeypatch.setattr(
        det_mod.SuccessDetector, "get_parameter", staticmethod(store.get)
    )
    with pytest.raises(FileNotFoundError):
        det_mod.SuccessDetector()


# ======================================================================
# 真值判定
# ======================================================================

def test_success_requires_settle_then_hold(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(det_mod, "time", clock)
    d = make_detector()
    d.gripper = 1.0
    begin(d, 3)

    run(d, clock, 1.5, feed=lambda: d.on_object(truth()))

    res = last_result(d)
    assert res["success"] is True
    assert res["reason"] == "success"
    assert res["id"] == 3
    assert res["judge"] == "ground_truth"
    assert res["dist_m"] == pytest.approx(0.0, abs=1e-4)
    assert res["gt_pose"] is not None
    assert d.watching is False


def test_timeout_without_truth(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(det_mod, "time", clock)
    d = make_detector()
    d.gripper = 1.0
    begin(d, 4)

    run(d, clock, 2.0)                    # 一帧真值都没有

    res = last_result(d)
    assert res["success"] is False
    assert res["reason"] == "timeout"
    assert res["dist_m"] is None
    assert d.watching is False


def test_cube_hovering_above_cup_is_not_success(monkeypatch):
    """中心在杯口正上方但没落进去：只比平面距离会误判，必须比顶面高度。"""
    clock = Clock()
    monkeypatch.setattr(det_mod, "time", clock)
    d = make_detector()
    d.gripper = 1.0
    begin(d, 5)
    hovering = truth(cube=(0.60, 0.0, 0.12), edge=0.03, cup=(0.60, 0.0, 0.12))

    run(d, clock, 0.6, feed=lambda: d.on_object(hovering))

    assert d.watching is True
    assert d.hold_since is None
    assert d.pub_result.sent == []


def test_cube_outside_cup_radius_is_not_success(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(det_mod, "time", clock)
    d = make_detector()
    d.gripper = 1.0
    begin(d, 6)
    offset = truth(cube=(0.70, 0.0, 0.10), cup=(0.60, 0.0, 0.12))

    run(d, clock, 0.6, feed=lambda: d.on_object(offset))

    assert d.watching is True
    assert d.dist_m == pytest.approx(0.10, abs=1e-6)


def test_closed_gripper_blocks_success(monkeypatch):
    """夹爪还夹着方块时不能算完成，否则会把"提着"判成"放进去了"。"""
    clock = Clock()
    monkeypatch.setattr(det_mod, "time", clock)
    d = make_detector()
    d.gripper = 0.2
    begin(d, 7)

    run(d, clock, 0.6, feed=lambda: d.on_object(truth()))

    assert d.watching is True
    assert d.hold_since is None


def test_moving_cube_never_settles(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(det_mod, "time", clock)
    d = make_detector()
    d.gripper = 1.0
    begin(d, 8)
    toggle = [False]

    def feed():
        jitter = 0.01 if toggle[0] else 0.0
        toggle[0] = not toggle[0]
        d.on_object(truth(cube=(0.60 + jitter, 0.0, 0.10)))

    run(d, clock, 0.6, feed=feed)

    assert d.watching is True
    assert d.hold_since is None


def test_end_event_finishes_unsuccessful_trial(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(det_mod, "time", clock)
    d = make_detector()
    begin(d, 9)

    run(d, clock, 0.1)
    end(d, 9)

    res = last_result(d)
    assert res["success"] is False
    assert res["reason"] == "ended_without_success"
    assert res["id"] == 9


def test_end_without_begin_is_ignored():
    d = make_detector()
    end(d, 10)
    assert d.pub_result.sent == []


def test_begin_clears_previous_trial_state(monkeypatch):
    """上一试次的真值与 hold 计时不能漏到下一试次。"""
    clock = Clock()
    monkeypatch.setattr(det_mod, "time", clock)
    d = make_detector()
    d.gripper = 1.0
    begin(d, 11)
    run(d, clock, 1.5, feed=lambda: d.on_object(truth()))
    assert last_result(d)["success"] is True

    begin(d, 12)

    assert d.episode_id == 12
    assert d.watching is True
    assert d.cube_3d is None
    assert d.cube_edge is None
    assert d.cup_gt is None
    assert d.gt_pose is None
    assert d.settle_hist == []
    assert d.hold_since is None
    assert d.dist_m == float("inf")


def test_bad_trial_json_is_ignored():
    d = make_detector()
    d.on_episode(String(data="{not json"))
    assert d.watching is False
    assert d.logger.messages("error")


# ======================================================================
# 视觉路径（保留但当前不启用）
# ======================================================================

def test_vision_path_still_judges(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(det_mod, "time", clock)
    d = make_detector(judge="vision")
    d.gripper = 1.0
    begin(d, 13)
    d.cube_2d = (320.0, 300.0, 0.0, 900)

    run(d, clock, 1.5)

    res = last_result(d)
    assert res["success"] is True
    assert res["judge"] == "vision"
    assert res["dist_px"] == pytest.approx(0.0, abs=1e-3)


def test_on_image_is_dropped_in_ground_truth_mode():
    d = make_detector()
    img = Image(height=1, width=1, encoding="rgb8", step=3, data=bytes(3))

    d.on_image(img)

    assert d.cube_2d is None
    assert d.pub_mask.sent == []


def test_on_state_requires_seven_values():
    d = make_detector()

    d.on_state(_joint_state([0.0] * 6))
    assert d.gripper is None

    d.on_state(_joint_state([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
    assert d.joints == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    assert d.gripper == pytest.approx(0.7)


def _joint_state(values):
    from sensor_msgs.msg import JointState

    msg = JointState()
    msg.position = [float(v) for v in values]
    return msg


# ======================================================================
# 状态
# ======================================================================

def test_status_idle_then_judging(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(det_mod, "time", clock)
    d = make_detector()

    d._publish_status()
    idle = json.loads(d.pub_status.sent[-1].data)
    assert idle["readiness"] is True
    assert idle["state"] == "idle"
    assert idle["judge"] == "ground_truth"
    assert idle["has_truth"] is False
    assert idle["stamp"] == pytest.approx(clock.time())

    begin(d, 14)
    d.on_object(truth())
    d._publish_status()
    busy = json.loads(d.pub_status.sent[-1].data)
    assert busy["readiness"] is False
    assert busy["state"] == "judging"
    assert busy["episode"] == 14
    assert busy["has_truth"] is True
    assert busy["liveness"] is True


def test_hold_measures_only_continuous_success(monkeypatch):
    """条件一旦被打破，hold 必须从零重计，不能累计。"""
    clock = Clock()
    monkeypatch.setattr(det_mod, "time", clock)
    # hold 拉长，保证观察窗口内不会直接判成功（成功会结束试次）
    d = make_detector(hold_s=5.0)
    d.gripper = 1.0
    begin(d, 15)

    run(d, clock, 0.5, feed=lambda: d.on_object(truth()))
    assert d.hold_since is not None
    first = d.hold_since

    # 方块被移出杯子：hold 清零
    d.on_object(truth(cube=(0.80, 0.0, 0.10)))
    run(d, clock, 0.1)
    assert d.hold_since is None
    assert d.watching is True

    # 再放回去：要等旧位置滑出 settle 窗口，才会重新判"停稳"并从当前时刻计时
    run(d, clock, 0.3, feed=lambda: d.on_object(truth()))
    assert d.hold_since is not None and d.hold_since > first


def test_result_distance_is_reported(monkeypatch):
    clock = Clock()
    monkeypatch.setattr(det_mod, "time", clock)
    d = make_detector()
    d.gripper = 1.0
    begin(d, 16)

    run(d, clock, 0.1, feed=lambda: d.on_object(truth(cube=(0.62, 0.0, 0.10))))
    assert math.isfinite(d.dist_m)
    assert d.dist_m == pytest.approx(0.02, abs=1e-6)
