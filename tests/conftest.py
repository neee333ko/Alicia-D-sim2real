"""orchestrator 单元测试的公共设施。

Orchestrator.__init__ 会 rclpy.init、建 client、读 task.yaml，直接实例化代价高且要真环境。
这里用 object.__new__ 绕过 __init__，只注入被测函数真正用到的字段，测纯逻辑。

必须在有 ROS 2 的环境里跑（容器内）：
    pip install pytest
    source /ws/install/setup.bash
    pytest tests -q
"""

from __future__ import annotations

import os
import sys
import threading

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORCH_SRC = os.path.join(ROOT, "ros2_ws", "src", "orchestrator")
if ORCH_SRC not in sys.path:
    sys.path.insert(0, ORCH_SRC)

# 宿主上没有 ROS，直接跳过而不是报一堆 ImportError
pytest.importorskip("rclpy", reason="需要 ROS 2 环境，请在容器内运行")
orchestrator_node = pytest.importorskip(
    "orchestrator_node", reason="需要先 colcon build alicia_interfaces 并 source"
)


class FakeLogger:
    """记录日志，便于断言，也避免 get_logger 依赖真实 Node。"""

    def __init__(self):
        self.records: list[tuple[str, str]] = []

    def _add(self, level: str, msg: str):
        self.records.append((level, msg))

    def info(self, msg, **_kwargs):
        self._add("info", msg)

    def warn(self, msg, **_kwargs):
        self._add("warn", msg)

    def error(self, msg, **_kwargs):
        self._add("error", msg)

    def debug(self, msg, **_kwargs):
        self._add("debug", msg)

    def messages(self, level: str) -> list[str]:
        return [m for lv, m in self.records if lv == level]


class FakePublisher:
    """只记录发布出去的消息，不碰 DDS。"""

    def __init__(self):
        self.sent = []

    def publish(self, msg):
        self.sent.append(msg)


@pytest.fixture
def orch():
    node = object.__new__(orchestrator_node.Orchestrator)

    logger = FakeLogger()
    node.get_logger = lambda: logger
    node.logger = logger

    node.state_lock = threading.Lock()
    node.status_lock = threading.Lock()
    node.abort = False
    node.stop_requested = False
    node.running = False
    node.run_thread = None
    node.result_event = threading.Event()
    node.last_result = None
    node.current_trial = None

    node.status_order = ("sim", "expert", "recorder", "detector")
    node.status = {name: {} for name in node.status_order}
    node.status_timeout = 3.0
    node.ready_wait = 0.2
    node.dry_run = False

    node.pub_trial = FakePublisher()
    # on_start 会起线程跑 self.run，测试里换成空实现，避免真去调服务
    node.run = lambda: None

    yield node

    if node.run_thread is not None and node.run_thread.is_alive():
        node.run_thread.join(timeout=1.0)


# ======================================================================
# 以下为“真 ROS”集成测试用的设施：起真节点 + 假下游，测 call/run_expert/run_trial/run
# ======================================================================

import json  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import rclpy  # noqa: E402
from rclpy.action import ActionServer, CancelResponse  # noqa: E402
from rclpy.callback_groups import ReentrantCallbackGroup  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402
from rclpy.qos import (  # noqa: E402
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String  # noqa: E402
from std_srvs.srv import Trigger  # noqa: E402

pytest.importorskip("alicia_interfaces", reason="需要先 colcon build alicia_interfaces 并 source")
from alicia_interfaces.action import RunOnce  # noqa: E402
from alicia_interfaces.srv import Place, Setup  # noqa: E402

# 独立 domain，避免和本机正在跑的节点串台
os.environ.setdefault("ROS_DOMAIN_ID", "77")

LATCHED = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

TASK_YAML = """
name: unit-test
spawn:
  cube:
    center: [0.50, 0.00, 0.03]
  cup:
    center: [0.60, 0.00, 0.04]
expert:
  count: 2
  radius_m: 0.03
  rounds_per_position: 1
  trial_timeout_s: 2.0
"""


def wait_until(predicate, timeout: float = 2.0, interval: float = 0.01) -> bool:
    """等某个条件成立。话题是异步的，发布完立刻断言会撞上回调还没跑。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


class ParamStore:
    """替换 declare_parameter/get_parameter，让 __init__ 读到测试指定的参数。

    注意 get() 必须返回真的 rclpy Parameter：Node.__init__ 内部的 time_source
    会读 use_sim_time 的 .type_，返回裸值会直接报 AttributeError。
    """

    def __init__(self, overrides: dict):
        self.values = dict(overrides)

    def declare(self, name, default):
        self.values.setdefault(name, default)

    def get(self, name):
        from rclpy.parameter import Parameter

        return Parameter(name, value=self.values.get(name, False))


class FakeWorld(Node):
    """假的下游：sim_bridge 的三个服务 + expert 的 action + detector 的结论。

    每个行为都做成可改的属性，测试里按需调整。所有回调都放在
    ReentrantCallbackGroup 里，否则阻塞式等待会把这个节点的执行线程占死。
    """

    def __init__(self):
        super().__init__("fake_world")
        cb = ReentrantCallbackGroup()

        self.place_ok = True
        self.place_requests: list[Place.Request] = []
        self.reset_ok = True
        self.reset_delay = 0.0
        self.expert_ok = True
        self.expert_seconds = 0.0
        self.expert_goals: list[int] = []
        self.expert_canceled = False
        self.result_ok = True
        self.result_delay = 0.05
        self.publish_status = True
        self.trial_events: list[dict] = []

        self.create_service(Setup, "/sim/setup", self.on_setup, callback_group=cb)
        self.create_service(Place, "/sim/place", self.on_place, callback_group=cb)
        self.create_service(Trigger, "/sim/reset", self.on_reset, callback_group=cb)

        self.action_server = ActionServer(
            self,
            RunOnce,
            "/expert/run_once",
            self.on_expert,
            callback_group=cb,
            # rclpy 默认拒绝所有 cancel 请求，不传这个的话 cancel 会静默失效
            cancel_callback=lambda _req: CancelResponse.ACCEPT,
        )

        self.pub_result = self.create_publisher(String, "/orch/result", 10)
        self.create_subscription(String, "/orch/trial", self.on_trial, 10, callback_group=cb)

        self.pub_status = {}
        for name in ("sim", "expert", "recorder", "detector"):
            self.pub_status[name] = self.create_publisher(String, f"/{name}/status", LATCHED)
        self.create_timer(0.05, self.publish_statuses, callback_group=cb)

    # ---- sim 服务 ------------------------------------------------------
    def on_setup(self, _req, resp):
        resp.success = True
        return resp

    def on_place(self, req, resp):
        self.place_requests.append(req)
        resp.success = self.place_ok
        resp.message = "" if self.place_ok else "place refused"
        return resp

    def on_reset(self, _req, resp):
        if self.reset_delay:
            time.sleep(self.reset_delay)
        resp.success = self.reset_ok
        resp.message = "" if self.reset_ok else "home not reached"
        return resp

    # ---- expert action -------------------------------------------------
    def on_expert(self, goal_handle):
        self.expert_goals.append(goal_handle.request.id)
        deadline = time.monotonic() + self.expert_seconds
        while time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                return self._canceled(goal_handle)
            try:
                fb = RunOnce.Feedback()
                fb.phase = "moving"
                fb.progress = 0.5
                goal_handle.publish_feedback(fb)
            except Exception:
                # cancel 和 feedback 有竞态：cancel 先到时发布器会失效
                if goal_handle.is_cancel_requested:
                    return self._canceled(goal_handle)
                raise
            time.sleep(0.02)

        if goal_handle.is_cancel_requested:
            return self._canceled(goal_handle)

        goal_handle.succeed()
        return RunOnce.Result(
            success=self.expert_ok,
            message="" if self.expert_ok else "grasp failed",
            duration_s=self.expert_seconds,
        )

    def _canceled(self, goal_handle):
        self.expert_canceled = True
        try:
            goal_handle.canceled()
        except Exception:
            pass
        return RunOnce.Result(success=False, message="canceled")

    # ---- detector 模拟 -------------------------------------------------
    def on_trial(self, msg):
        try:
            event = json.loads(msg.data)
        except ValueError:
            return
        self.trial_events.append(event)
        if event.get("event") != "begin":
            return

        def reply():
            time.sleep(self.result_delay)
            out = String()
            out.data = json.dumps(
                {"id": event["id"], "success": self.result_ok, "reason": "ok" if self.result_ok else "miss"}
            )
            try:
                self.pub_result.publish(out)
            except Exception:
                # 用例跑完、节点已销毁时线程可能还在，忽略即可
                pass

        threading.Thread(target=reply, daemon=True).start()

    # ---- 状态 ----------------------------------------------------------
    def publish_statuses(self):
        if not self.publish_status:
            return
        for name, pub in self.pub_status.items():
            msg = String()
            msg.data = json.dumps(
                {"node": name, "liveness": True, "readiness": True, "state": "idle", "stamp": time.time()}
            )
            pub.publish(msg)


@pytest.fixture
def ros_context():
    """初始化 rclpy；重复调用不报错，结束时统一 shutdown。"""
    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def live(tmp_path, monkeypatch, ros_context):
    """起真 Orchestrator + 假下游，返回 (orch, world, params)。"""
    task_file = tmp_path / "task.yaml"
    task_file.write_text(TASK_YAML, encoding="utf-8")

    store = ParamStore(
        {
            "mode": "expert",
            "task_file": str(task_file),
            "dry_run": False,
            "out_dir": str(tmp_path / "logs"),
            "status_timeout_s": 3.0,
            "ready_wait_s": 1.0,
        }
    )
    monkeypatch.setattr(
        orchestrator_node.Orchestrator, "declare_parameter", staticmethod(store.declare)
    )
    monkeypatch.setattr(
        orchestrator_node.Orchestrator, "get_parameter", staticmethod(store.get)
    )

    orch = orchestrator_node.Orchestrator()
    world = FakeWorld()

    exe = MultiThreadedExecutor(num_threads=6)
    exe.add_node(orch)
    exe.add_node(world)
    spin = threading.Thread(target=exe.spin, daemon=True)
    spin.start()

    yield orch, world

    exe.shutdown()
    orch.destroy_node()
    world.destroy_node()
    spin.join(timeout=2.0)


@pytest.fixture
def bare(tmp_path, monkeypatch, ros_context):
    """只有 Orchestrator，没有任何下游：用来测“缺服务”的分支。"""
    task_file = tmp_path / "task.yaml"
    task_file.write_text(TASK_YAML, encoding="utf-8")

    store = ParamStore(
        {
            "mode": "expert",
            "task_file": str(task_file),
            "dry_run": False,
            "out_dir": str(tmp_path / "logs"),
            "status_timeout_s": 3.0,
            "ready_wait_s": 0.2,
        }
    )
    monkeypatch.setattr(
        orchestrator_node.Orchestrator, "declare_parameter", staticmethod(store.declare)
    )
    monkeypatch.setattr(
        orchestrator_node.Orchestrator, "get_parameter", staticmethod(store.get)
    )

    orch = orchestrator_node.Orchestrator()
    exe = MultiThreadedExecutor(num_threads=4)
    exe.add_node(orch)
    spin = threading.Thread(target=exe.spin, daemon=True)
    spin.start()

    yield orch, store

    exe.shutdown()
    orch.destroy_node()
    spin.join(timeout=2.0)
