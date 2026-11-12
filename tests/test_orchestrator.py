"""orchestrator 纯逻辑测试：不启节点、不碰 DDS，只测调度判断本身。

    source /ws/install/setup.bash
    pytest tests -q
"""

from __future__ import annotations

import json
import threading
import time
import types

import pytest

from conftest import orchestrator_node


def msg_of(payload) -> types.SimpleNamespace:
    """造一条 String 消息。传 dict 会序列化，传字符串原样当作坏数据。"""
    data = payload if isinstance(payload, str) else json.dumps(payload)
    return types.SimpleNamespace(data=data)


def status_of(stamp=None, liveness=True, readiness=True, state="idle") -> dict:
    return {
        "liveness": liveness,
        "readiness": readiness,
        "state": state,
        "stamp": time.time() if stamp is None else stamp,
    }


def fill_all_ready(orch):
    for name in orch.status_order:
        orch.status[name] = status_of()


# ---------------------------------------------------------------- 事件构造
def test_trial_msg_payload(orch):
    payload = json.loads(orch._trial_msg(7, "begin").data)
    assert payload == {"id": 7, "event": "begin"}


# ---------------------------------------------------------------- 结论过滤
def test_on_result_accepts_current_trial(orch):
    orch.current_trial = 7
    orch.on_result(msg_of({"id": 7, "success": True, "reason": ""}))

    assert orch.result_event.is_set()
    assert orch.last_result["success"] is True


def test_on_result_drops_other_trial(orch):
    orch.current_trial = 7
    orch.on_result(msg_of({"id": 6, "success": True}))

    assert not orch.result_event.is_set()
    assert orch.last_result is None
    assert orch.logger.messages("warn")


def test_on_result_drops_bad_json(orch):
    orch.current_trial = 7
    orch.on_result(msg_of("not json at all"))

    assert not orch.result_event.is_set()
    assert orch.last_result is None


# ---------------------------------------------------------------- 状态接收
def test_on_status_trusts_topic_name(orch):
    orch.on_status("recorder", msg_of({"node": "detector", "liveness": True}))

    assert orch.status["recorder"]["node"] == "recorder"
    # 不该顺手写进 detector
    assert orch.status["detector"] == {}


def test_on_status_bad_json_keeps_previous(orch):
    orch.status["sim"] = status_of()
    orch.on_status("sim", msg_of("broken"))

    assert orch.status["sim"]["readiness"] is True
    assert orch.logger.messages("error")


# ---------------------------------------------------------------- 就绪判定
def test_check_ready_all_good(orch):
    fill_all_ready(orch)
    assert orch.check_ready() == []


def test_check_ready_no_status(orch):
    problems = orch.check_ready()
    assert len(problems) == 4
    assert all("no_status" in p for p in problems)


def test_check_ready_stale(orch):
    fill_all_ready(orch)
    orch.status["detector"] = status_of(stamp=time.time() - 10.0)

    problems = orch.check_ready()
    assert len(problems) == 1
    assert problems[0].startswith("detector: stale")


def test_check_ready_not_alive(orch):
    fill_all_ready(orch)
    orch.status["expert"] = status_of(liveness=False)

    assert orch.check_ready() == ["expert: not alive"]


def test_check_ready_not_ready(orch):
    fill_all_ready(orch)
    orch.status["recorder"] = status_of(readiness=False, state="recording")

    assert orch.check_ready() == ["recorder: not ready (recording)"]


def test_check_ready_skipped_in_dry_run(orch):
    orch.dry_run = True
    assert orch.check_ready() == []


# ---------------------------------------------------------------- 就绪等待
def test_wait_ready_returns_immediately(orch):
    fill_all_ready(orch)

    t0 = time.monotonic()
    assert orch.wait_ready(1.0) == []
    assert time.monotonic() - t0 < 0.2


def test_wait_ready_times_out(orch):
    t0 = time.monotonic()
    problems = orch.wait_ready(0.2)
    elapsed = time.monotonic() - t0

    assert len(problems) == 4
    assert elapsed >= 0.2


def test_wait_ready_exits_on_abort(orch):
    orch.abort = True

    t0 = time.monotonic()
    assert orch.wait_ready(5.0)
    assert time.monotonic() - t0 < 0.5


# ---------------------------------------------------------------- 启停控制
def test_on_start_first_time(orch):
    resp = types.SimpleNamespace(success=False, message="")
    orch.on_start(None, resp)

    assert resp.success is True
    assert orch.running is True
    assert orch.run_thread is not None


def test_on_start_rejects_when_running(orch):
    orch.running = True
    resp = types.SimpleNamespace(success=False, message="")
    orch.on_start(None, resp)

    assert resp.success is False
    assert "已经在运行" in resp.message


def test_on_stop_rejects_when_idle(orch):
    resp = types.SimpleNamespace(success=False, message="")
    orch.on_stop(None, resp)

    assert resp.success is False
    assert "没有在运行" in resp.message


def test_on_stop_requests_graceful_exit(orch):
    orch.running = True
    resp = types.SimpleNamespace(success=False, message="")
    orch.on_stop(None, resp)

    assert resp.success is True
    assert orch.stop_requested is True
    # 优雅停止不打断正在等的判定
    assert not orch.result_event.is_set()


def test_on_abort_wakes_result_wait(orch):
    orch.running = True
    resp = types.SimpleNamespace(success=False, message="")
    orch.on_abort(None, resp)

    assert resp.success is True
    assert orch.abort is True
    # 不唤醒的话 run_trial 要等到 trial_timeout 才返回
    assert orch.result_event.is_set()


def test_on_abort_rejects_when_idle(orch):
    resp = types.SimpleNamespace(success=False, message="")
    orch.on_abort(None, resp)

    assert resp.success is False


# ---------------------------------------------------------------- 等待原语
class DummyFuture:
    """只实现 done()，用来驱动 _wait_future 的三条分支。"""

    def __init__(self, delay: float):
        self.deadline = time.monotonic() + delay

    def done(self) -> bool:
        return time.monotonic() >= self.deadline


def test_wait_future_returns_true_when_ready(orch):
    assert orch._wait_future(DummyFuture(0.05), 1.0) is True


def test_wait_future_times_out(orch):
    t0 = time.monotonic()
    assert orch._wait_future(DummyFuture(5.0), 0.1) is False
    assert time.monotonic() - t0 < 1.0


def test_wait_future_exits_on_abort(orch):
    orch.abort = True
    t0 = time.monotonic()
    assert orch._wait_future(DummyFuture(5.0), 5.0) is False
    assert time.monotonic() - t0 < 0.5


# ---------------------------------------------------------------- dry_run
def test_call_skips_when_dry_run_and_unavailable(orch):
    orch.dry_run = True

    class NoService:
        def service_is_ready(self):
            return False

        def call_async(self, _req):  # 不该走到这里
            raise AssertionError("dry_run 不该真的发请求")

    assert orch.call(NoService(), "sim.reset") is True
