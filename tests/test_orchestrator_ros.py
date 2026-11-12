"""需要真 ROS 的集成测试：起真 Orchestrator + 假下游节点。

覆盖 call / check_nodes / run_expert / run_trial / run 的真实交互路径。
跑在容器里：

    source /ws/install/setup.bash
    pytest tests -q

每个用例自己起一整套节点，使用独立 ROS_DOMAIN_ID=77，不会和本机正在跑的节点串台。
"""

from __future__ import annotations

import json
import threading
import time
import types

import pytest

from conftest import wait_until


def _resp():
    return types.SimpleNamespace(success=False, message="")


def _force_missing(orch):
    """把等待服务/动作的探测改成立即失败，避免每个用例白等 2 秒。"""
    for cli in orch.required.values():
        cli.wait_for_service = lambda timeout_sec=0.0: False
    orch.ac_expert.wait_for_server = lambda timeout_sec=0.0: False


# ---------------------------------------------------------------- call
def test_call_success(live):
    orch, _world = live
    assert orch.call(orch.cli_reset, "sim.reset", timeout=2.0) is True


def test_call_reports_failure(live):
    orch, world = live
    world.reset_ok = False

    assert orch.call(orch.cli_reset, "sim.reset", timeout=2.0) is False


def test_call_timeout(live):
    orch, world = live
    world.reset_delay = 0.5

    t0 = time.monotonic()
    assert orch.call(orch.cli_reset, "sim.reset", timeout=0.2) is False
    assert time.monotonic() - t0 < 0.5


def test_call_exits_on_abort(live):
    orch, world = live
    world.reset_delay = 2.0

    threading.Timer(0.1, lambda: setattr(orch, "abort", True)).start()
    t0 = time.monotonic()
    assert orch.call(orch.cli_reset, "sim.reset", timeout=5.0) is False
    # 被 abort 提前打断，不应该等满 5 秒
    assert time.monotonic() - t0 < 1.0


# ---------------------------------------------------------------- 前置检查
def test_check_nodes_all_present(live):
    orch, _world = live
    assert orch.check_nodes() is True


def test_check_nodes_missing(bare):
    orch, _store = bare
    _force_missing(orch)

    assert orch.check_nodes() is False


def test_check_nodes_missing_ok_in_dry_run(bare):
    orch, _store = bare
    _force_missing(orch)
    orch.dry_run = True

    assert orch.check_nodes() is True


# ---------------------------------------------------------------- 专家 action
def test_run_expert_success(live):
    orch, world = live

    assert orch.run_expert(7, timeout=5.0) is True
    assert world.expert_goals == [7]


def test_run_expert_failure(live):
    orch, world = live
    world.expert_ok = False

    assert orch.run_expert(8, timeout=5.0) is False


def test_run_expert_cancels_on_abort(live):
    orch, world = live
    world.expert_seconds = 3.0

    threading.Timer(0.3, lambda: setattr(orch, "abort", True)).start()
    t0 = time.monotonic()
    assert orch.run_expert(9, timeout=10.0) is False
    assert time.monotonic() - t0 < 2.5
    # 关键：不是丢下不管，而是真的发了 cancel（服务端响应和 execute 轮询都有延迟）
    assert wait_until(lambda: world.expert_canceled, 2.0)


# ---------------------------------------------------------------- 单个试次
def test_run_trial_happy_path(live):
    orch, world = live

    trial = orch.run_trial(3, 0.52, -0.01)

    assert trial["success"] is True
    assert trial["reason"] == "ok"
    assert world.place_requests[0].x == pytest.approx(0.52)
    assert world.place_requests[0].y == pytest.approx(-0.01)
    assert world.place_requests[0].z == pytest.approx(0.03)
    assert wait_until(lambda: len(world.trial_events) >= 2)
    assert [e["event"] for e in world.trial_events] == ["begin", "end"]
    assert [e["id"] for e in world.trial_events] == [3, 3]


def test_run_trial_place_failure(live):
    orch, world = live
    world.place_ok = False

    trial = orch.run_trial(4, 0.5, 0.0)

    assert trial["reason"] == "place_failed"
    # 放都没放成功，不该复位、更不该开窗
    time.sleep(0.2)
    assert world.trial_events == []


def test_run_trial_reset_failure(live):
    orch, world = live
    world.reset_ok = False

    trial = orch.run_trial(5, 0.5, 0.0)

    assert trial["reason"] == "reset_failed"
    time.sleep(0.2)
    assert world.trial_events == []


def test_run_trial_skips_when_not_ready(live):
    orch, world = live
    world.publish_status = False
    with orch.status_lock:
        orch.status = {name: {} for name in orch.status_order}

    trial = orch.run_trial(6, 0.5, 0.0)

    assert trial["reason"].startswith("not_ready")
    assert world.place_requests == []


def test_run_trial_expert_failure_still_closes_window(live):
    orch, world = live
    world.expert_ok = False

    trial = orch.run_trial(7, 0.5, 0.0)

    assert trial["success"] is False
    assert trial["reason"] == "expert_failed"
    # 专家失败也要发 end，否则 detector 的窗口一直开着
    assert wait_until(lambda: len(world.trial_events) >= 2)
    assert [e["event"] for e in world.trial_events] == ["begin", "end"]


def test_run_trial_no_result_times_out(live):
    orch, world = live
    world.result_delay = 5.0  # 比 trial_timeout_s=2.0 长

    trial = orch.run_trial(8, 0.5, 0.0)

    assert trial["success"] is False
    assert trial["reason"] == "no_result"


# ---------------------------------------------------------------- 整轮
def test_run_writes_summary(live, tmp_path):
    orch, world = live

    assert orch.run() == 0

    files = list((tmp_path / "logs").glob("expert_run_*.json"))
    assert len(files) == 1
    summary = json.loads(files[0].read_text(encoding="utf-8"))
    assert summary["count"] == 2
    assert len(summary["trials"]) == 2
    assert summary["success"] == 2
    assert summary["aborted"] is False
    assert len(world.place_requests) == 2


def test_run_stops_early_when_stop_requested(live, tmp_path):
    orch, world = live
    orch.stop_requested = True

    assert orch.run() == 0

    summary = json.loads(
        next((tmp_path / "logs").glob("expert_run_*.json")).read_text(encoding="utf-8")
    )
    assert summary["trials"] == []
    assert world.place_requests == []


def test_on_start_runs_and_resets_flag(live):
    orch, _world = live

    resp = _resp()
    orch.on_start(None, resp)
    assert resp.success is True

    orch.run_thread.join(timeout=15.0)
    assert not orch.run_thread.is_alive()
    # finally 里必须放开 running，否则服务只能重启节点才能再触发
    assert orch.running is False


def test_on_start_rejects_while_running(live):
    orch, _world = live
    orch.running = True

    resp = _resp()
    orch.on_start(None, resp)

    assert resp.success is False
