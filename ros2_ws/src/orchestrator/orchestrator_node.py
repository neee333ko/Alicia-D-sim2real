#!/usr/bin/env python3
"""orchestrator_node.py — 整个流程的调度中心。

一个地方控制顺序：摆物体+复位 → 开窗 → 让专家动 → 等判定 → 关窗 → 记账。
录制和判定不归它管：两个节点订阅 /orch/trial 自己开窗收尾。

模式
  expert  仿真里用脚本专家批量生成数据（本期实现）
  eval    真机/仿真跑策略评测（占位）
  record  真机人工示范采集（占位）

启动时会先 ping 各个必需服务，缺一个就退出，不会带病跑一整轮。（diff 测试：改动这一行）

启动时不会自动开跑，等外部调用 /orch/start；/orch/stop 跑完当前试次退出，/orch/abort 立即中断。

下游状态（liveness + readiness）
  每个被调度的节点自己发布状态，orchestrator 只订阅、不 ping：
    /sim/status  /expert/status  /recorder/status  /detector/status
  std_msgs/String，JSON，transient_local，keep_last 1：
    {"node": "recorder", "liveness": true, "readiness": true,
     "state": "idle", "stamp": 1699999999.0, "detail": ""}
  约定：状态变化时立刻发一次，其余时间周期发（建议 1Hz）；stamp 用 time.time()。
  orchestrator 在每个试次开始前检查四路：超过 status_timeout_s 没更新算失联，
  liveness 为 false 算失活，readiness 为 false 算没准备好，任何一项不过就跳过该试次。

试次边界（一对多事件，depth=10）
  /orch/trial  {"id": 7, "event": "begin"} / {"id": 7, "event": "end"}
  recorder 和 detector 都订阅它，各自开窗、自己收尾，orchestrator 不再逐个调用。
  recorder 由此承担一个义务：readiness 为 true 时必须能接住下一条 begin，
  收到 begin 后立刻切到 state="recording"，end 后回到 "idle" 并 flush。

    python3 orchestrator_node.py --ros-args -p mode:=expert -p dry_run:=true
    ros2 service call /orch/start std_srvs/srv/Trigger
"""

from __future__ import annotations

import json
import os
import threading
import time

import rclpy
import yaml
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from alicia_interfaces.action import RunOnce
from alicia_interfaces.srv import Place, Setup


class Orchestrator(Node):
    def __init__(self):
        super().__init__("orchestrator")

        self.declare_parameter("mode", "expert")
        self.declare_parameter("task_file", "/ws/task.yaml")
        self.declare_parameter("dry_run", False)
        self.declare_parameter("out_dir", "/ws/logs")
        self.declare_parameter("status_timeout_s", 3.0)
        self.declare_parameter("ready_wait_s", 3.0)

        p = self.get_parameter
        self.mode = p("mode").value
        self.task_file = p("task_file").value
        self.dry_run = bool(p("dry_run").value)
        self.out_dir = p("out_dir").value
        self.status_timeout = float(p("status_timeout_s").value)
        self.ready_wait = float(p("ready_wait_s").value)

        if not os.path.isfile(self.task_file):
            raise FileNotFoundError(f"找不到任务文件：{self.task_file}")
        with open(self.task_file, encoding="utf-8") as f:
            self.task = yaml.safe_load(f)

        if self.mode != "expert":
            # 其他模式先占位，避免误启动后跑出一堆没意义的数据
            self.get_logger().error(f"mode={self.mode} 还没实现，只支持 expert")
            raise SystemExit(2)

        cfg = self.task.get("expert", {})
        self.count = int(cfg.get("count", 10))
        self.radius = float(cfg.get("radius_m", 0.03))
        self.rounds = int(cfg.get("rounds_per_position", 5))
        self.trial_timeout = float(cfg.get("trial_timeout_s", 20.0))
        self.cube_z = float(self.task["spawn"]["cube"]["center"][2])
        self.base = [float(v) for v in self.task["spawn"]["cube"]["center"][:2]]

        self.abort = False
        self.busy = False
        self.state_lock = threading.Lock()
        self.running = False
        self.stop_requested = False
        self.run_thread: threading.Thread | None = None
        self.last_result: dict | None = None
        self.current_trial: int | None = None
        self.status_lock = threading.Lock()
        self.status: dict[str, dict] = {}
        self.result_event = threading.Event()

        cb = ReentrantCallbackGroup()

        # ---- 客户端（控制走服务） -------------------------------------------
        self.cli_setup = self.create_client(Setup, "/sim/setup", callback_group=cb)
        # 摆方块和机械臂复位是同一个"准备下一条试次"动作，合在这个服务里原子完成
        self.cli_place = self.create_client(Place, "/sim/place", callback_group=cb)

        # 专家是长动作：要进度、要能取消，所以走 action 而不是服务
        self.ac_expert = ActionClient(self, RunOnce, "/expert/run_once", callback_group=cb)

        # ---- 反馈走话题 ------------------------------------------------------
        # 状态要能追上后启动的订阅者；事件要防止被连续的 begin/end 挤掉，所以深度不同。
        # 对方的 QoS 必须一致，否则不匹配、根本收不到。
        latched = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        events = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )
        self.pub_trial = self.create_publisher(String, "/orch/trial", events)
        self.create_subscription(String, "/orch/result", self.on_result, 10, callback_group=cb)

        # ---- 下游状态（订阅，不主动 ping） -----------------------------------
        self.status_order = ("sim", "expert", "recorder", "detector")
        for name in self.status_order:
            self.create_subscription(
                String,
                f"/{name}/status",
                lambda msg, n=name: self.on_status(n, msg),
                latched,
                callback_group=cb,
            )
            self.status[name] = {}

        self.create_service(Trigger, "/orch/start", self.on_start, callback_group=cb)
        self.create_service(Trigger, "/orch/stop", self.on_stop, callback_group=cb)
        self.create_service(Trigger, "/orch/abort", self.on_abort, callback_group=cb)

        self.required = {
            "sim.setup": self.cli_setup,
            "sim.place": self.cli_place,
        }

    # ---------------------------------------------------------------- 回调
    def on_expert_feedback(self, msg):
        """专家进度回调。只做限流日志，进度不参与调度决策。"""
        fb = msg.feedback
        self.get_logger().info(
            f"专家进度 {fb.phase} {fb.progress * 100:.0f}%",
            throttle_duration_sec=2.0,
        )

    def on_status(self, node_name: str, msg: String):
        """收下一路下游状态。

        node 以话题名为准，不信消息里的 node 字段——发布方写错名字不该让状态串到别人身上。
        """
        try:
            data = json.loads(msg.data)
        except ValueError:
            self.get_logger().error(
                f"/{node_name}/status 解析失败：{msg.data}", throttle_duration_sec=5.0
            )
            return
        data["node"] = node_name
        with self.status_lock:
            self.status[node_name] = data

    def on_result(self, msg: String):
        """只接受当前试次的结论。

        判定是异步的：某个试次超时后，detector 才发出的旧结论如果被收下，
        会把下一个试次的 result_event 叫醒，结论就张冠李戴了。
        """
        try:
            data = json.loads(msg.data)
        except ValueError:
            data = {"success": False, "reason": "bad_json"}
        if data.get("id") != self.current_trial:
            self.get_logger().warn(
                f"丢弃非当前试次的结论：id={data.get('id')}，当前 {self.current_trial}",
                throttle_duration_sec=2.0,
            )
            return
        self.last_result = data
        self.result_event.set()

    def on_start(self, _req, resp):
        """开始一轮编排。已经在跑就拒绝，避免两个线程同时驱动同一批服务。"""
        with self.state_lock:
            if self.running:
                resp.success = False
                resp.message = "已经在运行，先 /orch/stop 或 /orch/abort"
                return resp
            self.running = True
            self.abort = False
            self.stop_requested = False
            self.run_thread = threading.Thread(target=self.run, daemon=True)
        self.run_thread.start()
        self.get_logger().info("收到开始请求，编排线程已启动")
        resp.success = True
        resp.message = "started"
        return resp

    def on_stop(self, _req, resp):
        """优雅停止：只置标志，让循环跑完当前试次后自己退出。

        不打断正在等判定的试次，所以不会留下半截 episode；汇总照常写出。
        """
        with self.state_lock:
            if not self.running:
                resp.success = False
                resp.message = "没有在运行"
                return resp
            self.stop_requested = True
        self.get_logger().warn("收到停止请求：当前试次跑完后退出")
        resp.success = True
        resp.message = "stopping after current trial"
        return resp

    def on_abort(self, _req, resp):
        """立即中止：置标志并唤醒等判定的线程，不等当前试次跑完。"""
        with self.state_lock:
            if not self.running:
                resp.success = False
                resp.message = "没有在运行"
                return resp
            self.abort = True
        # run_trial 可能正卡在 result_event.wait 上，不唤醒就要等到 trial_timeout
        self.result_event.set()
        self.get_logger().warn("收到中止请求：当前试次作废")
        resp.success = True
        resp.message = "abort requested"
        return resp

    # ---------------------------------------------------------------- 状态检查
    def check_ready(self) -> list[str]:
        """返回没就绪的节点及原因；空列表表示可以开跑。

        失联、失活、没准备好是三种不同故障，分开报，别都归成"没状态"。
        """
        if self.dry_run:
            return []
        now = time.time()
        with self.status_lock:
            snapshot = {k: dict(v) for k, v in self.status.items()}
        problems = []
        for name in self.status_order:
            st = snapshot.get(name) or {}
            if not st:
                problems.append(f"{name}: no_status")
                continue
            age = now - float(st.get("stamp", 0.0))
            if age > self.status_timeout:
                problems.append(f"{name}: stale {age:.1f}s")
            elif not st.get("liveness", False):
                problems.append(f"{name}: not alive")
            elif not st.get("readiness", False):
                problems.append(f"{name}: not ready ({st.get('state', '?')})")
        return problems

    def wait_ready(self, timeout: float) -> list[str]:
        """等到所有下游就绪，返回仍未就绪的原因；空列表表示可以开跑。

        等状态而不是 sleep：recorder 收到 end 后要关文件，耗时不确定，
        固定等待要么不够、要么白等。abort 时不空等，立刻返回。
        """
        deadline = time.monotonic() + timeout
        while True:
            problems = self.check_ready()
            if not problems or self.abort or time.monotonic() >= deadline:
                return problems
            time.sleep(0.05)

    # ---------------------------------------------------------------- 前置检查
    def check_nodes(self) -> bool:
        """确认依赖的节点都在。缺一个就退出，不要带着半个系统跑 200 条。

        服务走 wait_for_service，action 走 wait_for_server，两者是两套发现机制，
        漏掉后者会出现"服务都在但专家根本没启动"的情况。
        """
        missing = []
        for name, cli in self.required.items():
            if not cli.wait_for_service(timeout_sec=2.0):
                missing.append(name)
        if not self.ac_expert.wait_for_server(timeout_sec=2.0):
            missing.append("expert.run_once")
        if missing:
            if self.dry_run:
                self.get_logger().warn(f"dry_run：缺少 {missing}，用空操作代替")
                return True
            self.get_logger().error(f"缺少必需服务/动作：{missing}，中止")
            return False
        self.get_logger().info(f"前置检查通过：{list(self.required) + ['expert.run_once']}")
        return True

    # ---------------------------------------------------------------- 服务调用
    def _wait_future(self, fut, timeout: float) -> bool:
        """等一个 future 完成。超时或收到 abort 都返回 False。

        用轮询而不是阻塞 wait，/orch/abort 才能立刻生效；
        abort 之后调用方负责善后（服务没法取消，action 要发 cancel）。
        """
        deadline = time.monotonic() + timeout
        while not fut.done():
            if self.abort:
                return False
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)
        return True

    def call(self, cli, name: str, req=None, timeout: float = 10.0) -> bool:
        """同步调用一个服务。req 为空时按 Trigger 发空请求。

        返回 True 表示服务端已经完成动作，不是"命令已接收"。
        等待期间会响应 abort：置位后立刻放弃等待并返回 False；但 ROS 超时/放弃
        都不会取消服务端执行，那边可能还在动，所以调用方要按"状态未知"处理。
        """
        if self.dry_run and not cli.service_is_ready():
            self.get_logger().info(f"[dry_run] {name}")
            return True
        if req is None:
            # 按客户端自己的类型造空请求：Trigger 只是本项目里 reset 的那种，
            # 写死成 Trigger 的话，给 Place 客户端发请求会直接 TypeError
            req = cli.srv_type.Request()
        fut = cli.call_async(req)
        deadline = time.monotonic() + timeout
        while not fut.done() and time.monotonic() < deadline:
            if self.abort:
                self.get_logger().warn(f"{name} 等待中被中止，服务端可能仍在执行")
                return False
            time.sleep(0.01)
        if not fut.done():
            self.get_logger().error(f"{name} 超时（{timeout:.1f}s）")
            return False
        res = fut.result()
        if not res.success:
            self.get_logger().warn(f"{name} 返回失败：{res.message}")
        return bool(res.success)

    def run_expert(self, idx: int, timeout: float = 60.0) -> bool:
        """让专家跑一次，返回它是否成功完成。

        用 action 的意义在这里：abort 时能真的 cancel goal，而不是丢下它继续动。
        """
        if self.dry_run and not self.ac_expert.server_is_ready():
            self.get_logger().info(f"[dry_run] expert.run_once id={idx}")
            return True

        goal = RunOnce.Goal()
        goal.id = idx
        send_fut = self.ac_expert.send_goal_async(goal, feedback_callback=self.on_expert_feedback)
        if not self._wait_future(send_fut, 5.0):
            self.get_logger().error(f"试次 {idx} 专家 goal 发送失败（abort={self.abort}）")
            return False

        goal_handle = send_fut.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error(f"试次 {idx} 专家拒绝了 goal")
            return False

        result_fut = goal_handle.get_result_async()
        if not self._wait_future(result_fut, timeout):
            why = "被中止" if self.abort else f"超过 {timeout:.0f}s"
            self.get_logger().warn(f"试次 {idx} 专家未完成（{why}），发送 cancel")
            # cancel 是协作式的，服务端不响应也拦不住，所以只等 2s 就放过
            self._wait_future(goal_handle.cancel_goal_async(), 2.0)
            return False

        res = result_fut.result().result
        if not res.success:
            self.get_logger().warn(f"试次 {idx} 专家失败：{res.message}")
        return bool(res.success)

    # ---------------------------------------------------------------- 主循环
    def run(self):
        try:
            if not self.check_nodes():
                return 1

            # 一轮只做一次场景初始化：杯子在本轮内不动，不需要每试次同步
            cup = [float(v) for v in self.task["spawn"]["cup"]["center"]]
            if not self.call(
                self.cli_setup,
                "sim.setup",
                Setup.Request(cup_x=cup[0], cup_y=cup[1], cup_z=cup[2]),
                timeout=5.0,
            ):
                self.get_logger().error("场景初始化失败，本轮不跑")
                return 1

            os.makedirs(self.out_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            summary = {
                "mode": self.mode,
                "task": self.task.get("name"),
                "started": stamp,
                "count": self.count,
                "rounds_per_position": self.rounds,
                "trials": [],
            }

            total = self.count * self.rounds
            idx = 0
            for i in range(self.count):
                x = self.base[0] + (i % 10 - 5) * self.radius / 5.0
                y = self.base[1] + ((i // 10) % 10 - 5) * self.radius / 5.0
                for r in range(self.rounds):
                    if self.abort or self.stop_requested:
                        break
                    idx += 1
                    self.get_logger().info(f"=== 试次 {idx}/{total}  位置 ({x:.3f}, {y:.3f}) 第 {r + 1} 轮")
                    trial = self.run_trial(idx, x, y)
                    summary["trials"].append(trial)
                    ok = sum(1 for t in summary["trials"] if t["success"])
                    self.get_logger().info(f"进度：{ok}/{len(summary['trials'])} 成功")
                if self.abort or self.stop_requested:
                    break

            summary["finished"] = time.strftime("%Y%m%d_%H%M%S")
            summary["success"] = sum(1 for t in summary["trials"] if t["success"])
            summary["aborted"] = self.abort
            path = os.path.join(self.out_dir, f"expert_run_{stamp}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            self.get_logger().info(
                f"结束：{summary['success']}/{len(summary['trials'])} 成功，明细 {path}"
            )
            return 0
        finally:
            # 无论正常跑完、被中止还是抛异常，都要放开 running，否则 /orch/start 再也起不来
            with self.state_lock:
                self.running = False
            self.get_logger().info("编排线程退出")

    def _trial_msg(self, idx: int, event: str) -> String:
        """构造试次边界事件：begin 表示开始判定/录制，end 表示本试次收尾。"""
        msg = String()
        msg.data = json.dumps({"id": idx, "event": event})
        return msg

    def run_trial(self, idx: int, x: float, y: float) -> dict:
        t0 = time.monotonic()
        trial = {"id": idx, "x": round(x, 4), "y": round(y, 4), "success": False, "reason": ""}

        # 0. 开跑前确认下游都活着且就绪。放在最前面：此时还没有任何副作用，
        #    跳过试次不会留下半截 episode，也不会把脏数据算进统计。
        #    上一个试次的 end 之后 recorder 还要收尾，所以这里等状态而不是固定 sleep。
        problems = self.wait_ready(self.ready_wait)
        if problems:
            trial["reason"] = "not_ready: " + "; ".join(problems)
            self.get_logger().warn(f"试次 {idx} 跳过：{trial['reason']}")
            return trial

        # 1. 摆方块 + 复位机械臂。合成一个服务是为了原子性：拆开会留下
        #    "方块已摆好但机械臂没回零位"的中间态，失败时还要单独善后。
        #    返回 true 表示两者都已完成，不用再额外等待。
        if not self.call(
            self.cli_place,
            "sim.place",
            Place.Request(x=x, y=y, z=self.cube_z),
            timeout=10.0,
        ):
            trial["reason"] = "place_failed"
            return trial

        # 2. 开窗 → 3. 专家
        #    先设 current_trial 再发 begin：detector 可能回得很快，id 对不上会被丢掉。
        #    录制由 recorder 自己订阅 begin/end 启停，不在这里调用。
        self.current_trial = idx
        self.result_event.clear()
        self.last_result = None
        self.pub_trial.publish(self._trial_msg(idx, "begin"))

        expert_ok = self.run_expert(idx, timeout=60.0)

        # 4. 等判定结论。专家没跑完就不等了：这时候判"有没有放进杯子"没有意义。
        got = self.result_event.wait(timeout=self.trial_timeout) if expert_ok else False
        if got and self.last_result is not None:
            trial["success"] = bool(self.last_result.get("success"))
            trial["reason"] = self.last_result.get("reason", "")
            trial["detector"] = self.last_result
        elif self.abort:
            trial["reason"] = "aborted"
        elif not expert_ok:
            trial["reason"] = "expert_failed"
        else:
            trial["reason"] = "no_result"

        # 5. 关窗。recorder 收到 end 自己 flush/close；它出了问题，下一个试次的
        #    check_ready 会发现（readiness 不再是 idle），不需要在这里同步等。
        self.pub_trial.publish(self._trial_msg(idx, "end"))
        trial["duration_s"] = round(time.monotonic() - t0, 2)
        return trial


def main():
    rclpy.init()
    node = Orchestrator()

    # 编排循环在主线程跑，rclpy 在后台线程转，这样服务调用可以用阻塞式写法而不会死锁
    exe = MultiThreadedExecutor(num_threads=4)
    exe.add_node(node)
    spin_thread = threading.Thread(target=exe.spin, daemon=True)
    spin_thread.start()

    code = 0
    try:
        # 不再自动开跑：等 /orch/start 触发，节点保持存活
        while rclpy.ok():
            time.sleep(0.2)
    except KeyboardInterrupt:
        # Ctrl-C 等同于 abort：打断当前试次，让编排线程自己收尾
        node.abort = True
        node.result_event.set()
        if node.run_thread is not None:
            node.run_thread.join(timeout=5.0)
        code = 130
    finally:
        exe.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
