#!/usr/bin/env python3
"""action_executor.py — 策略输出与机械臂之间的执行层。

它只做三件事：
  1. 按固定频率（默认 30Hz，必须等于训练时的 fps）把动作一点点发给 driver
  2. 队列水位低时向 policy 请求新的 chunk
  3. 做所有时序和安全上的防护（过期预测、跟踪误差、断流）

不关心模型是什么、chunk 有多长。换 ACT → SmolVLA → π0 这里一行不用改。

话题契约（临时，等 msg 包建好后换成 ActionChunk.msg）
  /policy/action_chunk  std_msgs/Float64MultiArray
      data[0] = 观测时刻（ROS 时钟秒，和 header.stamp 同一时间源）
      data[1] = chunk_size
      data[2:] = chunk_size × 7 个动作，行优先（6 关节弧度 + 夹爪 0~1）
  /policy/infer_request  std_msgs/Empty
  /joint_commands        std_msgs/Float64MultiArray（7 维）
  /joint_states          sensor_msgs/JointState（只用于算跟踪误差）
  /exec/status           std_msgs/String（JSON，5Hz）
                        其中 track_ratio / lag_ms 只作观测，不参与控制，也不告警

单独跑：
    python3 action_executor.py
"""

from __future__ import annotations

import json
import time
from collections import deque

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float64MultiArray, String
from std_srvs.srv import Trigger

ACTION_DIM = 7


def _sub6(a, b):
    return [x - y for x, y in zip(a[:6], b[:6])]


def _norm6(v):
    return sum(x * x for x in v) ** 0.5


class ActionExecutor(Node):
    def __init__(self):
        super().__init__("action_executor")

        # ---- 参数 ----------------------------------------------------------
        self.declare_parameter("control_hz", 30.0)
        # 请求节奏：下一个 chunk 要在当前 chunk 播完前多久发出。
        # 留够推理耗时即可——留太多会频繁覆盖还没播完的计划。
        self.declare_parameter("request_lead_s", 0.2)
        self.declare_parameter("request_timeout_s", 0.15)
        self.declare_parameter("chunk_max_age_s", 0.3)
        self.declare_parameter("hold_timeout_s", 1.0)
        self.declare_parameter("lag_min_dist", 0.03)     # 最近邻匹配的容差（rad）
        self.declare_parameter("ratio_window_s", 0.10)   # 增量比的统计窗口，要远大于采样周期
        self.declare_parameter("auto_start", True)

        p = self.get_parameter
        self.hz = float(p("control_hz").value)
        if self.hz <= 0:
            raise ValueError("control_hz 必须 > 0")
        self.dt = 1.0 / self.hz
        self.request_lead = float(p("request_lead_s").value)
        self.request_timeout = float(p("request_timeout_s").value)
        self.chunk_max_age = float(p("chunk_max_age_s").value)
        self.hold_timeout = float(p("hold_timeout_s").value)
        self.lag_min_dist = float(p("lag_min_dist").value)
        self.ratio_window = float(p("ratio_window_s").value)

        # ---- 状态 ----------------------------------------------------------
        self.queue: deque[list[float]] = deque()
        self.last_action: list[float] | None = None
        self.in_flight = False
        self.in_flight_since = 0.0
        self.queue_stamp = 0.0          # 当前队列对应 chunk 的观测时刻
        self.last_chunk_t = 0.0         # 上一段 chunk 的到达时刻（单调钟）
        self.last_chunk_dur = 0.0       # 上一段 chunk 覆盖多长时间（n × dt）
        self.running = bool(p("auto_start").value)

        self.latest_state: list[float] | None = None
        self.latest_state_t = 0.0
        # 跟踪质量的两个指标，见文件头说明
        self.ratio_ema = 1.0             # 增量比，1.0 = 完全跟随
        self.lag_ema_ms = 0.0            # 滞后时间
        self.lag_valid_t = 0.0           # 上次成功测到滞后的时刻
        self.cmd_hist: deque[tuple[float, list[float]]] = deque()
        self.state_hist: deque[tuple[float, list[float]]] = deque()
        self.hold_since: float | None = None

        # 统计
        self.n_chunks = 0
        self.n_dropped = 0
        self.n_late = 0
        self.n_requests = 0
        self.max_jitter = 0.0
        self.last_tick = 0.0
        self._last_status = 0.0

        # ---- ROS 接口 ------------------------------------------------------
        self.pub_cmd = self.create_publisher(Float64MultiArray, "/joint_commands", 10)
        self.pub_req = self.create_publisher(Empty, "/policy/infer_request", 10)
        self.pub_status = self.create_publisher(String, "/exec/status", 10)
        # 必须和 policy_node 的发布端一致（reliable + transient_local），
        # durability 不匹配时两边根本不会建立连接，症状是"话题在但收不到"
        qos_chunk = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.sub_chunk = self.create_subscription(
            Float64MultiArray, "/policy/action_chunk", self.on_chunk, qos_chunk
        )
        self.sub_state = self.create_subscription(
            JointState, "/joint_states", self.on_state, 10
        )
        self.srv_start = self.create_service(Trigger, "/exec/start", self.on_start)
        self.srv_stop = self.create_service(Trigger, "/exec/stop", self.on_stop)

        self.timer = self.create_timer(self.dt, self.tick)
        self.get_logger().info(
            f"action_executor 启动：{self.hz}Hz，提前量={self.request_lead * 1000:.0f}ms，"
            f"running={self.running}"
        )

    def _ros_now(self) -> float:
        """ROS 时钟（秒）。chunk 的 stamp 来自相机/仿真的 header.stamp，也是 ROS 时钟，
        两边必须用同一个源——否则有人开 use_sim_time，墙钟和 ROS 时钟差几十年，
        所有 chunk 都会被判成过期。"""
        return self.get_clock().now().nanoseconds / 1e9

    # ------------------------------------------------------------------ 回调
    def on_state(self, msg: JointState):
        if len(msg.position) < ACTION_DIM:
            return
        self.latest_state = [float(v) for v in msg.position[:ACTION_DIM]]
        self.latest_state_t = time.monotonic()
        self.state_hist.append((self._ros_now(), self.latest_state[:6]))
        cut = self._ros_now() - 2.0
        while self.state_hist and self.state_hist[0][0] < cut:
            self.state_hist.popleft()
        self._update_metrics()

    # ------------------------------------------------------------ 跟踪质量指标
    def _update_metrics(self):
        """估滞后 + 算增量比，两者都基于时间轴对齐，不做逐帧配对。

        为什么不能逐帧配对：状态流和指令流是独立采样的，各有抖动。当前状态匹配到
        第 i 条指令时，上一个状态未必匹配第 i-1 条。必须先把两条曲线在时间上对齐，
        再在一个较长的窗口上求增量比，抖动才会被平均掉。
        """
        if len(self.cmd_hist) < 3 or len(self.state_hist) < 2:
            return
        t_now, act = self.state_hist[-1]
        t_prev, prev_act = self.state_hist[-2]

        # 不动的时候两个指标都没意义：最近邻匹配不到唯一解，比值也全是噪声
        if _norm6(_sub6(act, prev_act)) < 0.001:
            return

        # --- 1. 滞后：当前角度最像多久之前发的那条指令
        best_t, best_d = None, 1e9
        for t, joints in self.cmd_hist:
            d = _norm6(_sub6(act, joints))
            if d < best_d:
                best_t, best_d = t, d
        if best_t is not None and best_d <= self.lag_min_dist:
            lag_ms = (t_now - best_t) * 1000.0
            if 0.0 <= lag_ms <= 500.0:
                self.lag_ema_ms = 0.9 * self.lag_ema_ms + 0.1 * lag_ms
                self.lag_valid_t = t_now
        # 匹配不上（落后太多）时不能直接退出：那样增量比就永远算不出来，
        # 越跟不上越"看起来正常"，正好是最该报警的情况。退化用已知延迟继续算。

        # --- 2. 增量比：实际运动 [now-W, now] 对齐到指令 [now-D-W, now-D]
        D = (self.lag_ema_ms / 1000.0) if self.lag_valid_t else 0.0
        W = self.ratio_window
        a_now = act
        a_old = self._interp(self.state_hist, t_now - W)
        c_now = self._interp(self.cmd_hist, t_now - D)
        c_old = self._interp(self.cmd_hist, t_now - D - W)
        if None in (a_old, c_now, c_old):
            return                                  # 历史还不够长

        d_act = _sub6(a_now, a_old)
        d_cmd = _sub6(c_now, c_old)
        n2 = sum(v * v for v in d_cmd)
        if n2 < 4e-6:                               # 窗口内指令没动，跳过
            return
        ratio = sum(a * b for a, b in zip(d_act, d_cmd)) / n2
        ratio = max(-2.0, min(2.0, ratio))
        self.ratio_ema = 0.95 * self.ratio_ema + 0.05 * ratio

    @staticmethod
    def _interp(hist, t):
        """在 (时刻, 关节) 历史里线性插值出 t 时刻的关节角。

        上限之外按零阶保持处理：指令发出去就一直有效，直到被下一条覆盖，
        所以 t 比最新一条还新时返回最新一条，而不是 None。
        （状态不同，状态是采样，超出范围就该算无效。）
        """
        if not hist or t < hist[0][0]:
            return None
        if t >= hist[-1][0]:
            return hist[-1][1]
        for i in range(1, len(hist)):
            t1, v1 = hist[i - 1]
            t2, v2 = hist[i]
            if t1 <= t <= t2:
                if t2 - t1 < 1e-9:
                    return v2
                k = (t - t1) / (t2 - t1)
                return [a + k * (b - a) for a, b in zip(v1, v2)]
        return None

    def on_chunk(self, msg: Float64MultiArray):
        data = msg.data
        if len(data) < 2:
            self._drop("chunk 长度不足")
            return
        stamp = float(data[0])
        n = int(data[1])
        if n <= 0 or len(data) != 2 + n * ACTION_DIM:
            self._drop(f"chunk 维度不符：len={len(data)} n={n}，应为 {2 + n * ACTION_DIM}")
            return

        # 时钟一致性检查：两条时间线差太多，说明有人开了 use_sim_time
        drift = abs(self._ros_now() - time.time())
        if drift > 60.0:
            self.get_logger().error(
                f"ROS 时钟和墙钟差 {drift / 3600:.1f} 小时，时间源不一致"
                "（可能开了 use_sim_time）", throttle_duration_sec=10.0,
            )

        age = self._ros_now() - stamp
        if age > self.chunk_max_age:
            # 过期预测：宁可继续用旧队列，也不要拿几秒前的观测来指挥现在
            self._drop(f"chunk 过期 {age * 1000:.0f}ms > {self.chunk_max_age * 1000:.0f}ms")
            return

        # 按时间戳对齐：算出该从第几步开始用。
        # 这一步同时处理了推理延迟和"旧队列还剩几个动作"两种情况，
        # 不需要区分覆盖/接续两种模式。
        k = int(round((self._ros_now() - stamp) / self.dt))
        k = max(0, min(k, n))
        if k >= n:
            self._drop(f"chunk 全部过期（k={k} n={n}）")
            return

        actions = [
            [float(v) for v in data[2 + i * ACTION_DIM : 2 + (i + 1) * ACTION_DIM]]
            for i in range(k, n)
        ]
        self.queue = deque(actions)
        self.queue_stamp = stamp
        self.last_chunk_t = time.monotonic()
        self.last_chunk_dur = n * self.dt
        self.hold_since = None
        self.in_flight = False
        self.n_chunks += 1
        # 老的两条保护：动作值本身是不是离谱
        bad = [i for i, a in enumerate(actions[-1]) if abs(a) > 10.0]
        if bad:
            self.get_logger().warn(f"chunk 末尾含异常值，索引 {bad}", throttle_duration_sec=2.0)

    def on_start(self, _req, resp):
        self.queue.clear()
        self.in_flight = False
        self.last_action = None
        self.hold_since = None
        self.last_chunk_t = 0.0
        self.last_chunk_dur = 0.0
        self.running = True
        resp.success = True
        resp.message = "started"
        self.get_logger().info("exec 启动")
        return resp

    def on_stop(self, _req, resp):
        self.queue.clear()
        self.in_flight = False
        self.running = False
        resp.success = True
        resp.message = "stopped"
        self.get_logger().info("exec 停止")
        return resp

    # ------------------------------------------------------------------ 主循环
    def tick(self):
        now = time.monotonic()
        if self.last_tick:
            self.max_jitter = max(self.max_jitter, abs((now - self.last_tick) - self.dt))
        self.last_tick = now

        if not self.running:
            return

        # 看不见真实状态就别再动
        if self.latest_state is None or now - self.latest_state_t > 1.0:
            self.get_logger().warn("收不到 /joint_states，停发", throttle_duration_sec=2.0)
            return

        self._step_command(now)
        self._maybe_request(now)
        self._publish_status(now)

    def _step_command(self, now):
        if self.queue:
            action = self.queue.popleft()
            self.last_action = action
            self.hold_since = None
            self._publish_cmd(action)
            return

        # 队列空：短暂保持最后一个动作，超过 hold_timeout 就彻底停发，
        # 让 driver 自己的看门狗接住（避免一直推着一个过时的目标）
        if self.last_action is None:
            return
        if self.hold_since is None:
            self.hold_since = now
        if now - self.hold_since <= self.hold_timeout:
            self._publish_cmd(self.last_action)
            return
        self.get_logger().warn("队列空且保持超时，停止下发", throttle_duration_sec=1.0)

    def _maybe_request(self, now):
        """按时间要 chunk，不按队列长度。

        队列长度做判据等价于"播完了才要"，于是每个 chunk 边界都要空等一次推理延迟，
        动作会周期性顿一下。正确的节奏是：下一段必须在当前段播完之前到达。
        请求周期 = chunk 时长 − 提前量，chunk 时长从收到的数据自己算。
        """
        if self.in_flight and now - self.in_flight_since > self.request_timeout:
            self.in_flight = False
            self.n_late += 1
            self.get_logger().warn(
                f"推理请求 {self.request_timeout}s 无回应", throttle_duration_sec=1.0
            )
        if self.in_flight:
            return

        if self.last_chunk_t == 0.0:
            # 还没收到过任何 chunk，立刻要一段
            self._send_request(now)
            return

        due = max(0.05, self.last_chunk_dur - self.request_lead)
        if now - self.last_chunk_t >= due:
            self._send_request(now)

    def _send_request(self, now):
        self.pub_req.publish(Empty())
        self.in_flight = True
        self.in_flight_since = now
        self.n_requests += 1

    def _publish_cmd(self, action):
        msg = Float64MultiArray()
        msg.data = [float(v) for v in action]
        self.pub_cmd.publish(msg)
        self._update_hist(action)

    def _update_hist(self, action):
        now = self._ros_now()
        self.cmd_hist.append((now, list(action[:6])))
        cut = now - 2.0
        while self.cmd_hist and self.cmd_hist[0][0] < cut:
            self.cmd_hist.popleft()

    def _publish_status(self, now):
        if now - self._last_status >= 0.2:
            self._last_status = now
            payload = {
                "running": self.running,
                "queue": len(self.queue),
                "in_flight": self.in_flight,
                "chunk_age_ms": round((self._ros_now() - self.queue_stamp) * 1000, 1)
                if self.queue_stamp
                else None,
                "since_chunk_ms": round((now - self.last_chunk_t) * 1000, 1)
                if self.last_chunk_t
                else None,
                "chunk_dur_ms": round(self.last_chunk_dur * 1000, 1),
                "track_ratio": round(self.ratio_ema, 3),
                # 落后太多时最近邻匹配不上，这时宁可报 null 也不要报一个假的 0
                "lag_ms": round(self.lag_ema_ms, 1)
                if self.lag_valid_t and self._ros_now() - self.lag_valid_t < 1.0
                else None,
                "chunks": self.n_chunks,
                "requests": self.n_requests,
                "dropped": self.n_dropped,
                "late": self.n_late,
                "jitter_ms": round(self.max_jitter * 1000, 2),
            }
            self.pub_status.publish(String(data=json.dumps(payload)))

    def _drop(self, why: str):
        self.n_dropped += 1
        self.get_logger().warn(f"丢弃 chunk：{why}", throttle_duration_sec=1.0)


def main():
    rclpy.init()
    node = ActionExecutor()
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
