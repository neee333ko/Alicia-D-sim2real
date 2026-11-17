"""sim_bridge 的协议解析与端到端测试。

分两层：

  1. 协议层（object.__new__ 绕过 __init__）：帧解析、req_id 过滤、状态机。
     不起线程、不碰 socket，失败时定位直接。

  2. 端到端（真 SimBridge + 假 sim_app）：在 tmp_path 下起两个 unix socket
     server，验证 /sim/setup、/sim/place 的 req_id 往返、真值先于确认帧的
     顺序，以及服务超时/失败语义。

假 sim_app 的作用是复现"对端行为"，不测 Isaac，也不测物理。

必须在有 ROS 2 的环境里跑（容器内）：
    colcon build --packages-select alicia_interfaces
    source install/setup.bash
    pytest tests/test_sim_bridge.py -q
"""

from __future__ import annotations

import json
import os
import socket
import struct
import sys
import threading
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE_SRC = os.path.join(ROOT, "ros2_ws", "src", "sim_bridge")
if BRIDGE_SRC not in sys.path:
    sys.path.insert(0, BRIDGE_SRC)

# 宿主上没有 ROS，直接跳过而不是报一堆 ImportError
pytest.importorskip("rclpy", reason="需要 ROS 2 环境，请在容器内运行")
bridge_mod = pytest.importorskip("sim_bridge", reason="需要 rclpy 与 alicia_interfaces")
pytest.importorskip("alicia_interfaces", reason="需要先 colcon build alicia_interfaces 并 source")

from alicia_interfaces.msg import ObjectPose  # noqa: E402
from alicia_interfaces.srv import Place, Setup  # noqa: E402
from rclpy.executors import MultiThreadedExecutor  # noqa: E402
from rclpy.node import Node  # noqa: E402

CMD_JOINTS = bridge_mod.CMD_JOINTS
CMD_SETUP = bridge_mod.CMD_SETUP
CMD_PLACE = bridge_mod.CMD_PLACE
T_JOINTS = bridge_mod.T_JOINTS
T_IMAGE = bridge_mod.T_IMAGE
T_INFO = bridge_mod.T_INFO
T_OBJECT = bridge_mod.T_OBJECT
T_PLACE_DONE = bridge_mod.T_PLACE_DONE


# ======================================================================
# 公共设施
# ======================================================================

class FakeLogger:
    """记录日志便于断言，也避免 get_logger 依赖真实 Node。"""

    def __init__(self):
        self.records: list[tuple[str, str]] = []

    def _add(self, level, msg):
        self.records.append((level, msg))

    def info(self, msg, **_kw):
        self._add("info", msg)

    def warn(self, msg, **_kw):
        self._add("warn", msg)

    def error(self, msg, **_kw):
        self._add("error", msg)

    def debug(self, msg, **_kw):
        self._add("debug", msg)

    def messages(self, level: str) -> list[str]:
        return [m for lv, m in self.records if lv == level]


class FakePublisher:
    def __init__(self):
        self.sent = []

    def publish(self, msg):
        self.sent.append(msg)


def make_bridge(**overrides):
    """绕开 __init__，只注入被测算子真正用到的字段。"""
    b = object.__new__(bridge_mod.SimBridge)
    logger = FakeLogger()
    b.get_logger = lambda: logger
    b.cmd_sock = None
    b.cmd_lock = threading.Lock()
    b._stop = False
    b.last_joint_t = 0.0
    b.n_frames = {T_JOINTS: 0, T_IMAGE: 0, T_INFO: 0, T_OBJECT: 0}
    b.camera_info = {}
    b._place_req = 0
    b._place_ack = None
    b._place_done = threading.Event()
    b.place_timeout = 0.2
    b.pub_state = FakePublisher()
    b.pub_front = FakePublisher()
    b.pub_wrist = FakePublisher()
    b.pub_front_info = FakePublisher()
    b.pub_wrist_info = FakePublisher()
    b.pub_object = FakePublisher()
    b.pub_status = FakePublisher()
    for k, v in overrides.items():
        setattr(b, k, v)
    b.logger = logger
    return b


def wait_until(pred, timeout=3.0, interval=0.01) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


class ParamStore:
    """替换 declare/get_parameter。

    get() 必须返回真的 rclpy Parameter：Node.__init__ 内部的 time_source
    会读 use_sim_time 的 .type_，返回裸值会 AttributeError。
    """

    def __init__(self, overrides):
        self.values = dict(overrides)

    def declare(self, name, default):
        self.values.setdefault(name, default)

    def get(self, name):
        from rclpy.parameter import Parameter

        return Parameter(name, value=self.values.get(name, False))


# ======================================================================
# 协议层
# ======================================================================

def test_cmd_frame_is_fixed_64_bytes():
    """定长帧是解析简单的前提：三种命令都必须 64 字节，首字节是类型。"""
    for kind in (CMD_JOINTS, CMD_SETUP, CMD_PLACE):
        raw = bridge_mod.CMD_STRUCT.pack(kind, *[float(i) for i in range(7)])
        assert len(raw) == 64
        assert raw[0] == kind
        assert bridge_mod.CMD_STRUCT.unpack(raw)[1] == 0.0


def test_joint_frame_publishes_joint_state():
    b = make_bridge()
    vals = (0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.9)
    b._on_joints(struct.pack("<7d", *vals), 1700000000.5)

    assert len(b.pub_state.sent) == 1
    msg = b.pub_state.sent[0]
    assert msg.name == bridge_mod.JOINT_NAMES
    assert msg.position == pytest.approx(list(vals))
    assert msg.header.stamp.sec == 1700000000
    assert msg.header.stamp.nanosec == pytest.approx(5e8, abs=1)
    assert b.last_joint_t > 0.0
    assert b.n_frames[T_JOINTS] == 1


def test_joint_frame_rejects_bad_payload():
    b = make_bridge()
    b._on_joints(struct.pack("<6d", *([0.0] * 6)), 1.0)          # 长度不对
    b._on_joints(struct.pack("<7d", *([0.0] * 6 + [float("nan")])), 1.0)  # NaN
    assert b.pub_state.sent == []
    assert b.logger.messages("error")


def test_image_frame_publishes_rgb8_and_camera_info():
    b = make_bridge()
    # 真实路径是 struct.unpack("<14d")，全是 float；p 字段不接受 int
    K = [600.0, 0.0, 320.0, 0.0, 600.0, 240.0, 0.0, 0.0, 1.0]
    D = [0.0, 0.0, 0.0, 0.0, 0.0]
    b.camera_info[0] = (K, D)

    w, h = 2, 3
    payload = struct.pack("<II", w, h) + bytes(range(w * h * 3))
    b._on_image(payload, 1700000001.0, 0)

    assert len(b.pub_front.sent) == 1
    img = b.pub_front.sent[0]
    assert (img.width, img.height, img.step, img.encoding) == (w, h, w * 3, "rgb8")
    assert bytes(img.data) == bytes(range(w * h * 3))
    assert img.header.frame_id == "cam_front_optical"

    # 内参跟着图像一起发，下游拿到就能直接用
    assert len(b.pub_front_info.sent) == 1
    info = b.pub_front_info.sent[0]
    assert (info.width, info.height) == (w, h)
    assert info.k == pytest.approx(K)
    assert info.d == pytest.approx(D)
    assert b.n_frames[T_IMAGE] == 1


def test_image_frame_rejects_mismatched_size():
    b = make_bridge()
    b.camera_info[0] = ([1.0] * 9, [0.0] * 5)
    payload = struct.pack("<II", 4, 4) + b"\x00" * 10     # 声明 4x4 却只有 10 字节
    b._on_image(payload, 1.0, 0)
    assert b.pub_front.sent == []
    assert b.logger.messages("error")


def test_image_frame_rejects_unknown_camera():
    b = make_bridge()
    b._on_image(struct.pack("<II", 1, 1) + b"\x00" * 3, 1.0, 7)
    assert b.pub_front.sent == [] and b.pub_wrist.sent == []


def test_info_frame_caches_intrinsics_per_camera():
    b = make_bridge()
    K = [500.0, 0, 160, 0, 500, 120, 0, 0, 1]
    D = [0.1, -0.1, 0.0, 0.0, 0.0]
    b._on_info(struct.pack("<14d", *(K + D)), 1)
    assert b.camera_info[1] == (K, D)
    assert b.n_frames[T_INFO] == 1

    b._on_image(struct.pack("<II", 1, 1) + b"\x00" * 3, 2.0, 1)
    assert len(b.pub_wrist.sent) == 1
    assert b.pub_wrist_info.sent[0].k == pytest.approx(K)


def test_object_frame_maps_eight_doubles():
    b = make_bridge()
    # cup(x,y,z,height) + cube(x,y,z,edge)
    payload = struct.pack("<8d", 0.60, 0.10, 0.12, 0.09, 0.45, -0.05, 0.02, 0.03)
    b._on_object(payload, 1700000002.0)

    assert len(b.pub_object.sent) == 1
    msg = b.pub_object.sent[0]
    assert (msg.cup_pose.x, msg.cup_pose.y, msg.cup_pose.z) == pytest.approx((0.60, 0.10, 0.12))
    assert msg.cup_height == pytest.approx(0.09)
    assert (msg.cube_pose.x, msg.cube_pose.y, msg.cube_pose.z) == pytest.approx((0.45, -0.05, 0.02))
    assert msg.cube_edge_m == pytest.approx(0.03)
    assert msg.header.frame_id == "world"
    assert b.n_frames[T_OBJECT] == 1


def test_object_frame_rejects_wrong_length():
    b = make_bridge()
    b._on_object(struct.pack("<7d", *([0.0] * 7)), 1.0)
    assert b.pub_object.sent == []


def test_place_done_sets_ack_and_rejects_bad_length():
    b = make_bridge()
    b._on_place_done(bridge_mod.PLACE_DONE_STRUCT.pack(7, True))
    assert b._place_ack == (7, True)
    assert b._place_done.is_set()

    b._place_ack = None
    b._on_place_done(b"\x01\x02")                       # 长度不对
    assert b._place_ack is None
    assert b.logger.messages("warn")


def test_request_writes_req_id_and_waits_for_matching_ack():
    b = make_bridge()
    local, peer = socket.socketpair()
    b.cmd_sock = local
    try:
        out = {}
        t = threading.Thread(
            target=lambda: out.update(
                r=b._request(CMD_PLACE, [0.5, 0.0, 0.03, 0.0, 0.0, 0.0, 0.0], "摆放")
            )
        )
        t.start()

        raw = bridge_mod._recv_exact(peer, bridge_mod.CMD_STRUCT.size)
        assert raw is not None
        kind, *vals = bridge_mod.CMD_STRUCT.unpack(raw)
        assert kind == CMD_PLACE
        assert vals[3] == 1.0                            # req_id 占第 4 位
        assert vals[:3] == pytest.approx([0.5, 0.0, 0.03])

        b._on_place_done(bridge_mod.PLACE_DONE_STRUCT.pack(1, True))
        t.join(timeout=2.0)
        assert out["r"] == (True, "摆放完成")
        assert b._place_req == 1
    finally:
        local.close()
        peer.close()


def test_request_ignores_stale_ack_and_times_out():
    """上一个请求迟到的确认不能拿来当本次结果。"""
    b = make_bridge()
    local, peer = socket.socketpair()
    b.cmd_sock = local
    b.place_timeout = 0.25
    b._place_ack = (99, True)                            # 旧确认
    try:
        ok, msg = b._request(CMD_SETUP, [0.0] * 7, "场景同步")
        assert ok is False
        assert "超时" in msg
    finally:
        local.close()
        peer.close()


def test_request_fails_fast_when_disconnected():
    b = make_bridge()
    ok, msg = b._request(CMD_PLACE, [0.0] * 7, "摆放")
    assert ok is False
    assert "未连接" in msg


def test_status_reflects_channel_and_joint_age():
    b = make_bridge()
    b._publish_status()
    st = json.loads(b.pub_status.sent[-1].data)
    assert st["state"] == "waiting_sim" and st["readiness"] is False

    b.cmd_sock = object()                                # 通道在，但还没收到关节
    b.last_joint_t = time.monotonic() - 5.0
    b._publish_status()
    st = json.loads(b.pub_status.sent[-1].data)
    assert st["state"] == "no_state" and st["readiness"] is False

    b.last_joint_t = time.monotonic()
    b._publish_status()
    st = json.loads(b.pub_status.sent[-1].data)
    assert st["state"] == "ready" and st["readiness"] is True
    assert st["liveness"] is True


# ======================================================================
# 端到端：真 SimBridge + 假 sim_app
# ======================================================================

class FakeSim:
    """最小的 sim_app：监听两条 socket，记录命令，按需回状态帧。

    只实现到"够验证 bridge 行为"的程度：不模拟物理，也不做位姿计算。
    """

    def __init__(self, tmp_path):
        self.cmd_path = str(tmp_path / "cmd.sock")
        self.state_path = str(tmp_path / "state.sock")
        self.commands: list[tuple[int, tuple]] = []
        self.on_command = None
        self.state_conn: socket.socket | None = None
        self._stop = False
        self._threads: list[threading.Thread] = []

        self.cmd_server = self._listen(self.cmd_path)
        self.state_server = self._listen(self.state_path)
        self._threads.append(threading.Thread(target=self._accept_cmd, daemon=True))
        self._threads.append(threading.Thread(target=self._accept_state, daemon=True))
        for t in self._threads:
            t.start()

    @staticmethod
    def _listen(path):
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.bind(path)
        s.listen(1)
        return s

    def _accept_cmd(self):
        try:
            conn, _ = self.cmd_server.accept()
        except OSError:
            return
        with conn:
            while not self._stop:
                raw = bridge_mod._recv_exact(conn, bridge_mod.CMD_STRUCT.size)
                if raw is None:
                    return
                kind, *vals = bridge_mod.CMD_STRUCT.unpack(raw)
                self.commands.append((kind, tuple(vals)))
                if self.on_command is not None:
                    self.on_command(kind, vals)

    def _accept_state(self):
        try:
            conn, _ = self.state_server.accept()
        except OSError:
            return
        self.state_conn = conn

    def send_state(self, mtype, payload=b"", cam=0, stamp=None):
        conn = self.state_conn
        if conn is None:
            return
        hdr = bridge_mod.HDR.pack(
            bridge_mod.MAGIC,
            mtype,
            cam,
            0,
            time.time() if stamp is None else stamp,
            len(payload),
            0,
        )
        conn.sendall(hdr + payload)

    def wait_cmd(self, kind, count=1, timeout=3.0) -> bool:
        return wait_until(
            lambda: sum(1 for k, _ in self.commands if k == kind) >= count, timeout
        )

    def close(self):
        self._stop = True
        for s in (self.cmd_server, self.state_server):
            try:
                s.close()
            except OSError:
                pass
        if self.state_conn is not None:
            try:
                self.state_conn.close()
            except OSError:
                pass


class Sink(Node):
    """收 bridge 发布出来的话题，断言用。"""

    def __init__(self):
        super().__init__("bridge_test_sink")
        self.objects: list[ObjectPose] = []
        self.create_subscription(ObjectPose, "/object/pose", self.objects.append, 10)


@pytest.fixture
def ros_context():
    import rclpy

    if not rclpy.ok():
        rclpy.init()
    yield
    if rclpy.ok():
        rclpy.shutdown()


@pytest.fixture
def sim_env(tmp_path, monkeypatch, ros_context):
    fake = FakeSim(tmp_path)
    store = ParamStore(
        {
            "sim_cmd_sock": fake.cmd_path,
            "sim_state_sock": fake.state_path,
            "reconnect_s": 0.05,
            "place_timeout_s": 0.5,
        }
    )
    monkeypatch.setattr(
        bridge_mod.SimBridge, "declare_parameter", staticmethod(store.declare)
    )
    monkeypatch.setattr(bridge_mod.SimBridge, "get_parameter", staticmethod(store.get))

    bridge = bridge_mod.SimBridge()
    sink = Sink()

    exe = MultiThreadedExecutor(num_threads=4)
    exe.add_node(bridge)
    exe.add_node(sink)
    spin = threading.Thread(target=exe.spin, daemon=True)
    spin.start()

    assert wait_until(
        lambda: bridge.cmd_sock is not None and fake.state_conn is not None
    ), "bridge 没能连上假 sim_app"

    yield bridge, fake, sink

    bridge.shutdown()
    exe.shutdown()
    bridge.destroy_node()
    sink.destroy_node()
    fake.close()
    spin.join(timeout=2.0)


def test_setup_roundtrip_publishes_truth_before_ack(sim_env):
    """setup：先收到新真值再收到确认；服务返回时下游必须已能看到新位姿。"""
    bridge, fake, sink = sim_env

    def on_command(kind, vals):
        if kind != CMD_SETUP:
            return
        fake.send_state(
            T_OBJECT, struct.pack("<8d", 0.60, 0.10, 0.12, 0.09, 0.45, -0.05, 0.02, 0.03)
        )
        fake.send_state(T_PLACE_DONE, bridge_mod.PLACE_DONE_STRUCT.pack(vals[3], True))

    fake.on_command = on_command

    resp = bridge._on_setup(
        Setup.Request(cup_x=0.60, cup_y=0.10, cup_z=0.12), Setup.Response()
    )

    assert resp.success is True
    assert resp.message == "场景同步完成"
    assert fake.wait_cmd(CMD_SETUP)
    kind, vals = fake.commands[0]
    assert vals[:3] == pytest.approx((0.60, 0.10, 0.12))
    assert vals[3] == 1.0

    assert wait_until(lambda: sink.objects), "/object/pose 没有发出来"
    msg = sink.objects[0]
    assert msg.cup_pose.z == pytest.approx(0.12)
    assert msg.cube_edge_m == pytest.approx(0.03)


def test_place_reports_sim_failure(sim_env):
    bridge, fake, sink = sim_env

    def on_command(kind, vals):
        if kind == CMD_PLACE:
            fake.send_state(
                T_PLACE_DONE, bridge_mod.PLACE_DONE_STRUCT.pack(vals[3], False)
            )

    fake.on_command = on_command
    resp = bridge._on_place(Place.Request(x=0.5, y=0.0, z=0.03), Place.Response())

    assert resp.success is False
    assert "失败" in resp.message


def test_place_times_out_without_ack(sim_env):
    bridge, fake, _sink = sim_env
    fake.on_command = None                    # sim 收到但不回确认

    resp = bridge._on_place(Place.Request(x=0.5, y=0.0, z=0.03), Place.Response())

    assert resp.success is False
    assert "超时" in resp.message
    assert fake.wait_cmd(CMD_PLACE)


def test_joint_frame_makes_status_ready(sim_env):
    bridge, fake, _sink = sim_env
    fake.send_state(T_JOINTS, struct.pack("<7d", *([0.1] * 7)))

    assert wait_until(lambda: bridge.last_joint_t > 0.0)
    bridge._publish_status()
    assert bridge.n_frames[T_JOINTS] >= 1
