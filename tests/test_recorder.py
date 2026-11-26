"""recorder_node 的录制链路测试。

分三层：

  1. FrameAligner：纯逻辑。两条 60Hz 关节状态夹住 30Hz 图像，配出
     observation.state / action；缺图、图像不同步、等右邻三种丢帧分类。

  2. EpisodeStaging：staging 落盘。jpg 按序号命名、npy 形状、空集、
     同名目录清理。

  3. RecorderNode：消息解析（行 padding、bgr8、关节重排）与 begin/end
     状态机、result 去重、status 的 readiness。

本文件不依赖 alicia_interfaces，也不需要 colcon build —— recorder 只用
标准消息。但它和 tests/conftest.py 那套「真节点 + 假下游」设施无关，所以
运行时加 --noconftest，避免被 conftest 里的 alicia_interfaces 跳过：

    docker run --rm --network host -v $PWD:/ws -w /ws --user root \
      alicia/recorder:1.0 \
      bash -lc "pip install -q pytest && \
                python3 -m pytest tests/test_recorder.py -q --noconftest"
"""

from __future__ import annotations

import array
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC_SRC = os.path.join(ROOT, "ros2_ws", "src", "recorder")
if REC_SRC not in sys.path:
    sys.path.insert(0, REC_SRC)

pytest.importorskip("rclpy", reason="需要 ROS 2 环境，请在容器内运行")
pytest.importorskip("cv2", reason="需要 opencv，recorder 镜像里有")

rec = pytest.importorskip("recorder_node", reason="需要 rclpy 与 cv2")

from sensor_msgs.msg import Image, JointState  # noqa: E402
from std_msgs.msg import String  # noqa: E402

FrameAligner = rec.FrameAligner
EpisodeStaging = rec.EpisodeStaging
RecorderNode = rec.RecorderNode


# ---------------------------------------------------------------- 公共设施
class FakeLogger:
    """收集日志，既方便断言，也避免依赖真实 Node。"""

    def __init__(self):
        self.records: list[tuple[str, str]] = []

    def info(self, msg, **_kwargs):
        self.records.append(("info", msg))

    def warn(self, msg, **_kwargs):
        self.records.append(("warn", msg))

    def error(self, msg, **_kwargs):
        self.records.append(("error", msg))

    def debug(self, msg, **_kwargs):
        self.records.append(("debug", msg))

    def messages(self, level: str) -> list[str]:
        return [m for lv, m in self.records if lv == level]


class FakePublisher:
    def __init__(self):
        self.sent = []

    def publish(self, msg):
        self.sent.append(msg)


class FakeDataset:
    """替身：只记录灌进来的帧，不碰 lerobot 与磁盘编码。"""

    def __init__(self):
        self.frames: list[dict] = []
        self.saved = 0

    def add_frame(self, frame):
        self.frames.append(frame)

    def save_episode(self):
        self.saved += 1


def make_image(rgb, encoding="rgb8", step_pad=0, stamp=1.0) -> Image:
    """按 sensor_msgs/Image 的布局造一帧，可选行尾 padding。

    step_pad > 0 时每行多出几个字节，专门用来验证 on_image 是按 step 切行
    再裁到 width*3，而不是直接 reshape。
    """
    arr = np.asarray(rgb, dtype=np.uint8)
    if encoding == "bgr8":
        arr = arr[:, :, ::-1]
    h, w = arr.shape[:2]
    row = np.ascontiguousarray(arr).reshape(h, w * 3)
    if step_pad:
        row = np.pad(row, ((0, 0), (0, step_pad)))

    msg = Image()
    msg.height, msg.width = h, w
    msg.encoding = encoding
    msg.step = w * 3 + step_pad
    msg.data = array.array("B", row.tobytes())
    msg.header.stamp.sec = int(stamp)
    msg.header.stamp.nanosec = int(round((stamp - int(stamp)) * 1e9))
    return msg


def make_state_msg(stamp: float, values, names=None) -> JointState:
    msg = JointState()
    msg.header.stamp.sec = int(stamp)
    msg.header.stamp.nanosec = int(round((stamp - int(stamp)) * 1e9))
    msg.name = list(names or rec.JOINTS)
    msg.position = [float(v) for v in values]
    return msg


IMG = np.zeros((6, 8, 3), dtype=np.uint8)


def frame_dict(i: int) -> dict:
    return {
        "stamp": 1.0 + i * 0.033,
        "state": np.full(7, float(i), dtype=np.float32),
        "action": np.full(7, float(i) + 1.0, dtype=np.float32),
        "images": {"front": IMG, "wrist": IMG},
    }


# ---------------------------------------------------------------- FrameAligner
def test_frame_pairs_neighbour_states():
    """左邻当 observation.state、右邻当 action。"""
    a = FrameAligner()
    a.push_state(10.000, [0.0] * 7)
    a.push_state(10.020, [1.0] * 7)
    a.push_image("front", 10.010, IMG)
    a.push_image("wrist", 10.010, IMG)

    frame = a.try_frame()

    assert frame is not None
    assert frame["state"][0] == pytest.approx(0.0)
    assert frame["action"][0] == pytest.approx(1.0)
    assert frame["stamp"] == pytest.approx(10.010)


def test_frame_waits_for_right_neighbour_without_burning_sample():
    """右邻没到时不推进 last_emitted，下一条 state 进来还能补出这一帧。"""
    a = FrameAligner()
    a.push_state(10.000, [0.0] * 7)
    a.push_image("front", 10.010, IMG)
    a.push_image("wrist", 10.010, IMG)

    assert a.try_frame() is None
    assert a.last_emitted is None

    a.push_state(10.020, [1.0] * 7)
    frame = a.try_frame()

    assert frame is not None
    assert frame["action"][0] == pytest.approx(1.0)


def test_frame_dropped_when_one_camera_missing():
    """少一路图不拿旧图凑数，计数 dropped_missing。"""
    a = FrameAligner()
    a.push_state(10.000, [0.0] * 7)
    a.push_state(10.020, [1.0] * 7)
    a.push_image("front", 10.010, IMG)

    assert a.try_frame() is None
    assert a.dropped_missing == 1
    assert a.last_emitted == pytest.approx(10.010)


def test_frame_dropped_when_images_unsynced():
    """两路图差超过 tol 说明有一路掉帧，计数 dropped_stale。"""
    a = FrameAligner()
    a.push_state(10.000, [0.0] * 7)
    a.push_state(10.200, [1.0] * 7)
    a.push_image("front", 10.010, IMG)
    a.push_image("wrist", 10.100, IMG)

    assert a.try_frame() is None
    assert a.dropped_stale == 1


def test_frame_not_emitted_twice_for_same_sample():
    """同一采样点只出一帧：图像和 state 两个回调都会尝试。"""
    a = FrameAligner()
    a.push_state(10.000, [0.0] * 7)
    a.push_state(10.020, [1.0] * 7)
    a.push_image("front", 10.010, IMG)
    a.push_image("wrist", 10.010, IMG)

    assert a.try_frame() is not None
    assert a.try_frame() is None


def test_state_clock_backwards_resets_pair():
    """时间倒退时丢掉旧的一对，避免后续配对出现负时间差。"""
    a = FrameAligner()
    a.push_state(10.020, [2.0] * 7)
    a.push_state(10.000, [0.0] * 7)

    assert a.state_prev is None
    assert a.state[1][0] == pytest.approx(0.0)


# ---------------------------------------------------------------- EpisodeStaging
def test_staging_write_and_finish(tmp_path):
    ep = EpisodeStaging(str(tmp_path), 3, ("front", "wrist"))
    for i in range(2):
        ep.write(frame_dict(i))

    n = ep.finish()

    assert n == 2
    assert os.path.isfile(os.path.join(ep.dir, "front", "000000.jpg"))
    assert os.path.isfile(os.path.join(ep.dir, "wrist", "000001.jpg"))
    states = np.load(os.path.join(ep.dir, "states.npy"))
    actions = np.load(os.path.join(ep.dir, "actions.npy"))
    stamps = np.load(os.path.join(ep.dir, "stamps.npy"))
    assert states.shape == (2, 7)
    assert actions[1][0] == pytest.approx(2.0)
    assert stamps[0] == pytest.approx(1.0)


def test_staging_empty_episode_writes_empty_arrays(tmp_path):
    """空集不抛异常，写成 (0, 7)，由调用方按帧数丢弃。"""
    ep = EpisodeStaging(str(tmp_path), 4, ("front",))

    assert ep.finish() == 0
    assert np.load(os.path.join(ep.dir, "states.npy")).shape == (0, 7)


def test_staging_clears_leftover_dir(tmp_path):
    """上一轮异常退出留下的同名目录要先清掉，否则两集的帧会混在一起。"""
    stale = tmp_path / "ep_000009"
    stale.mkdir()
    (stale / "leftover.txt").write_text("old", encoding="utf-8")

    ep = EpisodeStaging(str(tmp_path), 9, ("front",))

    assert not (stale / "leftover.txt").exists()
    assert os.path.isdir(os.path.join(ep.dir, "front"))


# ---------------------------------------------------------------- RecorderNode
@pytest.fixture
def node(tmp_path):
    """绕过 __init__（它要 rclpy.init + 读 task.yaml），只注入被测字段。"""
    n = object.__new__(RecorderNode)
    logger = FakeLogger()
    n.get_logger = lambda: logger
    n.logger = logger

    n.state = "idle"
    n.ep_id = None
    n.ep = None
    n.ep_result = None
    n.ep_event = threading.Event()
    n.result_timeout_s = 1.0
    n.aligner = FrameAligner()
    n.n_ok = 0
    n.n_fail = 0
    n.fps = 30
    n.language = "put the cube into the cup"
    n.main_root = str(tmp_path / "demos")
    n.failed_root = n.main_root + "_failed"
    n.staging_root = str(tmp_path / "staging")
    os.makedirs(n.staging_root, exist_ok=True)
    n.pub_status = FakePublisher()
    n._datasets = {}
    n._finalize_calls = []

    # 真回灌要 lerobot 与几十秒编码，这里只记录调用（状态机是被测对象）
    n._finalize = lambda ep, ep_id, count: n._finalize_calls.append((ep_id, count))
    n._flush_pool = ThreadPoolExecutor(max_workers=1)
    yield n
    n._flush_pool.shutdown(wait=True)


def test_on_image_handles_row_padding(node):
    """step 带行尾 padding 时，要按 step 切行再裁到 width*3。"""
    node.state = "recording"
    rgb = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)

    node.on_image("front", make_image(rgb, step_pad=7))

    stamp, got = node.aligner.images["front"]
    np.testing.assert_array_equal(got, rgb)
    assert stamp == pytest.approx(1.0)


def test_on_image_converts_bgr8(node):
    node.state = "recording"
    rgb = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)

    node.on_image("front", make_image(rgb, encoding="bgr8"))

    _, got = node.aligner.images["front"]
    np.testing.assert_array_equal(got, rgb)


def test_on_image_rejects_unsupported_encoding(node):
    node.state = "recording"
    msg = make_image(IMG)
    msg.encoding = "mono8"

    node.on_image("front", msg)

    assert "front" not in node.aligner.images
    assert node.logger.messages("warn")


def test_on_joint_state_reorders_by_name(node):
    """按名字取值，不信任消息里的排列顺序。"""
    node.state = "recording"
    names = ["gripper", "Joint3", "Joint1", "Joint6", "Joint2", "Joint5", "Joint4"]
    values = [0.9, 3.0, 1.0, 6.0, 2.0, 5.0, 4.0]

    node.on_joint_state(make_state_msg(10.0, values, names))

    assert node.aligner.state is not None
    assert list(node.aligner.state[1]) == pytest.approx(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.9]
    )


def test_on_joint_state_drops_message_missing_a_joint(node):
    """缺关节就整条丢弃，不能补 0。"""
    node.state = "recording"
    msg = make_state_msg(10.0, [0.0] * 6, names=rec.JOINTS[:6])

    node.on_joint_state(msg)

    assert node.aligner.state is None
    assert node.logger.messages("warn")


def test_on_trial_begin_then_end(node):
    node.on_trial(String(data=json.dumps({"id": 5, "event": "begin"})))

    assert node.state == "recording"
    assert node.ep is not None and node.ep.ep_id == 5

    node.on_trial(String(data=json.dumps({"id": 5, "event": "end"})))
    node._flush_pool.shutdown(wait=True)

    assert node.state == "flushing"
    assert node._finalize_calls == [(5, 0)]


def test_on_trial_begin_ignored_while_busy(node):
    node.state = "recording"
    node.ep_id = 5

    node.on_trial(String(data=json.dumps({"id": 6, "event": "begin"})))

    assert node.ep_id == 5
    assert node.logger.messages("warn")


def test_on_trial_end_without_begin_is_ignored(node):
    node.on_trial(String(data=json.dumps({"id": 1, "event": "end"})))

    assert node.state == "idle"
    assert node._finalize_calls == []


def test_on_result_ignores_stale_id(node):
    """上一个试次迟到的结论不能落到这一集上。"""
    node.ep_id = 3

    node.on_result(String(data=json.dumps({"id": 2, "success": True})))

    assert node.ep_result is None
    assert not node.ep_event.is_set()


def test_on_result_accepts_current_and_wakes_finalize(node):
    node.ep_id = 3

    node.on_result(String(data=json.dumps({"id": 3, "success": True})))

    assert node.ep_result["success"] is True
    assert node.ep_event.is_set()


def test_on_result_bad_json_counts_as_failure(node):
    """解析失败按失败处理：宁可误标失败，也不能把失败标成成功。"""
    node.ep_id = 3

    node.on_result(String(data="{not json"))

    assert node.ep_result["success"] is False
    assert node.ep_event.is_set()


def test_publish_status_readiness_follows_state(node):
    node.publish_status()
    payload = json.loads(node.pub_status.sent[-1].data)
    assert payload["readiness"] is True

    node.state = "recording"
    node.publish_status()
    payload = json.loads(node.pub_status.sent[-1].data)
    assert payload["readiness"] is False


def test_ingest_feeds_frames_in_order(tmp_path, node):
    ep = EpisodeStaging(str(tmp_path), 7, ("front",))
    for i in range(3):
        ep.write(
            {
                "stamp": 1.0,
                "state": np.full(7, float(i)),
                "action": np.full(7, float(i) + 1.0),
                "images": {"front": IMG},
            }
        )
    n = ep.finish()

    ds = FakeDataset()
    node._ingest(ds, ep, n, {"success": True})

    assert ds.saved == 1
    assert [f["action"][0] for f in ds.frames] == pytest.approx([1.0, 2.0, 3.0])
    assert ds.frames[0]["observation.images.front"].shape == IMG.shape
    assert ds.frames[0]["task"] == node.language


def test_ingest_raises_when_image_missing(tmp_path, node):
    ep = EpisodeStaging(str(tmp_path), 8, ("front",))
    ep.write(
        {
            "stamp": 1.0,
            "state": np.zeros(7),
            "action": np.ones(7),
            "images": {"front": IMG},
        }
    )
    n = ep.finish()
    os.remove(os.path.join(ep.dir, "front", "000000.jpg"))

    with pytest.raises(RuntimeError):
        node._ingest(FakeDataset(), ep, n, {"success": True})


# ---------------------------------------------------------------- 补充：配对边界
def test_try_frame_returns_none_without_images():
    """只有关节状态、还没来图时不出帧。"""
    a = FrameAligner()
    a.push_state(10.000, [0.0] * 7)
    a.push_state(10.020, [1.0] * 7)

    assert a.try_frame() is None


def test_frame_dropped_when_state_jumped_past_sample():
    """图像迟到、state 已经跑过采样点，右邻等不到了，记 stale 而不是干等。"""
    a = FrameAligner()
    a.push_state(10.100, [0.0] * 7)
    a.push_state(10.120, [1.0] * 7)
    a.push_image("front", 10.000, IMG)
    a.push_image("wrist", 10.000, IMG)

    assert a.try_frame() is None
    assert a.dropped_stale == 1


def test_push_image_ignores_unknown_key():
    """话题名写错时不该在 aligner 里多出一个计数不到的槽位。"""
    a = FrameAligner()

    a.push_image("side", 1.0, IMG)

    assert a.images == {}


def test_push_state_stores_float32():
    a = FrameAligner()

    a.push_state(1.0, [1, 2, 3, 4, 5, 6, 7])

    assert a.state[1].dtype == np.float32


# ---------------------------------------------------------------- 补充：回调门控
def test_callbacks_ignored_unless_recording(node):
    """idle 时不该往 aligner 里攒数据，否则第一帧会拿到上一集的残留。"""
    node.state = "idle"

    node.on_joint_state(make_state_msg(1.0, [0.0] * 7))
    node.on_image("front", make_image(IMG))

    assert node.aligner.state is None
    assert node.aligner.images == {}


def test_on_image_falls_back_to_wall_clock_without_stamp(node):
    """驱动不打时间戳时退回本机时刻。"""
    node.state = "recording"
    msg = make_image(IMG)
    msg.header.stamp.sec = 0
    msg.header.stamp.nanosec = 0

    before = time.time()
    node.on_image("front", msg)
    stamp, _ = node.aligner.images["front"]

    assert stamp >= before


def test_on_trial_ignores_unknown_event_and_bad_json(node):
    node.on_trial(String(data=json.dumps({"id": 1, "event": "pause"})))
    node.on_trial(String(data="not json"))

    assert node.state == "idle"
    assert len(node.logger.messages("warn")) == 2


def test_on_trial_end_with_wrong_id_ignored(node):
    node.on_trial(String(data=json.dumps({"id": 5, "event": "begin"})))

    node.on_trial(String(data=json.dumps({"id": 6, "event": "end"})))

    assert node.state == "recording"
    assert node._finalize_calls == []


def test_begin_creates_fresh_aligner(node):
    """begin 必须换掉对齐器：上一集残留的 state 会让第一帧张冠李戴。"""
    node.aligner.push_state(1.0, [0.0] * 7)

    node.on_trial(String(data=json.dumps({"id": 1, "event": "begin"})))

    assert node.aligner.state is None
    assert node.aligner.state_prev is None


def test_publish_status_includes_frames_while_recording(node):
    node.state = "recording"
    node.ep = EpisodeStaging(node.staging_root, 1, ("front", "wrist"))
    node.ep.write(frame_dict(0))

    node.publish_status()

    payload = json.loads(node.pub_status.sent[-1].data)
    assert payload["frames"] == 1
    assert payload["state"] == "recording"


# ---------------------------------------------------------------- 补充：_finalize 分流
def _staged(tmp_path, ep_id=5, frames=2):
    """造一集已经落盘完毕的 staging，供 _finalize 消费。"""
    ep = EpisodeStaging(str(tmp_path), ep_id, ("front",))
    for i in range(frames):
        ep.write(frame_dict(i))
    return ep, ep.finish()


def _wire_fake_ingest(node, sink):
    node._dataset = lambda root: root
    node._ingest = lambda ds, _ep, _n, _res: sink.append(ds)


def test_finalize_success_goes_to_main_and_cleans(node, tmp_path):
    ep, n = _staged(tmp_path)
    node.state, node.ep_id, node.ep = "flushing", 5, ep
    node.ep_result = {"id": 5, "success": True}
    node.ep_event.set()
    ingested = []
    _wire_fake_ingest(node, ingested)

    RecorderNode._finalize(node, ep, 5, n)

    assert node.state == "idle"
    assert (node.n_ok, node.n_fail) == (1, 0)
    assert ingested == [node.main_root]
    assert not os.path.exists(ep.dir)


def test_finalize_failure_goes_to_failed_dataset(node, tmp_path):
    ep, n = _staged(tmp_path)
    node.state, node.ep_id, node.ep = "flushing", 5, ep
    node.ep_result = {"id": 5, "success": False, "reason": "miss"}
    node.ep_event.set()
    ingested = []
    _wire_fake_ingest(node, ingested)

    RecorderNode._finalize(node, ep, 5, n)

    assert (node.n_ok, node.n_fail) == (0, 1)
    assert ingested == [node.failed_root]


def test_finalize_timeout_archives_as_failed(node, tmp_path):
    """判定丢了按失败归档：这一试次可疑，但帧留给事后分析。"""
    ep, n = _staged(tmp_path)
    node.state, node.ep_id, node.ep = "flushing", 5, ep
    node.result_timeout_s = 0.01
    got = []
    node._dataset = lambda root: root
    node._ingest = lambda ds, _ep, _n, res: got.append((ds, res["reason"]))

    RecorderNode._finalize(node, ep, 5, n)

    assert got == [(node.failed_root, "result_timeout")]
    assert node.n_fail == 1


def test_finalize_ingest_error_counts_fail_and_still_cleans(node, tmp_path):
    """回灌炸了也要删 staging 并回到 idle，否则整轮采集卡死。"""
    ep, n = _staged(tmp_path)
    node.state, node.ep_id, node.ep = "flushing", 5, ep
    node.ep_result = {"id": 5, "success": True}
    node.ep_event.set()
    node._dataset = lambda root: root

    def boom(*_args):
        raise RuntimeError("disk full")

    node._ingest = boom

    RecorderNode._finalize(node, ep, 5, n)

    assert node.n_fail == 1
    assert node.state == "idle"
    assert not os.path.exists(ep.dir)
    assert node.logger.messages("error")


def test_finalize_empty_episode_discarded_without_counting(node, tmp_path):
    ep = EpisodeStaging(str(tmp_path), 5, ("front",))
    n = ep.finish()
    node.state, node.ep_id, node.ep = "flushing", 5, ep
    node.ep_event.set()
    node._dataset = lambda _root: pytest.fail("空集不该碰数据集")

    RecorderNode._finalize(node, ep, 5, n)

    assert (node.n_ok, node.n_fail) == (0, 0)
    assert node.state == "idle"
    assert not os.path.exists(ep.dir)


def test_shutdown_drops_unfinished_recording(node):
    """退出时没录完的集直接删：没有 result 就无法归档。"""
    node.state = "recording"
    node.ep = EpisodeStaging(node.staging_root, 11, ("front", "wrist"))
    node.ep.write(frame_dict(0))
    ep_dir = node.ep.dir

    node.shutdown()

    assert not os.path.exists(ep_dir)
    assert node.ep is None
