#!/usr/bin/env python3
"""L2 集成测试：仿真专家轨迹采集全链路。

被测链路（真进程、真 ROS、真 socket）：

    orchestrator ──/sim/place──→ sim_bridge ──unix socket──→ sim_app
         │                            │
         ├──/orch/trial──→ recorder   ├──/joint_states──→ sim_expert
         │                            └──/camera/*──────→ sim_expert
         └──/orch/result←── success_detector ←──/object/pose────┘

为什么不用真 Isaac：L2 只回答「节点之间接得对不对」。物理和渲染的正确性
属于 L1（sim_app 自己的用例），在 GPU 机器上单独跑。fake_sim_app 用协议
兼容的假数据顶替，省掉一次 40 秒的 Isaac 启动和显存占用。

跑法（宿主机执行，需要 docker 与 robot_sim 环境）：

    RUN_L2=1 python3 -m pytest tests/integration/test_pipeline.py -s

默认 skip：这套会真的起两个容器、colcon build 一次、跑 2 个试次，
单次 2~4 分钟，不适合放进日常回路。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "tests" / "integration" / "_artifacts"
RUN_DIR = ROOT / "ros2_ws" / "run"
LOG_DIR = ART / "logs"
DATA_DIR = ART / "datasets"

PY_ROBOT_SIM = os.environ.get(
    "ROBOT_SIM_PY",
    "/home/tongou.wei/Software/anaconda3/envs/robot_sim/bin/python",
)
FAKE_SIM = ROOT / "tests" / "integration" / "fake_sim_app.py"
DRIVER_IMAGE = "alicia/driver:1.0"
RECORDER_IMAGE = "alicia/recorder:1.0"
C_DRIVER = "alicia_l2_driver"
C_RECORDER = "alicia_l2_recorder"

# 独立 domain，避免和正在跑的真机/仿真串话
DOMAIN = "177"

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_L2") != "1",
    reason="L2 集成测试会起容器跑完整链路，设置 RUN_L2=1 才执行",
)

# ---------------------------------------------------------------- 测试用配置
# 用字符串写死而不是读 task.yaml 再改：这里要的是一份**不随生产配置漂移**的
# 最小契约，生产配置加字段时这个测试不该跟着变。
TASK_YAML = """\
name: cube_into_cup
version: 1
frame: world

object:
  cube: {{size: 0.025, mass: 0.020, friction: 0.8, color: [0.02, 0.02, 0.02]}}
  cup: {{top_diameter: 0.060, bottom_diameter: 0.040, inner_diameter: 0.056,
       height: 0.080, mass: 0.050, wall: 0.002, color: [0.85, 0.02, 0.30]}}

spawn:
  # 这组坐标是按 Alicia_D 的实际可达域反推的（基座系 -x 方向、z=0.18 时
  # 横向只能到 ±0.10）。原来的 cup (0.30,-0.26) 换算到基座系是侧向 0.26，
  # "移到杯口上方"必然 IK 无解——真机上这个杯子位置同样够不到。
  cube: {{center: [0.33, -0.16, 0.0125], yaw_deg: 0}}
  cup: {{center: [0.28, -0.02, 0.040], fixed: true}}
  randomization:
    stage0: {{cube_x: 0.0, cube_y: 0.0, cube_yaw_deg: 0}}

expert:
  count: 2
  radius_m: 0.030
  rounds_per_position: 1
  trial_timeout_s: 12

robot:
  dof: 6
  joints: [Joint1, Joint2, Joint3, Joint4, Joint5, Joint6]
  joint_units: rad
  gripper: {{open: 1.0, close: 0.0}}
  reset_pose: poses/reset.json
  action_space: joint_position

camera:
  # 与生产一致：1280x720 / 30Hz。先按真实负载验证跨容器链路，跑不动再降。
  front: {{name: cam_front, resolution: [1280, 720], fps: 30}}
  wrist: {{name: cam_wrist, resolution: [1280, 720], fps: 30}}

success:
  hold_s: 0.5
  settle_s: 0.1
  sim: []
  real: []

episode:
  timeout_s: 12
  fps: 30
  max_steps: 360
  reset_between: true

data:
  project_root: {root}
  root: {data}
  demos_dir: demos_l2
  demo_count: 2
  language: "把黑色方块放进杯子里"
"""

# 这张表不走 .format()，所以花括号保持单层
DETECTOR_YAML = """\
judge: ground_truth
episode: {settle_s: 0.1, settle_px: 2.0, hold_s: 0.5, timeout_s: 10.0}
cup:
  center_px: [640, 360]
  radius_px: 70.0
  margin_px: 5.0
  radius_m: 0.028
cube:
  ranges_lower: [[0, 120, 60], [170, 120, 60]]
  ranges_upper: [[10, 255, 255], [180, 255, 255]]
  min_area_px: 200
  settle_m: 0.002
gripper:
  open_min: 0.8
"""


# ---------------------------------------------------------------- 命令封装
def _docker(*args: str, timeout: float = 120.0) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=timeout
    )


def _ros2(*args: str, timeout: float = 20.0) -> subprocess.CompletedProcess:
    """在 driver 容器里执行 ros2 CLI。

    容器和宿主挂的是同一个绝对路径，所以 install/setup.bash 的路径两边一致。
    """
    inner = (
        "source /opt/ros/humble/setup.bash && "
        f"source {ROOT}/ros2_ws/install/setup.bash 2>/dev/null; "
        f"ros2 {' '.join(args)}"
    )
    return _docker("exec", C_DRIVER, "bash", "-lc", inner, timeout=timeout)


def _logs(name: str, tail: int = 40) -> str:
    out = _docker("logs", "--tail", str(tail), name, timeout=30.0)
    return (out.stdout or "") + (out.stderr or "")


def _container_running(name: str) -> bool:
    out = _docker("ps", "--filter", f"name={name}", "--format", "{{.Names}}")
    return name in (out.stdout or "")


def _read_status(topic: str) -> dict | None:
    """读一条 status；没数据/还没发布时返回 None。

    ros2 topic echo 在每条消息后面会多打一行 `---`，直接丢给 json.loads
    会解析失败——之前这里一直返回 None 就是这个原因，看起来像"节点没发
    状态"，其实是解析问题。
    """
    out = _ros2("topic", "echo", topic, "--once", "--field", "data", timeout=20.0)
    if out.returncode != 0:
        return None
    raw = (out.stdout or "").split("\n---")[0].strip().strip('"')
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _container_log_has(name: str, text: str) -> bool:
    out = _docker("logs", name, timeout=30.0)
    return text in ((out.stdout or "") + (out.stderr or ""))


def _wait_until(cond, timeout: float, interval: float = 1.0, label: str = ""):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            last = cond()
        except Exception as exc:  # noqa: BLE001 - 轮询期间任何异常都当"还没好"
            last = exc
        if last:
            return last
        time.sleep(interval)
    raise AssertionError(f"等待超时({timeout:.0f}s)：{label}，最后一次={last!r}")


# ---------------------------------------------------------------- 启停
def _preflight() -> None:
    if not Path(PY_ROBOT_SIM).exists():
        pytest.skip(f"找不到 robot_sim 解释器：{PY_ROBOT_SIM}")
    images = _docker("images", "--format", "{{.Repository}}:{{.Tag}}")
    have = set((images.stdout or "").split())
    missing = [i for i in (DRIVER_IMAGE, RECORDER_IMAGE) if i not in have]
    if missing:
        pytest.fail(f"镜像不存在：{missing}，先构建再跑 L2")


def _prepare() -> None:
    shutil.rmtree(ART, ignore_errors=True)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for sock in RUN_DIR.glob("*.sock"):
        sock.unlink(missing_ok=True)
    (ART / "task_l2.yaml").write_text(
        TASK_YAML.format(root=ROOT, data=DATA_DIR), encoding="utf-8"
    )
    (ART / "detector_l2.yaml").write_text(DETECTOR_YAML, encoding="utf-8")


def _latest_summary() -> dict:
    """orchestrator 每轮写一个 expert_run_<时间>.json，取最新那份。"""
    files = sorted(LOG_DIR.glob("*.json"))
    assert files, f"orchestrator 没写出 summary：{list(LOG_DIR.iterdir())}"
    return json.loads(files[-1].read_text())


def _start_fake():
    """假仿真跑在宿主：它只需要 numpy，不需要 ROS，也不该占容器名额。

    图像压到 2Hz 是故意的：1280x720 两路 30Hz 约 160MB/s，会把 DDS 打满，
    连 `ros2 topic echo --once` 这种 CLI 订阅都会因为发现超时而读不到
    status。链路正确性不缺那 28 帧，诊断能力更重要。
    """
    log = open(ART / "fake_sim.log", "w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            PY_ROBOT_SIM,
            str(FAKE_SIM),
            "--cmd",
            str(RUN_DIR / "sim_cmd.sock"),
            "--state",
            str(RUN_DIR / "sim_state.sock"),
            "--images-hz",
            "30",
            # 关节必须比图像快：两者同为 30Hz 时，图像到达时右邻 state 还没来，
            # try_frame 会等下一帧 state，而那时 images 已被新图覆盖，一半的帧
            # 就这样丢了。生产里 sim_app 主循环是 60Hz、真机 driver 也是 60Hz。
            "--joints-hz",
            "60",
            "--width",
            "1280",
            "--height",
            "720",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    _wait_until(
        lambda: (RUN_DIR / "sim_cmd.sock").exists()
        and (RUN_DIR / "sim_state.sock").exists(),
        timeout=20.0,
        interval=0.2,
        label="fake_sim_app 建 socket",
    )
    return proc


def _start_driver() -> None:
    """四个节点必须共享同一个 source 过的 shell。

    注意不能用 `A && B && C & D & E` 的写法：`&` 会把整条 `&&` 链和 D、E
    拆成互不相干的后台任务，`source install/setup.bash` 只作用于第一个，
    结果另外三个节点 import alicia_interfaces 直接 ModuleNotFoundError。
    这里把四个节点包进 `{ ... ; }`，source 在包外的同一个 shell 里执行。
    """
    cmd = (
        "source /opt/ros/humble/setup.bash && "
        f"cd {ROOT}/ros2_ws && "
        "{ [ -d install/alicia_interfaces/share/alicia_interfaces ] || "
        "colcon build --packages-select alicia_interfaces ; } && "
        f"source {ROOT}/ros2_ws/install/setup.bash && "
        f"cd {ROOT} && "
        "{ "
        f"python3 ros2_ws/src/orchestrator/orchestrator_node.py --ros-args "
        f"-p task_file:={ART}/task_l2.yaml -p out_dir:={LOG_DIR} "
        # 回灌要编码整集图像，耗时可能十几秒；默认等待太短会让下一轮被判
        # not_ready: recorder: not ready (flushing)
        f"-p ready_wait_s:=60.0 "
        f"> {LOG_DIR}/node_orchestrator.log 2>&1 & "
        f"python3 ros2_ws/src/sim_expert/sim_expert.py --ros-args "
        f"-p task_file:={ART}/task_l2.yaml "
        # 默认抬升 0.15m 会到 z=0.175、撤离再加 5cm 到 z=0.23，都超出可达域
        # （实测 z=0.18 时侧向只能到 ±0.10）。这里压到 8cm / 4cm
        f"-p grasp_lift_m:=0.08 -p release_m:=0.04 "
        f"> {LOG_DIR}/node_expert.log 2>&1 & "
        f"python3 ros2_ws/src/sim_bridge/sim_bridge.py --ros-args "
        f"-p sim_cmd_sock:={RUN_DIR}/sim_cmd.sock "
        f"-p sim_state_sock:={RUN_DIR}/sim_state.sock "
        f"> {LOG_DIR}/node_bridge.log 2>&1 & "
        f"python3 ros2_ws/src/success_detector/success_detector.py --ros-args "
        f"-p config_file:={ART}/detector_l2.yaml "
        f"> {LOG_DIR}/node_detector.log 2>&1 & "
        "wait ; }"
    )
    out = _docker(
        "run",
        "-d",
        "--name",
        C_DRIVER,
        "--network",
        "host",
        # 必须和 recorder 共享 IPC 命名空间：FastDDS 优先走共享内存，两个
        # 容器各自的 /dev/shm 互相看不见，表现是 discovery 正常（topic
        # info 能看到 publisher）但消息一条都收不到，orchestrator 会一直
        # 报 recorder: no_status。
        "--ipc",
        "host",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        f"ROS_DOMAIN_ID={DOMAIN}",
        "-e",
        # 默认传输装不下 2.7MB 的单帧图像，见 docker/fastdds/profile.xml
        f"FASTRTPS_DEFAULT_PROFILES_FILE={ROOT}/docker/fastdds/profile.xml",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-v",
        f"{ROOT}:{ROOT}",
        "-w",
        str(ROOT),
        DRIVER_IMAGE,
        "bash",
        "-lc",
        cmd,
    )
    if out.returncode != 0:
        pytest.fail(f"driver 容器启动失败：{out.stderr}")


def _start_recorder() -> None:
    cmd = (
        "source /opt/ros/humble/setup.bash && "
        f"cd {ROOT} && "
        f"python3 ros2_ws/src/recorder/recorder_node.py --ros-args "
        f"-p task_file:={ART}/task_l2.yaml"
    )
    out = _docker(
        "run",
        "-d",
        "--name",
        C_RECORDER,
        "--network",
        "host",
        # 必须和 driver 共享 IPC 命名空间：FastDDS 优先走共享内存，两个容器
        # 各自的 /dev/shm 互相看不见，表现为 discovery 正常但一条消息都收不到
        "--ipc",
        "host",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        f"ROS_DOMAIN_ID={DOMAIN}",
        "-e",
        # 用宿主挂载的这份，而不是镜像里 /opt 那份：测试不依赖"镜像是否重建过"，
        # 旧的 alicia/*:1.0 也能直接跑（profile 已打进 base 的新镜像同样接受覆盖）
        f"FASTRTPS_DEFAULT_PROFILES_FILE={ROOT}/docker/fastdds/profile.xml",
        "-e",
        "PYTHONUNBUFFERED=1",
        "-v",
        f"{ROOT}:{ROOT}",
        "-w",
        str(ROOT),
        RECORDER_IMAGE,
        "bash",
        "-lc",
        cmd,
    )
    if out.returncode != 0:
        pytest.fail(f"recorder 容器启动失败：{out.stderr}")


def _teardown(fake) -> None:
    # 容器一删日志就没了，先把最后一段落到 artifacts 里，失败时能直接看
    for name in (C_DRIVER, C_RECORDER):
        if _container_running(name):
            try:
                (ART / f"{name}.log").write_text(_logs(name, 200), encoding="utf-8")
            except Exception:  # noqa: BLE001 - 诊断信息拿不到不该盖住真正的失败
                pass
    # 只对 recorder 优雅停：它的 shutdown() 要等在途回灌跑完并调 finalize()，
    # 少了这步 parquet writer 不写 footer，数据集文件停在半截读不出来。
    # rm -f 是 SIGKILL，会把这整段收尾跳过。
    # driver 里的四个节点是 bash 后台任务，SIGTERM 未必能立刻收干净，
    # 让它照旧走 SIGKILL，别把 teardown 拖成两分钟。
    if _container_running(C_RECORDER):
        _docker("stop", "-t", "120", C_RECORDER, timeout=150.0)
    for name in (C_RECORDER, C_DRIVER):
        _docker("rm", "-f", name, timeout=60.0)
    if fake is not None and fake.poll() is None:
        fake.terminate()
        try:
            fake.wait(timeout=10)
        except subprocess.TimeoutExpired:
            fake.kill()


def _on_failure_dump() -> str:
    parts = []
    if (ART / "fake_sim.log").exists():
        parts.append("=== fake_sim ===\n" + (ART / "fake_sim.log").read_text()[-2000:])
    for name in ("orchestrator", "expert", "bridge", "detector"):
        path = LOG_DIR / f"node_{name}.log"
        if path.exists():
            parts.append(f"=== {name} ===\n" + path.read_text()[-2000:])
    for name in (C_DRIVER, C_RECORDER):
        if _container_running(name):
            parts.append(f"=== {name} ===\n" + _logs(name, 20))
    return "\n".join(parts)


# ---------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def stack():
    _preflight()
    _prepare()
    # _prepare 会清空 run 目录，必须排在建 socket 之前；顺序反了会把
    # fake_sim 刚建好的 socket 文件删掉，bridge 永远停在 waiting_sim。
    fake = _start_fake()
    try:
        _start_driver()
        _start_recorder()
        # sim / detector 用 status 里的 readiness；recorder 没有服务也没有
        # action，`ros2 topic hz` 每次都要重新做 DDS 发现、几秒内未必出数，
        # 直接看它启动时打的那行日志——recorder 一启动就是 idle，日志出现
        # 即代表可接单。
        _wait_until(
            lambda: (
                (_read_status("/sim/status") or {}).get("readiness")
                and (_read_status("/detector/status") or {}).get("readiness")
                and _container_log_has(C_RECORDER, "recorder 启动")
            ),
            timeout=300.0,
            interval=1.0,
            label="三个节点就绪",
        )
        yield
    except BaseException:
        print(_on_failure_dump())
        raise
    finally:
        _teardown(fake)


# ---------------------------------------------------------------- 用例
def test_start_accepted(stack):
    """orchestrator 能接下 start：说明四个节点都被它认可为就绪。"""
    out = _ros2("service", "call", "/orch/start", "std_srvs/srv/Trigger", timeout=30.0)
    assert out.returncode == 0, out.stderr
    assert "success=True" in out.stdout, (
        "start 被拒绝，通常是某个节点还没就绪\n" + out.stdout + "\n" + _on_failure_dump()
    )


def test_two_trials_collected(stack):
    """两个试次跑完，orchestrator 写出 summary，两条都有结论。"""

    def summary_ready():
        files = sorted(LOG_DIR.glob("*.json"))
        if not files:
            return None
        try:
            data = json.loads(files[-1].read_text())
        except json.JSONDecodeError:
            return None
        trials = data.get("trials") or []
        return data if len(trials) >= 2 else None

    data = _wait_until(
        summary_ready, timeout=180.0, interval=2.0, label="orchestrator summary"
    )
    trials = data["trials"]
    assert len(trials) == 2, f"期望 2 条试次，实际 {len(trials)}：{trials}"


def test_trials_actually_ran(stack):
    """两个试次必须真的跑过，而不是被 readiness 检查直接跳过。

    只断言"有两条 trial"是不够的：节点没就绪时 orchestrator 也会写两条
    reason=not_ready 的记录，测试照样通过，但链路其实一次都没跑。
    """
    data = _latest_summary()
    reasons = [t.get("reason", "") for t in data["trials"]]
    assert not any("not_ready" in r for r in reasons), (
        f"试次被跳过，链路没跑起来：{reasons}\n" + _on_failure_dump()
    )
    ok = sum(1 for t in data["trials"] if t["success"])
    assert ok == 2, f"假仿真下两轮都应成功，实际 {ok}/2：{data['trials']}"


def test_dataset_written(stack):
    """成功样本进主数据集：至少要有 meta 目录，且 staging 已清空。"""
    root = DATA_DIR / "demos_l2"
    assert root.exists(), f"数据集目录没建出来：{root}\n{_logs(C_RECORDER)}"
    meta = root / "meta"
    assert meta.exists() and any(meta.iterdir()), f"meta 为空：{list(meta.iterdir())}"

    # 最后一集的回灌是异步的，要给编码留时间；直接断言会撞上"还在写"
    staging = DATA_DIR / ".staging"
    _wait_until(
        lambda: not list(staging.glob("ep_*")) if staging.exists() else True,
        timeout=120.0,
        interval=2.0,
        label="staging 清空",
    )
