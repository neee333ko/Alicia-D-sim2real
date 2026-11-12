# Alicia-D 多任务 Sim2Real 工程方案（ROS 2 版）v2

> 目标：从单任务实机 ACT，演进到「多任务 / 多物体 / 仿真预训练 / 真机闭环」的工程化系统。
> 前置：实机链路已跑通（真机 + 相机 + SDK + LeRobot 数据集 + ACT 训练，在 2080 那台）；仿真环境已装好（Isaac Sim 5.1 + Isaac Lab 2.3.2，在 3090 那台）。
> 总周期：一人全职约 4~5 个月（含 ROS 2 改造）。
> 更新：2026-11-12

---

## 0. 本版相对 v1 的变更

| 项 | v1 写的 | v2 实际 | 原因 |
|---|---|---|---|
| 机器 | 单机（2080 Ti） | **两台**：2080 挂真机，3090 跑仿真 | 机械臂在 2080，Isaac Sim 装在 3090 |
| ROS 2 装法 | RoboStack 或 Docker | **apt 装系统级 Humble** | 22.04 原生支持；不在 docker 组；Docker 也不解决显示 |
| 仿真选型 | MuJoCo 起步 → ManiSkill3 | **Isaac Lab 2.3.2 直接用** | 已装好并验证能起；MuJoCo 只做过 URDF 加载 |
| 任务 | 笔 + 容器 | **35mm 木方块 + 260ml 锥形塑料杯** | 方块在仿真里是原生 primitive，不用建模；橙色好做分割 |
| 阶段 0 | 先定 3 个任务 | **先定 1 个** | 单任务跑通再谈扩展 |
| 阶段 2 对照 | 各 20 次 | **各 50 次，配对同一批初始位姿** | 20 次分不出 15 个百分点的差别 |
| 首要投入 | 渲染真实性 | **评测无人化** | 数据端能无限堆，评测端是人的时间锁死的 |
| 存储路径 | `~/datasets/` | **全部在 `pi/` 下** | 数据和工程文件只放一处 |
| 判定方式 | 未定 | **只用视觉，真机仿真同一套** | 仿真读真值会和真机不可比 |
| 时序 | 写在 policy 里 | **拆出 `action_executor`** | 换模型不影响执行策略 |
| 频率 | 未定 | **全链路 30Hz** | 相机/关节/动作必须同一个 fps |

---

## 1. 总体目标与范围

| 维度 | 当前 | 目标 |
|---|---|---|
| 任务数 | 1（方块放进杯子） | 3~5 个 |
| 物体数 | 2 | 10 个 |
| 数据来源 | 纯真机遥操作 | 仿真合成 + 真机微调 |
| 策略 | ACT（单任务） | SmolVLA（语言条件） |
| 评测 | 人工判定 | 自动判定 + 无人值守 |
| 架构 | 单 Python 脚本 | ROS 2 多节点 |

**不在范围内**：强化学习训练、双臂协同、移动底盘。

---

## 2. 架构总览

### 2.1 两台机器的分工

```
2080（机械臂 + 两个相机）                          3090（6×RTX 3090，无显示器）
├── driver_node       串口 → SDK                  ├── sim_app    Isaac Lab（py3.11）
├── camera_node       图像 + 内参                 │              ↕ unix socket
├── action_executor   30Hz 派发 + 请求 chunk      └── sim_bridge ROS 话题（py3.10，容器）
├── policy_node       ACT / SmolVLA 推理
├── success_detector  成功判定
├── recorder_node     写 LeRobot 数据集
└── eval_orchestrator 评测编排
```

`sim_app` 和 `sim_bridge` 是两个进程：Isaac Sim 只支持 py3.11，ROS Humble 是 py3.10，同进程 import 不了。

`driver_node` 和 `sim_bridge` **二选一**，对外接口完全相同。

### 2.2 关键设计

- `policy_node` 只依赖 `/joint_states` 与 `/camera/*/image_raw`，不感知底层是真是假
- `driver_node` 是**唯一**允许 `import` SDK 的地方。任何其他节点里出现 SDK 调用，一进仿真就废
- `recorder_node` 独立订阅，驱动/相机异常不影响已录制数据
- `success_detector` 自动判定，`eval_orchestrator` 编排整轮评测，无需人工
- `action_executor` 是唯一按 30Hz 发 `/joint_commands` 的节点；`policy_node` 只产 chunk，不管时序
- 判定只用视觉，真机仿真同一套；仿真真值只做离线对拍

### 2.3 设计原则

| 原则 | 理由 |
|---|---|
| 控制接口统一为关节位置 | 与实机 API 一致，避免 sim2real 多一层 IK 误差 |
| sim/real 暴露相同接口 | 策略、录制、评测节点在两种模式下完全复用 |
| 驱动层隔离硬件 | 换机械臂只换 `driver_node` |
| 录制与推理解耦 | 相机/驱动崩溃不影响数据落盘 |
| 策略模型可换 | 换 ACT→SmolVLA 只换 adapter + ckpt，数据不重采 |

---

## 3. ROS 2 环境

### 3.1 选型

apt 装系统级 **Humble**（Ubuntu 22.04 jammy，Python 3.10），装在 `/opt/ros/humble`。

| 方式 | 结论 |
|---|---|
| **apt 系统装（主选）** | 二进制包，几分钟装完；rviz2 齐全；管理员一条 `apt install` |
| RoboStack | py3.11 社区重建，无官方兜底，不用 |
| Docker | 当前账号不在 docker 组；且容器不解决"没显示器" |

```bash
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions python3-rosdep
```

**注意**：`ros-humble-desktop` 不是最小的。`ros-base` 没有 rviz2。

### 3.2 和 conda 的关系

`robot_sim`（py3.11 + Isaac）和 ROS（py3.10）**永远不共用一个环境**。ROS 的 `setup.bash` 会改 `PYTHONPATH` / `LD_LIBRARY_PATH`，在 `robot_sim` 里 source 会污染解释器。

两边靠进程边界通信：unix socket / TCP / 共享内存，传 ndarray 和 JSON，不传对象。

### 3.3 跨机器通信（v1 没写，必踩）

`driver_node` 在 2080，`sim_bridge` 在 3090，两台不在同一网段时 DDS 自动发现会失败。

- 同网段：组播可用，什么都不用配，但要统一 `ROS_DOMAIN_ID` 和 `RMW_IMPLEMENTATION`
- 不同网段：配 `ROS_DISCOVERY_SERVER`（Fast DDS）或 CycloneDDS 的静态 peers 列表
- 第一阶段其实用不到跨机器：`driver + camera + recorder + policy` 全在 2080

---

## 4. 接口定义

这套接口是全局契约。真机模式和仿真模式必须逐字一致，否则后面每加一个节点都要改两套代码。

**频率约定：全链路 30Hz。** 相机采图 30fps、`/joint_states` 30Hz、`/joint_commands` 30Hz、训练数据集 30fps、chunk 里每一步代表 1/30 秒。这五个数必须一致，改一个就要一起改——错位不会报错，只会让策略的整个时间轴偏掉（比如按 50Hz 采、按 30Hz 训，动作会慢 1.67 倍）。相机能不能跑 30Hz 要先实测（`v4l2-ctl --list-formats-ext`），USB 带宽不够时 YUYV 常常只有 30fps 而 MJPG 能到 60。

### 4.1 节点

| 节点 | 跑在哪 | 镜像 | 状态 |
|---|---|---|---|
| `driver_node` | 2080 | `alicia/driver:1.0` | ✅ 已实现。dry_run 验证了限位/限速/NaN/看门狗；真机未接 |
| `camera_node` | 2080 | `alicia/camera:1.0` | ✅ 用官方 `v4l2_camera` + launch + 参数，没写自研节点；未在真机跑 |
| `action_executor` | 2080 | `alicia/base:1.0` | ✅ 已实现，假机械臂验证过 30Hz 派发和重连 |
| `policy_node` | 2080 | `alicia/policy:1.0` | ✅ 已实现；没有真 checkpoint，没跑通过一次完整推理 |
| `success_detector` | 2080 | `alicia/base:1.0` | ✅ 已实现，合成图像验证过成功/超时路径 |
| `sim_bridge` | 3090 | `alicia/base:1.0` | ✅ 已实现，假 sim_app 验证过协议和断线重连 |
| `recorder_node` | 2080 | — | ⬜ 未实现 |
| `orchestrator`（旧称 `eval_orchestrator`） | 2080 / 3090 | `alicia/base:1.0` | ✅ 编排、服务、状态检查、abort 已实现并在容器内测通；下游节点尚未对接（见 5.12） |
| `sim_app` | 3090 | 宿主 conda `robot_sim`（py3.11） | ⬜ 未实现 |
| `sim_scene` | 3090 | — | ⬜ 未实现 |
| `robot_state_publisher` | 2080 | 现成 ROS 包 | ⬜ 未接入 |

**仿真侧为什么要拆两个进程**：Isaac Sim 5.1 只跑在 Python 3.11，ROS 2 Humble 是 3.10，同进程 import 不了。所以 `sim_app`（无 ROS）+ `sim_bridge`（无 Isaac）两个进程，中间走本机 unix socket。协议写在 `ros2_ws/src/sim_bridge/sim_bridge.py` 文件头，`sim_app` 照着实现即可。

`driver_node` 与 `sim_bridge` 二选一，对外接口逐字一致。

### 4.2 话题

| 话题 | 类型 | 发布者 | 订阅者 |
|---|---|---|---|
| `/joint_states` | `sensor_msgs/JointState`（6 关节 + 夹爪） | driver_node / sim_bridge | policy、executor、detector、recorder |
| `/joint_commands` | `std_msgs/Float64MultiArray`（7 维） | **action_executor** | driver_node / sim_bridge |
| `/policy/infer_request` | `std_msgs/Empty`（reliable, depth 1） | action_executor | policy_node |
| `/policy/action_chunk` | `std_msgs/Float64MultiArray`（`[stamp, n, 7×n]`，reliable + volatile + depth 1） | policy_node | action_executor |
| `/exec/status` | `std_msgs/String`（JSON，5Hz） | action_executor | 人 / recorder（可选） |
| `/camera/front/image_raw` | `sensor_msgs/Image`（rgb8，best_effort） | camera_node / sim_bridge | policy、detector、recorder |
| `/camera/front/camera_info` | `sensor_msgs/CameraInfo` | 同上 | policy、recorder |
| `/camera/wrist/image_raw` | `sensor_msgs/Image` | 同上 | policy、recorder |
| `/camera/wrist/camera_info` | `sensor_msgs/CameraInfo` | 同上 | policy、recorder |
| `/detector/mask` | `sensor_msgs/Image`（mono8，有人订阅才发） | success_detector | 人（调阈值） |
| `/task/instruction` | `std_msgs/String`（transient_local） | eval_orchestrator | policy、detector |
| `/orch/trial` | `std_msgs/String`（JSON：id + event，transient_local） | eval_orchestrator | recorder、detector |
| `/object/pose` | `geometry_msgs/PoseStamped`（仿真真值） | sim_bridge | detector（只对拍，不判定） |
| `/detector/condition` | `std_msgs/Bool`（实时状态，变化才发） | success_detector | 人（调试） |
| `/orch/result` | `std_msgs/String`（JSON，每次试次一条，带 id；**唯一结论出口**） | success_detector | orchestrator、recorder |
| `/tf`、`/tf_static` | `tf2_msgs/TFMessage` | robot_state_publisher / sim_bridge | rviz |
| `/rosout` | `rcl_interfaces/Log` | 所有节点 | 自动发布 |

### 4.3 服务

| 服务 | 类型 | 提供方 | 状态 |
|---|---|---|---|
| `/arm/home` | `std_srvs/Trigger` | driver_node / sim_bridge | ✅ 开始动就返回，不等到位 |
| `/exec/start` / `/exec/stop` | `std_srvs/Trigger` | action_executor | ✅ |
| `/record/start` / `/record/stop` | — | recorder_node | ❌ 已删除，改由 `/orch/trial` 的 begin/end 驱动 |
| `/orch/start` / `/orch/stop` / `/orch/abort` | `std_srvs/Trigger` | orchestrator | ✅ 已实现 |
| `/sim/setup` | 自定义 `Setup.srv`（cup_x/cup_y/cup_z） | sim_bridge | ⬜ 待实现 |
| `/sim/place` | 自定义 `Place.srv`（x/y/z） | sim_bridge | ⬜ 待实现 |
| `/sim/reset` | `std_srvs/Trigger` | sim_bridge | ⚠️ 已有，语义要改成“到位才返回 success” |
| `/expert/run_once` | action `RunOnce` | sim_expert | ❌ 现为 Trigger 服务，待改成 action |

### 4.4 约定

- **7 维** = 6 关节弧度 + 夹爪。夹爪在 ROS 侧统一 0~1，映射到 SDK 角度在 `driver_node` 内部完成。数据集、策略输出、驱动三处必须一致，中间用 0~100 就全错且不报错。
- **chunk 编码**（临时，没有 msg 包）：`data[0]` = 观测时刻（ROS 时钟秒），`data[1]` = chunk 步数 n，`data[2:]` = n×7 行优先。以后建 `ActionChunk.msg` 换掉。
- **`success_detector` 有两种判定模式**（配置 `judge`）：`vision` 读图像（真机用），`ground_truth` 读 `/object/pose` 三维坐标（只能仿真用）。
  - `ground_truth` 是唯一能真正区分"在杯里"和"悬在杯口上方"的判据——单目做不到。适合仿真数据生成和仿真内评测。
  - **跨环境比较时必须都用 `vision`**：A/B 对照里如果仿真用真值、真机用视觉，两个成功率不是同一个量，差异会被误读成 sim2real 差距。
  - 每条 `/orch/result` 都带 `judge` 字段，事后能分清哪个数是用哪种判据测的。
- **判定相关的两个话题分工**：`/task/instruction` 决定"判什么"（任务级），`/orch/trial` 决定"什么时候判"（试次级）。detector 两个都订阅。
- **结论只有一个出口**：`/orch/result`。不再提供布尔版的 `/eval/success`——两个话题表达同一件事，迟早有人给它们统计出两个数。实时状态在 `/detector/condition`（一个试次里可能变很多次），不要拿它统计成功率。
- **`/arm/home` 是"开始动"不是"已到位"**。调用方要自己轮询 `/joint_states` 等它到位。
- **时间戳一律用 ROS 时钟**（`get_clock().now()`），不用 `time.time()`。chunk 的 stamp 来自相机/仿真的 `header.stamp`，`action_executor` 也用同一个源比较。混用会让 chunk 全被判过期，表现是"策略在推理但机械臂不动"。
- **所有消息必须填 `header.stamp`**（采集时刻，不是发布时刻）。
- **观测对齐**：三路各缓存 5 帧，以最新图像为锚，关节状态取离锚点最近的一帧，相差 ≤ 一帧（33ms）即同一时刻；找不到就跳过，由 executor 在 150ms 后重发。
- **给 `uint8[]` 字段赋值必须用 `array.array("B", data)`，不能直接赋 `bytes`**。rclpy 会对 `bytes` 做逐元素校验，921KB 从 3000+ fps 掉到 6.6 fps，而且不报错。
- **图像 QoS 用 best_effort**：丢一帧不影响策略，reliable 会在丢包时把相机节点堵住。
- **`/task/instruction`、`/orch/trial` 用 transient_local**，后启动的节点也能拿到最后一条。发布方必须用同样的 durability，否则 QoS 不兼容、根本匹配不上。

### 4.5 关键参数

```yaml
driver_node:
  rate_hz: 30                 # 必须等于训练 fps
  max_speed_rad_s: 3.0        # ≈172°/s，安全上限不是调速器
  cmd_timeout_s: 0.2          # 看门狗
  joint_limits_deg: "-157.5,157.5,-90,90,-44.7,180,-159.9,159.9,-90,90,-180,180"
  reset_pose_file: poses/reset.json

action_executor:
  control_hz: 30              # 必须等于 driver 的 rate_hz 和训练 fps
  request_lead_s: 0.2         # 下一段 chunk 提前多久要
  request_timeout_s: 0.15     # 请求没回应多久重发
  chunk_max_age_s: 0.3        # 过期 chunk 直接丢
  hold_timeout_s: 1.0         # 队列空了保持多久

policy_node:
  policy_path: ".../checkpoints/act_v1"
  policy_type: act            # act | smolvla | pi0
  device: cuda
  align_tol_s: 0.0333         # 三路观测对齐容差，一帧
  buffer_len: 5
  max_obs_age_s: 0.25

success_detector:
  config_file: /calib/detector.yaml   # 所有视觉阈值都在那
```

### 4.6 QoS

| 话题 | reliability | durability | depth |
|---|---|---|---|
| 图像 / 掩码 | best_effort | volatile | 5 |
| `/joint_states`、`/joint_commands`、`/object/pose` | reliable | volatile | 5~10 |
| `/policy/action_chunk` | reliable | volatile | 1 |
| `/policy/infer_request` | reliable | volatile | 1 |
| `/task/instruction`、`/orch/trial` | reliable | **transient_local** | 1 |

---

## 5. 编排器（orchestrator）架构与设计

代码：`ros2_ws/src/orchestrator/orchestrator_node.py`（节点名 `orchestrator`，旧称 `eval_orchestrator`）。

### 5.1 职责边界

| 归它管 | 不归它管 |
|---|---|
| 试次的顺序与边界（何时放、何时复位、何时开窗） | 录制落盘（recorder 订阅 `/orch/trial` 自己做） |
| 调用 `sim.*` 服务并判返回值 | 成功判定（detector 订阅 `/orch/trial` 自己做） |
| 驱动专家 action、超时、cancel | 视觉、图像、模型推理 |
| 每个试次前检查下游存活与就绪 | 进程守护（缺节点就退出，不负责重启） |
| 汇总统计并写 JSON | 数据集的格式与版本 |

一条明确的分界线：**编排方只发边界事件、只等“就绪”信号，不靠固定 sleep 猜下游进度。**

### 5.2 结构

```
        /orch/start /orch/stop /orch/abort  (std_srvs/Trigger)
                          │
      ┌───────────────────▼────────────────────┐
      │              orchestrator              │
      │  主线程：run() → run_trial() 循环       │
      │  后台：MultiThreadedExecutor spin       │
      └──┬─────────┬────────────┬──────────┬───┘
         │ 服务     │ action      │ 事件发布  │ 订阅
   /sim/setup   /expert/run_once  /orch/trial  /orch/result
   /sim/place                    begin/end    /{node}/status ×4
   /sim/reset
```

为什么主线程跑循环、rclpy 在后台 spin：编排逻辑可以用阻塞式写法（顺序调用、等待条件），而回调仍能被并发处理，不会自己把自己锁死。节点启动后**不自动开跑**，保持存活等服务调用。

### 5.3 三种通信方式的分工

| 用途 | 机制 | 例子 | 语义 |
|---|---|---|---|
| 执行确认（要同步知道结果） | 服务 | `/sim/setup`、`/sim/place`、`/sim/reset` | `success=true` 表示**动作已完成**，不是“命令已接收” |
| 长动作（要进度、要能取消） | action | `/expert/run_once` | goal 接受 ≠ 完成；可 cancel、可 feedback |
| 一对多事件（广播边界） | 话题 | `/orch/trial` | begin/end，订阅者各自决定怎么做 |
| 持续状态（可晚到、可重发） | 话题 | `/{node}/status`、`/orch/result` | 只缺最新一条就有意义，用 transient_local |

**为什么事件不用服务**：`/orch/trial` 的消费者有 recorder 和 detector 两个，且编排方不需要等它们“做完”（recorder 收尾耗时不确定）。用服务会把 1 对 N 变 1 对 N 串行调用，还把录制耗时耦合进试次时序。

**为什么状态不用服务**：发布者本来就不知道订阅者的状态；改成服务只是把“猜”挪到调用方，还多一次往返。正确做法是让下游**自己上报状态**，编排方订阅后判断，见 5.8。

### 5.4 一次试次的时序

```
run_trial(idx, x, y)
 0. wait_ready(ready_wait_s)   四路 status 全部 liveness+readiness+未过期，否则跳过
 1. /sim/place  {x, y, cube_z} 失败 → reason=place_failed，直接返回（还没开窗，无脏数据）
 2. /sim/reset                 失败 → reason=reset_failed，直接返回（不在错误姿态上开跑）
 3. 置 current_trial=idx → publish /orch/trial {id, "begin"}
 4. /expert/run_once  action   失败/超时 → reason=expert_failed
 5. result_event.wait(trial_timeout)  收到同 id 的 /orch/result → success/reason
 6. publish /orch/trial {id, "end"}
```

几个关键顺序：

- **先置 `current_trial` 再发 begin**。detector 可能回得很快，id 对不上会被 `on_result` 丢掉。
- **第 0 步放在最前面**。此时还没有任何副作用，跳过试次不会留下半截 episode，也不会把异常数据算进成功率。
- **第 1、2 步在 begin 之前**。放物体和复位是“准备”，不属于试次；失败时不需要 recorder/detector 开窗再关窗。
- **专家失败仍然发 end**。窗口必须成对关闭，否则 recorder 永远停在 recording、下一个试次 `wait_ready` 全部失败。
- **等待判定用 `result_event`，不用轮询 `last_result`**，abort 时直接 `set()` 唤醒，不必等满 `trial_timeout`。

失败原因（写进汇总 JSON 的 `reason`）：

| reason | 含义 |
|---|---|
| `not_ready: <明细>` | 下游有节点失联/失活/未就绪，本试次跳过 |
| `place_failed` | 物体没摆好 |
| `reset_failed` | 机械臂没回到复位位 |
| `expert_failed` | 专家没跑成功（含被 cancel、超时） |
| `aborted` | 收到 `/orch/abort` |
| `no_result` | 专家成功但 `trial_timeout` 内没有判定结论 |

### 5.5 控制状态机

| 请求 | 条件 | 行为 |
|---|---|---|
| `/orch/start` | 未运行 | `running=true`、清 abort/stop、起 `run` 线程，立即返回 `started` |
| `/orch/start` | 正在运行 | 拒绝，提示先 stop/abort（避免两个线程驱动同一批服务） |
| `/orch/stop` | 正在运行 | 只置 `stop_requested`，**跑完当前试次**再退出；汇总照常写 |
| `/orch/abort` | 正在运行 | 置 `abort` + `result_event.set()` 唤醒等待线程，当前试次作废 |
| Ctrl-C | — | 等价 abort：置位、唤醒、join 编排线程 5s，退出码 130 |

`running` 在 `run()` 的 `finally` 里复位——正常跑完、被中止、抛异常都要放开，否则 `/orch/start` 再也起不来。

`/orch/stop` 与 `/orch/abort` 的区别是**数据完整性**：stop 让当前试次正常收尾（录制文件完整），abort 放弃当前试次（可能留下半截数据）。前者用于“这批够了”，后者用于“出事了”。

### 5.6 超时设计

原则：**客户端超时 > 服务端内部超时**，让服务端先自己放弃并返回一个有语义的失败，而不是让客户端先放弃、留下“状态未知”。

| 位置 | 超时 | 说明 |
|---|---|---|
| `check_nodes` 逐个探测 | 2s / 个 | 服务与 action 两套发现机制都要探 |
| `/sim/setup` | 5s | 一轮一次 |
| `/sim/place` | 5s | 每试次一次 |
| `/sim/reset` | 8s | 需要真运动，比摆位宽 |
| action 发送 goal | 5s | 服务端没起来或满载 |
| action 等 result | 60s | 脚本专家的完整抓取+放置 |
| 等判定 | `trial_timeout_s`（task.yaml，默认 20s） | 试次级，独立于专家超时 |
| 状态过期 | `status_timeout_s`（默认 3s） | 超过就算失联 |
| 等就绪 | `ready_wait_s`（默认 3s） | 有界轮询，50ms 一次 |
| cancel 后等确认 | 2s | 不无限等，见 5.9 |

**超时/abort 之后的接口语义**：ROS 服务无法取消，`call` 返回 False 只代表“我不等了”，服务端可能还在动，调用方必须按“状态未知”处理。action 可以发 cancel，但那是协作式的，对方不响应就拦不住——所以 cancel 只等 2s，然后继续走收尾流程。

### 5.7 为什么删掉 recorder 的 start/stop 服务

改为 recorder 订阅 `/orch/trial` 自己开关窗口。

- **省掉两条调用链**：`/orch/trial` 已经是试次边界的事实来源，再加两个服务等于同一件事有两个真相。
- **失败分支归一**：录制失败不再由 orchestrator 逐个判返回值并 abort 整轮，而是体现在 recorder 的 status 上，由下一个试次的 `wait_ready` 接住。
- **代价**：orchestrator 不能同步知道“这一段录成功了”。这是刻意的取舍——录制属于旁路，不应该卡住主流程；但必须靠 status 保证“没准备好就别开下一个试次”。

由此 recorder 承担两条义务：`readiness=true` 时必须能立刻接住下一条 begin；收到 begin 立刻切 `state="recording"`，收到 end 回到 `idle` 并 flush。

### 5.8 下游状态契约

```
/{sim|expert|recorder|detector}/status   std_msgs/String, JSON
{"node": "recorder", "liveness": true, "readiness": true,
 "state": "idle", "stamp": 1699999999.0, "detail": ""}
```

- QoS：reliable + transient_local + keep_last 1。后启动的 orchestrator 也能立刻拿到最新一条。
- 发布时机：状态变化时立刻发一次，其余周期发（建议 1Hz）。
- `node` 字段**以话题名为准**，不信消息体里的名字——发布方写错不该让状态串到别人身上；JSON 解析失败不覆盖上一条。
- 三种故障分开报，不都归成“没状态”：

| 检查 | 判据 | 含义 |
|---|---|---|
| 失联 | `now - stamp > status_timeout_s` | 节点挂了、卡死或网络断了 |
| 失活 | `liveness=false` | 进程在但内部子系统坏了（相机打不开、串口断了） |
| 未就绪 | `readiness=false` | 活着但腾不出手（recorder 还在写盘、expert 还在上一次动作里） |

### 5.9 abort 为什么必须两端配合取消

`/orch/abort` 对专家走 action cancel。实测（容器内集成测试）发现：只发 cancel 而不做下面两件事，取消会**静默失效**，goal 一路跑到 SUCCEEDED。

1. **ActionServer 必须传 `cancel_callback` 并返回 `CancelResponse.ACCEPT`**。rclpy 默认是 `default_cancel_callback`，返回 `CANCEL_REJECTED`，客户端看到的是 `goals_canceling: []`，且没有任何报错。
2. **execute 回调必须周期性检查 `goal_handle.is_cancel_requested`**，否则请求被接受也不会真的停下来。

两条缺一条，表现都是“点了 abort，机械臂继续做完”。另外 cancel 与 feedback 有竞态：cancel 先到时 `publish_feedback` 会抛“发布器已失效”，服务端要包异常并复查 `is_cancel_requested`。

### 5.10 试次位置与参数

`count` 个位置铺在 `radius_m` 为半径的 10×10 网格上，每个位置重复 `rounds_per_position` 次：

```
x = base_x + (i % 10 - 5) * radius / 5.0
y = base_y + (i // 10 % 10 - 5) * radius / 5.0
```

即 `i=0..99` 覆盖以 `base` 为中心、边长 `2*radius` 的方形区域；`i%10` 是列、`i//10` 是行，`-5` 让中心对齐 `base`。`z` 取 task.yaml 里方块质心高度。

| 参数 | 默认 | 说明 |
|---|---|---|
| `mode` | `expert` | 目前只支持 expert，其他值启动即退出 |
| `task_file` | `/ws/task.yaml` | 不存在则启动失败 |
| `dry_run` | false | 服务缺失时用空操作代替，用于跑通时序 |
| `out_dir` | `/ws/logs` | 汇总 JSON 输出目录 |
| `status_timeout_s` | 3.0 | 状态过期阈值 |
| `ready_wait_s` | 3.0 | 每试次等就绪的上限 |

汇总输出 `expert_run_<timestamp>.json`：`mode/task/started/finished/count/rounds_per_position/success/aborted/trials[]`，每条 trial 含 `id/x/y/success/reason/duration_s`，有结论时附 detector 原始字段。

### 5.11 运行

```bash
python3 orchestrator_node.py --ros-args -p mode:=expert -p task_file:=/ws/task.yaml -p dry_run:=true

ros2 service call /orch/start std_srvs/srv/Trigger
ros2 service call /orch/stop  std_srvs/srv/Trigger   # 跑完当前试次
ros2 service call /orch/abort std_srvs/srv/Trigger   # 立即中断
```

### 5.12 测试与当前缺口

容器内（真 ROS，独立 `ROS_DOMAIN_ID`）共 45 个用例：25 个纯逻辑（状态检查、结果过滤、服务状态机，绕过 `__init__` 注入字段）+ 20 个集成（起真 `Orchestrator` 节点 + 假下游，覆盖服务成功/失败/超时、abort 真 cancel、begin/end 顺序、复位失败、未就绪跳过、汇总输出）。

```bash
pytest tests -q     # 需 source /ws/install/setup.bash
```

尚未对接的部分：

| 项 | 现状 |
|---|---|
| `/expert/run_once` | `sim_expert` 仍是 Trigger 服务，与 action 客户端类型不匹配 → 改为 action server（含 5.9 的两条取消要求） |
| `/sim/setup`、`/sim/place` | `sim_bridge` 未实现；socket 协议需要加 `CMD_SETUP` / `CMD_PLACE` / `PLACE_DONE` |
| `/sim/reset` | 已有服务，但当前“开始动就返回”，要改成“到位才 success” |
| `/{node}/status` | 四个下游都还没发布 |
| `run_trial` 里的 `time.sleep(1.0)` | 临时硬等，等 `/sim/reset` 语义修正后删掉 |
| `recorder_node` | 未实现，需要 idle→recording→idle 状态机 + status |

---

## 6. 分阶段实施

### 阶段 0：定义标准（已完成大半）

单任务：**方块放进杯子**（`pi/sim_test/task.yaml`）。

| 项目 | 状态 |
|---|---|
| 任务 | 方块放进杯子 |
| 物体 | 35mm 木方块（红色/黑色）、260ml 锥形塑料杯（橙色） |
| 动作空间 | 6 关节弧度 + 夹爪 0~1 |
| 观测 | front + wrist 640×480，位置固定，整个数据集期间不许动 |
| 成功判据 | 方块质心进杯口圆 + 停稳 0.5s + 保持 2s；仿真读真值，真机读画面 |
| 评测集 | 3 个变体 × 10 次 |

**待补**：物体到货后实测 5 个数（杯口内径、杯底内径、内深、杯质量、方块边长+质量），填进 task.yaml。

**验收**：成功判据能被 `success_detector` 直接实现，无需人工。

---

### 阶段 1：仿真环境（2~3 周）

#### 现状

Isaac Sim 5.1 + Isaac Lab 2.3.2 + torch 2.7.0+cu128 已装在 3090 的 `robot_sim`（py3.11，19GB）。能起 headless，`app ready` 约 34 秒。

环境变量必带：

```bash
export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
export CUDA_VISIBLE_DEVICES=<剩余显存最多的卡>
```

#### 要做的

```
1. Alicia-D URDF → USD，加 50mm 夹爪；关节零位/限位/速度对齐实机
2. 场景：桌子 + 方块（Box primitive）+ 杯子（外柱减内柱，参数化）
3. 两路相机：位姿与内参对齐真机
4. 脚本演示生成器：IK 抓取 → 抬升 → 移到杯口 → 松手
5. sim_bridge 节点
```

#### 脚本生成器（不是遥操作）

```
随机摆位 → 脚本专家执行 → 自动判定 → 存 (图像, 关节角, 动作) → 复位 → 下一条
```

一晚上出几千条，不需要人。注意：脚本专家的动作太"干净"，要注入扰动（时序随机、轨迹抖动、目标点偏移几毫米），否则策略学到的是机械动作。

#### 验收门槛

- 仿真画面与真机构图接近（同视角、同物体尺度）
- 脚本演示抓取成功率 > 90%

> 第 2 条不达标说明接触参数有问题，禁止进入下一阶段。

---

### 阶段 2：Sim2Real 基线

#### 实验设计（v1 的 20 次改掉）

```
B（先做）：仅真机 40 条训练 → 真机测 50 次     ← 基线，不是对照
A：      仅仿真 200 条训练 → 真机测 50 次
C（加）：40 真机 + 200 仿真混合训练 → 真机测 50 次
```

三组必须用**同一批初始位姿**（配对比较），否则连噪声都算不上。

| 结果 | 结论 |
|---|---|
| A ≈ B | 仿真可用，继续扩规模 |
| A < B − 20% | 存在 sim2real 差距，进阶段 3 |
| A ≈ 0 | 观测/动作对齐有严重问题，回阶段 1 |
| C > B | 混合训练立刻可上，不必等纯仿真追平 |

**B 组必须先跑**：它的作用不是对照，是证明整条闭环能稳定产出成功率数字。B 组跑不出数字，A 组的结果没有意义。

50 次是下限。20 次时 12/20 和 9/20 的差别（p≈0.34）在统计上不可分辨。

#### 评测自动化（与阶段 2 同步，优先级最高）

`eval_orchestrator` 状态机：

```
选场景 → /eval/reset → 策略运行 → success_detector 判定
      → 录制落盘 → 自动记录标签 → 下一轮
```

**产出**：`eval_<timestamp>/results.json`，无人介入。

---

### 阶段 3：缩小 Sim2Real 差距（3~4 周）

| 差距来源 | 对策 | 优先级 |
|---|---|---|
| 视觉 | 域随机化：光照、纹理、相机位姿抖动、背景照片替换 | 高 |
| 接触 | 随机化摩擦、质量、夹爪刚度 | 高 |
| 延迟 | 仿真注入 30~50 ms 控制延迟 | 中 |
| 标定 | 真机图像标定相机外参，仿真精确复现 | 中 |
| 零位 | 用实测零位修正 URDF | 中 |

**关键技巧**：把真机实拍图作为仿真背景纹理（texture baking），视觉差距显著下降。

**验收**：纯仿真训练的策略在真机达到 B 的 70% 以上。

---

### 阶段 4：多物体（2 周）

```
物体 2 → 10 个
每物体随机 500 个初始位姿
生成 5000~10000 条仿真演示
```

**验收**：seen ≥ 70%，unseen ≥ 40%。

---

### 阶段 5：多任务（3~4 周）

| 模型 | 说明 |
|---|---|
| **SmolVLA**（450M） | 语言条件，3090 可训，主选 |
| π0 / π0.5（3B 级） | 等到数据过万条或需要跨本体再考虑 |
| ACT + task one-hot | 快但扩展性差，只作过渡 |

数据配比：仿真 8000 + 真机 300，采样 sim : real = 4 : 1。

**切换信号**：单任务成功率稳定 85% 以上、要加第二第三个任务、或 ACT 明显吃不下 1000 条以上数据。在那之前换模型对成功率没有贡献。

**验收**：给定语言指令能激活正确任务；单任务成功率不低于阶段 4。

---

### 阶段 6：闭环飞轮（长期）

```
仿真大规模预训练 → 真机微调 → 真机评测 → 失败挖掘
      ↑                                    ↓
      └──────── 补仿真场景 ←── 人工纠正 ─────┘
```

| 环节 | 现有组件 | 目标形态 |
|---|---|---|
| 评测 | `eval.sh` | `eval_orchestrator` 节点 |
| 失败记录 | `failure_logs/` | 自动分类 + 标签 |
| 纠正 | `corrective_demos_v1/` | 接管录制节点 |
| 训练 | `lerobot-train` | `retrain.sh` + 版本管理 |

**目标指标**：

| 指标 | 目标 |
|---|---|
| 单轮周期 | < 24 小时 |
| 每轮评测次数 | ≥ 200 |
| 每轮人工投入 | < 1 小时 |
| 成功率提升 | 每轮 +3~5% |

**飞轮的瓶颈是评测，不是数据。** 一晚生 5000 条仿真数据很容易；判断这 5000 条训出来的策略好不好，还得真机跑 50 次。数据端可以无限堆，评测端被人的时间锁死。

---

## 7. 目录规划

所有数据和工程文件都在 `/home/tongou.wei/workspace/src/pi`。

```
pi/
├── task.yaml                   # 任务定义（物体尺寸/位置/成功判据/训练参数）
├── calib/                      # 标定与阈值，都是数据，运行时只读挂载
│   ├── front.yaml / wrist.yaml # 相机内参
│   └── detector.yaml           # 成功判定的所有阈值
├── poses/
│   └── reset.json.example      # /arm/home 用的复位姿态模板
├── udev/
│   └── 99-alicia-cameras.rules # 把 /dev/video* 固定成 cam_front / cam_wrist
├── docker/
│   ├── base/Dockerfile         # alicia/base：Humble + numpy + v4l2_camera
│   ├── driver/Dockerfile       # + SDK 依赖（scipy/omegaconf/synriard）
│   ├── camera/Dockerfile       # + v4l-utils
│   └── policy/Dockerfile       # FROM base + torch + lerobot
├── ros2_ws/src/
│   ├── alicia_d_driver/driver_node.py
│   ├── alicia_d_cameras/{launch/bringup.launch.py, config/cameras.yaml}
│   ├── action_executor/action_executor.py
│   ├── act_policy_node/policy_node.py
│   ├── success_detector/success_detector.py
│   ├── sim_bridge/sim_bridge.py
│   ├── orchestrator/orchestrator_node.py   # 编排中心（见第 5 节）
│   ├── alicia_interfaces/{srv/Setup.srv, srv/Place.srv, action/RunOnce.action}
│   └── recorder_node/          # ⬜ 未建
├── scripts/                    # ⬜ 待建：record/train/eval 的入口
├── sim_test/                   # 仿真环境验证脚本
│   ├── smoke_isaaclab.py
│   ├── livestream_test.py
│   ├── start_livestream.sh
│   ├── plot_zero_pose.py       # 从 URDF 算 FK 画侧视图
│   └── render_zero_pose.py     # WIP，Isaac Sim 渲染（当前跑出空白图）
├── docs/
│   └── ros2_sim2real_plan.md
├── tools/
│   └── proxy.env
├── datasets/                   # ⬜ 待采
└── checkpoints/                # ⬜ 待训
```

**代码只有一份**，在 3090 的 pi 里。2080 那边挂 NFS 或每次 rsync，不要两边各改一份 —— 版本漂移是这类项目最常见的隐性故障。

---

## 8. 实现状态与下一步

### 已完成

| 组件 | 验证到什么程度 |
|---|---|
| `driver_node` | dry_run：限位、速度限幅、NaN 拦截、看门狗、`/arm/home` 流程 |
| `action_executor` | 假机械臂：30Hz 派发、按时间请求 chunk、时间戳对齐、断线重连 |
| `policy_node` | 未跑通（没有真 checkpoint）；接口和观测对齐逻辑写完 |
| `success_detector` | 合成图像：成功路径 2.1s 出结果，超时/手动结束路径 |
| `sim_bridge` | 假 sim_app：协议往返、关节 30Hz、图像吞吐、断线重连 |
| 相机启动 | launch + 参数 + udev + 标定模板，未接真机 |
| `orchestrator` | 容器内真 ROS：45 个用例（纯逻辑 + 假下游集成），覆盖服务失败/超时、abort 真 cancel、begin/end 顺序、未就绪跳过、汇总输出 |
| 镜像 | base / driver / camera / policy 四个都构建通过 |

### 已知未做（按优先级）

1. **真机上跑通 camera + driver**：设备名固定、内参标定、串口确认。这是所有真机数据的入口。
2. **`recorder_node`**：写 LeRobot 数据集。没有它就没有训练数据。
3. **下游对接编排契约**（详见 5.12）：`sim_expert` 改 action server、`sim_bridge` 实现 `/sim/setup` `/sim/place` 与到位才返回的 `/sim/reset`、四个节点发布 `/{node}/status`。
4. **`sim_app` + `sim_scene`**：Isaac Lab 场景、关节映射、脚本专家生成数据。
5. **`pi/policy` 镜像重建**：Dockerfile 已改成 `FROM alicia/base`，但现有镜像是旧的。
6. **`ActionChunk.msg` 消息包**：现在 chunk 挤在 `Float64MultiArray` 里。

---

## 9. 时间估算

| 阶段 | 一人全职 |
|---|---|
| 0 定义 | 1 周（大半已完成） |
| ROS 2 基础设施（前 4 步） | 2 周 |
| 1 仿真环境 | 2~3 周 |
| 2 sim2real 基线 | 2 周 |
| 3 缩小差距 | 3~4 周 |
| 4 多物体 | 2 周 |
| 5 多任务 | 3~4 周 |
| **合计** | **约 4~5 个月** |

业余投入按 2 倍估算。

---

## 10. 风险清单

| # | 风险 | 影响 | 缓解 |
|---|---|---|---|
| 1 | 两台机器分离，数据/权重靠搬 | 每轮多一步手工，容易版本漂移 | 代码只一份 + NFS；先测 rsync 实际速度 |
| 2 | 相机掉线（ENODEV） | 整批数据/评测作废 | 换稳定 UVC，采集前跑十分钟压测 |
| 3 | 3090 的卡被别人占满 / 掉卡 | 仿真起不来（纹理创建失败） | 启动脚本自动挑剩余显存最多的卡；掉卡查 `dmesg \| grep -i xid` |
| 4 | inotify 配额 65536 不够 | Kit 启动刷几千条报错 | `fs.inotify.max_user_watches=524288`（需 root） |
| 5 | 动作空间不一致 | sim2real 多一层误差 | 统一绝对关节角，夹爪在 driver 内部换算 |
| 6 | 仿真评测被当真机结论 | 误判进展 | 两个数字分开汇报 |
| 7 | 仿真数据动作太"干净" | 策略呆板，真机迟钝 | 生成时注入时序/轨迹扰动 |
| 8 | 数据版本失控 | 无法复现 | manifest + 固定命名 |
| 9 | 渲染真实性不足 | 视觉策略失效 | texture baking + 域随机化 |
| 10 | websockets 12.0/13.1 冲突 | 只有 livestream 受影响 | 用 livestream 时降到 12.0 |

**三条硬性约束**：

1. **评测数字分开报**：仿真成功率和真机成功率永远不在同一张图里混着说
2. **动作空间必须与实机 API 完全一致**
3. **仿真评测不能替代真机评测**

---

## 11. 下一步

代码侧优先级见第 8 节。真机上按这个顺序落地：

```
1. 相机：udev 固定设备名 → 标定内参 → 640×480@30 跑十分钟不断
2. 机械臂：确认串口、填 poses/reset.json、dry_run 走一遍再切真机
3. 标定杯口像素坐标，填 calib/detector.yaml，用 /detector/mask 调 HSV
4. 采 40 条示范（recorder_node）→ 训 ACT → 评测 20 次，拿到第一个可信成功率
5. 并行：3090 上写 sim_app，先只出图像和状态，和真机构图比对
```

物体到货后先量 5 个数填 task.yaml。
