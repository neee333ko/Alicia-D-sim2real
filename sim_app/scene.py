#!/usr/bin/env python3
"""sim_app 的 Isaac Lab 场景：建场景、下发关节目标、回读状态与相机。

这个模块只能在 AppLauncher 起来之后 import——Isaac Sim 的 Python 包要先让
app 把环境和扩展初始化好，提前 import 会直接崩。

场景里不放任何本轮位姿。杯子和小方块的位置每一轮由 /sim/setup、/sim/place
传进来，这里只负责把值写到 prim 上。
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import yaml

# 仓库根：sim_app/ 的上一层。yaml 里的相对路径都以它为基准
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scene.yaml")


def load_config(path: str | None = None) -> dict:
    """读 scene.yaml，并把相对路径按仓库根解析成绝对路径。

    配置里的 assets/、poses/ 都是相对仓库根写的，换机器不用改；这里统一
    转绝对路径，sim_app 从任何 cwd 启动都能找到文件。只处理已知的路径字段，
    不做通用扫描——配置里出现新路径字段时要显式加进来。
    """
    cfg_path = path or DEFAULT_CONFIG
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    for key in ("usd", "home_pose_file"):
        value = cfg["robot"][key]
        if value and not os.path.isabs(value):
            cfg["robot"][key] = os.path.normpath(os.path.join(ROOT, value))
    return cfg


def set_prim_pose(prim_path: str, pos, quat_wxyz=None) -> None:
    """把世界坐标下的位姿直接写到 prim 的 xform 上。

    只用于「本来就该瞬移」的东西：静态桌面、杯子筒、腕部相机。刚体（方块）
    不能走这里——写 xform 不会同步到 PhysX，要用 write_root_pose_to_sim。

    要求 prim 是 /World 的直接子节点，此时 local 变换等于 world 变换；这个
    前提在这些 prim 都是自己建的时候成立。只在 prim 上补 translate/orient
    两个 op（不用 rotateXYZ），避免和已有旋转 op 叠加出意想不到的结果。
    """
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise ValueError(f"prim 不存在：{prim_path}")

    xf = UsdGeom.Xformable(prim)
    ops = {op.GetOpName(): op for op in xf.GetOrderedXformOps()}
    translate = ops.get("xformOp:translate") or xf.AddTranslateOp(
        UsdGeom.XformOp.PrecisionDouble
    )
    orient = ops.get("xformOp:orient") or xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)

    translate.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    if quat_wxyz is not None:
        w, x, y, z = (float(v) for v in quat_wxyz)
        orient.Set(Gf.Quatd(w, Gf.Vec3d(x, y, z)))


def create_simulation(cfg: dict):
    """按配置建物理世界，只返回 SimulationContext，不建任何资产。

    步长取 1/physics_hz：PD 位置控制在 60Hz 下比较稳，调高会更跟手但更吃
    GPU。这里不调 reset()——先把资产都 spawn 进 stage 再 reset，顺序反了
    模型会被清掉。
    """
    from isaaclab.sim import SimulationCfg, SimulationContext

    sim_cfg = cfg["sim"]
    dt = 1.0 / float(sim_cfg["physics_hz"])
    return SimulationContext(SimulationCfg(dt=dt, device=str(sim_cfg["device"])))


def create_environment(cfg: dict) -> None:
    """建静态环境和光源：地面、穹顶光、桌面。

    桌面顶面必须正好在 z=0——task.yaml 的 frame 就定义在台面上，方块中心
    z=0.015、杯子中心 z=0.045 全部以此为基准，桌子摆错整条坐标链都错。
    桌子只给碰撞体不给刚体，PhysX 会当成静态碰撞面，不会被机械臂推走。
    """
    import isaaclab.sim as sim_utils

    light = sim_utils.DomeLightCfg(intensity=600.0, color=(1.0, 1.0, 1.0))
    light.func("/World/DomeLight", light)

    # 深色地面：桌外区域别是纯白，相机里才好分辨边界
    ground = sim_utils.GroundPlaneCfg(color=(0.15, 0.15, 0.18))
    ground.func("/World/Ground", ground)

    table = cfg["table"]
    table_cfg = sim_utils.CuboidCfg(
        size=tuple(float(v) for v in table["size"]),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.32, 0.32, 0.35)),
    )
    table_cfg.func(table["prim"], table_cfg)
    set_prim_pose(table["prim"], table["pos"])


def create_robot(cfg: dict):
    """用 USD 建机械臂的 Articulation，PD 增益和夹爪关节都来自配置。

    两个 actuator 组分开配：6 个旋转关节用臂的增益，夹爪是两个 prismatic
    关节（行程只有 25mm），刚度必须高很多，否则夹不紧方块。
    init_state 只给基座位姿不给关节角——关节初始值用 USD 里的零位，回零
    由 place 负责，这里先摆角度反而会掩盖零位不对的问题。
    """
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import Articulation, ArticulationCfg

    robot = cfg["robot"]
    act = robot["actuators"]

    yaw = math.radians(float(robot["base_yaw_deg"]))
    base_quat = (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))

    robot_cfg = ArticulationCfg(
        prim_path=robot["prim"],
        spawn=sim_utils.UsdFileCfg(usd_path=robot["usd"]),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(float(v) for v in robot["base_pos"]),
            rot=base_quat,
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=list(robot["joints"]),
                stiffness=float(act["arm"]["stiffness"]),
                damping=float(act["arm"]["damping"]),
                effort_limit=float(act["arm"]["effort_limit"]),
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=[g["name"] for g in robot["gripper"]["joints"]],
                stiffness=float(act["gripper"]["stiffness"]),
                damping=float(act["gripper"]["damping"]),
                effort_limit=float(act["gripper"]["effort_limit"]),
            ),
        },
    )
    return Articulation(robot_cfg)


def create_cube(cfg: dict):
    """建方块刚体，尺寸/质量/摩擦全来自配置。

    初始位置只是 spawn 占位：第一次 /sim/place 就会用 write_root_pose_to_sim
    覆盖掉，所以不能把它当成「本轮位置」来读。z 取 edge/2 让它正好落在台面上，
    免得 reset 后先掉一段、第一帧图里还在半空。
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg

    cube = cfg["cube"]
    edge = float(cube["edge"])
    cube_cfg = RigidObjectCfg(
        prim_path=cube["prim"],
        spawn=sim_utils.CuboidCfg(
            size=(edge, edge, edge),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(cube["mass"])),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=float(cube["friction"]),
                dynamic_friction=float(cube["friction"]),
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(float(v) for v in cube["color"])
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.42, 0.0, edge / 2.0)),
    )
    return RigidObject(cube_cfg)


def create_cup(cfg: dict):
    """用一圈薄板拼出空心杯子，整杯是一个 kinematic 刚体，返回它的句柄。

    单个实心圆柱会让方块卡在杯口掉不进去，所以墙体用 wall_segments 段长方体
    沿圆周排开，段长取外接多边形的边长 2r·tan(π/N)，相邻两段在拐角处轻微
    重叠，内壁看上去是闭合的。

    杯子不该被方块推走或被机械臂撞退，但每轮 setup 都要把它摆到新位置。
    纯静态碰撞体在 PhysX 里没有「重设位姿」的接口，只写 USD xform 在 Fabric
    下物理侧也不一定跟随，所以根 prim 挂 kinematic 刚体：位置由我们写死，
    子碰撞体统一受它带动，方块撞上去它的位姿也不会变。

    局部坐标以杯体中心为原点、z 向上，墙体和杯底都相对它摆放；移动杯子只改
    根 prim 这一层，不用挨个搬子 prim。
    """
    import omni.usd
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObject, RigidObjectCfg
    from pxr import Gf, UsdGeom, UsdPhysics

    cup = cfg["cup"]
    root = cup["prim"]
    height = float(cup["height"])
    r_in = float(cup["inner_radius"])
    r_out = float(cup["outer_radius"])
    segments = int(cup["wall_segments"])
    bottom_t = float(cup["bottom_thickness"])
    color = tuple(float(v) for v in cup["color"])

    stage = omni.usd.get_context().get_stage()
    root_prim = UsdGeom.Xform.Define(stage, root).GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    root_prim.GetAttribute("physics:kinematicEnabled").Set(True)

    # 占位位置：先摆在台面空处，免得启动时和机械臂基座重叠、被 kinematic
    # 体顶开。第一次 setup 会用真实位置覆盖，别把它当成本轮位置读。
    set_prim_pose(root, (0.55, -0.15, height / 2.0))

    # 墙体：盒子局部 y 轴指向径向外侧、x 轴沿切向，中心落在中径上
    r_mid = 0.5 * (r_in + r_out)
    chord = 2.0 * r_mid * math.tan(math.pi / segments)
    wall_cfg = sim_utils.CuboidCfg(
        size=(chord, r_out - r_in, height),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )

    for i in range(segments):
        angle = 2.0 * math.pi * i / segments
        path = f"{root}/Wall_{i:02d}"
        wall_cfg.func(path, wall_cfg)

        # 绕 z 转 angle-90°，正好把局部 +y 转到径向外侧
        yaw = angle - math.pi / 2.0
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(r_mid * math.cos(angle), r_mid * math.sin(angle), 0.0)
        )
        xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Quatd(math.cos(yaw / 2.0), Gf.Vec3d(0.0, 0.0, math.sin(yaw / 2.0)))
        )

    # 杯底：半径取内径，正好托住方块，不插进墙体
    bottom_cfg = sim_utils.CylinderCfg(
        radius=r_in,
        height=bottom_t,
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
    )
    bottom_path = f"{root}/Bottom"
    bottom_cfg.func(bottom_path, bottom_cfg)
    UsdGeom.Xformable(stage.GetPrimAtPath(bottom_path)).AddTranslateOp(
        UsdGeom.XformOp.PrecisionDouble
    ).Set(Gf.Vec3d(0.0, 0.0, -height / 2.0 + bottom_t / 2.0))

    # spawn=None：prim 已经在上面建好了，这里只要一个能写物理位姿的句柄
    return RigidObject(RigidObjectCfg(prim_path=root, spawn=None))


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)) -> tuple[float, float, float, float]:
    """算从 eye 看向 target 的相机朝向四元数 (w, x, y, z)。

    Isaac 的相机约定是前方 = +X、右方 = +Y、上方 = +Z。于是构造一个右手基：
    X 取归一化视线，Y = up × X，Z = X × Y，把三列拼成旋转矩阵再转四元数。
    up 只决定滚转（画面不会歪），视线和 up 平行时叉积退化、四元数是 NaN，
    会让相机整帧出全黑，所以这种情况直接抛异常而不是硬算。
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)

    x = target - eye
    norm_x = np.linalg.norm(x)
    if norm_x < 1e-9:
        raise ValueError("eye 与 target 重合，朝向无法确定")
    x = x / norm_x

    y = np.cross(np.asarray(up, dtype=float), x)
    norm_y = np.linalg.norm(y)
    if norm_y < 1e-9:
        raise ValueError("视线与 up 平行，滚转无法确定")
    y = y / norm_y
    z = np.cross(x, y)

    # R 的三列就是相机三轴在世界系下的分量，按迹最大的分支转四元数
    r = np.stack([x, y, z], axis=1)
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s
    return (float(w), float(qx), float(qy), float(qz))


def create_cameras(cfg: dict):
    """建前视和腕部两路相机，返回 (front, wrist)。

    返回顺序就是 protocol 里 CAM_FRONT / CAM_WRIST 的下标，别调换。

    front 用世界坐标定死（eye 看 look_at），画面对齐台面，方便每轮对比；
    wrist 挂在机械臂末端 prim 下，pos/rot 是相对父 prim 的安装偏移，臂一动
    它跟着动。rot 按 opengl（USD 相机）约定书写：零偏时光轴是父节点的 -Z、
    上方 +Y，要改成"沿 +Z 看出去"就换成 convention="ros" 再写偏航。

    只取 rgb：深度和分割这里用不到，多开一个 buffer 只会拖慢渲染。视场角
    由 focal_length + aperture 决定，改分辨率不会改变画面范围。
    """
    import omni.usd
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg

    cams = cfg["camera"]

    front = cams["front"]
    front_cam = Camera(
        CameraCfg(
            prim_path="/World/Cam_Front",
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=float(front["focal_length"]),
                focus_distance=400.0,
                horizontal_aperture=float(front["horizontal_aperture"]),
                clipping_range=(0.01, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=tuple(float(v) for v in front["eye"]),
                rot=look_at_quat(front["eye"], front["look_at"]),
                convention="world",
            ),
            width=int(front["width"]),
            height=int(front["height"]),
            data_types=["rgb"],
        )
    )

    wrist = cams["wrist"]
    tool_path = f"{cfg['robot']['prim']}/{wrist['parent']}"
    stage = omni.usd.get_context().get_stage()
    if not stage.GetPrimAtPath(tool_path).IsValid():
        raise ValueError(f"腕部相机的父 prim 不存在：{tool_path}")

    # 固定轴 X-Y-Z：q = qz * qy * qx，展开成 (w, x, y, z)。
    # RPY 直接作用在 USD 相机坐标上（opengl 约定），下面不再做坐标系换算
    roll, pitch, yaw = (math.radians(float(v)) for v in wrist["offset_rpy_deg"])
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    wrist_rot = (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )

    wrist_cam = Camera(
        CameraCfg(
            prim_path=f"{tool_path}/Cam_Wrist",
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=float(wrist["focal_length"]),
                focus_distance=400.0,
                horizontal_aperture=float(wrist["horizontal_aperture"]),
                clipping_range=(0.01, 100.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=tuple(float(v) for v in wrist["offset_pos"]),
                rot=wrist_rot,
                convention="opengl",
            ),
            width=int(wrist["width"]),
            height=int(wrist["height"]),
            data_types=["rgb"],
        )
    )
    return front_cam, wrist_cam


class Scene:
    """仿真场景的门面：建一次，之后所有关节、物体、图像的读写都走它。"""

    def __init__(self, cfg: dict):
        """按固定顺序把场景搭起来，并把机械臂摆到复位姿态。

        顺序不能换：资产先全部 spawn 进 stage，再 sim.reset() 注册进 PhysX。
        reset 早了，后面 spawn 的模型进不了物理世界；资产建完却不 reset，
        刚体没有物理句柄，第一次 step 就会报错。

        reset 必须排在读 joint_names 之前：资产是听 timeline 的 PLAY 事件
        才初始化的（reset 会触发），在那之前 root_physx_view 还是 None，
        读关节名直接 AttributeError。相机同理由 reset 初始化，所以也要先建。

        复位角来自 reset.json，文件不存在就按全零启动并打一行日志——缺一个
        姿态文件不该让仿真起不来，但也不能悄悄用错的姿态。写进仿真后还要把
        PD 目标设成同一个值，否则下一步隐式执行器会把关节拽回零位。
        """
        import torch

        self.cfg = cfg
        self.sim = create_simulation(cfg)
        create_environment(cfg)
        self.robot = create_robot(cfg)
        self.cube = create_cube(cfg)
        self.cup = create_cup(cfg)
        self.cameras = create_cameras(cfg)
        self.sim.reset()

        # USD 里的关节顺序和配置顺序不保证一致，下发和回读一律按 USD 顺序
        self.joint_order = list(self.robot.joint_names)

        arm_joints = list(cfg["robot"]["joints"])
        home_file = cfg["robot"]["home_pose_file"]
        if os.path.isfile(home_file):
            with open(home_file, encoding="utf-8") as f:
                data = json.load(f)
            home_joints = [float(v) for v in data["joints"]]
            home_grip = max(0.0, min(1.0, float(data.get("gripper", 0.0))))
            if len(home_joints) != len(arm_joints):
                raise ValueError(
                    f"{home_file} 的 joints 有 {len(home_joints)} 个，"
                    f"配置要 {len(arm_joints)} 个"
                )
        else:
            print(f"[sim_app] 找不到复位姿态 {home_file}，按全零启动", flush=True)
            home_joints = [0.0] * len(arm_joints)
            home_grip = 0.0

        # 夹爪 0~1 线性映射到每个手指自己的行程（两指方向相反）
        targets = dict(zip(arm_joints, home_joints))
        for g in cfg["robot"]["gripper"]["joints"]:
            closed, opened = float(g["close"]), float(g["open"])
            targets[g["name"]] = closed + home_grip * (opened - closed)

        missing = [n for n in self.joint_order if n not in targets]
        if missing:
            raise ValueError(f"复位姿态里缺少关节：{missing}")

        self.home_q = [targets[n] for n in self.joint_order]

        q = torch.tensor([self.home_q], dtype=torch.float32, device=self.sim.device)
        self.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        self.robot.set_joint_position_target(q)

    def step(self) -> None:
        """推一个物理步：先下发关节目标，步进，再刷新资产 buffer。

        物理固定 60Hz、出图只有 30Hz，渲染交给 render_all 单独触发。这里若用
        默认的 render=True，每一步都会跑一遍渲染管线，GPU 时间白花一倍，相机
        帧的时间戳也会和关节帧错开。

        write_data_to_sim 必须在 step 之前：set_joint_position_target 只填
        内部 buffer，真正把目标写进 PhysX 的是它，漏了机械臂就不会动。

        update(dt) 必须在 step 之后：sim.step() 不会刷新资产数据，缺了它
        read_joint_state / read_object_truth 读到的永远是初始值，home_error
        也恒为 0，place 会立刻误判成功。
        """
        dt = self.sim.get_physics_dt()
        self.robot.write_data_to_sim()
        self.sim.step(render=False)
        self.robot.update(dt)
        self.cube.update(dt)
        self.cup.update(dt)

    def apply_joint_target(self, q7) -> None:
        """把一条 /joint_commands 的 7 维目标落到物理关节上。

        q7 的前 6 个是 Joint1..Joint6 的角度（弧度），第 7 个是夹爪 0~1 的
        归一化开合（0=闭合、1=张开）。夹爪要按配置里每根手指自己的行程分别
        换算，两个 prismatic 关节方向相反，不能用同一个符号。

        yaml 里的关节顺序不代表 USD 里的顺序，必须按 joint_order 重排；
        这里只写 PD 目标不瞬移关节——位置控制在后面的 step 里自己收敛。
        """
        import torch

        if len(q7) != 7:
            raise ValueError(f"关节目标需要 7 个数，实际 {len(q7)}")

        targets = dict(
            zip(self.cfg["robot"]["joints"], (float(v) for v in q7[:6]))
        )
        grip = max(0.0, min(1.0, float(q7[6])))
        for g in self.cfg["robot"]["gripper"]["joints"]:
            closed, opened = float(g["close"]), float(g["open"])
            targets[g["name"]] = closed + grip * (opened - closed)

        q = [targets[n] for n in self.joint_order]
        self.robot.set_joint_position_target(
            torch.tensor([q], dtype=torch.float32, device=self.sim.device)
        )

    def command_home(self) -> None:
        """把 6 个臂关节的 PD 目标设回复位角，夹爪保持当前位置。

        只在 place 的收尾阶段调用。不用 write_joint_state_to_sim 瞬移：
        瞬移会让画面跳变，还可能在夹爪仍抓着东西时把方块弹飞；交给 PD 慢慢
        收敛更接近真机，失败也能用超时暴露出来。

        夹爪不设目标：复位文件里的夹爪开合只是启动姿态，回零途中夹爪该开
        还是关由外部策略决定，这里强行改会抢走夹爪的控制权。

        目标按 joint_order 逐个拼：home_q 本来就是按这个顺序存的，关节顺序
        和 yaml 不一致也不会串位。
        """
        arm = set(self.cfg["robot"]["joints"])
        q = self.robot.data.joint_pos[0].clone()
        for i, name in enumerate(self.joint_order):
            if name in arm:
                q[i] = self.home_q[i]
        self.robot.set_joint_position_target(q.unsqueeze(0))

    def home_error(self) -> float:
        """返回 6 个臂关节当前角度与复位角的最大绝对偏差（弧度）。

        place 的到位判据用它：只看臂关节，夹爪不参加——夹爪这时可能还夹着
        东西，位置本来就不该等于复位值，算进来判定永远不会通过。取最大偏差
        而不是平均：任何一个关节没到位，末端位姿就已经偏了。
        """
        pos = self.robot.data.joint_pos[0]
        arm = set(self.cfg["robot"]["joints"])
        return max(
            abs(float(pos[i]) - self.home_q[i])
            for i, name in enumerate(self.joint_order)
            if name in arm
        )

    def read_joint_state(self) -> list[float]:
        """回读 7 维关节状态：Joint1..Joint6 的角度 + 夹爪 0~1 的归一化开合。

        顺序和 /joint_commands 一致，bridge 拿到就能直接填 JointState。物理侧
        的关节顺序和 yaml 顺序可能不同，所以先按名字建表再按配置顺序取值。

        夹爪读配置里的第一根手指再反归一化：两指由同一条指令驱动，正常时位置
        是对称的，读哪根都一样；某根被挡住时这个值也只是不精确，不影响判断。
        """
        pos = self.robot.data.joint_pos[0].detach().cpu().tolist()
        by_name = dict(zip(self.joint_order, pos))

        q = [float(by_name[n]) for n in self.cfg["robot"]["joints"]]
        g = self.cfg["robot"]["gripper"]["joints"][0]
        closed, opened = float(g["close"]), float(g["open"])
        grip = (by_name[g["name"]] - closed) / (opened - closed)
        q.append(max(0.0, min(1.0, grip)))
        return q

    def set_cube_pose(self, xyz) -> None:
        """把方块瞬移到指定位置，姿态摆正、速度清零。

        刚体不能像静态物那样写 xform：PhysX 有自己的位姿 buffer，只改 USD 会
        让画面和物理分叉。这里走 write_root_pose_to_sim 直接改物理位姿。

        速度必须一起清零——上一轮掉进杯子的残余速度会带进新一轮，方块刚放下
        就会滑走。传入的 z 是方块几何中心，传台面高度时要自己加 edge/2。
        """
        import torch

        if len(xyz) != 3:
            raise ValueError(f"方块位置需要 xyz 三个数，实际 {len(xyz)}")

        device = self.sim.device
        pos = torch.tensor([[float(v) for v in xyz]], dtype=torch.float32, device=device)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
        self.cube.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))
        self.cube.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=torch.float32, device=device))

    def set_cup_pose(self, xyz_center) -> None:
        """把杯子摆到指定位置，杯口保持朝上。

        杯子是 kinematic 刚体，走 write_root_pose_to_sim 改物理位姿；写 USD
        xform 在 Fabric 下物理侧可能不跟随。传入的 z 是杯体中心，不是杯口，
        和 setup 命令里 cup_z 的语义一致。

        姿态固定用单位四元数：杯子永远竖直放置，不该继承上一轮的任何旋转。
        速度不用清——kinematic 体不会自己动。
        """
        import torch

        if len(xyz_center) != 3:
            raise ValueError(f"杯子位置需要 xyz 三个数，实际 {len(xyz_center)}")

        device = self.sim.device
        pos = torch.tensor(
            [[float(v) for v in xyz_center]], dtype=torch.float32, device=device
        )
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device)
        self.cup.write_root_pose_to_sim(torch.cat([pos, quat], dim=-1))

    def read_object_truth(self):
        """回读杯子和小方块的真实位姿，组装成 type=4 帧要的两组数。

        返回 (cup_xyz_height, cube_xyz_edge)：
        - cup 的 z 必须是**杯口**高度。物理位姿给的是杯体中心，要加 height/2；
          detector 的 ground_truth 判据直接拿这个高度判断方块有没有落进杯里。
        - cube 的 z 是几何中心，最后一个数跟边长。

        读的是 root_pos_w（物理真值），不是下发的指令也不是 spawn 时的占位值：
        type=4 的意义就是「现在实际在哪」，拿别的值下游会做出错误判断。
        """
        cup = self.cup.data.root_pos_w[0].detach().cpu().tolist()
        cube = self.cube.data.root_pos_w[0].detach().cpu().tolist()

        cup_height = float(self.cfg["cup"]["height"])
        cube_edge = float(self.cfg["cube"]["edge"])

        cup_truth = (
            float(cup[0]),
            float(cup[1]),
            float(cup[2]) + cup_height / 2.0,
            cup_height,
        )
        cube_truth = (float(cube[0]), float(cube[1]), float(cube[2]), cube_edge)
        return cup_truth, cube_truth

    def render_all(self) -> list:
        """渲染一次，取出两路相机的当前帧 RGB，返回 [front, wrist]。

        必须自己先触发 sim.render()：Scene.step() 关掉了自动渲染，相机 buffer
        只有在渲染管线跑过之后才是当前帧，直接读会拿到上一帧。

        渲染完还要逐台 update(dt)：Camera.data 只在 _is_outdated 为真时才去
        读 buffer，而 _is_outdated 正是 update() 置位的，漏了它第一帧之后
        画面就永久冻结。dt 用物理步长，和 SensorBase 内部的时间戳基准一致。

        顺手拷成 numpy：相机 buffer 是复用的显存，下一帧就被覆盖，交给发送
        线程的数组必须已经离开 GPU。返回顺序和 cameras 一致，即
        CAM_FRONT / CAM_WRIST。
        """
        self.sim.render()
        dt = self.sim.get_physics_dt()

        frames = []
        for cam in self.cameras:
            cam.update(dt)
            rgb = cam.data.output["rgb"][0, ..., :3]
            frames.append(rgb.detach().cpu().numpy().astype(np.uint8))
        return frames
