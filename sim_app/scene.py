#!/usr/bin/env python3
"""sim_app 的 Isaac Lab 场景：建场景、下发关节目标、回读状态与相机。

这个模块只能在 AppLauncher 起来之后 import——Isaac Sim 的 Python 包要先让
app 把环境和扩展初始化好，提前 import 会直接崩。

场景里不放任何本轮位姿。杯子和小方块的位置每一轮由 /sim/setup、/sim/place
传进来，这里只负责把值写到 prim 上。
"""

from __future__ import annotations

import math
import os

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
