#!/usr/bin/env python3
"""【WIP — 当前跑出来是空白图，别用】

把 Alicia-D 的 URDF 导入 Isaac Sim，摆到零位，从侧面渲染一张图存下来。

卡住的地方：URDF 导入命令没有把 prim 写进当前 stage（生成的 usd 只有 1.5KB、
没有 /World），补光/相机都正常但画面全白；换 UrdfFileCfg spawn 之后 GPU 物理
又炸（PhysX error 700，多人共用的卡上跑不动）。
目前可用的替代方案是 plot_zero_pose.py（从 URDF 直接算 FK 画侧视图）。

原始需求：

用途：真机的"零位看起来是歪的"到底是不是模型本来的样子。URDF 导入后关节默认
就是 0，所以不需要设物理、不需要设目标位置，直接渲染即可。

    python render_zero_pose.py            # 默认输出 logs/zero_pose_side.png

相机放在 -Y 方向看向 +Y，看到的是 X-Z 平面（机械臂的侧面）。
同时出一条正视图（从 +X 往回看），方便对比。
"""

import math
import os
import sys

import numpy as np

from isaaclab.app import AppLauncher

URDF = (
    "/home/tongou.wei/Software/anaconda3/envs/robot_sim/lib/python3.11/"
    "site-packages/synriard/urdf/Alicia_D_v5_6/Alicia_D_v5_6_gripper_50mm.urdf"
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")

launcher = AppLauncher(headless=True, enable_cameras=True)
simulation_app = launcher.app

import omni.kit.app  # noqa: E402
import omni.kit.commands  # noqa: E402
import omni.usd  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import Camera, CameraCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)):
    """世界坐标下：前方 = +X，上方 = +Z 的相机四元数 (w, x, y, z)。"""
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    f = target - eye
    f /= np.linalg.norm(f)
    up = np.asarray(up, dtype=float)
    y = np.cross(up, f)          # 相机 Y = Z × X
    y /= np.linalg.norm(y)
    z = np.cross(f, y)           # 相机 Z（上方）
    R = np.stack([f, y, z], axis=1)

    # 旋转矩阵 → 四元数
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        yy = (R[0, 2] - R[2, 0]) / s
        zz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        yy = (R[0, 1] + R[1, 0]) / s
        zz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        yy = 0.25 * s
        zz = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        yy = (R[1, 2] + R[2, 1]) / s
        zz = 0.25 * s
    return (w, x, yy, zz)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 导入 URDF -----------------------------------------------------
    if not os.path.isfile(URDF):
        raise FileNotFoundError(URDF)

    # 导入命令由这个扩展提供，默认没加载，先开
    ext_mgr = omni.kit.app.get_app().get_extension_manager()
    if not ext_mgr.is_extension_enabled("isaacsim.asset.importer.urdf"):
        ext_mgr.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
    _, import_cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    import_cfg.fix_base = True
    import_cfg.merge_fixed_joints = False
    import_cfg.make_default_prim = True
    import_cfg.self_collision = False
    import_cfg.fix_base = True

    # 先建仿真再导入：SimulationContext 会重建 stage，先导入的话模型会被清掉
    sim = SimulationContext(SimulationCfg(dt=1.0 / 60.0, device="cpu"))
    sim.reset()

    omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=URDF,
        dest_path=os.path.join(OUT_DIR, "alicia_d_zero.usd"),
        import_config=import_cfg,
    )
    stage = omni.usd.get_context().get_stage()
    paths = [str(p.GetPath()) for p in stage.Traverse()]
    print("N_PRIMS", len(paths), flush=True)
    print("WORLD", [x for x in paths if x.startswith("/World")][:12], flush=True)

    # 用包围盒自动构图，避免相机对着空气
    from pxr import Usd, UsdGeom
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    world = stage.GetPrimAtPath("/World")
    bbox = cache.ComputeWorldBound(world).ComputeAlignedRange()
    lo, hi = np.array(bbox.GetMin()), np.array(bbox.GetMax())
    center = (lo + hi) / 2.0
    size = float(np.max(hi - lo))
    print("BBOX", np.round(lo, 3).tolist(), np.round(hi, 3).tolist(), "size", round(size, 3), flush=True)

    # 补光，否则头less 渲染出来是黑的
    # 补光调暗一点，纯白模型配强光会糊成一片白
    light = sim_utils.DomeLightCfg(intensity=600.0, color=(1.0, 1.0, 1.0))
    light.func("/World/DomeLight", light)
    # 深色地面做背景，白色机械臂才看得见
    ground = sim_utils.GroundPlaneCfg(color=(0.15, 0.15, 0.18))
    ground.func("/World/Ground", ground)

    # ---- 两路相机 ------------------------------------------------------
    d = max(size * 1.8, 0.8)
    views = {
        "side": ((center[0], center[1] - d, center[2] + d * 0.15), tuple(center)),
        "front": ((center[0] + d, center[1], center[2] + d * 0.15), tuple(center)),
    }

    for name, (eye, target) in views.items():
        cam = Camera(
            CameraCfg(
                prim_path=f"/World/Cam_{name}",
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=24.0,
                    focus_distance=400.0,
                    horizontal_aperture=36.0,
                    clipping_range=(0.01, 100.0),
                ),
                offset=CameraCfg.OffsetCfg(
                    pos=eye, rot=look_at_quat(eye, target), convention="world"
                ),
                width=960,
                height=720,
                data_types=["rgb"],
            )
        )
        if not cam.is_initialized:
            cam._initialize_impl()
        for _ in range(8):          # 让 RTX 把这一帧渲染完
            sim.step()
            simulation_app.update()

        rgb = cam.data.output["rgb"][0]
        arr = rgb[..., :3].detach().cpu().numpy().astype(np.uint8)
        path = os.path.join(OUT_DIR, f"zero_pose_{name}.png")
        from PIL import Image

        Image.fromarray(arr).save(path)
        print(f"WROTE {os.path.abspath(path)}  shape={arr.shape}  mean={arr.mean():.1f}", flush=True)

    simulation_app.close()


if __name__ == "__main__":
    main()
