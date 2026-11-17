#!/usr/bin/env python3
"""从 URDF 直接算零位，画一张侧视线条图，标出每段的倾角。

为什么不走 Isaac Sim：URDF 导入那条路在这台机器上生成的是空 stage（1.5KB 的
usd，没有 prim），而且 GPU 物理在多人共用的卡上会炸。这里只用到 URDF 里的
运动学（origin + axis），数学和仿真里 FK 是同一套，结论等价。

    python sim_test/plot_zero_pose.py
    → logs/zero_pose_schematic.png
"""

import math
import os
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 系统里有 Noto CJK，不设的话中文全是方块
# 这台机器上 matplotlib 找不到可用的中文字体（只有 Noto Serif CJK 的 ttc，没被索引），
# 图里一律用英文，免得标题变成方块
import numpy as np

URDF = (
    "/home/tongou.wei/Software/anaconda3/envs/robot_sim/lib/python3.11/"
    "site-packages/synriard/urdf/Alicia_D_v5_6/Alicia_D_v5_6_gripper_50mm.urdf"
)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "zero_pose_schematic.png")


def rpy2R(rpy):
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    Ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def main():
    root = ET.parse(URDF).getroot()
    child2joint = {j.find("child").get("link"): j for j in root.findall("joint")}

    # base_link → tool0 的关节链
    chain, cur = [], "tool0"
    while cur in child2joint:
        j = child2joint[cur]
        chain.append(j)
        cur = j.find("parent").get("link")
    chain.reverse()

    T = np.eye(4)
    pts = [T[:3, 3].copy()]
    names = ["base"]
    axes = []
    seg_lines = []

    for j in chain:
        o = j.find("origin")
        xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
        T0 = np.eye(4)
        T0[:3, :3] = rpy2R(rpy)
        T0[:3, 3] = xyz
        T = T @ T0                      # 零位：所有关节角 = 0，旋转部分不变
        pts.append(T[:3, 3].copy())
        names.append(j.find("child").get("link"))

        axis = j.find("axis")
        if axis is not None:
            a = np.array([float(v) for v in axis.get("xyz").split()])
            n = np.linalg.norm(a)
            if n < 1e-9:
                continue                      # 夹爪是 prismatic，axis 可能全 0
            a = a / n
            axes.append((j.get("name"), T[:3, :3] @ a))
        seg_lines.append((j.get("name"), pts[-2], pts[-1]))

    pts = np.array(pts)

    # ---- 画侧视图（X-Z 平面）------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 7), dpi=130)
    ax.plot(pts[:, 0], pts[:, 2], "-o", lw=4, ms=8, color="#1f77b4", zorder=3)

    for i, (name, a, b) in enumerate(seg_lines):
        mid = (a + b) / 2
        v = b - a
        ang = math.degrees(math.atan2(v[2], v[0]))
        ax.annotate(
            f"{name}\n{ang:+.1f}°",
            xy=(mid[0], mid[2]),
            xytext=(0, 14 if i % 2 == 0 else -20),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color="#333333",
        )

    for i, (x, z) in enumerate(pts[:, [0, 2]]):
        ax.plot(x, z, "o", ms=6, color="#d62728", zorder=4)
        ax.annotate(names[i], (x, z), textcoords="offset points", xytext=(8, -12), fontsize=7, color="#666666")

    ax.axhline(0, color="#aaaaaa", lw=1, ls="--")
    ax.axvline(0, color="#aaaaaa", lw=1, ls="--")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.set_title("Alicia-D URDF zero pose (side view, X-Z). Label = segment angle vs horizontal")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.abspath(OUT))
    print("WROTE", os.path.abspath(OUT))

    # ---- 文字结果 ------------------------------------------------------
    print("\n零位末端 tool0 (x, y, z) =", np.round(T[:3, 3], 4).tolist())
    print("零位工具轴朝向 =", np.round(T[:3, 2], 3).tolist())
    print("\n各段相对水平面倾角：")
    for name, a, b in seg_lines:
        v = b - a
        print(f"  {name:8s} {math.degrees(math.atan2(v[2], v[0])):+7.1f}°")
    print("\n各关节轴在世界坐标下的方向（零位）：")
    for name, ax_ in axes:
        print(f"  {name:8s} {np.round(ax_, 2).tolist()}")


if __name__ == "__main__":
    main()
