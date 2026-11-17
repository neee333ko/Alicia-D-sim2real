#!/usr/bin/env python3
"""sim_app 侧的 unix socket 协议编解码。

契约的唯一事实源是 ros2_ws/src/sim_bridge/sim_bridge.py 的文件头注释。
这里是它的镜像实现：改任何一边都必须同步改另一边，两个进程靠这 64 字节
和 24 字节头对齐，格式错了不会报错，只会让下游读到垃圾。

本模块不 import 任何 Isaac / ROS，纯 Python，便于单独核对。
"""

from __future__ import annotations

import struct

import numpy as np

# ---- cmd 通道：bridge -> sim_app，定长 64 字节，只写不读 -------------------
CMD_STRUCT = struct.Struct("<B7d7x")
CMD_SIZE = CMD_STRUCT.size                  # 64
CMD_JOINTS = 0x00                           # [q1..q6, gripper, 0]
CMD_SETUP = 0x01                            # [cup_x, cup_y, cup_z, req_id, 0, 0, 0]
CMD_PLACE = 0x02                            # [x, y, z, req_id, 0, 0, 0]
CMD_TYPES = (CMD_JOINTS, CMD_SETUP, CMD_PLACE)

# ---- state 通道：sim_app -> bridge，24 字节头 + payload ---------------------
HDR = struct.Struct("<4sBBHdII")            # magic / type / cam / flags / stamp / plen / rsv
HDR_SIZE = HDR.size                         # 24
MAGIC = b"SIMB"
MAX_PAYLOAD = 16 * 1024 * 1024              # 和 bridge 侧一致，坏帧不会吃光内存

T_JOINTS = 1                                # '<' + 'd'*7
T_IMAGE = 2                                 # '<II' + h*w*3 的 RGB
T_INFO = 3                                  # '<' + 'd'*14  K(9) + D(5)
T_OBJECT = 4                                # '<' + 'd'*8   cup(x,y,z,height) + cube(x,y,z,edge)
T_PLACE_DONE = 5                            # '<dB3x'       req_id, success

CAM_FRONT = 0
CAM_WRIST = 1

JOINTS_STRUCT = struct.Struct("<7d")
IMAGE_HDR_STRUCT = struct.Struct("<II")
INFO_STRUCT = struct.Struct("<14d")
OBJECT_STRUCT = struct.Struct("<8d")
PLACE_DONE_STRUCT = struct.Struct("<dB3x")


def parse_cmd(frame: bytes) -> tuple[int, list[float]]:
    """把一条 64 字节的 cmd 帧解成 (类型, 7 个 double)。

    frame 必须正好 64 字节；读到别的长度说明流已经错位，只能丢连接重来，
    所以这里直接抛异常而不是返回 None。类型不在 CMD_TYPES 里同样抛异常。
    """
    if len(frame) != CMD_SIZE:
        raise ValueError(f"cmd 帧长度应为 {CMD_SIZE}，实际 {len(frame)}")
    kind, *values = CMD_STRUCT.unpack(frame)
    if kind not in CMD_TYPES:
        raise ValueError(f"未知 cmd 类型 0x{kind:02x}")
    return kind, values


def encode_frame(ftype: int, cam: int, payload: bytes, stamp: float) -> bytes:
    """拼一帧 state：24 字节头 + payload，整帧一次性交给 socket。

    头里的 plen 就是 payload 长度，bridge 按它切帧。payload 超过 MAX_PAYLOAD
    说明分辨率配错了（或者传进来的不是 RGB 字节流），这里直接抛异常，
    不要让一帧几 MB 的垃圾把对端的内存吃掉。
    """
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload {len(payload)} 字节超过上限 {MAX_PAYLOAD}")
    head = HDR.pack(MAGIC, ftype, cam, 0, stamp, len(payload), 0)
    return head + payload


def frame_joints(q, stamp: float) -> bytes:
    """type=1 关节帧：6 个关节角（弧度）+ 夹爪（0~1）。

    q 必须是 7 个数，且顺序固定为 Joint1..Joint6, gripper——bridge 侧直接
    按这个顺序填 JointState，顺序错了下游看不出来。
    """
    if len(q) != 7:
        raise ValueError(f"关节帧需要 7 个数，实际 {len(q)}")
    payload = JOINTS_STRUCT.pack(*[float(v) for v in q])
    return encode_frame(T_JOINTS, 0, payload, stamp)


def frame_image(cam: int, rgb, stamp: float) -> bytes:
    """type=2 图像帧：宽、高，然后是 h*w*3 的 RGB 字节。

    rgb 传 (H, W, 3) 或 (H, W, 4) 的 uint8 数组——Isaac 的相机常带 alpha
    通道，这里丢掉第 4 通道，保证 bridge 侧能直接当 rgb8 用。宽高用
    IMAGE_HDR_STRUCT('<II') 打包，顺序必须是宽在前。
    """
    arr = np.ascontiguousarray(rgb[..., :3], dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"图像形状应为 (H, W, 3)，实际 {arr.shape}")
    h, w = arr.shape[:2]
    payload = IMAGE_HDR_STRUCT.pack(w, h) + arr.tobytes()
    return encode_frame(T_IMAGE, cam, payload, stamp)


def frame_info(cam: int, K, D, stamp: float) -> bytes:
    """type=3 相机内参帧：K 的 9 个元素（行优先 3x3）+ D 的 5 个元素。

    bridge 侧原样塞进 CameraInfo 的 k 和 d，不做重排，所以这里必须按
    行优先摊平；D 长度固定 5（k1,k2,p1,p2,k3），仿真里一般是全零。
    """
    k = np.asarray(K, dtype=float).reshape(-1)
    d = np.asarray(D, dtype=float).reshape(-1)
    if k.size != 9:
        raise ValueError(f"K 应为 3x3，实际 {k.size} 个元素")
    if d.size != 5:
        raise ValueError(f"D 应为 5 个元素，实际 {d.size}")
    payload = INFO_STRUCT.pack(*k.tolist(), *d.tolist())
    return encode_frame(T_INFO, cam, payload, stamp)


def frame_object(cup_xyz_height, cube_xyz_edge, stamp: float) -> bytes:
    """type=4 物体真值帧：cup(x,y,z) + height，再接 cube(x,y,z) + edge。

    两个约定别写反：
    - cup 的 z 是**杯口中心高度**，不是杯体中心——detector 的 ground_truth
      判据直接拿它和方块 z 比"是否落进杯里"。
    - cube 的 z 是方块几何中心，不是底面。
    两个点各 3 个数，后面紧跟各自尺度，一共 8 个 double。
    """
    cup = [float(v) for v in cup_xyz_height]
    cube = [float(v) for v in cube_xyz_edge]
    if len(cup) != 4:
        raise ValueError(f"cup 需要 xyz+height 共 4 个数，实际 {len(cup)}")
    if len(cube) != 4:
        raise ValueError(f"cube 需要 xyz+edge 共 4 个数，实际 {len(cube)}")
    payload = OBJECT_STRUCT.pack(*(cup + cube))
    return encode_frame(T_OBJECT, 0, payload, stamp)


def frame_done(req_id: int, success: bool, stamp: float) -> bytes:
    """type=5 完成帧：req_id + success，回给 bridge 结束一次 setup/place。

    必须在这条之前先把新的 type=4 物体帧放进同一条 state 通道——同 socket
    是 FIFO，bridge 收到成功的瞬间下游一定已经拿到新位姿。
    """
    payload = PLACE_DONE_STRUCT.pack(int(req_id), bool(success))
    return encode_frame(T_PLACE_DONE, 0, payload, stamp)
