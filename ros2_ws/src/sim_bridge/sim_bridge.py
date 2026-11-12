#!/usr/bin/env python3
"""sim_bridge.py — 把 Isaac Sim 伪装成 driver_node + camera_node。

为什么要有这一层：Isaac Sim 5.1 跑在 Python 3.11，ROS 2 Humble 是 3.10，
两者不能同进程。所以仿真侧（sim_app，py3.11，无 ROS）只做仿真，这一层（py3.10）
只做话题收发，中间用本机 unix socket 传数据。

对外接口和真机完全一致，下游（policy / action_executor / recorder）不需要知道
自己接的是仿真还是真机：

  订阅  /joint_commands        std_msgs/Float64MultiArray（7 维）
  发布  /joint_states          sensor_msgs/JointState（6 关节 + 夹爪）
        /camera/{front,wrist}/image_raw   sensor_msgs/Image（rgb8）
        /camera/{front,wrist}/camera_info sensor_msgs/CameraInfo
        /object/pose           geometry_msgs/PoseStamped（方块/杯子真值）
  服务  /arm/home              std_srvs/Trigger

———————————————— 内部协议 ——————————————————————————————
bridge 主动连两条 unix socket（sim_app 作为 server 监听）：

  cmd   路径由 sim_cmd_sock 参数给（默认 /tmp/alicia_sim_cmd.sock）
        bridge 往里写定长帧，不读：
            '<d' * 7    一个 7 关节目标（弧度 + 夹爪 0~1），56 字节
            'H'         0x01 = 请求复位（1 字节，后面补 7 字节 0 保持等长）
        为了让解析简单，所有命令都写成 64 字节定长：1 字节类型 + 7 个 double + 7 字节填充

  state 路径由 sim_state_sock 参数给（默认 /tmp/alicia_sim_state.sock）
        bridge 只读。每帧 24 字节头 + payload：
            magic  'SIMB'         4s
            type   uint8  1=关节 2=图像 3=相机内参 4=物体位姿
            cam    uint8  0=front 1=wrist（只有图像/内参用）
            flags  uint16 预留
            stamp  float64 采集时刻（墙钟秒）
            plen   uint32 payload 长度
            rsv    uint32 预留
        payload:
            type=1  '<' + 'd'*7              关节角弧度 + 夹爪 0~1
            type=2  '<II' + 字节流           宽、高、然后 h*w*3 的 RGB
            type=3  '<' + 'd'*14             K(9) + D(5)
            type=4  '<' + 'd'*7              x y z qx qy qz qw
"""

from __future__ import annotations

import array
import select
import socket
import struct
import threading
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from builtin_interfaces.msg import Time as TimeMsg
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Float64MultiArray
from std_srvs.srv import Trigger

ACTION_DIM = 7
JOINT_NAMES = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6", "gripper"]

HDR = struct.Struct("<4sBBHdII")          # 24 字节
MAGIC = b"SIMB"
# 单帧上限：640x480x3 的图像约 0.9MB，留 16MB 足够，坏帧也不会把内存吃光
MAX_PAYLOAD = 16 * 1024 * 1024

T_JOINTS, T_IMAGE, T_INFO, T_OBJECT = 1, 2, 3, 4
CMD_JOINTS = 0x00
CMD_HOME = 0x01
CMD_STRUCT = struct.Struct("<B7d7x")      # 64 字节定长


class SimBridge(Node):
    def __init__(self):
        super().__init__("sim_bridge")

        self.declare_parameter("sim_cmd_sock", "/tmp/alicia_sim_cmd.sock")
        self.declare_parameter("sim_state_sock", "/tmp/alicia_sim_state.sock")
        self.declare_parameter("reconnect_s", 1.0)

        p = self.get_parameter
        self.cmd_path = p("sim_cmd_sock").value
        self.state_path = p("sim_state_sock").value
        self.reconnect_s = float(p("reconnect_s").value)

        self.cmd_sock: socket.socket | None = None
        self.cmd_lock = threading.Lock()
        self._stop = False
        self.n_frames = {T_JOINTS: 0, T_IMAGE: 0, T_INFO: 0, T_OBJECT: 0}
        self.camera_info = {}                      # cam -> (K, D)

        # ---- 发布端 --------------------------------------------------------
        qos_img = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        qos_state = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.pub_state = self.create_publisher(JointState, "/joint_states", qos_state)
        self.pub_front = self.create_publisher(Image, "/camera/front/image_raw", qos_img)
        self.pub_wrist = self.create_publisher(Image, "/camera/wrist/image_raw", qos_img)
        self.pub_front_info = self.create_publisher(
            CameraInfo, "/camera/front/camera_info", qos_state
        )
        self.pub_wrist_info = self.create_publisher(
            CameraInfo, "/camera/wrist/camera_info", qos_state
        )
        self.pub_object = self.create_publisher(PoseStamped, "/object/pose", qos_state)

        # ---- 订阅端 --------------------------------------------------------
        self.sub_cmd = self.create_subscription(
            Float64MultiArray, "/joint_commands", self._on_command, 10
        )
        self.srv_home = self.create_service(Trigger, "/arm/home", self._on_home)

        # ---- 两条连接 ------------------------------------------------------
        self.writer = threading.Thread(target=self._cmd_loop, daemon=True)
        self.reader = threading.Thread(target=self._state_loop, daemon=True)
        self.writer.start()
        self.reader.start()

        # 每秒统计一次收到的帧率，真机/仿真对不上时先看这里
        self._last_stat = time.time()
        self._last_counts = dict(self.n_frames)
        self.create_timer(1.0, self._log_stats)

        self.get_logger().info(
            f"sim_bridge 启动：cmd={self.cmd_path} state={self.state_path}"
        )

    def _log_stats(self):
        now = time.time()
        dt = max(now - self._last_stat, 1e-6)
        rates = {k: (self.n_frames[k] - self._last_counts[k]) / dt for k in self.n_frames}
        self._last_stat, self._last_counts = now, dict(self.n_frames)
        if any(v > 0 for v in rates.values()):
            self.get_logger().info(
                f"帧率：关节 {rates[T_JOINTS]:.1f}/s 图像 {rates[T_IMAGE]:.1f}/s "
                f"内参 {rates[T_INFO]:.1f}/s 物体 {rates[T_OBJECT]:.1f}/s"
            )

    # ------------------------------------------------------------------ 命令
    def _on_command(self, msg: Float64MultiArray):
        if len(msg.data) != ACTION_DIM:
            self.get_logger().error(
                f"/joint_commands 需要 {ACTION_DIM} 个值，收到 {len(msg.data)}",
                throttle_duration_sec=5.0,
            )
            return
        vals = [float(v) for v in msg.data]
        if not all(np.isfinite(v) for v in vals):
            self.get_logger().error("命令含 NaN/Inf，丢弃", throttle_duration_sec=1.0)
            return
        self._send(CMD_JOINTS, vals)

    def _on_home(self, req, resp):
        ok = self._send(CMD_HOME, [0.0] * ACTION_DIM)
        resp.success = ok
        resp.message = "sent" if ok else "sim 未连接，复位请求未送达"
        return resp

    def _send(self, kind: int, values: list[float]) -> bool:
        with self.cmd_lock:
            if self.cmd_sock is None:
                self.get_logger().warn(
                    "sim 未连接，命令被丢弃", throttle_duration_sec=2.0
                )
                return False
            try:
                self.cmd_sock.sendall(CMD_STRUCT.pack(kind, *values))
                return True
            except OSError as e:
                self.get_logger().warn(f"写命令失败：{e}", throttle_duration_sec=2.0)
                # 注意：这里已经持有 cmd_lock，不能调 _close_cmd（它要再拿一次同一把锁，
                # threading.Lock 不可重入，会直接死锁）。就地关闭并置空。
                try:
                    self.cmd_sock.close()
                except OSError:
                    pass
                self.cmd_sock = None
                return False

    def _cmd_loop(self):
        """连上 sim 的命令端口。断开就每秒重连一次。

        光靠写失败来判断断开不够：没有命令下发时永远发现不了。这里用 select
        探一下可读性——对端关闭时 select 会立刻返回可读，peek 到空就是 EOF。
        """
        while not self._stop:
            if self.cmd_sock is not None:
                try:
                    r, _, _ = select.select([self.cmd_sock], [], [], 0)
                    if r and self.cmd_sock.recv(1, socket.MSG_PEEK) == b"":
                        self.get_logger().info("sim 的命令通道已关闭，重连")
                        self._close_cmd()
                except (OSError, ValueError):
                    self._close_cmd()
            if self.cmd_sock is None:
                try:
                    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    s.connect(self.cmd_path)
                    s.settimeout(2.0)
                    with self.cmd_lock:
                        self.cmd_sock = s
                    self.get_logger().info(f"已连上命令通道 {self.cmd_path}")
                except OSError:
                    self.get_logger().info(
                        f"等待 sim_app 的命令端口 {self.cmd_path}",
                        throttle_duration_sec=10.0,
                    )
            time.sleep(self.reconnect_s)

    def _close_cmd(self):
        with self.cmd_lock:
            if self.cmd_sock is not None:
                try:
                    self.cmd_sock.close()
                except OSError:
                    pass
                self.cmd_sock = None

    # ------------------------------------------------------------------ 状态
    def _state_loop(self):
        while not self._stop:
            s = None
            try:
                s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                s.connect(self.state_path)
                s.settimeout(None)
                self.get_logger().info(f"已连上状态通道 {self.state_path}")
                self._read_frames(s)
            except OSError:
                self.get_logger().info(
                    f"等待 sim_app 的状态端口 {self.state_path}", throttle_duration_sec=10.0
                )
            except Exception as e:                      # noqa: BLE001
                self.get_logger().error(f"状态通道异常：{e}", throttle_duration_sec=2.0)
            finally:
                if s is not None:
                    try:
                        s.close()
                    except OSError:
                        pass
            time.sleep(self.reconnect_s)

    def _read_frames(self, sock: socket.socket):
        while not self._stop:
            hdr = _recv_exact(sock, HDR.size)
            if hdr is None:
                return
            magic, mtype, cam, _flags, stamp, plen, _rsv = HDR.unpack(hdr)
            if magic != MAGIC:
                self.get_logger().error("帧头 magic 不对，断开重连")
                return
            if plen > MAX_PAYLOAD:
                self.get_logger().error(f"帧长度 {plen} 超出上限，断开重连")
                return
            payload = _recv_exact(sock, plen) if plen else b""
            if payload is None:
                return
            if mtype == T_JOINTS:
                self._on_joints(payload, stamp)
            elif mtype == T_IMAGE:
                self._on_image(payload, stamp, cam)
            elif mtype == T_INFO:
                self._on_info(payload, cam)
            elif mtype == T_OBJECT:
                self._on_object(payload, stamp)
            else:
                self.get_logger().warn(f"未知帧类型 {mtype}", throttle_duration_sec=5.0)

    def _on_joints(self, payload: bytes, stamp: float):
        if len(payload) != ACTION_DIM * 8:
            return
        vals = struct.unpack("<7d", payload)
        if not all(np.isfinite(v) for v in vals):
            self.get_logger().error("sim 发来的关节角含 NaN/Inf，丢弃", throttle_duration_sec=1.0)
            return
        msg = JointState()
        msg.header.stamp = _to_ros_time(stamp)
        msg.name = JOINT_NAMES
        msg.position = [float(v) for v in vals]
        self.pub_state.publish(msg)
        self.n_frames[T_JOINTS] += 1

    def _on_image(self, payload: bytes, stamp: float, cam: int):
        if cam not in (0, 1):
            self.get_logger().warn(f"未知相机号 {cam}", throttle_duration_sec=5.0)
            return
        if len(payload) < 8:
            return
        w, h = struct.unpack("<II", payload[:8])
        data = payload[8:]
        if len(data) != w * h * 3:
            self.get_logger().error(
                f"图像长度不符：{len(data)} != {w}x{h}x3", throttle_duration_sec=2.0
            )
            return

        msg = Image()
        msg.header.stamp = _to_ros_time(stamp)
        msg.header.frame_id = "cam_front_optical" if cam == 0 else "cam_wrist_optical"
        msg.height = h
        msg.width = w
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = w * 3
        # 必须用 array.array 而不是 bytes：rclpy 给 uint8[] 字段赋一个 bytes 对象会走
        # 极慢的逐元素转换路径，实测 921KB 从 3000+ fps 掉到 6.6 fps。array('B', ...) 是 C 层拷贝。
        msg.data = array.array("B", data)
        if cam == 0:
            self.pub_front.publish(msg)
            info_pub = self.pub_front_info
        else:
            self.pub_wrist.publish(msg)
            info_pub = self.pub_wrist_info

        # 内参跟着图像一起发，下游拿到就能用
        cached = self.camera_info.get(cam)
        if cached is not None:
            info_pub.publish(_make_info(msg.header.stamp, msg.header.frame_id, w, h, *cached))
        self.n_frames[T_IMAGE] += 1

    def _on_info(self, payload: bytes, cam: int):
        if len(payload) != 14 * 8:
            return
        vals = struct.unpack("<14d", payload)
        self.camera_info[cam] = (list(vals[:9]), list(vals[9:]))
        self.n_frames[T_INFO] += 1

    def _on_object(self, payload: bytes, stamp: float):
        if len(payload) != 7 * 8:
            return
        x, y, z, qx, qy, qz, qw = struct.unpack("<7d", payload)
        msg = PoseStamped()
        msg.header.stamp = _to_ros_time(stamp)
        # TODO: 现在没有 world→base_link 的 TF，下游要用 tf2 变换会失败。
        # 要么把这里改成 base_link，要么让 robot_state_publisher 发一个静态变换。
        msg.header.frame_id = "world"
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.x = qx
        msg.pose.orientation.y = qy
        msg.pose.orientation.z = qz
        msg.pose.orientation.w = qw
        self.pub_object.publish(msg)
        self.n_frames[T_OBJECT] += 1

    def shutdown(self):
        self._stop = True
        self._close_cmd()


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        try:
            chunk = sock.recv(n - len(buf))
        except OSError:
            return None
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def _to_ros_time(wall: float) -> TimeMsg:
    """收到的 stamp 是墙钟秒，转成 ROS 消息的 Time。"""
    sec = int(wall)
    return TimeMsg(sec=sec, nanosec=int((wall - sec) * 1e9))


def _make_info(stamp, frame_id, w, h, K, D) -> CameraInfo:
    msg = CameraInfo()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.width = w
    msg.height = h
    msg.distortion_model = "plumb_bob"
    msg.k = [float(v) for v in K]
    msg.d = [float(v) for v in D]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [K[0], 0.0, K[2], 0.0, 0.0, K[4], K[5], 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def main():
    rclpy.init()
    node = SimBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
