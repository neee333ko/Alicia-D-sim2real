#!/usr/bin/env python3
"""两路 UVC 相机的启动文件。

做三件事：起两个 v4l2_camera_node、把话题重映射成项目约定的名字、给每路指定标定文件。

不写自研节点：UVC 的格式协商、双缓冲、帧超时都交给 ros-humble-v4l2-camera。
我们只负责参数和标定。

    ros2 launch /ws/src/alicia_d_cameras/launch/bringup.launch.py

参数文件默认读同包的 config/cameras.yaml（按节点名分段，两路各一段）。
标定文件默认在 /calib 下（宿主 pi/calib 只读挂载进容器）。
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")

    # 同一份 YAML 传给两个节点：ROS 2 会按节点名各取自己那段
    common = {
        "parameters": [params_file],
        "output": "screen",
    }

    cam_front = Node(
        package="v4l2_camera",
        executable="v4l2_camera_node",
        name="cam_front",
        remappings=[
            ("image_raw", "/camera/front/image_raw"),
            ("camera_info", "/camera/front/camera_info"),
        ],
        **common,
    )

    cam_wrist = Node(
        package="v4l2_camera",
        executable="v4l2_camera_node",
        name="cam_wrist",
        remappings=[
            ("image_raw", "/camera/wrist/image_raw"),
            ("camera_info", "/camera/wrist/camera_info"),
        ],
        **common,
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value="/ws/src/alicia_d_cameras/config/cameras.yaml",
                description="相机参数文件",
            ),
            cam_front,
            cam_wrist,
        ]
    )
