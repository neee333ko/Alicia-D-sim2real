# camera：两路 UVC 相机的发布层
#
# 这一层几乎不做事——v4l2_camera 已经在 alicia/base 里装好了。
# 只多加一个 v4l-utils（v4l2-ctl），用来查设备支持哪些分辨率/帧率，
# 真机上排"相机是不是只能 30fps"这类问题全靠它。
#
# 构建：
#   docker build -f docker/camera/Dockerfile -t alicia/camera:1.0 .

ARG BASE=alicia/base:1.0
FROM ${BASE}

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
        v4l-utils \
        usbutils \
    && rm -rf /var/lib/apt/lists/*

USER runner

# 运行示例（两个相机都要挂设备 + 给 video 组权限）：
#   docker run --rm --network host \
#     --device=/dev/video0 --device=/dev/video2 --group-add video \
#     -v $PWD/ros2_ws:/ws \
#     -v $PWD/calib:/calib:ro \
#     alicia/camera:1.0 \
#     ros2 launch /ws/src/alicia_d_cameras/bringup.launch.py
#
# 设备号会变。插拔或重启后 /dev/video* 的编号可能对调，
# 用 udev 按序列号固定成 /dev/cam_front、/dev/cam_wrist 再挂进来更稳：
#   udevadm info -q property -n /dev/video0 | grep ID_SERIAL
#
# 查支持的格式和帧率（这是确认能不能跑 30Hz 的唯一可靠办法）：
#   docker run --rm --device=/dev/video0 --group-add video alicia/camera:1.0 \
#     v4l2-ctl --list-formats-ext -d /dev/video0
