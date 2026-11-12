# driver：真机机械臂的串口层
#
# 只需要 rclpy + pyserial + SDK 的运行依赖。不装 torch、不装 OpenCV、不装相机驱动。
# SDK 源码不打进镜像，运行时挂载并用 PYTHONPATH 引入（见文末）。
#
# 构建：
#   docker build -f docker/driver/Dockerfile -t alicia/driver:1.0 .

ARG BASE=alicia/base:1.0
FROM ${BASE}

USER root

# SDK 的两个依赖包从 PyPI 拿（GitHub 不通，git+https 那种装法会失败）
# 版本和 SDK v6.0.0 配套：synria-robocore 1.0.0 / synriard 1.0.2
# 除了 SDK 自己声明的 numpy/pyserial，还得手动补三个：
#   omegaconf → robocore/configs 里直接 import
#   scipy     → robocore 运动学
#   pyyaml>=6 → apt 自带的是 5.4.1，SDK 要求 >=6
# websockets 不装：只有 SDK 的 sparkvis（真机可视化网页）用，driver 用不到
RUN python3 -m pip install --no-cache-dir \
        "numpy<2" \
        "pyserial>=3.5,<4.0" \
        "omegaconf" \
        "pyyaml>=6" \
        "scipy>=1.11" \
        "synria-robocore==1.0.0" \
        "synriard==1.0.2"

USER runner

# SDK 源码在宿主的 pi 里，容器里通过挂载 + PYTHONPATH 使用：
#   -v $PWD/src/github.com/Synria-Robotics/Alicia-D-SDK:/sdk:ro
# PYTHONPATH 已指向 /sdk，所以容器里可以直接 import alicia_d_sdk
ENV PYTHONPATH=/sdk

# 运行示例（串口必须显式挂进去，光有关系没权限也打不开）：
#   docker run --rm --network host \
#     --device=/dev/ttyUSB1 --group-add dialout \
#     -v $PWD/ros2_ws:/ws \
#     -v $PWD/src/github.com/Synria-Robotics/Alicia-D-SDK:/sdk:ro \
#     alicia/driver:1.0 \
#     python3 /ws/src/alicia_d_driver/driver_node.py --ros-args -p port:=/dev/ttyUSB1
