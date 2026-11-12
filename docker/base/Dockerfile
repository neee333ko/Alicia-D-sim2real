# 公共基座：ROS 2 Humble 桌面版 + 相机驱动 + 常用工具
#
# driver / camera 都从这一层出来；policy 暂时还直接基于 osrf 镜像，
# 后面可以也改成 FROM alicia/base，省掉重复的 apt 步骤。
#
# 构建：
#   docker build -f docker/base/Dockerfile -t alicia/base:1.0 .
#
# 这一层只放"所有容器都要"的东西。torch、lerobot、SDK 一律不进 base。

ARG BASE=osrf/ros:humble-desktop
FROM ${BASE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip \
        python3-numpy \
        # 相机节点用官方包，不自己写：UVC 的格式协商、双缓冲、掉线重连它都处理好了
        ros-humble-v4l2-camera \
        ros-humble-camera-info-manager \
        curl \
        less \
    && rm -rf /var/lib/apt/lists/*

# apt 装的 numpy 是 distutils 包，只有 *.egg-info。上层镜像用 pip 升 numpy/sympy 时
# 会报 uninstall-distutils-installed-package，删掉 egg-info 让 pip 认为它没装过。
RUN rm -rf /usr/lib/python3/dist-packages/numpy-*.egg-info \
           /usr/lib/python3/dist-packages/sympy-*.egg-info \
           /usr/lib/python3/dist-packages/PyYAML-*.egg-info \
           /usr/lib/python3/dist-packages/scipy-*.egg-info

# 不用 root 跑：容器和宿主共用挂载目录，root 写出来的文件在宿主上改不动
ARG USER_UID=1018
ARG USER_GID=1018
RUN groupadd -o -g "${USER_GID}" runner \
    && useradd -o -u "${USER_UID}" -g "${USER_GID}" -m -s /bin/bash runner

# 工作目录要可写：SDK 会往 CWD 下建 logs/，root 拥有的目录会让它直接报错
RUN mkdir -p /ws && chown "${USER_UID}:${USER_GID}" /ws

WORKDIR /ws
USER runner

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
