# policy 容器：ROS 2 Humble + PyTorch + lerobot 运行时依赖
#
# 只装"依赖"，不放任何项目代码。代码（ros2_ws、lerobot 的 fork、checkpoint）在运行时用 -v 挂进来。
# 目的是改一行策略代码不需要重建镜像。
#
# 构建（GitHub 不通，必须走宿主代理 + host 网络，否则 git clone 那步直接失败）：
#   docker build --network host \
#     --build-arg HTTP_PROXY=http://127.0.0.1:7897 \
#     --build-arg HTTPS_PROXY=http://127.0.0.1:7897 \
#     -f docker/policy/Dockerfile -t alicia/policy:1.0 .
#
# 运行（示例，正式启动交给 compose）：
#   docker run --rm -it --network host --gpus all --ipc host \
#     --user $(id -u):$(id -g) \
#     -v $PWD/ros2_ws:/ws \
#     -v $PWD/checkpoints:/checkpoints \
#     alicia/policy:1.0 \
#     ros2 run act_policy_node policy_node

# 基于 alicia/base：它已经有 ROS desktop、python3-pip/numpy、非 root 用户 runner、
# 以及 distutils 包（numpy/sympy/PyYAML/scipy）的 egg-info 清理。这一层只加 torch 和 lerobot。
ARG BASE=alicia/base:1.0
FROM ${BASE}

# --- 可覆盖的构建参数 -------------------------------------------------------
# TORCH_INDEX_URL: 2080 的驱动需要和这里对齐；现有 lerobot6 环境是 torch 2.7.1+cu128
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1
ARG PIP_INDEX_URL=https://pypi.org/simple
# lerobot 必须是 Synria 的 fork（v6.0.0 内部版本号是 0.3.4，改了 Alicia-D 的接入）
# 从 PyPI 拉 HuggingFace 原版是不行的：没有机械臂那部分
ARG LEROBOT_SPEC="git+https://github.com/Synria-Robotics/lerobot.git@v6.0.0"

# OMP_NUM_THREADS=1：ACT 推理是单线程喂数据，给多了反而和 ROS 抢 CPU
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# --- 系统依赖 ---------------------------------------------------------------
# base 里已有 python3-pip / python3-numpy / curl / less，这里只补编译工具和 git
# （flatdict 之类只有源码包，要编译；lerobot 要从 git 拉）
# 不装 cuda-toolkit：torch wheel 自带 cuDNN/cuBLAS，宿主驱动够了
# 不装 ros-humble-cv-bridge：Image→numpy 用 np.frombuffer 三行就够
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-dev \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# --- PyTorch -----------------------------------------------------------------
# 用 --index-url 单独指向 pytorch 源，避免 pip 从 PyPI 拉到 CPU 版
# setuptools 锁 <81：lerobot v6.0.0 要求 >=71,<81（81 起删了 pkg_resources，flatdict 之类会编不过）
RUN python3 -m pip install --upgrade "pip<25" "setuptools>=71,<81" wheel \
    && python3 -m pip install --index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

# --- 策略运行时依赖 -----------------------------------------------------------
# 版本区间照 lerobot v6.0.0 的 pyproject.toml 抄，别自己放宽，否则 pip check 会一直报
# numpy 必须 <2：ROS Humble 那一套是按 numpy 1.x 编译的，升到 2.x 是 import 段错误，不是警告
#
# 这几个看着"推理用不到"但删不掉，因为 lerobot 是模块级 import：
#   datasets / torchcodec / jsonlines / imageio / termcolor → lerobot.datasets
#   pyserial → lerobot.motors（哪怕只跑策略也会被 import 到）
# 少任何一个，import ACTPolicy 就会 ModuleNotFoundError
RUN python3 -m pip install --index-url "${PIP_INDEX_URL}" \
        "numpy<2" \
        pillow \
        "einops>=0.8.0,<0.9.0" \
        "safetensors>=0.4.3" \
        "huggingface_hub>=0.34.2,<0.36.0" \
        "draccus==0.10.0" \
        "av>=15.0.0,<16.0.0" \
        "datasets>=4.0.0,<4.2.0" \
        "torchcodec>=0.2.1,<0.6.0" \
        "jsonlines>=4.0.0,<5.0.0" \
        "imageio[ffmpeg]>=2.34.0,<3.0.0" \
        "termcolor>=2.4.0,<4.0.0" \
        "pyserial>=3.5,<4.0" \
        "deepdiff>=7.0.1,<9.0.0"

# lerobot 本身：--no-deps 装，否则它会顺手拉 datasets / diffusers / wandb / gymnasium / rerun
# 这一串推理用不到，而且它把 alicia-d-sdk 的 git 地址写进了必需依赖（构建时 GitHub 不通就挂）
RUN python3 -m pip install --index-url "${PIP_INDEX_URL}" --no-deps "${LEROBOT_SPEC}"

# alicia-d-sdk 不装进镜像：它是硬件层，只允许 driver_node 用，而且源码要跟项目一起挂载
# 需要时在容器内执行（源码已在 /ws 下）：
#   pip install --no-deps -e /ws/src/github.com/Synria-Robotics/Alicia-D-SDK
#
# 如果某次构建确实要访问 GitHub（比如 --no-deps 失效），用宿主代理：
#   docker build --network host \
#     --build-arg HTTPS_PROXY=http://127.0.0.1:7897 \
#     --build-arg HTTP_PROXY=http://127.0.0.1:7897 -f docker/policy/Dockerfile -t alicia/policy:1.0 .

# --- 运行环境 -----------------------------------------------------------------
# 用户 runner 和 WORKDIR /ws 都在 base 里建好了，这里只切回去
WORKDIR /ws
USER runner

# 沿用官方镜像的 ros_entrypoint.sh，自动 source /opt/ros/humble/setup.bash
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
