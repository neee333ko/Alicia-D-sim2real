# recorder 容器：ROS 2 Humble + lerobot 数据集依赖 + OpenCV
#
# 为什么不复用 policy 镜像：policy 里没有 OpenCV，录制端 30Hz 双路 1280x720
# 只能靠 Pillow 编码，一张 15~25ms、两路就超过 33ms 采样周期，回调被拖住后
# 60Hz 的 /joint_states 会掉帧。这一层只多装 opencv-python-headless
# （imencode 约 4~8ms/张，两路也在预算内），其余依赖与 policy 完全一致，
# 避免两套 lerobot 行为漂移。
#
# 构建（GitHub 不通，必须走代理 + host 网络，否则 git clone 那步直接失败）：
#   docker build --network host \
#     --build-arg HTTP_PROXY=http://moveai.duckdns.org:7892 \
#     --build-arg HTTPS_PROXY=http://moveai.duckdns.org:7892 \
#     -f docker/recorder/Dockerfile -t alicia/recorder:1.0 .
#
# 代理用宿主环境里的 moveai.duckdns.org:7892。tools/proxy.env 里那套
# 127.0.0.1:7897 依赖 ssh 反向隧道，隧道没开时 pip 会直接 Connection refused。
#
# 运行（datasets 必须落在宿主上，容器一删不能把刚录的数据带走）：
#   docker run --rm -it --network host --ipc host \
#     --user $(id -u):$(id -g) \
#     -v $PWD/ros2_ws:/ws \
#     -v $PWD/datasets:/datasets \
#     -v $PWD/task.yaml:/ws/task.yaml:ro \
#     alicia/recorder:1.0 \
#     python3 /ws/src/recorder/recorder_node.py

ARG BASE=alicia/base:1.0
FROM ${BASE}

# --- 可覆盖的构建参数 -------------------------------------------------------
# 与 policy 保持一致：同一个 lerobot 版本、同一套 torch，只是多了 opencv
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG TORCH_VERSION=2.7.1
ARG TORCHVISION_VERSION=0.22.1
ARG PIP_INDEX_URL=https://pypi.org/simple
# lerobot 必须是 Synria 的 fork（v6.0.0 内部版本号是 0.3.4）
ARG LEROBOT_SPEC="git+https://github.com/Synria-Robotics/lerobot.git@v6.0.0"

# OMP_NUM_THREADS=1：录制是单线程事件循环，给多了反而和 ROS 抢 CPU
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1

# --- 系统依赖 ---------------------------------------------------------------
# base 里已有 python3-pip / python3-numpy / curl / less，这里补编译工具和 git
# （flatdict 之类只有源码包要编译；lerobot 要从 git 拉）
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-dev \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# --- PyTorch -----------------------------------------------------------------
# lerobot.datasets 在模块级 import torch，装不了纯 CPU 以外的替代。
# setuptools 锁 <81：lerobot v6.0.0 要求 >=71,<81
RUN python3 -m pip install --upgrade "pip<25" "setuptools>=71,<81" wheel \
    && python3 -m pip install --index-url "${TORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}"

# --- 录制运行时依赖 -----------------------------------------------------------
# 版本区间照 lerobot v6.0.0 的 pyproject.toml 抄，与 policy 逐行一致。
# numpy 必须 <2：ROS Humble 那一套是按 numpy 1.x 编译的，升到 2.x 是 import 段错误。
# opencv-python-headless：不需要 GUI，headless 版少拉一整套 GTK 依赖。
#   datasets / torchcodec / jsonlines / imageio / termcolor → lerobot.datasets
#   pyserial → lerobot.motors（模块级 import，删不掉）
RUN python3 -m pip install --index-url "${PIP_INDEX_URL}" \
        "numpy<2" \
        pillow \
        "opencv-python-headless>=4.10,<5.0" \
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
RUN python3 -m pip install --index-url "${PIP_INDEX_URL}" --no-deps "${LEROBOT_SPEC}"

# --- 运行环境 -----------------------------------------------------------------
# 用户 runner 和 WORKDIR /ws 都在 base 里建好了，这里只切回去
WORKDIR /ws
USER runner

# 沿用官方镜像的 ros_entrypoint.sh，自动 source /opt/ros/humble/setup.bash
ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
