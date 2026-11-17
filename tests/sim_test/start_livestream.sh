#!/usr/bin/env bash
# 启动 Isaac Sim headless + WebRTC livestream，供笔记本上的 Isaac Sim WebRTC Streaming Client 连接。
# 用法: ./start_livestream.sh [GPU编号] [公网IP]
#   GPU 不传就自动挑剩余显存最多的那张；公网IP 默认 218.194.61.78（内网用就传 192.168.1.101）
set -e

if [ -z "${1:-}" ]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
        | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
else
  GPU="$1"
fi
PUBLIC_IP="${2:-218.194.61.78}"
CONDA_ENV=robot_sim
CONDA_SH="$HOME/Software/anaconda3/etc/profile.d/conda.sh"

source "$CONDA_SH"
conda activate "$CONDA_ENV"

export OMNI_KIT_ACCEPT_EULA=YES
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json   # 绕开系统里重复的 nvidia ICD
export CUDA_VISIBLE_DEVICES="$GPU"                                 # 这台机器其他卡都被别人占满，必须指定一张空闲卡
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY DISPLAY

exec isaacsim isaacsim.exp.full.streaming --no-window \
  --/app/livestream/publicEndpointAddress="$PUBLIC_IP" \
  --/app/livestream/port=49100
