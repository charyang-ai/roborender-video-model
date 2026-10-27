#!/bin/bash
# Host side. One W7900 render node. The workdir is mounted at /workspace.
#
#   bash scripts/radeon_start_container.sh /path/to/workdir [/dev/dri/renderD131]
#
# Put this repo at <workdir>/src. Weights, the venv and results stay in the workdir,
# not in the container writable layer. Only one render node is passed through.
# Match it to a GPU with `rocm-smi --showbus` and /dev/dri/by-path.
# Core dumps are off: a crashed HIP process can write a multi-gigabyte dump on the host.
set -euo pipefail
HOST_DIR=${1:?usage: radeon_start_container.sh HOST_DIR [RENDER_NODE]}
RENDER_NODE=${2:-/dev/dri/renderD128}
NAME=${NAME:-radeon-roborender-video-model}
IMAGE=${IMAGE:-rocm/pytorch:rocm7.14_ubuntu24.04_py3.12_pytorch_release_2.11.0}
VIDEO_GID=$(getent group video | cut -d: -f3)
RENDER_GID=$(getent group render | cut -d: -f3)
mkdir -p "$HOST_DIR/models" "$HOST_DIR/ckpt" "$HOST_DIR/data" "$HOST_DIR/results" \
  "$HOST_DIR/cache/pip" "$HOST_DIR/cache/miopen" "$HOST_DIR/cache/triton" "$HOST_DIR/cache/torch"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker start "$NAME" >/dev/null
else
  docker run -d --name "$NAME" \
    --entrypoint bash \
    --device=/dev/kfd \
    --device="$RENDER_NODE" \
    --group-add "$VIDEO_GID" \
    --group-add "$RENDER_GID" \
    --ipc=host \
    --shm-size=32g \
    --ulimit core=0 \
    --security-opt seccomp=unconfined \
    -e HIP_VISIBLE_DEVICES=0 \
    -e ROCR_VISIBLE_DEVICES=0 \
    -e MIOPEN_USER_DB_PATH=/workspace/cache/miopen \
    -e MIOPEN_CUSTOM_CACHE_DIR=/workspace/cache/miopen \
    -e TRITON_CACHE_DIR=/workspace/cache/triton \
    -e TORCH_HOME=/workspace/cache/torch \
    -e XDG_CACHE_HOME=/workspace/cache \
    -e PIP_CACHE_DIR=/workspace/cache/pip \
    -e HF_HOME=/workspace/cache/hf \
    -e TRANSFORMERS_OFFLINE=1 \
    -e HF_HUB_OFFLINE=1 \
    -v "$HOST_DIR":/workspace \
    -w /workspace \
    "$IMAGE" \
    -lc 'sleep infinity'
fi

docker exec "$NAME" /opt/venv/bin/python -c 'import torch; print("torch", torch.__version__, "hip", torch.version.hip, "devices", torch.cuda.device_count()); print(torch.cuda.get_device_name(0))'
