#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python_environment=/home/liu/workspace/ltx2/LTX-2/_env
source "$python_environment/bin/activate"
export PYTHON_LIBRARY=/home/liu/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/libpython3.12.so
export LD_LIBRARY_PATH="$python_environment/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$python_environment/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"
# ccv requests cuDNN tensor-op math; disable TF32 for genuine FP32 controls.
export NVIDIA_TF32_OVERRIDE=0
bazel build examples:qwen_image21 --compilation_mode=dbg --keep_going
exec ./bazel-bin/examples/qwen_image21 "$@"
