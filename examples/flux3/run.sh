#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python_environment=/home/liu/workspace/ltx2/LTX-2/_env
source "$python_environment/bin/activate"
export PYTHON_LIBRARY=/home/liu/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/libpython3.12.so
export LD_LIBRARY_PATH="$python_environment/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$python_environment/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"
export NVIDIA_TF32_OVERRIDE=0
if [[ "${FLUX3_SKIP_BUILD:-0}" != "1" ]]; then
  bazel build examples:flux3 --compilation_mode=dbg --keep_going
fi
exec ./bazel-bin/examples/flux3 "$@"
