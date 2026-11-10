#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
cd "$(dirname "$0")/../.."
python_environment=/home/liu/workspace/ltx2/LTX-2/_env
source "$python_environment/bin/activate"
export PYTHONPATH="$PWD/examples/fibo2/artifacts/python:${PYTHONPATH:-}"
export PYTHON_LIBRARY=/home/liu/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/libpython3.12.so
export LD_LIBRARY_PATH="$python_environment/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$python_environment/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"
export NVIDIA_TF32_OVERRIDE=0
if [[ "${FIBO_SKIP_BUILD:-0}" != 1 ]]; then
  bazel build examples:fibo2 --compilation_mode=dbg --keep_going --jobs=8 \
    --disk_cache=/slow/Data/Fibo-2.0-parity/build-cache \
    --override_repository=ccv="$PWD/examples/fibo2/artifacts/ccv" \
    --override_repository=s4nnc="$PWD/examples/fibo2/artifacts/s4nnc"
fi
exec ./bazel-bin/examples/fibo2 "${@:-parity-vae}"
