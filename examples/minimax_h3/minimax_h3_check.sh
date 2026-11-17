#!/usr/bin/env bash
set -euo pipefail

cd /home/liu/workspace/swift-diffusion

export MINIMAX_H3_DEVICE=${MINIMAX_H3_DEVICE:-3}
# FP32 parity must disable TF32 in ccv/cuDNN too, not only in PyTorch.
export NVIDIA_TF32_OVERRIDE=0

output=$(mktemp -d /tmp/minimax_h3_image.XXXXXX)
exec > >(tee "${output}/run.log") 2>&1
echo "MiniMax H3 Image log: ${output}/run.log"
source /home/liu/workspace/ltx2/LTX-2/_env/bin/activate
export PYTHON_LIBRARY=/home/liu/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/libpython3.12.so
export LD_LIBRARY_PATH=/home/liu/workspace/ltx2/LTX-2/_env/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH:-}
export LD_PRELOAD=/home/liu/workspace/ltx2/LTX-2/_env/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2
# Reference environment: Diffusers 8069682140c675652a886736121f4e3b86f4187c,
# huggingface-hub 1.33.0. Installed without changing Torch or CUDA.
python -c 'from diffusers.models.transformers.transformer_minimax_h3_image import MiniMaxH3ImageTransformer2DModel; print("Official H3-Image DiT imports successfully", flush=True)'
# FP16 DiT export is complete. Default to non-mutating parity checks.
export H3_FP32_ATTENTION_CONTROL=1
bazel build examples:minimax_h3_converter --compilation_mode=dbg --keep_going --jobs=32
./bazel-bin/examples/minimax_h3_converter self-test
./bazel-bin/examples/minimax_h3_converter parity-h3-image-dit-f16
