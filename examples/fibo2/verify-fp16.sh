#!/usr/bin/env bash
# FP16 text/vision and DiT against FP32. VAE and sampler state stay FP32.
set -euo pipefail
cd "$(dirname "$0")/../.."
unset FIBO_CASE FIBO_VISION_TRACE FIBO_VERBOSE
export FIBO_DTYPE=float16 FIBO_REFERENCE_DTYPE=float32
export FIBO_VARIANT=dev FIBO_SKIP_BUILD=0
fp16_status=0
bash examples/fibo2/run.sh parity-text
export FIBO_SKIP_BUILD=1
bash examples/fibo2/run.sh parity-dit
bash examples/fibo2/run.sh parity-pipeline || fp16_status=1
export FIBO_VARIANT=turbo
bash examples/fibo2/run.sh parity-dit
bash examples/fibo2/run.sh parity-pipeline || fp16_status=1
python3 examples/fibo2/summarize.py
exit "$fp16_status"
