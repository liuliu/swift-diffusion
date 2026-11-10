#!/usr/bin/env bash
# Complete FP32 parity suite. No checkpoint export.
set -euo pipefail
cd "$(dirname "$0")/../.."
unset FIBO_CASE FIBO_VISION_TRACE FIBO_VERBOSE
export FIBO_DTYPE=float32
export FIBO_VARIANT=dev
export FIBO_SKIP_BUILD=0
bash examples/fibo2/run.sh parity-vae
export FIBO_SKIP_BUILD=1
bash examples/fibo2/run.sh parity-text
bash examples/fibo2/run.sh parity-dit
bash examples/fibo2/run.sh parity-pipeline
export FIBO_VARIANT=turbo
bash examples/fibo2/run.sh parity-dit
bash examples/fibo2/run.sh parity-pipeline
python3 examples/fibo2/summarize.py
