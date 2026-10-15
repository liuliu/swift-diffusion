#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export NVIDIA_TF32_OVERRIDE=0
if [[ "${FLUX3_SKIP_BUILD:-0}" != "1" ]]; then
  bazel build examples:flux3_generate --compilation_mode=dbg --keep_going
fi
# No Python environment, runtime, subprocess, or precomputed embeddings.
exec ./bazel-bin/examples/flux3_generate "$@"
