#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
s4nnc_directory=examples/fibo2/artifacts/s4nnc
s4nnc_revision=5bb3fb70e0095290ce9cb0ac5090748af1d8e9a8
if [[ ! -d "$s4nnc_directory/.git" ]]; then
  git clone --shared --no-checkout /home/liu/workspace/s4nnc "$s4nnc_directory"
  git -C "$s4nnc_directory" checkout --detach "$s4nnc_revision"
fi
test "$(git -C "$s4nnc_directory" rev-parse HEAD)" = "$s4nnc_revision"
ccv_directory=examples/fibo2/artifacts/ccv
ccv_revision=329ab6c52085dff8e0de931824f2ee1b6d64b937
if [[ ! -d "$ccv_directory/.git" ]]; then
  git init "$ccv_directory"
  git -C "$ccv_directory" fetch --depth=1 https://github.com/liuliu/ccv.git "$ccv_revision"
  git -C "$ccv_directory" checkout --detach FETCH_HEAD
fi
test "$(git -C "$ccv_directory" rev-parse HEAD)" = "$ccv_revision"
uv pip install --python /home/liu/workspace/ltx2/LTX-2/_env/bin/python \
  --target examples/fibo2/artifacts/python \
  'diffusers @ git+https://github.com/huggingface/diffusers.git@d0b9b1b5a540e5f9df2f8ec236347bf9c5aa7d1f' \
  'transformers==5.20.1' 'huggingface-hub==2.4.0' 'tokenizers==0.23.3'
