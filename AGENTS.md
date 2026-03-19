# Agent Notes: PythonKit + Swift PyTorch-Parity Workflow

These notes capture what worked for `examples/ltx2` (LTX-2 spatial upscaler) and should be reused for future model ports.

## 1) Environment and launch

- Use the same run pattern from shell history:
  - `source ../ltx2/_env/bin/activate`
  - `export PYTHONPATH=/home/liu/workspace/ltx2/LTX-2/packages/ltx-core/src:${PYTHONPATH}`
  - `bazel run examples:ltx2 --compilation_mode=dbg --keep_going`
- Keep fresh repo path (`../ltx2/LTX-2`) for latest code. Do not use `../ltx2/ltx-core` for new model behavior.

## 2) PythonKit import hygiene

- PythonKit may not see all site-packages by default. Before importing model code:
  - insert `site.getusersitepackages()` into `sys.path` when missing.
  - insert `/usr/lib/python3/dist-packages` into `sys.path` when missing (needed for modules like `setuptools` transitively required by `triton`/`ltx_core` imports).
- Keep optional/heavy imports below early `exit(0)` when iterating on one submodel to avoid unrelated dependency failures.
- For CUDA-related work in this repo, prefer running direct Python probes, parity binaries, and other GPU executions **outside the sandbox**.
  - Sandboxed runs can report misleading CUDA failures such as `cuInit(0) = 100` / `cudaErrorNoDevice` even when unrestricted runs see all GPUs normally.
  - If a CUDA / PyTorch / ccv issue appears suspicious, rerun the same command unrestricted before drawing conclusions.

## 3) Swift model parity rules (must match PyTorch op order)

- Exact operator order matters more than anything else.
- For LTX-2 upscaler `ResBlock`, correct order is:
  - `conv1 -> norm1 -> SiLU -> conv2 -> norm2 -> (x + residual) -> SiLU`
- Do not switch to pre-norm pattern unless PyTorch block is pre-norm.
- Keep tensors in `N,C,D,H,W` through the graph and only permute when required by pixel shuffle layout.

## 4) Padding and layout rules

- For H/W-only padding, prefer convolution `hint` borders.
- If temporal padding is needed, use `pad`, but because pad helper is 4D-oriented:
  - reshape to 4D, pad, then reshape back.
- Validate all conv filter shapes with model semantics:
  - upsampler `Conv2d` weights from PyTorch become Swift 3D conv weights via `unsqueeze(2)` (temporal kernel size 1).

## 5) LTX-2 spatial upscaler specifics

- Checkpoint: `ltx-2-spatial-upscaler-x2-1.0.safetensors`
- Config expected in metadata:
  - `in_channels=128`, `mid_channels=1024`, `num_blocks_per_stage=4`, `dims=3`
  - `spatial_upsample=true`, `temporal_upsample=false`, `spatial_scale=2.0`, `rational_resampler=true`
- Rational x2 spatial path behavior:
  - `upsampler.conv` + pixel shuffle on H/W.
  - `blur_down(stride=1)` is effectively identity for this checkpoint.

## 6) Validation-before-export checklist

- Numeric parity checks are **mandatory** for every model conversion in this repo.
  - Do this per converted submodel first (for example: text encoder, adapter / connector, diffusion model, VAE pieces).
  - Then do an end-to-end parity check for the active exported unit when feasible.
  - Do not treat a conversion as complete just because it builds, runs, or writes a ckpt.
- Always run Swift and PyTorch on the same seeded random latent.
- Confirm output shape equality first.
- Compare sample values and record at least max-abs diff before writing ckpt.
- Prefer also recording relative diff when reference magnitudes vary significantly.
- After parity passes, ask the user before running the export step.
- When performing the final export run, rerun the numeric parity check in that same session before writing ckpt when practical.
- Only export graph (`graph.openStore(...).write(...)`) after numeric parity is confirmed and the user has approved export.

## 6.1) Model config representation

- Prefer to keep model architecture / config values on the Swift side as explicit constants.
- It is fine to group them into constant structs or tables instead of inlining them, but they should stay easy to inspect in Swift.
- Do not rely on opaque Python / JSON config reads for core architecture choices during conversion unless there is a specific reason and the user asked for it.

## 7) Iteration pattern in `examples/ltx2/main.swift`

- Add new converter block near top.
- Run only that block with `exit(0)` immediately after it.
- Keep older experiments below for reference, but not on active execution path.

## 8) LTX-2.3 environment (working setup for `examples/ltx23`)

- Repository: `/home/liu/workspace/ltx2/LTX-2`
- Create and sync dedicated Python 3.12 env:
  - `cd /home/liu/workspace/ltx2/LTX-2`
  - `uv venv -p 3.12 _env`
  - `source _env/bin/activate`
  - `uv sync --frozen --active`
- Confirmed compatible stack in this env:
  - `torch==2.9.1+cu128` (`torch.version.cuda == 12.8`)

### Build and run `ltx23` reliably

- Build first (from `/home/liu/workspace/swift-diffusion`):
  - `bazel build examples:ltx23 --compilation_mode=dbg --keep_going`
- Then run the built binary with the following environment:
  - `source /home/liu/workspace/ltx2/LTX-2/_env/bin/activate`
  - `export PYTHONPATH=/home/liu/workspace/ltx2/LTX-2/packages/ltx-core/src:${PYTHONPATH}`
  - `export PYTHON_LIBRARY=/home/liu/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/libpython3.12.so`
  - `export LD_LIBRARY_PATH=/home/liu/workspace/ltx2/LTX-2/_env/lib/python3.12/site-packages/nvidia/nccl/lib:${LD_LIBRARY_PATH}`
  - `export LD_PRELOAD=/home/liu/workspace/ltx2/LTX-2/_env/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2`
  - `./bazel-bin/examples/ltx23`

### Why this is needed

- PythonKit embeds Python and may miss correct runtime/library selection unless `PYTHON_LIBRARY` is pinned.
- Without the NCCL path/preload above, `torch` import can fail with:
  - `undefined symbol: ncclCommWindowRegister`
- Running `./bazel-bin/examples/ltx23` after build is currently more stable than `bazel run` when these extra env vars are required.

## 9) LTX-2.3 audio/video VAE lessons (from this session)

- Keep latent per-channel stats (`std/mean`) **outside** Swift model definitions.
  - Decoder: denormalize input latents before feeding decoder.
  - Encoder: normalize encoder output after model forward.
  - Do not bake these into `Parameter` inside encoder/decoder model builders.
- For runtime `DynamicGraph` tensor reshapes in active parity code, prefer explicit shape formats (`.NCHW`, `.HWC`, `.CHW`) instead of raw `[Int]`.
- Keep non-essential/heavy Python imports (Gemma/transformer/text stack) out of the active path when iterating on video VAE; otherwise unrelated PIL/transformers import issues can crash before parity checks.
- LTX-2.3 video VAE block schedule differs from older LTX-2:
  - Decoder stages: `(1024 x2 res) -> up(2,2,2,/2) -> (512 x2 res) -> up(2,2,2,/1) -> (512 x4 res) -> up(2,1,1,/2) -> (256 x6 res) -> up(1,2,2,/2) -> (128 x4 res)`.
  - Encoder stages: `(128 x4, stride 1,1,1) -> (256 x6, stride 1,2,2) -> (512 x4, stride 2,1,1) -> (1024 x2, stride 2,2,2) -> (1024 x2, stride 2,2,2)`.
- Match current `CausalConv3d` padding semantics from LTX-2.3:
  - Temporal padding via replicate of first/last frame behavior.
  - Spatial padding via conv border (`[0, 1, 1]`) with zero padding mode behavior; avoid old reflect-prepad pattern in this path.
- In Swift video decoder parity checks, output may appear batch-squeezed (`[3, D, H, W]`); compare against PyTorch `decoded_video[0]` in that case.
- Keep ckpt export disabled until audio decoder/encoder + vocoder/BWE + video VAE are all validated together.

## 10) LTX-2.3 vocoder / BWE (SnakeBeta + AMP1) lessons

- `s4nnc` requirement for SnakeBeta graph ops:
  - Pin `WORKSPACE` `@s4nnc` to `9092224ec3cb3a260d505882ad55fa1266a16284`
  - Use `shallow_since = "1772754759 -0500"` (timezone matters in this repo workflow).
  - This commit provides `Model.IO` `.pow(_:)`, `.sin()`, `.cos()` used by SnakeBeta.
- SnakeBeta parity in Swift:
  - Keep runtime op simple: `x + beta * sin(x * alpha)^2`.
  - Fold reciprocal into load step:
    - `alpha = exp(alpha_raw)`
    - `beta = (exp(beta_raw) + 1e-9)^-1`
  - This removes runtime `+eps` and `pow(-1)` around beta while preserving parity.
- `Activation1d` parity details:
  - Exact op order: `upsample -> snakebeta -> downsample`.
  - Load anti-alias filters from state dict keys:
    - `...upsample.filter`
    - `...downsample.lowpass.filter`
  - In this codebase, grouped transpose-conv for this path can trigger `ccv_nnc_conv_transpose_tensor_auto_forw` assertion.
  - Reliable workaround: reshape `[1,C,1,W] -> [C,1,1,W]`, run single-channel conv/transposed-conv with shared filter, reshape back.
- PythonKit config access gotcha:
  - `dict["missing_key"]` can hard-fail.
  - Guard optional config reads with `dict.__contains__(key)` before subscripting (or helper wrapper).
- Final activation/bias must follow config (not defaults):
  - Core vocoder (`config.vocoder.vocoder`):
    - `use_bias_at_final=false`
    - `use_tanh_at_final=false`
    - missing `apply_final_activation` => default `true` (therefore final **clamp**, not tanh).
  - BWE generator (`config.vocoder.bwe`):
    - `use_bias_at_final=false`
    - `apply_final_activation=false` (no tanh/clamp).
- `VocoderWithBWE` resampler constants are computed in Python (`UpSample1d(window_type="hann")`):
  - `ratio = output_sampling_rate // input_sampling_rate`
  - `width = ceil(lowpass_filter_width / rolloff)` with `lowpass_filter_width=6`, `rolloff=0.99`
  - `kernel_size = 2 * width * ratio + 1`
  - `pad = width`
  - `pad_left = 2 * width * ratio`
  - `pad_right = kernel_size - ratio`
  - LTX-2.3 values (`input=16000`, `output=48000`): `ratio=3`, `width=7`, `kernel_size=43`, `pad=7`, `pad_left=42`, `pad_right=40`.
- Current parity status in `examples/ltx23/main.swift` early-exit block:
  - Core vocoder: max-abs diff `4.196167e-05`.
  - BWE generator: max-abs diff `5.5576675e-07`.
  - VocoderWithBWE: max-abs diff `0.016445458`.
- Regeneration hygiene:
  - Delete existing sqlite ckpt before re-export (`rm -f ...ckpt`) for clean, deterministic writes.
  - For one-off regen when VAE block is commented out:
    - temporarily uncomment the block,
    - run with early `exit(0)` after export,
    - re-comment it afterward so default flow stays on main model work.

## 11) LTX-2.3 text connectors + feature extractor + main DiT lessons

- Keep these as separate exported units in `examples/ltx23/main.swift`:
  - `text_feature_extractor` (one merged model),
  - `text_video_connector_learnable_registers` (variable),
  - `text_video_connector` (model),
  - `text_audio_connector_learnable_registers` (variable),
  - `text_audio_connector` (model),
  - `dit` (model).
- Text feature extractor structure for 2.3:
  - Load two separate projections from checkpoint:
    - `text_embedding_projection.video_aggregate_embed.*` (4096)
    - `text_embedding_projection.audio_aggregate_embed.*` (2048)
  - Merge into one Swift model by concatenating outputs (`4096 + 2048 = 6144`).
  - Run feature extractor in `BFloat16` for parity (weights/bias copied as BF16).
- Connector architecture changes vs older LTX-2:
  - Use 8 transformer blocks per connector (`layers: 8`), not older shallow setup.
  - Video connector: `k=128`, `h=32`, token length `1024`.
  - Audio connector: `k=64`, `h=32`, token length `1024`.
  - Enable gated attention (`to_gate_logits`) and load these weights/biases; missing this causes clear parity drift.
- RoPE parity strategy used in this repo:
  - Avoid connector RoPE complexity by patching reference `embeddings_connector.py` to pass `pe=None`.
  - In Swift, use fixed rotary tensors with repeating `1,0` pattern.
  - For main DiT reference path, set positions to zeros (`torch.zeros_like(...)`) to match fixed RoPE behavior.
- Main DiT 2.3 deltas to keep:
  - Additional AdaLN branches beyond old LTX-2:
    - `prompt_adaln_single`, `audio_prompt_adaln_single`,
    - `av_ca_video_scale_shift_adaln_single`,
    - `av_ca_audio_scale_shift_adaln_single`,
    - `av_ca_a2v_gate_adaln_single`,
    - `av_ca_v2a_gate_adaln_single`.
  - Keep gated attention paths in transformer blocks and load `to_gate_logits`.
- Context projection behavior:
  - Keep `useContextProjection` as a switch.
  - Guard caption projection loads with `state_dict.__contains__` before access.
  - Distilled path can run with `useContextProjection: false` if those projection tensors are absent/not wanted.
- Parity checks:
  - Check both max-abs and relative diff for connectors/DiT, not just absolute error.
  - Sanity test: disable gating in both implementations; diff should increase significantly. If it does not, suspect an implementation mismatch elsewhere.
- Export validation:
  - After DiT export, compare key counts against original safetensors while excluding audio/video VAE+vocoder payload.
  - Because text projections are merged into one `text_feature_extractor`, key accounting differs from source by 2 tensors.
- Runtime practical tip:
  - During GPU slicing/composed ops in connector path, use `.contiguous()` / `.copied()` when needed to avoid unsupported sliced-tensor behavior in some ops.
