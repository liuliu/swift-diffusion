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

- Always run Swift and PyTorch on the same seeded random latent.
- Confirm output shape equality first.
- Compare sample values (and ideally full max-abs diff) before writing ckpt.
- Only export graph (`graph.openStore(...).write(...)`) after numeric parity is confirmed.

## 7) Iteration pattern in `examples/ltx2/main.swift`

- Add new converter block near top.
- Run only that block with `exit(0)` immediately after it.
- Keep older experiments below for reference, but not on active execution path.
