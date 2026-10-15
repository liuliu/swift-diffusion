# Qwen-Image-2.1

Swift/NNC implementation of the released DiT, Qwen3-VL conditioner, and RGBA
VAE. The generation and editing examples contain the inference flow without
parity suites, activation probes, export hooks, or model-specific environment
switches. The examples load all neural-network weights directly from the three
converted checkpoints in `/slow/Data`, using strict model loading. Python handles
tokenization, image preprocessing, initial noise, latent normalization metadata,
and the scheduler; it does not load Hugging Face model weights. Neural-network
inference is Swift.

From the repository root:

```bash
bash examples/qwen_image21/run.sh generate
bash examples/qwen_image21/run.sh edit
```

Run generation first. It creates a red teapot; editing uses that image and asks
for a blue teapot. Each uses 512×512 output and 40 steps. Images are written to
`artifacts/examples/generation/swift.png` and `artifacts/examples/edit/swift.png`.
Each also gets a `.preview.png` composited over white.

The original-VAE checkpoint-backed examples passed all 40 steps. Generation was byte-identical
to the source-weight run; editing differed by at most one 8-bit channel value
(mean absolute difference `0.0000477` on the 0–255 scale). The full-DiT check
before/after splitting modulation had max-abs difference `0`. Results are in
`artifacts/exports/checkpoint_examples.json`.

Prompts, input paths, dimensions, steps, and seeds are ordinary local constants
at the start of `q21Generate()` in `generation.swift` and `q21Edit()` in
`edit.swift`. GPU choices are directly in `main.swift` (Swift GPU 1, Python GPU
0, for seeded noise). Checkpoint paths are also in `main.swift`. The existing
model files remain `dit.swift`, `text.swift`, `vision.swift`, and `vae.swift`,
with architecture constants beside the builders.

The runner uses `/home/liu/workspace/ltx2/LTX-2/_env`, the established Python 3.12
and PyTorch 2.9.1+cu128 environment. Its remaining environment assignments are
required PythonKit/NCCL library settings and TF32 disabling. Run CUDA commands
outside the sandbox. For missing dependencies, activate that environment and
run `python examples/qwen_image21/setup.py` to fetch the pinned model/reference
sources. Builds use the local s4nnc/ccv dependency configured by the repository
`WORKSPACE`, without a repository override. Those original-VAE examples were revalidated
with the local dependencies and produced byte-identical images.

## Implementation details

- DiT: 32 blocks, width 4096, 32×128 heads, SwiGLU width 12288; 64-channel
  unpatched latents. Generation uses FP16 main computation with FP32 residuals.
  Full 40-step profiles at 512px and 1024px required no FFN scaling; peak was
  5736 in block 6. There are no bias tensors. Editing uses the validated FP32 path.
- Language: Qwen3-VL-8B, 36 blocks, width 4096, 32 query / 8 KV heads. It includes
  token embeddings and returns hidden states before final RMSNorm; no `lm_head`
  is needed. All 750 released source tensors match original Qwen3-VL-8B-Instruct.
- Vision: 27 blocks, width 1152, 16×72 heads, with DeepStack from blocks 8/16/24
  injected after language blocks 0/1/2. FP16 **storage** passed with FP32
  computation: maximum weight error `2.98e-8`, minimum vision-token cosine
  `0.9999999965`. This is separate from FP16 computation, which failed the earlier
  real-image parity gate. The edit example computes vision/language in FP32.
- VAE: RGBA ↔ 64-channel latents, 16× spatial compression, single-image path.
  Both examples use madebyollin's October 5 texture/FP16 fix, pinned to
  [`702909b4`](https://huggingface.co/madebyollin/texture-fix-vae-for-qwen-image-2.1/blob/702909b4d408912c7a28fadea06e8b7fdb38ef0c/README.md).
  Its weights already include the author's power-of-two rescaling. Latent
  mean/std are unchanged and remain outside the graphs. Weights, convolution
  outputs, and residuals use FP16; normalization, SiLU, and attention use FP32.
  Temporal-kernel-one convolutions give FP32 accumulation in cuDNN while
  preserving the usual OIHW checkpoint shapes. Vision composites alpha over
  white; the VAE retains all four channels.

The examples load these three exports from `/slow/Data`:

| File | Keys |
| --- | --- |
| `qwen_3_vl_8b_instruct_f16.ckpt` | `text_model`, `vision_model` |
| `qwen_image_2.1_f16.ckpt` | `dit` |
| `qwen_image_2.1_vae_texture_fix_f16.ckpt` | `encoder`, `decoder` |

Their 1,382 tensors were audited against source names, shapes, dtypes, and values.
The new VAE contains 98 encoder and 128 decoder tensors, all FP16, with names
and shapes exactly matching the original export. The original
`qwen_image_2.1_vae_f32.ckpt` remains available.
Language/vision names follow the usual Qwen3-VL convention. DiT attention/FFN
names cover 0–31. Input projections use `x_embedder` and
`context_embedder_0/1`, with `context_norm` and `t_embedder_0/1`. The four shared
Dense projections `x_ada_ln_0`–`3` load consecutive row ranges of the source
modulation weight. The final modulation is separately named `ada_ln_0`.
The output projection is `linear`. The vision export now stores FP16 weights,
which reload into FP32 computation.
The replacement passed strict Swift loading and real-image output comparison
(minimum token cosine `0.99999999949`); language tensor bytes are unchanged.
Checkpoints are not rewritten by the examples.

`full_schedule_validation.json`, `vision_storage_validation.json`, and the other
JSON reports preserve local numerical evidence and are excluded from version control. Detailed export checksums and tensor
mappings remain in `artifacts/exports`. `export_summary.json` records the original export
checksums; `vision_f16_checkpoint.json` and `vision_f16_reload.json` record the
vision storage update after the original FP32 export audit. `dit_naming_parity.json`
and `dit_naming_checkpoint.json` record the split modulation and name update.
`q8p_name_comparison.json` confirms all 397 text names exactly match the existing
Q8P. Retired validation/export source is saved
under ignored `artifacts/cleanup_archive`, outside the active implementation.

The texture-fix conversion passed FP32 and FP16 encoder/decoder parity on seeded
square and rectangular inputs. FP32 maximum absolute errors were below `1.87e-4`.
Against PyTorch FP16, the 1024px transparent dragon and 2048px detailed meadow
had mean per-pixel channel cosine above `0.9999999` and normalized RMS error below
`0.000431`; all outputs were finite. The encoder also passed on an RGBA image and
a fully transparent synthetic image. The largest measured Swift FP16 activation
was `52.53125` in `decoder.up_blocks.2.resnets.2.norm2` on the 2048px fixture.
Reports, source revision, source weights,
checkpoint audit, and preview images are in ignored `artifacts/texture_fix_vae`.
Both checkpoint-backed 40-step generation and editing examples passed with the
new VAE and local s4nnc/ccv. `conversion_summary.json` in that directory records
the current checkpoint checksum, parity, activation peaks, and example outputs.
Checkpoint loads are explicitly read-only.
The active examples contain no conversion or validation hooks.
