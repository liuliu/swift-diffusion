# Qwen3-VL vision / DeepStack export

The H3 text checkpoint intentionally retains the embedding and first 50 language
blocks: H3 consumes `hidden_states[50]`, before the final language-model norm.
Adding vision does not require changing these text weights.

The combined export is a new file:

`/slow/Data/minimax_h3_qwen_3_vl_with_vision_f16.ckpt`

- `text_model`: all 551 tensors copied from the existing
  `minimax_h3_qwen_3_vl_f16.ckpt`, verified byte-for-byte including tensor metadata.
- `vision_model`: 459 FP16 tensors covering all 351 source visual tensors. Splitting
  each of the 27 QKV weight/bias pairs into Q/K/V accounts for the extra 108 tensors.
  This includes the final patch merger and all three DeepStack mergers.

The original checkpoint is never modified. Export first copies it into a temporary
SQLite store, validates and writes the vision model, strictly reloads it, checks
every stored vision tensor, and compares every text tensor with the original.
Only a fully verified file is published, without replacing an existing destination.

## Vision model contract

Model definitions and positional helpers live in `../minimax_h3/qwen_vision.swift`
and have no Python dependency. Architecture constants are explicit on the Swift
side. The implementation follows the existing HiDream-O1 Qwen3-VL vision converter,
with DeepStack and per-frame packed attention added.

For `N = sum(t * h * w)` patch tokens, inputs are:

1. Processor patches `[N, 1536]` (`3 * 2 * 16 * 16`), FP16.
2. Four-corner learned-position indices `[4 * N]`, Int32.
3. Bilinear interpolation weights `[4 * N]`, FP32.
4. Interleaved spatial RoPE `[1, N, 1, 72]`, FP32.

Patches use the processor's 2x2 spatial-merge ordering. Each temporal slice attends
only within itself, including packed images/videos with different spatial sizes.
The patch Conv3D is stored as an equivalent flattened dense projection.

All four outputs have shape `[N / 4, 5120]`, in this order:

1. Final image embeddings (main merger after vision block 26).
2. DeepStack 0 (after vision block 8).
3. DeepStack 1 (after vision block 16).
4. DeepStack 2 (after vision block 24).

The main merger normalizes 1152 features before the 2x2 merge. DeepStack mergers
normalize 4608 features after the merge. Block MLPs use tanh-approximate GELU;
mergers use exact GELU, matching the reference.

An eventual multimodal language path must replace visual placeholder embeddings
with output 0 and add outputs 1/2/3 at the visual positions after language blocks
0/1/2 respectively. This export does not itself enable I2V/reference generation
in the current T2VA-only runtime.

## Naming

Shared tower names align with `examples/qwen_converter` and the existing
`qwen_2.5_vl_7b_vit_f16.ckpt`: `conv_in`, `q_proj`, `k_proj`, `v_proj`, `out_proj`,
`norm1`, `norm2`, `norm_out`, `merger_mlp_0`, and `merger_mlp_1`.
Block indices use the standard repeated-layer suffixes. Qwen3-VL-specific names
are `pos_embed`, `mlp_fc1` / `mlp_fc2` (its MLP is not gated), and
`deepstack_0/1/2_norm` / `deepstack_0/1/2_mlp_0/1`.

## Validation / execution

Run the approved no-argument `../minimax_h3/minimax_h3_check.sh` outside the sandbox.
It supplies the PythonKit/NCCL environment and builds the converter. Select
`parity-qwen-vision` in that script for these checks (the script also serves other
H3 conversions). After parity review and export approval, use mode
`export-qwen-vision`, which reruns the entire suite before exporting.

The official reference is imported from local Transformers and reads only the
downloaded visual weights; no model downloads are performed. Tests cover grids
`[(1, 8, 8)]`, `[(2, 8, 12), (1, 6, 10)]`, and `[(1, 28, 28)]`. They compare the
stem, sampled full-tower residuals, and every merger in FP32 and FP16, and also
compare FP16 outputs against the FP32 baseline on identical input values.

Metrics include max absolute difference, max-abs error divided by max reference
magnitude, NRMSE, and minimum/mean token cosine similarity. FP16 checks cover both
weight casting and complete forward execution with FP16 SDPA. The export also
scans the inherited text weights for non-finite FP16 values while comparing bytes.

FP16 public-output gates use the existing Qwen export criteria: max-relative error
below 0.02, minimum token cosine above 0.99, and mean token cosine above 0.999,
plus NRMSE below 0.02. They apply both against Python FP16 and the FP32 baseline.
Pre-merger residual maxima are reported but not used as output gates: block 26
has approximately 10k-magnitude outliers before the final merger normalization.
Intermediate tensors must still be finite and pass cosine/NRMSE checks. FP32
parity remains stricter (max-relative and NRMSE below 0.001, minimum cosine above
0.999) for both public outputs and intermediates.

Environment overrides: `MINIMAX_H3_DEVICE`, `MINIMAX_H3_MODEL`,
`MINIMAX_H3_QWEN_EXPORT_PATH` (original text file), and
`MINIMAX_H3_QWEN_VISION_EXPORT_PATH` (new combined destination).

## Measured parity (2026-09-07)

Worst values over all three grids, compared with the official FP32 baseline:

| Output | FP32 max-relative | FP16 max-relative | FP16 max-abs | FP16 minimum token cosine | FP16 NRMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Final merger | 4.89e-6 | 0.012086 | 0.269690 | 0.998332 | 0.007026 |
| DeepStack 0 | 1.08e-6 | 0.002764 | 0.018652 | 0.999948 | 0.002515 |
| DeepStack 1 | 4.73e-6 | 0.006662 | 0.070606 | 0.999882 | 0.003993 |
| DeepStack 2 | 7.68e-6 | 0.009990 | 0.026449 | 0.999818 | 0.005934 |

FP32 token cosines report 1.0. FP16 mean token cosines versus FP32 are at least
0.999958. Against Python FP16, Swift's worst output max-relative error is 0.010223,
minimum token cosine 0.997987, and mean cosine 0.999951. Python FP16 itself has a
worst final-merger max-relative error of 0.007392 versus FP32, with minimum cosine
0.999704. Thus the FP16 forward is approximate, with somewhat more rounding error
than the Python FP16 implementation; this is distinct from weight-cast safety.

All tested outputs and intermediate residuals are finite. All 351 source tensors
(595,266,800 elements) cast to FP16 without NaN, infinity, or overflow. Maximum
weight magnitude is 25; maximum cast error is 2.98e-8. There are 22,460 nonzero
source values that underflow to zero on cast.

Combined export completed and verified: 1,010 tensors total. All 459 vision
tensors, including all 18 DeepStack tensors, are stored as FP16 and finite.
Strict vision-model reload reproduces all four outputs exactly (zero max-abs
difference). There are 276 exact shared tensor-name suffixes with the Qwen-Image
vision export.

All 551 text tensors and their metadata are byte-identical to the original;
all 25,157,829,120 text elements are finite FP16 (maximum magnitude 20.75).
The original checkpoint's size, modification time, and inode are unchanged.
The new file is 51,620,843,520 bytes. Export-session log:
`/tmp/minimax_h3_qwen_vision.S1kLSu/run.log`.

## Pika H3 Flex merged FP16 exports

The release at `Pika-Labs/Pika-H3-Flex`, revision
`0ddf55ad7067ca1e76812253bd7b4600df87f759`, supplies separate FL2VA and Ref2VA
rank-128 adapters. Downloaded sources and adapter files live under
`/slow/Data/Pika-H3-Flex`. `scripts/download_pika_h3_flex.py` pins the release and
checks adapter SHA256 hashes.

`pika_h3_flex.swift` reuses `H3JointTransformer` and its existing reader. The
reference helper reconstructs source tensors from the existing lossless FP32 H3
exports; this reconstruction was verified against every byte of all 14 official
base safetensors shards via their SHA256 hashes. No duplicate base download is
needed. Each adapter is merged as `base + B @ A` in FP32, before Q/K permutation
and FFN splitting, then rounded to nearest FP16. Runtime activation scales are
not baked into the weights.

Destinations, both with model key `dit` and the existing 2,392 H3 tensor names:

- `/slow/Data/pika_h3_flex_dit_f16.ckpt` (FL2VA, including T2VA).
- `/slow/Data/pika_h3_flex_ref2va_dit_f16.ckpt` (Ref2VA).

Every tensor is stored in FP16, including the parameters originally published in
FP32. FP16-versus-FP32 storage differences were measured and accepted; there are
no mixed-precision storage exceptions. Shared text/vision encoders and VAEs use
the existing H3 checkpoints.

Set `PIKA_H3_TASK=fl2va` or `ref2va` and use converter mode
`parity-pika-h3-flex`. After export approval, mode `export-pika-h3-flex-f16`
reruns the same parity check before writing. Validation covers:

- All 312 adapted matrices: merged versus separate LoRA application.
- Every source tensor: finite values, FP16 overflow, rounding, and coverage.
- Full 50-layer Swift/Python outputs on identical seeded inputs and FP16-rounded
  weights, using FP32 arithmetic to isolate conversion correctness. Metrics are
  max-abs, max-abs divided by max reference magnitude, NRMSE, and token cosine.
- Strict checkpoint reload, requiring exactly unchanged outputs.
- All saved names, shapes, FP16 dtypes, and expected merged-tensor SHA256 hashes,
  plus SQLite integrity. An export is staged as `.partial` until these pass.

Each checkpoint gets an `.inference.json` sidecar preserving the source identity,
base verification, merge policy, and official sampling manifest. Pika uses six
renoising steps with knots `[1000, 750, 500, 250, 100, 40, 0]`, video shift 12,
and audio shift 3; it is not a generic six-step Euler schedule. These conversion
checks do not constitute an end-to-end generation quality assessment or an
FP16-activation runtime parity test.

Measured Swift/Python conversion parity (FP16-rounded weights, FP32 arithmetic):

| Variant / output | Max-abs / max-ref | Max-abs | NRMSE | Minimum token cosine |
| --- | ---: | ---: | ---: | ---: |
| FL2VA video | 8.018496e-6 | 3.862381e-5 | 7.609412e-6 | 1.0 |
| FL2VA audio | 2.819487e-5 | 4.744530e-5 | 2.917250e-5 | 1.0 |
| Ref2VA video | 6.256023e-6 | 2.646446e-5 | 5.074113e-6 | 1.0 |
| Ref2VA audio | 1.374066e-5 | 3.334880e-5 | 1.529465e-5 | 1.0 |

Token cosine values above are reported at the comparison helper's FP32 precision.
Export-session log: `/tmp/pika_h3_flex.ah5DT0/run.log`.

Both exports completed successfully, each 66,393,960,448 bytes. Strict reload
reproduced both output streams exactly (zero max-abs difference). All 2,392
tensors in each file passed the FP16 dtype, shape, name, and exact-value hash
checks; both SQLite integrity checks passed. The no-argument helper script now
defaults to parity-only mode.

## MiniMax H3 Image (2026-11-11)

Pinned source: `MiniMaxAI/MiniMax-H3-Image`, revision
`492fd8c4eca666f1085f57e30738c1e0d86b3439`. Downloads live under
`/slow/Data/MiniMax-H3-Image`; `scripts/download_minimax_h3_image.py` downloads
the selected Diffusers-format component and verifies every LFS SHA256. The
official reference model source is pinned to Diffusers
`8069682140c675652a886736121f4e3b86f4187c` and loaded unchanged by
`h3_image_reference.py`.

### VAE validation

`parity-h3-image-vae` reuses `H3ConditioningVideoEncoder` for a single frame.
The existing transformer decoder now accepts explicit architecture constants:
the image variant has 24 layers, width 1024, 16 heads, 4096 FFN channels,
16x16 spatial patches, temporal patch 1, and 2D RoPE over 32 of 64 head
channels. Original video defaults are preserved. Latent means/stds stay outside
the models; the encoder returns all 48 mean/log-variance channels.

Both implementations run FP32. Set `NVIDIA_TF32_OVERRIDE=0` before launching:
PyTorch's TF32 flags alone do not control ccv/cuDNN convolution precision.
The PythonKit path setup preserves venv package precedence over system Python
packages (in particular, the Python 3.12 Pillow extension).

Passed square 64x64, rectangular 64x96, and full 256x256-tile cases, using the
same seeded pixels/latents. Worst metrics across these cases:

| Comparison | Max-abs / max-ref | Max-abs | NRMSE | Minimum token cosine |
| --- | ---: | ---: | ---: | ---: |
| Encoder moments | 9.20709e-5 | 9.2691183e-4 | 7.2648063e-6 | 1.0 |
| Decoder, random latent | 2.170271e-6 | 1.1980534e-5 | 9.861794e-7 | 1.0 |
| Decoder, identical Swift-encoded latent | 9.732442e-6 | 2.3007393e-5 | 3.2433702e-6 | 1.0 |
| Composed encoder to decoder | 0.0018096943 | 0.0044582523 | 0.00017910544 | 0.99999744 |

Cosines are reported at the comparison helper's FP32 precision. The composed
test's localized difference is reproduced using only Python's decoder on the
two encoder outputs: max-relative 0.0018108058, max-abs 0.0044609904, NRMSE
0.00017913165. This isolates propagation of the small encoder difference from
decoder conversion error. The composition max-relative gate is 0.005; its
NRMSE < 0.001 and cosine > 0.99999 gates remain unchanged. Isolated models and
the identical-latent decoder use max-relative < 0.001.

All 511 source tensors are consumed. The approved export is now available at
`/slow/Data/minimax_h3_image_vae_f32.ckpt`, with `video_encoder` (118 tensors)
and `video_decoder` (441 tensors). All 559 stored tensors are finite FP32 and
account for all 584,228,408 source parameters; the 48 extra tensors come from
splitting the 24 SwiGLU input weight/bias pairs. Both models strictly reload
with exactly zero output difference. SQLite integrity and the complete stored
name set passed. The `.ckpt.json` sidecar records the source revision, VAE
config (including external latent statistics), and audit results.

The export session reran all three parity cases before writing. Log:
`/tmp/minimax_h3_image.BkwRMG/run.log`. Converter mode: `export-h3-image-vae`.
Existing destinations and partial exports are never overwritten. FP16
execution, full-canvas tiling, and generation quality are not covered by this
FP32 per-tile parity result.

### DiT FP32 validation

`H3JointTransformer(includeAudio: false, pddSteps: 6)` omits audio inputs,
projections, and outputs while preserving the three-modality AdaLN table and
existing parameter names. The six source PDD projections are concatenated in
head-major order into `proj_out`; they are already fused, not base-plus-delta.

`parity-h3-image-dit` is the new validation entry point, requiring
`H3_FP32_ATTENTION_CONTROL=1`. Its reference harness calls the unchanged official
forward with per-module GPU weight-streaming hooks. Tests passed for T2I
and reference-image layouts, with isolated timestep/refiner/first-block checks,
all 50 transformer layers, and each of the six output heads. All 12 source
shards have downloaded and passed SHA256 verification. After root-disk space
was restored, the full converter build passed. All 644 source tensors are
consumed in both tests, all Swift parameter names are unique, and the image
graph contains no audio projections. The approved helper defaults to DiT
parity only; `export-h3-image-dit` performs the approved export workflow.

Worst individual-head metrics over all six heads, using identical seeded input
rows and full FP32 arithmetic, including attention:

| Layout | Max-abs / max-ref | Max-abs | NRMSE | Minimum token cosine |
| --- | ---: | ---: | ---: | ---: |
| Text to image | 4.4761887e-6 | 2.5868416e-5 | 2.9752418e-6 | 1.0 |
| Reference images | 8.1510425e-6 | 6.681681e-5 | 8.899163e-6 | 1.0 |

The reference layout exercises image-modality vision/text rows, fixed-reference
rows at timestep 0.999, and target rows at a different timestep. The isolated
timestep embedder, full token refiner, and first DiT block all pass with
max-relative error below 4.1e-7. Final-norm max-relative error is 4.8733777e-6
(T2I) and 5.60948e-6 (references); large pre-norm residuals remain in FP32 and
are also compared. Cosines are reported at FP32 precision.

Successful log: `/tmp/minimax_h3_image.pevFYj/run.log`. This is conversion parity
on small packed sequences, not an end-to-end image quality assessment or an
FP16 activation/scaling validation.

The reference environment was updated to the pinned Diffusers revision
`8069682140c675652a886736121f4e3b86f4187c` (0.42.0.dev0) and its required Hub
package (`huggingface-hub==1.33.0`). Torch/CUDA were not changed. Download/build
cache for that Python upgrade lives under `/slow/Data/MiniMax-H3-Image/.uv-cache`.
The normal helper does not install or upgrade packages; it checks the official
DiT import before launching the converter.

### DiT export completed (2026-11-13)

`/slow/Data/minimax_h3_image_dit_f32.ckpt` contains model key `dit`, with all
2,388 tensors stored in FP32 (33,125,224,384 parameters). No runtime scaling is
baked into the weights. Parameter names and shapes match the original H3 export
after removing the four audio projection tensors and widening `proj_out` to six
head-major 96-channel outputs. The three-modality AdaLN table is retained.

The export session reran both full 50-layer layouts and all six heads, with the
same parity metrics above. Strict checkpoint reload reproduced all six output
heads exactly (zero max-abs difference). An independent audit reused the
existing H3 name/permutation mapping to verify every stored tensor's shape,
FP32 dtype, and exact source-derived SHA256, including all split FFN/AdaLN
weights and concatenated PDD heads. All values are finite; SQLite integrity
passed. The staged file was published only after these checks.

The `.ckpt.inference.json` sidecar records source revisions and hashes, model
config, the six-step PDD schedule, output-head layout, and audit results.
Export log: `/tmp/minimax_h3_image.kWhKpO/run.log`. Existing checkpoints are
never overwritten, and export mode refuses to overwrite a leftover partial
file. FP16 runtime and end-to-end generation quality remain outside this
export validation's scope.

### H3-Image DiT FP16 storage

`parity-h3-image-dit-f16` checks the same graph using all source weights rounded
to FP16, with FP32 arithmetic on both sides. It separately reports each PDD
head's rounding impact against the native-weight official FP32 baseline, for
both T2I and reference layouts. Every source tensor is checked for finite values
before and after casting; the summary records the maximum weight magnitude,
maximum rounding error, and number of nonzero values rounded to zero.

After parity and export approval, `export-h3-image-dit-f16` writes
`/slow/Data/minimax_h3_image_dit_f16.ckpt`, under the same `dit` key. It uses the
Pika streaming FP16 writer, with no intermediate FP32 file and no runtime scales
baked into weights. All 2,388 tensors, including norms, timestep embedding, and
all six output heads, are stored in FP16. The writer does not remove other exports.
Strict reload and exact source-derived FP16 byte/shape/name checks gate
publication. This validates FP16 **storage**, not FP16 activation execution or
end-to-end image quality. The helper defaults to parity-only mode.

Validation passed on November 16, 2026, including isolated timestep/refiner/
first-block checks and both complete 50-layer layouts. Worst individual-head
metrics across all six PDD heads:

| Comparison | Layout | Max-abs / max-ref | Max-abs | NRMSE | Minimum token cosine |
| --- | --- | ---: | ---: | ---: | ---: |
| Swift vs Python, identical FP16-rounded weights | T2I | 2.2200632e-6 | 1.2636185e-5 | 2.2477102e-6 | 1.0 |
| Swift vs Python, identical FP16-rounded weights | Reference | 1.1774419e-5 | 9.649992e-5 | 1.0010136e-5 | 1.0 |
| Official FP16-rounded vs native weights | T2I | 0.00040074685 | 0.002280891 | 0.00042103074 | 0.9999999 |
| Official FP16-rounded vs native weights | Reference | 0.002733284 | 0.022331461 | 0.0032309315 | 0.99997723 |

All 644 source tensors (33,125,224,384 values) remain finite when cast to FP16.
Maximum weight magnitude is 55.5; maximum weight rounding error is
0.00363445282; 8,193 nonzero values round to zero. The safety helper also passed
small tests for underflow accounting, exact FP16 hashes, native-baseline
switching, and rejection of overflow/NaN/Inf. Build, foundation self-test,
Python syntax, shell syntax, and whitespace checks passed.

Parity log: `/tmp/minimax_h3_image.hIeVt6/run.log`.

The approved FP16 export completed on November 16, 2026:
`/slow/Data/minimax_h3_image_dit_f16.ckpt` (66,398,429,184 bytes), model key `dit`.
The same-session full parity checks reproduced the metrics above. Strict
reload reproduced all six heads exactly (zero max-abs difference). Every one
of the 2,388 stored tensors passed exact source-derived FP16 hash, shape, and
name verification; all values are finite, and SQLite integrity passed.
The sidecar records the model/source identity, PDD schedule, FP16 safety
statistics, and audit results. No runtime scaling is baked into the weights.
Export log: `/tmp/minimax_h3_image.OjPVef/run.log`.

After the FP16 checkpoint was verified and published, the user-approved
`minimax_h3_image_dit_f32.ckpt` and its `.inference.json` sidecar were removed
(132,796,300,658 bytes total). The original source shards remain available
under `/slow/Data/MiniMax-H3-Image/transformer_diffusers` for regeneration.
The FP32 image VAE and original H3 checkpoints were not changed. The helper
was restored to non-mutating FP16-storage parity mode; it contains no cleanup
or automatic export command.

Original H3 DiT FP16 provenance: the October 20 conversation recorded that its
FP16 files appeared to have been converted separately. The original converter
exported FP32; FP16 runtime/scaling and Qwen FP16 exports were implemented in
this thread. No original DiT FP16-writing invocation was located. On November
16, all 2,392 stored tensors were confirmed FP16, and sampled input/context,
timestep, layer-49 Q-norm, and layer-49 FFN-down weights exactly matched plain
casts of the corresponding FP32 export. This does not establish who ran that
conversion, nor does the sampling establish byte equality for every tensor.
