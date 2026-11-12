"""Pinned upstream Fibo 2 / Qwen3-VL parity reference and export validation."""
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "artifacts/python"))
import numpy as np
import torch
from safetensors import safe_open
from torch.nn.attention import SDPBackend, sdpa_kernel
from diffusers import AutoencoderKLFlux2, BriaFibo2Transformer2DModel, BriaFibo2Pipeline, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.bria_fibo2.pipeline_bria_fibo2 import (
    TEXT_ENCODER_OUT_LAYERS, PROMPT_TEMPLATE, DEFAULT_NEGATIVE_PROMPT)
from diffusers.models.transformers.transformer_bria_fibo2 import compute_gist_count
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

VARIANT = os.environ.get("FIBO_VARIANT", "dev")
assert VARIANT in ("dev", "turbo")
MODEL_REVISION = {"dev": "72415bd021af33b2cbc63525683a114ff8ac294e",
                  "turbo": "9a24ee6810a782dd6aad4639fadcf48db0957ef4"}[VARIANT]
DIFFUSERS_REVISION = "d0b9b1b5a540e5f9df2f8ec236347bf9c5aa7d1f"
DIT_LAYOUT = "draw_things_split_adaln_v2"
ROOT = Path("/slow/Data") / ("Fibo-2.0-dev" if VARIANT == "dev" else "Fibo-2.0-edit-turbo")
assert json.loads((ROOT / "DOWNLOAD_MANIFEST.json").read_text())["revision"] == MODEL_REVISION
ARTIFACTS = HERE / "artifacts" / VARIANT
ARTIFACTS.mkdir(parents=True, exist_ok=True)
torch.set_grad_enabled(False)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
REPORT = []
MODELS = {}


def array(x):
    return x.detach().float().cpu().contiguous().numpy()


def release():
    MODELS.clear()
    gc.collect()
    torch.cuda.empty_cache()


class State:
    def __init__(self, component):
        self.root = ROOT / component
        index = self.root / "diffusion_pytorch_model.safetensors.index.json"
        if index.exists():
            self.index = json.loads(index.read_text())["weight_map"]
        else:
            filename = "model.safetensors" if component == "text_encoder" else "diffusion_pytorch_model.safetensors"
            with safe_open(self.root / filename, framework="pt") as f:
                self.index = {k: filename for k in f.keys()}

    def __getitem__(self, key):
        with safe_open(self.root / self.index[key], framework="pt") as f:
            return f.get_tensor(key)

    def keys(self):
        return self.index.keys()


def vae_key(key):
    key = key.replace("encoder.quant_conv", "quant_conv").replace("decoder.post_quant_conv", "post_quant_conv")
    key = key.replace(".norm_out.", ".conv_norm_out.").replace(".nin_shortcut.", ".conv_shortcut.")
    key = re.sub(r"encoder.down.(\d+).block.(\d+).", r"encoder.down_blocks.\1.resnets.\2.", key)
    key = re.sub(r"encoder.down.(\d+).downsample.", r"encoder.down_blocks.\1.downsamplers.0.", key)
    key = re.sub(r"decoder.up.(\d+).block.(\d+).",
                 lambda m: f"decoder.up_blocks.{3-int(m[1])}.resnets.{m[2]}.", key)
    key = re.sub(r"decoder.up.(\d+).upsample.",
                 lambda m: f"decoder.up_blocks.{3-int(m[1])}.upsamplers.0.", key)
    key = key.replace(".mid.block_1.", ".mid_block.resnets.0.").replace(".mid.block_2.", ".mid_block.resnets.1.")
    key = key.replace(".mid.attn_1.", ".mid_block.attentions.0.")
    if ".attentions." in key:
        for a, b in [(".norm.", ".group_norm."), (".q.", ".to_q."), (".k.", ".to_k."),
                     (".v.", ".to_v."), (".proj_out.", ".to_out.0.")]:
            key = key.replace(a, b)
    return key


def weight(state, key, transform="identity", heads=0, vae=False, row_start=0, row_count=0):
    x = state[vae_key(key) if vae else key].float()
    if row_count:
        x = x.narrow(0, int(row_start), int(row_count))
    if transform == "qk":
        x = x.reshape(int(heads), 2, 64, -1).transpose(1, 2).reshape(x.shape)
    elif transform == "qk_norm":
        x = x.reshape(2, 64).transpose(0, 1).reshape(x.shape)
    if vae and ".attn_1." in key and x.ndim == 2:
        x = x[:, :, None, None]
    return array(x)


def vision_weight(state, prefix, kind, bias=False):
    x = state[prefix + (".bias" if bias else ".weight")].float()
    if kind == "patch" and not bias:
        x = x.flatten(1)
    if kind in ("query", "key", "value"):
        x = x.narrow(0, ["query", "key", "value"].index(kind) * 1024, 1024)
        if kind != "value":
            x = x.reshape(16, 2, 32, -1).transpose(1, 2).reshape(x.shape)
    return array(x)


def check_keys(state, keys, component):
    keys = set(keys)
    if component == "vae":
        keys = {vae_key(k) for k in keys}
        expected = set(state.keys()) - {"bn.running_mean", "bn.running_var", "bn.num_batches_tracked"}
    elif component == "text":
        expected = {k for k in state.keys() if k.startswith("model.language_model.layers.")
                    and int(k.split(".")[3]) < 35}
        expected.add("model.language_model.embed_tokens.weight")
    elif component == "vision":
        expected = {k for k in state.keys() if k.startswith("model.visual.")}
    else:
        expected = set(state.keys())
    assert keys == expected, (component, sorted(expected - keys), sorted(keys - expected))
    print(f"{component}: all {len(keys)} required tensors accounted for", flush=True)


def check_dit_bindings(state, keys, starts, counts):
    from collections import defaultdict
    keys = list(map(str, keys))
    check_keys(state, keys, "dit")
    slices = defaultdict(list)
    for key, start, count in zip(keys, starts, counts, strict=True):
        slices[key].append((int(start), int(count)))
    split_keys = {f"{prefix}.{i}.adaLN_modulation.0.{suffix}"
                  for prefix, layers in (("noise_refiner", 2), ("layers", 44))
                  for i in range(layers) for suffix in ("weight", "bias")}
    for key, ranges in slices.items():
        expected = [(i * 3584, 3584) for i in range(4)] if key in split_keys else [(0, 0)]
        assert sorted(ranges) == expected, (key, ranges)
    assert len(keys) == 1059 and len(slices) == 783
    print("Verified 46 AdaLN projections split into four complete weight/bias slices each", flush=True)


def metrics(label, actual, expected, tolerance=0.001):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
    if not (np.isfinite(actual).all() and np.isfinite(expected).all()):
        row = dict(label=label, shape=list(actual.shape), passed=False,
                   actual_nonfinite=int((~np.isfinite(actual)).sum()),
                   reference_nonfinite=int((~np.isfinite(expected)).sum()))
        REPORT.append(row)
        print(json.dumps(row), flush=True)
        (ARTIFACTS / "failure.json").write_text(json.dumps(REPORT, indent=2) + "\n")
        return False
    a, b = actual.astype(np.float64).ravel(), expected.astype(np.float64).ravel()
    delta = a - b
    max_abs = float(np.abs(delta).max())
    max_rel = max_abs / max(float(np.abs(b).max()), 1e-12)
    relative_l2 = float(np.linalg.norm(delta) / max(np.linalg.norm(b), 1e-12))
    cosine = float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))
    row = dict(label=label, shape=list(actual.shape), max_abs=max_abs, max_relative=max_rel,
               relative_l2=relative_l2, cosine=cosine, tolerance=float(tolerance),
               actual_sample=a[:8].tolist(), reference_sample=b[:8].tolist(),
               passed=max_rel <= tolerance and relative_l2 <= tolerance)
    REPORT.append(row)
    print(json.dumps(row), flush=True)
    if not row["passed"]:
        (ARTIFACTS / "failure.json").write_text(json.dumps(REPORT, indent=2) + "\n")
    return row["passed"]


def finish(name, reference_dtype=None):
    assert REPORT, "No parity results"
    passed = all(r["passed"] for r in REPORT)
    selected_case = os.environ.get("FIBO_CASE")
    if selected_case is not None:
        name += "_case" + selected_case
    report = dict(variant=VARIANT, selected_case=selected_case, passed=passed,
                  dit_layout=DIT_LAYOUT,
                  compute_dtype=os.environ.get("FIBO_DTYPE", "float32"),
                  reference_dtype=reference_dtype or os.environ.get("FIBO_REFERENCE_DTYPE",
                      "float32" if os.environ.get("FIBO_DTYPE") == "float16" else os.environ.get("FIBO_DTYPE", "float32")),
                  model_revision=MODEL_REVISION, diffusers_revision=DIFFUSERS_REVISION,
                  versions={n: importlib.metadata.version(n) for n in
                            ["torch", "diffusers", "transformers", "huggingface_hub", "tokenizers"]},
                  results=REPORT)
    path = ARTIFACTS / (name + ".json")
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("PARITY PASSED:" if passed else "PARITY FAILED:", path, flush=True)
    return passed


def check_dit_store(path, written_names):
    import math
    import sqlite3
    names = list(map(str, written_names))
    state = State("transformer")
    assert len(state.keys()) == 783
    assert len(names) == len(set(names)) == 1059
    expected = set()
    def add(stem, count=1, bias=False):
        expected.update(f"__dit__[t-{stem}-{i}-{parameter}]"
                        for i in range(count) for parameter in range(2 if bias else 1))
    for prefix, layers in (("", 44), ("noise_refiner_", 2)):
        for stem in ("q", "k", "v", "o", "norm_q", "norm_k", "attention_norm2",
                     "ffn_norm1", "ffn_norm2", "ffn_gate_proj", "ffn_up_proj", "ffn_down_proj"):
            add(prefix + stem, layers)
        add(prefix + ("attention_norm_1" if prefix else "attention_norm1"), layers)
        for i in range(4):
            add(prefix + f"ada_ln_{i}", layers, bias=True)
    for stem in ("cross_q", "cross_k", "cross_v", "cross_o", "cross_norm_q", "cross_norm_k",
                 "cross_attn_norm1", "cross_attn_norm2", "cross_attn_gate"):
        add(stem, 5)
    for stem in ("perceiver_cross_attn_norm", "perceiver_self_attn_norm", "perceiver_ffn_norm",
                 "perceiver_ffn_gate_proj", "perceiver_ffn_up_proj", "perceiver_ffn_down_proj"):
        add(stem, 2)
    for prefix in ("perceiver", "perceiver_cross"):
        for stem in ("q", "k", "v", "o", "norm_q", "norm_k"):
            add(prefix + "_" + stem, 2)
    for stem in ("t_embedder_0", "t_embedder_1", "x_embedder", "ada_ln_final", "linear_final"):
        add(stem, bias=True)
    for stem in ("perceiver_queries", "perceiver_norm_out"):
        add(stem)
    assert set(names) == expected, (sorted(expected - set(names)), sorted(set(names) - expected))
    elements = 0
    for filename in set(state.index.values()):
        with safe_open(state.root / filename, framework="pt") as f:
            elements += sum(math.prod(f.get_slice(key).get_shape()) for key in f.keys())
    with sqlite3.connect(Path(str(path)).resolve().as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute("SELECT name, datatype, length(data) FROM tensors").fetchall()
        assert {row[0] for row in rows} == set(names)
        assert len(rows) == len(names)
        assert all(row[1] == 131072 for row in rows), "Checkpoint must contain only FP16 tensors"
        for name, dim in db.execute("SELECT name,dim FROM tensors"):
            if re.fullmatch(r"__dit__\[t-(noise_refiner_)?ada_ln_[0-3]-\d+-[01]\]", name):
                shape = tuple(int(n) for n in np.frombuffer(dim, dtype=np.int32) if n)
                assert shape == ((3584, 256) if name.endswith("-0]") else (3584,)), (name, shape)
        data_bytes = sum(row[2] for row in rows)
        assert data_bytes == elements * 2, (data_bytes, elements * 2)
        assert db.execute("PRAGMA quick_check").fetchall() == [("ok",)]
    inventory = dict(tensors=len(names), dtype="float16", dit_layout=DIT_LAYOUT, parameters=elements,
                     tensor_bytes=data_bytes, sqlite_quick_check="ok", names=sorted(names))
    (ARTIFACTS / "dit_parameter_names.json").write_text(json.dumps(inventory, indent=2) + "\n")
    print(f"Verified {len(names)} FP16 checkpoint tensors, {elements} parameters", flush=True)
    return inventory


def record_dit_export(path, inventory):
    import hashlib
    from datetime import datetime, timezone
    path = Path(str(path)).resolve()
    report = json.loads((ARTIFACTS / "export_dit_float16.json").read_text())
    assert report["passed"] and len(report["results"]) == 209
    assert report["results"][-1]["label"] == "checkpoint/dit/reload"
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    manifest = dict(variant=VARIANT, component="dit", checkpoint=str(path),
                    dit_layout=DIT_LAYOUT, source_tensors=783, split_adaln_blocks=46,
                    model_revision=MODEL_REVISION, diffusers_revision=DIFFUSERS_REVISION,
                    created_utc=datetime.now(timezone.utc).isoformat(), bytes=path.stat().st_size,
                    sha256=digest, dtype="float16", store_key="dit",
                    tensors=inventory["tensors"], parameters=inventory["parameters"],
                    tensor_bytes=inventory["tensor_bytes"], sqlite_quick_check="ok",
                    parity_checks=208, reload=report["results"][-1],
                    scope="DiT only; text/vision encoder and VAE are separate components.",
                    full_report=str((ARTIFACTS / "export_dit_float16.json").relative_to(HERE)))
    encoded = json.dumps(manifest, indent=2) + "\n"
    Path(str(path) + ".json").write_text(encoded)
    (ARTIFACTS / "export_manifest.json").write_text(encoded)
    print("Checkpoint SHA256:", digest, flush=True)


def text_case(index):
    generator = torch.Generator().manual_seed(42 + index)
    if index == 0:
        inputs = dict(input_ids=torch.randint(0, 151936, (1, 17), generator=generator))
        inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
        grids, spans = [], []
    else:
        from PIL import Image
        processor = Qwen3VLProcessor.from_pretrained(ROOT / "processor", local_files_only=True)
        images = []
        if index >= 2:
            for h, w in ([(64, 96)] if index == 2 else [(64, 96), (96, 64)]):
                pixels = torch.randint(0, 256, (h, w, 3), generator=generator, dtype=torch.uint8).numpy()
                images.append(Image.fromarray(pixels))
        prompt = ("A red ceramic teapot on a wooden table, soft morning light." if not images
                  else "Change the object to a blue ceramic teapot.")
        text = BriaFibo2Pipeline._get_edit_prompt(prompt, len(images)) if images else prompt
        inputs = dict(processor(text=[PROMPT_TEMPLATE.format(text)], images=images or None,
                                padding="longest", max_length=4096, truncation=True, return_tensors="pt"))
        grids = inputs["image_grid_thw"].tolist() if images else []
        spans, cursor = [], 0
        ids = inputs["input_ids"][0].numpy()
        for t, h, w in grids:
            assert t == 1
            start = int(np.flatnonzero(ids[cursor:] == 151655)[0]) + cursor
            spans.append([start, h // 2, w // 2])
            cursor = start + h * w // 4
    length = inputs["input_ids"].shape[1]
    mask = torch.zeros(length, length).masked_fill(torch.ones(length, length, dtype=torch.bool).triu(1), -float("inf"))
    return dict(inputs=inputs, tokens=inputs["input_ids"][0].numpy().astype(np.int32),
                grids=grids, spans=spans, patches=array(inputs["pixel_values"]) if grids else None,
                mask=mask[None, None].numpy())


def text_reference(pack, precision, device):
    spec = ("text", precision, device)
    if spec not in MODELS:
        release()
        print("Loading upstream Qwen3-VL:", precision, flush=True)
        MODELS[spec] = Qwen3VLForConditionalGeneration.from_pretrained(
            ROOT / "text_encoder", dtype=getattr(torch, precision), local_files_only=True,
            attn_implementation="eager" if precision == "float32" else "sdpa").to(f"cuda:{device}").eval()
    model = MODELS[spec]
    inputs = {k: v.to(f"cuda:{device}") for k, v in pack["inputs"].items()}
    result = {}
    def capture_vision(module, args, output):
        values = [output.pooler_output] + output.deepstack_features
        result.update({f"vision_{i}": array(x) for i, x in enumerate(values)})
    hook = model.model.visual.register_forward_hook(capture_vision)
    vision_hooks = []
    if os.environ.get("FIBO_VISION_TRACE") == "1":
        visual = model.model.visual
        vision_hooks.append(visual.patch_embed.register_forward_hook(
            lambda m, a, o: result.update(vision_stage_0=array(o))))
        vision_hooks.append(visual.blocks[0].register_forward_pre_hook(
            lambda m, a: result.update(vision_stage_1=array(a[0]))))
        for i, block in enumerate(visual.blocks):
            vision_hooks.append(block.register_forward_hook(
                lambda m, a, o, i=i: result.update({f"vision_stage_{i+2}": array(o)})))
    positions, _ = model.model.get_rope_index(
        inputs["input_ids"],
        mm_token_type_ids=inputs.get("mm_token_type_ids", torch.zeros_like(inputs["input_ids"])),
        image_grid_thw=inputs.get("image_grid_thw"),
        attention_mask=inputs["attention_mask"])
    pack["positions"] = positions[:, 0].cpu().numpy().astype(np.int32)
    try:
        output = model(**inputs, output_hidden_states=True, use_cache=False)
    finally:
        hook.remove()
        for handle in vision_hooks:
            handle.remove()
    normalized = {}
    for i in sorted({n for bundle in TEXT_ENCODER_OUT_LAYERS for n in bundle}):
        states = output.hidden_states[i]
        result[f"hidden_{i}"] = array(states[0])
        f = states.float()
        normalized[i] = (f * torch.rsqrt(f.square().mean(-1, keepdim=True) + 1e-6)).to(states.dtype)
    for i, bundle in enumerate(TEXT_ENCODER_OUT_LAYERS):
        result[f"bundle_{i}"] = array(torch.cat([normalized[n] for n in bundle], dim=-1)[0])
    del output
    return result


def vae_reference(height, width, seed, device):
    spec = ("vae", device)
    if spec not in MODELS:
        release()
        MODELS[spec] = AutoencoderKLFlux2.from_pretrained(
            ROOT / "vae", torch_dtype=torch.float32, local_files_only=True).to(f"cuda:{device}").eval()
    vae = MODELS[spec]
    generator = torch.Generator().manual_seed(seed)
    image = torch.randn(1, 3, height, width, generator=generator).to(f"cuda:{device}")
    latent = torch.randn(1, 32, height // 8, width // 8, generator=generator).to(f"cuda:{device}")
    moments = vae.encode(image).latent_dist.parameters
    decoded = vae.decode(latent).sample
    pipe = BriaFibo2Pipeline(transformer=None, scheduler=None, vae=vae, text_encoder=None, processor=None)
    normalized = pipe.prepare_image_latents([image], 1, image.device, torch.float32)[0]
    return dict(image=array(image), latent=array(latent), moments=array(moments), decoded=array(decoded),
                normalized=array(normalized),
                mean=array(vae.bn.running_mean), invstd=array((vae.bn.running_var + 1e-4).rsqrt()))


def pack_latent(x):
    b, c, h, w = x.shape
    return x.reshape(b, c, h//2, 2, w//2, 2).permute(0, 2, 4, 1, 3, 5).reshape(-1, c*4)


def dit_case(index, precision):
    specs = [(4, 6, [], 17, 0.75), (6, 4, [(4, 4)], 23, 0.375),
             (8, 12, [(4, 6), (6, 4)], 257, 0.125),
             (64, 64, [(32, 48), (48, 32)], 257, 0.015625)]
    h, w, shapes, text_length, timestep = specs[index]
    generator = torch.Generator().manual_seed(42 + index)
    dtype = getattr(torch, precision)
    x = torch.randn(1, 32, h, w, generator=generator).to(dtype)
    references = [torch.randn(1, 32, rh, rw, generator=generator).to(dtype) for rh, rw in shapes]
    context = torch.randn(1, 6, text_length, 7680, generator=generator).to(dtype)
    gist = compute_gist_count([text_length], 256, 512, 64, 256, 2048)[0]
    coordinates = [[513 + k, y, z] for k, (rh, rw) in enumerate([(h, w)] + shapes)
                   for y in range(rh//2) for z in range(rw//2)]
    return dict(x=array(x), references=[array(r) for r in references], context=array(context),
                packed=array(torch.cat([pack_latent(x)] + [pack_latent(r) for r in references])),
                coordinates=coordinates, gist_coordinates=[[i+1, 0, 0] for i in range(gist)],
                height=h, width=w, text_length=text_length, gist=gist, timestep=timestep)


def dit_reference(pack, precision, device):
    spec = ("dit", precision, device)
    if spec not in MODELS:
        release()
        print("Loading upstream Fibo DiT:", VARIANT, precision, flush=True)
        MODELS[spec] = BriaFibo2Transformer2DModel.from_pretrained(
            ROOT / "transformer", torch_dtype=getattr(torch, precision),
            local_files_only=True).to(f"cuda:{device}").eval()
    model = MODELS[spec]
    result, hooks, times = {}, [], []
    def capture(name):
        def hook(module, inputs, output):
            result[name] = array(output[0] if output.ndim == 3 else output)
        return hook
    hooks.append(model.t_embedder.register_forward_hook(lambda m, a, o: times.append(array(o))))
    hooks.append(model.x_embedder.register_forward_hook(capture("patch_embedding")))
    for i, block in enumerate(model.noise_refiner):
        hooks.append(block.register_forward_hook(capture(f"refiner_{i}")))
    for i, block in enumerate(model.perceiver.layers):
        hooks.append(block.register_forward_hook(capture(f"perceiver_{i}")))
    hooks.append(model.perceiver.register_forward_hook(capture("gist")))
    for i, block in enumerate(model.layers):
        hooks.append(block.register_forward_hook(capture(f"layer_{i}")))
    def tensor(x):
        return torch.from_numpy(np.asarray(x)).to(device=f"cuda:{device}", dtype=getattr(torch, precision))
    try:
        with sdpa_kernel(SDPBackend.MATH if precision == "float32" else SDPBackend.FLASH_ATTENTION):
            output = model(hidden_states=tensor(pack["x"]),
                           timestep=torch.tensor([pack["timestep"]], device=f"cuda:{device}"),
                           encoder_hidden_states=tensor(pack["context"]),
                           context_latents=[tensor(x) for x in pack["references"]], return_dict=False)[0]
        result["time"] = np.concatenate(times)
        result["output"] = array(pack_latent(output))
        return result
    finally:
        for hook in hooks:
            hook.remove()


def pipeline_reference(device):
    """Native pipeline oracle; small image and short trajectory test arithmetic, not image quality."""
    from PIL import Image
    precision = "float32"
    positive_pack = text_case(2)
    positive = text_reference(positive_pack, precision, device)
    processor = Qwen3VLProcessor.from_pretrained(ROOT / "processor", local_files_only=True)
    negative_pack = text_case(1)
    negative_inputs = dict(processor(text=[PROMPT_TEMPLATE.format(DEFAULT_NEGATIVE_PROMPT)],
                                    return_tensors="pt"))
    negative_pack["inputs"] = negative_inputs
    negative_pack["tokens"] = negative_inputs["input_ids"][0].numpy().astype(np.int32)
    length = len(negative_pack["tokens"])
    negative_pack["mask"] = torch.zeros(length, length).masked_fill(
        torch.ones(length, length, dtype=torch.bool).triu(1), -float("inf"))[None, None].numpy()
    negative = text_reference(negative_pack, precision, device) if VARIANT == "dev" else None
    context = np.stack([positive[f"bundle_{i}"] for i in range(6)])[None]
    negative_context = (np.stack([negative[f"bundle_{i}"] for i in range(6)])[None]
                        if negative is not None else None)
    release()
    target = f"cuda:{device}"
    transformer = BriaFibo2Transformer2DModel.from_pretrained(
        ROOT / "transformer", torch_dtype=torch.float32, local_files_only=True).to(target).eval()
    vae = AutoencoderKLFlux2.from_pretrained(
        ROOT / "vae", torch_dtype=torch.float32, local_files_only=True).to(target).eval()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(ROOT / "scheduler", local_files_only=True)
    pipe = BriaFibo2Pipeline(transformer=transformer, vae=vae, scheduler=scheduler,
                            text_encoder=None, processor=None)
    generator = torch.Generator().manual_seed(44)
    pixels = torch.randint(0, 256, (64, 96, 3), generator=generator, dtype=torch.uint8).numpy()
    image = Image.fromarray(pixels)
    vae_images, _ = pipe.image_processor.preprocess(image, height=64, width=96)
    initial = torch.randn(1, 32, 8, 12, generator=torch.Generator().manual_seed(1042))
    steps, guidance = (2, 5.0) if VARIANT == "dev" else (4, 1.0)
    result = dict(positive_pack=positive_pack, positive=positive,
                  negative_pack=negative_pack, negative=negative,
                  image=array(vae_images[0]), initial=array(initial), steps=steps, guidance=guidance,
                  mean=array(vae.bn.running_mean),
                  invstd=array((vae.bn.running_var + vae.config.batch_norm_eps).rsqrt()),
                  velocities=[], latents=[], times=[])
    hooks = [
        vae.encoder.register_forward_hook(lambda m, a, o: result.update(encoder_pre_quant=array(o))),
        vae.quant_conv.register_forward_hook(lambda m, a, o: result.update(moments=array(o))),
        vae.decoder.register_forward_hook(lambda m, a, o: result.update(decoded=array(o))),
    ]
    def capture_velocity(module, args, kwargs, output):
        result["velocities"].append(array(output[0]))
        if "references" not in result:
            result["references"] = [array(x) for x in kwargs["context_latents"]]
        result["times"].append(float(kwargs["timestep"][0]))
    hooks.append(transformer.register_forward_hook(capture_velocity, with_kwargs=True))
    def callback(pipeline, index, timestep, values):
        result["latents"].append(array(values["latents"]))
        return values
    try:
        with sdpa_kernel(SDPBackend.MATH):
            output = pipe(prompt_embeds=torch.from_numpy(context).to(target),
                          negative_prompt_embeds=(torch.from_numpy(negative_context).to(target)
                                                  if negative_context is not None else None),
                          image=image, height=64, width=96, latents=initial.to(target),
                          num_inference_steps=steps, guidance_scale=guidance,
                          output_type="pt", callback_on_step_end=callback)
        result["pixels"] = array(output.images)
        result["sigmas"] = array(scheduler.sigmas)
    finally:
        for hook in hooks:
            hook.remove()
    del pipe, transformer, vae
    release()
    return result
