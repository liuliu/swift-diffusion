"""Pinned BFL reference forwards and parity fixtures; no checkpoint export.

Model definitions remain upstream; FP32 DiT controls change only the SDPA backend.
Architecture choices used by Swift are explicit constants in the Swift files.
"""
import gc
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file

ROOT = Path(os.environ.get("FLUX3_MODEL", "/slow/Data/flux-3-image-klein"))
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "reference/src"))
MODEL_REVISION = "8454017f9ebaada30c998fccd9f67a80c7580f68"
REFERENCE_REVISION = "45e57c9fb06f9751cdced29a65c5d2171003be05"
assert json.loads((ROOT / "DOWNLOAD_MANIFEST.json").read_text())["revision"] == MODEL_REVISION
assert json.loads((ROOT / "reference/SOURCE_REVISION.json").read_text())["commit"] == REFERENCE_REVISION
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
REPORT = []


def dtype(name):
    return getattr(torch, name)


def array(value):
    return value.detach().float().cpu().contiguous().numpy()


class ShardedState:
    def __init__(self, root):
        self.root = Path(root)
        self.index = json.loads((self.root / "model.safetensors.index.json").read_text())["weight_map"]

    def __getitem__(self, key):
        with safe_open(self.root / self.index[key], framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    def keys(self):
        return self.index.keys()


def state(component):
    if component == "text":
        return ShardedState(ROOT / "text_encoder")
    return load_file(str(ROOT / ("model.safetensors" if component == "dit" else "ae.safetensors")))


def weight(weights, key, transform="identity", heads=0):
    w = weights[key]
    if transform == "qk":
        w = w.reshape(int(heads), 2, 64, -1).transpose(1, 2).reshape(w.shape)
    elif transform == "qk_norm":
        w = w.reshape(2, 64).transpose(0, 1).reshape(w.shape)
    elif transform == "first_half":
        w = w[:w.shape[0] // 2]
    elif transform == "second_half":
        w = w[w.shape[0] // 2:]
    return array(w)


def check_keys(weights, keys, component, complete=False):
    used = set(keys)
    available = set(weights.keys())
    assert used <= available, sorted(used - available)
    if component == "dit" and complete:
        assert used == available, sorted(available - used)
    if component == "text":
        limit = 36 if complete else 32
        expected = {k for k in available if k.startswith("model.language_model.layers.")
                    and int(k.split(".")[3]) < limit}
        expected.add("model.language_model.embed_tokens.weight")
        if complete:
            expected.add("model.language_model.norm.weight")
        assert used == expected, (sorted(expected - used), sorted(used - expected))
    print(f"{component}: checked {len(used)} unique source tensors", flush=True)


def metrics(label, actual, expected, tolerance):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (label, actual.shape, expected.shape)
    assert np.isfinite(actual).all() and np.isfinite(expected).all(), label
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    max_abs = float(np.abs(delta).max())
    max_rel = max_abs / max(float(np.abs(expected).max()), 1e-12)
    rel_l2 = float(np.linalg.norm(delta.ravel()) / max(np.linalg.norm(expected.astype(np.float64).ravel()), 1e-12))
    result = dict(label=label, shape=list(actual.shape), max_abs=max_abs,
                  max_relative=max_rel, relative_l2=rel_l2,
                  actual_sample=actual.ravel()[:8].tolist(), reference_sample=expected.ravel()[:8].tolist(),
                  tolerance=float(tolerance), passed=max_rel <= tolerance and rel_l2 <= tolerance)
    REPORT.append(result)
    print(json.dumps(result), flush=True)
    return result["passed"]


def finish(name):
    assert REPORT and all(r["passed"] for r in REPORT), "Numeric parity failed"
    result = dict(model_revision=MODEL_REVISION, reference_revision=REFERENCE_REVISION,
                  torch=torch.__version__, results=REPORT)
    path = ARTIFACTS / f"{name}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print("PARITY PASSED:", path, flush=True)


def release():
    gc.collect()
    torch.cuda.empty_cache()


def text_case(kind, length=80, seed=42):
    if kind == "random":
        generator = torch.Generator().manual_seed(seed)
        tokens = torch.randint(0, 151936, (1, length), generator=generator)
        attention = torch.ones_like(tokens)
    else:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(ROOT / "text_encoder", local_files_only=True)
        text = processor.apply_chat_template(
            [{"role": "user", "content": kind}], tokenize=False, add_generation_prompt=True)
        tokenizer = processor.tokenizer
        tokenizer.padding_side = "right"
        real = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)["input_ids"].shape[1]
        length = min((real + 79) // 80 * 80, 4096)
        batch = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=length)
        tokens, attention = batch["input_ids"], batch["attention_mask"]
    positions = (attention.cumsum(-1) - 1).masked_fill(attention == 0, 1)
    mask = torch.zeros(length, length).masked_fill(
        torch.ones(length, length, dtype=torch.bool).triu(1) | (attention[0] == 0)[None], -float("inf"))
    return dict(tokens=tokens[0].numpy().astype(np.int32), attention=attention,
                positions=positions[0].numpy().astype(np.int32), mask=mask[None, None].numpy())


def text_reference(weights, case, precision, device, complete=False):
    from transformers import AutoConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel, Qwen3VLTextRotaryEmbedding
    config = AutoConfig.from_pretrained(ROOT / "text_encoder", local_files_only=True).text_config
    config._attn_implementation = "eager" if precision == "float32" else "sdpa"
    with torch.device("meta"):
        model = Qwen3VLTextModel(config)
    sd = {k.removeprefix("model.language_model."): weights[k].to(dtype(precision))
          for k in weights.keys() if k.startswith("model.language_model.")}
    model.load_state_dict(sd, strict=True, assign=True)
    # Non-persistent rotary buffers are not in the checkpoint's state dictionary.
    model.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)
    model = model.to(f"cuda:{device}").eval()
    positions = torch.from_numpy(case["positions"]).long()[None].to(f"cuda:{device}")
    output = model(input_ids=torch.from_numpy(case["tokens"]).long()[None].to(f"cuda:{device}"),
                   attention_mask=case["attention"].to(f"cuda:{device}"),
                   position_ids=positions[None].expand(4, -1, -1),
                   use_cache=False, output_hidden_states=True)
    result = {f"hidden_{i}": array(output.hidden_states[i][0]) for i in [4,8,12,16,20,24,28,32]}
    if complete:
        result["final_norm"] = array(output.last_hidden_state[0])
    del model, output, sd
    release()
    return result


def make_ids(h, w, reference=None):
    from flux3.autoencoder import latent_ids
    ids = latent_ids(h, w).float()
    ids[:, 1:3] *= 0.5
    if reference is not None:
        ids[:, 3] = -(1 + reference)
    return ids


def dit_case(text_length, height, width, references, seed, precision="float32", time=0.75, target=0.5):
    generator = torch.Generator().manual_seed(seed)
    def rand(n, d, scale=1):
        return (torch.randn(n, d, generator=generator) * scale).to(dtype(precision)).float().numpy()
    reference_ids = [make_ids(int(h), int(w), i) for i, (h, w) in enumerate(references)]
    ref_ids = torch.cat(reference_ids) if reference_ids else torch.zeros(0, 4)
    text_ids = torch.zeros(text_length, 4)
    text_ids[:, 3] = torch.arange(text_length)
    ids = torch.cat([text_ids, make_ids(height, width), ref_ids])
    # Match the reference's BF16 timestep representation for native BF16 runs.
    time = float(torch.tensor(time).to(dtype(precision)))
    # denoise_flow_map keeps the destination time in the FP32 schedule dtype.
    target = float(torch.tensor(target, dtype=torch.float32))
    return dict(x=rand(height * width, 128), context=rand(text_length, 20480, 0.1),
                reference=rand(len(ref_ids), 128), ids=ids.numpy(), time=time, target=target)


def dit_reference(weights, case, precision, device, flow_map=True, debug=True):
    import flux3.dit as module
    original_attention = module.packed_attention
    if precision == "float32":
        # cuDNN attention in the release selects a reduced precision backend.
        # Use PyTorch's FP32 math backend for the FP32 control only.
        def math_attention(q, k, v, bounds):
            out = torch.zeros(q.shape[2], q.shape[1] * q.shape[3], device=q.device, dtype=q.dtype)
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                for a, b in bounds:
                    if b > a:
                        y = torch.nn.functional.scaled_dot_product_attention(q[:,:,a:b],k[:,:,a:b],v[:,:,a:b])
                        out[a:b] = y[0].transpose(0,1).reshape(b-a,-1)
            return out
        module.packed_attention = math_attention
    try:
        with torch.device("meta"):
            model = module.DiT(flow_map)
        sd = {k: v.to(dtype(precision)) for k, v in weights.items()
              if flow_map or not k.startswith("ref_time_in.")}
        model.load_state_dict(sd, strict=True, assign=True)
        model = model.to(f"cuda:{device}").eval()
        result = {}
        nc, nx, nr = len(case["context"]), len(case["x"]), len(case["reference"])
        def hook(name, shared=False):
            def capture(_, inputs, output):
                if shared:
                    output = torch.cat([output[nx+nr:], output[:nx], output[nx:nx+nr]])
                result[name] = array(output[None])
            return capture
        if debug:
            model.txt_mode_blocks[-1].register_forward_hook(hook("context_refiner"))
            model.content_mode_blocks["image"][-1].register_forward_hook(hook("x_refiner"))
            if nr:
                model.content_mode_blocks["image_cond"][-1].register_forward_hook(hook("ref_refiner"))
            for i in [0,7,27]:
                model.single_blocks[i].register_forward_hook(hook(f"single_{i}", True))
        data = {m: torch.from_numpy(case[k]).to(device=f"cuda:{device}",dtype=dtype(precision))
                for m,k in [("text","context"),("image","x"),("image_cond","reference")]}
        all_ids = torch.from_numpy(case["ids"]).to(f"cuda:{device}")
        ids = dict(text=all_ids[:nc], image=all_ids[nc:nc+nx], image_cond=all_ids[nc+nx:])
        times = {m: torch.full((len(x),), case["time"] if m == "image" else 0,
                              device=x.device,dtype=x.dtype) for m,x in data.items()}
        pack = module.Packed(data,ids,times,{m:[(0,len(x))] for m,x in data.items()})
        target = torch.tensor([case["target"]],device=f"cuda:{device}",dtype=torch.float32) if flow_map else None
        # Match the native sampler. DiT.forward casts destination times to the
        # current-time dtype before computing their sinusoidal embeddings.
        with torch.autocast("cuda", dtype=dtype(precision), enabled=precision != "float32"):
            result["output"] = array(model(pack, target)[None])
        del model, sd, pack
        release()
        return result
    finally:
        module.packed_attention = original_attention


def vae_reference(weights, height, width, seed, device):
    from flux3.autoencoder import AutoEncoder, latent_ids
    model = AutoEncoder().eval()
    model.load_state_dict(weights, strict=True)
    model = model.to(f"cuda:{device}")
    generator = torch.Generator().manual_seed(seed)
    image = torch.randn(1,3,height,width,generator=generator)
    latent = torch.randn(1,32,height//8,width//8,generator=generator)
    tokens = torch.randn(height//16 * (width//16),128,generator=generator)
    moments = model.encoder(image.to(f"cuda:{device}"))
    decoded = model.decoder(latent.to(f"cuda:{device}"))
    encoded_tokens, ids = model.encode(image[0].to(f"cuda:{device}"))
    decoded_tokens = model.decode(tokens.to(f"cuda:{device}"),ids)
    result = dict(image=array(image), latent=array(latent), tokens=array(tokens),
                  moments=array(moments), decoded=array(decoded),
                  encoded_tokens=array(encoded_tokens), decoded_tokens=array(decoded_tokens[None]))
    del model, moments, decoded, encoded_tokens, decoded_tokens
    release()
    return result


def normalize_moments(moments, weights):
    mean = torch.from_numpy(np.asarray(moments))[:,:32]
    h, w = mean.shape[-2:]
    z = mean.reshape(1,32,h//2,2,w//2,2).permute(0,1,3,5,2,4).reshape(1,128,h//2,w//2)
    z = (z - weights["bn.running_mean"][None,:,None,None]) / (weights["bn.running_var"][None,:,None,None] + 1e-4).sqrt()
    return array(z[0].permute(1,2,0).reshape(-1,128))


def denormalize_tokens(tokens, weights, height, width):
    h,w = height//16,width//16
    z = torch.from_numpy(np.asarray(tokens).reshape(h,w,128)).permute(2,0,1)[None]
    z = z * (weights["bn.running_var"][None,:,None,None]+1e-4).sqrt() + weights["bn.running_mean"][None,:,None,None]
    return array(z.reshape(1,32,2,2,h,w).permute(0,1,4,2,5,3).reshape(1,32,h*2,w*2))


def concatenate_features(features):
    return np.ascontiguousarray(np.concatenate([np.asarray(x) for x in features], axis=-1))


def pipeline_case(device, steps=2, height=64, width=96):
    """FP32 control of the complete edited-image path, with BF16-rounded initial noise."""
    from flux3.sampling import timesteps
    from flux3.autoencoder import AutoEncoder, latent_ids
    text = text_case("A red ceramic teapot on a wooden table, in soft morning light.")
    text_state = state("text")
    text_result = text_reference(text_state, text, "float32", device)
    context = concatenate_features([text_result[f"hidden_{i}"] for i in [4,8,12,16,20,24,28,32]])
    vae_state = state("vae")
    vae = AutoEncoder().eval().to(f"cuda:{device}")
    vae.load_state_dict(vae_state)
    # Same seeded image supplied to both implementations, not a precomputed latent.
    reference_image = torch.randn(1,3,64,64,generator=torch.Generator().manual_seed(123)).clamp(-1,1)
    reference_tokens, _ = vae.encode(reference_image[0].to(f"cuda:{device}"))
    reference_tokens = array(reference_tokens)
    del vae
    release()
    case = dit_case(len(context),height//16,width//16,[[4,4]],42)
    noise = torch.randn(height//16*(width//16),128,generator=torch.Generator().manual_seed(42))
    case["x"] = array(noise.bfloat16().float())
    initial = case["x"].copy()
    case["context"], case["reference"] = context, reference_tokens
    schedule = timesteps(steps,6.93).numpy()
    dit_state = state("dit")
    velocities, latents = [], []
    for t,r in zip(schedule[:-1],schedule[1:]):
        case["time"],case["target"] = float(t),float(r)
        velocity = dit_reference(dit_state,case,"float32",device,debug=False)["output"][0]
        case["x"] = case["x"] + float(r-t) * velocity
        velocities.append(velocity.copy()); latents.append(case["x"].copy())
    del dit_state
    vae = AutoEncoder().eval().to(f"cuda:{device}")
    vae.load_state_dict(vae_state)
    decoded = array(vae.decode(torch.from_numpy(case["x"]).to(f"cuda:{device}"),
                               latent_ids(height//16,width//16,f"cuda:{device}"))[None])
    del vae
    release()
    return dict(text=text,context=context,reference_image=array(reference_image),
                reference_tokens=reference_tokens,initial=initial,ids=case["ids"],schedule=schedule,
                velocities=velocities,latents=latents,decoded=decoded,height=height,width=width)


def save_image(array_value, name):
    from PIL import Image
    image = np.asarray(array_value)[0].transpose(1,2,0)
    image = ((np.clip(image,-1,1)+1)*127.5).astype(np.uint8)
    path = ARTIFACTS / name
    Image.fromarray(image).save(path)
    return str(path)
