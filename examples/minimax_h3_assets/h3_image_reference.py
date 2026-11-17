"""Parity data from the pinned official H3-Image implementation, not a second model."""

import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import torch
from safetensors.torch import load_file

ROOT = Path("/slow/Data/MiniMax-H3-Image")


def official_module(relative, module_name):
    component = "vae" if "autoencoder" in relative else "dit"
    verified = json.loads((ROOT / f"{component}_verified.json").read_text())
    path = ROOT / "reference" / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == verified["reference/" + relative]
    # Load the verified release's model source unchanged, using the matching
    # installed Diffusers dependencies (see the converter README).
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class VAEState:
    def __init__(self, state):
        self.state = state
        self.accessed = set()

    def __getitem__(self, key):
        key = re.sub(r"encoder\.down\.(\d+)\.block\.(\d+)\.", r"encoder.down_blocks.\1.resnets.\2.", key)
        key = re.sub(r"encoder\.down\.(\d+)\.downsample\.", r"encoder.down_blocks.\1.downsamplers.0.", key)
        key = key.replace(".nin_shortcut.", ".conv_shortcut.")
        self.accessed.add(key)
        return self.state[key]

    def validate_coverage(self):
        assert self.accessed == set(self.state), (set(self.state) - self.accessed, self.accessed - set(self.state))
        print(f"H3-Image VAE: all {len(self.state)} source tensors consumed", flush=True)


def load_vae(device):
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads((ROOT / "vae_diffusers/config.json").read_text())
    for key, expected in {
        "latent_channels": 24, "block_out_channels": [128, 256, 256, 512, 512, 1024],
        "layers_per_block": 2, "spatial_downsample_factors": [2, 2, 2, 2, 1, 1],
        "temporal_downsample_factors": [1, 2, 2, 1, 1, 1], "norm_num_groups": 32,
        "norm_eps": 1e-6, "spatial_padding_mode": "reflect", "decoder_num_layers": 24,
        "decoder_num_attention_heads": 16, "decoder_attention_head_dim": 64,
        "decoder_num_register_tokens": 4, "decoder_ffn_mult": 4,
        "decoder_rope_theta": 100.0, "decoder_rope_dim_ratio": 0.5, "decoder_norm_eps": 1e-5,
    }.items():
        assert config[key] == expected, (key, config[key], expected)
    relative = "vae_diffusers/diffusion_pytorch_model.safetensors"
    verified = json.loads((ROOT / "vae_verified.json").read_text())
    with (ROOT / relative).open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == verified[relative]
    state = load_file(str(ROOT / relative))
    assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in state.values())
    module = official_module("models/autoencoders/autoencoder_kl_minimax_h3_image.py",
                             "diffusers.models.autoencoders.autoencoder_kl_minimax_h3_image")
    model = module.AutoencoderKLMiniMaxH3Image.from_config(config).float().eval()
    model.load_state_dict(state, strict=True, assign=True)
    model.to(f"cuda:{device}")
    model.disable_tiling()
    print("H3-Image VAE loaded: official FP32 encoder and 24-layer decoder", flush=True)
    return {"model": model, "state": VAEState(state), "config": config, "device": device}


def image_rows(image):
    return image.permute(0, 2, 3, 1).contiguous().reshape(-1, image.shape[1]).cpu()


def image_patches(image):
    n, c, h, w = image.shape
    return image.reshape(n, c, h // 16, 16, w // 16, 16).permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, 768).cpu()


@torch.no_grad()
def vae_case(pack, height, width, seed):
    model = pack["model"]
    generator = torch.Generator(device=f"cuda:{pack['device']}").manual_seed(seed)
    pixels = torch.randn(1, 3, height, width, device=generator.device, generator=generator).clamp(-2, 2)
    z = torch.randn(1, 24, height // 16, width // 16, device=generator.device, generator=generator)
    mean = torch.tensor(pack["config"]["latents_mean"], device=generator.device).view(1, 24, 1, 1)
    std = torch.tensor(pack["config"]["latents_std"], device=generator.device).view(1, 24, 1, 1)
    denormalized = z * std + mean
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        moments = model._encode(pixels)
        decoded = model.decode(denormalized).sample
        # Deterministic mean reconstruction, so only the model conversion is compared.
        reconstructed = model.decode(moments[:, :24]).sample
    # Independently exercise the official 2D RoPE generator.
    grids = [2 * (torch.arange(0.5, size, device=generator.device) / size) - 1
             for size in (height // 16, width // 16)]
    positions = torch.stack(torch.meshgrid(*grids, indexing="ij"), -1).flatten(0, 1)[None]
    positions = torch.cat([positions, positions.new_zeros(1, 5, 2)], 1)
    cos, sin = model.decoder.rope(positions)
    rotary = torch.zeros(positions.shape[1], 64, device=generator.device)
    rotary[:, ::2] = 1
    rotary[:, :32:2] = cos[0, :, 0, :16]
    rotary[:, 1:32:2] = sin[0, :, 0, :16]
    return {"pixels": pixels[:, :, None].cpu(), "moments": image_rows(moments),
            "latent_rows": image_rows(denormalized), "decoded": image_patches(decoded),
            "reconstructed": image_patches(reconstructed), "rotary": rotary.cpu()}


@torch.no_grad()
def decode_rows(pack, rows, height, width):
    z = torch.as_tensor(rows).reshape(1, height, width, 24).permute(0, 3, 1, 2).contiguous()
    z = z.to(f"cuda:{pack['device']}")
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        return image_patches(pack["model"].decode(z).sample)


def release_vae(pack):
    pack["model"] = None
    gc.collect()
    torch.cuda.empty_cache()


def audit_vae_export(path, pack, encoder_names, decoder_names):
    import sqlite3
    import numpy as np
    expected = {f"__video_encoder__[{name}]" for name in encoder_names}
    expected |= {f"__video_decoder__[{name}]" for name in decoder_names}
    assert len(expected) == len(encoder_names) + len(decoder_names), "Duplicate parameter names"
    names, elements, maximum = set(), 0, 0.0
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as store:
        assert store.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        for name, datatype, dimensions, payload in store.execute("SELECT name,datatype,dim,data FROM tensors"):
            assert datatype == 16384, (name, datatype)  # CCV_32F
            shape = tuple(int(d) for d in np.frombuffer(dimensions, dtype="<i4") if d)
            values = np.frombuffer(payload, dtype="<f4")
            assert values.size == np.prod(shape), (name, shape, values.size)
            assert np.isfinite(values).all(), f"Non-finite exported weight: {name}"
            names.add(name)
            elements += values.size
            maximum = max(maximum, float(np.abs(values).max()))
    assert names == expected, (expected - names, names - expected)
    assert elements == sum(value.numel() for value in pack["state"].state.values())
    assert len(names) == 559  # 511 source tensors + two extra per split SwiGLU, for 24 blocks.
    result = {"tensors": len(names), "encoder_tensors": len(encoder_names),
              "decoder_tensors": len(decoder_names), "elements": int(elements),
              "storage_dtype": "float32", "max_weight_magnitude": maximum,
              "all_finite": True, "strict_reload_max_abs": 0.0}
    print("H3-Image VAE store audit:", result, flush=True)
    return result


def write_vae_export_metadata(path, pack, audit):
    metadata = {"source": json.loads((ROOT / "download_identity.json").read_text()),
                "vae_config": pack["config"], "audit": audit,
                "models": ["video_encoder", "video_decoder"],
                "latent_statistics_location": "outside the exported models"}
    with Path(str(path) + ".json").open("x") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")


class ImageDiTState:
    """Keep the existing H3 reader, concatenating the six already-fused heads."""

    def __init__(self, source, float16=False):
        self.source = source
        self.float16 = float16
        self.rounded = float16
        self.storage_statistics = {}
        self.export_tensors = None

    def enable_export_audit(self):
        self.export_tensors = {}

    def source_value(self, key):
        value = self.source[key]
        if not self.rounded:
            return value
        full = value.float()
        half = full.half()
        if key not in self.storage_statistics:
            assert torch.isfinite(full).all() and torch.isfinite(half).all(), (key, "FP16 overflow/non-finite")
            restored = half.float()
            self.storage_statistics[key] = {
                "elements": full.numel(), "max_abs": full.abs().max().item(),
                "max_rounding_abs": (full - restored).abs().max().item(),
                "underflow_to_zero": ((full != 0) & (restored == 0)).sum().item(),
            }
        return half.float()

    def __getitem__(self, key):
        if key in {"proj_out.weight", "proj_out.bias"}:
            suffix = key.split(".")[-1]
            value = torch.cat([self.source_value(f"pdd_proj_out.{i}.{suffix}") for i in range(6)], 0)
        else:
            value = self.source_value(key)
        if self.export_tensors is not None:
            # Reuse the independently verified H3 naming / permutation map.
            # Importing these helpers does not load Pika weights or adapters.
            from pika_h3_reference import store_parts, forward_qk
            names, permutation = store_parts(key)
            if names[0] not in self.export_tensors:
                stored = value.half() if self.float16 else value.float()
                assert torch.isfinite(stored).all(), key
                if permutation != "none":
                    stored = forward_qk(stored, permutation == "qk_norm").reshape(value.shape)
                chunks = stored.chunk(len(names), dim=0) if len(names) > 1 else [stored]
                for name, chunk in zip(names, chunks):
                    shape = tuple(chunk.shape)
                    # ccv stores RMSNorm weights in their broadcast shape,
                    # whereas PyTorch keeps only the normalized dimension.
                    if ".attn.norm_" in key:
                        shape = (1, 1, 1, chunk.numel())
                    elif key.endswith((".norm1.weight", ".norm2.weight", ".final_norm.weight", "norm_out.norm.weight")):
                        shape = (1, 1, chunk.numel())
                    self.export_tensors[name] = {
                        "shape": shape, "elements": chunk.numel(),
                        "sha256": hashlib.sha256(memoryview(chunk.contiguous().numpy())).hexdigest(),
                    }
        return value

    def validate_coverage(self):
        expected = set(self.source.weight_map)
        assert self.source.accessed == expected, (expected - self.source.accessed, self.source.accessed - expected)
        print(f"H3-Image DiT: all {len(expected)} source tensors consumed", flush=True)
        if self.float16:
            assert self.rounded and set(self.storage_statistics) == expected
            self.storage_summary = {
                "all_finite": True, "source_tensors": len(expected),
                "elements": sum(x["elements"] for x in self.storage_statistics.values()),
                "max_abs": max(x["max_abs"] for x in self.storage_statistics.values()),
                "max_rounding_abs": max(x["max_rounding_abs"] for x in self.storage_statistics.values()),
                "underflow_to_zero": sum(x["underflow_to_zero"] for x in self.storage_statistics.values()),
            }
            print("H3-Image FP16 storage safety:", self.storage_summary, flush=True)


def audit_dit_export(path, state, names):
    import sqlite3
    import numpy as np
    expected = state.export_tensors
    assert expected is not None
    actual_names = {f"__dit__[{name}]" for name in names}
    assert len(names) == len(actual_names) == len(expected) == 2388
    assert actual_names == set(expected)
    elements = 0
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as store:
        rows = store.execute("SELECT name, datatype, dim FROM tensors").fetchall()
        assert {name for name, _, _ in rows} == actual_names
        for index, (name, datatype, dimensions) in enumerate(rows):
            assert datatype == (131072 if state.float16 else 16384), (name, datatype)  # CCV_16F / CCV_32F
            shape = tuple(int(d) for d in np.frombuffer(dimensions, dtype="<i4") if d)
            item = expected[name]
            assert shape == item["shape"], (name, shape, item["shape"])
            payload = store.execute("SELECT data FROM tensors WHERE name=?", (name,)).fetchone()[0]
            assert len(payload) == item["elements"] * (2 if state.float16 else 4), name
            assert hashlib.sha256(payload).hexdigest() == item["sha256"], (name, "weight mismatch")
            elements += item["elements"]
            if (index + 1) % 200 == 0:
                print(f"H3-Image DiT export audit: {index + 1}/{len(rows)} exact tensors", flush=True)
        print("H3-Image DiT: checking SQLite integrity", flush=True)
        assert store.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    result = {"tensors": len(expected), "source_tensors": len(state.source.weight_map),
              "elements": elements, "storage_dtype": "float16" if state.float16 else "float32", "all_finite": True,
              "all_source_transforms_exact": True, "strict_reload_max_abs": 0.0}
    if state.float16:
        result["storage_safety"] = state.storage_summary
    print("H3-Image DiT store audit:", result, flush=True)
    return result


def write_dit_export_metadata(path, audit):
    metadata = {
        "model_key": "dit", "dtype": audit["storage_dtype"], "runtime_scaling_baked_in": False,
        "source": json.loads((ROOT / "download_identity.json").read_text()),
        "transformer_config": json.loads((ROOT / "transformer_diffusers/config.json").read_text()),
        "pdd_config": json.loads((ROOT / "pdd_config.json").read_text()),
        "source_verification": json.loads((ROOT / "dit_verified.json").read_text()),
        "output_layout": "head-major: six already-fused 96-channel heads in proj_out",
        "audit": audit,
        "validation_scope": "Matched stored-weight parity with FP32 arithmetic; not FP16 runtime or generation quality",
    }
    with Path(str(path) + ".inference.json").open("x") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")


def load_image_dit(source, device, float16=False):
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads((ROOT / "transformer_diffusers/config.json").read_text())
    for key, expected in {
        "hidden_size": 5376, "num_layers": 50, "num_refiner_layers": 2,
        "num_attention_heads": 56, "attention_head_dim": 128, "ffn_dim": 14336,
        "in_channels": 24, "patch_size": [1, 2, 2], "text_dim": 5120,
        "freq_dim": 256, "time_embed_hidden_dim": 5376, "time_embed_dim": 2688,
        "rope_freq_dim": 16, "rope_theta": 10000.0, "norm_eps": 1e-5,
        "qk_norm_eps": 1e-5, "final_norm_eps": 1e-5, "num_pdd_heads": 6,
    }.items():
        assert config[key] == expected, (key, config[key], expected)
    module = official_module("models/transformers/transformer_minimax_h3_image.py",
                             "diffusers.models.transformers.transformer_minimax_h3_image")
    # Run the unchanged official forward. Hooks only stream weights into/out of
    # GPU memory; they do not replace attention, normalization, or block code.
    with torch.device("meta"):
        model = module.MiniMaxH3ImageTransformer2DModel.from_config(config).float().eval()
    assert set(model.state_dict()) == set(source.weight_map)
    model.rope = module.MiniMaxH3ImageRotaryPosEmbed().to(f"cuda:{device}")
    state = ImageDiTState(source, float16=float16)
    streamed = ["proj_in", "context_embedder", "time_embedder", "token_refiner.final_norm", "norm_out"]
    streamed += [f"token_refiner.refiner_blocks.{i}" for i in range(2)]
    streamed += [f"transformer_blocks.{i}" for i in range(50)]
    streamed += [f"pdd_proj_out.{i}" for i in range(6)]
    for prefix in streamed:
        child = model.get_submodule(prefix)

        def load(child, args, prefix=prefix):
            values = {key: state.source_value(prefix + "." + key).to(device=f"cuda:{device}", dtype=torch.float32)
                      for key in child.state_dict()}
            child.load_state_dict(values, strict=True, assign=True)

        def offload(child, args, output, prefix=prefix):
            child.to("meta")
            if prefix.startswith("transformer_blocks."):
                print(f"H3-Image official {prefix} complete", flush=True)

        child.register_forward_pre_hook(load)
        child.register_forward_hook(offload)
    return {"model": model, "source": source, "state": state, "device": device}


@torch.no_grad()
def image_dit_case(pack, reference_images=False):
    model = pack["model"]
    device = f"cuda:{pack['device']}"
    generator = torch.Generator(device=device).manual_seed(73 if reference_images else 72)
    text_length, image_length = (4, 6) if reference_images else (3, 4)
    length = text_length + image_length
    image = torch.randn(1, image_length, 96, device=device, generator=generator)
    text = torch.randn(1, text_length, 5120, device=device, generator=generator)
    if reference_images:
        # Two vision-conditioned text rows use image modality, with two fixed
        # reference-latent rows and four target rows at a different noise level.
        timestep = torch.tensor([0.25, 0.999, 1.0], device=device)
        timestep_indices = torch.tensor([2] * 4 + [1] * 2 + [0] * 4, device=device)
        tags = torch.tensor([1, 0, 0, 1] + [0] * 6, device=device)
    else:
        timestep = torch.tensor([0.75, 1.0], device=device)
        timestep_indices = torch.tensor([1] * 3 + [0] * 4, device=device)
        tags = torch.tensor([1] * 3 + [0] * 4, device=device)
    positions = torch.randn(length, 3, device=device, generator=generator)
    image_indices = torch.arange(text_length, length, device=device)
    text_indices = torch.arange(text_length, device=device)
    captured = {}
    hooks = []

    def capture(name, inputs=False):
        def hook(child, args, output):
            captured[name] = (args[0] if inputs else output).detach().float().cpu()
        return hook

    for name, child, inputs in [
        ("refiner_input", model.token_refiner, True),
        ("refiner_output", model.token_refiner, False),
        ("block_input", model.transformer_blocks[0], True),
        ("block_output", model.transformer_blocks[0], False),
        ("temb", model.time_embedder, False),
        ("pre_norm", model.norm_out, True),
        ("hidden", model.norm_out, False),
    ]:
        hooks.append(child.register_forward_hook(capture(name, inputs)))
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        first = model(image, text, timestep, timestep_indices, tags, positions,
                      image_indices, text_indices, pdd_step_index=0).sample
        # All heads use the same post-norm hidden state. Head 0 still comes from
        # the complete official forward, so packing and final row selection are checked.
        hidden = captured["hidden"].to(device)
        outputs = [first]
        for head in model.pdd_proj_out[1:]:
            outputs.append(head(hidden).index_select(1, image_indices))
    for hook in hooks:
        hook.remove()
    cos, sin = model.rope(positions)
    rotary = torch.zeros(length, 128, device=device)
    rotary[:, ::2] = 1
    rotary[:, :96:2] = cos[:, :48]
    rotary[:, 1:96:2] = sin[:, :48]
    indices = timestep_indices * 3 + tags
    pack["source"].release()
    return {"image": image.cpu(), "text": text.cpu(), "rotary": rotary.cpu(),
            "adaln_selection": torch.nn.functional.one_hot(indices, num_classes=len(timestep) * 3).float().cpu(),
            "timestep_selection": torch.nn.functional.one_hot(timestep_indices, num_classes=len(timestep)).float().cpu(),
            "frequencies": model.time_proj(timestep).cpu(),
            "image_output": torch.cat(outputs, -1).cpu(), **captured}


if __name__ == "__main__":
    import os
    pack = load_vae(int(os.environ.get("MINIMAX_H3_DEVICE", "3")))
    case = vae_case(pack, 64, 96, 61)
    print({name: tuple(value.shape) for name, value in case.items()}, flush=True)
