"""FastH3-specific schema/PDD checks; the transformer reuses our H3 reference."""

import ast
from collections import Counter
import json
from pathlib import Path
import sqlite3
import struct
from types import SimpleNamespace

import torch
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open


def required_keys(weight_map, layers):
    return {key for key in weight_map if not key.startswith("transformer_blocks.") or int(key.split(".")[1]) < layers}


def audit_source(root, layers=50):
    root = Path(root)
    config = json.loads((root / "transformer_ref/config.json").read_text())
    expected = dict(
        hidden_size=5376, num_layers=50, num_refiner_layers=2,
        num_attention_heads=56, attention_head_dim=128, ffn_dim=14336,
        in_channels=24, audio_in_channels=32, patch_size=[1, 2, 2],
        text_dim=5120, freq_dim=256, time_embed_hidden_dim=5376,
        time_embed_dim=2688, rope_freq_dim=16, rope_theta=10000.0,
        norm_eps=1e-5, qk_norm_eps=1e-5, final_norm_eps=1e-5, pdd_steps=32,
    )
    for key, value in expected.items():
        assert config[key] == value, (key, config[key], value)
    shapes = {}

    def linear(prefix, rows, cols, bias=False):
        shapes[prefix + ".weight"] = [rows, cols]
        if bias:
            shapes[prefix + ".bias"] = [rows]

    linear("time_embedder.linear_1", 5376, 256, True)
    linear("time_embedder.linear_2", 2688, 5376, True)
    linear("proj_in", 5376, 96, True)
    linear("audio_proj_in", 5376, 32, True)
    linear("context_embedder", 5376, 5120, True)
    linear("proj_out", 32 * 96, 5376, True)
    linear("audio_proj_out", 32 * 32, 5376, True)
    linear("norm_out.linear", 2 * 5376, 2688, True)
    shapes["norm_out.norm.weight"] = [5376]
    shapes["token_refiner.final_norm.weight"] = [5376]
    for prefix, count in (("transformer_blocks", 50), ("token_refiner.refiner_blocks", 2)):
        for layer in range(count):
            base = f"{prefix}.{layer}"
            for norm in ("norm1", "norm2"):
                shapes[f"{base}.{norm}.weight"] = [5376]
            for qkv in ("q", "k", "v"):
                linear(f"{base}.attn.to_{qkv}", 7168, 5376)
            for qk in ("q", "k"):
                shapes[f"{base}.attn.norm_{qk}.weight"] = [128]
            linear(f"{base}.attn.to_out.0", 5376, 7168)
            linear(f"{base}.ff.net.0.proj", 2 * 14336, 5376)
            linear(f"{base}.ff.net.2", 5376, 14336)
            if prefix == "transformer_blocks":
                linear(f"{base}.attn.to_gate_compress", 7168, 5376)
                linear(f"{base}.adaln_proj.linear", 3 * 6 * 5376, 2688, True)
    weight_map = json.loads((root / "transformer_ref/diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
    assert set(weight_map) == set(shapes), (set(weight_map) - set(shapes), set(shapes) - set(weight_map))
    counts = Counter()
    required = required_keys(weight_map, layers)
    for shard in sorted({weight_map[key] for key in required}):
        with safe_open(root / "transformer_ref" / shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                assert weight_map[key] == shard
                tensor = f.get_slice(key)
                assert tensor.get_shape() == shapes[key], (key, tensor.get_shape(), shapes[key])
                counts[tensor.get_dtype()] += 1
    if layers == 50:
        assert sum(counts.values()) == 688
    contract = json.loads((root / "fastvideo_inference.json").read_text())
    assert contract["pdd_steps"] == 32 and contract["pdd_step_indices"] == list(range(0, 33, 4))
    assert contract["grid_max_t"] == 0.999
    assert contract["video_scheduler_shift"] == 12 and contract["audio_scheduler_shift"] == 3
    print(f"FastH3 source schema: all 688 names match; validated shards needed for {layers} layers; dtypes:", dict(counts), flush=True)


def check_coverage(state, layers=50):
    expected = required_keys(state.weight_map, layers)
    assert state.accessed == expected, (sorted(expected - state.accessed), sorted(state.accessed - expected))
    print(f"FastH3 Swift loader consumed every source tensor for {layers} layers:", len(state.accessed), flush=True)


def check_model_names(names):
    base = "/slow/Data/minimax_h3_ref2va_dit_f32.ckpt"
    with sqlite3.connect(f"file:{base}?mode=ro", uri=True) as original:
        expected = {row[0] for row in original.execute("SELECT name FROM tensors")}
    expected |= {f"__dit__[t-gate_compress-{layer}-0]" for layer in range(50)}
    actual = {f"__dit__[{name}]" for name in names}
    assert len(names) == len(actual), "Duplicate Swift parameter names"
    assert actual == expected, (sorted(expected - actual), sorted(actual - expected))
    print("FastH3 parameter names match the original H3 export plus 50 compression gates:", len(actual), flush=True)


def audit_export(path, state):
    base = "/slow/Data/minimax_h3_ref2va_dit_f32.ckpt"
    with sqlite3.connect(f"file:{base}?mode=ro", uri=True) as original:
        original_rows = {name: (dtype, dims) for name, dtype, dims in original.execute("SELECT name, datatype, dim FROM tensors")}
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as exported:
        rows = exported.execute("SELECT name, datatype, dim FROM tensors").fetchall()
        gate_names = {f"__dit__[t-gate_compress-{layer}-0]": layer for layer in range(50)}
        assert {name for name, _, _ in rows} == set(original_rows) | set(gate_names), "Unexpected export names"
        head_count = 0
        for name, dtype, dims in rows:
            assert dtype == 16384, (name, dtype)  # CCV_32F
            shape = [x for x in struct.unpack("<12i", dims) if x]
            if name in gate_names:
                data = exported.execute("SELECT data FROM tensors WHERE name=?", (name,)).fetchone()[0]
                actual = np.frombuffer(data, dtype=np.float32).reshape(shape)
                expected = state[f"transformer_blocks.{gate_names[name]}.attn.to_gate_compress.weight"].float().numpy()
                assert np.array_equal(actual, expected), f"Compression gate bytes differ: {name}"
                continue
            baseline = [x for x in struct.unpack("<12i", original_rows[name][1]) if x]
            head = next((prefix for prefix in ("proj_out", "audio_proj_out") if name.startswith(f"__dit__[t-{prefix}-")), None)
            if head:
                head_count += 1
                baseline[0] *= 32
                suffix = "bias" if name.endswith("-1]") else "weight"
                data = exported.execute("SELECT data FROM tensors WHERE name=?", (name,)).fetchone()[0]
                actual = np.frombuffer(data, dtype=np.float32).reshape(shape)
                expected = state[f"{head}.{suffix}"].float().numpy()
                assert np.array_equal(actual, expected), f"PDD head bytes differ: {name}"
            assert shape == baseline, (name, shape, baseline)
        assert head_count == 4
        assert exported.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    print("FastH3 export audit:", len(rows), "FP32 tensors; H3 names retained plus 50 gates; PDD/gate values exact", flush=True)


def official_pdd_functions(root):
    # Execute only these dependency-free arithmetic routines from the downloaded,
    # revision-pinned official source, without importing FastVideo's server stack.
    path = Path(root) / "reference/fastvideo/layers/pdd.py"
    tree = ast.parse(path.read_text(), filename=str(path))
    wanted = {"validate_finite_nonzero", "validate_pdd_steps", "pdd_fine_grid", "shifted_noise_delta"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PDDReplicatedLinear")
    nodes += [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"_head_view", "_fused_params"}]
    assert len(nodes) == 6
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes, type_ignores=[])
    namespace = {"torch": torch, "PDD_GRID_MAX_T": 0.999}
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


@torch.no_grad()
def head_case(root, state, device):
    torch.manual_seed(20261020)
    dev = f"cuda:{device}"
    x = torch.randn(7, 5376, device=dev, dtype=torch.float32)
    official = official_pdd_functions(root)
    grid = official["pdd_fine_grid"](32, device=dev)
    result = {"x": x.cpu()}
    for name, prefix, channels, shift in (("video", "proj_out", 96, 12.0), ("audio", "audio_proj_out", 32, 3.0)):
        weight = state[prefix + ".weight"].float().to(dev)
        bias = state[prefix + ".bias"].float().to(dev)
        result[name] = F.linear(x, weight, bias).cpu()
        weights = official["shifted_noise_delta"](grid[:-1], grid[1:], shift)
        obj = SimpleNamespace(weight=weight, bias=bias, grid_size=32, channel_layout="channel_major")
        obj._head_view = lambda t, obj=obj: official["_head_view"](obj, t)
        fused = []
        for start in range(0, 32, 4):
            w, b = official["_fused_params"](obj, start, start + 4, weights, torch.float64)
            fused.append(F.linear(x, w, b).cpu())
        result[name + "_fused"] = torch.stack(fused, dim=1)
        result[name + "_weights"] = weights.cpu()
    return result
