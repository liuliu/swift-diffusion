"""Pika adapter checks using the already validated H3 graph and base exports."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import struct
import textwrap
import types

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

ROOT = Path("/slow/Data/Pika-H3-Flex")
TASKS = {
    "fl2va": ("transformer", Path("/slow/Data/minimax_h3_dit_f32.ckpt")),
    "ref2va": ("transformer_ref", Path("/slow/Data/minimax_h3_ref2va_dit_f32.ckpt")),
}


def store_parts(key):
    """Inverse of the existing H3 reader: names, split rows, and Q/K permutation."""
    prefix, suffix = key.rsplit(".", 1)
    parameter = 0 if suffix == "weight" else 1
    assert suffix in {"weight", "bias"}, key

    def name(label, layer=0):
        return f"__dit__[t-{label}-{layer}-{parameter}]"

    if prefix.startswith(("transformer_blocks.", "token_refiner.refiner_blocks.")):
        refiner = prefix.startswith("token_refiner.")
        parts = prefix.split(".")
        layer = int(parts[2] if refiner else parts[1])
        leaf = ".".join(parts[3:] if refiner else parts[2:])
        stem = "refiner_" if refiner else ""
        simple = {
            "norm1": "norm1", "norm2": "norm2", "attn.to_q": "q",
            "attn.to_k": "k", "attn.to_v": "v", "attn.to_out.0": "o",
            "attn.norm_q": "norm_q", "attn.norm_k": "norm_k", "ff.net.2": "down",
        }
        if leaf in simple:
            permutation = "none"
            if not refiner and leaf in {"attn.to_q", "attn.to_k"}:
                permutation = "qk"
            if not refiner and leaf in {"attn.norm_q", "attn.norm_k"}:
                permutation = "qk_norm"
            return [name(stem + simple[leaf], layer)], permutation
        if leaf == "ff.net.0.proj":
            return [name(stem + "up", layer), name(stem + "gate", layer)], "none"
        if leaf == "adaln_proj.linear" and not refiner:
            return [name(f"adaln_{chunk}_{modality}", layer)
                    for modality in range(3) for chunk in range(6)], "none"
        raise ValueError(key)
    if prefix == "norm_out.linear":
        return [name("norm_out_shift"), name("norm_out_scale")], "none"
    label = {
        "time_embedder.linear_1": "time_embedder_linear_1",
        "time_embedder.linear_2": "time_embedder_linear_2",
        "token_refiner.final_norm": "refiner_final_norm",
        "norm_out.norm": "norm_out",
        "context_embedder": "context_embedder", "proj_in": "proj_in",
        "audio_proj_in": "audio_proj_in", "proj_out": "proj_out",
        "audio_proj_out": "audio_proj_out",
    }[prefix]
    return [name(label)], "none"


def undo_qk(value, norm=False):
    heads = 1 if norm else 56
    value = value.reshape(heads, 128, -1)
    leading = value[:, :96].reshape(heads, 48, 2, -1).transpose(1, 2)
    return torch.cat((leading.reshape(heads, 96, -1), value[:, 96:]), dim=1)


def base_fingerprint(path):
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "inode": stat.st_ino, "device": stat.st_dev}


class BaseState:
    def __init__(self, task, require_verified=True):
        self.task = task
        self.trunk, self.path = TASKS[task]
        config = json.loads((ROOT / self.trunk / "config.json").read_text())
        for key, value in {
            "hidden_size": 5376, "num_layers": 50, "num_refiner_layers": 2,
            "num_attention_heads": 56, "attention_head_dim": 128, "ffn_dim": 14336,
            "in_channels": 24, "audio_in_channels": 32, "patch_size": [1, 2, 2],
            "text_dim": 5120, "freq_dim": 256, "time_embed_hidden_dim": 5376,
            "time_embed_dim": 2688, "rope_freq_dim": 16, "rope_theta": 10000.0,
            "norm_eps": 1e-5, "qk_norm_eps": 1e-5, "final_norm_eps": 1e-5,
        }.items():
            assert config[key] == value, (key, config[key], value)
        self.weight_map = json.loads((ROOT / self.trunk / "diffusion_pytorch_model.safetensors.index.json").read_text())["weight_map"]
        self.headers = {}
        for shard in sorted(set(self.weight_map.values())):
            content = (ROOT / "base_headers" / self.trunk / (shard + ".header")).read_bytes()
            self.headers.update({k: v for k, v in json.loads(content[8:]).items() if k != "__metadata__"})
        assert set(self.headers) == set(self.weight_map)
        assert len(self.weight_map) == 638
        self.db = sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)
        self.rows = {name: (dtype, dims) for name, dtype, dims in self.db.execute("SELECT name, datatype, dim FROM tensors")}
        expected = [name for key in self.weight_map for name in store_parts(key)[0]]
        assert len(expected) == len(set(expected)) == 2392
        assert set(expected) == set(self.rows)
        assert all(dtype == 16384 for dtype, _ in self.rows.values())
        self.accessed = set()
        if require_verified:
            certificate = json.loads((ROOT / f"{task}_base_verified.json").read_text())
            assert certificate["base"] == base_fingerprint(self.path), "Base checkpoint changed since hash verification"
            assert certificate["identity"] == json.loads((ROOT / "download_identity.json").read_text())
            expected_shards = json.loads((ROOT / "base_shards.json").read_text())
            assert certificate["shards"] == {f"{self.trunk}/{name}": expected_shards[f"{self.trunk}/{name}"]["sha256"] for name in set(self.weight_map.values())}

    def __contains__(self, key):
        return key in self.weight_map

    def __getitem__(self, key):
        self.accessed.add(key)
        names, permutation = store_parts(key)
        tensors = []
        for name in names:
            blob = self.db.execute("SELECT data FROM tensors WHERE name=?", (name,)).fetchone()[0]
            shape = tuple(x for x in struct.unpack("<12i", self.rows[name][1]) if x)
            assert len(blob) == int(np.prod(shape)) * 4
            # SQLite returns immutable bytes; copy before exposing them to torch.
            tensors.append(torch.from_numpy(np.frombuffer(blob, dtype=np.float32).copy()).reshape(shape))
        value = torch.cat(tensors, dim=0) if len(tensors) > 1 else tensors[0]
        if permutation != "none":
            value = undo_qk(value, permutation == "qk_norm")
        return value.reshape(self.headers[key]["shape"]).contiguous()

    def release(self):
        gc.collect()


def forward_qk(value, norm=False):
    heads = 1 if norm else 56
    value = value.reshape(heads, 128, -1)
    leading = value[:, :96].reshape(heads, 2, 48, -1).transpose(1, 2)
    return torch.cat((leading.reshape(heads, 96, -1), value[:, 96:]), dim=1)


class MergedState(BaseState):
    def __init__(self, task, device):
        super().__init__(task)
        self.manifest = json.loads((ROOT / task / "manifest.json").read_text())
        assert self.manifest["rank"] == self.manifest["alpha"] == 128
        assert self.manifest["trunk"] == self.trunk
        assert self.manifest["base"] == {
            "repo_id": "MiniMaxAI/MiniMax-H3", "revision": "42ed227ee7df40d41602854ae760620d6eb651fe"}
        assert self.manifest["numerics_policy"] == (
            "bf16_conditioning_v1" if task == "fl2va" else "fp32_conditioning_v1")
        assert self.manifest["sampling"] == {
            "knots": [1000, 750, 500, 250, 100, 40, 0],
            "video_shift": 12.0, "audio_shift": 3.0, "method": "renoise"}
        adapter_path = ROOT / task / "adapter.safetensors"
        with adapter_path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == self.manifest["sha256"], "Adapter checksum mismatch"
        self.adapter = load_file(str(adapter_path))
        self.adapter_keys = set()
        for prefix, count in (("transformer_blocks", 50), ("token_refiner.refiner_blocks", 2)):
            for layer in range(count):
                for leaf in ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2"):
                    base_key = f"{prefix}.{layer}.{leaf}.weight"
                    rows, cols = self.headers[base_key]["shape"]
                    for side, shape in (("A", [128, cols]), ("B", [rows, 128])):
                        name = base_key.removesuffix(".weight") + f".lora_{side}.default.weight"
                        self.adapter_keys.add(name)
                        assert list(self.adapter[name].shape) == shape
                        assert self.adapter[name].dtype == torch.float32
        assert set(self.adapter) == self.adapter_keys and len(self.adapter_keys) == 624
        self.adapter_accessed = set()
        self.checked = set()
        self.device = f"cuda:{device}"
        self.generator = torch.Generator(device=self.device).manual_seed(20261029)
        self.rounded = False
        self.keep_native_fp32 = False
        self.expected_hashes = {}
        self.safety = {"elements": 0, "rounded_to_zero": 0, "max_abs": 0.0,
                       "max_rounding_abs": 0.0, "worst_merge_relative": 0.0,
                       "minimum_merge_cosine": 1.0, "adapted_matrices": 0}

    @torch.no_grad()
    def __getitem__(self, key):
        value = super().__getitem__(key)
        adapter_prefix = key.removesuffix(".weight")
        key_a, key_b = (adapter_prefix + f".lora_{side}.default.weight" for side in ("A", "B"))
        check = key not in self.checked
        if key.endswith(".weight") and key_a in self.adapter:
            a = self.adapter[key_a].to(self.device)
            b = self.adapter[key_b].to(self.device)
            base = value.to(self.device)
            # Merge the released FP32 adapters at alpha/rank = 1, before the
            # existing H3 Q/K permutation and FFN split. Do not bake runtime scales.
            value = base + b @ a
            self.adapter_accessed.update((key_a, key_b))
            if check:
                x = torch.randn(4, base.shape[1], device=self.device, generator=self.generator)
                separate = F.linear(x, base) + F.linear(F.linear(x, a), b)
                merged = F.linear(x, value)
                relative = float((merged - separate).abs().max() / separate.abs().max().clamp_min(1e-6))
                cosine = float(F.cosine_similarity(merged.double(), separate.double(), dim=-1).min())
                assert torch.isfinite(merged).all() and relative < 1e-4 and cosine > 0.99999, (key, relative, cosine)
                self.safety["worst_merge_relative"] = max(self.safety["worst_merge_relative"], relative)
                self.safety["minimum_merge_cosine"] = min(self.safety["minimum_merge_cosine"], cosine)
                self.safety["adapted_matrices"] += 1
            value = value.cpu()
        half = value.half()
        if check:
            assert torch.isfinite(value).all() and torch.isfinite(half).all(), (key, "nonfinite or FP16 overflow")
            self.safety["elements"] += value.numel()
            self.safety["rounded_to_zero"] += int(((value != 0) & (half == 0)).sum())
            self.safety["max_abs"] = max(self.safety["max_abs"], float(value.abs().max()))
            self.safety["max_rounding_abs"] = max(self.safety["max_rounding_abs"], float((value - half.float()).abs().max()))
            self.checked.add(key)
        if self.rounded:
            names, permutation = store_parts(key)
            retained = self.keep_native_fp32 and self.headers[key]["dtype"] == "F32"
            stored_value = value if retained else half
            if names[0] not in self.expected_hashes:
                stored = stored_value
                if permutation != "none":
                    stored = forward_qk(stored, permutation == "qk_norm")
                stored = stored.reshape(value.shape)
                chunks = stored.chunk(len(names), dim=0) if len(names) > 1 else [stored]
                for name, chunk in zip(names, chunks):
                    self.expected_hashes[name] = hashlib.sha256(memoryview(chunk.contiguous().view(torch.uint8).numpy())).hexdigest()
            return stored_value.float()
        return value

    def validate_coverage(self, names):
        assert self.accessed == set(self.weight_map), (set(self.weight_map) - self.accessed)
        assert self.adapter_accessed == self.adapter_keys
        assert self.checked == set(self.weight_map)
        assert self.safety["adapted_matrices"] == 312
        actual = {f"__dit__[{name}]" for name in names}
        assert len(names) == len(actual) == 2392
        assert actual == set(self.rows)
        if self.rounded:
            assert set(self.expected_hashes) == actual
        print(self.task, "coverage: 638 base tensors, 624 adapter tensors, 312 merged matrices; 2392 unchanged H3 names", flush=True)
        print(self.task, "FP16 safety and separate-vs-merged LoRA checks:", json.dumps(self.safety), flush=True)


def audit_export(path, state):
    with sqlite3.connect(Path(path).as_uri() + "?mode=ro", uri=True) as db:
        rows = db.execute("SELECT name, datatype, dim FROM tensors").fetchall()
        assert len(rows) == 2392 and {name for name, _, _ in rows} == set(state.expected_hashes)
        for name, dtype, dims in rows:
            assert dtype == 131072, (name, dtype)  # CCV_16F, every exported tensor.
            assert dims == state.rows[name][1], name
            blob = db.execute("SELECT data FROM tensors WHERE name=?", (name,)).fetchone()[0]
            assert hashlib.sha256(blob).hexdigest() == state.expected_hashes[name], (name, "FP16 value mismatch")
        assert db.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    print(state.task, "export audit: all 2392 FP16 tensors exactly match expected merged weights; SQLite OK", flush=True)


def write_sidecar(path, state):
    record = {"model_key": "dit", "dtype": "float16", "runtime_scaling_baked_in": False,
              "merge": "FP32 base + FP32 B @ A * (alpha / rank), then FP16 round-to-nearest",
              "identity": json.loads((ROOT / "download_identity.json").read_text()),
              "manifest": state.manifest, "weight_safety": state.safety,
              "base_verification": json.loads((ROOT / f"{state.task}_base_verified.json").read_text())}
    Path(path + ".inference.json").write_text(json.dumps(record, indent=2) + "\n")


def verify_base(task, limit=0):
    torch.set_num_threads(8)
    state = BaseState(task, require_verified=False)
    expected = json.loads((ROOT / "base_shards.json").read_text())
    verified = {}
    for index, shard in enumerate(sorted(set(state.weight_map.values()))):
        if limit and index >= limit:
            break
        relative = f"{state.trunk}/{shard}"
        header = (ROOT / "base_headers" / (relative + ".header")).read_bytes()
        schema = json.loads(header[8:])
        schema.pop("__metadata__", None)
        checksum = hashlib.sha256(header)
        offset = 0
        for key, entry in sorted(schema.items(), key=lambda pair: pair[1]["data_offsets"][0]):
            assert entry["data_offsets"][0] == offset
            value = state[key]
            dtype = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}[entry["dtype"]]
            source = value.to(dtype)
            assert torch.equal(source.float(), value), f"Lossy source reconstruction: {key}"
            raw = source.view(torch.uint8).numpy()
            offset += raw.nbytes
            assert offset == entry["data_offsets"][1]
            checksum.update(memoryview(raw))
            del value, source, raw
        assert len(header) + offset == expected[relative]["size"]
        actual = checksum.hexdigest()
        assert actual == expected[relative]["sha256"], f"Existing H3 base differs from Pika's pinned source: {relative}: {actual}"
        verified[relative] = actual
        print("Exact official base SHA256 reproduced from existing FP32 checkpoint:", relative, flush=True)
    if len(verified) == len(set(state.weight_map.values())):
        certificate = {"base": base_fingerprint(state.path), "shards": verified,
                       "identity": json.loads((ROOT / "download_identity.json").read_text())}
        (ROOT / f"{task}_base_verified.json").write_text(json.dumps(certificate, indent=2) + "\n")
        print(task, "base fully verified; no base-weight download required", flush=True)


def python_storage_parity(task, device, keep_native_fp32=False):
    """Run the converter's existing Python reference independently of its build."""
    torch.set_grad_enabled(False)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ["H3_FP32_ATTENTION_CONTROL"] = "1"
    os.environ["H3_PARITY_PROGRESS"] = "1"
    path = Path(__file__).resolve().parent.parent / "minimax_h3_converter/main.swift"
    swift = path.read_text()
    source = swift.split('builtins.exec(\n  #"""', 1)[1].split('\n  """#,', 1)[0]
    reference = types.ModuleType("existing_h3_reference")
    exec(compile(textwrap.dedent(source), str(path) + ":embedded_python", "exec"), reference.__dict__)
    state = MergedState(task, device)
    state.keep_native_fp32 = keep_native_fp32
    print(task, "retain original FP32 projections:", keep_native_fp32, flush=True)
    results = []
    for rounded in (False, True):
        state.rounded = rounded
        print(task, "Python full 50-layer reference, FP16-rounded weights:", rounded, flush=True)
        pack = reference.load_one_block_shell(str(ROOT), device, state.trunk, 1, True, state)
        results.append(reference.run_full_shell_case(pack, 50, 52, device, True))
        reference.release_one_block_shell(pack)
    state.validate_coverage([name[len("__dit__["):-1] for name in state.rows])
    metrics = {}
    passed = True
    for name in ("video", "audio"):
        original = results[0][name + "_output"].double()
        rounded = results[1][name + "_output"].double()
        assert original.shape == rounded.shape
        error = rounded - original
        max_abs = float(error.abs().max())
        relative = max_abs / max(float(original.abs().max()), 1e-6)
        nrmse = float(error.square().mean().sqrt() / original.square().mean().sqrt())
        cosine = F.cosine_similarity(rounded, original, dim=-1)
        metrics[name] = {"max_abs": max_abs, "max_abs_over_max_ref": relative,
                         "nrmse": nrmse, "token_cosine_min": float(cosine.min()),
                         "token_cosine_mean": float(cosine.mean())}
        print(task, name, "FP16 storage vs FP32 weights:", json.dumps(metrics[name]), flush=True)
        passed = bool(torch.isfinite(rounded).all()) and relative < 0.02 and nrmse < 0.01 and float(cosine.min()) > 0.999 and passed
    suffix = "_native_f32" if keep_native_fp32 else ""
    (ROOT / f"{task}_python_storage_validation{suffix}.json").write_text(
        json.dumps({"metrics": metrics, "weight_safety": state.safety,
                    "passed": passed, "keep_native_fp32": keep_native_fp32,
                    "identity": json.loads((ROOT / "download_identity.json").read_text()),
                    "scope": "Python FP16 storage validation only; Swift parity is still required"}, indent=2) + "\n")
    np.savez(ROOT / f"{task}_python_storage_outputs{suffix}.npz", **{
        f"{precision}_{name}": result[name + "_output"].numpy()
        for precision, result in zip(("fp32", "storage"), results) for name in ("video", "audio")})
    assert passed, "FP16 storage output comparison failed; report and output arrays saved"
    print(task, "Python storage validation passed; no checkpoint exported", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify-base", choices=list(TASKS))
    action.add_argument("--python-storage-parity", choices=list(TASKS))
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--keep-native-fp32", action="store_true")
    parser.add_argument("--limit-shards", type=int, default=0)
    args = parser.parse_args()
    if args.verify_base:
        verify_base(args.verify_base, args.limit_shards)
    else:
        python_storage_parity(args.python_storage_parity, args.device, args.keep_native_fp32)
