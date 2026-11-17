"""Official, local Transformers Qwen3-VL reference; no downloads."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

_fp32_cases = {}


def load_state(root):
    root = Path(root) / "text_encoder"
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    keys = {key for key in index if key.startswith("model.visual.")}
    state = {}
    for shard in sorted({index[key] for key in keys}):
        with safe_open(root / shard, framework="pt", device="cpu") as tensors:
            for key in sorted(keys):
                if index[key] == shard:
                    state[key] = tensors.get_tensor(key)
    assert len(state) == 351, len(state)
    return state


def load(root, device, precision):
    torch.set_num_threads(8)
    config_path = Path(root) / "text_encoder" / "config.json"
    config = Qwen3VLVisionConfig(**json.loads(config_path.read_text())["vision_config"])
    config._attn_implementation = "eager" if precision == "float32" else "sdpa"
    state = load_state(root)
    model = Qwen3VLVisionModel(config)
    model.load_state_dict({key.removeprefix("model.visual."): value for key, value in state.items()},
                          strict=True, assign=True)
    model = model.to(device=f"cuda:{device}", dtype=getattr(torch, precision)).eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)
    print(f"Official Qwen3-VL vision: {len(state)} tensors, {precision}, {config._attn_implementation}", flush=True)
    return {"model": model, "state": state}


def check_fp16_weights(pack):
    maximum = 0.0
    maximum_key = None
    maximum_error = 0.0
    underflows = 0
    count = 0
    for key, value in pack["state"].items():
        value = value.float()
        half = value.half()
        assert torch.isfinite(value).all(), f"Source non-finite: {key}"
        assert torch.isfinite(half).all(), f"FP16 overflow: {key}"
        magnitude = value.abs().max().item()
        if magnitude > maximum:
            maximum, maximum_key = magnitude, key
        maximum_error = max(maximum_error, (half.float() - value).abs().max().item())
        underflows += ((value != 0) & (half == 0)).sum().item()
        count += value.numel()
    print(f"FP16 source safety: tensors={len(pack['state'])}, elements={count}, "
          f"max_abs={maximum} ({maximum_key}), max_cast_error={maximum_error}, "
          f"nonzero_to_zero={underflows}, no NaN/Inf/overflow", flush=True)


def text_store_names(path):
    import sqlite3
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as store:
        # Covering-index query avoids reading the >1 GB token embedding record
        # (larger than standard Python SQLite's single-record limit).
        return [row[0] for row in store.execute(
            "SELECT name FROM tensors WHERE name >= ? AND name < ? ORDER BY name",
            ("__text_model__[", "__text_model__\\"))]


def audit_store(path, previous_text_names):
    import re
    import sqlite3
    import numpy as np
    assert text_store_names(path) == list(previous_text_names), "Text tensor names changed"
    maximum, count, elements = 0.0, 0, 0
    keys = set()
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as store:
        for name, datatype, dimensions, payload in store.execute(
            "SELECT name,datatype,dim,data FROM tensors WHERE name >= ? AND name < ? ORDER BY name",
            ("__vision_model__[", "__vision_model__\\")):
            assert datatype == 131072, (name, datatype)  # CCV_16F
            shape = tuple(int(d) for d in np.frombuffer(dimensions, dtype="<i4") if d)
            values = np.frombuffer(payload, dtype="<f2")
            assert values.size == np.prod(shape), (name, shape, values.size)
            assert np.isfinite(values).all(), f"Non-finite exported weight: {name}"
            maximum = max(maximum, float(np.abs(values).max()))
            count += 1
            elements += values.size
            keys.add(name.removeprefix("__vision_model__"))
    assert count == 459, count  # 351 source + 4 extra tensors per split-QKV block.
    for stem in ["q_proj", "k_proj", "v_proj", "out_proj", "norm1", "norm2", "mlp_fc1", "mlp_fc2"]:
        matches = [key for key in keys if re.fullmatch(rf"\[t-{stem}-\d+-[01]\]", key)]
        assert len(matches) == 54, (stem, len(matches))
    for index in range(3):
        assert len([key for key in keys if key.startswith(f"[t-deepstack_{index}_")]) == 6
    for stem in ["conv_in", "norm_out", "merger_mlp_0", "merger_mlp_1"]:
        for parameter in range(2):
            assert f"[t-{stem}-0-{parameter}]" in keys, stem
    with sqlite3.connect("file:/slow/Data/qwen_2.5_vl_7b_vit_f16.ckpt?mode=ro", uri=True) as store:
        old = {row[0].removeprefix("__vit__") for row in store.execute("SELECT name FROM tensors")}
    shared = sorted(keys & old)
    print(f"Export audit: {count} FP16 vision tensors / {elements} elements, max_abs={maximum}, "
          "all finite; 18 distinct DeepStack tensors", flush=True)
    print(f"Naming audit: {len(shared)} exact shared tensor suffixes with Qwen-Image's vision export; "
          f"{len(previous_text_names)} text tensor names preserved", flush=True)


def prepare_combined_store(source, destination):
    import os
    import sqlite3
    import tempfile
    source, destination = Path(source).resolve(), Path(destination).resolve()
    assert source != destination, "Never overwrite the original text checkpoint"
    assert not destination.exists(), f"Destination already exists: {destination}"
    fd, staging = tempfile.mkstemp(prefix=f".{destination.stem}.", suffix=".ckpt", dir=destination.parent)
    os.close(fd)
    print(f"Copying original text checkpoint via SQLite backup to {staging}", flush=True)
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as old:
        with sqlite3.connect(staging) as new:
            old.backup(new, pages=8192)
    return staging


def finish_combined_store(source, staging, destination):
    import os
    import sqlite3
    import numpy as np
    maximum, elements, count = 0.0, 0, 0
    with sqlite3.connect(f"file:{source}?mode=ro", uri=True) as old:
        with sqlite3.connect(f"file:{staging}?mode=ro", uri=True) as new:
            def rows(connection):
                return dict(connection.execute(
                    "SELECT name,rowid FROM tensors WHERE name >= ? AND name < ? ORDER BY name",
                    ("__text_model__[", "__text_model__\\")))
            original_rows, copied_rows = rows(old), rows(new)
            assert original_rows.keys() == copied_rows.keys()
            assert len(original_rows) == 551
            for name, old_id in original_rows.items():
                new_id = copied_rows[name]
                metadata = "SELECT type,format,datatype,dim FROM tensors WHERE rowid=?"
                old_metadata = old.execute(metadata, (old_id,)).fetchone()
                assert old_metadata == new.execute(metadata, (new_id,)).fetchone(), name
                assert old_metadata[2] == 131072, (name, old_metadata[2])
                with old.blobopen("tensors", "data", old_id, readonly=True) as a:
                    with new.blobopen("tensors", "data", new_id, readonly=True) as b:
                        assert len(a) == len(b), name
                        while payload := a.read(8 * 1024 * 1024):
                            assert payload == b.read(len(payload)), f"Text tensor bytes changed: {name}"
                            values = np.frombuffer(payload, dtype="<f2")
                            assert np.isfinite(values).all(), f"Non-finite text weight: {name}"
                            maximum = max(maximum, float(np.abs(values).max()))
                            elements += values.size
                count += 1
                if count % 50 == 0:
                    print(f"Text byte-identity + FP16 safety verified: {count}/551 tensors", flush=True)
    # Atomically publish without replacing any existing file. Both paths are on
    # the same filesystem. Retain staging on failures for inspection.
    os.link(staging, destination)
    os.unlink(staging)
    print(f"Combined checkpoint ready: {destination}", flush=True)
    print(f"All {count} text tensors byte-identical, including metadata; "
          f"{elements} FP16 elements all finite, max_abs={maximum}", flush=True)


def check_coverage(pack, loaded):
    expected, actual = set(pack["state"]), set(loaded)
    assert actual == expected, (sorted(expected - actual), sorted(actual - expected))
    print(f"Vision source coverage: {len(actual)}/{len(expected)} tensors", flush=True)


@torch.no_grad()
def run_case(pack, grids, seed):
    model = pack["model"]
    grid = torch.tensor(grids, dtype=torch.int64, device=model.device)
    count = sum(t * h * w for t, h, w in grids)
    # Quantize input once so both implementations receive exactly the same values.
    generator = torch.Generator(device="cpu").manual_seed(seed)
    patches = torch.randn(count, 1536, generator=generator).half().to(model.dtype).to(model.device)
    captured, hooks = {}, []
    def capture(label):
        def hook(module, inputs, output):
            captured[label] = output.detach().float().cpu()
        return hook
    for layer in [8, 16, 24, 26]:
        hooks.append(model.blocks[layer].register_forward_hook(capture(f"block{layer}")))
    def capture_stem(module, inputs):
        captured["stem"] = inputs[0].detach().float().cpu()
    hooks.append(model.blocks[0].register_forward_pre_hook(capture_stem))
    try:
        output = model(patches, grid_thw=grid, return_dict=True)
    finally:
        for hook in hooks:
            hook.remove()
    outputs = [output.pooler_output, *output.deepstack_features]
    outputs = [value.detach().float().cpu() for value in outputs]
    outputs += [captured[label] for label in ["stem", "block8", "block16", "block24", "block26"]]
    key = (tuple(tuple(grid) for grid in grids), seed)
    if model.dtype == torch.float32:
        _fp32_cases[key] = outputs
    return {"patches": patches.float().cpu(), "outputs": outputs, "baseline": _fp32_cases[key]}


def release(pack):
    pack.clear()
    import gc
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/slow/Data/MiniMax-H3")
    args = parser.parse_args()
    torch.set_num_threads(8)
    check_fp16_weights({"state": load_state(args.model)})
