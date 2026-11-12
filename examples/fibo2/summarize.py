"""Consolidate completed parity runs without importing the model stack."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REQUIRED = {
    "dev/vae_float32": 9,
    "dev/text_float32": 96,
    "dev/dit_float32": 156,
    "turbo/dit_float32": 156,
    "dev/pipeline_float32": 59,
    "turbo/pipeline_float32": 39,
}
summary = {
    "status": "FP32 parity passed",
    "checkpoint_exported": False,
    "ccv_revision": "329ab6c52085dff8e0de931824f2ee1b6d64b937",
    "s4nnc_revision": "5bb3fb70e0095290ce9cb0ac5090748af1d8e9a8",
    "protocol": {
        "compute_dtype": "float32",
        "weight_values": "Original checkpoint values, promoted to float32 without quantization.",
        "tf32": False,
        "acceptance": "Equal shapes, finite outputs, max_abs/max_reference_abs <= 0.001 and relative L2 <= 0.001.",
        "vae": "Seeds 42..44; 32x48, 64x64, 96x128; encoder moments, normalized posterior mode, decoder.",
        "text": "Seeds 42..45; random IDs, real prompt, one image, two images; all 15 hidden taps and six bundles.",
        "dit": "Seeds 42..44; zero/one/two references; raw latent 4x6, 6x4, 8x12; gist counts 256,256,320.",
        "pipeline": "Seed 1042; 64x96 edit; native preprocessing; Swift vision/text/VAE conditioning; "
                    "dev two steps with CFG=5 and turbo four steps with CFG=1; every velocity and latent, decoded pixels.",
        "scope": "Single-item batches and small numerical fixtures; production-resolution performance and padded batches are not validated.",
    },
    "runs": {},
}
for name, count in REQUIRED.items():
    path = HERE / "artifacts" / (name + ".json")
    report = json.loads(path.read_text())
    rows = report["results"]
    assert report.get("selected_case") is None, (name, "partial run")
    assert len(rows) == count and all(r["passed"] for r in rows), name
    summary["runs"][name] = {
        **({"dit_layout": report.get("dit_layout", "fused_adaln_v1")}
           if "/dit_" in name or "/pipeline_" in name else {}),
        "model_revision": report["model_revision"],
        "diffusers_revision": report["diffusers_revision"],
        "versions": report["versions"],
        "checks": len(rows),
        "max_absolute_error_across_all_taps": max(r["max_abs"] for r in rows),
        "max_relative_l2_across_all_taps": max(r["relative_l2"] for r in rows),
        "outputs": [
            {k: r[k] for k in ("label", "shape", "max_abs", "max_relative", "relative_l2", "cosine")}
            for r in rows
            if r["label"].endswith(("/encoder", "/normalized", "/decoder", "/bundle_5", "/output", "/decoded", "/pixels"))
        ],
        "full_report": str(path.relative_to(HERE)),
    }
summary["total_checks"] = sum(run["checks"] for run in summary["runs"].values())
summary["experimental_bfloat16"] = {
    "status": "Observed drift exceeds the original 5% error limit; user accepted this drift. Full BF16 suite not completed.",
    "failures": [],
}
for name in ("vision_bfloat16_trace.log", "dit_dev_bfloat16.log"):
    path = HERE / "artifacts" / name
    if path.exists():
        for line in path.read_text().splitlines():
            if line.startswith('{"label":'):
                row = json.loads(line)
                if not row["passed"]:
                    summary["experimental_bfloat16"]["failures"].append({
                        k: row[k] for k in ("label", "max_abs", "max_relative", "relative_l2", "tolerance")
                    })
                    break
output = HERE / "parity-results.json"
FP16 = {
    "dev/text_float16": 96,
    "dev/dit_float16": 208,
    "turbo/dit_float16": 208,
    "dev/pipeline_float16": 59,
    "turbo/pipeline_float16": 39,
}
if all((HERE / "artifacts" / (name + ".json")).exists() for name in FP16):
    runs = {}
    for name, count in FP16.items():
        path = HERE / "artifacts" / (name + ".json")
        report = json.loads(path.read_text())
        rows = report["results"]
        assert report.get("selected_case") is None and report["reference_dtype"] == "float32", name
        assert len(rows) == count, name
        assert all("max_abs" in row for row in rows), (name, "nonfinite outputs")
        runs[name] = {
            **({"dit_layout": report.get("dit_layout", "fused_adaln_v1")}
               if "/dit_" in name or "/pipeline_" in name else {}),
            "checks": count,
            "passed_checks": sum(row["passed"] for row in rows),
            "failed_metrics": [
                {k: row[k] for k in ("label", "max_abs", "max_relative", "relative_l2", "tolerance")}
                for row in rows if not row["passed"]
            ],
            "model_revision": report["model_revision"],
            "diffusers_revision": report["diffusers_revision"],
            "versions": report["versions"],
            "max_absolute_error_across_all_taps": max(row["max_abs"] for row in rows),
            "max_relative_l2_across_all_taps": max(row["relative_l2"] for row in rows),
            "outputs": [
                {k: row[k] for k in ("label", "shape", "max_abs", "max_relative", "relative_l2", "cosine")}
                for row in rows
                if row["label"].endswith(("/output", "/bundle_5", "/decoded", "/pixels"))
            ],
            "full_report": str(path.relative_to(HERE)),
        }
    summary["float16"] = {
        "status": ("FP16 parity passed against FP32" if all(run["passed_checks"] == run["checks"] for run in runs.values())
                   else "FP16 outputs finite; some final image metrics exceed the unchanged 5% threshold. See failed_metrics."),
        "checks": sum(FP16.values()),
        "passed_checks": sum(run["passed_checks"] for run in runs.values()),
        "acceptance": "Equal shapes, finite outputs, max_abs/max_reference_abs <= 0.05 and relative L2 <= 0.05; FP32 VAE checks retain 0.001.",
        "precision": "FP16 text/vision/DiT; FP32 norm reductions, DiT RoPE and SiLU; FP32 VAE and sampler state.",
        "extra_activation_rescaling": False,
        "residuals_promoted_to_fp32": False,
        "additional_dit_case": "512x512 target, 256x384 and 384x256 references, 257 text tokens, timestep 1/64.",
        "scope": "CUDA; four text/vision fixtures, four DiT fixtures per checkpoint, and the same 64x96 edit pipeline fixtures as FP32. Metal and longer production runs are not tested.",
        "runs": runs,
    }
exports = []
for variant in ("dev", "turbo"):
    manifest_path = HERE / "artifacts" / variant / "export_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        checkpoint = Path(manifest["checkpoint"])
        assert checkpoint.is_file() and checkpoint.stat().st_size == manifest["bytes"]
        assert manifest["reload"]["passed"] and manifest["parity_checks"] == 208
        exports.append(manifest)
summary["checkpoint_exported"] = bool(exports)
if exports:
    summary["exports"] = exports
output.write_text(json.dumps(summary, indent=2) + "\n")
print(f"{summary['total_checks']} FP32 checks passed; {output}")
if "float16" in summary:
    print(f"{summary['float16']['passed_checks']}/{summary['float16']['checks']} FP16 checks passed against FP32")
