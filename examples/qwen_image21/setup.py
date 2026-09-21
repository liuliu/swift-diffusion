"""Fetch pinned reference sources and model assets."""
import argparse
import json
from pathlib import Path
import urllib.request

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
MODEL_REVISION = "b3179ad355be050328e483a9dfdd9e60cd62adfa"
DIFFUSERS_REVISION = "80c7ed262aeffbeb43ef13ae04baeb9b84515a69"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=["all", "transformer", "text_encoder", "vae"], default="all")
    args = parser.parse_args()
    reference = ROOT / "artifacts/reference"
    reference.mkdir(parents=True, exist_ok=True)
    sources = {
        "transformer_qwenimage21.py": "models/transformers/transformer_qwenimage21.py",
        "autoencoder_kl_qwenimage21.py": "models/autoencoders/autoencoder_kl_qwenimage21.py",
        "pipeline_qwenimage21.py": "pipelines/qwenimage21/pipeline_qwenimage21.py",
    }
    for name, path in sources.items():
        url = f"https://raw.githubusercontent.com/huggingface/diffusers/{DIFFUSERS_REVISION}/src/diffusers/{path}"
        (reference / name).write_bytes(urllib.request.urlopen(url, timeout=60).read())
    (reference / "revision.json").write_text(json.dumps({"diffusers": DIFFUSERS_REVISION, "model": MODEL_REVISION}, indent=2) + "\n")
    patterns = None if args.component == "all" else [f"{args.component}/*", "*.json", "LICENSE", "README.md", "processor/*"]
    snapshot_download("Qwen/Qwen-Image-2.1", revision=MODEL_REVISION, local_dir=ROOT / "artifacts/model", allow_patterns=patterns, max_workers=4)


if __name__ == "__main__":
    main()
