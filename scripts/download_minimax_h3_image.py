"""Fetch one pinned H3-Image component and its official reference, under /slow/Data."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path("/slow/Data/MiniMax-H3-Image")
REPO = "MiniMaxAI/MiniMax-H3-Image"
REVISION = "492fd8c4eca666f1085f57e30738c1e0d86b3439"
CODE_REVISION = "8069682140c675652a886736121f4e3b86f4187c"
os.environ["HF_HOME"] = str(ROOT / ".hf-home")
os.environ["HF_XET_CACHE"] = str(ROOT / ".hf-home" / "xet")

from huggingface_hub import HfApi, snapshot_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=["vae", "dit"], default="vae")
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    identity = {"repo": REPO, "revision": REVISION, "code_revision": CODE_REVISION}
    identity_path = ROOT / "download_identity.json"
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity
    else:
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    prefix = "vae_diffusers/" if args.component == "vae" else "transformer_diffusers/"
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION
    selected = [item for item in info.siblings if item.rfilename.startswith(prefix)
                or item.rfilename in {"README.md", "LICENSE", "pdd_config.json", "modular_model_index.json"}]
    print("Pinned source:", identity, flush=True)
    print("Component / bytes:", args.component, sum(item.size or 0 for item in selected), flush=True)
    snapshot_download(REPO, revision=REVISION, local_dir=ROOT,
                      allow_patterns=[item.rfilename for item in selected], max_workers=4)
    hashes = {}
    for item in selected:
        if item.lfs is None:
            continue
        with (ROOT / item.rfilename).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        assert actual == item.lfs.sha256, item.rfilename
        hashes[item.rfilename] = actual
        print("SHA256 verified:", item.rfilename, flush=True)
    sources = ["models/autoencoders/autoencoder_kl_minimax_h3_image.py"] if args.component == "vae" else [
        "models/transformers/transformer_minimax_h3_image.py"]
    for relative in sources:
        url = f"https://raw.githubusercontent.com/huggingface/diffusers/{CODE_REVISION}/src/diffusers/{relative}"
        path = ROOT / "reference" / relative
        with urllib.request.urlopen(url, timeout=45) as response:
            content = response.read()
        if path.exists():
            assert path.read_bytes() == content, f"Reference changed: {relative}"
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        hashes["reference/" + relative] = hashlib.sha256(content).hexdigest()
    (ROOT / f"{args.component}_verified.json").write_text(json.dumps(hashes, indent=2) + "\n")
    print("Download and verification complete:", args.component, flush=True)


if __name__ == "__main__":
    main()
