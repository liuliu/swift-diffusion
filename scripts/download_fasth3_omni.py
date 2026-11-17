"""Download the pinned, unquantized Omni release and its reference sources."""

import json
import hashlib
import os
from pathlib import Path
import urllib.request

ROOT = Path("/slow/Data/FastH3-Omni")
REPO = "FastVideo/FastVideo-FastH3-Omni-8-Step-V1"
REVISION = "53e0d26a1691aa473b6262d8196a8feb2c297df0"
os.environ["HF_HOME"] = str(ROOT / ".hf-home")
os.environ["HF_XET_CACHE"] = str(ROOT / ".hf-home" / "xet")
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

from huggingface_hub import snapshot_download


def fetch(url, destination):
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


ROOT.mkdir(parents=True, exist_ok=True)
# Pin the reference code too; subsequent runs use the original resolved commit.
identity = ROOT / "download_identity.json"
if identity.exists():
    manifest = json.loads(identity.read_text())
    assert manifest["repo"] == REPO and manifest["revision"] == REVISION
else:
    with urllib.request.urlopen(
        "https://api.github.com/repos/hao-ai-lab/FastVideo/commits/main", timeout=60
    ) as response:
        code_revision = json.load(response)["sha"]
    manifest = {"repo": REPO, "revision": REVISION, "fastvideo_revision": code_revision}
    identity.write_text(json.dumps(manifest, indent=2) + "\n")

code_revision = manifest["fastvideo_revision"]
for source in (
    "fastvideo/models/dits/minimax_h3.py",
    "fastvideo/layers/pdd.py",
    "fastvideo/configs/models/dits/minimax_h3.py",
    "docs/inference/fasth3_omniref.md",
):
    fetch(
        f"https://raw.githubusercontent.com/hao-ai-lab/FastVideo/{code_revision}/{source}",
        ROOT / "reference" / source,
    )
print("Source identity:", manifest, flush=True)
snapshot_download(REPO, revision=REVISION, local_dir=ROOT, ignore_patterns=["*.safetensors"], max_workers=4)
snapshot_download(REPO, revision=REVISION, local_dir=ROOT, max_workers=8)
expected_hashes = json.loads((ROOT / "checkpoint-identity.json").read_text())["files"]
for relative, expected in expected_hashes.items():
    with (ROOT / relative).open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    assert actual == expected, f"Source checksum mismatch: {relative}"
    print("SHA256 verified:", relative, flush=True)
print("Download complete:", ROOT, flush=True)
