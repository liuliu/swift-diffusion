"""Download pinned Pika H3 Flex adapters; reuse shared H3 encoder/VAE exports."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import urllib.request

ROOT = Path("/slow/Data/Pika-H3-Flex")
REPO = "Pika-Labs/Pika-H3-Flex"
REVISION = "0ddf55ad7067ca1e76812253bd7b4600df87f759"
CODE_REVISION = "64d81b2e98aa8aa1cf078b07a6ef3aaeae1033da"
os.environ["HF_HOME"] = str(ROOT / ".hf-home")
os.environ["HF_XET_CACHE"] = str(ROOT / ".hf-home" / "xet")
os.environ["HF_XET_HIGH_PERFORMANCE"] = "1"

from huggingface_hub import HfApi, snapshot_download


def fetch_range(relative, start, end):
    url = f"https://huggingface.co/{REPO}/resolve/{REVISION}/{relative}?header_range={start}-{end}"
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        assert response.status == 206
        assert response.headers["Content-Range"].startswith(f"bytes {start}-{end}/")
        result = response.read()
    assert len(result) == end - start + 1
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", choices=["transformer", "transformer_ref"])
    args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    identity_path = ROOT / "download_identity.json"
    identity = {"repo": REPO, "revision": REVISION, "code_revision": CODE_REVISION}
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity, "Existing source identity differs"
    else:
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    assert info.sha == REVISION
    selected = [
        item for item in info.siblings
        if item.rfilename in {"README.md", "LICENSE", "LICENSE-CODE", "NOTICE", "SHA256SUMS", "THIRD_PARTY_NOTICES.md", "modular_model_index.json"}
        or item.rfilename.startswith(("fl2va/", "ref2va/"))
        or (item.rfilename.startswith(("transformer/", "transformer_ref/", "scheduler/", "audio_scheduler/", "examples/"))
            and item.rfilename.endswith((".json", ".py", "requirements-diffusers.txt")))
        or (args.base and item.rfilename.startswith(args.base + "/") and item.rfilename.endswith(".safetensors"))
    ]
    print("Pinned source:", identity, flush=True)
    print("Selected download bytes:", sum(item.size or 0 for item in selected), flush=True)
    snapshot_download(REPO, revision=REVISION, local_dir=ROOT,
                      allow_patterns=[item.rfilename for item in selected], max_workers=8)
    for item in selected:
        if item.lfs is None:
            continue
        path = ROOT / item.rfilename
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        assert actual == item.lfs.sha256, f"Checksum mismatch: {item.rfilename}"
        print("SHA256 verified:", item.rfilename, flush=True)
    # Only the immutable headers are needed to reconstruct and hash the exact
    # official base shards from the existing lossless FP32 H3 exports.
    shards = {}
    for item in info.siblings:
        relative = item.rfilename
        if not (relative.startswith(("transformer/", "transformer_ref/")) and relative.endswith(".safetensors")):
            continue
        path = ROOT / "base_headers" / (relative + ".header")
        if not path.exists():
            prefix = fetch_range(relative, 0, 7)
            size = struct.unpack("<Q", prefix)[0]
            assert 0 < size < 2**20
            content = prefix + fetch_range(relative, 8, 7 + size)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        shards[relative] = {"size": item.size, "sha256": item.lfs.sha256}
    (ROOT / "base_shards.json").write_text(json.dumps(shards, indent=2) + "\n")
    for name in ("loader", "numerics", "wrapper", "sampler", "sampling", "schedules", "conditioning"):
        relative = f"src/pika_h3/core/{name}.py"
        path = ROOT / "reference" / relative
        with urllib.request.urlopen(
            f"https://raw.githubusercontent.com/Pika-Labs/Pika-H3-Flex/{CODE_REVISION}/{relative}",
            timeout=45,
        ) as response:
            content = response.read()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    print("Pika download and checksum verification complete:", ROOT, flush=True)


if __name__ == "__main__":
    main()
