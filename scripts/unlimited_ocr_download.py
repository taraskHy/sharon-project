"""Snapshot-download baidu/Unlimited-OCR at the pinned revision.

Run with .venv-unlimited. Downloads files ONLY (huggingface_hub
snapshot_download) — never imports or executes the model's remote code.
The bundled sglang wheel/ directory is excluded (not needed; serving
stacks are out of scope). Writes evaluation/unlimited_ocr/
download_manifest.json with the resolved snapshot path, per-file sizes,
and SHA256 hashes of every .py/config/tokenizer file plus the safetensors
weight shard.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ID = "baidu/Unlimited-OCR"
REVISION = "07dea832e22aefee32ad281d4b80551282e1c168"
OUT = Path(__file__).resolve().parents[1] / "evaluation" / "unlimited_ocr"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    t0 = time.monotonic()
    snap = Path(snapshot_download(
        REPO_ID, revision=REVISION, ignore_patterns=["wheel/*"],
    ))
    dl_s = round(time.monotonic() - t0, 1)
    files = []
    for p in sorted(snap.rglob("*")):
        if not p.is_file():
            continue
        rec = {"file": p.name, "bytes": p.stat().st_size}
        if p.suffix in (".py", ".json") or "tokenizer" in p.name or p.suffix == ".safetensors":
            t1 = time.monotonic()
            rec["sha256"] = sha256(p)
            rec["hash_s"] = round(time.monotonic() - t1, 1)
        files.append(rec)
    manifest = {
        "repo_id": REPO_ID,
        "revision": REVISION,
        "snapshot_path": str(snap),
        "download_s": dl_s,
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
        "fetched": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "download_manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=1))
    for f in files:
        print(f"  {f['file']}: {f['bytes']:,} B {f.get('sha256', '')[:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
