"""Verify a timestamped annotation backup against the LIVE annotation store.

Confirms (1) the backup manifest matches the backup copies (restorable)
and (2) the live files are byte-identical to the backup — i.e. nothing in
a campaign modified owner annotations. Exit 1 on any difference.

    .venv/Scripts/python.exe scripts/annotation_backup_verify.py \
        [evaluation/annotation_backups/<stamp>]
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

LIVE = Path("evaluation/htr_pilot/annotations")


def main() -> int:
    if len(sys.argv) > 1:
        bk = Path(sys.argv[1])
    else:
        bk = sorted(Path("evaluation/annotation_backups").glob("*"))[-1]
    manifest = {}
    for line in (bk / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, rel = line.split(maxsplit=1)
        manifest[rel.lstrip("*")] = digest

    bad = []
    for rel, digest in manifest.items():
        if hashlib.sha256((bk / rel).read_bytes()).hexdigest() != digest:
            bad.append(("backup-corrupt", rel))
    live_files = {
        "annotations/" + str(p.relative_to(LIVE)).replace("\\", "/"):
            hashlib.sha256(p.read_bytes()).hexdigest()
        for p in LIVE.rglob("*.json")}
    for rel in sorted(set(manifest) - set(live_files)):
        bad.append(("missing-live", rel))
    for rel in sorted(set(live_files) - set(manifest)):
        bad.append(("new-since-backup", rel))
    for rel in sorted(set(manifest) & set(live_files)):
        if manifest[rel] != live_files[rel]:
            bad.append(("changed", rel))

    print(f"backup {bk.name}: {len(manifest)} files; live: {len(live_files)}")
    if bad:
        for kind, rel in bad[:20]:
            print(f"  {kind}: {rel}")
        print("RESULT: DIFFERS")
        return 1
    print("RESULT: IDENTICAL (backup restorable, live untouched)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
