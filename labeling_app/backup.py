"""Backups: a consistent snapshot of labels.db (SQLite online backup API) plus
the current final_labels.json, into <data_dir>/backups/<timestamp>/ and —
optionally — a second copy into another directory (e.g. a OneDrive folder).
The LIVE database stays outside any continuously synced folder; only the
closed snapshot is copied there.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

from .bundle import Bundle
from .db import LabelDB
from .export import write_export


def make_backup(db: LabelDB, bundle: Bundle, data_dir: Path, *, copy_to: Path | None = None,
                now: str | None = None) -> dict[str, Any]:
    stamp = (now or time.strftime("%Y-%m-%d %H:%M:%S")).replace(":", "").replace(" ", "-")
    dest = Path(data_dir) / "backups" / stamp
    dest.mkdir(parents=True, exist_ok=True)
    db.snapshot_to(dest / "labels.db")
    write_export(db, bundle, dest / "final_labels.json", now=now)
    files = {}
    for p in sorted(dest.iterdir()):
        files[p.name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
    manifest = {"created_at": now or time.strftime("%Y-%m-%d %H:%M:%S"), "source_db": str(db.path), "files": files}
    (dest / "backup_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8",
                                               newline="\n")
    out = {"backup_dir": str(dest), "files": files}
    if copy_to:
        target = Path(copy_to) / dest.name
        shutil.copytree(dest, target, dirs_exist_ok=True)
        out["copied_to"] = str(target)
    return out


__all__ = ["make_backup"]
