"""Backups: a consistent snapshot of labels.db (SQLite online backup API) plus
— when the bundle can be loaded — the current final_labels.json, into
<data_dir>/backups/<timestamp>/ and, optionally, a second copy into another
directory (e.g. a OneDrive folder). The LIVE database stays outside any
continuously synced folder; only the closed snapshot is copied there.

ROBUSTNESS CONTRACT: backing up labels.db never depends on the bundle.
A missing, moved or corrupt bundle only means the final_labels.json export is
skipped (recorded in backup_manifest.json with the reason); the database
snapshot is always taken first, through a READ-ONLY connection, so a backup
can never be blocked by — or mutate — anything else.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any


def snapshot_sqlite(src: Path, dest: Path) -> dict[str, Any]:
    """Consistent copy of a (possibly live, WAL-mode) SQLite file using the
    online backup API over a READ-ONLY connection. No schema migration, no
    write to the source. Returns {bytes, sha256, counts}."""
    src, dest = Path(src), Path(dest)
    if not src.exists():
        raise FileNotFoundError(f"no database at {src} — nothing to back up")
    dest.parent.mkdir(parents=True, exist_ok=True)
    ro = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True, timeout=15.0)
    try:
        out = sqlite3.connect(str(dest))
        try:
            ro.backup(out)
            # The backup API copies the source's WAL journal mode into the
            # snapshot; switch it to a single self-contained file so no
            # -wal/-shm sidecars ever appear next to a backup.
            out.execute("PRAGMA journal_mode=DELETE")
        finally:
            out.close()
    finally:
        ro.close()
    for side in (dest.with_name(dest.name + "-wal"), dest.with_name(dest.name + "-shm")):
        side.unlink(missing_ok=True)
    counts = _quick_counts(dest)
    return {"bytes": dest.stat().st_size, "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(), "counts": counts}


def _quick_counts(db_path: Path) -> dict[str, Any]:
    """Self-describing numbers for the manifest (read-only; tolerant of older
    or newer schemas — missing tables simply report null)."""
    out: dict[str, Any] = {}
    try:
        c = sqlite3.connect(f"file:{Path(db_path).as_posix()}?mode=ro", uri=True)
        try:
            tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for t in ("items", "labels", "final_labels", "graders", "events"):
                out[t] = c.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] if t in tables else None
            if "meta" in tables:
                row = c.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
                out["schema_version"] = row[0] if row else None
        finally:
            c.close()
    except sqlite3.Error as e:  # pragma: no cover — counts are informational
        out["error"] = str(e)
    return out


def _try_load_bundle(bundle, bundle_dir: Path | None):
    """Return (Bundle | None, reason). Never raises."""
    from .bundle import Bundle
    if bundle is not None:
        return bundle, None
    if bundle_dir is None:
        return None, "no bundle directory given"
    try:
        return Bundle(Path(bundle_dir)), None
    except FileNotFoundError as e:
        return None, f"bundle missing: {e}"
    except Exception as e:  # noqa: BLE001 — corrupt/unreadable bundle must not block the DB snapshot
        return None, f"bundle unreadable ({type(e).__name__}: {e})"


def make_backup(db, bundle, data_dir: Path, *, copy_to: Path | None = None, now: str | None = None,
                bundle_dir: Path | None = None) -> dict[str, Any]:
    """Snapshot the database (always) and export FINAL labels (when a bundle is
    available). ``db`` may be a LabelDB or the path of labels.db; ``bundle``
    may be a Bundle or None (then ``bundle_dir`` is tried, if given)."""
    from .db import LabelDB
    stamp = (now or time.strftime("%Y-%m-%d %H:%M:%S")).replace(":", "").replace(" ", "-")
    dest = Path(data_dir) / "backups" / stamp
    dest.mkdir(parents=True, exist_ok=True)
    src_path = Path(db.path) if isinstance(db, LabelDB) else Path(db)
    # 1. the database — read-only online backup; never depends on anything else
    snap = snapshot_sqlite(src_path, dest / "labels.db")
    # 2. the export — best effort, never blocks
    export: dict[str, Any] = {"status": "skipped", "reason": None}
    b, reason = _try_load_bundle(bundle, bundle_dir)
    if b is None:
        export["reason"] = reason
    else:
        try:
            from .export import write_export
            live_db = db if isinstance(db, LabelDB) else LabelDB(src_path)
            data = write_export(live_db, b, dest / "final_labels.json", now=now)
            export = {"status": "written", "final_count": data.get("final_count")}
        except Exception as e:  # noqa: BLE001 — the snapshot already succeeded; record, don't fail
            export = {"status": "skipped", "reason": f"export failed ({type(e).__name__}: {e})"}
    files = {}
    for p in sorted(dest.iterdir()):
        if p.name == "backup_manifest.json":
            continue
        files[p.name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
    manifest = {"created_at": now or time.strftime("%Y-%m-%d %H:%M:%S"), "source_db": str(src_path),
                "db_counts": snap["counts"], "export": export, "files": files}
    (dest / "backup_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8",
                                               newline="\n")
    out = {"backup_dir": str(dest), "files": files, "db_counts": snap["counts"], "export": export}
    if copy_to:
        target = Path(copy_to) / dest.name
        shutil.copytree(dest, target, dirs_exist_ok=True)
        out["copied_to"] = str(target)
    return out


__all__ = ["make_backup", "snapshot_sqlite"]
