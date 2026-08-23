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
import tempfile
import sqlite3
import time
from pathlib import Path
from typing import Any


#: tables whose row counts must survive a backup unchanged
COUNTED_TABLES = ("items", "labels", "final_labels", "graders", "events")


class BackupError(RuntimeError):
    """The snapshot is not a usable backup. Never raised for a good one, and
    never swallowed: an unverifiable snapshot must not be reported as success."""


def _counts(con: sqlite3.Connection) -> dict[str, Any]:
    """Row counts + schema stamp read through an OPEN connection (so a WAL-mode
    source is counted as the connection sees it, WAL content included)."""
    out: dict[str, Any] = {}
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for t in COUNTED_TABLES:
        out[t] = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] if t in tables else None
    if "meta" in tables:
        row = con.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        out["schema_version"] = row[0] if row else None
    return out


def verify_database(path: Path) -> dict[str, Any]:
    """Full health report of a database FILE: integrity_check, quick_check,
    foreign_key_check and row counts. Raises sqlite3.DatabaseError if the file
    is damaged badly enough that the checks cannot even run — the caller decides
    whether that is fatal."""
    p = Path(path)
    con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True, timeout=15.0)
    try:
        integrity = [r[0] for r in con.execute("PRAGMA integrity_check")]
        quick = [r[0] for r in con.execute("PRAGMA quick_check")]
        fk = con.execute("PRAGMA foreign_key_check").fetchall()
        return {"integrity_check": "ok" if integrity == ["ok"] else integrity,
                "quick_check": "ok" if quick == ["ok"] else quick,
                "foreign_key_check": "ok" if not fk else [list(r) for r in fk],
                "counts": _counts(con)}
    finally:
        con.close()


def snapshot_sqlite(src: Path, dest: Path) -> dict[str, Any]:
    """A VERIFIED, self-contained copy of a (possibly live, WAL-mode) SQLite
    database, taken with SQLite's online backup API over a READ-ONLY connection.

    WAL safety: the online backup API copies the database as the source
    CONNECTION sees it, which includes everything committed to the write-ahead
    log — it is never a main-file-only copy. The source is opened read-only, so
    this is safe while the labeling server holds its own WAL connection open,
    and the source is never written, migrated or checkpointed.

    Self-contained: the snapshot is switched to a rollback journal and its
    ``-wal``/``-shm`` sidecars are removed, so the backup is one file that can
    be copied anywhere.

    VERIFIED: the snapshot is then integrity-checked, foreign-key-checked and
    its row counts compared against the source. If any check fails the partial
    file is deleted and ``BackupError`` is raised — an incomplete or damaged
    snapshot is NEVER reported as a successful backup.
    """
    src, dest = Path(src), Path(dest)
    if not src.exists():
        raise FileNotFoundError(f"no database at {src} — nothing to back up")
    # A snapshot is written TO dest: if the arguments are swapped, that write lands on the
    # deployment. Guard the destination the same way LabelDB guards its own path.
    from .db import assert_not_live_database
    assert_not_live_database(dest)
    if dest.resolve() == src.resolve():
        raise BackupError(f"refusing to back {src} up onto itself")
    dest.parent.mkdir(parents=True, exist_ok=True)
    wal = src.with_name(src.name + "-wal")
    source_wal_bytes = wal.stat().st_size if wal.exists() else 0

    def _discard():
        for p in (dest, dest.with_name(dest.name + "-wal"), dest.with_name(dest.name + "-shm")):
            p.unlink(missing_ok=True)

    try:
        ro = sqlite3.connect(f"file:{src.as_posix()}?mode=ro", uri=True, timeout=15.0)
    except sqlite3.Error as e:
        raise BackupError(f"cannot open {src} for backup: {type(e).__name__}: {e}") from e
    try:
        try:
            source_counts = _counts(ro)          # what the SOURCE connection can actually see
            out = sqlite3.connect(str(dest))
            try:
                ro.backup(out)
                out.execute("PRAGMA journal_mode=DELETE")
            finally:
                out.close()
        except sqlite3.DatabaseError as e:
            _discard()
            raise BackupError(
                f"the source database {src} could not be snapshotted ({type(e).__name__}: {e}). "
                "It is damaged or its write-ahead log cannot be read; NO backup was written. "
                "Stop the labeling server so SQLite can checkpoint cleanly, then retry, and "
                "restore from the newest verified backup if this persists.") from e
    finally:
        ro.close()
    for side in (dest.with_name(dest.name + "-wal"), dest.with_name(dest.name + "-shm")):
        side.unlink(missing_ok=True)

    try:
        health = verify_database(dest)
    except sqlite3.DatabaseError as e:
        _discard()
        hint = ""
        if source_wal_bytes:
            hint = (f" A -wal sidecar of {source_wal_bytes} bytes sits beside the source: if the main database "
                    "was restored from a backup while that write-ahead log was left in place, the log belongs "
                    "to the OLD database and cannot be applied to the new one. Stop the labeling server, move "
                    "labels.db-wal and labels.db-shm aside, and retry — the main file alone is then the "
                    "database.")
        raise BackupError(f"the snapshot of {src} is unreadable ({type(e).__name__}: {e}); it was discarded, "
                          f"and NO backup was written.{hint}") from e
    problems = [f"{k}={health[k]}" for k in ("integrity_check", "quick_check", "foreign_key_check")
                if health[k] != "ok"]
    # An empty or table-less file passes every integrity check trivially. Counting that as a
    # verified backup is exactly the failure this function exists to prevent: it would report
    # success while preserving nothing.
    if all(health["counts"].get(t) is None for t in COUNTED_TABLES):
        problems.append("the snapshot contains none of the expected tables "
                        f"({', '.join(COUNTED_TABLES)}) — it preserves nothing")
    elif health["counts"].get("labels") is None:
        problems.append("the snapshot has no labels table — it is not a labeling database")
    if health["counts"] != source_counts:
        problems.append(f"row counts changed: source {source_counts} -> snapshot {health['counts']}")
    if problems:
        _discard()
        raise BackupError(f"the snapshot of {src} did not verify ({'; '.join(problems)}); it was discarded, "
                          "and NO backup was written")
    return {"bytes": dest.stat().st_size,
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
            "counts": health["counts"],
            "verified": {k: health[k] for k in ("integrity_check", "quick_check", "foreign_key_check")},
            "source_wal_bytes": source_wal_bytes}


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
    # 1. the database — read-only online backup; never depends on anything else.
    #    A snapshot that does not verify raises: no half-backup is left behind,
    #    and the caller is never told a damaged database was backed up.
    try:
        snap = snapshot_sqlite(src_path, dest / "labels.db")
    except Exception:
        try:
            next(dest.iterdir())
        except StopIteration:
            dest.rmdir()
        raise
    # 2. the export — best effort, never blocks
    export: dict[str, Any] = {"status": "skipped", "reason": None}
    b, reason = _try_load_bundle(bundle, bundle_dir)
    if b is None:
        export["reason"] = reason
    else:
        try:
            from .export import write_export
            # Export from a THROWAWAY copy of the verified snapshot. Not from the
            # live file (LabelDB.__init__ sets journal_mode and runs DDL — exactly
            # what this function promises not to do), and not from the snapshot
            # itself, which must stay a single self-contained file with no
            # -wal/-shm beside it.
            with tempfile.TemporaryDirectory(prefix="labeling-export-") as tmp:
                work = Path(tmp) / "labels.db"
                shutil.copy2(dest / "labels.db", work)
                data = write_export(LabelDB(work), b, dest / "final_labels.json", now=now)
            export = {"status": "written", "final_count": data.get("final_count")}
        except Exception as e:  # noqa: BLE001 — the snapshot already succeeded; record, don't fail
            export = {"status": "skipped", "reason": f"export failed ({type(e).__name__}: {e})"}
    files = {}
    for p in sorted(dest.iterdir()):
        if p.name == "backup_manifest.json":
            continue
        files[p.name] = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
    manifest = {"created_at": now or time.strftime("%Y-%m-%d %H:%M:%S"), "source_db": str(src_path),
                "db_counts": snap["counts"], "db_verified": snap["verified"],
                "source_wal_bytes": snap["source_wal_bytes"], "export": export, "files": files}
    (dest / "backup_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8",
                                               newline="\n")
    out = {"backup_dir": str(dest), "files": files, "db_counts": snap["counts"],
           "db_verified": snap["verified"], "export": export}
    if copy_to:
        target = Path(copy_to) / dest.name
        shutil.copytree(dest, target, dirs_exist_ok=True)
        out["copied_to"] = str(target)
    return out


__all__ = ["make_backup", "snapshot_sqlite", "verify_database", "BackupError", "COUNTED_TABLES"]
