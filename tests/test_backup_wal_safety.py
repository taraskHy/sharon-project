"""Backups must be WAL-safe, self-contained, verified — and loud when they fail.

The incident behind this file: `snapshot_sqlite` reported success while
producing a snapshot that could not even be opened. A backup that cannot be
verified is not a backup, and silently returning one is worse than raising,
because the rotation then looks healthy while every snapshot in it is unusable.

Covered here:

* committed data that lives only in the write-ahead log IS in the backup, while
  another connection holds the database open (the exact live-server shape);
* the backup is self-contained — a single file, no -wal/-shm sidecars;
* integrity_check / quick_check / foreign_key_check all clean, row counts equal
  to the source, both recorded in the manifest;
* a damaged source raises BackupError and leaves NO file behind;
* the labels.db snapshot never depends on the bundle.

No model, network or OCR calls.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from labeling_app.backup import BackupError, make_backup, snapshot_sqlite, verify_database
from labeling_app.db import LabelDB


def _wal_source(tmp_path: Path, rows: int = 40) -> tuple[Path, sqlite3.Connection]:
    """A WAL-mode database whose committed rows are still in the -wal, with a
    second connection left OPEN so SQLite cannot checkpoint them away — the
    shape a live labeling server presents."""
    src = tmp_path / "src.db"
    writer = sqlite3.connect(str(src), isolation_level=None)
    writer.execute("PRAGMA journal_mode=WAL")
    writer.execute("PRAGMA wal_autocheckpoint=0")          # nothing drains to the main file
    writer.execute("CREATE TABLE items (item_id TEXT PRIMARY KEY, max_score REAL NOT NULL)")
    writer.execute("CREATE TABLE labels (item_id TEXT, grader TEXT, score REAL, PRIMARY KEY (item_id, grader))")
    for t in ("final_labels", "graders", "events"):
        writer.execute(f"CREATE TABLE {t} (id INTEGER PRIMARY KEY)")
    writer.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")
    writer.execute("INSERT INTO meta VALUES ('schema_version', '3')")
    for i in range(rows):
        writer.execute("INSERT INTO items VALUES (?, 4.0)", (f"g{i:04d}",))
        writer.execute("INSERT INTO labels VALUES (?, 'Erik', ?)", (f"g{i:04d}", float(i % 5)))
    reader = sqlite3.connect(str(src))                     # stays open for the whole test
    reader.execute("SELECT COUNT(*) FROM items").fetchone()
    writer.close()                                         # reader still holds the db -> no checkpoint
    return src, reader


def test_committed_wal_rows_are_in_the_backup_while_a_connection_is_open(tmp_path):
    src, reader = _wal_source(tmp_path, rows=40)
    try:
        wal = src.with_name(src.name + "-wal")
        assert wal.exists() and wal.stat().st_size > 0, "the fixture must leave data in the WAL"
        # the MAIN FILE ALONE does not have the rows — this is what a naive copy would grab
        naive = tmp_path / "naive.db"
        naive.write_bytes(src.read_bytes())
        with sqlite3.connect(f"file:{naive.as_posix()}?mode=ro", uri=True) as bad:
            try:
                main_only = bad.execute("SELECT COUNT(*) FROM items").fetchone()[0]
            except sqlite3.OperationalError:
                main_only = 0                              # not even the schema reached the main file
        assert main_only < 40, "precondition: the committed rows are not yet in the main file"

        out = snapshot_sqlite(src, tmp_path / "backup.db")
        assert out["counts"]["items"] == 40 and out["counts"]["labels"] == 40, \
            "the backup must contain everything committed, WAL included"
        assert out["source_wal_bytes"] > 0
        assert out["verified"] == {"integrity_check": "ok", "quick_check": "ok", "foreign_key_check": "ok"}
    finally:
        reader.close()


def test_the_backup_is_a_single_self_contained_file(tmp_path):
    src, reader = _wal_source(tmp_path, rows=12)
    try:
        dest = tmp_path / "out" / "backup.db"
        snapshot_sqlite(src, dest)
        assert dest.exists()
        assert not list(dest.parent.glob("backup.db-*")), "no -wal/-shm sidecars beside a backup"
        moved = tmp_path / "elsewhere.db"                  # it must survive being moved alone
        moved.write_bytes(dest.read_bytes())
        with sqlite3.connect(f"file:{moved.as_posix()}?mode=ro", uri=True) as c:
            assert c.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 12
            assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        reader.close()


def test_the_source_is_never_written(tmp_path):
    src, reader = _wal_source(tmp_path, rows=10)
    try:
        before = {p.name: (p.stat().st_size, p.stat().st_mtime_ns)
                  for p in tmp_path.glob("src.db*")}
        snapshot_sqlite(src, tmp_path / "backup.db")
        after = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in tmp_path.glob("src.db*")}
        assert after == before, "a backup must not touch the source database"
    finally:
        reader.close()


def test_a_damaged_source_raises_and_leaves_no_backup_behind(tmp_path):
    src = tmp_path / "src.db"
    con = sqlite3.connect(str(src))
    con.execute("CREATE TABLE items (item_id TEXT PRIMARY KEY)")
    con.execute("INSERT INTO items VALUES ('a')")
    con.commit()
    con.close()
    raw = bytearray(src.read_bytes())
    raw[4096:4200] = b"\x00" * 104                          # scribble over a page
    src.write_bytes(bytes(raw))

    dest = tmp_path / "backup.db"
    with pytest.raises(BackupError) as e:
        snapshot_sqlite(src, dest)
    assert "NO backup was written" in str(e.value)
    assert not dest.exists(), "a partial/damaged snapshot is discarded, never left to look like a backup"
    assert not list(tmp_path.glob("backup.db-*"))


def test_a_missing_source_is_reported_not_silently_skipped(tmp_path):
    with pytest.raises(FileNotFoundError):
        snapshot_sqlite(tmp_path / "nope.db", tmp_path / "backup.db")


def test_verify_database_reports_every_check(tmp_path):
    src, reader = _wal_source(tmp_path, rows=5)
    try:
        snapshot_sqlite(src, tmp_path / "backup.db")
        rep = verify_database(tmp_path / "backup.db")
        assert rep["integrity_check"] == "ok" and rep["quick_check"] == "ok"
        assert rep["foreign_key_check"] == "ok"
        assert rep["counts"]["items"] == 5 and rep["counts"]["schema_version"] == "3"
    finally:
        reader.close()


# ------------------------------------------------- bundle independence --

def test_backup_succeeds_with_a_missing_bundle(tmp_path):
    """A valid labels.db must always be backed up, whatever state the bundle is in."""
    data = tmp_path / "data"
    db = LabelDB(data / "labels.db")
    db.load_items([{"item_id": "g1", "max_score": 4.0, "rubric_items": [{"id": "R1", "text": "r"}]}])
    db.save_label("g1", "Erik", score=3.0, rubric=["R1"], status="saved")

    out = make_backup(db, None, data, bundle_dir=tmp_path / "no_such_bundle", now="2026-08-23 04:00:00")
    snap = Path(out["backup_dir"]) / "labels.db"
    assert snap.exists()
    assert out["db_counts"]["labels"] == 1 and out["db_counts"]["items"] == 1
    assert out["db_verified"]["integrity_check"] == "ok"
    assert out["export"]["status"] == "skipped" and "bundle missing" in out["export"]["reason"]
    manifest = json.loads((Path(out["backup_dir"]) / "backup_manifest.json").read_text(encoding="utf-8"))
    assert manifest["db_counts"]["labels"] == 1
    assert manifest["db_verified"] == {"integrity_check": "ok", "quick_check": "ok", "foreign_key_check": "ok"}
    assert "source_wal_bytes" in manifest
    assert verify_database(snap)["integrity_check"] == "ok"


def test_backup_succeeds_with_a_corrupt_bundle(tmp_path):
    data = tmp_path / "data"
    db = LabelDB(data / "labels.db")
    db.load_items([{"item_id": "g1", "max_score": 4.0, "rubric_items": [{"id": "R1", "text": "r"}]}])
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / "bundle.json").write_text("{ this is not json", encoding="utf-8")

    out = make_backup(db, None, data, bundle_dir=bundle_dir, now="2026-08-23 04:01:00")
    assert (Path(out["backup_dir"]) / "labels.db").exists()
    assert out["db_verified"]["integrity_check"] == "ok"
    assert out["export"]["status"] == "skipped" and out["export"]["reason"]


def test_backup_of_a_live_wal_database_keeps_everything(tmp_path):
    """End to end in the deployment's shape: a WAL database with an open
    connection, backed up through make_backup, verified afterwards."""
    data = tmp_path / "data"
    db = LabelDB(data / "labels.db")
    db.load_items([{"item_id": f"g{i}", "max_score": 4.0, "rubric_items": [{"id": "R1", "text": "r"}]}
                   for i in range(20)])
    for i in range(20):
        db.save_label(f"g{i}", "Erik", score=float(i % 5), rubric=["R1"], status="saved")
    holder = sqlite3.connect(str(data / "labels.db"))       # the "server" keeps its connection
    try:
        holder.execute("SELECT COUNT(*) FROM labels").fetchone()
        out = make_backup(db, None, data, now="2026-08-23 04:02:00")
        assert out["db_counts"]["items"] == 20 and out["db_counts"]["labels"] == 20
        assert out["db_verified"]["integrity_check"] == "ok"
        snap = Path(out["backup_dir"]) / "labels.db"
        with sqlite3.connect(f"file:{snap.as_posix()}?mode=ro", uri=True) as c:
            assert c.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 20
            assert c.execute("SELECT COUNT(*) FROM events").fetchone()[0] >= 20
    finally:
        holder.close()
