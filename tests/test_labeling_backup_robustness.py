"""Regression: `labeling_app backup` must back up labels.db even when the
bundle is missing or corrupt (the export is skipped and recorded, the
database snapshot always succeeds), and must never write to the source DB."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from labeling_app.backup import make_backup, snapshot_sqlite
from labeling_app.cli import main
from labeling_app.db import LabelDB


def _seed_db(data_dir: Path) -> Path:
    db = LabelDB(data_dir / "labels.db")
    db.load_items([{"item_id": "gaaaaaaaaaa", "max_score": 4.0, "rubric_items": [{"id": "R1", "text": "r"}]},
                   {"item_id": "gbbbbbbbbbb", "max_score": 4.0, "rubric_items": []}])
    db.save_label("gaaaaaaaaaa", "Erik", score=3.0, rubric=["R1"], note="n", status="saved", expected_revision=0)
    db.save_label("gbbbbbbbbbb", "Erik", score=1.5, rubric=[], status="saved", expected_revision=0)
    db.close()
    return data_dir / "labels.db"


def test_backup_succeeds_with_valid_db_and_missing_bundle(tmp_path):
    data_dir = tmp_path / "data"
    db_path = _seed_db(data_dir)
    before = db_path.read_bytes()
    out = make_backup(db_path, None, data_dir, bundle_dir=data_dir / "bundle", now="2026-08-23 00:10:00")
    d = Path(out["backup_dir"])
    assert (d / "labels.db").exists() and (d / "backup_manifest.json").exists()
    assert not (d / "final_labels.json").exists()
    assert out["export"]["status"] == "skipped" and "bundle missing" in out["export"]["reason"]
    assert out["db_counts"]["labels"] == 2 and out["db_counts"]["items"] == 2 and out["db_counts"]["schema_version"]
    man = json.loads((d / "backup_manifest.json").read_text(encoding="utf-8"))
    assert man["export"]["status"] == "skipped" and man["db_counts"]["labels"] == 2
    assert set(man["files"]) == {"labels.db"}
    # the snapshot is a consistent, readable copy with the labels
    snap = LabelDB(d / "labels.db")
    assert snap.get_label("gaaaaaaaaaa", "Erik")["score"] == 3.0 and snap.get_label("gbbbbbbbbbb", "Erik")["score"] == 1.5
    snap.close()
    # the source database was not modified by the backup
    assert db_path.read_bytes() == before


def test_backup_succeeds_with_corrupt_bundle(tmp_path):
    data_dir = tmp_path / "data"
    db_path = _seed_db(data_dir)
    (data_dir / "bundle").mkdir()
    (data_dir / "bundle" / "bundle.json").write_text("{ this is not json", encoding="utf-8")
    out = make_backup(db_path, None, data_dir, bundle_dir=data_dir / "bundle", now="2026-08-23 00:11:00")
    assert Path(out["backup_dir"], "labels.db").exists()
    assert out["export"]["status"] == "skipped" and "unreadable" in out["export"]["reason"]


def test_backup_cli_does_not_need_the_bundle(tmp_path, capsys):
    data_dir = tmp_path / "data"
    _seed_db(data_dir)
    rc = main(["backup", "--data-dir", str(data_dir)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert Path(out["backup_dir"], "labels.db").exists() and out["export"]["status"] == "skipped"
    # copy-to works without the bundle too
    rc = main(["backup", "--data-dir", str(data_dir), "--copy-to", str(tmp_path / "onedrive")])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert Path(out["copied_to"], "labels.db").exists()
    # no database -> clear message, non-zero exit, nothing created
    rc = main(["backup", "--data-dir", str(tmp_path / "empty")])
    assert rc == 2 and not (tmp_path / "empty" / "backups").exists()


def test_snapshot_is_taken_read_only_while_another_connection_writes(tmp_path):
    data_dir = tmp_path / "data"
    db_path = _seed_db(data_dir)
    live = LabelDB(db_path)                                   # another process/thread holding the live DB (WAL)
    live.save_label("gaaaaaaaaaa", "Dana", score=2.0, rubric=[], status="saved", expected_revision=0)
    snap = snapshot_sqlite(db_path, tmp_path / "snap.db")
    assert snap["counts"]["labels"] == 3 and snap["sha256"] == hashlib.sha256((tmp_path / "snap.db").read_bytes()).hexdigest()
    # the live DB keeps working and the snapshot is independent
    live.save_label("gbbbbbbbbbb", "Dana", score=4.0, rubric=[], status="saved", expected_revision=0)
    c = sqlite3.connect(str(tmp_path / "snap.db"))
    assert c.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 3
    c.close()
    live.close()
    with pytest.raises(FileNotFoundError):
        snapshot_sqlite(tmp_path / "nope.db", tmp_path / "x.db")
