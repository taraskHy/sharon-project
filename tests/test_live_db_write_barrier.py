"""A test must never open the LIVE labeling database.

This file exists because of a real incident: a test constructed
``LabelDB`` on ``%LOCALAPPDATA%\\autograder\\labeling\\labels.db`` while the
labeling server held it open. ``LabelDB.__init__`` is a WRITER — it sets
``journal_mode``, runs DDL and migrations, and closing its connection
checkpoints the WAL — and the deployment's ``items`` table was physically
corrupted (``PRAGMA integrity_check`` became unrunnable, 67 phantom keys
appeared in the index, 4 real rows became unreadable). The human ground truth
survived only because every other table was intact and hourly backups existed.

Two independent barriers, tested here:

* the LIBRARY refuses: ``assert_not_live_database`` fires inside
  ``LabelDB.__init__`` under pytest, BEFORE any SQLite connection is opened;
* the SESSION redirects: an autouse fixture in conftest points
  ``LABELING_DATA_DIR`` at a throwaway directory, so ``default_data_dir()``
  cannot return the real one in the first place.

Production behaviour is unchanged: the guard only fires while pytest runs.
"""
from __future__ import annotations

import os
import re
import sqlite3
from pathlib import Path

import pytest

from labeling_app.cli import default_data_dir
from labeling_app.db import LIVE_DB_OPT_IN, LabelDB, LabelError, assert_not_live_database, live_db_path

TESTS_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------- barrier 1: library --

def test_constructing_labeldb_on_the_live_database_fails_before_any_sqlite_write(tmp_path):
    live = live_db_path()
    before = (live.stat().st_size, live.stat().st_mtime_ns) if live.exists() else None
    sidecars_before = {p.name for p in live.parent.glob("labels.db-*")} if live.parent.exists() else set()

    with pytest.raises(LabelError, match="refusing to open the LIVE labeling database"):
        LabelDB(live)

    if before is not None:                          # the file was not touched at all
        assert (live.stat().st_size, live.stat().st_mtime_ns) == before
        assert {p.name for p in live.parent.glob("labels.db-*")} == sidecars_before


def test_the_guard_names_the_safe_alternatives():
    with pytest.raises(LabelError) as e:
        LabelDB(live_db_path())
    msg = str(e.value)
    assert "snapshot_sqlite" in msg and "tmp_path" in msg and LIVE_DB_OPT_IN in msg


def test_the_guard_accepts_any_spelling_of_the_live_path(tmp_path):
    live = live_db_path()
    if not live.exists():
        pytest.skip("no live labeling database on this machine")
    for spelling in (str(live), str(live).replace("\\", "/"), live.parent / "." / "labels.db"):
        with pytest.raises(LabelError):
            LabelDB(spelling)


def test_an_ordinary_temporary_database_is_unaffected(tmp_path):
    db = LabelDB(tmp_path / "labels.db")
    db.load_items([{"item_id": "g1", "max_score": 4.0, "rubric_items": [{"id": "R1", "text": "r"}]}])
    assert db.summary()["total_items"] == 1


def test_the_opt_in_exists_for_deliberate_forensics(tmp_path, monkeypatch):
    """A forensic reader may still ask for the real file explicitly."""
    monkeypatch.setenv(LIVE_DB_OPT_IN, "1")
    assert assert_not_live_database(live_db_path()) is None


def test_the_guard_is_inert_outside_pytest(monkeypatch):
    """Production must not be affected: with no pytest marker in the
    environment the guard returns without objecting."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("PYTEST_VERSION", raising=False)
    assert assert_not_live_database(live_db_path()) is None


# ------------------------------------------------------- barrier 2: session --

def test_the_default_data_dir_is_redirected_for_the_whole_session():
    """`default_data_dir()` — what every CLI entry point uses when no
    --data-dir is given — must not resolve to the real deployment."""
    resolved = default_data_dir().resolve()
    assert os.environ.get("LABELING_DATA_DIR"), "the autouse sandbox fixture did not run"
    assert resolved != live_db_path().parent
    assert (resolved / "labels.db") != live_db_path()


# ------------------------------------------------- barrier 3: the whole suite --

def test_no_test_module_can_reach_the_live_labeling_database():
    """Static sweep: nothing in tests/ may hand a real-deployment path to a
    writer. Read-only forensic access must go through snapshot_sqlite."""
    offenders = []
    for path in sorted(TESTS_DIR.glob("test_*.py")) + [TESTS_DIR / "conftest.py"]:
        if path.name == Path(__file__).name or not path.exists():
            continue
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if "re.search" in line or line.lstrip().startswith("#"):
                continue                            # a scanner's own pattern is not a call site
            if re.search(r"LabelDB\s*\(\s*(LIVE_DB|live_db_path\(\))", line):
                offenders.append(f"{path.name}:{n}: {line.strip()}")
            if re.search(r"(LOCALAPPDATA|AppData)", line) and "snapshot" not in line and "LIVE_DB =" not in line:
                if re.search(r"LabelDB|load_items|save_label|sync_evidence|set_final", line):
                    offenders.append(f"{path.name}:{n}: {line.strip()}")
    assert offenders == [], (
        "these lines could open the live labeling database with a writer; snapshot it first with "
        "labeling_app.backup.snapshot_sqlite and open the copy:\n  " + "\n  ".join(offenders))


@pytest.mark.skipif(not os.environ.get(LIVE_DB_OPT_IN),
                    reason=f"opt-in only: set {LIVE_DB_OPT_IN}=1 to let the suite read the live database. "
                           "By default the test run does not open the deployment's file at all — not even "
                           "read-only, which still touches its -shm sidecar.")
def test_snapshotting_the_live_database_is_read_only_and_allowed(tmp_path):
    """The sanctioned way to inspect the deployment: an online-backup snapshot
    over a read-only connection, then open the COPY."""
    from labeling_app.backup import snapshot_sqlite
    live = live_db_path()
    if not live.exists():
        pytest.skip("no live labeling database on this machine")
    before = (live.stat().st_size, live.stat().st_mtime_ns)
    snap = tmp_path / "labels.db"
    try:
        snapshot_sqlite(live, snap)
    except sqlite3.DatabaseError:
        pytest.skip("the live labeling database is not currently readable")
    assert (live.stat().st_size, live.stat().st_mtime_ns) == before, "the source was not written"
    assert snap.exists() and not list(tmp_path.glob("labels.db-*")), "no WAL sidecars beside a snapshot"
    LabelDB(snap)                                   # opening the COPY is fine
