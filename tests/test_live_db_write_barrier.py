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
            names_live = re.search(r"(LIVE_DB|live_db_path\(\)|LOCALAPPDATA|AppData)", line)
            if not names_live or "LIVE_DB = " in line:
                continue
            # snapshot_sqlite(LIVE, dest) is the SANCTIONED read-only path: the deployment
            # is the SOURCE. The dangerous shape is the deployment in the DESTINATION
            # position (swapped arguments), which writes onto it — allow only the former.
            if re.search(r"snapshot_sqlite\s*\(\s*(LIVE_DB|live_db_path\(\))\s*,", line):
                continue
            # a deployment path handed to any other writer, in any argument position
            if re.search(r"(LabelDB|load_items|save_label|sync_evidence|sync_eligibility|set_final|"
                         r"snapshot_to|snapshot_sqlite|make_backup)\s*\(", line):
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


# ------------------------------------------- barrier 4: the DESTINATION too --

def test_a_snapshot_can_never_be_written_onto_the_live_database(tmp_path):
    """The guard's own advice is `snapshot_sqlite(live, copy)`. Swap the two
    arguments by accident and the deployment becomes the DESTINATION of a
    backup — a write, straight onto the file the guard exists to protect.
    The destination is checked exactly like the source."""
    from labeling_app.backup import BackupError, snapshot_sqlite
    live = live_db_path()
    source = tmp_path / "source.db"
    LabelDB(source).load_items([{"item_id": "g1", "max_score": 4.0, "rubric_items": [{"id": "R1", "text": "r"}]}])
    before = (live.stat().st_size, live.stat().st_mtime_ns) if live.exists() else None

    with pytest.raises(LabelError, match="refusing to open the LIVE labeling database"):
        snapshot_sqlite(source, live)                       # arguments swapped

    if before is not None:
        assert (live.stat().st_size, live.stat().st_mtime_ns) == before, "the deployment was written to"


def test_labeldb_snapshot_to_refuses_the_live_database_as_a_destination(tmp_path):
    source = tmp_path / "source.db"
    db = LabelDB(source)
    db.load_items([{"item_id": "g1", "max_score": 4.0, "rubric_items": [{"id": "R1", "text": "r"}]}])
    with pytest.raises(LabelError, match="refusing to open the LIVE labeling database"):
        db.snapshot_to(live_db_path())
    db.snapshot_to(tmp_path / "ok.db")                      # an ordinary destination still works
    assert (tmp_path / "ok.db").exists()


def test_a_backup_onto_its_own_source_is_refused(tmp_path):
    from labeling_app.backup import BackupError, snapshot_sqlite
    src = tmp_path / "labels.db"
    LabelDB(src).load_items([{"item_id": "g1", "max_score": 4.0, "rubric_items": [{"id": "R1", "text": "r"}]}])
    with pytest.raises(BackupError, match="onto itself"):
        snapshot_sqlite(src, src)


def test_an_empty_or_tableless_snapshot_is_never_called_a_backup(tmp_path):
    """An empty file passes every integrity check trivially. Reporting that as a
    verified backup is precisely the failure this machinery exists to prevent."""
    from labeling_app.backup import BackupError, snapshot_sqlite
    empty = tmp_path / "empty.db"
    sqlite3.connect(str(empty)).close()                     # a valid, completely empty database
    with pytest.raises(BackupError, match="preserves nothing"):
        snapshot_sqlite(empty, tmp_path / "out.db")
    assert not (tmp_path / "out.db").exists()

    partial = tmp_path / "partial.db"
    con = sqlite3.connect(str(partial))
    con.execute("CREATE TABLE items (item_id TEXT PRIMARY KEY)")
    con.commit(); con.close()
    with pytest.raises(BackupError, match="no labels table"):
        snapshot_sqlite(partial, tmp_path / "out2.db")
    assert not (tmp_path / "out2.db").exists()


# --------------------------------- barrier 5: spellings and the opt-in switch --

def test_the_opt_in_is_not_flipped_by_a_falsy_value(monkeypatch):
    """`LABELING_ALLOW_LIVE_DB=0` read as truthy would silently disarm the guard."""
    for falsy in ("0", "false", "no", "off", ""):
        monkeypatch.setenv(LIVE_DB_OPT_IN, falsy)
        with pytest.raises(LabelError):
            LabelDB(live_db_path())
    for truthy in ("1", "true", "YES", "on"):
        monkeypatch.setenv(LIVE_DB_OPT_IN, truthy)
        assert assert_not_live_database(live_db_path()) is None


def test_extended_length_and_unc_spellings_are_caught(tmp_path):
    r"""Extended-length and UNC spellings reach the same bytes; string
    comparison alone misses both, so the filesystem is asked."""
    live = live_db_path()
    if not live.exists():
        pytest.skip("no live labeling database on this machine")
    spellings = [rf"\?\{live}", str(live).replace("C:\\", r"\localhost\C$" + "\\", 1)]
    for spelling in spellings:
        try:
            reachable = Path(spelling).exists()
        except OSError:
            reachable = False
        if not reachable:
            continue                                   # that syntax is not available here
        with pytest.raises(LabelError, match="refusing to open the LIVE"):
            LabelDB(spelling)


def test_export_does_not_register_the_bundle(tmp_path, capsys):
    """`export` is step two of the label-import sequence. Registering there once
    inserted 67 orphan items AND retired all 67 real ones."""
    from tests.test_evidence_report_case_ids import _counts, _dataset
    from labeling_app.bundle import Bundle, build_bundle
    from labeling_app.cli import main
    dataset = _dataset(tmp_path)
    data = tmp_path / "data"
    repo = Path(__file__).resolve().parents[1]
    build_bundle(dataset, data / "bundle", evaluation_root=repo / "evaluation", page_max_edge=200,
                 now="2026-08-23 06:00:00")
    loaded = Bundle(data / "bundle")
    db = LabelDB(data / "labels.db")
    db.load_items(loaded.items)
    db.sync_evidence(loaded.fingerprints)
    capsys.readouterr()
    before = _counts(data / "labels.db")

    other = tmp_path / "other_bundle"
    build_bundle(dataset, other, evaluation_root=repo / "evaluation", page_max_edge=200,
                 now="2026-08-23 06:01:00", salt="a-different-salt")
    assert main(["export", "--data-dir", str(data), "--bundle", str(other),
                 "--out", str(tmp_path / "final_labels.json")]) == 0
    capsys.readouterr()
    assert _counts(data / "labels.db") == before, "`export` registered the bundle"
    import sqlite3
    con = sqlite3.connect(f"file:{(data / 'labels.db').as_posix()}?mode=ro", uri=True)
    try:
        assert dict(con.execute("SELECT eligible, COUNT(*) FROM items GROUP BY eligible")) == {1: 67}, \
            "`export` retired the real items"
    finally:
        con.close()
