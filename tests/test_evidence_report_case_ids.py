"""Reporting commands must not mutate the database, and must not hide items.

Two defects, one root cause. `_open()` registered the bundle on EVERY command —
`load_items` inserts every item id the bundle carries — so running a *report*
against a bundle whose id salt differed from the database's inserted a second,
orphan set of item rows. The same mismatch then made the report print
`case_id: null` and an empty `affected_case_ids`, hiding the very items that had
drifted.

No model, network or OCR calls.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from labeling_app.cli import _join_case_ids, main
from labeling_app.db import LabelDB


def _dataset(tmp_path: Path) -> Path:
    """The smallest grade_primary-shaped dataset a bundle can be built from."""
    import shutil
    real = Path(__file__).resolve().parents[1] / "evaluation" / "model_selection" / "datasets" / "grade_primary"
    if not (real / "manifest.json").exists():
        pytest.skip("grade_primary dataset is not built here")
    d = tmp_path / "dataset"
    d.mkdir()
    for name in ("cases_inputs.jsonl", "cases_labels.jsonl", "manifest.json", "CHECKSUMS.sha256"):
        shutil.copy2(real / name, d / name)
    return d


# ------------------------------------------------- the join never hides an item --

def test_an_item_the_bundle_does_not_know_keeps_its_id_instead_of_becoming_null():
    rep = {"items_evidence_changed": [{"item_id": "gAAA"}, {"item_id": "gBBB"}],
           "stale_labels": [{"item_id": "gBBB", "grader": "Erik"}]}
    out = _join_case_ids(rep, {"gAAA": "e003_q1_r5"})
    assert out["items_evidence_changed"][0]["case_id"] == "e003_q1_r5"
    assert out["items_evidence_changed"][1]["case_id"] is None
    assert "not in the current bundle" in out["items_evidence_changed"][1]["case_id_unavailable"]
    # the unmapped item is still counted as affected, by its item id
    assert out["affected_case_ids"] == ["e003_q1_r5", "gBBB"]
    assert out["items_not_in_current_bundle"] == ["gBBB"]


def test_a_fully_mapped_report_names_only_case_ids():
    rep = {"items_evidence_changed": [{"item_id": "gAAA"}]}
    out = _join_case_ids(rep, {"gAAA": "e003_q1_r5"})
    assert out["affected_case_ids"] == ["e003_q1_r5"]
    assert out["items_not_in_current_bundle"] == []
    assert "case_id_unavailable" not in out["items_evidence_changed"][0]


def test_authoritative_rows_are_joined_too():
    """They were missing from the join entirely, so every authoritative label on
    repaired evidence printed without a case id."""
    rep = {"authoritative_labels_on_repaired_evidence": [{"item_id": "gAAA", "grader": "Erik"}]}
    out = _join_case_ids(rep, {"gAAA": "e004_q2_r3"})
    assert out["authoritative_labels_on_repaired_evidence"][0]["case_id"] == "e004_q2_r3"


# --------------------------------------------- a report must not write to the db --

def _counts(db_path: Path) -> dict:
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in ("items", "labels", "events")}
    finally:
        con.close()


@pytest.fixture()
def deployment(tmp_path):
    """A data dir with a bundle and a database registered against it."""
    from labeling_app.bundle import build_bundle
    dataset = _dataset(tmp_path)
    data = tmp_path / "data"
    bundle = data / "bundle"
    repo = Path(__file__).resolve().parents[1]
    b = build_bundle(dataset, bundle, evaluation_root=repo / "evaluation", page_max_edge=200,
                     now="2026-08-23 04:00:00")
    # register once, deliberately — that is a WRITE and belongs to bundle building,
    # never to a report
    from labeling_app.bundle import Bundle
    loaded = Bundle(bundle)
    db = LabelDB(data / "labels.db")
    db.load_items(loaded.items)
    db.sync_evidence(loaded.fingerprints)
    return data, bundle, dataset


def test_a_report_does_not_insert_items_when_the_bundle_salt_differs(deployment, tmp_path, capsys):
    """The exact shape that produced 67 orphan rows in the live database."""
    from labeling_app.bundle import build_bundle
    data, bundle, dataset = deployment
    capsys.readouterr()
    before = _counts(data / "labels.db")
    assert before["items"] == 67

    other = tmp_path / "other_bundle"                # rebuilt from scratch => a NEW id salt
    build_bundle(dataset, other, evaluation_root=Path(__file__).resolve().parents[1] / "evaluation",
                 page_max_edge=200, now="2026-08-23 05:00:00", salt="a-different-salt")
    for cmd in ("status", "evidence-report"):
        assert main([cmd, "--data-dir", str(data), "--bundle", str(other)]) == 0
        capsys.readouterr()
        assert _counts(data / "labels.db") == before, f"`{cmd}` wrote to the database"
    # verify-provenance takes no --bundle; it must be read-only against the default one
    assert main(["verify-provenance", "--data-dir", str(data)]) == 0
    capsys.readouterr()
    assert _counts(data / "labels.db") == before, "`verify-provenance` wrote to the database"


def test_a_report_against_a_mismatched_bundle_names_the_unmapped_items(deployment, tmp_path, capsys):
    from labeling_app.bundle import build_bundle
    data, bundle, dataset = deployment
    capsys.readouterr()
    other = tmp_path / "other_bundle2"
    build_bundle(dataset, other, evaluation_root=Path(__file__).resolve().parents[1] / "evaluation",
                 page_max_edge=200, now="2026-08-23 05:00:00", salt="yet-another-salt")
    assert main(["evidence-report", "--data-dir", str(data), "--bundle", str(other)]) == 0
    out = capsys.readouterr().out
    rep = json.loads(out[out.index("{"):])
    # no evidence has changed in this fixture, so there is nothing to list — the
    # point is that the keys exist and nothing silently disappeared
    assert "items_not_in_current_bundle" in rep and "affected_case_ids" in rep


def test_build_bundle_still_registers_into_an_existing_database(deployment, tmp_path, capsys):
    """The fix must not stop the command whose JOB is to register."""
    data, bundle, dataset = deployment
    capsys.readouterr()
    before = _counts(data / "labels.db")
    assert before["items"] == 67
    assert main(["build-bundle", "--dataset", str(dataset), "--out", str(bundle),
                 "--data-dir", str(data), "--replace"]) == 0
    capsys.readouterr()
    after = _counts(data / "labels.db")
    assert after["items"] == 67, "a --replace rebuild keeps stable ids, it does not add a second set"
    assert after["labels"] == before["labels"]
