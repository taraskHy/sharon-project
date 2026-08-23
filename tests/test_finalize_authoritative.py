"""Bulk promotion of instructor-copied grades to FINAL.

An `original_instructor_grade` was COPIED from the original graded exam. It is
not a judgement formed from the app's evidence, so neither ordinary route to
FINAL fits: `finalize_agreement` wants two independent graders who will never
exist here, and calling it "adjudicated" would claim a judgement nobody made.
This path promotes it verbatim, under its own source.

The invariant that matters most: the score is never recomputed.

No model, network or OCR calls.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from labeling_app.db import FINAL_SOURCES, LabelDB

REPO = Path(__file__).resolve().parents[1]


def _db(tmp_path, n=3, max_score=4.0):
    db = LabelDB(tmp_path / "labels.db")
    db.load_items([{"item_id": f"g{i}", "max_score": max_score,
                    "rubric_items": [{"id": "R1", "text": "r"}]} for i in range(n)])
    return db


def _authoritative(db, item, score, grader="Erik", rubric=None, status="saved"):
    db.save_label(item, grader, score=score, rubric=rubric or [], status=status)
    db.set_label_provenance(grader=grader, label_source="original_instructor_grade",
                            entered_by=grader, asserted_by="owner")


def _label_fingerprint(db) -> str:
    con = sqlite3.connect(f"file:{db.path.as_posix()}?mode=ro", uri=True)
    try:
        rows = con.execute("SELECT item_id, grader, score, rubric, status, revision, label_source, "
                           "entered_by, provenance_asserted_by FROM labels ORDER BY item_id, grader").fetchall()
    finally:
        con.close()
    return hashlib.sha256(json.dumps([list(r) for r in rows], sort_keys=True).encode()).hexdigest()


def test_a_single_instructor_grade_is_promoted_verbatim(tmp_path):
    db = _db(tmp_path, n=3)
    for i, score in enumerate((4.0, 0.0, 2.0)):
        _authoritative(db, f"g{i}", score, rubric=["R1"] if i == 0 else [])

    dry = db.finalize_authoritative()
    assert dry["applied"] is False and dry["eligible_for_promotion"] == 3
    assert db.summary()["final"] == 0, "a dry run writes nothing"

    out = db.finalize_authoritative(apply=True)
    assert out["applied"] is True and out["promoted"] == 3
    finals = {f["item_id"]: f for f in db.final_rows()}
    assert len(finals) == 3
    assert [finals[f"g{i}"]["score"] for i in range(3)] == [4.0, 0.0, 2.0], "copied, never recomputed"
    assert sorted(finals["g0"]["rubric"]) == ["R1"], "rubric decisions carried over"
    for f in finals.values():
        assert f["source"] == "authoritative" and f["source"] in FINAL_SOURCES
        assert f["ground_truth_source"] == "original_instructor_grade"
        assert list(f["contributing_graders"]) == ["Erik"]


def test_it_is_idempotent(tmp_path):
    db = _db(tmp_path, n=2)
    for i in range(2):
        _authoritative(db, f"g{i}", 3.0)
    db.finalize_authoritative(apply=True)
    before = {f["item_id"]: dict(f) for f in db.final_rows()}
    out = db.finalize_authoritative(apply=True)
    assert out["promoted"] == 0 and out["already_final"] == 2 and out["eligible_for_promotion"] == 0
    assert {f["item_id"]: dict(f) for f in db.final_rows()} == before, "a second run changes nothing"


def test_an_independent_grading_is_never_promoted(tmp_path):
    db = _db(tmp_path, n=1)
    db.save_label("g0", "Ann", score=3.0, rubric=[], status="saved")      # default source
    out = db.finalize_authoritative(apply=True)
    assert out["promoted"] == 0
    assert any("no authoritative label" in s["reason"] for s in out["skipped_detail"])
    assert db.summary()["final"] == 0


def test_two_labels_on_an_item_are_left_to_the_ordinary_routes(tmp_path):
    db = _db(tmp_path, n=1)
    _authoritative(db, "g0", 4.0)
    db.save_label("g0", "Ann", score=1.0, rubric=[], status="saved")
    out = db.finalize_authoritative(apply=True)
    assert out["promoted"] == 0
    assert any("exactly one" in s["reason"] for s in out["skipped_detail"])


def test_incomplete_provenance_is_refused(tmp_path):
    db = _db(tmp_path, n=1)
    db.save_label("g0", "Erik", score=4.0, rubric=[], status="saved")
    db.set_label_provenance(grader="Erik", label_source="original_instructor_grade", entered_by="Erik")
    out = db.finalize_authoritative(apply=True)                            # no asserted_by
    assert out["promoted"] == 0
    assert any("provenance" in s["reason"] for s in out["skipped_detail"])


def test_an_ineligible_item_is_refused(tmp_path):
    db = _db(tmp_path, n=2)
    for i in range(2):
        _authoritative(db, f"g{i}", 4.0)
    db.sync_eligibility(["g0", "g1"], ["g1"], eligibility_known=True)
    out = db.finalize_authoritative(apply=True)
    assert out["promoted"] == 1
    assert any(s["item_id"] == "g1" and "not eligible" in s["reason"] for s in out["skipped_detail"])


def test_a_conflicting_existing_final_is_never_overwritten(tmp_path):
    db = _db(tmp_path, n=1)
    _authoritative(db, "g0", 4.0)
    ov = db.overview("g0")
    db.set_final("g0", score=1.0, rubric=[], note="", source="adjudicated",
                 adjudicator="admin", expected_item_revision=ov["revision"])
    out = db.finalize_authoritative(apply=True)
    assert out["promoted"] == 0
    assert any("different FINAL already exists" in s["reason"] for s in out["skipped_detail"])
    assert db.final_rows()[0]["score"] == 1.0, "the existing FINAL stands"


def test_a_label_that_is_not_a_saved_score_is_refused(tmp_path):
    """A skipped/flagged label, or one with no score, is never ground truth."""
    db = _db(tmp_path, n=1)
    db.save_label("g0", "Erik", score=None, rubric=[], status="skipped")
    db.set_label_provenance(grader="Erik", label_source="original_instructor_grade",
                            entered_by="Erik", asserted_by="owner")
    out = db.finalize_authoritative(apply=True)
    assert out["promoted"] == 0 and db.summary()["final"] == 0
    assert out["skipped_detail"], "the refusal is reported, not silent"


def test_the_grader_labels_are_untouched_by_promotion(tmp_path):
    db = _db(tmp_path, n=3)
    for i, score in enumerate((4.0, 0.0, 2.5)):
        _authoritative(db, f"g{i}", score)
    before = _label_fingerprint(db)
    db.finalize_authoritative(apply=True)
    assert _label_fingerprint(db) == before, "promotion must not touch a single grader label"


def test_the_promotion_is_recorded_in_the_audit_trail(tmp_path):
    db = _db(tmp_path, n=1)
    _authoritative(db, "g0", 4.0)
    db.finalize_authoritative(apply=True, adjudicator="owner")
    con = sqlite3.connect(f"file:{db.path.as_posix()}?mode=ro", uri=True)
    try:
        rows = [dict(zip(("grader", "detail"), r)) for r in
                con.execute("SELECT grader, detail FROM events WHERE action='finalize_authoritative'")]
    finally:
        con.close()
    assert len(rows) == 1 and rows[0]["grader"] == "owner"
    assert json.loads(rows[0]["detail"])["promoted"] == 1


def test_a_promoted_final_survives_export_and_import(tmp_path):
    """End to end: promotion -> export -> benchmark import, scores identical."""
    from autograder.benchmark.finallabels import import_final_labels
    from labeling_app.bundle import Bundle, build_bundle
    from labeling_app.export import write_export
    from tests.test_evidence_report_case_ids import _dataset

    dataset = _dataset(tmp_path)
    data = tmp_path / "data"
    build_bundle(dataset, data / "bundle", evaluation_root=REPO / "evaluation",
                 page_max_edge=200, now="2026-08-23 07:00:00")
    b = Bundle(data / "bundle")
    db = LabelDB(data / "labels.db")
    db.load_items(b.items)
    db.sync_evidence(b.fingerprints)
    scores = {}
    for n, oid in enumerate(sorted(b.id_map)):
        s = float((n % 9) * 0.5)
        _authoritative(db, oid, s)
        scores[b.id_map[oid]] = s

    out = db.finalize_authoritative(apply=True)
    assert out["promoted"] == 67

    exp = tmp_path / "final_labels.json"
    data_exp = write_export(db, b, exp)
    assert data_exp["final_count"] == 67

    res = import_final_labels(exp, dataset)
    assert res["imported"] == 67, res
    assert not res["unknown_case_ids"] and not res["ignored_unknown_source"]
    imported = json.loads((dataset / "final_labels.json").read_text(encoding="utf-8"))
    assert len(imported["labels"]) == 67
    for cid, row in imported["labels"].items():
        assert row["score"] == scores[cid], f"{cid}: imported score differs from the grader's"
        assert row["source"] == "authoritative"
        assert row["ground_truth_source"] == "original_instructor_grade"


def test_an_unrecognised_final_source_is_recorded_not_silently_dropped(tmp_path):
    """A whole export once imported zero labels and still looked successful."""
    from autograder.benchmark.finallabels import import_final_labels
    from tests.test_evidence_report_case_ids import _dataset
    dataset = _dataset(tmp_path)
    case = json.loads((dataset / "cases_labels.jsonl").read_text(encoding="utf-8").splitlines()[0])["case_id"]
    exp = tmp_path / "bad_export.json"
    exp.write_text(json.dumps({
        "schema_version": 3, "kind": "grade_primary_final_labels", "final_count": 1,
        "items": [{"item_id": case, "final_score": 4.0, "source": "invented_source",
                   "rubric_decisions": [], "contributing_graders": ["Erik"],
                   "eligible_for_human_label": True, "evidence_stale": False}]},
        ensure_ascii=False), encoding="utf-8")
    res = import_final_labels(exp, dataset)
    assert res["imported"] == 0
    assert case in res["ignored_unknown_source"]
    assert res["ignored_unknown_source"][case]["source"] == "invented_source"


def test_the_cli_dry_run_writes_nothing_then_apply_promotes(tmp_path, capsys):
    from labeling_app.bundle import Bundle, build_bundle
    from labeling_app.cli import main
    from tests.test_evidence_report_case_ids import _dataset

    dataset = _dataset(tmp_path)
    data = tmp_path / "data"
    build_bundle(dataset, data / "bundle", evaluation_root=REPO / "evaluation",
                 page_max_edge=200, now="2026-08-23 07:10:00")
    b = Bundle(data / "bundle")
    db = LabelDB(data / "labels.db")
    db.load_items(b.items)
    db.sync_evidence(b.fingerprints)
    for oid in sorted(b.id_map)[:3]:
        _authoritative(db, oid, 4.0)
    capsys.readouterr()

    assert main(["finalize-authoritative", "--data-dir", str(data), "--bundle", str(data / "bundle")]) == 0
    capsys.readouterr()
    assert db.summary()["final"] == 0, "the dry run wrote a FINAL"

    assert main(["finalize-authoritative", "--data-dir", str(data), "--bundle", str(data / "bundle"),
                 "--apply"]) == 0
    capsys.readouterr()
    assert db.summary()["final"] == 3
