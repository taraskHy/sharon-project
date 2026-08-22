"""Provenance VERIFICATION and the three independent dimensions.

Complements tests/test_label_provenance_source.py (which covers the schema-3
semantics themselves) with the pieces the live deployment needs:

* unlike label sources are never compared as inter-grader votes (spec F);
* a read-only audit that re-derives, from the database's own event trail, that
  recording provenance changed no score — and that names any label where it
  did (spec H, the "verify the backfill" requirement);
* ground truth / model evidence / transcription completeness stay three
  separate dimensions in the benchmark's dataset status (spec §12/J).

No model, network or OCR calls.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from labeling_app.cli import main
from labeling_app.db import (DEFAULT_LABEL_SOURCE, STATE_AUTHORITATIVE, STATE_EVIDENCE_REVIEW, LabelDB)

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "evaluation" / "model_selection" / "datasets" / "grade_primary"
real_dataset = pytest.mark.skipif(not (DATASET / "manifest.json").exists(),
                                  reason="grade_primary dataset not built")


def _db(tmp_path, n: int = 3) -> LabelDB:
    db = LabelDB(tmp_path / "labels.db")
    db.load_items([{"item_id": f"g{i}", "max_score": 4.0, "rubric_items": [{"id": "R1", "text": "r"}]}
                   for i in range(n)])
    return db


# ------------------------------------------------- F: unlike sources never compared --

def test_agreement_never_compares_an_instructor_grade_with_an_independent_grading(tmp_path):
    """A copied instructor grade and an independent re-grade answer different
    questions. Even when their scores differ, that is NOT an inter-grader
    disagreement and must never route the item to adjudication."""
    db = _db(tmp_path, n=1)
    db.save_label("g0", "Erik", score=4.0, rubric=[], status="saved")
    db.set_label_provenance(grader="Erik", label_source="original_instructor_grade",
                            entered_by="Erik", asserted_by="owner")
    db.save_label("g0", "Ann", score=1.0, rubric=[], status="saved")      # very different number

    ov = db.overview("g0")
    assert ov["n_authoritative"] == 1 and ov["n_independent"] == 1
    assert ov["agreement"] is None                    # one independent judgment => nothing to compare
    assert ov["state"] == STATE_AUTHORITATIVE         # not NEEDS_ADJUDICATION
    s = db.summary()
    assert s["disagreements"] == 0 and s["agreements"] == 0
    assert s["needs_adjudication"] == 0 and s["authoritative_ground_truth"] == 1
    # ground truth stays the authoritative one, whatever the independent grader said
    assert ov["ground_truth_source"] == "original_instructor_grade"


def test_two_independent_labels_are_still_compared_normally(tmp_path):
    """The exclusion is about SOURCE, not about disabling agreement."""
    db = _db(tmp_path, n=2)
    db.save_label("g0", "Ann", score=3.0, rubric=[], status="saved")
    db.save_label("g0", "Bob", score=1.0, rubric=[], status="saved")
    assert db.overview("g0")["agreement"] is False and db.overview("g0")["state"] == "NEEDS_ADJUDICATION"
    db.save_label("g1", "Ann", score=2.0, rubric=["R1"], status="saved")
    db.save_label("g1", "Bob", score=2.0, rubric=["R1"], status="saved")
    assert db.overview("g1")["agreement"] is True and db.overview("g1")["state"] == "AGREEMENT"
    s = db.summary()
    assert s["disagreements"] == 1 and s["agreements"] == 1


def test_two_instructor_copies_are_not_an_agreement_statistic(tmp_path):
    """Two people copying the same authoritative grade is a transcription check,
    not independent agreement — it never inflates the agreement counters."""
    db = _db(tmp_path, n=1)
    db.save_label("g0", "Erik", score=4.0, rubric=[], status="saved")
    db.save_label("g0", "Dana", score=4.0, rubric=[], status="saved")
    for g in ("Erik", "Dana"):
        db.set_label_provenance(grader=g, label_source="original_instructor_grade", asserted_by="owner")
    ov = db.overview("g0")
    assert ov["n_authoritative"] == 2 and ov["agreement"] is None
    assert db.summary()["agreements"] == 0 and ov["state"] == STATE_AUTHORITATIVE


# ------------------------------------------ H: verifying the backfill from the audit trail --

def test_verify_provenance_reports_sources_actors_and_proves_scores_unchanged(tmp_path):
    db = _db(tmp_path, n=3)
    for i, score in enumerate((4.0, 2.5, 0.0)):
        db.save_label(f"g{i}", "Erik", score=score, rubric=[], status="saved")
    db.set_label_provenance(grader="Erik", label_source="original_instructor_grade",
                            entered_by="Erik", asserted_by="owner",
                            source_refs={"g0": "test/003_70.pdf#e003_q1_r5"})
    rep = db.verify_provenance()
    assert rep["labels_total"] == 3
    assert rep["labels_by_source"] == {"human_independent_grading": 0, "original_instructor_grade": 3,
                                       "adjudicated": 0}
    assert rep["per_grader"]["Erik"]["by_source"] == {"original_instructor_grade": 3}
    assert rep["per_grader"]["Erik"]["entered_by"] == ["Erik"]
    assert rep["per_grader"]["Erik"]["asserted_by"] == ["owner"]
    assert rep["per_grader"]["Erik"]["revisions"] == [1] and rep["per_grader"]["Erik"]["statuses"] == ["saved"]
    assert rep["provenance_events_checked"] == 3 and rep["scores_unchanged"] is True
    assert rep["scores_changed_since_provenance_recorded"] == []
    assert rep["authoritative_missing_entered_by"] == [] and rep["authoritative_missing_asserted_by"] == []
    assert rep["backfill_events"][0]["labels"] == 3 and rep["backfill_events"][0]["scores_modified"] == 0
    assert "never claims" in rep["assertion_note"]


def test_verify_provenance_names_a_label_whose_score_moved_after_provenance_was_recorded(tmp_path):
    """The audit is real: tamper with a score behind the app's back and the
    verification must point at it (it does not silently pass)."""
    db = _db(tmp_path, n=2)
    db.save_label("g0", "Erik", score=4.0, rubric=[], status="saved")
    db.save_label("g1", "Erik", score=1.0, rubric=[], status="saved")
    db.set_label_provenance(grader="Erik", label_source="original_instructor_grade", asserted_by="owner")
    assert db.verify_provenance()["scores_unchanged"] is True
    con = sqlite3.connect(db.path)                     # out-of-band edit, as a corruption would be
    con.execute("UPDATE labels SET score=0.5 WHERE item_id='g1'")
    con.commit(); con.close()
    rep = db.verify_provenance()
    assert rep["scores_unchanged"] is False
    bad = rep["scores_changed_since_provenance_recorded"]
    assert len(bad) == 1 and bad[0]["item_id"] == "g1"
    assert bad[0]["score_when_provenance_recorded"] == 1.0 and bad[0]["score_now"] == 0.5


def test_verify_provenance_flags_missing_actors_and_reports_repaired_evidence(tmp_path):
    db = _db(tmp_path, n=2)
    db.sync_evidence({"g0": "fp_old", "g1": "fp_old"})
    db.save_label("g0", "Erik", score=4.0, rubric=[], status="saved")
    db.save_label("g1", "Erik", score=3.0, rubric=[], status="saved")
    # provenance recorded WITHOUT naming who asserted it
    db.set_label_provenance(grader="Erik", label_source="original_instructor_grade", entered_by="Erik")
    db.sync_evidence({"g0": "fp_new", "g1": "fp_old"})           # g0's evidence repaired
    rep = db.verify_provenance()
    assert rep["authoritative_missing_asserted_by"] == ["g0", "g1"]
    assert rep["authoritative_missing_entered_by"] == []
    assert rep["authoritative_labels_on_repaired_evidence"] == ["g0"]
    assert rep["stale_labels"] == []                              # authoritative: repair does not invalidate


def test_verify_provenance_cli_is_read_only_and_needs_no_bundle(tmp_path, capsys):
    db = _db(tmp_path, n=2)
    db.save_label("g0", "Erik", score=4.0, rubric=[], status="saved")
    db.set_label_provenance(grader="Erik", label_source="original_instructor_grade", asserted_by="owner")
    db.close()
    before = (tmp_path / "labels.db").read_bytes()
    assert not (tmp_path / "bundle").exists()                     # no bundle at all
    rc = main(["verify-provenance", "--data-dir", str(tmp_path)])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["labels_total"] == 1 and out["scores_unchanged"] is True
    assert out["labels_by_source"]["original_instructor_grade"] == 1
    con = sqlite3.connect(tmp_path / "labels.db")                 # scores/labels untouched by the check
    assert con.execute("SELECT score FROM labels").fetchone()[0] == 4.0
    assert con.execute("SELECT COUNT(*) FROM labels").fetchone()[0] == 1
    con.close()
    assert main(["verify-provenance", "--data-dir", str(tmp_path / "nowhere")]) == 2


def test_verify_provenance_matches_the_live_deployment_shape(tmp_path):
    """A database shaped like the live one — one grader, every label copied from
    the original instructor grading, some items' evidence repaired — verifies
    clean: consistent provenance, no stale label, no score moved."""
    n = 12
    db = _db(tmp_path, n=n)
    db.sync_evidence({f"g{i}": f"fp{i}" for i in range(n)})
    for i in range(n):
        db.save_label(f"g{i}", "Erik", score=float(i % 5), rubric=[], status="saved")
    db.set_label_provenance(grader="Erik", label_source="original_instructor_grade",
                            entered_by="Erik", asserted_by="owner")
    repaired = [f"g{i}" for i in range(3)]
    db.sync_evidence({**{f"g{i}": f"fp{i}" for i in range(n)}, **{i: f"{i}_repaired" for i in repaired}})
    rep = db.verify_provenance()
    assert rep["labels_total"] == n
    assert rep["labels_by_source"]["original_instructor_grade"] == n
    assert rep["per_grader"] == {"Erik": {"labels": n, "by_source": {"original_instructor_grade": n},
                                          "entered_by": ["Erik"], "asserted_by": ["owner"],
                                          "revisions": [1], "statuses": ["saved"]}}
    assert rep["scores_unchanged"] and rep["stale_labels"] == []
    assert rep["authoritative_labels_on_repaired_evidence"] == sorted(repaired)
    s = db.summary()
    assert s["authoritative_ground_truth"] == n and s["awaiting_second_label"] == 0
    assert s["stale_labels"] == 0 and s["needs_evidence_review"] == 0
    assert db.progress("Erik")["remaining_for_me"] == 0


# ------------------- provenance rule applied CONSISTENTLY everywhere it is read --

def test_per_grader_stale_count_uses_the_same_provenance_rule(tmp_path):
    """Regression: the admin summary counted an authoritative label as stale
    (raw SQL without the label_source rule) while my_items/progress reported
    none — telling a grader they had re-review work they did not have."""
    db = _db(tmp_path, n=2)
    db.sync_evidence({"g0": "fp_old", "g1": "fp_old"})
    db.save_label("g0", "Erik", score=4.0, rubric=[], status="saved")
    db.set_label_provenance(grader="Erik", label_source="original_instructor_grade", asserted_by="owner")
    db.save_label("g1", "Ann", score=3.0, rubric=[], status="saved")
    db.sync_evidence({"g0": "fp_new", "g1": "fp_new"})            # both items repaired
    s = db.summary()
    assert s["per_grader"]["Erik"]["stale"] == 0                  # authoritative: never stale
    assert s["per_grader"]["Ann"]["stale"] == 1                   # independent: stale
    assert s["stale_labels"] == 1
    assert db.my_items("Erik")["stale"] == [] and db.progress("Erik")["my_stale"] == 0
    assert db.my_items("Ann")["stale"] == ["g1"]


def test_an_authoritative_final_is_not_invalidated_by_an_evidence_repair(tmp_path):
    """Regression: overview() judged FINAL staleness without ground_truth_source,
    so the state machine said NEEDS_REVIEW while the export said valid."""
    from labeling_app.export import export_final
    db = _db(tmp_path, n=1)
    db.sync_evidence({"g0": "fp_old"})
    db.save_label("g0", "Erik", score=4.0, rubric=[], status="saved")
    db.set_label_provenance(grader="Erik", label_source="original_instructor_grade", asserted_by="owner")
    ov = db.overview("g0")
    db.set_final("g0", score=4.0, rubric=[], note="", source="adjudicated", adjudicator="owner",
                 expected_item_revision=ov["revision"])
    assert db.final_rows()[0]["ground_truth_source"] == "original_instructor_grade"
    db.sync_evidence({"g0": "fp_new"})                            # evidence repaired AFTER the FINAL
    ov = db.overview("g0")
    assert ov["final"]["evidence_stale"] is False and ov["state"] == "FINAL"
    assert ov["evidence_repaired"] is True and ov["evidence_previous_sha256"] == "fp_old"
    assert db.summary()["stale_finals"] == 0 and db.summary()["needs_evidence_review"] == 0

    class _B:
        meta = {"items_sha256": "x", "evidence_fingerprint": "y", "eligibility": {}, "source": {}}
        id_map = {"g0": "e003_q1_r5"}
        eligibility = {"g0": {"eligible_for_human_label": True}}
    data = export_final(db, _B(), now="2026-08-23 00:00:00")
    # state machine and export now agree, and the repair is still recorded
    assert data["items"][0]["evidence_stale"] is False and data["stale_evidence_final_count"] == 0
    assert data["items"][0]["evidence_repaired"] is True


def test_an_independent_final_still_goes_stale_on_repair(tmp_path):
    db = _db(tmp_path, n=1)
    db.sync_evidence({"g0": "fp_old"})
    db.save_label("g0", "Ann", score=3.0, rubric=[], status="saved")
    ov = db.overview("g0")
    db.set_final("g0", score=3.0, rubric=[], note="", source="adjudicated", adjudicator="owner",
                 expected_item_revision=ov["revision"])
    db.sync_evidence({"g0": "fp_new"})
    ov = db.overview("g0")
    assert ov["final"]["evidence_stale"] is True and ov["state"] == STATE_EVIDENCE_REVIEW
    assert db.summary()["stale_finals"] == 1


def test_provenance_is_never_rewritten_under_an_existing_final(tmp_path):
    """A FINAL freezes ground_truth_source; changing the label's provenance
    underneath it would leave the two disagreeing, so the label is skipped."""
    db = _db(tmp_path, n=1)
    db.save_label("g0", "Ann", score=3.0, rubric=[], status="saved")
    ov = db.overview("g0")
    db.set_final("g0", score=3.0, rubric=[], note="", source="adjudicated", adjudicator="owner",
                 expected_item_revision=ov["revision"])
    out = db.set_label_provenance(grader="Ann", label_source="original_instructor_grade", asserted_by="owner")
    assert out["applied_count"] == 0 and out["skipped_count"] == 1
    assert "FINAL" in out["skipped"][0]["reason"]
    assert db.get_label("g0", "Ann")["label_source"] == DEFAULT_LABEL_SOURCE      # untouched
    assert db.final_rows()[0]["ground_truth_source"] == DEFAULT_LABEL_SOURCE      # still coherent
    db.reopen("g0")                                                              # the documented path
    out = db.set_label_provenance(grader="Ann", label_source="original_instructor_grade", asserted_by="owner")
    assert out["applied_count"] == 1


# ------------------------ §12/J: three dimensions stay separate in dataset status --

def _fake_manifest(cases):
    from autograder.benchmark.manifests import BenchCase, BenchmarkManifest
    return BenchmarkManifest(role="grade_primary", name="t", status="FROZEN", root=Path("."), hashes={},
                             components=["ALL"],
                             cases=[BenchCase(case_id=c["case_id"], split="DEV", component="ALL",
                                              inputs={}, label=c) for c in cases])


def test_dataset_status_keeps_ground_truth_and_transcription_separate():
    from autograder.benchmark.status import role_dataset_status
    full = [{"case_id": "a", "score": 4.0, "transcription_complete": True},
            {"case_id": "b", "score": 2.0, "transcription_complete": True}]
    st = role_dataset_status("grade_primary", _fake_manifest(full))
    assert st["status"] == "READY" and st["scorable_for_accuracy"] == 2
    assert st["transcription_incomplete"] == 0 and st["labeled_not_scorable"] == 0

    mixed = full + [{"case_id": "c", "score": 3.0, "transcription_complete": False}]
    st = role_dataset_status("grade_primary", _fake_manifest(mixed))
    # every case HAS ground truth, so it is not "needs owner labels" — but it is
    # not READY either: one case cannot be measured yet
    assert st["labeled"] == 3 and st["status"] == "PARTIALLY_READY"
    assert st["scorable_for_accuracy"] == 2 and st["labeled_not_scorable"] == 1
    assert st["transcription_incomplete_cases"] == ["c"]
    assert "INCOMPLETE" in st["detail"] and "transcribe" in (st["owner_action"] or "")

    unlabeled = [{"case_id": "a", "score": None, "transcription_complete": True}]
    st = role_dataset_status("grade_primary", _fake_manifest(unlabeled))
    assert st["status"] == "NEEDS_OWNER_LABELS" and st["scorable_for_accuracy"] == 0


@real_dataset
def test_live_dataset_reports_58_scorable_and_9_incomplete():
    """The real grade_primary dataset: 67 cases, 9 of them not measurable until
    their restored line is transcribed — derived, never hardcoded."""
    from autograder.benchmark.manifests import load_manifest
    from autograder.benchmark.status import role_dataset_status
    m = load_manifest("grade_primary")
    st = role_dataset_status("grade_primary", m)
    assert st["cases"] == 67
    assert st["transcription_incomplete"] == 9
    assert st["scorable_for_accuracy"] + st["labeled_not_scorable"] == st["labeled"]
    # and every incomplete case names exactly which line is missing
    for c in m.cases:
        missing = c.label.get("lines_without_audited_transcription") or []
        assert bool(missing) is (c.label.get("transcription_complete") is False)
        if missing:
            ids = {e["sample_id"] for e in c.label["evidence_lines"] if e.get("sample_id")}
            assert set(missing) <= ids
            assert c.label["line_count"] == len(c.label["evidence_lines"])


@real_dataset
def test_missing_transcriptions_command_lists_the_exact_lines(capsys):
    from autograder.cli import main as autograder_main
    assert autograder_main(["bench", "missing-transcriptions", "--role", "grade_primary", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["cases_total"] == 67 and out["cases_incomplete"] == 9 and out["cases_complete"] == 58
    for row in out["cases"]:
        assert row["missing"] and row["line_count"] == row["lines_transcribed"] + len(row["missing"])
        for m in row["missing"]:
            assert m["sample_id"] and m["image"] and (REPO / "evaluation" / m["image"]).exists()
            assert m["transcription_status"].startswith("no_audited_transcription:")
