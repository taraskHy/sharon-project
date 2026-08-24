"""Label PROVENANCE (schema 3) — how a score was DERIVED vs which evidence it was shown against.

Two independent facts that this project must never conflate:

* ``label_source``   — how the number was produced. ``human_independent_grading``
  is a judgment formed FROM the evidence the app displayed;
  ``original_instructor_grade`` is a score copied from the authoritative original
  instructor-graded exam for the whole grading unit, which never depended on what
  the app displayed at all.
* ``evidence_sha256`` — exactly which crops were on screen, and (via the item)
  whether that evidence was later repaired.

Therefore repairing an item's model-visible evidence makes an INDEPENDENT label
stale (it judged something incomplete) but must NOT invalidate an authoritative
one — while the repair itself stays recorded either way, because the benchmark
needs ``evidence_repaired`` for reproducibility.

Covers: the schema-3 migration of a legacy database, the guarded backfill (never
touches a score), provenance-aware staleness, the nine evidence-repaired
grade_primary cases, the second-grader rule, export/import separation, and the
leak rule that a grader never receives ``source_ref`` (its filename carries the
instructor's total grade). No model/API calls.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from labeling_app.app import create_app
from labeling_app.bundle import Bundle, build_bundle
from labeling_app.db import (AUTHORITATIVE_LABEL_SOURCES, DEFAULT_LABEL_SOURCE, STATE_AUTHORITATIVE,
                             STATE_EVIDENCE_REVIEW, LabelDB, LabelError)
from labeling_app.export import export_final

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "evaluation" / "model_selection" / "datasets" / "grade_primary"
real_dataset = pytest.mark.skipif(not (DATASET / "manifest.json").exists(),
                                  reason="grade_primary dataset not built")

#: the nine grade_primary cases whose evidence was repaired (a recorded
#: handwritten line was missing from the bundle) — commit 78f0185
REPAIRED_CASES = ("e003_q1_r5", "e003_q2_r2", "e003_q2_r3", "e003_q2_r4", "e003_q2_r7",
                  "e004_q2_r3", "e004_q2_r5", "e006_q2_r6", "e007_q1_r1")


def _items(n: int = 3) -> list[dict]:
    return [{"item_id": f"g{i}", "max_score": 4.0, "rubric_items": [{"id": "R1", "text": "r"}]}
            for i in range(n)]


def _db(tmp_path, n: int = 3) -> LabelDB:
    db = LabelDB(tmp_path / "labels.db")
    db.load_items(_items(n))
    return db


# ------------------------------------------------------------- migration --

def test_legacy_schema2_database_migrates_and_defaults_to_independent(tmp_path):
    """A pre-provenance database gains the columns; existing rows keep their
    scores and are treated as ordinary independent grading (never silently
    promoted to authoritative)."""
    p = tmp_path / "labels.db"
    con = sqlite3.connect(p)
    con.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE items (item_id TEXT PRIMARY KEY, max_score REAL NOT NULL,
            rubric_ids TEXT NOT NULL DEFAULT '[]', wanted_labels INTEGER NOT NULL DEFAULT 1,
            revision INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
        CREATE TABLE graders (name TEXT PRIMARY KEY, created_at TEXT NOT NULL, last_seen TEXT NOT NULL);
        CREATE TABLE claims (item_id TEXT NOT NULL, grader TEXT NOT NULL, claimed_at TEXT NOT NULL,
            expires_at REAL NOT NULL, PRIMARY KEY (item_id, grader));
        CREATE TABLE labels (item_id TEXT NOT NULL, grader TEXT NOT NULL, score REAL,
            rubric TEXT NOT NULL DEFAULT '[]', note TEXT NOT NULL DEFAULT '', status TEXT NOT NULL,
            flag_reason TEXT NOT NULL DEFAULT '', revision INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY (item_id, grader));
        CREATE TABLE final_labels (item_id TEXT PRIMARY KEY, score REAL NOT NULL,
            rubric TEXT NOT NULL DEFAULT '[]', note TEXT NOT NULL DEFAULT '', source TEXT NOT NULL,
            adjudicator TEXT NOT NULL DEFAULT '', contributing_graders TEXT NOT NULL DEFAULT '[]',
            from_revisions TEXT NOT NULL DEFAULT '{}', finalized_at TEXT NOT NULL,
            schema_version INTEGER NOT NULL);
        CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, grader TEXT NOT NULL,
            action TEXT NOT NULL, item_id TEXT, revision INTEGER, detail TEXT NOT NULL DEFAULT '{}');
        INSERT INTO items VALUES ('g0', 4.0, '["R1"]', 1, 0, '2026-08-01 10:00:00');
        INSERT INTO labels VALUES ('g0', 'Erik', 2.5, '[]', '', 'saved', '', 1,
                                   '2026-08-01 10:00:00', '2026-08-01 10:00:00');
    """)
    con.commit()
    con.close()

    db = LabelDB(p)                                   # migration runs on open
    lab = db.get_label("g0", "Erik")
    assert lab["score"] == 2.5                        # score preserved byte-for-byte
    assert lab["label_source"] == DEFAULT_LABEL_SOURCE
    assert lab["authoritative"] is False
    assert lab["entered_by"] == "" and lab["source_ref"] == ""


# ---------------------------------------------------------------- backfill --

def test_backfill_records_provenance_and_never_touches_the_score(tmp_path):
    db = _db(tmp_path)
    db.save_label("g0", "Erik", score=4.0, rubric=["R1"], status="saved")
    db.save_label("g1", "Erik", score=0.0, rubric=[], status="saved")
    before = {r["item_id"]: dict(r) for r in db.labels_for_item("g0") + db.labels_for_item("g1")}

    out = db.set_label_provenance(grader="Erik", label_source="original_instructor_grade",
                                  entered_by="Erik", asserted_by="owner",
                                  source_refs={"g0": "test/003_70.pdf#e003_q1_r5"})
    assert out["applied_count"] == 2 and out["scores_modified"] == 0

    g0 = db.get_label("g0", "Erik")
    g1 = db.get_label("g1", "Erik")
    assert g0["score"] == 4.0 and g1["score"] == 0.0                    # untouched
    assert g0["revision"] == before["g0"]["revision"]                   # no revision churn
    assert g0["label_source"] == "original_instructor_grade" and g0["authoritative"] is True
    assert g0["entered_by"] == "Erik" and g0["source_ref"] == "test/003_70.pdf#e003_q1_r5"
    assert g0["provenance_asserted_by"] == "owner" and g0["provenance_asserted_at"]
    assert g1["source_ref"] == ""                                       # no ref available -> empty, never invented


def test_backfill_is_audited_and_records_that_provenance_was_asserted_not_proved(tmp_path):
    db = _db(tmp_path)
    db.save_label("g0", "Erik", score=4.0, rubric=[], status="saved")
    db.set_label_provenance(grader="Erik", label_source="original_instructor_grade",
                            asserted_by="owner", actor="owner")
    actions = [e["action"] for e in db.events(limit=500)]
    assert "label_provenance_set" in actions and "label_provenance_backfill" in actions
    ev = next(e for e in db.events(limit=500) if e["action"] == "label_provenance_set")
    d = json.loads(ev["detail"])
    assert d["previous_label_source"] == DEFAULT_LABEL_SOURCE
    assert d["label_source"] == "original_instructor_grade"
    assert d["asserted_by"] == "owner"          # WHO asserted it is part of the record
    assert d["score_unchanged"] == 4.0


def test_backfill_dry_run_writes_nothing(tmp_path):
    db = _db(tmp_path)
    db.save_label("g0", "Erik", score=4.0, rubric=[], status="saved")
    out = db.set_label_provenance(grader="Erik", label_source="original_instructor_grade", dry_run=True)
    assert out["applied_count"] == 1 and out["dry_run"] is True
    assert db.get_label("g0", "Erik")["label_source"] == DEFAULT_LABEL_SOURCE


def test_backfill_guards_skip_non_saved_labels_rather_than_rewriting_them(tmp_path):
    db = _db(tmp_path)
    db.save_label("g0", "Erik", score=4.0, rubric=[], status="saved")
    db.save_label("g1", "Erik", score=None, rubric=[], status="skipped")
    out = db.set_label_provenance(grader="Erik", label_source="original_instructor_grade")
    assert out["applied_count"] == 1 and out["skipped_count"] == 1
    assert out["skipped"][0]["item_id"] == "g1"
    assert db.get_label("g1", "Erik")["label_source"] == DEFAULT_LABEL_SOURCE


def test_backfill_rejects_an_unknown_source(tmp_path):
    db = _db(tmp_path)
    db.save_label("g0", "Erik", score=4.0, rubric=[], status="saved")
    with pytest.raises(LabelError):
        db.set_label_provenance(grader="Erik", label_source="whatever_the_owner_said")


# ------------------------------------------- provenance-aware staleness --

def _change_evidence(db: LabelDB, item_id: str, fp: str) -> None:
    db.sync_evidence({item_id: fp})


def test_independent_label_still_goes_stale_when_evidence_changes(tmp_path):
    db = _db(tmp_path)
    db.sync_evidence({"g0": "fp_old"})
    db.save_label("g0", "Ann", score=3.0, rubric=[], status="saved")
    _change_evidence(db, "g0", "fp_new")
    assert db.get_label("g0", "Ann")["evidence_stale"] is True
    assert db.evidence_report()["labels_stale"] == 1
    assert db.overview("g0")["state"] == STATE_EVIDENCE_REVIEW


def test_authoritative_label_does_not_go_stale_when_evidence_changes(tmp_path):
    db = _db(tmp_path)
    db.sync_evidence({"g0": "fp_old"})
    db.save_label("g0", "Erik", score=4.0, rubric=[], status="saved")
    db.set_label_provenance(grader="Erik", label_source="original_instructor_grade", asserted_by="owner")
    _change_evidence(db, "g0", "fp_new")

    lab = db.get_label("g0", "Erik")
    assert lab["evidence_stale"] is False and lab["score"] == 4.0
    rep = db.evidence_report()
    assert rep["labels_stale"] == 0 and rep["labels_fresh"] == 1
    assert rep["labels_authoritative"] == 1
    # ...but the repair itself is still on the record, separately
    assert len(rep["authoritative_labels_on_repaired_evidence"]) == 1
    assert len(rep["items_evidence_changed"]) == 1
    assert db.overview("g0")["evidence_repaired"] is True
    assert db.overview("g0")["evidence_previous_sha256"] == "fp_old"


def test_authoritative_item_is_not_re_served_for_re_review(tmp_path):
    db = _db(tmp_path, n=1)
    db.sync_evidence({"g0": "fp_old"})
    db.save_label("g0", "Erik", score=4.0, rubric=[], status="saved")
    db.set_label_provenance(grader="Erik", label_source="original_instructor_grade")
    _change_evidence(db, "g0", "fp_new")
    assert db.claim_next("Erik") is None                       # never handed back
    assert db.my_items("Erik")["stale"] == []
    assert db.progress("Erik")["remaining_for_me"] == 0


def test_authoritative_item_never_requests_a_second_independent_grader(tmp_path):
    db = _db(tmp_path, n=1)
    db.set_wanted_labels("all", n=2)                           # ordinarily wants two labels
    db.save_label("g0", "Erik", score=4.0, rubric=[], status="saved")
    db.set_label_provenance(grader="Erik", label_source="original_instructor_grade")
    assert db.claim_next("Someone-Else") is None
    ov = db.overview("g0")
    assert ov["state"] == STATE_AUTHORITATIVE
    assert ov["n_authoritative"] == 1 and ov["agreement"] is None
    s = db.summary()
    assert s["awaiting_second_label"] == 0 and s["authoritative_ground_truth"] == 1


def test_independent_item_still_requests_a_second_grader(tmp_path):
    db = _db(tmp_path, n=1)
    db.set_wanted_labels("all", n=2)
    db.save_label("g0", "Ann", score=3.0, rubric=[], status="saved")
    assert db.claim_next("Bob") == "g0"
    assert db.summary()["awaiting_second_label"] == 1


# ------------------------------------------------------- the nine cases --

@real_dataset
def test_the_nine_repaired_cases_keep_authoritative_labels_valid(tmp_path):
    """End to end on the real dataset: label all cases, repair the evidence of
    the nine, and confirm authoritative labels survive while independent ones
    do not — with the repair still recorded for every one of the nine."""
    out = tmp_path / "b"
    build_bundle(DATASET, out, evaluation_root=REPO / "evaluation", page_max_edge=600,
                 now="2026-08-22 12:00:00")
    bundle = Bundle(out)
    case_to_item = {v: k for k, v in bundle.id_map.items()}
    assert all(c in case_to_item for c in REPAIRED_CASES)

    db = LabelDB(tmp_path / "labels.db")
    db.load_items([{"item_id": i["item_id"], "max_score": i["max_score"],
                    "rubric_ids": [r["id"] for r in i.get("rubric_items") or []]} for i in bundle.items])
    db.sync_evidence(bundle.fingerprints)

    for iid in bundle.by_id:
        db.save_label(iid, "Erik", score=4.0, rubric=[], status="saved")
    db.save_label(case_to_item["e003_q1_r5"], "Ann", score=2.0, rubric=[], status="saved")

    db.set_label_provenance(grader="Erik", label_source="original_instructor_grade",
                            entered_by="Erik", asserted_by="owner")

    # repair: the nine items' evidence changes; everything else stays put
    changed = {case_to_item[c]: f"repaired_{c}" for c in REPAIRED_CASES}
    db.sync_evidence({**bundle.fingerprints, **changed})

    rep = db.evidence_report()
    assert rep["labels_total"] == len(bundle.by_id) + 1
    assert rep["labels_preserved"] == rep["labels_total"]
    # Erik's authoritative labels: none stale. Ann's independent one: stale.
    assert rep["labels_stale"] == 1
    assert rep["stale_labels"][0]["grader"] == "Ann"
    assert rep["stale_labels"][0]["label_source"] == DEFAULT_LABEL_SOURCE
    assert rep["labels_authoritative"] == len(bundle.by_id)
    # the repair is recorded for all nine regardless of provenance
    assert len(rep["items_evidence_changed"]) == len(REPAIRED_CASES)
    assert {r["item_id"] for r in rep["authoritative_labels_on_repaired_evidence"]} == set(changed)

    for c in REPAIRED_CASES:
        iid = case_to_item[c]
        assert db.get_label(iid, "Erik")["evidence_stale"] is False
        ov = db.overview(iid)
        assert ov["evidence_repaired"] is True and ov["evidence_previous_sha256"]
        assert ov["ground_truth_source"] == "original_instructor_grade"
    # nobody is asked to re-grade any of the nine on Erik's behalf
    assert db.progress("Erik")["remaining_for_me"] == 0


@real_dataset
def test_the_nine_repaired_cases_are_scorable_once_their_repair_is_complete():
    """Before the human repair these nine were invisible in part to the grading
    model — it consumes the TRANSCRIPTION — so they could not count toward model
    accuracy. The owner has since transcribed or ruled on every one of them, so
    the exclusion no longer applies: they are scorable BECAUSE each carries a
    complete, human-verified decision. Both halves of that are asserted here."""
    from tests.prerepair import pre_repair_rows, repair_store

    _, before = pre_repair_rows()
    before_rows = {r["case_id"]: r for r in before}
    assert {c for c, r in before_rows.items() if r["transcription_complete"] is False} == set(REPAIRED_CASES)
    for c in REPAIRED_CASES:
        assert len(before_rows[c]["lines_without_audited_transcription"]) == 1, c

    after = {json.loads(l)["case_id"]: json.loads(l)
             for l in (DATASET / "cases_labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    store = repair_store()
    assert {c for c, r in after.items() if r.get("transcription_complete") is False} == set()
    for c in REPAIRED_CASES:
        row = after[c]
        assert row["transcription_complete"] is True and row["lines_without_audited_transcription"] == []
        assert len(row["evidence_repairs"]) == 1, c
        rec = store[row["evidence_repairs"][0]]
        assert rec["human_verified"] is True and rec["verified_by"], c
        assert rec["disposition"] in ("transcribed", "no_text_segmentation_artifact"), c
    # the score itself never moved: the repair changed the model's input, not the label
    assert [after[c]["score"] for c in REPAIRED_CASES] == [before_rows[c]["score"] for c in REPAIRED_CASES]
    assert [after[c]["max_score"] for c in REPAIRED_CASES] == [before_rows[c]["max_score"] for c in REPAIRED_CASES]


def test_a_transcription_incomplete_case_is_excluded_from_grading_accuracy():
    """The exclusion MECHANISM stays covered even though no real case trips it
    any more — it is what protects accuracy metrics from partial evidence."""
    from autograder.benchmark.roles import GradeAdapter

    def _row(cid, complete, score):
        # verdict target: the primary metric is the explanation verdict, and a
        # partially-evidenced case must stay out of it exactly as before
        verdict = "valid" if score == 4.0 else "invalid"
        return {"case_id": cid, "split": "DEV", "component": "grade", "decision": "AUTO",
                "schema_failure": False, "score": score, "label_score": 4.0,
                "label_verdict": "valid", "predicted_verdict": verdict,
                "verdict_exact": verdict == "valid",
                "final_exact": score == 4.0, "final_abs_error": abs(score - 4.0),
                "harmful_upgrade": score > 4.0, "harmful_downgrade": score < 4.0,
                "uncertain": False, "validation_ok": True,
                "transcription_complete": complete}

    # one ordinary case, one whose restored line has no audited transcription
    agg = GradeAdapter("grade_primary").aggregate([_row("a", True, 4.0), _row("b", False, 0.0)], [])
    assert agg["labeled_excluded_transcription_incomplete"] == 1
    # the incomplete case does NOT drag accuracy down — it is simply not measured
    assert agg["verdict_cases"] == 1
    assert agg["verdict_exact_pct"] == 100.0
    assert agg["final_score_exact_pct"] == 100.0


# --------------------------------------------------------- export / leak --

def test_export_separates_ground_truth_source_from_evidence_version(tmp_path):
    db = _db(tmp_path, n=1)
    db.sync_evidence({"g0": "fp_old"})
    db.save_label("g0", "Erik", score=4.0, rubric=[], status="saved")
    db.set_label_provenance(grader="Erik", label_source="original_instructor_grade",
                            asserted_by="owner", source_refs={"g0": "test/003_70.pdf#e003_q1_r5"})
    ov = db.overview("g0")
    db.set_final("g0", score=4.0, rubric=[], note="", source="adjudicated",
                 adjudicator="owner", expected_item_revision=ov["revision"])
    db.sync_evidence({"g0": "fp_new"})        # evidence repaired AFTER the final

    class _B:
        meta = {"items_sha256": "x", "evidence_fingerprint": "y", "eligibility": {}, "source": {}}
        id_map = {"g0": "e003_q1_r5"}
        eligibility = {"g0": {"eligible_for_human_label": True}}

    data = export_final(db, _B(), now="2026-08-22 13:00:00")
    row = data["items"][0]
    assert row["ground_truth_score"] == 4.0
    assert row["ground_truth_source"] == "original_instructor_grade"
    assert row["evidence_repaired"] is True and row["evidence_previous_sha256"] == "fp_old"
    # an authoritative FINAL is NOT invalidated by the repair
    assert row["evidence_stale"] is False
    assert data["stale_evidence_final_count"] == 0 and data["evidence_repaired_count"] == 1
    assert data["ground_truth_sources"] == {"original_instructor_grade": 1}
    assert row["labels"][0]["label_source"] == "original_instructor_grade"
    assert row["labels"][0]["source_ref"] == "test/003_70.pdf#e003_q1_r5"


@real_dataset
def test_grader_never_receives_the_private_source_reference(tmp_path):
    """``source_ref`` names the original graded PDF and that filename carries the
    instructor's TOTAL grade — it must never reach a grader."""
    out = tmp_path / "b"
    build_bundle(DATASET, out, evaluation_root=REPO / "evaluation", page_max_edge=600,
                 now="2026-08-22 12:00:00")
    app = create_app(data_dir=tmp_path / "data", bundle_dir=out, admin_key="k")
    db: LabelDB = app.state.db
    with TestClient(app) as c:
        c.post("/api/session", json={"name": "Erik"})
        iid = c.post("/api/next").json()["item"]["item_id"]
        c.post(f"/api/items/{iid}/label", json={"score": 4.0, "status": "saved", "expected_revision": 0})
        db.set_label_provenance(grader="Erik", label_source="original_instructor_grade",
                                asserted_by="owner", source_refs={iid: "test/003_70.pdf#e003_q1_r5"})
        body = json.dumps(c.get(f"/api/items/{iid}").json())
        assert "003_70" not in body and ".pdf" not in body and "source_ref" not in body
        assert "provenance_asserted_by" not in body
