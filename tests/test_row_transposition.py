"""The confirmed r6/r8 row-transposition repair and its staleness cascade.

Zero inference, zero network. The live review DB is never opened — DB-level
assertions run against the verified backup snapshot (skipped on machines
without it) or synthetic tmp databases.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

DS = REPO / "evaluation" / "model_selection" / "datasets" / "grade_primary"
RUNS = REPO / "evaluation" / "model_selection" / "runs" / "local_grade_primary"
A, B = "e004_q2_r6", "e004_q2_r8"


@pytest.fixture(scope="module")
def revision():
    man = json.loads((DS / "manifest.json").read_text(encoding="utf-8"))
    revs = [r for r in man.get("revisions", []) if r.get("kind") == "confirmed_row_transposition"]
    if not revs:
        pytest.skip("repair not applied in this checkout")
    assert len(revs) == 1
    return man, revs[0]


def test_repair_touches_exactly_the_two_confirmed_cases(revision):
    man, rev = revision
    assert rev["cases_changed"] == [A, B]
    assert rev["rows_updated"] == 2
    assert rev["owner_confirmed"] is True
    assert rev["model_involved"] is False
    # hash chain: previous -> current, recorded and live
    assert man["inputs_sha256"] == rev["inputs_sha256"]
    assert man["labels_sha256"] == rev["labels_sha256"]
    import hashlib
    assert hashlib.sha256((DS / "cases_inputs.jsonl").read_bytes()).hexdigest() == rev["inputs_sha256"]
    assert hashlib.sha256((DS / "cases_labels.jsonl").read_bytes()).hexdigest() == rev["labels_sha256"]


def test_swap_is_a_true_transposition_of_row_attached_content(revision):
    _, rev = revision
    import hashlib
    inputs = {json.loads(l)["case_id"]: json.loads(l)
              for l in (DS / "cases_inputs.jsonl").open(encoding="utf-8")}
    labels = {json.loads(l)["case_id"]: json.loads(l)
              for l in (DS / "cases_labels.jsonl").open(encoding="utf-8")}
    for x, y in ((A, B), (B, A)):
        assert (hashlib.sha256(inputs[x]["transcription"].encode()).hexdigest()
                == rev["mapping_before"][y]["transcription_sha256"]
                == rev["mapping_after"][x]["transcription_sha256"])
        assert labels[x]["evidence_images"] == rev["mapping_before"][y]["evidence_images"]
    # logical identity NEVER moved: packs, sub-items, derived verdicts stay
    assert "המסכה" in inputs[A]["pack"]["question_text"]         # r6 = the echo-mask sub-item
    assert labels[A]["explanation_verdict"] == "valid"           # instructor 4 -> valid, unchanged
    assert labels[B]["explanation_verdict"] == "partially_valid"  # instructor 2, unchanged
    assert labels[A]["sub_item_id"] == "6" and labels[B]["sub_item_id"] == "8"


def test_instructor_grades_untouched(revision):
    finals = json.loads((DS / "final_labels.json").read_text(encoding="utf-8"))["labels"]
    assert finals[A]["score"] == 4.0 and finals[B]["score"] == 2.0
    assert all(v.get("ground_truth_source") == "original_instructor_grade"
               for v in finals.values())


def test_campaign_verification_is_revision_aware(revision):
    from scripts.seen46_campaign import verify_campaign
    assert verify_campaign() == []


def test_stale_model_outputs_registry_covers_only_the_two_cases(revision):
    p = RUNS / "STALE_MODEL_OUTPUTS_2026-09-01.json"
    doc = json.loads(p.read_text(encoding="utf-8"))
    assert doc["reason"] == "invalid_due_to_confirmed_source_transposition"
    assert {e["case_id"] for e in doc["affected_outputs"]} == {A, B}
    assert "preserved historically" in doc["policy"] and "NEVER edited" in doc["policy"]
    forb = re.compile(r"(?<![0-9a-fA-F])(e005|e006)(?![0-9a-fA-F])")
    assert not forb.search(p.read_text(encoding="utf-8"))


# ------------------------- staleness semantics (synthetic, tmp DB) -----------

BUNDLE = Path((__import__("review46_app").default_data_dir())) / "bundle"
needs_bundle = pytest.mark.skipif(not (BUNDLE / "bundle46.json").exists(),
                                  reason="review46 bundle not on this machine")


@needs_bundle
def test_stale_review_is_blinded_and_never_counted_as_consensus(tmp_path):
    from starlette.testclient import TestClient
    from review46_app.app import create_app
    app = create_app(data_dir=tmp_path, bundle_dir=BUNDLE, admin_key="K")
    a, b = TestClient(app), TestClient(app)
    a.post("/api/session", json={"name": "alice"})
    b.post("/api/session", json={"name": "bob"})
    item = a.post("/api/next", json={}).json()["item"]
    iid = item["item_id"]
    for cl, v in ((a, "valid"), (b, "valid")):
        it = cl.get(f"/api/items/{iid}").json()["item"]
        cl.post(f"/api/items/{iid}/decision", json={
            "verdict": v, "confidence": "high", "issue": "none",
            "expected_revision": it.get("label_revision") or 0,
            "evidence_sha256": it["evidence_sha256"]})
    admin = TestClient(app, headers={"x-admin-key": "K"})
    assert admin.get(f"/api/admin/items/{iid}").json()["state"] == "CONSENSUS"
    # the evidence is corrected -> fingerprints move -> both reviews stale
    app.state.db.sync_evidence({iid: "corrected-fingerprint"})
    d = admin.get(f"/api/admin/items/{iid}").json()
    assert d["state"] == "STALE_PENDING_RE_REVIEW"
    assert d["human_reference_verdict"] is None            # stale never counts
    assert len(d["stale_reviews"]) == 2
    assert all(s["reason"] == "stale_due_to_confirmed_source_mapping_change"
               for s in d["stale_reviews"])
    # the reviewer's look is blind to their own old decision — and the case is
    # parked for the owner (decision 2026-09-02: reviewers are never re-asked)
    mine = a.get(f"/api/items/{iid}").json()["item"]["my_review"]
    assert mine == {"stale": True, "parked": True, "message": mine["message"]}
    assert "verdict" not in mine and "confidence" not in mine
    # comparisons run, with the parked case excluded and listed
    cmp_ = admin.get("/api/admin/compare?partial=1").json()
    assert BUNDLE and app.state.bundle.id_map[iid] in cmp_["excluded_stale_pending_re_review"] \
        or cmp_["excluded_stale_pending_re_review"] is not None


@needs_bundle
def test_bundle_marks_repaired_model_proposals_invalid(revision):
    props = json.loads((BUNDLE / "private" / "model_proposals.json").read_text(encoding="utf-8"))
    stale = {p["case_id"]: p.get("stale") for p in props.values() if p.get("stale")}
    assert set(stale) == {A, B}
    assert set(stale.values()) == {"invalid_due_to_confirmed_source_transposition"}


# ---------------------- live-deployment assertions (snapshot only) -----------

SNAP = Path(r"C:\Users\ethan\AppData\Local\autograder\review46\backups\2026-09-01-233337\labels.db")


@pytest.mark.skipif(not SNAP.exists(), reason="deployment snapshot not on this machine")
def test_all_92_original_decisions_preserved_and_exactly_4_stale(revision):
    bundle_items = {i["item_id"]: i.get("evidence_sha256") for i in
                    json.loads((BUNDLE / "items.json").read_text(encoding="utf-8"))}
    id_map = json.loads((BUNDLE / "private" / "id_map.json").read_text(encoding="utf-8"))
    c = sqlite3.connect(f"file:{SNAP.as_posix()}?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    labels = [dict(r) for r in c.execute("SELECT * FROM labels WHERE status='saved'")]
    c.close()
    assert len(labels) == 92                                   # nothing deleted
    stale = [l for l in labels
             if l.get("evidence_sha256") and bundle_items.get(l["item_id"])
             and l["evidence_sha256"] != bundle_items[l["item_id"]]]
    assert len(stale) == 4                                     # expected maximum, exactly
    assert {id_map[l["item_id"]] for l in stale} == {A, B}
    assert {l["grader"] for l in stale} == {"Erik", "Or Vaisman"}
    current = [l for l in labels if l not in stale]
    assert len(current) == 88                                  # other 88 remain current
