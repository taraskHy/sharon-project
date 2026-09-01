"""Owner decision 2026-09-02 for the repaired cases (e004_q2_r6 / e004_q2_r8):

* reviewers are NEVER asked to redo them — their stale reviews stay preserved,
  the cases never re-enter any reviewer's claim queue, and a direct submit is
  refused;
* the OWNER assigns the repaired reference verdict once, stored distinctly as
  ``owner_adjudicated_after_source_repair`` — a separate source that is never
  presented as (or mixed into) two-reviewer consensus metrics;
* comparisons keep three clearly-separated layers: consensus-track metrics,
  the owner-adjudicated block, and a combined all-sources diagnostic with
  per-source counts.

All tests here run on a fresh tmp database against the real built bundle —
the live review database is never opened. No model / provider / network call.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from review46_app import default_data_dir  # noqa: E402
from review46_app.app import create_app  # noqa: E402

BUNDLE = default_data_dir() / "bundle"
pytestmark = pytest.mark.skipif(not (BUNDLE / "bundle46.json").exists(),
                                reason="review46 bundle not built on this machine")


@pytest.fixture()
def app(tmp_path):
    return create_app(data_dir=tmp_path, bundle_dir=BUNDLE, admin_key="K")


@pytest.fixture()
def clients(app):
    from starlette.testclient import TestClient
    a, b = TestClient(app), TestClient(app)
    a.post("/api/session", json={"name": "alice"})
    b.post("/api/session", json={"name": "bob"})
    admin = TestClient(app, headers={"x-admin-key": "K"})
    return a, b, admin


def _decide(client, iid, verdict):
    it = client.get(f"/api/items/{iid}").json()["item"]
    return client.post(f"/api/items/{iid}/decision", json={
        "verdict": verdict, "confidence": "high", "issue": "none",
        "expected_revision": it.get("label_revision") or 0,
        "evidence_sha256": it.get("evidence_sha256")})


def _park_one_case(app, a, b, verdict_a="valid", verdict_b="valid"):
    """Both reviewers decide one case, then its evidence is corrected —
    exactly the live repair cascade. Returns the item id."""
    iid = a.post("/api/next", json={}).json()["item"]["item_id"]
    _decide(a, iid, verdict_a)
    _decide(b, iid, verdict_b)
    app.state.db.sync_evidence({iid: "corrected-after-owner-confirmed-repair"})
    return iid


# --------------------------------------------------- reviewers stay out -----


def test_parked_case_never_reenters_a_reviewer_queue(app, clients):
    a, b, admin = clients
    iid = _park_one_case(app, a, b)
    assert admin.get(f"/api/admin/items/{iid}").json()["state"] == "STALE_PENDING_RE_REVIEW"
    # alice's own stale label would be claim_next's TOP priority — the app
    # must end her queue instead of serving the parked case
    r = a.post("/api/next", json={}).json()
    assert r["done"] is True and r["item"] is None
    assert "owner" in r["message"]
    # and the transient claim was released — nothing is left held on the case
    import time
    with app.state.db._conn() as c:  # noqa: SLF001 — read-only assertion
        held = c.execute("SELECT COUNT(*) FROM claims WHERE item_id=? AND expires_at > ?",
                         (iid, time.time())).fetchone()[0]
    assert held == 0


def test_direct_submit_on_a_parked_case_is_refused(app, clients):
    a, b, _ = clients
    iid = _park_one_case(app, a, b)
    r = _decide(a, iid, "invalid")
    assert r.status_code == 409
    assert r.json()["parked"] is True
    r2 = a.post(f"/api/items/{iid}/skip", json={})
    assert r2.status_code == 409 and r2.json()["parked"] is True
    r3 = a.post(f"/api/items/{iid}/flag", json={"reason": "x"})
    assert r3.status_code == 409 and r3.json()["parked"] is True


def test_stale_reviews_stay_preserved_and_uncounted(app, clients):
    a, b, admin = clients
    iid = _park_one_case(app, a, b, "valid", "partially_valid")
    d = admin.get(f"/api/admin/items/{iid}").json()
    assert len(d["stale_reviews"]) == 2                       # preserved
    assert d["reviews"] == []                                 # never counted fresh
    assert d["human_reference_verdict"] is None               # no silent consensus


# ------------------------------------------- the owner's repaired verdict ----


def test_owner_adjudication_is_stored_as_its_own_source(app, clients):
    a, b, admin = clients
    iid = _park_one_case(app, a, b)
    d = admin.get(f"/api/admin/items/{iid}").json()
    r = admin.post(f"/api/admin/items/{iid}/adjudicate", json={
        "verdict": "partially_valid", "note": "repaired evidence supports partial credit",
        "adjudicator": "owner", "expected_item_revision": d["item_revision"]})
    assert r.status_code == 200
    case = r.json()["case"]
    assert case["state"] == "OWNER_ADJUDICATED_AFTER_REPAIR"
    assert case["human_reference_source"] == "owner_adjudicated_after_source_repair"
    assert case["adjudicated"]["kind"] == "owner_adjudicated_after_source_repair"
    assert case["human_reference_verdict"] == "partially_valid"
    # the db final's note carries the distinct kind + the stale history
    note = json.loads(r.json()["final"]["note"])
    assert note["kind"] == "owner_adjudicated_after_source_repair"
    assert {h["reviewer"] for h in note["stale_historical_reviews"]} == {"alice", "bob"}
    # once resolved, the case never returns to any queue (final blocks claims)
    assert a.post("/api/next", json={}).json()["item"]["item_id"] != iid


def test_ordinary_disagreement_adjudication_keeps_its_own_kind(app, clients):
    a, b, admin = clients
    iid = a.post("/api/next", json={}).json()["item"]["item_id"]
    _decide(a, iid, "valid")
    _decide(b, iid, "invalid")
    d = admin.get(f"/api/admin/items/{iid}").json()
    assert d["state"] == "NEEDS_ADJUDICATION"
    r = admin.post(f"/api/admin/items/{iid}/adjudicate", json={
        "verdict": "valid", "adjudicator": "owner",
        "expected_item_revision": d["item_revision"]})
    case = r.json()["case"]
    assert case["state"] == "ADJUDICATED"
    assert case["adjudicated"]["kind"] == "adjudicated_human_reference"
    assert case["human_reference_source"] == "adjudicated_human_reference"
    assert "kind" not in json.loads(r.json()["final"]["note"]) or \
        json.loads(r.json()["final"]["note"]).get("kind") != "owner_adjudicated_after_source_repair"


# ----------------------------------------------- three separated layers -----


def test_comparisons_keep_owner_refs_out_of_consensus_metrics(app, clients):
    a, b, admin = clients
    # one consensus case
    cid = a.post("/api/next", json={}).json()["item"]["item_id"]
    _decide(a, cid, "valid")
    _decide(b, cid, "valid")
    # one parked + owner-adjudicated case
    iid = _park_one_case(app, a, b)
    d = admin.get(f"/api/admin/items/{iid}").json()
    admin.post(f"/api/admin/items/{iid}/adjudicate", json={
        "verdict": "invalid", "adjudicator": "owner",
        "expected_item_revision": d["item_revision"]})
    cmp_ = admin.get("/api/admin/compare?partial=1").json()
    # layer 1: consensus-track counts exclude the owner-adjudicated case
    assert cmp_["human_reference_cases"] == 1
    # layer 2: the owner block reports it separately, with full provenance
    blockcases = cmp_["owner_adjudicated_after_source_repair"]["cases"]
    assert len(blockcases) == 1
    oc = blockcases[0]
    assert oc["owner_reference_verdict"] == "invalid"
    assert oc["adjudicator"] == "owner"
    assert len(oc["stale_historical_reviews"]) == 2
    assert all(h["stale"] for h in oc["stale_historical_reviews"])
    assert "consensus" not in json.dumps(blockcases)          # never called consensus
    # layer 3: the combined diagnostic distinguishes the sources by count
    comb = cmp_["combined_diagnostic_all_sources"]
    assert comb["human_reference_by_source"] == {
        "independent_two_reviewer_consensus": 1,
        "owner_adjudicated_after_source_repair": 1}


def test_export_carries_source_provenance_per_case(app, clients):
    a, b, admin = clients
    iid = _park_one_case(app, a, b)
    d = admin.get(f"/api/admin/items/{iid}").json()
    admin.post(f"/api/admin/items/{iid}/adjudicate", json={
        "verdict": "valid", "adjudicator": "owner",
        "expected_item_revision": d["item_revision"]})
    exp = admin.get("/api/admin/export").json()
    row = next(c for c in exp["cases"] if c["item_id"] == iid)
    assert row["human_reference_source"] == "owner_adjudicated_after_source_repair"
    assert len(row["stale_historical_reviews"]) == 2
    assert row["independent_human_reviews"] == []


def test_owner_resolution_completes_the_campaign_state(app, clients):
    """A parked case is analysis-ready but not campaign-complete; the owner's
    verdict resolves it fully."""
    from review46_app.app import _campaign_complete, _analysis_ready
    parked = {"n_fresh_saved": 0, "state": "STALE_PENDING_RE_REVIEW"}
    resolved = {"n_fresh_saved": 0, "state": "OWNER_ADJUDICATED_AFTER_REPAIR"}
    done = {"n_fresh_saved": 2, "state": "CONSENSUS"}
    assert _analysis_ready([done, parked]) and not _campaign_complete([done, parked])
    assert _analysis_ready([done, resolved]) and _campaign_complete([done, resolved])


def test_summary_exposes_the_owner_queue(app, clients):
    a, b, admin = clients
    iid = _park_one_case(app, a, b)
    s = admin.get("/api/admin/summary").json()
    case_id = app.state.bundle.id_map[iid]
    assert s["owner_adjudication_queue"] == [case_id]
    assert s["owner_adjudicated_after_repair"] == []
    d = admin.get(f"/api/admin/items/{iid}").json()
    admin.post(f"/api/admin/items/{iid}/adjudicate", json={
        "verdict": "valid", "adjudicator": "owner",
        "expected_item_revision": d["item_revision"]})
    s2 = admin.get("/api/admin/summary").json()
    assert s2["owner_adjudication_queue"] == []
    assert s2["owner_adjudicated_after_repair"] == [case_id]
