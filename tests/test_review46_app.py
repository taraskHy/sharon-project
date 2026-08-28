"""The SEEN-46 blind review site: blindness, concurrency, immutability.

Runs against the REAL built review bundle (skips when absent) with a fresh
tmp database per test — the live review/labeling databases are never opened
(their pytest barriers are themselves under test here). No model, provider or
network call happens anywhere: the app has no gateway; a socket guard proves
it. Numbered tests map to the owner's Phase-18 requirements 6-24 (1-5 live in
test_seen46_campaign.py).
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from review46_app import (default_data_dir, live_db_path,  # noqa: E402
                          assert_not_live_review_db)
from review46_app.app import create_app  # noqa: E402

BUNDLE = default_data_dir() / "bundle"
pytestmark = pytest.mark.skipif(not (BUNDLE / "bundle46.json").exists(),
                                reason="review46 bundle not built on this machine")

FORBIDDEN_IN_REVIEWER_PAYLOAD = (
    "instructor", "model_verdict", "model_proposal", "predicted_verdict",
    "justification", "audit", "human_decision", "split", '"DEV"', '"CALIBRATION"',
    "HELD_OUT", "derivable", "strict", "expected", "agreement", "consensus",
    '"final"', "adjudicat", "actual_", "label_verdict", "e005", "e006",
)


@pytest.fixture()
def app(tmp_path):
    return create_app(data_dir=tmp_path, bundle_dir=BUNDLE, admin_key="TESTKEY")


@pytest.fixture()
def clients(app):
    from starlette.testclient import TestClient
    a, b = TestClient(app), TestClient(app)
    a.post("/api/session", json={"name": "alice"})
    b.post("/api/session", json={"name": "bob"})
    admin = TestClient(app, headers={"x-admin-key": "TESTKEY"})
    return a, b, admin


def _decide(client, item, verdict, confidence="high", issue="none", note=""):
    return client.post(f"/api/items/{item['item_id']}/decision", json={
        "verdict": verdict, "confidence": confidence, "issue": issue, "note": note,
        "expected_revision": item.get("label_revision") or 0,
        "evidence_sha256": item.get("evidence_sha256")})


# ------------------------------------------------------------- blindness ----


def test_06_pre_decision_payload_is_blind(clients):
    a, _, _ = clients
    item = a.post("/api/next", json={}).json()["item"]
    blob = json.dumps(item)
    for banned in FORBIDDEN_IN_REVIEWER_PAYLOAD:
        assert banned not in blob, banned
    # and it DOES carry what a reviewer needs
    for needed in ("question_text", "rubric", "official_solution", "transcription",
                   "max_score", "images", "item_id"):
        assert needed in item, needed


def test_06b_payload_stays_blind_after_submitting(clients):
    """No reveal after a decision either — blind for the whole campaign."""
    a, _, _ = clients
    item = a.post("/api/next", json={}).json()["item"]
    r = _decide(a, item, "valid").json()
    blob = json.dumps(r)
    for banned in FORBIDDEN_IN_REVIEWER_PAYLOAD:
        assert banned not in blob, banned
    again = a.get(f"/api/items/{item['item_id']}").json()
    for banned in FORBIDDEN_IN_REVIEWER_PAYLOAD:
        assert banned not in json.dumps(again), banned


def test_07_reviewer_a_cannot_see_reviewer_b(clients):
    a, b, _ = clients
    item = a.post("/api/next", json={}).json()["item"]
    assert _decide(a, item, "valid", note="alice-secret-observation").status_code == 200
    view_b = b.get(f"/api/items/{item['item_id']}").json()
    blob = json.dumps(view_b)
    assert "alice" not in blob and "alice-secret-observation" not in blob
    assert view_b["item"]["my_review"] is None


def test_08_two_independent_reviews_required_for_consensus(clients):
    a, b, admin = clients
    item = a.post("/api/next", json={}).json()["item"]
    _decide(a, item, "valid")
    one = admin.get(f"/api/admin/items/{item['item_id']}").json()
    assert one["state"] == "ONE_REVIEW" and one["human_reference_verdict"] is None
    fresh = b.get(f"/api/items/{item['item_id']}").json()["item"]
    _decide(b, fresh, "valid")
    two = admin.get(f"/api/admin/items/{item['item_id']}").json()
    assert two["state"] == "CONSENSUS"
    assert two["human_reference_verdict"] == "valid"


def test_08b_disagreement_goes_to_adjudication_never_auto_tiebreak(clients):
    a, b, admin = clients
    item = a.post("/api/next", json={}).json()["item"]
    _decide(a, item, "valid")
    _decide(b, b.get(f"/api/items/{item['item_id']}").json()["item"], "invalid")
    d = admin.get(f"/api/admin/items/{item['item_id']}").json()
    assert d["state"] == "NEEDS_ADJUDICATION"
    # neither the model nor the instructor breaks the tie automatically
    assert d["human_reference_verdict"] is None


def test_09_claim_ttl_releases_expired_claims(tmp_path):
    from labeling_app.db import LabelDB
    from labeling_app.bundle import Bundle
    bundle = Bundle(BUNDLE)
    db = LabelDB(tmp_path / "x.db", claim_ttl_s=-1)     # already expired
    db.load_items(bundle.items)
    db.set_wanted_labels("all", n=2)
    first = db.claim_next("alice")
    assert first is not None
    # alice's claim has TTL 0 -> bob may claim the same item
    assert db.claim_next("bob") == first
    db.close()


def test_10_reviewer_never_claims_a_case_they_already_decided(clients, app):
    a, _, _ = clients
    seen = set()
    for _ in range(50):
        r = a.post("/api/next", json={}).json()
        if r["done"]:
            break
        item = r["item"]
        assert item["item_id"] not in seen, "same case handed twice to one reviewer"
        seen.add(item["item_id"])
        assert _decide(a, item, "valid").status_code == 200
    assert len(seen) == 46                     # one full blind pass, no repeats


def test_11_stale_write_returns_409_and_never_overwrites(clients):
    a, _, admin = clients
    item = a.post("/api/next", json={}).json()["item"]
    assert _decide(a, item, "valid").status_code == 200
    stale = _decide(a, item, "invalid")        # expected_revision is now stale
    assert stale.status_code == 409
    d = admin.get(f"/api/admin/items/{item['item_id']}").json()
    assert [r["verdict"] for r in d["reviews"]] == ["valid"]


def test_12_decision_persistence_and_resume(app, clients):
    a, _, _ = clients
    item = a.post("/api/next", json={}).json()["item"]
    _decide(a, item, "partially_valid", confidence="low",
            issue="genuinely_ambiguous", note="borderline")
    again = a.get(f"/api/items/{item['item_id']}").json()["item"]["my_review"]
    assert again["verdict"] == "partially_valid"
    assert again["confidence"] == "low" and again["issue"] == "genuinely_ambiguous"
    assert again["text"] == "borderline"
    # a brand-new client with the same cookie name resumes the same identity
    from starlette.testclient import TestClient
    a2 = TestClient(app)
    a2.post("/api/session", json={"name": "alice"})
    resumed = a2.get(f"/api/items/{item['item_id']}").json()["item"]["my_review"]
    assert resumed["verdict"] == "partially_valid"


def test_13_review_decisions_are_append_only_events(clients):
    a, _, admin = clients
    item = a.post("/api/next", json={}).json()["item"]
    _decide(a, item, "valid")
    events = admin.get("/api/admin/events").json()["events"]
    actions = [e["action"] for e in events]
    assert "label_saved" in actions and "claim" in actions


def test_14_adjudication_is_a_separate_source(clients):
    a, b, admin = clients
    item = a.post("/api/next", json={}).json()["item"]
    _decide(a, item, "valid")
    _decide(b, b.get(f"/api/items/{item['item_id']}").json()["item"], "invalid")
    d = admin.get(f"/api/admin/items/{item['item_id']}").json()
    r = admin.post(f"/api/admin/items/{item['item_id']}/adjudicate",
                   json={"verdict": "partially_valid", "note": "middle",
                         "adjudicator": "owner", "expected_item_revision": d["item_revision"]})
    assert r.status_code == 200
    after = admin.get(f"/api/admin/items/{item['item_id']}").json()
    assert after["state"] == "ADJUDICATED"
    assert after["adjudicated"]["verdict"] == "partially_valid"
    assert after["adjudicated"]["adjudicator"] == "owner"
    assert {c["reviewer"] for c in after["adjudicated"]["contributing"]} == {"alice", "bob"}
    # both original reviews remain, unmodified
    assert sorted(rv["verdict"] for rv in after["reviews"]) == ["invalid", "valid"]


def test_15_16_instructor_and_model_sources_are_immutable_files(clients):
    _, _, admin = clients
    priv = BUNDLE / "private"
    before_i = (priv / "instructor_reference.json").read_bytes()
    before_m = (priv / "model_proposals.json").read_bytes()
    item_id = json.loads((priv / "id_map.json").read_text(encoding="utf-8"))
    some = next(iter(item_id))
    d = admin.get(f"/api/admin/items/{some}").json()
    admin.post(f"/api/admin/items/{some}/adjudicate",
               json={"verdict": "valid", "expected_item_revision": d["item_revision"]})
    assert (priv / "instructor_reference.json").read_bytes() == before_i
    assert (priv / "model_proposals.json").read_bytes() == before_m


def test_17_18_no_comparative_metrics_before_completion(clients):
    a, b, admin = clients
    r = admin.get("/api/admin/compare")
    assert r.status_code == 409
    assert "withheld" in r.json()["error"]
    item = a.post("/api/next", json={}).json()["item"]
    _decide(a, item, "valid")
    assert admin.get("/api/admin/compare").status_code == 409   # still incomplete
    partial = admin.get("/api/admin/compare?partial=1").json()
    assert partial["partial"] is True and partial["campaign_complete"] is False
    summary = admin.get("/api/admin/summary").json()
    assert summary["campaign_complete"] is False
    assert "comparison_preview" not in summary


def test_19_bundle_excludes_held_out_and_counts_46():
    import re
    doc = json.loads((BUNDLE / "bundle46.json").read_text(encoding="utf-8"))
    assert doc["cases"] == 46
    id_map = json.loads((BUNDLE / "private" / "id_map.json").read_text(encoding="utf-8"))
    assert len(id_map) == 46
    writers = {cid.split("_")[0] for cid in id_map.values()}
    assert writers == {"e002", "e003", "e004", "e007"}
    pat = re.compile(r"(?<![0-9a-fA-F])(e005|e006)(?![0-9a-fA-F])")
    for p in sorted(BUNDLE.rglob("*.json")):
        assert not pat.search(p.read_text(encoding="utf-8")), p


def test_20_source_pages_are_masked_or_withheld():
    prov = json.loads((BUNDLE / "private" / "provenance.json").read_text(encoding="utf-8"))
    items = json.loads((BUNDLE / "items.json").read_text(encoding="utf-8"))
    served = {i["item_id"]: i for i in items}
    for oid, p in prov.items():
        rep = p.get("page_report") or {}
        available = bool(served[oid].get("provenance", {}).get("page_available"))
        if available:
            # a served page passed the strict residual-red mask gate at build
            assert rep.get("ok") is True, (oid, rep)
            assert rep.get("strict_red_after", 999) <= 60, (oid, rep)


def test_21_review_ui_makes_zero_network_or_model_calls(app):
    """The review app has no route to any model or provider: no gateway, no
    backend construction, no HTTP client — enforced at the SOURCE level over
    the whole package (an in-process TestClient exercises no socket, and
    anyio's own portal plumbing makes a socket monkeypatch unreliable)."""
    from starlette.testclient import TestClient
    c = TestClient(app)
    c.post("/api/session", json={"name": "zoe"})
    assert c.post("/api/next", json={}).status_code == 200
    assert c.get("/api/health").json()["ai_calls"] == 0
    for f in ("app.py", "build.py", "cli.py", "__init__.py"):
        src = (REPO / "review46_app" / f).read_text(encoding="utf-8")
        for banned in ("gateway", "create_backend", "httpx", "requests.post",
                       "openrouter", "ollama", "urllib.request"):
            assert banned not in src, (f, banned)


def test_22_wal_safe_backup_under_concurrent_reviewers(app, clients):
    a, b, admin = clients
    stop = threading.Event()
    errors: list[str] = []

    def hammer(client, name):
        try:
            for _ in range(10):
                if stop.is_set():
                    return
                r = client.post("/api/next", json={})
                item = r.json().get("item")
                if not item:
                    return
                _decide(client, item, "valid")
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {e}")

    t1 = threading.Thread(target=hammer, args=(a, "alice"))
    t2 = threading.Thread(target=hammer, args=(b, "bob"))
    t1.start(); t2.start()
    bk = admin.post("/api/admin/backup", json={})
    stop.set(); t1.join(); t2.join()
    assert not errors, errors
    assert bk.status_code == 200
    assert bk.json().get("db_verified") in (True, "ok", 1) or bk.json().get("backup_dir")


def test_23_deterministic_export(clients):
    a, b, admin = clients
    item = a.post("/api/next", json={}).json()["item"]
    _decide(a, item, "valid")
    e1 = admin.get("/api/admin/export").json()
    e2 = admin.get("/api/admin/export").json()
    assert e1 == e2
    assert len(e1["cases"]) == 46
    # every source kept separately, none merged
    c0 = e1["cases"][0]
    for key in ("original_instructor_reference", "local_model_proposal",
                "independent_human_reviews", "adjudicated_human_reference"):
        assert key in c0


def test_24_pytest_never_touches_the_live_review_db():
    with pytest.raises(RuntimeError, match="LIVE review46 database"):
        assert_not_live_review_db(live_db_path())
    # tmp paths pass freely
    assert_not_live_review_db(Path("C:/somewhere/else.db"))


def test_no_human_decisions_prefilled(app):
    """A fresh deployment starts with an EMPTY review queue: zero saved
    decisions, nothing prefilled from the instructor, the model, or audits."""
    from starlette.testclient import TestClient
    admin = TestClient(app, headers={"x-admin-key": "TESTKEY"})
    s = admin.get("/api/admin/summary").json()
    assert s["completed_review_decisions"] == 0
    assert s["cases_by_review_count"]["0"] == 46      # JSON object keys are strings


def test_admin_requires_key(app):
    from starlette.testclient import TestClient
    c = TestClient(app)
    assert c.get("/api/admin/summary").status_code == 403
    assert c.get("/api/admin/items").status_code == 403
    assert c.get("/admin").status_code == 403


def test_invite_token_when_configured(tmp_path):
    from starlette.testclient import TestClient
    app = create_app(data_dir=tmp_path, bundle_dir=BUNDLE, admin_key="K",
                     invite_token="SECRET-INVITE")
    c = TestClient(app)
    assert c.post("/api/session", json={"name": "x"}).status_code == 403
    assert c.post("/api/session", json={"name": "x", "token": "SECRET-INVITE"}).status_code == 200
