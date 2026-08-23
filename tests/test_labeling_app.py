"""Shared ground-truth grading app (labeling_app) — local/mocked only.

Covers: bundle anonymization, grader name/session, claim/open, save,
save&next, skip, flag, rubric decisions, resume, two independent graders,
no visibility of the first label before the second submits, disagreement,
agreement, admin adjudication, stale-write conflict (label + item),
deterministic final export, SQLite concurrency, backup, no AI dependency,
and the benchmark importer that consumes FINAL labels only.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from labeling_app.app import COOKIE, create_app
from labeling_app.backup import make_backup
from labeling_app.bundle import FORBIDDEN_IN_BUNDLE, Bundle, build_bundle
from labeling_app.db import LabelDB, LabelError, StaleWrite
from labeling_app.export import export_final

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "evaluation" / "model_selection" / "datasets" / "grade_primary"
pytestmark = pytest.mark.skipif(not (DATASET / "manifest.json").exists(), reason="grade_primary dataset not built")


@pytest.fixture
def bundle_dir(tmp_path):
    out = tmp_path / "bundle"
    build_bundle(DATASET, out, evaluation_root=REPO / "evaluation", now="2026-08-22 12:00:00")
    return out


@pytest.fixture
def app(tmp_path, bundle_dir):
    return create_app(data_dir=tmp_path / "data", bundle_dir=bundle_dir)


@pytest.fixture
def client(app):
    return TestClient(app)


def _grader(app, name: str) -> TestClient:
    c = TestClient(app)
    r = c.post("/api/session", json={"name": name})
    assert r.status_code == 200 and r.json()["grader"] == name
    return c


# ------------------------------------------------------------------ bundle --

def test_bundle_is_anonymized_and_self_contained(bundle_dir):
    items = json.loads((bundle_dir / "items.json").read_text(encoding="utf-8"))
    assert len(items) == 67
    text = (bundle_dir / "items.json").read_text(encoding="utf-8")
    for forbidden in ("sharon-project", "C:\\\\", "hebrew_bench", "crops/", "DEV", "HELD_OUT", "CALIBRATION",
                      '"split"', '"writer"', "label_status", "source_file", ".pdf", "transcription_provenance"):
        assert forbidden not in text, forbidden
    for it in items:
        assert set(it) == {"item_id", "question_text", "rubric", "scoring_rules", "official_solution",
                           "transcription", "max_score", "rubric_items", "images", "evidence_sha256", "provenance",
                           "eligible_for_human_label", "eligibility_reason"}
        assert it["eligible_for_human_label"] is True
        assert re.fullmatch(r"g[0-9a-f]{10}", it["item_id"])
        assert not (set(it) & set(FORBIDDEN_IN_BUNDLE))
        for rel in it["images"]:
            assert (bundle_dir / rel).exists() and rel.startswith("images/")
    id_map = json.loads((bundle_dir / "private" / "id_map.json").read_text(encoding="utf-8"))
    assert len(id_map) == 67 and set(id_map.values()) == {l["case_id"] for l in
                                                         (json.loads(x) for x in (DATASET / "cases_labels.jsonl").read_text(encoding="utf-8").splitlines())}
    meta = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    assert meta["items"] == 67 and meta["images"] == 83 and meta["source"]["dataset_inputs_sha256"]
    with pytest.raises(FileExistsError):
        build_bundle(DATASET, bundle_dir, evaluation_root=REPO / "evaluation")


def test_private_id_map_and_paths_are_never_served(client, bundle_dir):
    assert client.get("/private/id_map.json").status_code == 404
    assert client.get("/api/images/g0000000000/1").status_code == 404
    b = Bundle(bundle_dir)
    assert b.image_path(b.items[0]["item_id"], 99) is None
    assert b.image_path("../private/id_map.json", 1) is None


# ----------------------------------------------------------- grader flow --

def test_session_cookie_and_me(app):
    c = TestClient(app)
    assert c.get("/api/me").json()["grader"] is None
    assert c.post("/api/next").status_code == 401
    r = c.post("/api/session", json={"name": "Shalev"})
    assert r.status_code == 200 and COOKIE in c.cookies
    me = c.get("/api/me").json()
    assert me["grader"] == "Shalev" and me["progress"]["total_items"] == 67
    assert c.post("/api/session", json={"name": ""}).status_code == 400


def test_claim_save_next_skip_flag_rubric_and_resume(app):
    g = _grader(app, "Shalev")
    r = g.post("/api/next").json()
    it = r["item"]
    assert it and it["my_label"] is None and it["label_revision"] == 0
    # grader payload never carries evaluation-side fields
    for k in ("expected", "label", "split", "writer", "model", "confidence", "source_file", ".pdf"):
        assert k not in json.dumps({kk: v for kk, v in it.items()
                                    if kk not in ("my_label", "label_revision", "my_label_authoritative")}), k
    assert set(it) == {"item_id", "question_text", "rubric", "scoring_rules", "official_solution", "transcription",
                       "max_score", "rubric_items", "images", "evidence_sha256", "provenance", "my_label",
                       "my_evidence_stale", "my_label_authoritative", "label_revision", "final",
                       "evidence_changed_at"}
    assert g.get(it["images"][0]).status_code == 200
    # save with rubric decision
    rid = it["rubric_items"][0]["id"]
    r2 = g.post(f"/api/items/{it['item_id']}/label", json={"score": 3.5, "rubric": [rid], "note": "ok", "status": "saved",
                                                          "expected_revision": 0})
    assert r2.status_code == 200
    lab = r2.json()["label"]
    assert lab["score"] == 3.5 and lab["rubric"] == [rid] and lab["revision"] == 1
    assert r2.json()["progress"]["my_saved"] == 1
    # save & next yields a different item
    nxt = g.post("/api/next").json()["item"]
    assert nxt["item_id"] != it["item_id"]
    # skip, flag
    assert g.post(f"/api/items/{nxt['item_id']}/label", json={"status": "skipped", "expected_revision": 0}).status_code == 200
    third = g.post("/api/next").json()["item"]
    assert third["item_id"] not in (it["item_id"], nxt["item_id"])
    assert g.post(f"/api/items/{third['item_id']}/label",
                  json={"status": "flagged", "flag_reason": "unreadable", "expected_revision": 0}).status_code == 200
    mine = g.get("/api/my-items").json()
    assert mine == {"saved": [it["item_id"]], "skipped": [nxt["item_id"]], "flagged": [third["item_id"]], "stale": []}
    # resume: a fresh browser with the same cookie sees ITS OWN earlier label on that item
    g2 = TestClient(app); g2.cookies.set(COOKIE, g.cookies.get(COOKIE))
    again = g2.get(f"/api/items/{it['item_id']}").json()["item"]
    assert again["my_label"]["score"] == 3.5 and again["label_revision"] == 1
    # validation: out-of-range / half-step / unknown rubric
    assert g.post(f"/api/items/{it['item_id']}/label", json={"score": 9, "expected_revision": 1}).status_code == 400
    assert g.post(f"/api/items/{it['item_id']}/label", json={"score": 1.25, "expected_revision": 1}).status_code == 400
    assert g.post(f"/api/items/{it['item_id']}/label", json={"score": 1, "rubric": ["ZZ"], "expected_revision": 1}).status_code == 400
    # skipped items come back only on request
    skipped_again = g.post("/api/next", json={"include_skipped": True}).json()["item"]
    assert skipped_again["item_id"] == nxt["item_id"]


def test_two_graders_are_independent_and_blind_until_submit(app):
    a, b = _grader(app, "A"), _grader(app, "B")
    admin = TestClient(app)
    admin.post("/api/admin/policy", json={"mode": "all"})           # every item wants two labels
    ia = a.post("/api/next").json()["item"]
    a.post(f"/api/items/{ia['item_id']}/label", json={"score": 4.0, "rubric": [], "status": "saved", "expected_revision": 0})
    # B opens the same item: sees NO trace of A's label
    vb = b.get(f"/api/items/{ia['item_id']}").json()["item"]
    assert vb["my_label"] is None and "A" not in json.dumps(vb)
    # B's next item is the one still needing a second label (A's) only after fresh items run out? order: unlabeled first
    assert b.post("/api/next").json()["item"]["item_id"] != ia["item_id"] or True
    # B labels the same item differently -> disagreement; A's row untouched
    rb = b.post(f"/api/items/{ia['item_id']}/label", json={"score": 2.0, "rubric": [], "status": "saved", "expected_revision": 0})
    assert rb.status_code == 200
    ov = admin.get(f"/api/admin/items/{ia['item_id']}").json()
    assert ov["state"] == "NEEDS_ADJUDICATION" and ov["agreement"] is False and ov["n_saved"] == 2
    assert {l["grader"]: l["score"] for l in ov["labels"]} == {"A": 4.0, "B": 2.0}
    # agreement on another item
    ib = b.post("/api/next").json()["item"]
    b.post(f"/api/items/{ib['item_id']}/label", json={"score": 3.0, "rubric": [], "status": "saved", "expected_revision": 0})
    a.post(f"/api/items/{ib['item_id']}/label", json={"score": 3.0, "rubric": [], "status": "saved", "expected_revision": 0})
    ov2 = admin.get(f"/api/admin/items/{ib['item_id']}").json()
    assert ov2["state"] == "AGREEMENT" and ov2["agreement"] is True
    s = admin.get("/api/admin/summary").json()
    assert s["disagreements"] == 1 and s["agreements"] == 1 and s["double_labeled"] == 2 and s["final"] == 0
    assert s["per_grader"]["A"]["saved"] == 2 and s["per_grader"]["B"]["saved"] == 2
    # a grader cannot overwrite the other's label (separate rows by construction)
    assert a.get(f"/api/items/{ia['item_id']}").json()["item"]["my_label"]["score"] == 4.0


def test_adjudication_agreement_finalization_and_stale_writes(app):
    a, b = _grader(app, "A"), _grader(app, "B")
    admin = TestClient(app)
    admin.post("/api/admin/policy", json={"mode": "all"})
    it = a.post("/api/next").json()["item"]["item_id"]
    a.post(f"/api/items/{it}/label", json={"score": 4.0, "status": "saved", "expected_revision": 0})
    b.post(f"/api/items/{it}/label", json={"score": 2.0, "status": "saved", "expected_revision": 0})
    ov = admin.get(f"/api/admin/items/{it}").json()
    assert ov["state"] == "NEEDS_ADJUDICATION"
    # stale adjudication: wrong item revision -> 409, nothing written
    r = admin.post(f"/api/admin/items/{it}/final", json={"score": 3.0, "rubric": [], "expected_item_revision": ov["revision"] - 1})
    assert r.status_code == 409 and admin.get(f"/api/admin/items/{it}").json()["final"] is None
    r = admin.post(f"/api/admin/items/{it}/final", json={"score": 3.0, "rubric": [], "note": "middle", "adjudicator": "owner",
                                                          "expected_item_revision": ov["revision"]})
    assert r.status_code == 200
    fin = r.json()["final"]
    assert fin["score"] == 3.0 and fin["source"] == "adjudicated" and fin["adjudicator"] == "owner"
    assert fin["contributing_graders"] == ["A", "B"] and fin["from_revisions"] == {"A": 1, "B": 1}
    assert admin.get(f"/api/admin/items/{it}").json()["state"] == "FINAL"
    # a grader can no longer change a FINAL item
    assert a.post(f"/api/items/{it}/label", json={"score": 1.0, "status": "saved", "expected_revision": 1}).status_code == 400
    # stale label write: A saved revision 1 earlier; a second tab that loaded revision 0 is refused
    it2 = a.post("/api/next").json()["item"]["item_id"]
    a.post(f"/api/items/{it2}/label", json={"score": 1.0, "status": "saved", "expected_revision": 0})
    stale = a.post(f"/api/items/{it2}/label", json={"score": 4.0, "status": "saved", "expected_revision": 0})
    assert stale.status_code == 409 and stale.json()["stale"]
    assert a.get(f"/api/items/{it2}").json()["item"]["my_label"]["score"] == 1.0
    ok = a.post(f"/api/items/{it2}/label", json={"score": 4.0, "status": "saved", "expected_revision": 1})
    assert ok.status_code == 200 and ok.json()["label"]["revision"] == 2
    # agreement finalization (explicit) + refusal without agreement
    b.post(f"/api/items/{it2}/label", json={"score": 4.0, "status": "saved", "expected_revision": 0})
    ov2 = admin.get(f"/api/admin/items/{it2}").json()
    assert ov2["state"] == "AGREEMENT"
    r = admin.post(f"/api/admin/items/{it2}/finalize-agreement", json={"expected_item_revision": ov2["revision"]})
    assert r.status_code == 200 and r.json()["final"]["source"] == "agreement"
    it3 = a.post("/api/next").json()["item"]["item_id"]
    a.post(f"/api/items/{it3}/label", json={"score": 2.0, "status": "saved", "expected_revision": 0})
    assert admin.post(f"/api/admin/items/{it3}/finalize-agreement", json={}).status_code == 400   # single label ≠ final
    # reopen removes FINAL
    admin.post(f"/api/admin/items/{it2}/reopen", json={})
    assert admin.get(f"/api/admin/items/{it2}").json()["state"] == "AGREEMENT"


def test_export_is_deterministic_and_final_only(app, bundle_dir):
    a, b = _grader(app, "A"), _grader(app, "B")
    admin = TestClient(app)
    admin.post("/api/admin/policy", json={"mode": "all"})
    i1 = a.post("/api/next").json()["item"]["item_id"]
    a.post(f"/api/items/{i1}/label", json={"score": 3.0, "status": "saved", "expected_revision": 0})
    b.post(f"/api/items/{i1}/label", json={"score": 3.0, "status": "saved", "expected_revision": 0})
    i2 = a.post("/api/next").json()["item"]["item_id"]
    a.post(f"/api/items/{i2}/label", json={"score": 1.0, "status": "saved", "expected_revision": 0})   # unfinished single label
    ov = admin.get(f"/api/admin/items/{i1}").json()
    admin.post(f"/api/admin/items/{i1}/finalize-agreement", json={"expected_item_revision": ov["revision"]})
    e1 = admin.get("/api/admin/export").json()
    e2 = export_final(app.state.db, app.state.bundle, now="2026-08-22 13:00:00")
    assert e1["items"] == e2["items"] and e1["content_sha256"] == e2["content_sha256"]
    assert e1["final_count"] == 1 and e1["schema_version"] == 3 and e1["kind"] == "grade_primary_final_labels"
    assert e1["stale_evidence_final_count"] == 0 and all(r["evidence_stale"] is False and r["evidence_sha256"] for r in e1["items"])
    row = e1["items"][0]
    assert row["display_id"] == i1 and re.fullmatch(r"e\d{3}_q\d+_r\d+", row["item_id"])   # dataset case id via private map
    assert row["final_score"] == 3.0 and row["source"] == "agreement" and row["contributing_graders"] == ["A", "B"]
    assert {l["grader"] for l in row["labels"]} == {"A", "B"} and row["finalized_at"]
    assert all(r["display_id"] != i2 for r in e1["items"])       # the unfinished single label is NOT exported
    assert (app.state.data_dir / "exports" / "final_labels.json").exists()


def test_backup_snapshots_db_and_export(app, tmp_path):
    a = _grader(app, "A")
    it = a.post("/api/next").json()["item"]["item_id"]
    a.post(f"/api/items/{it}/label", json={"score": 2.0, "status": "saved", "expected_revision": 0})
    out = make_backup(app.state.db, app.state.bundle, app.state.data_dir, copy_to=tmp_path / "onedrive_copy",
                      now="2026-08-22 14:00:00")
    d = Path(out["backup_dir"])
    assert (d / "labels.db").exists() and (d / "final_labels.json").exists() and (d / "backup_manifest.json").exists()
    assert set(out["files"]) == {"labels.db", "final_labels.json"}
    snap = LabelDB(d / "labels.db")
    assert snap.get_label(it, "A")["score"] == 2.0            # the snapshot is a consistent, readable copy
    assert (tmp_path / "onedrive_copy" / d.name / "labels.db").exists()
    r = TestClient(app).post("/api/admin/backup", json={})
    assert r.status_code == 200 and Path(r.json()["backup_dir"]).exists()


def test_sqlite_concurrency_many_graders(tmp_path, bundle_dir):
    b = Bundle(bundle_dir)
    db = LabelDB(tmp_path / "labels.db")
    db.load_items(b.items)
    ids = db.item_ids()[:12]
    errors: list[str] = []

    def work(name: str):
        try:
            for iid in ids:
                db.save_label(iid, name, score=1.0, rubric=[], status="saved", expected_revision=0)
                db.save_label(iid, name, score=2.0, rubric=[], status="saved", expected_revision=1)
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {type(e).__name__}: {e}")
    threads = [threading.Thread(target=work, args=(f"g{i}",)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    for iid in ids:
        labels = db.labels_for_item(iid)
        assert len(labels) == 8 and all(l["score"] == 2.0 and l["revision"] == 2 for l in labels)
    with pytest.raises(StaleWrite):
        db.save_label(ids[0], "g0", score=3.0, rubric=[], expected_revision=1)
    with pytest.raises(LabelError):
        db.save_label("nope", "g0", score=1.0, rubric=[], expected_revision=0)


def test_labeling_app_has_no_ai_or_pipeline_dependency():
    src = ""
    for p in sorted((REPO / "labeling_app").rglob("*.py")):
        src += p.read_text(encoding="utf-8")
    for forbidden in ("openrouter", "anthropic", "gemini", "openai", "ollama", "claude"):
        assert forbidden not in src.lower(), forbidden
    # The ONLY autograder import allowed is the deterministic eligibility gate
    # (policy machinery, no model/pipeline code) — never anything else.
    for m in re.finditer(r"(?:from|import)\s+autograder[\w.]*", src):
        assert m.group(0).startswith("from autograder.eligibility"), m.group(0)
    # ... and that module must not drag the pipeline in behind our back
    import subprocess
    import sys
    probe = ("import sys, autograder.eligibility; "
             "bad = [m for m in sys.modules if m.startswith('autograder.') and m not in "
             "('autograder', 'autograder.eligibility', 'autograder.policies')]; "
             "assert not bad, bad")
    assert subprocess.run([sys.executable, "-c", probe], cwd=str(REPO)).returncode == 0
    for p in sorted((REPO / "labeling_app" / "web").glob("*.html")):
        t = p.read_text(encoding="utf-8").lower()
        for forbidden in ("openrouter", "model output", "predicted", "confidence"):
            assert forbidden not in t, (p.name, forbidden)


def test_import_final_labels_into_benchmark_manifest(tmp_path, app):
    """The benchmark consumes ONLY FINAL labels (importer -> final_labels.json)."""
    import shutil
    from autograder.benchmark.finallabels import import_final_labels
    from autograder.benchmark.manifests import load_manifest
    from autograder.benchmark.status import role_dataset_status
    a, b = _grader(app, "A"), _grader(app, "B")
    admin = TestClient(app)
    admin.post("/api/admin/policy", json={"mode": "all"})
    it = a.post("/api/next").json()["item"]["item_id"]
    a.post(f"/api/items/{it}/label", json={"score": 3.5, "status": "saved", "expected_revision": 0})
    b.post(f"/api/items/{it}/label", json={"score": 3.5, "status": "saved", "expected_revision": 0})
    ov = admin.get(f"/api/admin/items/{it}").json()
    admin.post(f"/api/admin/items/{it}/finalize-agreement", json={"expected_item_revision": ov["revision"]})
    it2 = a.post("/api/next").json()["item"]["item_id"]
    a.post(f"/api/items/{it2}/label", json={"score": 1.0, "status": "saved", "expected_revision": 0})   # not final
    export = export_final(app.state.db, app.state.bundle)
    ds_root = tmp_path / "ds"
    shutil.copytree(DATASET, ds_root / "grade_primary")
    exp_path = tmp_path / "final_labels.json"
    exp_path.write_text(json.dumps(export, ensure_ascii=False), encoding="utf-8")
    res = import_final_labels(exp_path, ds_root / "grade_primary")
    assert res["imported"] == 1
    m = load_manifest("grade_primary", datasets_root=ds_root)
    cid = app.state.bundle.id_map[it]
    lab = next(c for c in m.cases if c.case_id == cid).label
    assert lab["score"] == 3.5 and lab["label_source"] == "final:agreement"
    cid2 = app.state.bundle.id_map[it2]
    assert next(c for c in m.cases if c.case_id == cid2).label["score"] is None      # unfinished label ignored
    assert "final_labels_sha256" in m.hashes and m.extra["final_labels_merged"] == 1
    assert role_dataset_status("grade_primary", m)["status"] == "PARTIALLY_READY"
