"""Evidence completeness + evidence fingerprints of the shared labeling app.

Regression (e004_q2_r3): the dataset records TWO handwritten lines; the
bundle copies two crops in the recorded order; the grader API serves two
crops (same bytes, same order); the grader page renders every image in order
(structural contract of grader.html — no JS engine is available offline).
Generic: every grade_primary case's rendered crops == the authoritative
upstream evidence inventory (one image per recorded line for line cells; one
cell crop for exam-002 cells), in recorded order.

Fingerprints: every label records the evidence it was made against; a
rebuilt bundle that changes an item's evidence makes ONLY that item's labels
stale — re-served to their grader, never counted as fresh, never finalized
from agreement, exported as stale and refused by the benchmark importer —
while labels on unchanged items stay untouched. Legacy databases without
fingerprints are backfilled on first registration. No model/API calls.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from labeling_app.app import create_app
from labeling_app.bundle import Bundle, build_bundle, evidence_fingerprint, previous_bundle_info
from labeling_app.db import STATE_EVIDENCE_REVIEW, LabelDB, LabelError, StaleEvidence
from labeling_app.export import export_final

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "evaluation" / "model_selection" / "datasets" / "grade_primary"
HTR = REPO / "evaluation" / "htr_pilot"
real_dataset = pytest.mark.skipif(not (DATASET / "manifest.json").exists(), reason="grade_primary dataset not built")


def _rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _labels() -> dict[str, dict]:
    return {r["case_id"]: r for r in _rows(DATASET / "cases_labels.jsonl")}


def _upstream_line_records() -> dict[str, list[dict]]:
    """The authoritative per-cell line records (htr_pilot split manifests)."""
    out: dict[str, list[dict]] = {}
    for split in ("train", "val", "internal_test"):
        p = HTR / "splits" / f"{split}.json"
        if not p.exists():
            continue
        for rec in json.loads(p.read_text(encoding="utf-8")):
            cid = f"{rec['writer']}_q{rec['question']}_r{rec['row']}"
            out.setdefault(cid, []).append(rec)
    for recs in out.values():
        recs.sort(key=lambda r: r["line_index"])
    return out


def _authoritative_count(case_id: str, label_row: dict, upstream: dict[str, list[dict]]) -> int:
    """One image per recorded line (line cells); exam-002 cells are one crop."""
    if label_row["writer"] == "e002":
        return 1
    recs = upstream[case_id]
    assert recs and recs[0]["n_lines"] == len(recs)
    return recs[0]["n_lines"]


@pytest.fixture(scope="module")
def bundle_dir(tmp_path_factory):
    if not (DATASET / "manifest.json").exists():
        pytest.skip("grade_primary dataset not built")
    out = tmp_path_factory.mktemp("bundle") / "b"
    build_bundle(DATASET, out, evaluation_root=REPO / "evaluation", page_max_edge=800, now="2026-08-22 12:00:00")
    return out


# ------------------------------------------------------------- e004_q2_r3 --

@real_dataset
def test_e004_q2_r3_two_lines_end_to_end(bundle_dir, tmp_path):
    cid = "e004_q2_r3"
    # 1. upstream: the HTR pilot package records TWO lines for this cell
    up = _upstream_line_records()[cid]
    assert [r["line_index"] for r in up] == [1, 2] and up[0]["n_lines"] == 2
    up_images = [HTR / r["images"]["line"] for r in up]
    assert all(p.exists() for p in up_images)
    # 2. dataset: the label row carries both lines, in order, with provenance
    lab = _labels()[cid]
    assert lab["line_count"] == 2 and lab["evidence_kind"] == "line_crops"
    assert lab["evidence_images"] == ["hebrew_bench_v2/crops/hl_e004_q2_r3__l1.png", "htr_pilot/images/e004/q2_r3_l2.png"]
    assert [e["index"] for e in lab["evidence_lines"]] == [1, 2]
    assert lab["evidence_lines"][0]["bench_item"] == "hl_e004_q2_r3__l1"
    assert lab["evidence_lines"][1]["bench_item"] is None
    assert lab["evidence_lines"][1]["transcription_status"].startswith("no_audited_transcription:")
    assert lab["transcription_items"] == ["hl_e004_q2_r3__l1"]           # the audited OCR item is unchanged
    assert lab["transcription_complete"] is False
    assert lab["lines_without_audited_transcription"] == ["e004_q2_r3__l2"]
    assert lab["line_inventory_source"] == "evaluation/htr_pilot/splits/train.json"
    expected_bytes = [(REPO / "evaluation" / rel).read_bytes() for rel in lab["evidence_images"]]
    assert expected_bytes[0] == up_images[0].read_bytes()             # bench crop == upstream line image
    assert expected_bytes[1] == up_images[1].read_bytes()
    # 3. bundle: two crops copied in the recorded order, fingerprinted
    b = Bundle(bundle_dir)
    oid = next(k for k, v in b.id_map.items() if v == cid)
    it = b.item(oid)
    assert len(it["images"]) == 2
    assert [(bundle_dir / rel).read_bytes() for rel in it["images"]] == expected_bytes
    assert it["provenance"]["line_count"] == 2 and it["provenance"]["lines_transcribed"] == 1
    assert it["provenance"]["transcription_complete"] is False
    assert it["evidence_sha256"] == evidence_fingerprint([bundle_dir / rel for rel in it["images"]])
    assert b.private_provenance[oid]["crop_files"] == lab["evidence_images"]
    # 4. grader API: two crops, in order, same bytes
    app = create_app(data_dir=tmp_path / "data", bundle_dir=bundle_dir)
    c = TestClient(app)
    c.post("/api/session", json={"name": "R"})
    payload = c.get(f"/api/items/{oid}").json()["item"]
    assert payload["images"] == [f"/api/images/{oid}/1", f"/api/images/{oid}/2"]
    assert [c.get(u).content for u in payload["images"]] == expected_bytes
    assert payload["evidence_sha256"] == it["evidence_sha256"]
    assert payload["provenance"]["line_count"] == 2 and payload["provenance"]["transcription_complete"] is False
    assert c.get(f"/api/images/{oid}/3").status_code == 404          # exactly two
    # 5. grader UI contract: every image in `images` is rendered, in order
    _assert_grader_page_renders_all_images_in_order()


def _assert_grader_page_renders_all_images_in_order() -> None:
    html = (REPO / "labeling_app" / "web" / "grader.html").read_text(encoding="utf-8")
    script = html.split("<script>", 1)[1]
    # the render loop walks the WHOLE images array in order and appends one <img> per entry
    assert re.search(r"const urls = it\.images \|\| \[\];", script)
    assert re.search(r"urls\.forEach\(\(u, i\) =>", script)
    assert re.search(r"img\.src = u; img\.dataset\.index = String\(i\+1\)", script)
    assert "im.appendChild(img)" in script
    # nothing truncates the list to the first image
    assert "images[0]" not in script and ".slice(0, 1)" not in script and ".slice(0,1)" not in script
    assert 'id="images"' in html


# ------------------------------------------------------------- all cases --

@real_dataset
def test_every_case_renders_one_crop_per_authoritative_evidence_image(bundle_dir, tmp_path):
    labels = _labels()
    upstream = _upstream_line_records()
    b = Bundle(bundle_dir)
    app = create_app(data_dir=tmp_path / "data", bundle_dir=bundle_dir)
    c = TestClient(app)
    c.post("/api/session", json={"name": "R"})
    total = 0
    for it in b.items:
        cid = b.id_map[it["item_id"]]
        lab = labels[cid]
        want = _authoritative_count(cid, lab, upstream)
        assert len(lab["evidence_images"]) == want == lab["line_count"], cid
        payload = c.get(f"/api/items/{it['item_id']}").json()["item"]
        assert len(payload["images"]) == want, cid
        served = [c.get(u).content for u in payload["images"]]
        assert served == [(REPO / "evaluation" / rel).read_bytes() for rel in lab["evidence_images"]], cid
        total += want
    assert b.meta["images"] == total == 91


@real_dataset
def test_incomplete_transcriptions_are_flagged_not_hidden():
    labels = _labels()
    incomplete = sorted(c for c, l in labels.items() if l["transcription_complete"] is False)
    assert incomplete == ["e003_q1_r5", "e003_q2_r2", "e003_q2_r3", "e003_q2_r4", "e003_q2_r7",
                          "e004_q2_r3", "e004_q2_r5", "e006_q2_r6", "e007_q1_r1"]
    for cid in incomplete:
        lab = labels[cid]
        assert len(lab["evidence_images"]) == lab["line_count"] > len(lab["transcription_items"])
        assert lab["lines_without_audited_transcription"]


# ------------------------------------------------------------ fingerprint --

def test_fingerprint_is_deterministic_and_order_sensitive(tmp_path):
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    a.write_bytes(b"PNG-A"); b.write_bytes(b"PNG-B")
    assert evidence_fingerprint([a, b]) == evidence_fingerprint([a, b])
    assert evidence_fingerprint([a, b]) != evidence_fingerprint([b, a])
    assert evidence_fingerprint([a]) != evidence_fingerprint([a, b])
    assert re.fullmatch(r"[0-9a-f]{64}", evidence_fingerprint([]))


# ---------------------------------------------------- synthetic datasets --

def _pack() -> dict:
    return {"question_id": "9", "question_text": "t", "question_type": "matching_with_explanation",
            "max_score": 4.0, "correct_by_version": {"S1": {}}, "rubric": ["line"], "scoring_rules": [],
            "grading_policy": "choice_and_explanation_independent", "official_solution": {"S1": "sol"},
            "rubric_items": [{"id": "r1", "text": "crit", "points": None, "requires_evidence": True,
                              "excludes": [], "requires": [], "kind": "binary"}],
            "evidence_policy": "required", "score_granularity": None}


def _write_dataset(ds: Path, evidence: dict[str, list[str]]) -> Path:
    """A synthetic grade_primary dataset; ``evidence`` maps case id -> evidence
    image paths (relative to the evaluation root). Inputs are identical across
    calls so the opaque-id salt (inputs_sha256) stays the same."""
    ds.mkdir(parents=True, exist_ok=True)
    inputs, labels = [], []
    for cid, imgs in evidence.items():
        inputs.append({"case_id": cid, "pack": _pack(), "selected": None, "transcription": "הסבר", "version": None})
        labels.append({"case_id": cid, "split": "DEV", "writer": cid.split("_")[0], "question_id": "9",
                       "sub_item_id": "S1", "score": None, "rubric_met": None, "label_status": "NEEDS_OWNER_LABEL",
                       "transcription_source": "test", "transcription_items": [], "transcription_provenance": [],
                       "evidence_images": list(imgs), "evidence_kind": "line_crops", "line_count": len(imgs),
                       "line_inventory_source": "test", "evidence_lines": [], "transcription_complete": True,
                       "lines_without_audited_transcription": [], "max_score": 4.0})
    (ds / "cases_inputs.jsonl").write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in inputs), encoding="utf-8")
    (ds / "cases_labels.jsonl").write_text("".join(json.dumps(l, ensure_ascii=False) + "\n" for l in labels), encoding="utf-8")
    (ds / "manifest.json").write_text(json.dumps({"name": "synthetic grade_primary", "status": "FROZEN", "cases": len(inputs),
                                                  "inputs_sha256": "testsalt", "labels_sha256": "x"}), encoding="utf-8")
    return ds


def _fake_repo(root: Path) -> Path:
    ev = root / "evaluation"
    (ev / "hebrew_bench").mkdir(parents=True, exist_ok=True)
    (ev / "hebrew_bench" / "crops_manifest.json").write_text("[]", encoding="utf-8")
    (ev / "htr_pilot_sources.json").write_text("{}", encoding="utf-8")
    (ev / "crops").mkdir(exist_ok=True)
    for name in ("a1", "a2", "b1", "c1"):
        (ev / "crops" / f"{name}.png").write_bytes(f"PNG-{name}".encode())
    return root


def _grader(app, name: str) -> TestClient:
    c = TestClient(app)
    assert c.post("/api/session", json={"name": name}).status_code == 200
    return c


def test_evidence_change_marks_only_the_affected_labels_stale(tmp_path):
    root = _fake_repo(tmp_path)
    ds1 = _write_dataset(tmp_path / "ds1", {"e901_q1_r1": ["crops/a1.png"], "e902_q1_r1": ["crops/b1.png"],
                                           "e903_q1_r1": ["crops/c1.png"]})
    out = tmp_path / "bundle"
    data = tmp_path / "data"
    build_bundle(ds1, out, evaluation_root=root / "evaluation", repo_root=root, now="2026-08-22 12:00:00")
    b1 = Bundle(out)
    ids = {v: k for k, v in b1.id_map.items()}
    old_fp = b1.fingerprints[ids["e901_q1_r1"]]
    # --- round 1: friends grade against bundle v1 --------------------------
    app1 = create_app(data_dir=data, bundle_dir=out)
    a, b = _grader(app1, "A"), _grader(app1, "B")
    admin = TestClient(app1)
    admin.post("/api/admin/policy", json={"mode": "all"})
    for g, score in ((a, 3.0), (b, 3.0)):
        for cid in ("e901_q1_r1", "e902_q1_r1"):
            it = g.get(f"/api/items/{ids[cid]}").json()["item"]
            r = g.post(f"/api/items/{ids[cid]}/label", json={"score": score, "status": "saved", "expected_revision": 0,
                                                            "evidence_sha256": it["evidence_sha256"]})
            assert r.status_code == 200 and r.json()["label"]["evidence_sha256"] == it["evidence_sha256"]
    b.post(f"/api/items/{ids['e903_q1_r1']}/label", json={"score": 1.0, "status": "saved", "expected_revision": 0})
    for cid in ("e901_q1_r1", "e902_q1_r1"):
        ov = admin.get(f"/api/admin/items/{ids[cid]}").json()
        assert ov["state"] == "AGREEMENT"
        assert admin.post(f"/api/admin/items/{ids[cid]}/finalize-agreement",
                          json={"expected_item_revision": ov["revision"]}).status_code == 200
    app1.state.db.close()
    # --- the dataset gains e901's missing second line; nothing else changes --
    ds2 = _write_dataset(tmp_path / "ds2", {"e901_q1_r1": ["crops/a1.png", "crops/a2.png"],
                                           "e902_q1_r1": ["crops/b1.png"], "e903_q1_r1": ["crops/c1.png"]})
    with pytest.raises(FileExistsError):
        build_bundle(ds2, out, evaluation_root=root / "evaluation", repo_root=root)
    meta2 = build_bundle(ds2, out, evaluation_root=root / "evaluation", repo_root=root, now="2026-08-22 13:00:00",
                         replace=True)
    prev_dir = Path(meta2["replaced"]["previous_dir"])
    assert prev_dir.exists() and (prev_dir / "items.json").exists()          # what graders saw is kept
    assert meta2["replaced"]["previous_items_sha256"] == b1.meta["items_sha256"]
    b2 = Bundle(out)
    assert b2.id_map == b1.id_map                                               # opaque ids stable
    assert len(b2.item(ids["e901_q1_r1"])["images"]) == 2
    assert b2.fingerprints[ids["e901_q1_r1"]] != old_fp
    assert b2.fingerprints[ids["e902_q1_r1"]] == b1.fingerprints[ids["e902_q1_r1"]]
    # --- round 2: the SAME labels.db against bundle v2 ------------------------
    app2 = create_app(data_dir=data, bundle_dir=out)
    sync = app2.state.evidence_sync
    assert [c["item_id"] for c in sync["changed"]] == [ids["e901_q1_r1"]]
    assert sync["changed"][0]["graders"] == ["A", "B"] and sync["changed"][0]["final_present"]
    rep = sync["report"]
    assert rep["labels_total"] == rep["labels_preserved"] == 5                  # nothing deleted
    assert rep["labels_stale"] == 2 and rep["labels_fresh"] == 3 and rep["labels_unknown_evidence"] == 0
    assert {(r["item_id"], r["grader"]) for r in rep["stale_labels"]} == {(ids["e901_q1_r1"], "A"), (ids["e901_q1_r1"], "B")}
    assert rep["finals_stale"] == 1 and rep["stale_finals"][0]["item_id"] == ids["e901_q1_r1"]
    admin2 = TestClient(app2)
    ev = admin2.get("/api/admin/evidence").json()
    assert ev["affected_case_ids"] == ["e901_q1_r1"]
    assert admin2.get(f"/api/admin/items/{ids['e901_q1_r1']}").json()["state"] == STATE_EVIDENCE_REVIEW
    assert admin2.get(f"/api/admin/items/{ids['e902_q1_r1']}").json()["state"] == "FINAL"     # untouched
    assert admin2.get(f"/api/admin/items/{ids['e903_q1_r1']}").json()["state"] == "LABELED"   # untouched
    summ = admin2.get("/api/admin/summary").json()
    assert summ["stale_labels"] == 2 and summ["stale_finals"] == 1 and summ["needs_evidence_review"] == 1
    assert summ["per_grader"]["A"]["stale"] == 1 and summ["per_grader"]["B"]["stale"] == 1
    # the stale FINAL is exported for provenance but flagged, and the importer refuses it
    exp = export_final(app2.state.db, b2, now="2026-08-22 14:00:00")
    rows = {r["item_id"]: r for r in exp["items"]}
    assert rows["e901_q1_r1"]["evidence_stale"] is True and rows["e902_q1_r1"]["evidence_stale"] is False
    assert exp["stale_evidence_final_count"] == 1
    from autograder.benchmark.finallabels import import_final_labels
    (tmp_path / "export.json").write_text(json.dumps(exp, ensure_ascii=False), encoding="utf-8")
    res = import_final_labels(tmp_path / "export.json", ds2)
    assert res["imported"] == 1 and res["ignored_stale_evidence"] == ["e901_q1_r1"]
    # a FINAL made on superseded evidence blocks grading until the admin reopens it
    a2, b2c = _grader(app2, "A"), _grader(app2, "B")
    first = a2.post("/api/next").json()["item"]
    assert first["item_id"] == ids["e903_q1_r1"]                              # e901 is FINAL (stale) -> admin; e903 still wants A
    admin2.post(f"/api/admin/items/{ids['e901_q1_r1']}/reopen", json={})
    # re-review: the grader's OWN stale label comes back first, flagged as stale
    nxt = a2.post("/api/next").json()
    assert nxt["item"]["item_id"] == ids["e901_q1_r1"] and nxt["item"]["my_evidence_stale"] is True
    assert len(nxt["item"]["images"]) == 2 and nxt["progress"]["my_stale"] == 1
    assert nxt["progress"]["remaining_for_me"] == 2                            # e901 (re-review) + e903 (second label)
    # a save that echoes the OLD evidence fingerprint is refused (the page is stale)
    r = a2.post(f"/api/items/{ids['e901_q1_r1']}/label", json={"score": 2.0, "status": "saved", "expected_revision": 1,
                                                            "evidence_sha256": old_fp})
    assert r.status_code == 409 and r.json()["stale_evidence"] is True
    # agreement cannot be finalized while a contributing label is stale
    assert admin2.post(f"/api/admin/items/{ids['e901_q1_r1']}/finalize-agreement", json={}).status_code == 400
    # re-saving against the corrected evidence makes the label fresh again (history kept: revision 2)
    r = a2.post(f"/api/items/{ids['e901_q1_r1']}/label", json={"score": 2.0, "status": "saved", "expected_revision": 1,
                                                            "evidence_sha256": b2.fingerprints[ids["e901_q1_r1"]]})
    assert r.status_code == 200 and r.json()["label"]["revision"] == 2 and r.json()["progress"]["my_stale"] == 0
    ov = admin2.get(f"/api/admin/items/{ids['e901_q1_r1']}").json()
    assert ov["state"] == STATE_EVIDENCE_REVIEW and ov["stale_graders"] == ["B"]   # B still has to re-review
    nb = b2c.post("/api/next").json()["item"]
    assert nb["item_id"] == ids["e901_q1_r1"] and nb["my_evidence_stale"] is True
    r = b2c.post(f"/api/items/{ids['e901_q1_r1']}/label", json={"score": 2.0, "status": "saved", "expected_revision": 1,
                                                             "evidence_sha256": nb["evidence_sha256"]})
    assert r.status_code == 200
    ov = admin2.get(f"/api/admin/items/{ids['e901_q1_r1']}").json()
    assert ov["state"] == "AGREEMENT" and ov["n_stale"] == 0
    assert admin2.post(f"/api/admin/items/{ids['e901_q1_r1']}/finalize-agreement",
                       json={"expected_item_revision": ov["revision"]}).status_code == 200
    exp2 = export_final(app2.state.db, b2, now="2026-08-22 15:00:00")
    assert all(r["evidence_stale"] is False for r in exp2["items"]) and exp2["final_count"] == 2
    # the audit trail names the change
    kinds = [e["action"] for e in app2.state.db.events(500)]
    assert "evidence_changed" in kinds and "evidence_registered" in kinds
    app2.state.db.close()


def test_legacy_database_without_fingerprints_is_backfilled_then_detects_change(tmp_path):
    """A labels.db created before schema 2 (no evidence columns, labels made
    against the bundle that is still in place): migration adds the columns,
    the first registration records that bundle's fingerprints on every label
    (explicitly logged), and only a LATER change makes a label stale."""
    p = tmp_path / "labels.db"
    con = sqlite3.connect(str(p))
    con.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO meta VALUES ('schema_version', '1');
        CREATE TABLE items (item_id TEXT PRIMARY KEY, max_score REAL NOT NULL, rubric_ids TEXT NOT NULL DEFAULT '[]',
            wanted_labels INTEGER NOT NULL DEFAULT 1, revision INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL);
        CREATE TABLE graders (name TEXT PRIMARY KEY, created_at TEXT NOT NULL, last_seen TEXT NOT NULL);
        CREATE TABLE claims (item_id TEXT NOT NULL, grader TEXT NOT NULL, claimed_at TEXT NOT NULL, expires_at REAL NOT NULL,
            PRIMARY KEY (item_id, grader));
        CREATE TABLE labels (item_id TEXT NOT NULL, grader TEXT NOT NULL, score REAL, rubric TEXT NOT NULL DEFAULT '[]',
            note TEXT NOT NULL DEFAULT '', status TEXT NOT NULL, flag_reason TEXT NOT NULL DEFAULT '',
            revision INTEGER NOT NULL DEFAULT 1, created_at TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY (item_id, grader));
        CREATE TABLE final_labels (item_id TEXT PRIMARY KEY, score REAL NOT NULL, rubric TEXT NOT NULL DEFAULT '[]',
            note TEXT NOT NULL DEFAULT '', source TEXT NOT NULL, adjudicator TEXT NOT NULL DEFAULT '',
            contributing_graders TEXT NOT NULL DEFAULT '[]', from_revisions TEXT NOT NULL DEFAULT '{}',
            finalized_at TEXT NOT NULL, schema_version INTEGER NOT NULL);
        CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, grader TEXT NOT NULL,
            action TEXT NOT NULL, item_id TEXT, revision INTEGER, detail TEXT NOT NULL DEFAULT '{}');
        INSERT INTO items(item_id, max_score, rubric_ids, wanted_labels, revision, created_at) VALUES ('gaaaaaaaaaa', 4, '[]', 1, 1, 't');
        INSERT INTO items(item_id, max_score, rubric_ids, wanted_labels, revision, created_at) VALUES ('gbbbbbbbbbb', 4, '[]', 1, 1, 't');
        INSERT INTO labels(item_id, grader, score, status, revision, created_at, updated_at) VALUES ('gaaaaaaaaaa', 'Friend', 3, 'saved', 1, 't', 't');
        INSERT INTO labels(item_id, grader, score, status, revision, created_at, updated_at) VALUES ('gbbbbbbbbbb', 'Friend', 2, 'saved', 1, 't', 't');
    """)
    con.commit(); con.close()
    db = LabelDB(p)
    first = db.sync_evidence({"gaaaaaaaaaa": "fp-a-v1", "gbbbbbbbbbb": "fp-b-v1"})
    assert sorted(first["registered"]) == ["gaaaaaaaaaa", "gbbbbbbbbbb"] and first["backfilled_labels"] == 2
    assert first["report"]["labels_stale"] == 0 and first["report"]["labels_unknown_evidence"] == 0
    assert db.get_label("gaaaaaaaaaa", "Friend")["evidence_sha256"] == "fp-a-v1"
    second = db.sync_evidence({"gaaaaaaaaaa": "fp-a-v2", "gbbbbbbbbbb": "fp-b-v1"})     # only a's evidence changed
    assert [c["item_id"] for c in second["changed"]] == ["gaaaaaaaaaa"]
    rep = second["report"]
    assert rep["labels_preserved"] == 2 and rep["labels_stale"] == 1 and rep["labels_fresh"] == 1
    assert rep["stale_labels"][0]["grader"] == "Friend" and rep["stale_labels"][0]["item_id"] == "gaaaaaaaaaa"
    assert db.overview("gaaaaaaaaaa")["state"] == STATE_EVIDENCE_REVIEW
    assert db.overview("gbbbbbbbbbb")["state"] == "LABELED"
    assert db.my_items("Friend")["stale"] == ["gaaaaaaaaaa"]
    assert db.claim_next("Friend") == "gaaaaaaaaaa"                           # re-served to its grader
    with pytest.raises(StaleEvidence):
        db.save_label("gaaaaaaaaaa", "Friend", score=3.0, rubric=[], expected_revision=1, client_evidence_sha256="fp-a-v1")
    with pytest.raises(LabelError):
        db.finalize_agreement("gaaaaaaaaaa")
    lab = db.save_label("gaaaaaaaaaa", "Friend", score=3.0, rubric=[], expected_revision=1, client_evidence_sha256="fp-a-v2")
    assert lab["evidence_sha256"] == "fp-a-v2" and lab["revision"] == 2 and not lab["evidence_stale"]
    assert db.evidence_report()["labels_stale"] == 0
    db.close()


def test_cli_replace_registers_old_bundle_before_rebuilding(tmp_path, capsys):
    """`build-bundle --replace`: the old bundle's fingerprints are registered on
    the labels that exist (what they were made against), the old bundle is kept
    aside, and the report names exactly the changed case."""
    from labeling_app.cli import main
    real = REPO / "evaluation"
    l1, l2 = "hebrew_bench_v2/crops/hl_e004_q2_r1__l1.png", "hebrew_bench_v2/crops/hl_e004_q2_r1__l2.png"
    if not (real / l1).exists() or not (real / l2).exists():
        pytest.skip("frozen OCR bench crops not present")
    ds1 = _write_dataset(tmp_path / "ds1", {"e901_q1_r1": [l1], "e902_q1_r1": [l1]})
    ds2 = _write_dataset(tmp_path / "ds2", {"e901_q1_r1": [l1, l2], "e902_q1_r1": [l1]})
    out, data = tmp_path / "bundle", tmp_path / "data"
    assert main(["build-bundle", "--dataset", str(ds1), "--out", str(out), "--data-dir", str(data)]) == 0
    b1 = Bundle(out)
    ids = {v: k for k, v in b1.id_map.items()}
    db = LabelDB(data / "labels.db")                                            # the friend's labels (no fingerprints yet)
    db.load_items(b1.items)
    db.save_label(ids["e901_q1_r1"], "Friend", score=3.0, rubric=[], expected_revision=0)
    db.save_label(ids["e902_q1_r1"], "Friend", score=1.0, rubric=[], expected_revision=0)
    db.close()
    assert main(["build-bundle", "--dataset", str(ds2), "--out", str(out), "--data-dir", str(data)]) == 2   # refuses silently replacing
    assert main(["build-bundle", "--dataset", str(ds2), "--out", str(out), "--data-dir", str(data), "--replace"]) == 0
    printed = capsys.readouterr()
    assert "labels preserved : 2" in printed.err and "labels stale     : 1" in printed.err
    assert "STALE label  case e901_q1_r1  grader Friend" in printed.err
    assert previous_bundle_info(out)["salt"] == "testsalt"
    assert any(p.name.startswith("bundle.previous-") for p in tmp_path.iterdir())
    assert main(["evidence-report", "--data-dir", str(data), "--bundle", str(out)]) == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["labels_stale"] == 1 and rep["affected_case_ids"] == ["e901_q1_r1"]
    assert rep["stale_labels"][0]["case_id"] == "e901_q1_r1"
