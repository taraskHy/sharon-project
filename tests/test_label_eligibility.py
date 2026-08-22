"""Explanation-label eligibility — ONE source of truth (autograder.eligibility)
wrapping the production policy gate.

Covers: the full policy x MC matrix (correct / wrong zero-rule / wrong_choice_zero /
rescue / independent / process rule / ambiguous-never-zero / choice_only / absent MC /
undeterminable accepted set), the bundle gate + accounting (no silent loss), server-side
rejection of ineligible submissions (fresh AND stale bundles), workload counts,
obsolete-label surfacing, the export/import human-vs-deterministic distinction, and
GRADE_PRIMARY benchmark integrity on the real frozen dataset. Offline; no model calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from autograder.eligibility import (
    ELIGIBILITY_REASONS,
    decide_explanation_label_eligibility,
    eligibility_counts,
    eligibility_for_case,
    split_cases,
)
from autograder.policies import MCResolution, POLICIES
from labeling_app.app import create_app
from labeling_app.bundle import Bundle, build_bundle
from labeling_app.db import LabelDB, LabelError
from labeling_app.export import export_final

REPO = Path(__file__).resolve().parents[1]
REAL_DATASET = REPO / "evaluation" / "model_selection" / "datasets" / "grade_primary"


# ------------------------------------------------------------ test fixtures --

def _pack(policy: str, accepted_by_version: dict, *, rule: str | None = None, max_score: float = 4.0) -> dict:
    pack = {"question_id": "9", "question_text": "t", "question_type": "matching_with_explanation",
            "max_score": max_score, "correct_by_version": {"S1": accepted_by_version},
            "rubric": ["line"], "scoring_rules": [], "grading_policy": policy,
            "official_solution": {"S1": "sol"},
            "rubric_items": [{"id": "r1", "text": "crit", "points": None, "requires_evidence": True,
                              "excludes": [], "requires": [], "kind": "binary"}],
            "evidence_policy": "required", "score_granularity": None}
    if rule:
        pack["wrong_answer_rule"] = rule
    return pack


def _case(cid: str, policy: str, *, selected=None, version=None, accepted=None, rule=None,
          mc_state=None, mc_confidence=None, max_score: float = 4.0) -> tuple[dict, dict]:
    inp = {"case_id": cid, "pack": _pack(policy, accepted or {}, rule=rule, max_score=max_score),
           "selected": selected, "transcription": "הסבר", "version": version}
    if mc_state is not None:
        inp["mc_state"] = mc_state
    if mc_confidence is not None:
        inp["mc_confidence"] = mc_confidence
    lab = {"case_id": cid, "split": "DEV", "writer": cid.split("_")[0], "question_id": "9",
           "sub_item_id": "S1", "score": None, "rubric_met": None, "label_status": "NEEDS_OWNER_LABEL",
           "transcription_source": "audited human reference (reference_for_scoring mode=final)",
           "transcription_items": [], "transcription_provenance": ["audited_confirmed"],
           "evidence_images": [], "max_score": max_score}
    return inp, lab


def _write_dataset(root: Path, cases: list[tuple[dict, dict]]) -> Path:
    ds = root / "dataset"
    ds.mkdir(parents=True, exist_ok=True)
    (ds / "manifest.json").write_text(json.dumps({"name": "synthetic grade_primary", "status": "FROZEN",
                                                  "cases": len(cases), "inputs_sha256": "testsalt"}),
                                      encoding="utf-8")
    (ds / "cases_inputs.jsonl").write_text(
        "".join(json.dumps(i, ensure_ascii=False) + "\n" for i, _ in cases), encoding="utf-8")
    (ds / "cases_labels.jsonl").write_text(
        "".join(json.dumps(l, ensure_ascii=False) + "\n" for _, l in cases), encoding="utf-8")
    return ds


def _fake_repo(root: Path) -> Path:
    (root / "evaluation" / "hebrew_bench").mkdir(parents=True, exist_ok=True)
    (root / "evaluation" / "hebrew_bench" / "crops_manifest.json").write_text("[]", encoding="utf-8")
    (root / "evaluation" / "htr_pilot_sources.json").write_text("{}", encoding="utf-8")
    return root


#: the standard 4-case synthetic dataset: 3 human-labelable + 1 deterministic zero
def _four_cases() -> list[tuple[dict, dict]]:
    return [
        _case("e901_q1_r1", "choice_and_explanation_independent"),                        # absent MC
        _case("e902_q1_r1", "wrong_choice_zero", selected="B", version="1",
              accepted={"1": ["A"]}),                                                     # deterministic 0
        _case("e903_q1_r1", "explanation_can_rescue_wrong_choice", selected="B", version="1",
              accepted={"1": ["A"]}),                                                     # rescue
        _case("e904_q1_r1", "wrong_choice_zero", selected="B", version="1", accepted={"1": ["A"]},
              mc_state="multiple_marks", mc_confidence=0.0),                              # unresolved
    ]


@pytest.fixture
def synth(tmp_path):
    ds = _write_dataset(tmp_path, _four_cases())
    root = _fake_repo(tmp_path)
    out = tmp_path / "bundle"
    meta = build_bundle(ds, out, evaluation_root=root / "evaluation", repo_root=root,
                        now="2026-08-22 12:00:00")
    return ds, out, meta


def _elig(policy, **kw):
    inp, lab = _case("e999_q1_r1", policy, **kw)
    return eligibility_for_case(inp, lab)


# --------------------------------------------------- decision matrix (unit) --

def test_correct_mc_under_required_policy_is_labelable():
    e = _elig("explanation_required_if_correct", selected="A", version="1", accepted={"1": ["A"]})
    assert e.eligible_for_human_label and e.deterministic_score is None
    assert e.reason == "mc_correct_explanation_required" and e.mc_state == "correct"
    assert e.selected_option == "A" and e.accepted_options == ("A",)


def test_wrong_mc_zero_rule_is_deterministic_zero():
    for rule in (None, "zero", "selection"):        # production default is "zero"
        e = _elig("explanation_required_if_correct", selected="B", version="1",
                  accepted={"1": ["A"]}, rule=rule)
        assert not e.eligible_for_human_label
        assert e.deterministic_score == 0.0
        assert e.reason == "wrong_mc_deterministic_zero" and e.mc_state == "wrong"


def test_wrong_choice_zero_wrong_mc_is_deterministic_zero():
    e = _elig("wrong_choice_zero", selected="B", version="1", accepted={"1": ["A"]})
    assert not e.eligible_for_human_label and e.deterministic_score == 0.0
    assert e.reason == "wrong_mc_deterministic_zero"


def test_wrong_choice_zero_correct_mc_needs_a_human():
    e = _elig("wrong_choice_zero", selected="A", version="1", accepted={"1": ["A"]})
    assert e.eligible_for_human_label and e.reason == "mc_correct_explanation_required"


def test_rescue_policy_keeps_wrong_mc_labelable():
    e = _elig("explanation_can_rescue_wrong_choice", selected="B", version="1", accepted={"1": ["A"]})
    assert e.eligible_for_human_label and e.deterministic_score is None
    assert e.reason == "wrong_mc_rescue_allowed" and e.mc_state == "wrong"


def test_independent_policy_always_labelable():
    for kw in ({}, {"selected": "B", "version": "1", "accepted": {"1": ["A"]}},
               {"selected": "A", "version": "1", "accepted": {"1": ["A"]}}):
        e = _elig("choice_and_explanation_independent", **kw)
        assert e.eligible_for_human_label and e.reason == "independent_explanation"


def test_process_rule_keeps_wrong_mc_labelable():
    e = _elig("explanation_required_if_correct", selected="B", version="1",
              accepted={"1": ["A"]}, rule="process")
    assert e.eligible_for_human_label and e.deterministic_score is None
    assert e.reason == "wrong_mc_process_rule" and e.mc_state == "wrong"


def test_ambiguous_mc_is_never_deterministic_zero():
    unresolved = MCResolution(selected="B", state="multiple_marks", confidence=0.0,
                              source="dataset", candidates=["A", "B"])
    for policy in POLICIES:
        e = decide_explanation_label_eligibility(policy=policy, mc=unresolved, accepted=["A"],
                                                 points_selection=4.0, points_max=4.0)
        assert e.deterministic_score is None, policy
        if policy == "choice_only":
            assert not e.eligible_for_human_label and e.reason == "choice_only"
        else:
            assert e.eligible_for_human_label and e.reason in ("mc_unresolved", "independent_explanation")
    low_conf = MCResolution(selected="B", state="single_mark", confidence=0.5, source="dataset")
    e = decide_explanation_label_eligibility(policy="wrong_choice_zero", mc=low_conf, accepted=["A"],
                                             points_selection=4.0, points_max=4.0)
    assert e.eligible_for_human_label and e.reason == "mc_unresolved" and e.deterministic_score is None


def test_absent_mc_is_never_deterministic():
    # grade_primary shape: selected=None, version=None, no observation fields
    for policy in ("wrong_choice_zero", "explanation_required_if_correct",
                   "explanation_can_rescue_wrong_choice"):
        e = _elig(policy, accepted={"1": ["A"]})
        assert e.eligible_for_human_label and e.mc_state == "absent" and e.reason == "mc_absent"
    e = _elig("choice_and_explanation_independent", accepted={"1": ["A"]})
    assert e.eligible_for_human_label and e.mc_state == "absent" and e.reason == "independent_explanation"


def test_undeterminable_accepted_set_never_wrong():
    # version unknown and the key has no "default" entry -> MC cannot be judged
    e = _elig("wrong_choice_zero", selected="B", version=None, accepted={"1": ["A"]})
    assert e.eligible_for_human_label and e.reason == "mc_unresolved" and e.deterministic_score is None
    # ... but a "default" entry resolves it (production _accepted semantics)
    e = _elig("wrong_choice_zero", selected="B", version=None, accepted={"default": ["A"]})
    assert not e.eligible_for_human_label and e.deterministic_score == 0.0


def test_blank_observation_is_a_real_wrong_under_zero_policy():
    e = _elig("wrong_choice_zero", mc_state="blank", mc_confidence=1.0, version="1",
              accepted={"1": ["A"]})
    assert not e.eligible_for_human_label and e.deterministic_score == 0.0


def test_choice_only_never_enters_the_explanation_queue():
    e = _elig("choice_only", selected="A", version="1", accepted={"1": ["A"]})
    assert not e.eligible_for_human_label and e.reason == "choice_only"
    assert e.deterministic_score == 4.0                       # local MC score, not a human label
    e = _elig("choice_only")
    assert not e.eligible_for_human_label and e.deterministic_score is None


def test_hebrew_letters_normalize_like_production():
    e = _elig("wrong_choice_zero", selected="א", version="1", accepted={"1": ["A"]})
    assert e.eligible_for_human_label and e.mc_state == "correct"


def test_junk_selected_is_unclear_never_a_confident_wrong():
    # production maps answered-but-normalizes-to-nothing to state "unclear"
    for junk in (".", "'", " )( ", ""):
        e = _elig("wrong_choice_zero", selected=junk, version="1", accepted={"1": ["A"]})
        assert e.eligible_for_human_label, junk
        assert e.reason == "mc_unresolved" and e.deterministic_score is None, junk


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        decide_explanation_label_eligibility(policy="bogus", mc=None, accepted=[],
                                             points_selection=1.0, points_max=1.0)


def test_wrong_answer_rule_is_a_real_pack_field():
    # the rule travels on QuestionGradingPack -> dataset pack dicts (emitted
    # only when set, so existing dataset rebuilds stay byte-identical)
    from autograder.gradingpack import QuestionGradingPack
    pack = QuestionGradingPack(question_id="9", question_text="t", question_type="matching_with_explanation",
                               max_score=4.0, correct_by_version={"S1": {"1": ["A"]}}, rubric=[],
                               scoring_rules=[], grading_policy="explanation_required_if_correct",
                               official_solution={"S1": "sol"}, wrong_answer_rule="process")
    assert pack.wrong_answer_rule == "process"
    assert QuestionGradingPack(question_id="9", question_text="t", question_type="x", max_score=1.0,
                               correct_by_version={}, rubric=[], scoring_rules=[],
                               grading_policy="choice_only", official_solution={}).wrong_answer_rule is None


def test_builder_routing_accounts_for_every_case():
    from autograder.benchmark.datasets import route_case_by_eligibility
    cases = _four_cases()
    kept, exited = [], []
    for inp, lab in cases:
        elig, record = route_case_by_eligibility(inp, lab)
        assert (record is None) == elig.eligible_for_human_label      # exactly one outcome per case
        (kept if record is None else exited).append((inp["case_id"], elig, record))
    assert [c for c, _, _ in kept] == ["e901_q1_r1", "e903_q1_r1", "e904_q1_r1"]
    assert len(kept) + len(exited) == len(cases)                       # no silent loss
    (_, elig, record), = exited
    assert record == {"case_id": "e902_q1_r1", "final_score": 0.0, "source": "deterministic_mc_wrong",
                      "policy": "wrong_choice_zero", "mc_correct": False, "mc_state": "wrong",
                      "reason": "wrong_mc_deterministic_zero", "selected_option": "B",
                      "accepted_options": ["A"], "split": "DEV", "writer": "e902",
                      "question_id": "9", "sub_item_id": "S1", "max_score": 4.0}
    # choice_only routes as policy_no_explanation_component
    inp, lab = _case("e905_q1_r1", "choice_only", selected="A", version="1", accepted={"1": ["A"]})
    _, record = route_case_by_eligibility(inp, lab)
    assert record["source"] == "policy_no_explanation_component" and record["final_score"] == 4.0


def test_counts_account_for_every_case():
    inputs = [i for i, _ in _four_cases()]
    labels = {l["case_id"]: l for _, l in _four_cases()}
    labelable, decided = split_cases(inputs, labels)
    assert [r["case_id"] for r, _ in labelable] == ["e901_q1_r1", "e903_q1_r1", "e904_q1_r1"]
    assert [r["case_id"] for r, _ in decided] == ["e902_q1_r1"]
    c = eligibility_counts([e for _, e in labelable + decided])
    assert c["source_cases"] == 4 and c["human_labelable"] == 3 and c["deterministic_zero"] == 1
    assert c["mc_unresolved"] == 1 and c["wrong_mc_rescue_allowed"] == 1
    assert c["source_cases"] == c["human_labelable"] + c["deterministic_zero"] + c["excluded_choice_only"]
    assert all(e.reason in ELIGIBILITY_REASONS for _, e in labelable + decided)


# ------------------------------------------------------------- bundle gate --

def test_bundle_excludes_policy_decided_cases(synth):
    ds, out, meta = synth
    items = json.loads((out / "items.json").read_text(encoding="utf-8"))
    id_map = json.loads((out / "private" / "id_map.json").read_text(encoding="utf-8"))
    assert len(items) == 3 and set(id_map.values()) == {"e901_q1_r1", "e903_q1_r1", "e904_q1_r1"}
    assert all(i["eligible_for_human_label"] is True for i in items)
    assert "e902" not in (out / "items.json").read_text(encoding="utf-8")
    excluded = json.loads((out / "private" / "excluded.json").read_text(encoding="utf-8"))
    assert [e["case_id"] for e in excluded] == ["e902_q1_r1"]
    assert excluded[0]["reason"] == "wrong_mc_deterministic_zero"
    assert excluded[0]["deterministic_score"] == 0.0
    el = meta["eligibility"]
    assert el["source_cases"] == 4 and el["human_labelable"] == 3 and el["deterministic_zero"] == 1
    assert meta["items"] == 3


def test_fresh_server_never_serves_a_policy_decided_case(synth, tmp_path):
    ds, out, _ = synth
    app = create_app(data_dir=tmp_path / "data", bundle_dir=out, dataset_dir=ds)
    c = TestClient(app)
    c.post("/api/session", json={"name": "friend"})
    seen = []
    for _ in range(10):
        r = c.post("/api/next", json={})
        if r.json().get("done"):
            break
        item = r.json()["item"]
        seen.append(item["item_id"])
        assert "eligibility_reason" not in item          # graders never see policy context
        r2 = c.post(f"/api/items/{item['item_id']}/label",
                    json={"score": 1.0, "status": "saved", "expected_revision": 0})
        assert r2.status_code == 200
    assert len(seen) == 3                                # the deterministic-zero case is no one's workload
    prog = c.get("/api/me").json()["progress"]
    assert prog["total_items"] == 3 and prog["remaining_for_me"] == 0


def _make_stale(out: Path, case_id: str, oid: str = "gdeadbeef00") -> None:
    """Simulate a bundle built BEFORE eligibility filtering: inject an item for
    ``case_id`` with no eligibility flags."""
    items = json.loads((out / "items.json").read_text(encoding="utf-8"))
    stale = dict(items[0])
    stale["item_id"] = oid
    stale.pop("eligible_for_human_label", None)
    stale.pop("eligibility_reason", None)
    items.append(stale)
    (out / "items.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    id_map = json.loads((out / "private" / "id_map.json").read_text(encoding="utf-8"))
    id_map[oid] = case_id
    (out / "private" / "id_map.json").write_text(json.dumps(id_map, ensure_ascii=False), encoding="utf-8")


def test_stale_bundle_submission_rejected_server_side(synth, tmp_path):
    ds, out, _ = synth
    _make_stale(out, "e902_q1_r1")
    app = create_app(data_dir=tmp_path / "data", bundle_dir=out, dataset_dir=ds)
    assert app.state.bundle.ineligible_item_ids() == ["gdeadbeef00"]
    c = TestClient(app)
    c.post("/api/session", json={"name": "friend"})
    # direct submission against the ineligible item fails, even though the
    # stale bundle still contains it
    r = c.post("/api/items/gdeadbeef00/label",
               json={"score": 0.0, "status": "saved", "expected_revision": 0})
    assert r.status_code == 400 and "not eligible" in r.json()["error"]
    # Save & Next never hands it out
    served = []
    for _ in range(10):
        r = c.post("/api/next", json={})
        if r.json().get("done"):
            break
        iid = r.json()["item"]["item_id"]
        served.append(iid)
        c.post(f"/api/items/{iid}/label", json={"score": 0.5, "status": "saved", "expected_revision": 0})
    assert "gdeadbeef00" not in served and len(served) == 3
    # admin cannot set a human FINAL on it either — the policy score is authoritative
    r = c.post("/api/admin/items/gdeadbeef00/final", json={"score": 0, "expected_item_revision": 0})
    assert r.status_code == 400 and "not eligible" in r.json()["error"]
    summary = c.get("/api/admin/summary").json()
    assert summary["ineligible_items"] == 1 and summary["eligible_items"] == 3
    assert summary["ineligible_item_ids"] == ["gdeadbeef00"]
    # workload accounting: 4 rows in the DB, but only the 3 eligible ones count
    prog = c.get("/api/me").json()["progress"]
    assert prog["total_items"] == 3 and prog["remaining_for_me"] == 0 and prog["my_saved"] == 3
    assert summary["total_items"] == 4 and summary["labels_completed"] == 3
    # double labeling: the second grader is owed exactly the 3 eligible items
    assert c.post("/api/admin/policy", json={"mode": "all"}).status_code == 200
    c2 = TestClient(app)
    c2.post("/api/session", json={"name": "friend2"})
    served2 = []
    for _ in range(10):
        r = c2.post("/api/next", json={})
        if r.json().get("done"):
            break
        iid = r.json()["item"]["item_id"]
        served2.append(iid)
        c2.post(f"/api/items/{iid}/label", json={"score": 0.5, "status": "saved", "expected_revision": 0})
    assert "gdeadbeef00" not in served2 and len(served2) == 3
    summary = c.get("/api/admin/summary").json()
    assert summary["double_labeled"] == 3
    prog2 = c2.get("/api/me").json()["progress"]
    assert prog2["total_items"] == 3 and prog2["remaining_for_me"] == 0


def test_readonly_cli_open_never_erases_enforcement(synth, tmp_path):
    """A status/export/backup style open with UNKNOWN eligibility (stale bundle,
    dataset unusable) must never flip an ineligible item back to eligible."""
    ds, out, _ = synth
    _make_stale(out, "e902_q1_r1")
    data_dir = tmp_path / "data"
    app = create_app(data_dir=data_dir, bundle_dir=out, dataset_dir=ds)   # marks gdeadbeef00 ineligible
    db: LabelDB = app.state.db
    assert db.item("gdeadbeef00")["eligible"] == 0
    # simulate `labeling_app status` on a machine without the dataset: the
    # bundle alone has no flag for the stale item -> eligibility UNKNOWN
    stale_bundle = Bundle(out)
    assert not stale_bundle.eligibility_known()
    changes = db.sync_eligibility([i["item_id"] for i in stale_bundle.items],
                                  stale_bundle.ineligible_item_ids(),
                                  eligibility_known=stale_bundle.eligibility_known())
    assert changes == {"retired": [], "marked_ineligible": [], "restored": []}
    assert db.item("gdeadbeef00")["eligible"] == 0                        # enforcement intact
    with pytest.raises(LabelError):
        db.save_label("gdeadbeef00", "friend", score=1.0, rubric=[], expected_revision=0)


def test_orphaned_db_items_are_retired_not_resurrected(synth, tmp_path):
    """An item registered by an old bundle but absent from the current one is
    retired: never claimable, never admin-FINAL-able, surfaced as ineligible."""
    ds, out, _ = synth
    data_dir = tmp_path / "data"
    app = create_app(data_dir=data_dir, bundle_dir=out, dataset_dir=ds)
    db: LabelDB = app.state.db
    db.load_items([{"item_id": "gorphan0000", "max_score": 4.0, "rubric_items": []}])
    changes = db.sync_eligibility([i["item_id"] for i in app.state.bundle.items],
                                  app.state.bundle.ineligible_item_ids(),
                                  eligibility_known=app.state.bundle.eligibility_known())
    assert changes["retired"] == ["gorphan0000"]
    c = TestClient(app)
    c.post("/api/session", json={"name": "friend"})
    served = []
    for _ in range(10):
        r = c.post("/api/next", json={})
        if r.json().get("done"):
            break
        iid = r.json()["item"]["item_id"]
        served.append(iid)
        c.post(f"/api/items/{iid}/label", json={"score": 0.5, "status": "saved", "expected_revision": 0})
    assert "gorphan0000" not in served                                    # no phantom claims
    with pytest.raises(LabelError):
        db.set_final("gorphan0000", score=1.0, rubric=[], note="", source="adjudicated",
                     adjudicator="admin", expected_item_revision=1)
    # the HTTP admin route additionally 404s anything outside the served bundle
    r = c.post("/api/admin/items/gorphan0000/final", json={"score": 1, "expected_item_revision": 1})
    assert r.status_code == 404


def test_recompute_refuses_wrong_or_unusable_dataset(synth, tmp_path):
    ds, out, _ = synth
    b = Bundle(out)
    # wrong dataset (different inputs_sha256) -> refused, flags untouched
    other = _write_dataset(tmp_path / "other", [_case("e901_q1_r1", "wrong_choice_zero", selected="B",
                                                      version="1", accepted={"1": ["A"]})])
    man = json.loads((other / "manifest.json").read_text(encoding="utf-8"))
    man["inputs_sha256"] = "differentsha"
    (other / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    before = dict(b.eligibility)
    res = b.apply_dataset_eligibility(other)
    assert res["applied"] is False and "does not match" in res["reason"]
    assert b.eligibility == before
    # missing dataset files -> refused
    res = b.apply_dataset_eligibility(tmp_path / "nowhere")
    assert res["applied"] is False and "missing" in res["reason"]
    # missing private/id_map.json -> refused (not silently all-eligible)
    (out / "private" / "id_map.json").unlink()
    b2 = Bundle(out)
    res = b2.apply_dataset_eligibility(ds)
    assert res["applied"] is False and "id_map" in res["reason"]


def test_explicitly_marked_ineligible_item_is_rejected_without_dataset(synth, tmp_path):
    ds, out, _ = synth
    _make_stale(out, "e902_q1_r1")
    items = json.loads((out / "items.json").read_text(encoding="utf-8"))
    for it in items:
        if it["item_id"] == "gdeadbeef00":
            it["eligible_for_human_label"] = False
    (out / "items.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    app = create_app(data_dir=tmp_path / "data", bundle_dir=out)      # no dataset_dir at all
    c = TestClient(app)
    c.post("/api/session", json={"name": "friend"})
    r = c.post("/api/items/gdeadbeef00/label",
               json={"score": 0.0, "status": "saved", "expected_revision": 0})
    assert r.status_code == 400 and "not eligible" in r.json()["error"]


# ------------------------------------- existing labels become obsolete-safe --

def test_existing_label_and_final_surface_as_obsolete_not_deleted(synth, tmp_path):
    ds, out, _ = synth
    _make_stale(out, "e902_q1_r1")
    data_dir = tmp_path / "data"
    # 1) old server (no dataset knowledge): the stale item collects a label + FINAL
    app_old = create_app(data_dir=data_dir, bundle_dir=out)
    c = TestClient(app_old)
    c.post("/api/session", json={"name": "friend"})
    r = c.post("/api/items/gdeadbeef00/label",
               json={"score": 2.0, "status": "saved", "expected_revision": 0})
    assert r.status_code == 200
    rev = c.get("/api/admin/items/gdeadbeef00").json()["revision"]
    r = c.post("/api/admin/items/gdeadbeef00/final",
               json={"score": 2.0, "expected_item_revision": rev})
    assert r.status_code == 200
    # 2) corrected server: same DB, eligibility now known
    app_new = create_app(data_dir=data_dir, bundle_dir=out, dataset_dir=ds)
    c2 = TestClient(app_new)
    c2.post("/api/session", json={"name": "friend"})
    summary = c2.get("/api/admin/summary").json()
    assert summary["obsolete_ineligible_finals"] == 1
    assert summary["obsolete_ineligible_labels"] == 1
    ov = c2.get("/api/admin/items/gdeadbeef00").json()
    assert ov["state"] == "INELIGIBLE" and ov["final"]["obsolete_ineligible"] is True
    assert len(ov["labels"]) == 1                          # history NOT deleted
    # 3) export marks it; the importer refuses to promote it to ground truth
    db: LabelDB = app_new.state.db
    export = export_final(db, app_new.state.bundle, now="2026-08-22 13:00:00")
    row = next(i for i in export["items"] if i["item_id"] == "e902_q1_r1")
    assert row["label_kind"] == "human_final_label" and row["eligible_for_human_label"] is False
    assert export["obsolete_ineligible_count"] == 1
    export_path = tmp_path / "final_labels.json"
    export_path.write_text(json.dumps(export, ensure_ascii=False), encoding="utf-8")
    from autograder.benchmark.finallabels import import_final_labels
    res = import_final_labels(export_path, ds)
    assert res["imported"] == 0 and res["ignored_ineligible"] == ["e902_q1_r1"]
    written = json.loads((ds / "final_labels.json").read_text(encoding="utf-8"))
    assert written["labels"] == {} and "e902_q1_r1" in written["ignored_ineligible"]


def test_import_promotes_only_eligible_human_labels(synth, tmp_path):
    ds, _, _ = synth
    export = {"schema_version": 1, "kind": "grade_primary_final_labels", "content_sha256": "x",
              "exported_at": "2026-08-22", "final_count": 2, "bundle_items_sha256": "y",
              "items": [
                  {"item_id": "e901_q1_r1", "final_score": 3.0, "rubric_decisions": [], "note": "",
                   "source": "agreement", "contributing_graders": ["a", "b"], "adjudicator": None,
                   "finalized_at": "2026-08-22"},
                  {"item_id": "e902_q1_r1", "final_score": 2.0, "rubric_decisions": [], "note": "",
                   "source": "adjudicated", "contributing_graders": ["a"], "adjudicator": "admin",
                   "finalized_at": "2026-08-22"},
              ]}
    p = tmp_path / "export.json"
    p.write_text(json.dumps(export, ensure_ascii=False), encoding="utf-8")
    from autograder.benchmark.finallabels import import_final_labels
    res = import_final_labels(p, ds)
    assert res["imported"] == 1 and res["ignored_ineligible"] == ["e902_q1_r1"]
    written = json.loads((ds / "final_labels.json").read_text(encoding="utf-8"))
    assert set(written["labels"]) == {"e901_q1_r1"}
    assert written["ignored_ineligible"]["e902_q1_r1"]["deterministic_policy_score"] == 0.0


# ------------------------------------------------- real GRADE_PRIMARY facts --

@pytest.mark.skipif(not (REAL_DATASET / "manifest.json").exists(), reason="grade_primary dataset not built")
def test_real_grade_primary_contains_no_policy_decided_cases():
    inputs = [json.loads(l) for l in (REAL_DATASET / "cases_inputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    labels = {r["case_id"]: r for r in (json.loads(l) for l in
              (REAL_DATASET / "cases_labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    labelable, decided = split_cases(inputs, labels)
    assert len(decided) == 0, [r["case_id"] for r, _ in decided]
    assert len(labelable) == len(inputs) == 67
    c = eligibility_counts([e for _, e in labelable])
    # explanation-only cells under the independent policy: every case is human work
    assert c == c | {"source_cases": 67, "human_labelable": 67, "deterministic_zero": 0,
                     "excluded_choice_only": 0, "independent_explanation": 67}
