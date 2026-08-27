"""The CALIBRATION case audit is blinded, append-only, and changes no label.

Six cases from the completed v3-vs-v4 A/B: four that BOTH models downgraded
under BOTH prompts, and two that one model upgraded under grade-v4. The audit
checks strictness in both directions, so it must not tell the auditor which
direction a case came from — knowing "both models called this weak" is exactly
the anchor that would decide the case for them.

The blinding is the same code path as the DEV ground-truth audit: one tool, one
set of guards. These tests pin the properties for THIS queue, including the two
this queue adds — the audit group is hidden, and no label is rewritten.

No provider is contacted anywhere in this file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
UI = REPO / "scripts" / "ground_truth_audit_ui.py"
QUEUE = REPO / "evaluation" / "model_selection" / "runs" / "grade_primary" / \
    "CALIBRATION_AUDIT_2026-08-26.json"


def _mod():
    spec = importlib.util.spec_from_file_location("ground_truth_audit_ui", UI)
    m = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("ground_truth_audit_ui", m)
    spec.loader.exec_module(m)          # importing must not need streamlit
    return m


gta = _mod()
pytestmark = pytest.mark.skipif(not QUEUE.exists(), reason="audit queue not in this checkout")


@pytest.fixture()
def doc():
    return json.loads(QUEUE.read_text(encoding="utf-8"))


def _undecided(case: dict) -> dict:
    """A deep copy of a shipped case restored to its PRE-decision state.

    The six audit decisions were recorded (blind) on 2026-08-27, so the
    shipped artifact is decided. The blinding/append-only properties are
    properties of the CODE PATH over an undecided case; they are exercised on
    faithful pre-decision copies, never by mutating the real artifact."""
    import copy
    c = copy.deepcopy(case)
    for k in ("decided_at", "decided_by", "decided_blind", "models_revealed_at",
              "decision_revision", "decision_history"):
        c.pop(k, None)
    c["human_decision"] = None
    c["human_note"] = ""
    return c


# ------------------------------------------------------ the queue itself ----


def test_the_queue_holds_the_six_expected_cases(doc):
    ids = [c["case_id"] for c in doc["cases"]]
    assert ids == ["e004_q1_r1", "e004_q1_r3", "e004_q2_r6", "e004_q2_r8",
                   "e004_q1_r5", "e004_q1_r6"]
    groups = [c["audit_group"] for c in doc["cases"]]
    assert groups[:4] == ["shared_downgrade"] * 4
    assert groups[4:] == ["gemini_upgrade"] * 2


def test_it_offers_the_four_option_taxonomy(doc):
    assert set(doc["options"]) == {"A", "B", "C", "D"}
    assert "more lenient" in doc["options"]["B"]
    assert doc["note_guidance"]["B"]


def test_every_case_carries_what_the_auditor_needs(doc):
    for c in doc["cases"]:
        for k in ("question_text", "rubric", "official_solution", "frozen_transcription",
                  "instructor_final_score", "derived_explanation_verdict"):
            assert c[k], (c["case_id"], k)


# ------------------------------------------------- 1. blinding, pre-decision ----


def test_no_model_output_reaches_the_pre_decision_screen(doc):
    names = set()
    for c in doc["cases"]:
        names |= set((c.get("model_predictions") or {}).keys())
    assert names, "the queue must carry model predictions in order to withhold them"
    for real in doc["cases"]:
        c = _undecided(real)
        blob = json.dumps(gta.view_payload(c), ensure_ascii=False)
        assert "model_predictions" not in blob
        for n in names:
            assert n not in blob, f"{c['case_id']}: {n} leaked"
        # model-prediction keys only. `derived_explanation_verdict` IS shown
        # pre-decision by design — it is the ground truth under review, not a
        # model's opinion.
        for banned in ("raw_score", "justification", "cited_spans", "spans_verified",
                       "uncertain"):
            assert banned not in blob, (c["case_id"], banned)
        assert "derived_explanation_verdict" in blob, "the label under review must be shown"


def test_the_audit_group_is_hidden_before_the_decision(doc):
    """Telling the auditor a case is a 'shared_downgrade' announces that every
    model called it weak — the exact anchor this audit exists to avoid."""
    for c in map(_undecided, doc["cases"]):
        blob = json.dumps(gta.view_payload(c), ensure_ascii=False)
        assert "audit_group" not in blob
        assert "shared_downgrade" not in blob and "gemini_upgrade" not in blob


def test_the_direction_of_the_disagreement_is_not_inferable_from_the_payload(doc):
    """A downgrade case and an upgrade case must expose the same field set."""
    down = _undecided(next(c for c in doc["cases"] if c["audit_group"] == "shared_downgrade"))
    up = _undecided(next(c for c in doc["cases"] if c["audit_group"] == "gemini_upgrade"))
    assert set(gta.view_payload(down)) == set(gta.view_payload(up))


def test_the_allow_list_excludes_anything_added_later(doc):
    c = _undecided(doc["cases"][0])
    c["some_future_model_field"] = "both models downgraded this"
    payload = gta.view_payload(c)
    assert "some_future_model_field" not in payload
    assert set(payload) == set(gta.BLINDED_FIELDS) | {"_blinded"}


# --------------------------------- 2/3. append-only history, explicit reset ----


def test_decision_history_is_append_only(doc):
    c = _undecided(doc["cases"][0])
    gta.record_decision(c, "A", now="2026-08-26 10:00:00")
    gta.reset_decision(c, reason="re-read", now="2026-08-26 10:05:00")
    gta.record_decision(c, "B", note="lenient on the mechanism", now="2026-08-26 10:10:00")
    actions = [e["action"] for e in c["decision_history"]]
    assert actions == ["decide", "reset", "decide"]
    assert c["decision_history"][0]["decision"] == "A", "the first decision is never edited away"
    assert c["decision_history"][1]["previous_decision"] == "A"


def test_the_first_decision_is_made_blind_and_a_later_one_is_not(doc):
    c = _undecided(doc["cases"][0])
    gta.record_decision(c, "A", now="2026-08-26 10:00:00")
    assert c["decided_blind"] is True
    gta.reset_decision(c, reason="x", now="2026-08-26 10:05:00")
    gta.record_decision(c, "D", now="2026-08-26 10:10:00")
    assert c["decided_blind"] is False, "made with the model outputs already visible"


def test_revealing_the_models_does_not_unlock_the_decision(doc):
    c = _undecided(doc["cases"][0])
    gta.record_decision(c, "A", now="2026-08-26 10:00:00")
    gta.view_payload(c)                       # reveals
    assert c["human_decision"] == "A" and gta.is_decided(c)


def test_a_reset_never_pretends_the_models_were_unseen(doc):
    c = _undecided(doc["cases"][0])
    gta.record_decision(c, "A", now="2026-08-26 10:00:00")
    gta.reset_decision(c, reason="x", now="2026-08-26 10:05:00")
    assert c["models_revealed_at"] == "2026-08-26 10:00:00"
    assert c["decision_history"][-1]["models_already_revealed"] is True


# ------------------------------------------------------------ 4/5/6. safety ----


def test_no_held_out_case_is_present(doc):
    from autograder.benchmark.manifests import load_manifest

    by = {c.case_id: c for c in load_manifest("grade_primary").cases}
    for c in doc["cases"]:
        assert by[c["case_id"]].split == "CALIBRATION", c["case_id"]
    assert doc["held_out_exposed"] is False


def test_the_audit_records_that_it_costs_nothing(doc):
    assert doc["provider_calls"] == 0


def test_the_tooling_makes_no_provider_call():
    import inspect

    import scripts.calibration_audit_recompute as rc

    for mod in (gta, rc):
        src = inspect.getsource(mod)
        for banned in ("httpx", "requests", "openrouter", "gateway.call", "urllib"):
            assert banned not in src.lower(), (mod.__name__, banned)


def test_benchmark_labels_are_not_rewritten_by_the_audit(doc, tmp_path):
    """The audit changes which cases COUNT, never what the truth IS."""
    from autograder.benchmark.manifests import load_manifest

    before = {c.case_id: c.label.get("explanation_verdict")
              for c in load_manifest("grade_primary").cases}
    d = json.loads(json.dumps(doc))
    d["cases"] = [_undecided(c) for c in d["cases"]]     # re-adjudicate from scratch
    for c in d["cases"]:
        gta.record_decision(c, "B", now="2026-08-26 11:00:00")
    p = tmp_path / "audit.json"
    gta.save_audit(d, p)
    after = {c.case_id: c.label.get("explanation_verdict")
             for c in load_manifest("grade_primary").cases}
    assert before == after


# ------------------------------------------------------------- recompute ------


def test_recompute_reports_both_versions_and_excludes_only_c_and_d(tmp_path):
    import scripts.calibration_audit_recompute as rc

    d = json.loads(QUEUE.read_text(encoding="utf-8"))
    d["cases"] = [_undecided(c) for c in d["cases"]]     # scripted scenario, not the real audit
    by_id = {c["case_id"]: c for c in d["cases"]}
    for cid, dec in (("e004_q1_r1", "A"), ("e004_q1_r3", "B"),
                     ("e004_q2_r6", "C"), ("e004_q2_r8", "D")):
        gta.record_decision(by_id[cid], dec, now="2026-08-26 12:00:00")
    p = tmp_path / "audit.json"
    gta.save_audit(d, p)
    res = rc.recompute(p)
    assert res["provider_calls"] == 0
    assert res["excluded_from_strict_accuracy"] == ["e004_q2_r6", "e004_q2_r8"]
    assert res["strict_denominator"] == 10
    assert res["rubric_to_practice_mismatches"] == ["e004_q1_r3"]
    for arm in res["arms"].values():
        assert arm["pre_audit"]["n"] == 12, "the pre-audit view always keeps all 12"
        assert arm["revised"]["n"] == 10


def test_recompute_is_a_no_op_while_the_audit_is_undecided():
    import scripts.calibration_audit_recompute as rc

    res = rc.recompute(QUEUE)
    if any(res["decisions"].values()):
        pytest.skip("the audit has been completed; the undecided invariant no longer applies")
    assert res["strict_denominator"] == 12
    for arm in res["arms"].values():
        assert arm["pre_audit"] == arm["revised"]


def test_recompute_never_reads_a_provider():
    import scripts.calibration_audit_recompute as rc

    res = rc.recompute(QUEUE)
    assert set(res["arms"]) == {"A1", "A2", "B1", "B2"}
    for arm in res["arms"].values():
        assert arm["pre_audit"]["n"] == 12


# ------------------------------------ 7. provenance, images, overwrite guard ----


def _content_hash(doc):
    import hashlib
    mutable = {"human_decision", "human_note", "decision_history", "decided_at",
               "decided_by", "decided_blind", "models_revealed_at", "decision_revision"}
    frozen = [{k: v for k, v in c.items() if k not in mutable} for c in doc["cases"]]
    return hashlib.sha256(
        json.dumps(frozen, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def test_the_manifest_carries_frozen_provenance(doc):
    assert doc["schema_version"] == 1
    assert doc["audit_id"] == "CALIBRATION_AUDIT_2026-08-26"
    assert doc["split"] == "CALIBRATION" and doc["writer"] == "e004"
    assert doc["case_order"] == [c["case_id"] for c in doc["cases"]]
    assert len(doc["git_commit"]) == 40
    ds = doc["source_dataset"]
    assert ds["name"] == "grade_primary" and ds["status"] == "FROZEN"
    assert len(ds["inputs_sha256"]) == 64 and len(ds["labels_sha256"]) == 64
    assert doc["content_hash"] == _content_hash(doc), \
        "content_hash must match the auditor-facing material"


def test_the_content_hash_ignores_decisions(doc):
    """Deciding a case must not change what the population claims to be —
    pinned twice: the SHIPPED decided artifact still matches its recorded
    hash (the provenance test), and re-deciding a pre-decision copy leaves
    the hash unchanged."""
    import copy
    before = _content_hash(doc)
    d2 = copy.deepcopy(doc)
    d2["cases"][0] = _undecided(d2["cases"][0])
    gta.record_decision(d2["cases"][0], "A", now="2026-08-27 10:00:00")
    assert _content_hash(d2) == before


def test_every_case_names_its_split_and_masked_source_page(doc):
    for c in doc["cases"]:
        assert c["split"] == "CALIBRATION", c["case_id"]
        sp = c["source_page"]
        assert sp["masked"] is True
        assert (REPO / sp["pdf"]).exists(), sp["pdf"]
        assert isinstance(sp["page"], int)


def test_answer_crops_exist_and_carry_no_red_ink(doc):
    np = pytest.importorskip("numpy")
    PIL = pytest.importorskip("PIL.Image")
    sys.path.insert(0, str(REPO))
    from labeling_app.bundle import strict_red_count

    for c in doc["cases"]:
        crop = REPO / c["answer_crop"]
        assert crop.exists(), c["answer_crop"]
        assert "clean" in crop.name and "orig" not in crop.name
        arr = np.array(PIL.open(crop).convert("RGB"))
        assert strict_red_count(arr) == 0, f"{c['case_id']}: red ink in the shown crop"


def test_the_pre_decision_payload_never_references_the_raw_marked_page(doc):
    for c in doc["cases"]:
        blob = json.dumps(gta.view_payload(c), ensure_ascii=False)
        assert "orig" not in blob, "raw instructor-marked image leaked into the payload"
        assert "audit_group" not in blob


def test_a_saved_decision_refuses_a_silent_overwrite(doc):
    c = _undecided(doc["cases"][0])
    gta.record_decision(c, "A", now="2026-08-27 10:00:00")
    with pytest.raises(ValueError, match="reset explicitly"):
        gta.record_decision(c, "B", now="2026-08-27 10:01:00")
    assert c["human_decision"] == "A"
    assert [e["action"] for e in c["decision_history"]] == ["decide"]


def test_a_decision_pins_its_blind_payload_and_revision(doc):
    import hashlib
    c = _undecided(doc["cases"][0])
    expected = hashlib.sha256(
        json.dumps(gta.pre_decision_payload(c), ensure_ascii=False, sort_keys=True)
        .encode("utf-8")).hexdigest()
    e1 = gta.record_decision(c, "A", now="2026-08-27 10:00:00",
                             manifest_hash=doc.get("content_hash"))
    assert e1["payload_sha256"] == expected
    assert e1["revision"] == 1 and c["decision_revision"] == 1
    assert e1["manifest_content_hash"] == doc["content_hash"]
    assert c["decided_by"] == "owner"
    gta.reset_decision(c, reason="second look", now="2026-08-27 10:05:00")
    e2 = gta.record_decision(c, "B", now="2026-08-27 10:10:00")
    assert e2["revision"] == 2 and e2["made_blind"] is False
    # the original blind decision is still fully present in the history
    assert c["decision_history"][0]["decision"] == "A"
    assert c["decision_history"][0]["made_blind"] is True
    assert c["decision_history"][0]["payload_sha256"] == expected
