"""The ground-truth audit is BLINDED: no model output before a human decision.

Three models agreeing is not evidence about a label — it can equally mean three
models are wrong in the same direction. Shown first it anchors, so the audit
withholds model outputs until the human decision for that case is saved.

These tests check the CODE PATH, not the layout. ``view_payload`` is the only
thing the UI renders from, so anything absent from its pre-decision output
cannot reach the browser at all — not through a collapsed expander, a hidden
element, or the page source.

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


def _mod():
    spec = importlib.util.spec_from_file_location("ground_truth_audit_ui", UI)
    m = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("ground_truth_audit_ui", m)
    spec.loader.exec_module(m)      # importing must not need streamlit
    return m


gta = _mod()


@pytest.fixture()
def case():
    """A realistic undecided case, shaped like the real audit file."""
    return {
        "case_id": "e007_q1_r1", "writer": "e007", "question_id": "1", "sub_item_id": "1",
        "question_text": "שאלה מספר 1", "rubric": [{"id": "R1", "text": "כלל כלשהו"}],
        "official_solution": {"1": "תמונה F"}, "max_score": 4.0,
        "frozen_transcription": "ניתן לראות שיש סוג של מתיחה",
        "instructor_final_score": 4.0, "derived_explanation_verdict": "valid",
        "derivation_reason": "full_credit_implies_valid",
        "model_predictions": {
            "google/gemini-3.7-flash": {"raw_score": 2.0, "verdict": "partially_valid",
                                        "justification": "vague about the blur direction",
                                        "rubric_items_met": [], "uncertain": False},
            "anthropic/claude-sonnet-5": {"raw_score": 0.0, "verdict": "invalid",
                                          "justification": "does not address motion blur",
                                          "rubric_items_met": [], "uncertain": False},
        },
        "human_decision": None, "human_note": "",
    }


# ------------------------------------------------- the pre-decision screen ----


def test_the_pre_decision_payload_carries_no_model_output(case):
    payload = gta.view_payload(case)
    assert payload["_blinded"] is True
    blob = json.dumps(payload, ensure_ascii=False)
    for banned in ("model_predictions", "google/gemini-3.7-flash",
                   "anthropic/claude-sonnet-5", "partially_valid", "invalid",
                   "vague about the blur direction", "does not address motion blur",
                   "raw_score", "justification", "uncertain"):
        assert banned not in blob, f"{banned!r} reached the pre-decision screen"


def test_the_pre_decision_payload_is_an_allow_list_not_a_deny_list(case):
    """A model-derived field added to the audit file later must be excluded by
    default, not leak until someone remembers to ban it."""
    case["some_future_model_field"] = "gemini says the explanation is weak"
    payload = gta.view_payload(case)
    assert "some_future_model_field" not in payload
    assert "gemini" not in json.dumps(payload, ensure_ascii=False)
    assert set(payload) == set(gta.BLINDED_FIELDS) | {"_blinded"}


def test_the_auditor_still_gets_everything_needed_to_judge(case):
    payload = gta.view_payload(case)
    for needed in ("question_text", "rubric", "official_solution", "frozen_transcription",
                   "instructor_final_score", "derived_explanation_verdict"):
        assert payload[needed], needed


def test_every_real_audit_case_is_blinded_while_undecided():
    """The shipped artifact, not just a fixture."""
    doc = json.loads(gta.AUDIT.read_text(encoding="utf-8"))
    names = set()
    for c in doc["cases"]:
        names |= set((c.get("model_predictions") or {}).keys())
    assert names, "the audit file should carry model predictions to withhold"
    for c in doc["cases"]:
        if gta.is_decided(c):
            continue
        blob = json.dumps(gta.view_payload(c), ensure_ascii=False)
        for model in names:
            assert model not in blob, f"{c['case_id']}: {model} leaked pre-decision"
        assert "model_predictions" not in blob


# ------------------------------------------------------------ after the save ---


def test_models_are_revealed_only_after_a_decision_is_saved(case):
    assert "model_predictions" not in gta.view_payload(case)
    gta.record_decision(case, "A", now="2026-08-25 16:00:00")
    revealed = gta.view_payload(case)
    assert revealed["_blinded"] is False
    assert set(revealed["model_predictions"]) == set(case["model_predictions"])


def test_a_blind_decision_is_recorded_as_blind(case):
    gta.record_decision(case, "B", note="lenient", now="2026-08-25 16:00:00")
    assert case["human_decision"] == "B"
    assert case["decided_blind"] is True
    assert case["decided_at"] == "2026-08-25 16:00:00"
    assert case["models_revealed_at"] == "2026-08-25 16:00:00"
    assert case["decision_history"][-1]["made_blind"] is True
    assert case["decision_history"][-1]["action"] == "decide"


def test_revealing_the_models_does_not_unlock_the_decision(case):
    gta.record_decision(case, "A", now="2026-08-25 16:00:00")
    # the reveal happened; the decision must still be there and still marked blind
    gta.view_payload(case)
    assert case["human_decision"] == "A"
    assert case["decided_blind"] is True
    assert gta.is_decided(case)


# ------------------------------------------------------- re-adjudication ------


def test_reset_is_explicit_and_recorded(case):
    gta.record_decision(case, "A", now="2026-08-25 16:00:00")
    gta.reset_decision(case, reason="rubric re-read", now="2026-08-25 16:30:00")
    assert case["human_decision"] is None
    entry = case["decision_history"][-1]
    assert entry["action"] == "reset"
    assert entry["previous_decision"] == "A"
    assert entry["reason"] == "rubric re-read"
    assert entry["models_already_revealed"] is True


def test_a_decision_made_after_the_reveal_is_marked_sighted(case):
    """A judgement revised once the model outputs are known is weaker evidence
    than a blind one, and the file has to say which it was."""
    gta.record_decision(case, "A", now="2026-08-25 16:00:00")
    gta.reset_decision(case, reason="reconsidering", now="2026-08-25 16:30:00")
    gta.record_decision(case, "B", now="2026-08-25 17:00:00")
    assert case["human_decision"] == "B"
    assert case["decided_blind"] is False, "this one was made with the models visible"
    assert [e["action"] for e in case["decision_history"]] == ["decide", "reset", "decide"]
    assert case["decision_history"][-1]["made_blind"] is False


def test_the_case_goes_back_to_blinded_rendering_after_a_reset(case):
    """It re-blinds the SCREEN, but the history still records that the auditor
    has already seen the outputs — the file never pretends otherwise."""
    gta.record_decision(case, "A", now="2026-08-25 16:00:00")
    gta.reset_decision(case, reason="x", now="2026-08-25 16:30:00")
    assert gta.view_payload(case)["_blinded"] is True
    assert case["models_revealed_at"] == "2026-08-25 16:00:00"


# ----------------------------------------------------------- provenance -------


def test_existing_audit_provenance_is_preserved():
    doc = json.loads(gta.AUDIT.read_text(encoding="utf-8"))
    for key in ("artifact", "created_at", "scope", "instructions", "warning",
                "options", "held_out_exposed", "cases"):
        assert key in doc, key
    assert doc["held_out_exposed"] is False
    for c in doc["cases"]:
        for key in ("case_id", "derived_explanation_verdict", "derivation_reason",
                    "instructor_final_score", "frozen_transcription", "model_predictions"):
            assert key in c, (c.get("case_id"), key)


def test_no_held_out_case_is_present():
    from autograder.benchmark.manifests import load_manifest

    doc = json.loads(gta.AUDIT.read_text(encoding="utf-8"))
    by = {c.case_id: c for c in load_manifest("grade_primary").cases}
    for c in doc["cases"]:
        assert by[c["case_id"]].split == "DEV", c["case_id"]


def test_save_round_trips_without_losing_history(tmp_path, case):
    doc = {"artifact": "x", "options": {"A": "a"}, "cases": [case]}
    gta.record_decision(case, "A", now="2026-08-25 16:00:00")
    p = tmp_path / "audit.json"
    gta.save_audit(doc, p)
    back = gta.load_audit(p)
    assert back["cases"][0]["decision_history"] == case["decision_history"]
    assert back["cases"][0]["decided_blind"] is True


def test_the_ui_renders_only_from_view_payload():
    """Guard against a future edit reaching past the blinding helper."""
    src = UI.read_text(encoding="utf-8")
    body = src.split("def main(")[1]
    assert 'case["model_predictions"]' not in body, \
        "main() must read model output through view_payload, never from the case dict"
    assert 'view["model_predictions"]' in body
