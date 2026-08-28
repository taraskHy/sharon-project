"""The LOCAL grader output contract and the symmetric grounding rules.

Everything here is synthetic — no real benchmark answer, writer, case id or
Hebrew benchmark phrase appears (test_no_benchmark_content_in_the_local_prompt
enforces the same property on the prompt itself). ZERO inference, zero
network.

Covers, by number, the 24 owner-directed regression requirements of
2026-08-28 (see the test names).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autograder.escalation import (GRADE_SYSTEM_BY_VERSION,  # noqa: E402
                                   GRADE_SYSTEM_V4_CHARITABLE,
                                   GRADE_VALIDATION_VERSION, GradeResult,
                                   LOCAL_OUTPUT_CONTRACT, RubricItemGrade,
                                   validate_grade)
from autograder.evidence import UNGROUNDED_INVALID_VERDICT  # noqa: E402
from autograder.gradingpack import QuestionGradingPack, RubricItemSpec  # noqa: E402

LOCAL_VERSION = "grade-v4-charitable-local"
TRANSCRIPTION = "the filter keeps the slow changes and drops the fast ones"


def mk_pack(**over) -> QuestionGradingPack:
    kw = dict(question_id="q1", question_text="why does the filter keep slow changes?",
              question_type="open", max_score=4.0, correct_by_version={},
              rubric=["states the central idea"], scoring_rules=[],
              grading_policy="choice_and_explanation_independent",
              official_solution={"1": "a low pass filter attenuates high frequencies"},
              rubric_items=[RubricItemSpec(id="R1", text="states the central idea",
                                           points=None, requires_evidence=True,
                                           excludes=[], requires=[], kind="semantic")],
              evidence_policy="required", rag_policy="RAG_DISABLED")
    kw.update(over)
    return QuestionGradingPack(**kw)


def check(g: GradeResult, transcription: str | None = TRANSCRIPTION, pack=None):
    return validate_grade(g, pack or mk_pack(), selection_correct=None, selected=None,
                          transcription=transcription)


# ------------------------------------------------- credit-side grounding ----


def test_01_positive_credit_with_empty_rubric_items_fails_to_review():
    v = check(GradeResult(score=4.0))
    assert not v.ok and v.evidence["ungrounded_credit"]


def test_02_positive_credit_with_quote_only_in_top_level_evidence_fails():
    v = check(GradeResult(score=4.0, evidence="the filter keeps the slow changes"))
    assert not v.ok and v.evidence["ungrounded_credit"]


def test_03_positive_credit_with_exact_span_in_rubric_items_is_accepted():
    v = check(GradeResult(score=4.0, rubric_items=[
        RubricItemGrade(id="R1", met=True, student_evidence="keeps the slow changes")]))
    assert v.ok and v.problems == []
    assert v.evidence["verified"] == ["R1"]


def test_04_paraphrased_student_evidence_fails():
    v = check(GradeResult(score=4.0, rubric_items=[
        RubricItemGrade(id="R1", met=True,
                        student_evidence="the student retains gradual variation")]))
    assert not v.ok and "R1" in v.evidence["fabricated"]


def test_05_overlong_student_evidence_fails_even_when_verbatim():
    long_text = "slow " * 60                      # 300 chars
    v = check(GradeResult(score=4.0, rubric_items=[
        RubricItemGrade(id="R1", met=True, student_evidence=long_text.strip())]),
        transcription=long_text)
    assert not v.ok
    assert any("exceeds" in p for p in v.problems)


def test_06_official_solution_quote_as_student_evidence_fails():
    pack = mk_pack()
    v = check(GradeResult(score=4.0, rubric_items=[
        RubricItemGrade(id="R1", met=True,
                        student_evidence="a low pass filter attenuates high frequencies")]),
        pack=pack)
    assert not v.ok and "R1" in v.evidence["fabricated"]


def test_07_unknown_rubric_id_fails_even_on_an_unmet_entry():
    v = check(GradeResult(score=0.0, rubric_items=[
        RubricItemGrade(id="R9", met=False, student_evidence="drops the fast ones")]))
    assert not v.ok
    assert any("unknown rubric ids" in p for p in v.problems)


def test_08_duplicate_rubric_id_fails():
    v = check(GradeResult(score=4.0, rubric_items=[
        RubricItemGrade(id="R1", met=True, student_evidence="keeps the slow changes"),
        RubricItemGrade(id="R1", met=False)]))
    assert not v.ok
    assert any("more than once" in p for p in v.problems)


def test_09_justification_prose_cannot_substitute_for_student_evidence():
    v = check(GradeResult(score=4.0, evidence=(
        "The student states 'keeps the slow changes' which correctly identifies "
        "the central idea, so full credit is appropriate.")))
    assert not v.ok and v.evidence["ungrounded_credit"]


def test_10_top_level_evidence_cannot_substitute_even_as_an_exact_span():
    v = check(GradeResult(score=4.0, evidence=TRANSCRIPTION))
    assert not v.ok and v.evidence["ungrounded_credit"]


# --------------------------------------------------- zero-side grounding ----


def test_11_invalid_on_non_empty_text_without_grounding_routes_to_review():
    v = check(GradeResult(score=0.0))
    assert not v.ok
    assert v.evidence["ungrounded_invalid"]
    assert any(UNGROUNDED_INVALID_VERDICT in p for p in v.problems)


def test_12_invalid_with_exact_contradictory_span_is_accepted():
    v = check(GradeResult(score=0.0, rubric_items=[
        RubricItemGrade(id="R1", met=False, student_evidence="drops the fast ones")]))
    assert v.ok and v.problems == []
    assert v.evidence["verified"] == ["R1"]
    assert v.evidence["ungrounded_invalid"] is False


def test_13_invalid_with_only_generic_prose_routes_to_review():
    v = check(GradeResult(score=0.0, rubric_items=[
        RubricItemGrade(id="R1", met=False,
                        student_evidence="the answer is generally too vague to credit")]))
    assert not v.ok
    assert v.evidence["ungrounded_invalid"] and "R1" in v.evidence["fabricated"]


def test_14_blank_answer_behavior_unchanged():
    """Grounding cannot be demanded from text that does not exist: blank /
    whitespace transcriptions never trigger the zero-side rule (the
    deterministic upstream blank/illegible statuses keep handling them)."""
    for blank in ("", "   \n"):
        v = check(GradeResult(score=0.0), transcription=blank)
        assert v.ok, (blank, v.problems)
        assert not v.evidence.get("ungrounded_invalid")


def test_14b_unavailable_transcription_is_not_blank_and_fails_closed():
    """None means UNVERIFIABLE, not empty: a zero that cannot be checked
    against the student's text must not auto-finalize."""
    v = check(GradeResult(score=0.0), transcription=None)
    assert not v.ok
    assert v.evidence["ungrounded_invalid"]


def test_15_uncertain_zero_routes_to_review_not_auto():
    v = check(GradeResult(score=0.0, uncertain=True))
    assert not v.ok
    assert any("uncertainty" in p for p in v.problems)


# ------------------------- adversarial-review closures (2026-08-28) ----------


def test_epsilon_score_invalid_verdict_cannot_slip_into_the_credit_branch():
    """Any score in (0, 0.001*max] is an 'invalid' verdict in production; it
    must be held to the ZERO-side rule, not validated as credit."""
    from autograder.reliability import _verdict_from_score
    assert _verdict_from_score(0.004, 4.0) == "invalid"
    v = check(GradeResult(score=0.004))
    assert not v.ok and v.evidence["ungrounded_invalid"]
    # even a credited (met=true) entry cannot launder the zero: it is an
    # internal contradiction, never grounding
    v2 = check(GradeResult(score=0.004, rubric_items=[
        RubricItemGrade(id="R1", met=True, student_evidence="keeps the slow changes")]))
    assert not v2.ok
    assert any("contradictory" in p for p in v2.problems)


def test_unknown_evidence_policy_fails_closed():
    for bogus in ("Required", "requried", "on", ""):
        v = check(GradeResult(score=4.0, rubric_items=[
            RubricItemGrade(id="R1", met=True, student_evidence="keeps the slow changes")]),
            pack=mk_pack(evidence_policy=bogus))
        assert not v.ok, bogus
        assert any("unknown evidence_policy" in p for p in v.problems), bogus


def test_zero_verdict_grounded_only_by_met_false_spans():
    """A verified span on a met=true entry credits text the score denies —
    contradiction, REVIEW. Only a met=false wrong-claim span grounds a zero."""
    v = check(GradeResult(score=0.0, rubric_items=[
        RubricItemGrade(id="R1", met=True, student_evidence="keeps the slow changes")]))
    assert not v.ok
    assert any("contradictory" in p for p in v.problems)
    assert v.evidence["ungrounded_invalid"]


def test_trivial_spans_cannot_ground_anything():
    from autograder.evidence import evidence_supported
    assert evidence_supported("e", TRANSCRIPTION) is False
    assert evidence_supported("s", TRANSCRIPTION) is False
    v = check(GradeResult(score=4.0, rubric_items=[
        RubricItemGrade(id="R1", met=True, student_evidence="e")]))
    assert not v.ok and "R1" in v.evidence["fabricated"]


def test_legacy_credit_contradicting_structured_met_false_is_refused():
    v = check(GradeResult(score=0.0,
                          rubric_items=[RubricItemGrade(id="R1", met=False,
                                                        student_evidence="drops the fast ones")],
                          rubric_items_met=["R1"]))
    assert not v.ok
    assert any("contradictory" in p for p in v.problems)


def test_adapter_treats_extra_field_outputs_as_schema_failures_not_crashes():
    from autograder.benchmark.roles import GradeAdapter
    adapter = GradeAdapter("grade_primary", prompt_version=LOCAL_VERSION)
    row = adapter.score(_bench_case(), {"score": 4.0, "bogus_field": "x"}, None)
    assert row["schema_failure"] is True
    assert row["decision"] == "REVIEW"


# --------------------------- semantics unchanged (structural proof) ----------


def test_16_17_18_charitable_semantics_are_verbatim_v4():
    """FULL-for-informal-central-idea, PARTIAL-for-incomplete, and
    no-promotion-of-vague-generic are v4 semantics; the local version contains
    the v4 text VERBATIM and only appends the mechanical output contract."""
    local = GRADE_SYSTEM_BY_VERSION[LOCAL_VERSION]
    assert local.startswith(GRADE_SYSTEM_V4_CHARITABLE)
    assert local == GRADE_SYSTEM_V4_CHARITABLE + "\n\n" + LOCAL_OUTPUT_CONTRACT
    for semantic in ("FULL explanation quality", "PARTIAL explanation quality",
                     "ZERO explanation quality",
                     "informal or imprecise terminology",
                     "A CONCISE answer can earn full quality",
                     "not fully establish",
                     "would fit almost any unrelated problem"):
        assert semantic in local, semantic
    # the contract adds no new grading level and never redefines the scale
    assert "OUTPUT CONTRACT" in LOCAL_OUTPUT_CONTRACT
    assert "semantics above are unchanged" in LOCAL_OUTPUT_CONTRACT


def test_contract_states_the_mechanical_requirements():
    c = LOCAL_OUTPUT_CONTRACT
    assert "rubric_items` must NOT be empty" in c
    assert "character-for-character" in c
    assert "at most 200" in c
    assert "leave it null" in c or "leave it \nnull" in c
    assert "uncertain=true" in c
    assert "never student evidence" in c
    assert "met=false" in c                      # grounded zero
    assert "do NOT fabricate" in c


def test_19_no_benchmark_content_in_the_local_prompt():
    local = GRADE_SYSTEM_BY_VERSION[LOCAL_VERSION]
    assert not re.search(r"\be\d{3}(_q\d+_r\d+)?\b", local), "case/writer id leaked"
    assert not re.search(r"[֐-׿]", local), "Hebrew benchmark phrase leaked"
    for banned in ("DEV", "CALIBRATION", "HELD_OUT", "expected verdict",
                   "audit", "instructor score", "ground truth"):
        assert banned not in local, banned
    # no expected-verdict letter codes either
    assert not re.search(r"\bdecision [ABCD]\b", local)


# ------------------------------------------- shared path / route safety -----


def _bench_case(transcription=TRANSCRIPTION):
    from autograder.benchmark.manifests import BenchCase
    pack = {"question_id": "q1", "question_text": "why does the filter keep slow changes?",
            "question_type": "open", "max_score": 4.0, "correct_by_version": {},
            "rubric": ["states the central idea"], "scoring_rules": [],
            "grading_policy": "choice_and_explanation_independent",
            "official_solution": {"1": "a low pass filter attenuates high frequencies"},
            "rubric_items": [{"id": "R1", "text": "states the central idea",
                              "points": None, "requires_evidence": True,
                              "excludes": [], "requires": [], "kind": "semantic"}],
            "evidence_policy": "required"}
    return BenchCase(case_id="syn_q1_r1", split="DEV", component="ALL",
                     inputs={"case_id": "syn_q1_r1", "pack": pack, "selected": None,
                             "transcription": transcription, "version": None},
                     label={"selection_correct": None, "transcription_complete": True,
                            "score": 4.0})


def test_20_production_and_benchmark_share_the_same_validation():
    """The adapter routes through escalation.validate_grade — the SAME rules
    decide AUTO/REVIEW in the benchmark and in production."""
    from autograder.benchmark.roles import GradeAdapter
    adapter = GradeAdapter("grade_primary", prompt_version=LOCAL_VERSION)
    case = _bench_case()
    ungrounded = GradeResult(score=0.0).model_dump()
    row = adapter.score(case, ungrounded, None)
    assert row["decision"] == "REVIEW"
    assert row["evidence_ungrounded_invalid"] is True
    grounded = GradeResult(score=0.0, rubric_items=[
        RubricItemGrade(id="R1", met=False,
                        student_evidence="drops the fast ones")]).model_dump()
    row2 = adapter.score(case, grounded, None)
    assert row2["decision"] == "AUTO"
    assert row2["evidence_ungrounded_invalid"] is False


def test_21_openrouter_grading_remains_blocked_in_production():
    from autograder.cloudboundary import CloudBoundaryError, check_cloud_call
    with pytest.raises(CloudBoundaryError):
        check_cloud_call(task="grade_primary", backend="openrouter", base_url=None,
                         execution_mode="production")


def test_22_local_route_is_not_a_cloud_route():
    from autograder.usage import is_cloud_route
    assert is_cloud_route("ollama", "http://localhost:11434/v1") is False


def test_23_no_target_reaches_the_model_request():
    from autograder.benchmark.roles import GradeAdapter
    from autograder.benchmark.runner import leakage_check
    adapter = GradeAdapter("grade_primary", prompt_version=LOCAL_VERSION)
    case = _bench_case()
    req = adapter.build_request(dict(case.inputs), REPO)
    leakage_check(case, req, adapter.model_visible_fields)   # raises on any leak
    text = req.text_for_inspection()
    for banned in ("selection_correct", "label_score", "explanation_verdict"):
        assert banned not in text
    assert "DEV" not in text and "CALIBRATION" not in text


def test_24_audit_decisions_never_become_expected_labels():
    """The active freeze's targets re-derive from the instructor grades; the
    A/B/C/D letters are flags that appear ONLY in the audit/strict blocks."""
    from scripts.local_grade_freeze import FREEZE_PATH, build_freeze
    if not FREEZE_PATH.exists():
        pytest.skip("freeze record not written yet in this checkout")
    doc = build_freeze()
    verdicts = set(doc["populations"]["calibration_verdict_v4"]["verdicts"].values())
    assert verdicts <= {"valid", "partially_valid", "invalid"}
    assert not (verdicts & set("ABCD"))
    assert set(doc["human_audit"]["decisions"].values()) <= set("ABCD")
    # excluding a C case never rewrites its target
    for cid in doc["strict_metrics"]["excluded"]:
        assert doc["populations"]["calibration_verdict_v4"]["verdicts"][cid] in (
            "valid", "partially_valid", "invalid")
    assert "flags only" in doc["ground_truth"] or "flags" in doc["ground_truth"]


def test_validation_version_is_recorded():
    assert GRADE_VALIDATION_VERSION == "grade-validation-v2"
