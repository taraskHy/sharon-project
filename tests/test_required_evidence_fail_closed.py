"""Required evidence fails CLOSED: ungrounded credit is never AUTO.

The rule used to be open at the bottom. ``validate_evidence`` iterated the
CREDITED rubric items, so a grader that awarded a positive score while marking
no item met had nothing to iterate — it passed validation and went straight to
AUTO. The models that most needed catching were exactly the ones citing least.

Measured over the frozen 26-case DEV population (persisted outputs, no new
calls): 19/19 of one candidate's credit-awarding grades carried no verified
span at all, and every one was AUTO.

The rule now: under ``evidence_policy="required"``, credit must rest on at
least one VERIFIED span, unless a credited item's spec explicitly opted out.

Since grade-validation-v2 (2026-08-28) the zero side is symmetric: a zero on
NON-EMPTY text is an assertion of demerit and must also rest on a verified
span (normally a ``met=false`` entry citing the wrong/contradictory claim) —
the FullDev run's only AUTO decision was an ungrounded zero that undergraded
a full-credit answer. A zero on BLANK text still demands no grounding: the
grader is explaining an absence.

Generic by construction: nothing here mentions a vendor.

No provider is contacted anywhere in this file.
"""

from __future__ import annotations

import pytest

from autograder.escalation import GradeResult, validate_grade
from autograder.evidence import CreditedItem, validate_evidence
from autograder.gradingpack import QuestionGradingPack, RubricItemSpec

TRANSCRIPTION = "ניתן לראות שיש סוג של מתיחה בתמונה"
SPAN = "יש סוג של מתיחה"


def _pack(policy="required", *, requires_evidence=True, max_score=4.0):
    return QuestionGradingPack(
        question_id="1", question_text="q", question_type="matching_with_explanation",
        max_score=max_score, rubric=[], scoring_rules=[],
        rubric_items=[RubricItemSpec(id="R1", text="explains the stretch",
                                     requires_evidence=requires_evidence)],
        grading_policy="choice_and_explanation_independent",
        official_solution={}, correct_by_version={}, evidence_policy=policy)


def _decide(g: GradeResult, pack) -> str:
    v = validate_grade(g, pack, selection_correct=None, selected=None,
                       transcription=TRANSCRIPTION)
    return "AUTO" if (v.ok and not g.uncertain) else "REVIEW"


# ---------------------------------------- 1/2. required + no evidence -> REVIEW


def test_required_evidence_with_empty_evidence_is_review():
    g = GradeResult(score=4.0, rubric_items=[{"id": "R1", "met": True, "student_evidence": ""}])
    assert _decide(g, _pack()) == "REVIEW"


def test_required_evidence_with_null_evidence_is_review():
    g = GradeResult(score=4.0, rubric_items=[{"id": "R1", "met": True, "student_evidence": None}])
    assert _decide(g, _pack()) == "REVIEW"


def test_credit_with_no_rubric_item_credited_at_all_is_review():
    """The hole the old rule left open: nothing credited => nothing iterated
    => AUTO, even though a positive score asserts merit."""
    g = GradeResult(score=2.0, rubric_items=[{"id": "R1", "met": False}])
    assert _decide(g, _pack()) == "REVIEW"

    g2 = GradeResult(score=2.0)          # no rubric_items at all
    assert _decide(g2, _pack()) == "REVIEW"


def test_the_failure_is_reported_as_ungrounded_credit():
    v = validate_evidence(credited=[], transcription=TRANSCRIPTION,
                          specs=_pack().rubric_specs(),
                          policy="required", credit_awarded=True)
    assert v.ungrounded_credit is True
    assert not v.ok
    assert any("no verified evidence span" in p for p in v.problems)


def test_a_pack_with_no_rubric_items_cannot_be_held_to_the_rule():
    """Grounding must be EXPRESSIBLE before it can be demanded. A pack that
    declares no rubric items offers nothing to cite, so requiring a span would
    put every positive score in permanent REVIEW — a demand no grader could
    ever satisfy is a deadlock, not a safety check."""
    v = validate_evidence(credited=[], transcription=TRANSCRIPTION, specs={},
                          policy="required", credit_awarded=True)
    assert v.ok and v.ungrounded_credit is False

    from autograder.gradingpack import QuestionGradingPack

    bare = QuestionGradingPack(
        question_id="1", question_text="q", question_type="matching_with_explanation",
        max_score=4.0, rubric=[], rubric_items=[], scoring_rules=[],
        grading_policy="choice_and_explanation_independent", official_solution={},
        correct_by_version={}, evidence_policy="required")
    assert bare.rubric_item_ids() == []
    assert _decide(GradeResult(score=4.0), bare) == "AUTO"


def test_awarding_nothing_on_non_empty_text_needs_symmetric_grounding():
    """grade-validation-v2: a zero on NON-EMPTY text asserts DEMERIT and is
    grounded exactly like credit; a zero on blank text still explains an
    absence and needs no grounding."""
    ungrounded = GradeResult(score=0.0, rubric_items=[{"id": "R1", "met": False}])
    assert _decide(ungrounded, _pack()) == "REVIEW"
    grounded = GradeResult(score=0.0, rubric_items=[
        {"id": "R1", "met": False, "student_evidence": SPAN}])
    assert _decide(grounded, _pack()) == "AUTO"
    # blank text: unchanged — the deterministic upstream path owns it
    v = validate_grade(GradeResult(score=0.0), _pack(), selection_correct=None,
                       selected=None, transcription="")
    assert v.ok
    # and the CREDIT-side helper semantics are untouched: score 0 asserts no merit
    v2 = validate_evidence(credited=[], transcription=TRANSCRIPTION,
                           policy="required", credit_awarded=False)
    assert v2.ok and v2.ungrounded_credit is False


# ------------------------------------------- 3. optional policy still permits


def test_optional_policy_allows_credit_without_a_span():
    g = GradeResult(score=4.0, rubric_items=[{"id": "R1", "met": True, "student_evidence": ""}])
    assert _decide(g, _pack("optional")) == "AUTO"


def test_disabled_policy_checks_nothing():
    g = GradeResult(score=4.0, rubric_items=[{"id": "R1", "met": True, "student_evidence": ""}])
    assert _decide(g, _pack("disabled")) == "AUTO"


def test_a_spec_may_opt_out_of_evidence_and_still_be_grounded_enough():
    """An item legitimately gradeable without a quote (declared, never assumed)
    satisfies the fail-closed rule on its own."""
    g = GradeResult(score=4.0, rubric_items=[{"id": "R1", "met": True, "student_evidence": ""}])
    assert _decide(g, _pack(requires_evidence=False)) == "AUTO"


# ------------------------------------------------- 4/5. supported vs not


def test_supported_non_empty_evidence_is_valid():
    g = GradeResult(score=4.0, rubric_items=[{"id": "R1", "met": True, "student_evidence": SPAN}])
    assert _decide(g, _pack()) == "AUTO"


def test_unsupported_non_empty_evidence_is_review():
    g = GradeResult(score=4.0,
                    rubric_items=[{"id": "R1", "met": True,
                                   "student_evidence": "משפט שהתלמיד מעולם לא כתב"}])
    assert _decide(g, _pack()) == "REVIEW"


def test_a_fabricated_span_does_not_satisfy_the_grounding_requirement():
    """Fabrication must not count as grounding — otherwise inventing a quote
    would be a way OUT of review."""
    v = validate_evidence(
        credited=[CreditedItem("R1", "משפט שהתלמיד מעולם לא כתב")],
        transcription=TRANSCRIPTION, policy="required", credit_awarded=True)
    assert v.fabricated == ["R1"] and not v.verified and not v.ok


def test_normalization_still_accepts_only_protocol_differences():
    """The audit of the 2026-08-25 DEV run found ZERO normalization-caused
    failures, so the normalizer was deliberately left alone. This pins that it
    tolerates quoting/whitespace noise and nothing more."""
    from autograder.evidence import evidence_supported

    assert evidence_supported(f'  "{SPAN}"  ', TRANSCRIPTION)
    assert evidence_supported(SPAN.replace(" ", "  "), TRANSCRIPTION)
    assert not evidence_supported("מתיחה סוג של יש", TRANSCRIPTION), "reordering is not support"
    assert not evidence_supported(SPAN[:-3] + "אחר", TRANSCRIPTION), "altered letters are not support"


# --------------------------------------- 6. production == benchmark semantics


def test_production_and_benchmark_share_one_validator():
    """The benchmark adapter must not grow a second opinion about validity."""
    import inspect

    from autograder.benchmark.roles import GradeAdapter

    src = inspect.getsource(GradeAdapter.score)
    assert "validate_grade(" in src
    assert "validate_evidence" not in src, "the adapter must not re-implement evidence validation"


def test_the_adapter_decision_matches_validate_grade(monkeypatch):
    from autograder.benchmark.manifests import BenchCase
    from autograder.benchmark.roles import GradeAdapter

    pack_dict = {
        "question_id": "1", "question_text": "q", "question_type": "matching_with_explanation",
        "max_score": 4.0, "rubric": [], "rubric_items": [
            {"id": "R1", "text": "explains", "requires_evidence": True, "points": None,
             "excludes": [], "requires": [], "kind": "semantic"}],
        "scoring_rules": [], "grading_policy": "choice_and_explanation_independent",
        "official_solution": {}, "evidence_policy": "required", "correct_by_version": {},
    }
    case = BenchCase(case_id="c1", split="DEV", component="ALL",
                     inputs={"case_id": "c1", "pack": pack_dict, "selected": None,
                             "transcription": TRANSCRIPTION, "version": None},
                     label={"explanation_verdict": "valid",
                            "explanation_verdict_derivable": True}, meta={})
    a = GradeAdapter()
    ungrounded = a.score(case, {"score": 4.0, "uncertain": False,
                                "rubric_items": [{"id": "R1", "met": False}]}, None)
    assert ungrounded["decision"] == "REVIEW"
    assert ungrounded["evidence_ungrounded_credit"] is True
    assert ungrounded["evidence_failure"] is True

    grounded = a.score(case, {"score": 4.0, "uncertain": False,
                              "rubric_items": [{"id": "R1", "met": True,
                                                "student_evidence": SPAN}]}, None)
    assert grounded["decision"] == "AUTO"
    assert grounded["evidence_failure"] is False
    assert grounded["evidence_verified"] == ["R1"]


def test_evidence_failure_no_longer_counts_verbosity_as_a_grounding_problem():
    """`evidence exceeds length limit` is verbosity. The old substring test on
    "evidence" counted it as a grounding failure and inflated one candidate's
    count by 3 in the DEV run."""
    from autograder.benchmark.manifests import BenchCase
    from autograder.benchmark.roles import GradeAdapter

    pack_dict = {
        "question_id": "1", "question_text": "q", "question_type": "matching_with_explanation",
        "max_score": 4.0, "rubric": [], "rubric_items": [], "scoring_rules": [],
        "grading_policy": "choice_and_explanation_independent", "official_solution": {},
        "evidence_policy": "required", "correct_by_version": {},
    }
    case = BenchCase(case_id="c1", split="DEV", component="ALL",
                     inputs={"case_id": "c1", "pack": pack_dict, "selected": None,
                             "transcription": TRANSCRIPTION, "version": None},
                     label={}, meta={})
    row = GradeAdapter().score(
        case, {"score": 0.0, "uncertain": False, "evidence": "x" * 400}, None)
    assert row["decision"] == "REVIEW"                     # still flagged...
    assert any("exceeds length" in p for p in row["validation_problems"])
    assert row["evidence_failure"] is False                # ...but not as GROUNDING
