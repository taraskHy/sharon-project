"""grade-v4-charitable: grade the MEANING, not the wording — without inflating.

A blinded five-case human audit of the DEV cases where every candidate scored
below the label returned A=3 / B=2: three said the label was right and the
models were simply too strict, two said instructor practice is more lenient
than the encoded rubric. Both readings point the same way — v3 graded more
literally than the person whose grades are the ground truth.

The danger in the fix is obvious: "be charitable" degenerates into "award more
points". So the policy is split down one line, and these tests pin both halves:

    charity applies to how a correct idea is EXPRESSED
    never    to whether the idea is PRESENT

Everything credited must still be in the student's own text, quoted.

No provider is contacted anywhere in this file.
"""

from __future__ import annotations

import re

import pytest

from autograder.benchmark.roles import GradeAdapter
from autograder.escalation import (ACTIVE_GRADE_PROMPT_VERSION, GRADE_SYSTEM,
                                   GRADE_SYSTEM_BY_VERSION, GRADE_SYSTEM_V3,
                                   GRADE_SYSTEM_V4_CHARITABLE, explanation_scale,
                                   grade_prompt, grade_system_for)

V4 = "grade-v4-charitable"
SYS = GRADE_SYSTEM_V4_CHARITABLE.lower()


def _pack(max_score=4.0):
    from autograder.benchmark.roles import pack_from_inputs

    return pack_from_inputs({
        "question_id": "1", "question_text": "q", "question_type": "matching_with_explanation",
        "max_score": max_score, "rubric": ["explain the effect"], "rubric_items": [],
        "scoring_rules": ["explanation weight 0"],
        "grading_policy": "choice_and_explanation_independent",
        "official_solution": {"1": "the reference wording"},
        "evidence_policy": "required", "correct_by_version": {}})


def _body(**kw):
    return grade_prompt(_pack(), selected=None, transcription="הסבר כלשהו",
                        version=None, prompt_version=V4, **kw)[0]["text"]


def _has(*phrases):
    return all(p.lower() in SYS for p in phrases)


# --------------------------------------------- 1-4. what must be ACCEPTED ----


def test_semantically_equivalent_paraphrases_are_accepted():
    assert _has("grade the meaning, not the wording")
    assert _has("accept paraphrases")
    assert _has("expressed quite differently from the official")
    assert _has("not a template the student must match")


def test_informal_terminology_is_not_penalised():
    assert _has("informal or imprecise terminology")
    assert "do not require the official solution's exact vocabulary" in SYS


def test_grammar_and_spelling_errors_do_not_reduce_the_verdict():
    assert _has("imperfect grammar, spelling and typing mistakes")
    # and explicitly excluded as grounds for zero
    zero = SYS.split("zero explanation quality")[1].split("borderline")[0]
    assert "grammar or spelling is poor" in zero


def test_a_concise_answer_can_be_full():
    assert "a concise answer can earn full quality" in SYS
    assert _has("short explanations")


# ------------------------------------------ 5-7. the partial / full border ----


def test_a_correct_direction_but_incomplete_answer_is_partial_not_zero():
    part = SYS.split("partial explanation quality")[1].split("zero explanation quality")[0]
    assert "right direction but is too vague for full credit" in part
    assert "should normally receive at least partial quality rather than none" in part


def test_missing_nonessential_details_do_not_prevent_full():
    assert "omits secondary details the rubric does not make essential" in SYS
    assert _has("every intermediate step", "every possible consequence")


def test_an_explicitly_essential_rubric_requirement_still_gates_full_credit():
    """Charity is bounded by the rubric: 'the rubric does not make essential'
    is a conditional, so a requirement the rubric DOES make essential still
    binds. And a letter/option is required when a rubric item demands it."""
    assert "the rubric does not make essential" in SYS
    assert "unless a rubric item explicitly demands it" in SYS


# ------------------------------ 8-9. what must STILL be refused (no inflation) ----


def test_a_wrong_direction_or_contradictory_answer_stays_zero():
    zero = SYS.split("zero explanation quality")[1].split("borderline")[0]
    for phrase in ("wrong direction", "opposite effect", "materially contradicts",
                   "irrelevant to the question"):
        assert phrase in zero, phrase


def test_a_generic_statement_with_no_task_specific_content_is_not_promoted():
    """The single most likely failure mode of a charitable policy."""
    assert "so general that it would fit almost any unrelated problem" in SYS
    assert "must say something specific to this question to earn credit" in SYS


def test_restating_the_question_earns_nothing():
    zero = SYS.split("zero explanation quality")[1].split("borderline")[0]
    assert "restates the question without making a claim" in zero


def test_credit_may_not_be_invented():
    zero = SYS.split("zero explanation quality")[1].split("borderline")[0]
    assert "inventing content the student did not write" in zero


# -------------------------------------------- 10-11. the borderline rule ----


def test_borderline_favours_the_higher_verdict():
    b = SYS.split("borderline:")[1].split("never supply")[0]
    assert "adjacent levels, choose the higher one" in b


def test_but_only_when_the_students_own_text_supports_it():
    b = SYS.split("borderline:")[1].split("never supply")[0]
    assert "only when the higher level is supported by something the student genuinely wrote" in b


def test_the_model_may_not_infer_unstated_content_from_solution_or_rag():
    n = SYS.split("never supply the missing reasoning yourself")[1]
    for source in ("official solution", "the rubric", "course context",
                   "general domain knowledge", "what the student probably intended"):
        assert source in n, source
    assert "charity applies to how an idea is expressed, never to whether it is present" in n


# --------------------------------------- 12-14. evidence + REVIEW unchanged ----


def test_positive_credit_still_requires_an_exact_student_span():
    e = SYS.split("evidence.")[1]
    assert "whenever you award any credit above zero" in e
    assert "copied verbatim from the student transcription" in e
    assert "leniency never relaxes this" in e


def test_evidence_may_not_be_taken_from_the_solution_or_course_context():
    e = SYS.split("evidence.")[1]
    assert "a span from the official solution or the course context does not count" in e


def test_empty_required_evidence_still_routes_to_review():
    """The fail-closed rule is unchanged by the policy change."""
    from autograder.escalation import GradeResult, validate_grade
    from autograder.gradingpack import QuestionGradingPack, RubricItemSpec

    pack = QuestionGradingPack(
        question_id="1", question_text="q", question_type="matching_with_explanation",
        max_score=4.0, rubric=[], scoring_rules=[],
        rubric_items=[RubricItemSpec(id="R1", text="explains", requires_evidence=True)],
        grading_policy="choice_and_explanation_independent", official_solution={},
        correct_by_version={}, evidence_policy="required")
    v = validate_grade(GradeResult(score=4.0, rubric_items=[{"id": "R1", "met": True,
                                                             "student_evidence": ""}]),
                       pack, selection_correct=None, selected=None, transcription="טקסט כלשהו")
    assert not v.ok


def test_unreadable_transcription_causes_uncertainty_not_invented_credit():
    assert "materially unreadable or incomplete" in SYS
    assert "do not reconstruct an unreadable answer charitably" in SYS
    assert "say you are uncertain instead" in SYS


def test_the_normalizer_was_not_loosened():
    """The audit found zero normalization mismatches; the failures were empty
    citations. Loosening a check that was never the cause would only weaken it."""
    from autograder.evidence import evidence_supported

    t = "ניתן לראות שיש סוג של מתיחה"
    assert evidence_supported('  "יש סוג של מתיחה"  ', t)
    assert not evidence_supported("מתיחה של סוג יש", t)


# ------------------------------------ 15. production / benchmark parity ------


def test_production_and_benchmark_use_the_identical_prompt():
    assert GradeAdapter.prompt_version == V4 == ACTIVE_GRADE_PROMPT_VERSION
    assert grade_system_for(GradeAdapter.prompt_version) is GRADE_SYSTEM
    assert GRADE_SYSTEM is GRADE_SYSTEM_V4_CHARITABLE


def test_both_halves_of_the_prompt_come_from_the_declared_version():
    from autograder.benchmark.manifests import load_manifest

    m = load_manifest("grade_primary")
    case = next(c for c in m.cases if c.split == "DEV")
    req = GradeAdapter().build_request(dict(case.inputs), None)
    assert req.prompt_version == V4
    assert req.system is grade_system_for(V4)
    assert "= full:" in req.content_blocks[0]["text"]


def test_v3_is_preserved_verbatim_and_still_reproducible():
    assert grade_system_for("grade-v3") is GRADE_SYSTEM_V3
    assert GRADE_SYSTEM_V3 is not GRADE_SYSTEM_V4_CHARITABLE
    assert set(GRADE_SYSTEM_BY_VERSION) == {"grade-v3", V4}
    # the v3 user-block scale wording is preserved too
    v3_scale = explanation_scale(4.0, "grade-v3")
    assert "= valid: correct and sufficient reasoning" in v3_scale
    assert v3_scale != explanation_scale(4.0, V4)


def test_an_unknown_prompt_version_is_refused():
    with pytest.raises(ValueError, match="unknown grading prompt version"):
        grade_system_for("grade-v9-imaginary")


def test_the_prompt_change_moves_the_run_identity():
    """A changed prompt under an unchanged hash would make two different
    experiments look interchangeable."""
    import hashlib

    h3 = hashlib.sha256(GRADE_SYSTEM_V3.encode("utf-8")).hexdigest()
    h4 = hashlib.sha256(GRADE_SYSTEM_V4_CHARITABLE.encode("utf-8")).hexdigest()
    assert h3 != h4


# ---------------------------- 16-17. leakage + no benchmark specifics --------


def test_leakage_checks_pass_across_the_complete_dataset():
    from autograder.benchmark.manifests import load_manifest
    from autograder.benchmark.runner import leakage_check

    a = GradeAdapter()
    m = load_manifest("grade_primary")
    assert len(m.cases) == 67
    for case in m.cases:
        leakage_check(case, a.build_request(dict(case.inputs), None), a.model_visible_fields)


#: things that would tie the prompt to THIS benchmark instead of to grading
_BENCHMARK_SPECIFIC = [
    r"\be\d{3}_q\d+_r\d+\b",          # any case id
    r"\bhistogram\b", r"\bmotion blur\b", r"\bgaussian\b", r"\bpixel\b",
    r"\bwavelet\b", r"\bfourier\b", r"\bblur\b", r"\bimage [a-h]\b",
    r"\berik\b", r"\binstructor score\b", r"\bgemini\b", r"\bsonnet\b",
    r"\bluna\b", r"\bopenrouter\b", r"\bDEV\b",
]


@pytest.mark.parametrize("pattern", _BENCHMARK_SPECIFIC)
def test_the_prompt_contains_nothing_benchmark_specific(pattern):
    """The prompt must apply to a new exam in a new subject unchanged."""
    assert not re.search(pattern, GRADE_SYSTEM_V4_CHARITABLE, re.I), \
        f"{pattern!r} appears in the grading prompt"


def test_no_real_student_text_or_dev_answer_appears_in_the_prompt():
    from autograder.benchmark.manifests import load_manifest

    m = load_manifest("grade_primary")
    for case in m.cases:
        text = (case.inputs.get("transcription") or "").strip()
        if len(text) >= 12:
            assert text not in GRADE_SYSTEM_V4_CHARITABLE, case.case_id


def test_the_prompt_carries_no_worked_example():
    """A worked example copied from DEV would tune the prompt to the benchmark."""
    assert "for example" not in SYS
    assert "e.g." not in SYS
    # no Hebrew at all: every DEV answer is Hebrew, so any Hebrew character
    # would be a strong sign a real answer leaked in
    assert not re.search(r"[֐-׿]", GRADE_SYSTEM_V4_CHARITABLE)


def test_the_scale_text_is_generic_too():
    body = _body()
    assert not re.search(r"[֐-׿]", body.split("Student explanation")[0]
                         .replace(_pack().question_text, ""))
    assert not re.search(r"\be\d{3}_q\d+_r\d+\b", body)
