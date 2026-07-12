"""Offline tests for the deterministic grading core.

The fixtures replicate the structures of the sample exam in sample_data/:
- a matching question with required explanations (8 items x 4 pts, max 32),
- a 20-item multiple-choice question at 2 pts each capped at 36,
- three exam versions (A1/A2/A3) with per-version answers,
- one MC item accepting two answers, one MC item with version-dependent answers.
"""

import pytest

from autograder.config import GraderConfig
from autograder.grade import (
    VersionDecision,
    detect_version,
    grade_exam,
    normalize_answer,
)
from autograder.schema import (
    AnswerKey,
    ConventionNote,
    ExamExtraction,
    ExamSurvey,
    ExplanationEvaluation,
    KeyQuestion,
    KeySubItem,
    PageInfo,
    QuestionExtraction,
    SubItemExtraction,
)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def make_key() -> AnswerKey:
    q1_answers = {
        # sub_item -> per-version correct letter
        "1": {"A1": ["F"], "A2": ["F"], "A3": ["G"]},
        "2": {"A1": ["G"], "A2": ["G"], "A3": ["F"]},
        "3": {"A1": ["D"], "A2": ["E"], "A3": ["I"]},
        "4": {"A1": ["H"], "A2": ["A"], "A3": ["E"]},
        "5": {"A1": ["C"], "A2": ["I"], "A3": ["C"]},
        "6": {"A1": ["I"], "A2": ["C"], "A3": ["D"]},
        "7": {"A1": ["A"], "A2": ["H"], "A3": ["B"]},
        "8": {"A1": ["E"], "A2": ["B"], "A3": ["H"]},
    }
    q1 = KeyQuestion(
        id="1",
        title="Match operations to wavelet pyramids",
        type="matching_with_explanation",
        max_points=32,
        explanation_required=True,
        sub_items=[
            KeySubItem(
                id=i,
                prompt=f"operation {i}",
                correct_by_version=q1_answers[i],
                points=4,
                reference_explanation=f"reference reasoning for op {i}",
            )
            for i in q1_answers
        ],
    )

    q3_items = []
    for i in range(1, 21):
        if i == 1:
            correct = {"A1": ["A", "B"], "A2": ["A", "B"], "A3": ["A", "B"]}
        elif i == 16:
            correct = {"A1": ["C"], "A2": ["B"], "A3": ["C"]}
        else:
            correct = {"A1": ["B"], "A2": ["B"], "A3": ["B"]}
        q3_items.append(
            KeySubItem(id=str(i), prompt=f"MC question {i}", correct_by_version=correct, points=2)
        )
    q3 = KeyQuestion(
        id="3",
        title="Multiple choice",
        type="multiple_choice",
        max_points=36,  # 20 x 2 = 40 capped at 36
        sub_items=q3_items,
        answer_source="separate answer table only",
    )

    return AnswerKey(
        exam_title="Image processing exam",
        versions=["A1", "A2", "A3"],
        questions=[q1, q3],
        total_points=68,
        general_rules=["no credit without explanation on question 1"],
    )


def answered(sub_id: str, answer: str, explanation: str | None = None, confidence: float = 1.0):
    return SubItemExtraction(
        sub_item_id=sub_id,
        status="answered",
        final_answer=answer,
        explanation_transcription=explanation,
        explanation_legibility="full" if explanation else "none",
        interpretation_rationale="clean single mark",
        confidence=confidence,
    )


def make_extraction(q1_answers: dict[str, str], q3_answers: dict[str, str]) -> ExamExtraction:
    return ExamExtraction(
        questions=[
            QuestionExtraction(
                question_id="1",
                source_pages=[2, 3, 12],
                authoritative_source="answer table (retitled by student)",
                sub_items=[
                    answered(i, a, explanation=f"student explanation {i}")
                    for i, a in q1_answers.items()
                ],
            ),
            QuestionExtraction(
                question_id="3",
                source_pages=[13],
                authoritative_source="bubble table, X convention",
                sub_items=[answered(i, a) for i, a in q3_answers.items()],
            ),
        ]
    )


A1_Q1 = {"1": "F", "2": "G", "3": "D", "4": "H", "5": "C", "6": "I", "7": "A", "8": "E"}
A1_Q3 = {str(i): "B" for i in range(1, 21)}
A1_Q3["16"] = "C"


def valid_judgements(extraction: ExamExtraction) -> dict:
    """Every transcribed explanation judged valid."""
    out: dict[str, dict[str, ExplanationEvaluation]] = {}
    for q in extraction.questions:
        evals = {}
        for s in q.sub_items:
            verdict = "valid" if s.explanation_transcription else "missing"
            evals[s.sub_item_id] = ExplanationEvaluation(
                sub_item_id=s.sub_item_id, verdict=verdict, reasoning="test"
            )
        out[q.question_id] = evals
    return out


def config() -> GraderConfig:
    return GraderConfig()


# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------


def test_normalize_hebrew_letters():
    assert normalize_answer("א") == "A"
    assert normalize_answer("ב") == "B"
    assert normalize_answer("ג") == "C"
    assert normalize_answer("ד") == "D"


def test_normalize_latin_and_noise():
    assert normalize_answer(" c. ") == "C"
    assert normalize_answer("F") == "F"
    assert normalize_answer("") is None
    assert normalize_answer(None) is None


# --------------------------------------------------------------------------
# version detection
# --------------------------------------------------------------------------


def test_detect_version_picks_best_match():
    key = make_key()
    extraction = make_extraction(A1_Q1, A1_Q3)
    decision = detect_version(key, extraction, config())
    assert decision.version == "A1"
    assert not decision.uncertain


def test_detect_version_flags_uncertainty_on_small_margin():
    key = make_key()
    # Only sub-items whose answers agree across versions -> zero margin.
    extraction = make_extraction(
        {"1": "F", "2": "G"},  # F/G are correct in both A1 and A2
        {"1": "A"},
    )
    decision = detect_version(key, extraction, config())
    assert decision.uncertain


def test_detect_version_forced_by_user():
    key = make_key()
    cfg = GraderConfig(version="A2")
    extraction = make_extraction(A1_Q1, A1_Q3)
    decision = detect_version(key, extraction, cfg)
    assert decision.version == "A2"
    assert not decision.uncertain


def test_detect_version_rejects_unknown_version():
    key = make_key()
    cfg = GraderConfig(version="B9")
    with pytest.raises(ValueError):
        detect_version(key, make_extraction(A1_Q1, A1_Q3), cfg)


def test_detect_version_single_version_key():
    key = make_key()
    key.versions = ["default"]
    for q in key.questions:
        for s in q.sub_items:
            s.correct_by_version = {"default": next(iter(s.correct_by_version.values()))}
    decision = detect_version(key, make_extraction(A1_Q1, A1_Q3), config())
    assert decision.version == "default"


# --------------------------------------------------------------------------
# matching question with required explanations
# --------------------------------------------------------------------------


def graded(key, extraction, judgements, version="A1"):
    return grade_exam(
        key,
        extraction,
        judgements,
        VersionDecision(version, "test", False),
        config(),
    )


def test_perfect_exam_full_marks():
    key = make_key()
    extraction = make_extraction(A1_Q1, A1_Q3)
    result = graded(key, extraction, valid_judgements(extraction))
    q1 = next(q for q in result.questions if q.question_id == "1")
    q3 = next(q for q in result.questions if q.question_id == "3")
    assert q1.points_awarded == 32
    # 20 correct x 2 = 40, capped at 36
    assert q3.points_awarded == 36
    assert q3.capped
    assert result.total_awarded == 68


def test_correct_selection_without_explanation_earns_nothing():
    key = make_key()
    extraction = make_extraction(A1_Q1, A1_Q3)
    judgements = valid_judgements(extraction)
    judgements["1"]["3"] = ExplanationEvaluation(
        sub_item_id="3", verdict="missing", reasoning="nothing written"
    )
    result = graded(key, extraction, judgements)
    q1 = next(q for q in result.questions if q.question_id == "1")
    item3 = next(s for s in q1.sub_results if s.sub_item_id == "3")
    assert item3.selection_correct
    assert item3.points_total == 0
    assert "without a valid explanation" in item3.reason
    assert q1.points_awarded == 28


def test_partially_valid_explanation_gets_partial_credit():
    key = make_key()
    extraction = make_extraction(A1_Q1, A1_Q3)
    judgements = valid_judgements(extraction)
    judgements["1"]["5"] = ExplanationEvaluation(
        sub_item_id="5", verdict="partially_valid", reasoning="half right"
    )
    result = graded(key, extraction, judgements)
    q1 = next(q for q in result.questions if q.question_id == "1")
    item5 = next(s for s in q1.sub_results if s.sub_item_id == "5")
    assert item5.points_total == 2  # 4 * 0.5


def test_wrong_selection_with_explanation_matching_other_option_is_flagged():
    """The swapped-answers pattern: letters reversed in the table but the
    explanations correctly justify the other option. Must not silently award
    or deny — flag for human review."""
    key = make_key()
    q1_answers = dict(A1_Q1)
    q1_answers["5"], q1_answers["6"] = q1_answers["6"], q1_answers["5"]  # swap I and C
    extraction = make_extraction(q1_answers, A1_Q3)
    judgements = valid_judgements(extraction)
    judgements["1"]["5"] = ExplanationEvaluation(
        sub_item_id="5",
        verdict="valid",
        reasoning="explanation describes the other image",
        explanation_matches_different_answer="C",
    )
    result = graded(key, extraction, judgements)
    q1 = next(q for q in result.questions if q.question_id == "1")
    item5 = next(s for s in q1.sub_results if s.sub_item_id == "5")
    assert item5.selection_correct is False
    assert item5.points_total == 0
    assert item5.needs_review
    assert any(
        r.question_id == "1" and r.sub_item_id == "5" for r in result.needs_human_review
    )


def test_explanation_split_weight():
    key = make_key()
    q1 = key.question("1")
    q1.explanation_weight = 0.5  # rubric that splits points between components
    extraction = make_extraction(A1_Q1, A1_Q3)
    judgements = valid_judgements(extraction)
    judgements["1"]["2"] = ExplanationEvaluation(
        sub_item_id="2", verdict="invalid", reasoning="circular"
    )
    result = graded(key, extraction, judgements)
    q1r = next(q for q in result.questions if q.question_id == "1")
    item2 = next(s for s in q1r.sub_results if s.sub_item_id == "2")
    assert item2.points_selection == 2
    assert item2.points_explanation == 0
    assert item2.points_total == 2


# --------------------------------------------------------------------------
# multiple choice
# --------------------------------------------------------------------------


def test_mc_accepted_alternatives():
    key = make_key()
    q3_answers = dict(A1_Q3)
    q3_answers["1"] = "A"  # item 1 accepts both A and B
    extraction = make_extraction(A1_Q1, q3_answers)
    result = graded(key, extraction, valid_judgements(extraction))
    q3 = next(q for q in result.questions if q.question_id == "3")
    item1 = next(s for s in q3.sub_results if s.sub_item_id == "1")
    assert item1.selection_correct


def test_mc_version_dependent_answer():
    key = make_key()
    extraction = make_extraction(A1_Q1, A1_Q3)  # item 16 answered C
    r_a1 = graded(key, extraction, valid_judgements(extraction), version="A1")
    r_a2 = graded(key, extraction, valid_judgements(extraction), version="A2")
    item16_a1 = next(
        s
        for q in r_a1.questions
        if q.question_id == "3"
        for s in q.sub_results
        if s.sub_item_id == "16"
    )
    item16_a2 = next(
        s
        for q in r_a2.questions
        if q.question_id == "3"
        for s in q.sub_results
        if s.sub_item_id == "16"
    )
    assert item16_a1.selection_correct
    assert not item16_a2.selection_correct


def test_mc_cap_not_applied_below_cap():
    key = make_key()
    q3_answers = dict(A1_Q3)
    for i in ("2", "3", "4"):
        q3_answers[i] = "D"  # three wrong -> 17 x 2 = 34 < 36
    extraction = make_extraction(A1_Q1, q3_answers)
    result = graded(key, extraction, valid_judgements(extraction))
    q3 = next(q for q in result.questions if q.question_id == "3")
    assert q3.points_awarded == 34
    assert not q3.capped


# --------------------------------------------------------------------------
# unanswered / ambiguous
# --------------------------------------------------------------------------


def test_unanswered_and_ambiguous_are_distinct():
    key = make_key()
    extraction = make_extraction(A1_Q1, A1_Q3)
    q3 = extraction.question("3")
    q3.sub_items[4] = SubItemExtraction(
        sub_item_id="5",
        status="unanswered",
        interpretation_rationale="no mark in the table row",
        confidence=1.0,
    )
    q3.sub_items[5] = SubItemExtraction(
        sub_item_id="6",
        status="ambiguous",
        candidate_answers=["B", "C"],
        interpretation_rationale="two uncancelled marks",
        confidence=0.4,
    )
    result = graded(key, extraction, valid_judgements(extraction))
    q3r = next(q for q in result.questions if q.question_id == "3")
    item5 = next(s for s in q3r.sub_results if s.sub_item_id == "5")
    item6 = next(s for s in q3r.sub_results if s.sub_item_id == "6")

    assert item5.status == "unanswered"
    assert item5.points_total == 0
    assert item5.selection_correct is None
    assert not item5.needs_review

    assert item6.status == "ambiguous"
    assert item6.points_total == 0
    assert item6.selection_correct is None
    assert item6.needs_review

    assert any(r.sub_item_id == "5" for r in result.unanswered)
    assert any(r.sub_item_id == "6" for r in result.needs_human_review)
    assert not any(r.sub_item_id == "6" for r in result.unanswered)


def test_low_confidence_extraction_is_flagged():
    key = make_key()
    extraction = make_extraction(A1_Q1, A1_Q3)
    extraction.question("3").sub_items[0] = answered("1", "B", confidence=0.5)
    result = graded(key, extraction, valid_judgements(extraction))
    assert any(
        r.question_id == "3" and r.sub_item_id == "1" for r in result.needs_human_review
    )


def test_uncertain_version_adds_global_review_item():
    key = make_key()
    extraction = make_extraction(A1_Q1, A1_Q3)
    result = grade_exam(
        extraction=extraction,
        key=key,
        judgements=valid_judgements(extraction),
        version_decision=VersionDecision("A1", "margin too small", True),
        config=config(),
    )
    assert any(r.question_id == "*" for r in result.needs_human_review)


# --------------------------------------------------------------------------
# survey-driven mark interpretation record
# --------------------------------------------------------------------------


def test_mark_interpretations_include_conventions():
    key = make_key()
    extraction = make_extraction(A1_Q1, A1_Q3)
    survey = ExamSurvey(
        pages=[PageInfo(page_number=1, content_summary="cover")],
        marking_conventions=[
            ConventionNote(
                page_number=13,
                verbatim_text="סימון לזה X",
                interpretation="X marks are the final answers; circles are drafts",
                scope="answer table on page 13",
            )
        ],
        student_ink_description="blue pen",
        grader_annotations_description="red ink ticks and scores",
        authoritative_answer_locations=["Q3: table on page 13"],
    )
    result = grade_exam(
        key,
        extraction,
        valid_judgements(extraction),
        VersionDecision("A1", "test", False),
        config(),
        survey=survey,
    )
    joined = "\n".join(result.mark_interpretations)
    assert "סימון לזה X" in joined
    assert "red ink" in joined
