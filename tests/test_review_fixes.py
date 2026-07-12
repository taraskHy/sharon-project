"""Regression tests for defects found by the adversarial review pass."""

from pathlib import Path

import pytest

from autograder.config import GraderConfig
from autograder.extract import _canon_id, _pages_for_question, _reconcile_sub_items
from autograder.grade import (
    PipelineStateError,
    VersionDecision,
    detect_version,
    grade_exam,
    judge_question,
)
from autograder.ingest import _natural_key
from autograder.schema import (
    AnswerKey,
    ConventionNote,
    ExamExtraction,
    ExamSurvey,
    KeyQuestion,
    KeySubItem,
    PageInfo,
    QuestionExtraction,
    SubItemExtraction,
)
from tests.test_grade import (
    A1_Q1,
    A1_Q3,
    answered,
    config,
    graded,
    make_extraction,
    make_key,
    valid_judgements,
)


# --------------------------------------------------------------------------
# illegible explanation without transcription must route to review, not zero
# --------------------------------------------------------------------------


def test_illegible_untranscribed_explanation_is_flagged_not_zeroed_silently():
    key = make_key()
    q1 = key.question("1")
    ext_q = QuestionExtraction(
        question_id="1",
        source_pages=[12],
        authoritative_source="answer table",
        sub_items=[
            SubItemExtraction(
                sub_item_id="1",
                status="answered",
                final_answer="F",
                explanation_transcription=None,
                explanation_legibility="illegible",
                interpretation_rationale="clear letter, unreadable justification",
                confidence=0.9,
            )
        ],
    )
    # no LLM call is needed: nothing has a transcription
    evaluations = judge_question(None, q1, ext_q, "A1")
    assert evaluations["1"].verdict == "illegible"

    extraction = make_extraction(A1_Q1, A1_Q3)
    extraction.question("1").sub_items[0] = ext_q.sub_items[0]
    judgements = valid_judgements(extraction)
    judgements["1"]["1"] = evaluations["1"]
    result = graded(key, extraction, judgements)
    item1 = next(
        s
        for q in result.questions
        if q.question_id == "1"
        for s in q.sub_results
        if s.sub_item_id == "1"
    )
    assert item1.points_total == 0
    assert item1.needs_review
    assert any(
        r.question_id == "1" and r.sub_item_id == "1" for r in result.needs_human_review
    )


def test_truly_missing_explanation_still_judged_missing():
    key = make_key()
    q1 = key.question("1")
    ext_q = QuestionExtraction(
        question_id="1",
        source_pages=[12],
        authoritative_source="answer table",
        sub_items=[
            SubItemExtraction(
                sub_item_id="1",
                status="answered",
                final_answer="F",
                explanation_legibility="none",
                interpretation_rationale="letter only, nothing written",
                confidence=1.0,
            )
        ],
    )
    evaluations = judge_question(None, q1, ext_q, "A1")
    assert evaluations["1"].verdict == "missing"


# --------------------------------------------------------------------------
# empty accepted-answer set is a key defect -> review, not "incorrect"
# --------------------------------------------------------------------------


def test_missing_version_entry_flags_review_instead_of_marking_wrong():
    key = make_key()
    q3 = key.question("3")
    del q3.sub_items[1].correct_by_version["A1"]  # item "2" now lacks A1
    extraction = make_extraction(A1_Q1, A1_Q3)
    result = graded(key, extraction, valid_judgements(extraction))
    item2 = next(
        s
        for q in result.questions
        if q.question_id == "3"
        for s in q.sub_results
        if s.sub_item_id == "2"
    )
    assert item2.selection_correct is None
    assert item2.needs_review
    assert "no accepted answers" in item2.reason


def test_version_detection_skips_partially_covered_sub_items():
    key = make_key()
    # Sub-item 1 of Q3 loses its A2/A3 entries; detection must not let A1
    # score a free point on it.
    q3 = key.question("3")
    q3.sub_items[0].correct_by_version = {"A1": ["B"]}
    extraction = make_extraction(A1_Q1, A1_Q3)
    decision = detect_version(key, extraction, config())
    assert decision.version == "A1"  # still wins on fully covered items


# --------------------------------------------------------------------------
# open questions score via the judged explanation
# --------------------------------------------------------------------------


def test_open_question_scores_from_judged_explanation():
    key = make_key()
    key.questions.append(
        KeyQuestion(
            id="4",
            title="Open question",
            type="open",
            max_points=10,
            sub_items=[
                KeySubItem(
                    id="1",
                    prompt="explain X",
                    correct_by_version={"A1": [], "A2": [], "A3": []},
                    points=10,
                    reference_explanation="the right answer explains X via Y",
                )
            ],
        )
    )
    key.total_points = 78
    extraction = make_extraction(A1_Q1, A1_Q3)
    extraction.questions.append(
        QuestionExtraction(
            question_id="4",
            source_pages=[9],
            authoritative_source="page 9",
            sub_items=[
                SubItemExtraction(
                    sub_item_id="1",
                    status="answered",
                    final_answer=None,
                    explanation_transcription="X happens because of Y",
                    explanation_legibility="full",
                    interpretation_rationale="written answer",
                    confidence=0.95,
                )
            ],
        )
    )
    judgements = valid_judgements(extraction)
    result = graded(key, extraction, judgements)
    q4 = next(q for q in result.questions if q.question_id == "4")
    assert q4.points_awarded == 10
    item = q4.sub_results[0]
    assert item.points_explanation == 10
    assert item.points_selection == 0
    assert item.selection_correct is None


# --------------------------------------------------------------------------
# reconciliation: duplicates and id mismatches
# --------------------------------------------------------------------------


def _q(sub_ids):
    return KeyQuestion(
        id="1",
        title="t",
        type="multiple_choice",
        max_points=4,
        sub_items=[
            KeySubItem(id=i, prompt=f"q{i}", correct_by_version={"default": ["A"]}, points=2)
            for i in sub_ids
        ],
    )


def test_conflicting_duplicate_observations_become_ambiguous():
    ext = QuestionExtraction(
        question_id="1",
        source_pages=[1],
        authoritative_source="page 1",
        sub_items=[
            answered("1", "B"),
            answered("1", "C"),
            answered("2", "A"),
        ],
    )
    out = _reconcile_sub_items(_q(["1", "2"]), ext)
    item1 = next(s for s in out.sub_items if s.sub_item_id == "1")
    assert item1.status == "ambiguous"
    assert set(item1.candidate_answers) == {"B", "C"}
    assert item1.final_answer is None


def test_agreeing_duplicates_are_merged():
    ext = QuestionExtraction(
        question_id="1",
        source_pages=[1],
        authoritative_source="page 1",
        sub_items=[answered("1", "B"), answered("1", "B"), answered("2", "A")],
    )
    out = _reconcile_sub_items(_q(["1", "2"]), ext)
    item1 = next(s for s in out.sub_items if s.sub_item_id == "1")
    assert item1.status == "answered"
    assert item1.final_answer == "B"


def test_id_mismatch_zero_padding_still_matches():
    ext = QuestionExtraction(
        question_id="1",
        source_pages=[1],
        authoritative_source="page 1",
        sub_items=[answered("01", "B"), answered(" 2", "A")],
    )
    out = _reconcile_sub_items(_q(["1", "2"]), ext)
    assert [s.sub_item_id for s in out.sub_items] == ["1", "2"]
    assert out.sub_items[0].final_answer == "B"
    assert out.sub_items[0].status == "answered"


def test_canon_id():
    assert _canon_id("01") == _canon_id("1")
    assert _canon_id(" 2 ") == "2"
    assert _canon_id("0") == "0"


# --------------------------------------------------------------------------
# page selection fallback
# --------------------------------------------------------------------------


class FakePage:
    def __init__(self, n):
        self.page_number = n


def test_unplaced_question_gets_all_pages_despite_convention_notes():
    survey = ExamSurvey(
        pages=[
            PageInfo(page_number=1, content_summary="cover"),
            PageInfo(page_number=2, content_summary="q1", question_ids=["1"]),
        ],
        marking_conventions=[
            ConventionNote(
                page_number=3,
                verbatim_text="X is final",
                interpretation="X marks are final",
                scope="everywhere",
            )
        ],
        student_ink_description="blue",
        grader_annotations_description="",
    )
    pages = [FakePage(n) for n in (1, 2, 3, 4)]
    # question "9" was never placed by the survey -> must fall back to all pages
    got = _pages_for_question("9", survey, pages)
    assert [p.page_number for p in got] == [1, 2, 3, 4]
    # question "1" gets its own page plus the convention page
    got1 = _pages_for_question("1", survey, pages)
    assert [p.page_number for p in got1] == [2, 3]


# --------------------------------------------------------------------------
# stale/incomplete pipeline state fails loudly
# --------------------------------------------------------------------------


def test_missing_question_in_extraction_raises_pipeline_error():
    key = make_key()
    extraction = make_extraction(A1_Q1, A1_Q3)
    extraction.questions = [q for q in extraction.questions if q.question_id != "3"]
    with pytest.raises(PipelineStateError):
        grade_exam(
            key,
            extraction,
            valid_judgements(extraction),
            VersionDecision("A1", "test", False),
            config(),
        )


# --------------------------------------------------------------------------
# totals, unanswered reasons, versions guard
# --------------------------------------------------------------------------


def test_total_max_uses_per_question_sum_and_notes_discrepancy():
    key = make_key()
    key.total_points = 100  # key document claims a different total
    extraction = make_extraction(A1_Q1, A1_Q3)
    result = graded(key, extraction, valid_judgements(extraction))
    assert result.total_max == 68  # 32 + 36
    assert any("total_points=100" in note for note in result.mark_interpretations)


def test_unanswered_reason_carries_uncertainty():
    key = make_key()
    extraction = make_extraction(A1_Q1, A1_Q3)
    extraction.question("3").sub_items[0] = SubItemExtraction(
        sub_item_id="1",
        status="unanswered",
        interpretation_rationale="row looks empty",
        confidence=0.5,
        uncertainty_note="page corner is folded over the row",
    )
    result = graded(key, extraction, valid_judgements(extraction))
    item1 = next(
        s
        for q in result.questions
        if q.question_id == "3"
        for s in q.sub_results
        if s.sub_item_id == "1"
    )
    assert "folded" in item1.reason
    assert item1.needs_review


def test_empty_versions_list_defaults():
    key = make_key()
    data = key.model_dump()
    data["versions"] = []
    revived = AnswerKey.model_validate(data)
    assert revived.versions == ["default"]


# --------------------------------------------------------------------------
# natural page ordering for directory input
# --------------------------------------------------------------------------


def test_natural_sort_orders_page_10_after_page_2():
    names = ["page_10.png", "page_2.png", "page_1.png"]
    ordered = sorted((Path(n) for n in names), key=_natural_key)
    assert [p.name for p in ordered] == ["page_1.png", "page_2.png", "page_10.png"]
