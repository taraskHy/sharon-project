"""Regressions for the two defects found in the first real GPU run of the
representative exam (docs/validation/smoke-2026-07-13-strongpc-diagnosis.md):

1. The student SWAPPED the two answer tables; the low-resolution survey
   cannot see handwritten title corrections, so extraction graded Q1 against
   Q2's table (0/8 + 0/8). Fix: full-resolution close-read of the sheet
   pages reassigns each table to the question it actually serves.
2. A 20-row multiple-choice extraction in one call collapsed to the same
   letter for every row with an identical template rationale. Fix: chunked
   extraction (<=8 sub-items per call) plus a deterministic uniform-answer
   tripwire that routes to human review.
"""

from autograder.backends import BackendConfig
from autograder.backends.mock import MockBackend
from autograder.extract import (
    EXTRACTION_CHUNK_SIZE,
    _flag_uniform_collapse,
    _pages_for_question,
    extract_question,
)
from autograder.ingest import PageImage
from autograder.schema import (
    AnswerSheetPolicy,
    ConventionNote,
    ExamSurvey,
    KeyQuestion,
    KeySubItem,
    PageInfo,
    QuestionExtraction,
    SheetCloseRead,
    SheetPageReading,
    SubItemExtraction,
)
from autograder.survey import candidate_sheet_pages, closeread_sheets, merge_closeread
from tests.test_grade import make_key


def _survey_two_sheets() -> ExamSurvey:
    """Low-res survey view: sheets found, printed titles taken at face value."""
    return ExamSurvey(
        pages=[
            PageInfo(page_number=1, content_summary="q1 text", question_ids=["1"]),
            PageInfo(page_number=2, content_summary="q2 text", question_ids=["2"]),
            PageInfo(
                page_number=11,
                content_summary="answer sheet titled Question 1",
                page_kind="answer_sheet",
                is_answer_area=True,
                question_ids=["1"],
                answer_area_for_question="1",
            ),
            PageInfo(
                page_number=12,
                content_summary="answer sheet titled Question 2",
                page_kind="answer_sheet",
                is_answer_area=True,
                question_ids=["2"],
                answer_area_for_question="2",
            ),
        ],
        answer_sheet_policy=AnswerSheetPolicy(authoritative_pages=[11, 12]),
        student_ink_description="blue",
        grader_annotations_description="red scores",
    )


def _closeread_swapped() -> SheetCloseRead:
    """Full-res close-read: the student crossed out the printed titles and
    swapped the tables."""
    return SheetCloseRead(
        pages=[
            SheetPageReading(
                page_number=11,
                printed_title_question="1",
                serves_questions=["2"],
                correction_evidence="printed '1' crossed out, handwritten '2', note 'החלפתי בין הטבלאות'",
                sheet_condition="present",
            ),
            SheetPageReading(
                page_number=12,
                printed_title_question="2",
                serves_questions=["1"],
                correction_evidence="printed '2' crossed out, handwritten '1'",
                sheet_condition="present",
            ),
        ],
        marking_conventions=[
            ConventionNote(
                page_number=12,
                verbatim_text="תשובות המסומנות ב-X הן הסופיות",
                interpretation="X marks are the final answers; circles are drafts",
                scope="answer table on page 12",
            )
        ],
    )


# --------------------------------------------------------------------------
# close-read merge: the swapped-tables regression
# --------------------------------------------------------------------------


def test_closeread_swap_reroutes_extraction_pages():
    survey = merge_closeread(_survey_two_sheets(), _closeread_swapped())
    pages = [
        PageImage(page_number=n, png_bytes=b"png", width=5, height=5, text="")
        for n in (1, 2, 11, 12)
    ]
    # After the merge, question 1's answers are read from page 12 and
    # question 2's from page 11 — plus page 12 which carries the convention
    # note that governs reading.
    q1_pages = [p.page_number for p in _pages_for_question("1", survey, pages)]
    q2_pages = [p.page_number for p in _pages_for_question("2", survey, pages)]
    assert q1_pages == [12]
    assert q2_pages == [11, 12], "sheet page 11 plus the convention-note page 12"

    p11 = next(p for p in survey.pages if p.page_number == 11)
    p12 = next(p for p in survey.pages if p.page_number == 12)
    assert p11.answer_area_for_question == "2" and p11.question_ids == ["2"]
    assert p12.answer_area_for_question == "1" and p12.question_ids == ["1"]
    assert "crossed out" in p11.content_summary
    assert p11.sheet_condition == "present"


def test_closeread_merge_dedups_conventions_and_ignores_phantom_pages():
    survey = _survey_two_sheets()
    survey.marking_conventions.append(
        ConventionNote(
            page_number=12,
            verbatim_text="תשובות המסומנות ב-X הן הסופיות",
            interpretation="X final",
            scope="table p12",
        )
    )
    closeread = _closeread_swapped()
    closeread.pages.append(
        SheetPageReading(page_number=99, serves_questions=["3"], sheet_condition="present")
    )
    merged = merge_closeread(survey, closeread)
    texts = [n.verbatim_text for n in merged.marking_conventions]
    assert len(texts) == len(set(texts)) == 1, "identical note must not duplicate"
    assert all(p.page_number != 99 for p in merged.pages), "phantom page must be ignored"
    assert 99 not in merged.answer_sheet_policy.authoritative_pages


def test_closeread_skipped_when_no_sheets():
    survey = ExamSurvey(
        pages=[PageInfo(page_number=1, content_summary="q1", question_ids=["1"])],
        student_ink_description="pen",
        grader_annotations_description="",
    )
    assert candidate_sheet_pages(survey) == []
    backend = MockBackend(config=BackendConfig(backend="mock", model="m"))
    pages = [PageImage(page_number=1, png_bytes=b"p", width=5, height=5, text="")]
    assert closeread_sheets(backend, survey, pages, make_key()) is None
    assert backend.calls == [], "no model call when there is nothing to close-read"


# --------------------------------------------------------------------------
# chunked extraction: the 20-row collapse regression
# --------------------------------------------------------------------------


def _mc_question(n: int = 20) -> KeyQuestion:
    return KeyQuestion(
        id="3",
        title="MC",
        type="multiple_choice",
        max_points=36,
        sub_items=[
            KeySubItem(id=str(i), prompt=f"row {i}", correct_by_version={"default": ["B"]}, points=2)
            for i in range(1, n + 1)
        ],
    )


def _extraction_for_ids(qid: str, ids: list[str], letter_for) -> QuestionExtraction:
    return QuestionExtraction(
        question_id=qid,
        source_pages=[13],
        authoritative_source="answer table p13",
        answer_sheet_status="present",
        sub_items=[
            SubItemExtraction(
                sub_item_id=i,
                status="answered",
                final_answer=letter_for(i),
                answer_origin="answer_sheet",
                interpretation_rationale=f"row {i} read individually",
                confidence=0.9,
            )
            for i in ids
        ],
    )


def test_large_question_is_extracted_in_chunks_with_scoped_prompts():
    q = _mc_question(20)
    survey = _survey_two_sheets()
    pages = [PageImage(page_number=11, png_bytes=b"p", width=5, height=5, text="")]

    def responder(model, system, blocks):
        text = "\n".join(b["text"] for b in blocks if b.get("type") == "text")
        start = text.index("Report every sub-item (") + len("Report every sub-item (")
        ids = [s.strip() for s in text[start : text.index(")", start)].split(",")]
        return _extraction_for_ids("3", ids, lambda i: "ABCD"[int(i) % 4])

    backend = MockBackend(config=BackendConfig(backend="mock", model="m"), responder=responder)
    extraction = extract_question(backend, q, survey, pages)

    assert len(backend.calls) == 3, "20 sub-items at chunk size 8 -> 3 calls"
    assert [s.sub_item_id for s in extraction.sub_items] == [str(i) for i in range(1, 21)]
    assert all(s.status == "answered" for s in extraction.sub_items)
    # Later chunks carry the ignore-other-rows scoping instruction.
    scoped = [c for c in backend.calls if "IGNORE" in c.all_text()]
    assert len(scoped) == 3, "every partial chunk is scoped"
    # No sub-item is reported twice and none lost (reconciliation guarantee).
    assert len({s.sub_item_id for s in extraction.sub_items}) == 20


def test_chunk_merge_takes_worst_sheet_condition():
    q = _mc_question(16)
    survey = _survey_two_sheets()
    pages = [PageImage(page_number=11, png_bytes=b"p", width=5, height=5, text="")]
    statuses = iter(["present", "blank"])

    def responder(model, system, blocks):
        text = "\n".join(b["text"] for b in blocks if b.get("type") == "text")
        start = text.index("Report every sub-item (") + len("Report every sub-item (")
        ids = [s.strip() for s in text[start : text.index(")", start)].split(",")]
        ext = _extraction_for_ids("3", ids, lambda i: "AB"[int(i) % 2])
        ext.answer_sheet_status = next(statuses)
        return ext

    backend = MockBackend(config=BackendConfig(backend="mock", model="m"), responder=responder)
    extraction = extract_question(backend, q, survey, pages)
    assert extraction.answer_sheet_status == "blank"


def test_sheet_status_is_derived_from_closeread_not_model_echo():
    from autograder.extract import _derive_sheet_status
    from autograder.schema import QuestionExtraction

    survey = _survey_two_sheets()
    p11 = next(p for p in survey.pages if p.page_number == 11)
    p11.sheet_condition = "present"

    qx = QuestionExtraction(
        question_id="1",
        source_pages=[11],
        authoritative_source="sheet",
        answer_sheet_status="not_applicable",  # unreliable model echo
        sub_items=[],
    )
    _derive_sheet_status(qx, survey)
    assert qx.answer_sheet_status == "present"

    p11.sheet_condition = "blank"
    _derive_sheet_status(qx, survey)
    assert qx.answer_sheet_status == "blank", "worst close-read condition wins"

    # A question with no sheet pages at all is genuinely not_applicable.
    from autograder.schema import ExamSurvey, PageInfo

    no_sheet = ExamSurvey(
        pages=[PageInfo(page_number=1, content_summary="q", question_ids=["1"])],
        student_ink_description="pen",
        grader_annotations_description="",
    )
    qx.answer_sheet_status = "present"
    _derive_sheet_status(qx, no_sheet)
    assert qx.answer_sheet_status == "not_applicable"


def test_score_fractions_are_dropped_from_conventions():
    from autograder.survey import merge_closeread
    from autograder.schema import ConventionNote, SheetCloseRead, SheetPageReading

    survey = _survey_two_sheets()
    closeread = SheetCloseRead(
        pages=[
            SheetPageReading(page_number=11, serves_questions=["1"], sheet_condition="present")
        ],
        marking_conventions=[
            ConventionNote(
                page_number=11, verbatim_text="28/32",
                interpretation="score", scope="page 11",
            ),
            ConventionNote(
                page_number=11, verbatim_text="תשובות המסומנות ב-X הן הסופיות",
                interpretation="X final", scope="table",
            ),
        ],
    )
    merged = merge_closeread(survey, closeread)
    texts = [n.verbatim_text for n in merged.marking_conventions]
    assert "28/32" not in texts, "instructor score fractions are not conventions"
    assert any("X" in t for t in texts)
    assert "dropped as an instructor score fraction" in (merged.notes or "")


def test_uniform_collapse_is_flagged_for_review_only():
    q = _mc_question(20)
    ext = _extraction_for_ids("3", [str(i) for i in range(1, 21)], lambda i: "B")
    _flag_uniform_collapse(q, ext)
    assert all("collapse" in (s.uncertainty_note or "") for s in ext.sub_items)
    assert all(s.confidence <= 0.5 for s in ext.sub_items)
    # Protections only ADD review pressure — answers and status are untouched.
    assert all(s.status == "answered" and s.final_answer == "B" for s in ext.sub_items)


def test_varied_answers_and_small_questions_are_not_flagged():
    q = _mc_question(20)
    varied = _extraction_for_ids("3", [str(i) for i in range(1, 21)], lambda i: "ABCD"[int(i) % 4])
    _flag_uniform_collapse(q, varied)
    assert all(s.uncertainty_note is None for s in varied.sub_items)

    small_q = _mc_question(8)
    small = _extraction_for_ids("3", [str(i) for i in range(1, 9)], lambda i: "B")
    _flag_uniform_collapse(small_q, small)
    assert all(s.uncertainty_note is None for s in small.sub_items)
