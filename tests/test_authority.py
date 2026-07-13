"""Answer-source authority: dedicated answer sheets are authoritative.

Covers the policy the exam owner specified:
- tentative markings on question pages never (silently) override final
  answers on a dedicated answer sheet;
- when the sheet is missing/blank/damaged/ambiguous, question-page markings
  may stand only as secondary evidence flagged for human review;
- exams without a dedicated answer sheet use question-page response areas
  normally;
- extraction reads full-resolution images only for the authoritative pages
  (plus convention notes), not the whole booklet.
"""

from autograder.authority import enforce_answer_authority, sheet_pages_for_question
from autograder.config import GraderConfig
from autograder.extract import _pages_for_question
from autograder.grade import VersionDecision, grade_exam
from autograder.ingest import PageImage
from autograder.schema import (
    AnswerSheetPolicy,
    ExamExtraction,
    ExamSurvey,
    ConventionNote,
    PageInfo,
    PageRegion,
    QuestionExtraction,
    SubItemExtraction,
)
from tests.test_grade import make_key


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


def make_survey_with_sheet(booklet_not_graded: bool = False) -> ExamSurvey:
    """A 5-page exam: cover, two question pages (with scratch), one dedicated
    answer sheet for both questions, one instructor-only grading page."""
    return ExamSurvey(
        pages=[
            PageInfo(page_number=1, content_summary="cover", page_kind="other"),
            PageInfo(
                page_number=2,
                content_summary="question 1 text; student scratch circles",
                page_kind="question_or_instructions",
                question_ids=["1"],
                has_student_writing=True,
                regions=[
                    PageRegion(kind="question_text", question_ids=["1"], description="whole page"),
                    PageRegion(kind="scratch_work", question_ids=["1"], description="margins"),
                ],
            ),
            PageInfo(
                page_number=3,
                content_summary="question 3 MC options; tentative student marks",
                page_kind="question_or_instructions",
                question_ids=["3"],
                has_student_writing=True,
            ),
            PageInfo(
                page_number=4,
                content_summary="dedicated answer sheet: matching rows + MC table",
                page_kind="answer_sheet",
                question_ids=["1", "3"],
                is_answer_area=True,
                has_student_writing=True,
                regions=[
                    PageRegion(kind="answer_table", question_ids=["3"], description="MC table"),
                    PageRegion(kind="explanation_area", question_ids=["1"], description="rows"),
                ],
            ),
            PageInfo(
                page_number=5,
                content_summary="instructor grading grid",
                page_kind="instructor_only",
                has_grader_annotations=True,
            ),
        ],
        answer_sheet_policy=AnswerSheetPolicy(
            authoritative_pages=[4],
            booklet_answers_not_graded=booklet_not_graded,
            policy_source="printed instruction: answers only on the answer sheet"
            if booklet_not_graded
            else "answer-sheet heading",
        ),
        student_ink_description="blue pen",
        grader_annotations_description="red ink scores on page 5",
    )


def make_survey_without_sheet() -> ExamSurvey:
    """An exam answered directly on the question pages (no dedicated sheet)."""
    return ExamSurvey(
        pages=[
            PageInfo(
                page_number=1,
                content_summary="question 1 with answer lines under each item",
                page_kind="question_or_instructions",
                question_ids=["1"],
                has_student_writing=True,
            ),
            PageInfo(
                page_number=2,
                content_summary="question 3 MC circled directly",
                page_kind="question_or_instructions",
                question_ids=["3"],
                has_student_writing=True,
            ),
        ],
        student_ink_description="pencil",
        grader_annotations_description="",
    )


def sub(sub_id: str, answer: str | None, origin: str, status: str = "answered",
        confidence: float = 1.0) -> SubItemExtraction:
    return SubItemExtraction(
        sub_item_id=sub_id,
        status=status,
        final_answer=answer,
        answer_origin=origin,
        interpretation_rationale="test fixture",
        confidence=confidence,
    )


def one_question_extraction(sheet_status: str, items: list[SubItemExtraction],
                            qid: str = "3") -> ExamExtraction:
    return ExamExtraction(
        questions=[
            QuestionExtraction(
                question_id=qid,
                source_pages=[4],
                authoritative_source="answer sheet page 4",
                answer_sheet_status=sheet_status,
                sub_items=items,
            )
        ]
    )


# --------------------------------------------------------------------------
# deterministic enforcement
# --------------------------------------------------------------------------


def test_question_page_answer_never_silently_overrides_present_sheet():
    key = make_key()
    survey = make_survey_with_sheet()
    extraction = one_question_extraction(
        "present",
        [sub("1", "B", origin="answer_sheet"), sub("2", "C", origin="question_page")],
    )
    log = enforce_answer_authority(key, survey, extraction)

    kept, demoted = extraction.questions[0].sub_items
    assert kept.status == "answered" and kept.final_answer == "B"
    assert kept.uncertainty_note is None, "sheet-sourced answers pass untouched"

    assert demoted.status == "ambiguous"
    assert demoted.final_answer is None, "scratch answer must not stand as final"
    assert demoted.candidate_answers == ["C"], "the scratch answer is preserved as a candidate"
    assert demoted.confidence == 0.0
    assert "scratch" in demoted.uncertainty_note
    assert len(log) == 1 and "demoted" in log[0]
    assert "answer-authority enforcement" in extraction.questions[0].notes


def test_strict_booklet_rule_is_cited_when_printed_instructions_say_so():
    key = make_key()
    survey = make_survey_with_sheet(booklet_not_graded=True)
    extraction = one_question_extraction("present", [sub("1", "A", origin="question_page")])
    enforce_answer_authority(key, survey, extraction)
    item = extraction.questions[0].sub_items[0]
    assert item.status == "ambiguous"
    assert "not graded" in item.uncertainty_note


def test_broken_sheet_allows_question_page_as_flagged_secondary_evidence():
    key = make_key()
    survey = make_survey_with_sheet()
    for broken in ("missing", "blank", "damaged", "ambiguous"):
        extraction = one_question_extraction(broken, [sub("1", "D", origin="question_page")])
        log = enforce_answer_authority(key, survey, extraction)
        item = extraction.questions[0].sub_items[0]
        assert item.status == "answered", broken
        assert item.final_answer == "D", "secondary evidence keeps the answer"
        assert item.confidence <= 0.5
        assert "secondary evidence" in item.uncertainty_note
        assert "human review" in item.uncertainty_note
        assert any("secondary evidence" in line for line in log)


def test_no_dedicated_sheet_means_question_page_answers_are_normal():
    key = make_key()
    survey = make_survey_without_sheet()
    extraction = one_question_extraction(
        "not_applicable", [sub("1", "B", origin="question_page")]
    )
    log = enforce_answer_authority(key, survey, extraction)
    item = extraction.questions[0].sub_items[0]
    assert item.status == "answered" and item.final_answer == "B"
    assert item.uncertainty_note is None
    assert log == []


def test_legacy_origin_none_is_left_alone():
    key = make_key()
    survey = make_survey_with_sheet()
    extraction = one_question_extraction("present", [sub("1", "A", origin="none")])
    log = enforce_answer_authority(key, survey, extraction)
    item = extraction.questions[0].sub_items[0]
    assert item.status == "answered" and item.final_answer == "A"
    assert log == []


def test_demoted_item_reaches_human_review_through_grading():
    key = make_key()
    survey = make_survey_with_sheet(booklet_not_graded=True)
    items = [sub(str(i), "B", origin="answer_sheet") for i in range(1, 21)]
    items[15] = sub("16", "C", origin="question_page")  # scratch on the booklet
    extraction = ExamExtraction(
        questions=[
            QuestionExtraction(
                question_id="1",
                source_pages=[4],
                authoritative_source="answer sheet page 4",
                answer_sheet_status="present",
                sub_items=[
                    sub(str(i), a, origin="answer_sheet")
                    for i, a in enumerate("FGDHCIAE", start=1)
                ],
            ),
            QuestionExtraction(
                question_id="3",
                source_pages=[4],
                authoritative_source="answer sheet page 4",
                answer_sheet_status="present",
                sub_items=items,
            ),
        ]
    )
    enforce_answer_authority(key, survey, extraction)
    result = grade_exam(
        key,
        extraction,
        judgements={},
        version_decision=VersionDecision(version="A1", description="fixed for test", uncertain=False),
        config=GraderConfig(),
        survey=survey,
        exam_file="exam.pdf",
        graded_at="2026-07-13T00:00:00",
        model="mock:test",
    )
    flagged = [r for r in result.needs_human_review if r.sub_item_id == "16"]
    assert flagged, "demoted scratch answer must be routed to human review"
    q3 = next(q for q in result.questions if q.question_id == "3")
    item16 = next(s for s in q3.sub_results if s.sub_item_id == "16")
    assert item16.status == "ambiguous"
    assert item16.points_total == 0.0, "no silent credit from scratch markings"


# --------------------------------------------------------------------------
# page selection (inference-cost structure)
# --------------------------------------------------------------------------


def _fake_pages(n: int) -> list[PageImage]:
    return [
        PageImage(page_number=i, png_bytes=b"png", width=10, height=10, text="")
        for i in range(1, n + 1)
    ]


def test_sheet_pages_found_by_kind_regions_and_policy():
    survey = make_survey_with_sheet()
    assert sheet_pages_for_question("3", survey) == [4]
    assert sheet_pages_for_question("1", survey) == [4]


def test_shared_unlabeled_policy_page_serves_every_question():
    survey = make_survey_with_sheet()
    sheet = survey.pages[3]
    sheet.question_ids = []
    sheet.answer_area_for_question = None
    sheet.regions = []
    assert sheet_pages_for_question("3", survey) == [4]


def test_extraction_reads_only_sheet_and_convention_pages_when_sheet_exists():
    survey = make_survey_with_sheet()
    survey.marking_conventions.append(
        ConventionNote(
            page_number=4,
            verbatim_text="X marks are final",
            interpretation="X wins over circles",
            scope="answer table",
        )
    )
    pages = _fake_pages(5)
    selected = [p.page_number for p in _pages_for_question("3", survey, pages)]
    assert selected == [4], (
        "with a dedicated sheet, the booklet's question pages are not sent "
        f"to the vision model (got {selected})"
    )


def test_extraction_falls_back_to_question_pages_without_sheet():
    survey = make_survey_without_sheet()
    pages = _fake_pages(2)
    selected = [p.page_number for p in _pages_for_question("3", survey, pages)]
    assert selected == [2]


def test_extraction_falls_back_to_all_pages_when_survey_places_nothing():
    survey = make_survey_without_sheet()
    pages = _fake_pages(2)
    selected = [p.page_number for p in _pages_for_question("99", survey, pages)]
    assert selected == [1, 2]


# --------------------------------------------------------------------------
# remaining authority matrix scenarios
# --------------------------------------------------------------------------


def test_multiple_scratch_markings_never_become_final_answers():
    key = make_key()
    survey = make_survey_with_sheet()
    extraction = one_question_extraction(
        "present",
        [
            sub("1", "A", origin="question_page"),
            sub("2", "D", origin="question_page"),
            sub("3", "B", origin="answer_sheet"),
        ],
    )
    enforce_answer_authority(key, survey, extraction)
    demoted = [s for s in extraction.questions[0].sub_items if s.answer_origin == "question_page"]
    assert all(s.status == "ambiguous" and s.final_answer is None for s in demoted)
    kept = extraction.questions[0].sub_items[2]
    assert kept.status == "answered" and kept.final_answer == "B"


def test_conflicting_answers_on_two_answer_sheets_become_ambiguous():
    """Scenario: the student filled BOTH answer sheets for the same question
    (e.g. the original and a spare), disagreeing between them. Extraction
    reports the sub-item twice; reconciliation must yield ambiguity, never a
    silent tiebreak."""
    from autograder.extract import _reconcile_sub_items

    key = make_key()
    q1 = key.question("1")
    qx = QuestionExtraction(
        question_id="1",
        source_pages=[11, 12],
        authoritative_source="two conflicting answer sheets",
        answer_sheet_status="present",
        sub_items=(
            [sub(s.id, "A", origin="answer_sheet") for s in q1.sub_items]
            + [sub("1", "B", origin="answer_sheet")]  # second sheet disagrees on item 1
        ),
    )
    reconciled = _reconcile_sub_items(q1, qx)
    item1 = next(s for s in reconciled.sub_items if s.sub_item_id == "1")
    assert item1.status == "ambiguous"
    assert item1.final_answer is None
    assert sorted(item1.candidate_answers) == ["A", "B"]
    assert item1.uncertainty_note


def test_instructor_only_marks_never_become_student_answers():
    from autograder.schema import MarkObservation

    key = make_key()
    survey = make_survey_with_sheet()
    graded_only = sub("1", "C", origin="answer_sheet")
    graded_only.marks_observed = [
        MarkObservation(location="row 1", mark_type="grader_mark", meaning="grader_annotation"),
        MarkObservation(location="row 1 margin", mark_type="text_note", meaning="grader_annotation"),
    ]
    genuine = sub("2", "B", origin="answer_sheet")
    genuine.marks_observed = [
        MarkObservation(location="row 2", mark_type="x", meaning="selected_final"),
        MarkObservation(location="row 2 margin", mark_type="grader_mark", meaning="grader_annotation"),
    ]
    extraction = one_question_extraction("present", [graded_only, genuine])
    log = enforce_answer_authority(key, survey, extraction)
    demoted, kept = extraction.questions[0].sub_items
    assert demoted.status == "ambiguous" and demoted.final_answer is None
    assert "instructor" in demoted.uncertainty_note
    assert kept.status == "answered" and kept.final_answer == "B", (
        "a student mark alongside grader ink keeps the answer"
    )
    assert any("instructor annotations" in line for line in log)


def test_suspected_sheet_swap_is_flagged_for_review_never_regraded():
    from autograder.authority import flag_suspected_sheet_swap

    key = make_key()
    # make_key has one matching question (1) and one MC (3); add a sibling
    # matching question with the same shape so the pair check applies.
    q2 = key.question("1").model_copy(deep=True)
    q2.id = "2"
    for s in q2.sub_items:
        s.correct_by_version = {
            v: [chr(((ord(a[0]) - 65 + 1) % 9) + 65) for a in ans]
            for v, ans in s.correct_by_version.items()
        }
    key.questions.insert(1, q2)

    def extraction_for(qid, source_q):
        return QuestionExtraction(
            question_id=qid,
            source_pages=[11],
            authoritative_source="sheet",
            answer_sheet_status="present",
            sub_items=[
                sub(s.id, s.correct_by_version["A1"][0], origin="answer_sheet")
                for s in source_q.sub_items
            ],
        )

    # Crossed: question 1's sheet actually contains question 2's answers.
    crossed = ExamExtraction(
        questions=[
            extraction_for("1", key.question("2")),
            extraction_for("2", key.question("1")),
        ]
    )
    log = flag_suspected_sheet_swap(key, crossed, "A1")
    assert log and "mix-up" in log[0]
    for q in crossed.questions:
        for s in q.sub_items:
            assert "mix-up" in s.uncertainty_note
            assert s.status == "answered" and s.final_answer, "answers unchanged"
            assert s.confidence <= 0.4
    # Idempotent (resume path runs it again).
    flag_suspected_sheet_swap(key, crossed, "A1")
    assert crossed.questions[0].sub_items[0].uncertainty_note.count("mix-up") == 1

    # Straight extraction: no flags.
    straight = ExamExtraction(
        questions=[
            extraction_for("1", key.question("1")),
            extraction_for("2", key.question("2")),
        ]
    )
    assert flag_suspected_sheet_swap(key, straight, "A1") == []
    assert all(
        s.uncertainty_note is None for q in straight.questions for s in q.sub_items
    )

    # Noisy real-world swap (measured live): misreads erode the crossed
    # count to 7/16 with own=1 — must still fire.
    noisy = ExamExtraction(
        questions=[
            extraction_for("1", key.question("2")),
            extraction_for("2", key.question("1")),
        ]
    )
    flipped = 0
    for q in noisy.questions:
        for s in q.sub_items:
            if flipped < 9 and s.final_answer != "Z":
                s.final_answer = "Z" if flipped % 2 == 0 else s.final_answer
                flipped += 1
    # (9 corruptions leave crossed ≈ 7-11 of 16 while own stays ~0)
    assert flag_suspected_sheet_swap(key, noisy, "A1"), "noisy swap must still flag"


def test_convention_notes_reach_extraction_context():
    from autograder.extract import _survey_context_for_question

    survey = make_survey_with_sheet()
    survey.marking_conventions.append(
        ConventionNote(
            page_number=4,
            verbatim_text="תשובות המסומנות ב-X הן הסופיות",
            interpretation="X marks are final",
            scope="answer table p4",
        )
    )
    ctx = _survey_context_for_question("3", survey)
    assert any(
        "X marks are final" in n["interpretation"] for n in ctx["marking_conventions"]
    )
    assert ctx["answer_sheet_policy"]["authoritative_pages"] == [4]
