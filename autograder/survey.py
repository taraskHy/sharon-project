"""Document-level survey over the student exam scan.

Two model passes with distinct cost profiles:

1. ``survey_exam`` — every page at LOW resolution: page classification,
   answer-sheet policy, ink separation, version hints. Cheap, whole-document.
2. ``closeread_sheets`` — ONLY the located answer-sheet pages at FULL
   resolution: printed-title vs actual question (student renumbering /
   swapped tables), sheet condition, convention notes. Low resolution cannot
   read handwritten corrections, and getting these wrong silently misgrades
   whole questions — this is the fine-print pass.

``merge_closeread`` folds pass 2 into the survey deterministically, so the
persisted ``survey.json`` is the single merged source downstream stages use.
"""

from __future__ import annotations

from .ingest import PageImage, image_block, labeled_page_blocks
from .backends import VisionBackend
from .prompts import SHEET_CLOSEREAD_SYSTEM, SURVEY_SYSTEM
from .schema import AnswerKey, ExamSurvey, SheetCloseRead

_SHEET_KINDS = {"answer_sheet", "mixed"}


def _key_orientation_block(key: AnswerKey) -> dict:
    return {
        "type": "text",
        "text": (
            "Exam structure from the answer key (for orientation only — it "
            "contains NO student information):\n"
            + "\n".join(
                f"- Question {q.id}: {q.title} ({q.type}, {len(q.sub_items)} "
                f"sub-items, {q.max_points} pts)"
                + (f" — answers expected in: {q.answer_source}" if q.answer_source else "")
                for q in key.questions
            )
            + (
                f"\nExam versions that exist: {', '.join(key.versions)}"
                if key.versions != ["default"]
                else ""
            )
        ),
    }


def survey_exam(llm: VisionBackend, pages: list[PageImage], key: AnswerKey) -> ExamSurvey:
    blocks: list[dict] = [_key_orientation_block(key)]
    blocks.extend(labeled_page_blocks(pages))
    blocks.append(
        {
            "type": "text",
            "text": "Produce the document survey now.",
        }
    )
    return llm.parse(
        system=SURVEY_SYSTEM,
        content_blocks=blocks,
        output_model=ExamSurvey,
        max_tokens=16000,
    )


def candidate_sheet_pages(survey: ExamSurvey) -> list[int]:
    """Pages worth a full-resolution close read: everything the survey
    considers (or policy declares) an answer area."""
    nums = set(survey.answer_sheet_policy.authoritative_pages)
    for p in survey.pages:
        if p.page_kind in _SHEET_KINDS or p.is_answer_area:
            nums.add(p.page_number)
    return sorted(nums)


def closeread_sheets(
    llm: VisionBackend,
    survey: ExamSurvey,
    fullres_pages: list[PageImage],
    key: AnswerKey,
) -> SheetCloseRead | None:
    """Full-resolution second look at the answer-sheet pages. Returns None
    when the survey found no sheets (exam answered in the booklet)."""
    wanted = set(candidate_sheet_pages(survey))
    sheets = [p for p in fullres_pages if p.page_number in wanted]
    if not sheets:
        return None
    blocks: list[dict] = [_key_orientation_block(key)]
    # Topic anchors: the title digits of a mislabeled sheet may be corrected
    # in faint ink the model misses, but the student's written explanations
    # discuss the question's TOPIC — give the model each question's topic
    # vocabulary (prompts only, never answers) as an independent signal.
    topics = "\n".join(
        f"- Question {q.id} is about: "
        + "; ".join((s.prompt or "")[:60] for s in q.sub_items[:4])
        for q in key.questions
    )
    blocks.append(
        {
            "type": "text",
            "text": (
                "Question TOPICS from the key (prompts only — no answers):\n"
                + topics
                + "\nIf the handwritten explanations on a sheet discuss a "
                "DIFFERENT question's topic than the printed title claims, "
                "that is strong evidence of a student mix-up: report the "
                "content-matched question in serves_questions and describe "
                "the evidence."
            ),
        }
    )
    for p in sheets:
        blocks.append({"type": "text", "text": f"--- Page {p.page_number} (full resolution) ---"})
        blocks.append(image_block(p))
    blocks.append(
        {
            "type": "text",
            "text": (
                "These are the exam's dedicated answer-sheet pages. Produce "
                "the close-read report now (title vs reality — check the "
                "title digits for strikethrough AND the explanation topics "
                "against the question topics above — condition, regions, "
                "convention notes)."
            ),
        }
    )
    return llm.parse(
        system=SHEET_CLOSEREAD_SYSTEM,
        content_blocks=blocks,
        output_model=SheetCloseRead,
        max_tokens=6000,
    )


_SCORE_FRACTION = __import__("re").compile(r"^\s*\d{1,3}\s*/\s*\d{1,3}\s*$")


def merge_closeread(survey: ExamSurvey, closeread: SheetCloseRead) -> ExamSurvey:
    """Fold the close-read into the survey (mutates and returns it).

    The close-read saw the pages at full resolution, so for the sheet pages
    its page-role findings REPLACE the survey's; convention notes are
    appended (deduplicated); the policy's authoritative pages absorb any
    sheet the survey classified but forgot to list (never the reverse —
    pages are only added, so a policy set by printed instructions stays)."""
    by_num = {p.page_number: p for p in survey.pages}
    for reading in closeread.pages:
        page = by_num.get(reading.page_number)
        if page is None:
            continue  # close-read hallucinated a page number: ignore it
        page.page_kind = "answer_sheet" if page.page_kind != "mixed" else "mixed"
        page.is_answer_area = True
        page.sheet_condition = reading.sheet_condition
        if reading.regions:
            page.regions = reading.regions
        if reading.serves_questions:
            page.question_ids = list(reading.serves_questions)
            page.answer_area_for_question = (
                reading.serves_questions[0] if len(reading.serves_questions) == 1 else None
            )
        if reading.correction_evidence:
            page.content_summary += (
                f" [close-read: serves Q{','.join(reading.serves_questions)} "
                f"per student correction — {reading.correction_evidence}]"
            )
        if reading.page_number not in survey.answer_sheet_policy.authoritative_pages:
            survey.answer_sheet_policy.authoritative_pages.append(reading.page_number)

    seen = {(n.page_number, n.verbatim_text.strip()) for n in survey.marking_conventions}
    for note in closeread.marking_conventions:
        # A bare score fraction ("28/32") is instructor grading ink, not a
        # student marking convention — models occasionally transcribe them
        # here despite the prompt; drop them deterministically.
        if _SCORE_FRACTION.match(note.verbatim_text.strip()):
            stamp = (
                f"close-read note on page {note.page_number} dropped as an "
                f"instructor score fraction: {note.verbatim_text.strip()!r}"
            )
            survey.notes = f"{survey.notes}\n{stamp}" if survey.notes else stamp
            continue
        if (note.page_number, note.verbatim_text.strip()) not in seen:
            survey.marking_conventions.append(note)

    survey.answer_sheet_policy.authoritative_pages.sort()
    stamp = "close-read pass merged for pages " + ", ".join(
        str(r.page_number) for r in closeread.pages
    )
    survey.notes = f"{survey.notes}\n{stamp}" if survey.notes else stamp
    return survey
