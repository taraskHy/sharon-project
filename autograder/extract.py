"""Per-question extraction of the student's final answers and explanations."""

from __future__ import annotations

import json

from .ingest import PageImage, labeled_page_blocks
from .backends import VisionBackend
from .prompts import EXTRACTION_SYSTEM
from .schema import (
    AnswerKey,
    ExamExtraction,
    ExamSurvey,
    KeyQuestion,
    QuestionExtraction,
    SubItemExtraction,
)


def _pages_for_question(qid: str, survey: ExamSurvey, pages: list[PageImage]) -> list[PageImage]:
    """Select pages relevant to a question: pages the survey assigned to it,
    every answer area assigned to it, plus pages carrying convention notes.

    The fallback to the full document must be decided on the question's OWN
    pages — convention-note pages are shared context and would otherwise mask
    a question the survey failed to place anywhere.
    """
    question_pages: set[int] = set()
    for p in survey.pages:
        if qid in p.question_ids or p.answer_area_for_question == qid:
            question_pages.add(p.page_number)
    if not question_pages:
        return list(pages)  # survey placed this question nowhere: send everything
    wanted = set(question_pages)
    for note in survey.marking_conventions:
        wanted.add(note.page_number)
    return [p for p in pages if p.page_number in wanted]


def _question_structure(q: KeyQuestion) -> str:
    """Question structure WITHOUT correct answers (extraction must stay blind
    to the key so it reports what the student wrote, not what is right)."""
    lines = [
        f"Question {q.id}: {q.title}",
        f"Type: {q.type}",
        f"Sub-items ({len(q.sub_items)}):",
    ]
    for s in q.sub_items:
        lines.append(f"  - sub-item {s.id}: {s.prompt}")
    if q.answer_source:
        lines.append(f"Authoritative answer location per exam instructions: {q.answer_source}")
    if q.explanation_required:
        lines.append("Each sub-item requires a short written justification by the student.")
    return "\n".join(lines)


def extract_question(
    llm: VisionBackend,
    q: KeyQuestion,
    survey: ExamSurvey,
    pages: list[PageImage],
) -> QuestionExtraction:
    relevant = _pages_for_question(q.id, survey, pages)
    blocks: list[dict] = [
        {"type": "text", "text": _question_structure(q)},
        {
            "type": "text",
            "text": (
                "Document survey (conventions, ink separation, authoritative "
                "locations):\n"
                + json.dumps(survey.model_dump(), ensure_ascii=False, indent=1)
            ),
        },
        {
            "type": "text",
            "text": f"Relevant scan pages follow ({len(relevant)} pages).",
        },
    ]
    blocks.extend(labeled_page_blocks(relevant))
    blocks.append(
        {
            "type": "text",
            "text": (
                f"Extract the student's final answers for question {q.id} now. "
                f"Report every sub-item ({', '.join(s.id for s in q.sub_items)}) "
                "exactly once."
            ),
        }
    )
    extraction = llm.parse(
        system=EXTRACTION_SYSTEM,
        content_blocks=blocks,
        output_model=QuestionExtraction,
        max_tokens=16000,
    )
    return _reconcile_sub_items(q, extraction)


def _canon_id(sub_id: str) -> str:
    """Match ids robustly across the key and the extraction ('01' == '1 ')."""
    s = sub_id.strip()
    stripped = s.lstrip("0")
    return stripped if stripped else s


def _merge_duplicates(items: list[SubItemExtraction]) -> SubItemExtraction:
    """The extraction pass reported the same sub-item more than once. If all
    copies agree the first stands; conflicting copies mean the student's final
    intention is disputed between sources — that is ambiguity, not a tiebreak."""
    answers = {a for s in items if (a := s.final_answer)}
    if len(answers) <= 1 and len({s.status for s in items}) == 1:
        first = items[0]
        first.interpretation_rationale += (
            f" (reported {len(items)} times with agreeing content; merged)"
        )
        return first
    candidates = sorted(answers | {c for s in items for c in s.candidate_answers})
    return SubItemExtraction(
        sub_item_id=items[0].sub_item_id,
        status="ambiguous",
        final_answer=None,
        candidate_answers=candidates,
        explanation_transcription=next(
            (s.explanation_transcription for s in items if s.explanation_transcription),
            None,
        ),
        explanation_legibility=next(
            (s.explanation_legibility for s in items if s.explanation_legibility != "none"),
            "none",
        ),
        marks_observed=[m for s in items for m in s.marks_observed],
        interpretation_rationale=(
            "the extraction pass reported this sub-item multiple times with "
            "conflicting content: "
            + " / ".join(f"[{s.status}: {s.final_answer}] {s.interpretation_rationale}" for s in items)
        ),
        confidence=0.0,
        uncertainty_note="conflicting duplicate observations from extraction",
    )


def _reconcile_sub_items(q: KeyQuestion, extraction: QuestionExtraction) -> QuestionExtraction:
    """Guarantee a 1:1 match between key sub-items and extracted sub-items."""
    grouped: dict[str, list[SubItemExtraction]] = {}
    for s in extraction.sub_items:
        grouped.setdefault(_canon_id(s.sub_item_id), []).append(s)

    reconciled = []
    for key_sub in q.sub_items:
        found_group = grouped.get(_canon_id(key_sub.id))
        if not found_group:
            found = SubItemExtraction(
                sub_item_id=key_sub.id,
                status="ambiguous",
                final_answer=None,
                interpretation_rationale=(
                    "the extraction pass did not report this sub-item; flagged for review"
                ),
                confidence=0.0,
                uncertainty_note="missing from extraction output",
            )
        elif len(found_group) > 1:
            found = _merge_duplicates(found_group)
        else:
            found = found_group[0]
        found.sub_item_id = key_sub.id  # key's spelling is canonical
        reconciled.append(found)
    extraction.sub_items = reconciled
    extraction.question_id = q.id
    return extraction


def extract_exam(
    llm: VisionBackend,
    key: AnswerKey,
    survey: ExamSurvey,
    pages: list[PageImage],
    progress=None,
) -> ExamExtraction:
    questions = []
    for q in key.questions:
        if progress:
            progress(f"extracting question {q.id} ({q.title})")
        questions.append(extract_question(llm, q, survey, pages))
    return ExamExtraction(questions=questions)
