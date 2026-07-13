"""Per-question extraction of the student's final answers and explanations."""

from __future__ import annotations

import json

from .authority import enforce_answer_authority, sheet_pages_for_question
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
    """Select the pages extraction reads for a question, cheapest-first:

    1. When the survey located dedicated answer-sheet pages for the question,
       send THOSE (plus convention-note pages — the notes govern how marks on
       the sheet must be read). Question pages are not sent: the student's
       gradeable responses live on the sheet, sub-item prompts come from the
       key, and skipping the booklet pages keeps the payload small enough for
       modest context windows.
    2. Otherwise fall back to the pages the survey assigned to the question
       (the exam expects answers on the question pages themselves).
    3. If the survey placed the question nowhere, send the whole document.
       The fallback is decided on the question's OWN pages — convention-note
       pages are shared context and would otherwise mask a question the
       survey failed to place anywhere.
    """
    sheet_pages = set(sheet_pages_for_question(qid, survey))
    question_pages: set[int] = set()
    for p in survey.pages:
        if qid in p.question_ids or p.answer_area_for_question == qid:
            question_pages.add(p.page_number)

    if sheet_pages:
        wanted = sheet_pages
    elif question_pages:
        wanted = question_pages
    else:
        return list(pages)  # survey placed this question nowhere: send everything
    wanted = set(wanted)
    for note in survey.marking_conventions:
        wanted.add(note.page_number)
    return [p for p in pages if p.page_number in wanted]


def _survey_context_for_question(qid: str, survey: ExamSurvey) -> dict:
    """The survey fields extraction actually needs, instead of the full survey
    dump: document-wide conventions and policy, plus THIS question's pages.
    Keeps per-question payloads small (answer sheets are few; the page
    inventory of a long booklet is mostly irrelevant to any one question)."""
    relevant_pages = [
        p.model_dump()
        for p in survey.pages
        if qid in p.question_ids
        or p.answer_area_for_question == qid
        or any(qid in r.question_ids for r in p.regions)
        or p.page_kind in ("answer_sheet", "mixed", "instructor_only")
    ]
    return {
        "answer_sheet_policy": survey.answer_sheet_policy.model_dump(),
        "marking_conventions": [n.model_dump() for n in survey.marking_conventions],
        "student_ink_description": survey.student_ink_description,
        "grader_annotations_description": survey.grader_annotations_description,
        "authoritative_answer_locations": survey.authoritative_answer_locations,
        "pages_relevant_to_this_question": relevant_pages,
    }


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


# Questions with more sub-items than this are extracted in consecutive
# chunks: one call asked to read 20 table rows at once showed template
# collapse on the live model (every row reported as the SAME letter with an
# identical rationale and no per-row marks — docs/validation/, 2026-07-13).
# Small chunks force row-by-row visual work; reconciliation merges them.
EXTRACTION_CHUNK_SIZE = 8


def _extract_chunk(
    llm: VisionBackend,
    q: KeyQuestion,
    sub_items: list,
    survey: ExamSurvey,
    relevant: list[PageImage],
) -> QuestionExtraction:
    ids = [s.id for s in sub_items]
    chunk_q = q.model_copy(deep=True)
    chunk_q.sub_items = sub_items
    blocks: list[dict] = [
        {"type": "text", "text": _question_structure(chunk_q)},
        {
            "type": "text",
            "text": (
                "Document survey (conventions, ink separation, answer-sheet "
                "policy, authoritative locations):\n"
                + json.dumps(
                    _survey_context_for_question(q.id, survey),
                    ensure_ascii=False,
                    indent=1,
                )
            ),
        },
        {
            "type": "text",
            "text": f"Relevant scan pages follow ({len(relevant)} pages).",
        },
    ]
    blocks.extend(labeled_page_blocks(relevant))
    scope_note = (
        f"Extract the student's final answers for question {q.id} now. "
        f"Report every sub-item ({', '.join(ids)}) exactly once."
    )
    if len(ids) < len(q.sub_items):
        scope_note += (
            f" The sheet also shows other rows of question {q.id} — IGNORE "
            f"them; report ONLY rows {', '.join(ids)}. Read each row "
            "individually from its own marks; adjacent rows often differ, so "
            "never assume a pattern continues."
        )
    blocks.append({"type": "text", "text": scope_note})
    return llm.parse(
        system=EXTRACTION_SYSTEM,
        content_blocks=blocks,
        output_model=QuestionExtraction,
        max_tokens=16000,
    )


_CONDITION_SEVERITY = ["damaged", "missing", "ambiguous", "blank", "present", "not_applicable"]


def _merge_chunk_extractions(q: KeyQuestion, parts: list[QuestionExtraction]) -> QuestionExtraction:
    merged = parts[0]
    for part in parts[1:]:
        merged.sub_items.extend(part.sub_items)
        merged.source_pages = sorted(set(merged.source_pages) | set(part.source_pages))
        if part.notes:
            merged.notes = f"{merged.notes}\n{part.notes}" if merged.notes else part.notes
    # Worst reported sheet condition wins: a chunk that saw damage outranks
    # chunks that read fine.
    merged.answer_sheet_status = min(
        (p.answer_sheet_status for p in parts),
        key=_CONDITION_SEVERITY.index,
    )
    return merged


def _flag_uniform_collapse(q: KeyQuestion, extraction: QuestionExtraction) -> None:
    """Deterministic tripwire for the observed template-collapse failure:
    many rows, every one answered with the SAME letter. Real students
    occasionally do fill a whole column — a human should confirm that, so
    this only ADDS review flags; it never changes answers or status."""
    answered = [s for s in extraction.sub_items if s.status == "answered" and s.final_answer]
    if len(answered) < 10 or len(answered) < len(extraction.sub_items):
        return
    letters = {s.final_answer for s in answered}
    if len(letters) > 1:
        return
    note = (
        f"all {len(answered)} sub-items report the same answer "
        f"{letters.pop()!r} — uniform pattern is a known extraction-collapse "
        "signature; verify against the scan"
    )
    for s in extraction.sub_items:
        s.uncertainty_note = f"{s.uncertainty_note}; {note}" if s.uncertainty_note else note
        s.confidence = min(s.confidence, 0.5)


def extract_question(
    llm: VisionBackend,
    q: KeyQuestion,
    survey: ExamSurvey,
    pages: list[PageImage],
    chunk_size: int = EXTRACTION_CHUNK_SIZE,
) -> QuestionExtraction:
    relevant = _pages_for_question(q.id, survey, pages)
    chunks = [
        q.sub_items[i : i + chunk_size] for i in range(0, len(q.sub_items), chunk_size)
    ] or [[]]
    parts = [_extract_chunk(llm, q, chunk, survey, relevant) for chunk in chunks]
    extraction = _merge_chunk_extractions(q, parts) if len(parts) > 1 else parts[0]
    extraction = _reconcile_sub_items(q, extraction)
    _flag_uniform_collapse(q, extraction)
    return extraction


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
    alignment=None,
) -> ExamExtraction:
    """``alignment`` (a validated VariantAlignment) relabels each question
    into the variant's PRINTED sub-item numbering for the model — the
    numbering the student actually saw and filled into the answer sheet —
    and the results are remapped back to the key's canonical ids before
    reconciliation and scoring."""
    from .variant import printed_view, remap_extraction

    entries = {e.question_id: e for e in alignment.questions} if alignment else {}
    questions = []
    for q in key.questions:
        entry = entries.get(q.id)
        if progress:
            progress(f"extracting question {q.id} ({q.title})")
        if entry is not None and not entry.identical_order:
            view = printed_view(q, entry)
            if progress:
                progress(
                    f"question {q.id}: variant prints sub-items in a different "
                    f"order — extracting in printed numbering, remapping to key ids"
                )
            qx = extract_question(llm, view, survey, pages)
            qx = remap_extraction(qx, entry)
            qx.question_id = q.id
        else:
            qx = extract_question(llm, q, survey, pages)
        questions.append(qx)
    extraction = ExamExtraction(questions=questions)
    # The sheet's condition is established by the close-read at high
    # resolution; the extraction model's echo of it is unreliable (observed
    # 'not_applicable' while reading answers OFF the sheet). Derive it
    # deterministically before authority enforcement relies on it.
    for qx in extraction.questions:
        _derive_sheet_status(qx, survey)
    # Structural guarantee, independent of prompt compliance: question-page
    # scratch never silently overrides or stands in for a usable answer sheet.
    for line in enforce_answer_authority(key, survey, extraction):
        if progress:
            progress(f"authority: {line}")
    return extraction


_CONDITION_WORST_FIRST = ["damaged", "ambiguous", "blank", "present"]


def _derive_sheet_status(qx: QuestionExtraction, survey: ExamSurvey) -> None:
    sheet_nums = set(sheet_pages_for_question(qx.question_id, survey))
    if not sheet_nums:
        qx.answer_sheet_status = "not_applicable"
        return
    conditions = [
        p.sheet_condition
        for p in survey.pages
        if p.page_number in sheet_nums and p.sheet_condition
    ]
    if not conditions:
        # No close-read condition available (e.g. legacy survey): a located
        # sheet is at least 'present' unless the model claimed a problem.
        if qx.answer_sheet_status == "not_applicable":
            qx.answer_sheet_status = "present"
        return
    for level in _CONDITION_WORST_FIRST:
        if level in conditions:
            qx.answer_sheet_status = level
            return
