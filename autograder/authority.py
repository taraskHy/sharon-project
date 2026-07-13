"""Deterministic enforcement of answer-source authority.

The extraction prompt instructs the model to read final answers from the
dedicated answer sheet when one exists, and to treat student ink on question
pages as scratch work. Prompts are not guarantees, so this post-extraction
pass enforces the policy structurally:

- When a dedicated answer sheet exists for a question and is usable
  (``answer_sheet_status == "present"``), a sub-item whose answer was read
  from a question page contradicts the policy. It is demoted to ambiguous —
  never silently accepted — and routed to human review. If the exam's
  printed instructions say booklet answers are not graded, the demotion note
  says so explicitly.
- When the sheet exists but is missing/blank/damaged/ambiguous for the
  question, question-page evidence MAY stand, but only as flagged secondary
  evidence: the sub-item keeps its answer and is routed to human review.
- When no dedicated answer sheet exists (``not_applicable`` — the exam
  expects answers on the question pages), question-page answers are the
  normal case and pass untouched.
- Origin ``none`` (legacy artefacts predating provenance, or unanswered
  items) is left alone: there is nothing to enforce against.

The pass mutates the extraction in place (uncertainty notes and status
demotions feed the existing review-flag machinery in ``grade.py``) and
returns an audit log of every enforcement action.
"""

from __future__ import annotations

from .schema import AnswerKey, ExamExtraction, ExamSurvey, QuestionExtraction

_SHEET_PAGE_KINDS = {"answer_sheet", "mixed"}
_SHEET_PROBLEM_STATUSES = {"missing", "blank", "damaged", "ambiguous"}


def sheet_pages_for_question(qid: str, survey: ExamSurvey) -> list[int]:
    """Pages holding a dedicated answer sheet relevant to question ``qid``.

    A page qualifies when the survey classified it as (or as containing) an
    answer area AND it is tied to the question — via ``answer_area_for_question``
    (which reflects student renumbering corrections), page/region question
    ids, or membership in the document-level authoritative pages with no
    contrary question assignment.
    """
    pages: set[int] = set()
    policy_pages = set(survey.answer_sheet_policy.authoritative_pages)
    for p in survey.pages:
        sheet_like = (
            p.page_kind in _SHEET_PAGE_KINDS
            or p.is_answer_area
            or p.page_number in policy_pages
        )
        if not sheet_like:
            continue
        if p.answer_area_for_question == qid or qid in p.question_ids:
            pages.add(p.page_number)
            continue
        if any(qid in r.question_ids for r in p.regions):
            pages.add(p.page_number)
            continue
        # An authoritative page with no question assignment at all is shared
        # (e.g. one unlabeled answer sheet for the whole exam).
        if p.page_number in policy_pages and not p.question_ids and not p.answer_area_for_question:
            pages.add(p.page_number)
    return sorted(pages)


def enforce_answer_authority(
    key: AnswerKey, survey: ExamSurvey, extraction: ExamExtraction
) -> list[str]:
    """Apply the precedence policy to every sub-item. Returns audit log lines
    (also appended to each question's ``notes`` so the persisted
    ``extraction.json`` carries the enforcement record)."""
    log: list[str] = []
    for q in extraction.questions:
        lines = _enforce_question(q, survey)
        if lines:
            joined = "answer-authority enforcement: " + " | ".join(lines)
            q.notes = f"{q.notes}\n{joined}" if q.notes else joined
        log.extend(lines)
    return log


def _enforce_question(q: QuestionExtraction, survey: ExamSurvey) -> list[str]:
    sheet_exists = bool(sheet_pages_for_question(q.question_id, survey))
    strict = survey.answer_sheet_policy.booklet_answers_not_graded
    log: list[str] = []

    # Instructor ink is never a student answer: an "answered" sub-item whose
    # only observed marks are grader annotations has no student evidence —
    # demote it to ambiguous and route to review (applies with or without an
    # answer sheet).
    for s in q.sub_items:
        if (
            s.status == "answered"
            and s.marks_observed
            and all(m.meaning == "grader_annotation" for m in s.marks_observed)
        ):
            candidates = sorted({a for a in [s.final_answer, *s.candidate_answers] if a})
            s.status = "ambiguous"
            s.final_answer = None
            s.candidate_answers = candidates
            s.confidence = 0.0
            s.uncertainty_note = _merge_note(
                s.uncertainty_note,
                "every observed mark for this sub-item is an instructor "
                "annotation — instructor ink is never a student answer; "
                "demoted to ambiguous for human review",
            )
            log.append(
                f"question {q.question_id} sub-item {s.sub_item_id}: demoted "
                f"answer supported only by instructor annotations"
            )

    if not sheet_exists:
        # No dedicated sheet for this question: answers on the question pages
        # are the intended design (or the survey found no sheet). Nothing to
        # demote; extraction's own uncertainty flags remain in force.
        return log

    sheet_broken = q.answer_sheet_status in _SHEET_PROBLEM_STATUSES

    for s in q.sub_items:
        if s.answer_origin != "question_page":
            continue  # sheet-sourced, agreeing, none/legacy: nothing to enforce

        if sheet_broken:
            # Secondary evidence is allowed to stand, but never silently.
            note = (
                f"final answer read from a question page because the answer "
                f"sheet is {q.answer_sheet_status} for question {q.question_id}; "
                "secondary evidence only — requires human review"
            )
            s.uncertainty_note = _merge_note(s.uncertainty_note, note)
            s.confidence = min(s.confidence, 0.5)
            log.append(
                f"question {q.question_id} sub-item {s.sub_item_id}: kept "
                f"question-page answer as flagged secondary evidence "
                f"(answer sheet {q.answer_sheet_status})"
            )
            continue

        # Sheet exists and is usable, yet the answer came from a question
        # page: scratch work must never silently override / stand in for the
        # sheet. Demote to ambiguous and route to review.
        reason = (
            "the exam's printed instructions say booklet markings are not graded"
            if strict
            else "a usable dedicated answer sheet is the authoritative source"
        )
        candidates = sorted({a for a in [s.final_answer, *s.candidate_answers] if a})
        s.status = "ambiguous"
        s.final_answer = None
        s.candidate_answers = candidates
        s.answer_origin = "question_page"
        s.confidence = 0.0
        s.uncertainty_note = _merge_note(
            s.uncertainty_note,
            (
                f"answer was read from a question page although the answer "
                f"sheet is present ({reason}); question-page ink is scratch "
                "work — demoted to ambiguous for human review"
            ),
        )
        log.append(
            f"question {q.question_id} sub-item {s.sub_item_id}: demoted "
            f"question-page answer {candidates or '(none)'} — {reason}"
        )
    return log


def _merge_note(existing: str | None, new: str) -> str:
    return f"{existing}; {new}" if existing else new


def flag_suspected_sheet_swap(
    key: AnswerKey, extraction: ExamExtraction, version: str
) -> list[str]:
    """Deterministic tripwire for an UNDETECTED answer-table mix-up.

    Students occasionally fill one question's answer table under another
    question's printed title. The visual corrections are sometimes too faint
    for the close-read; this post-extraction check compares, for every PAIR
    of same-shape matching questions, how well each question's extracted
    letters agree with its OWN key column versus the SIBLING's. A strongly
    crossed pattern is only ever FLAGGED for human review — answers, status
    and points are never changed by this check, because regrading on
    key-agreement would be score-driven inference.
    Returns audit log lines; mutates only uncertainty notes/confidence.
    """
    matching = [
        q for q in key.questions
        if q.type in ("matching", "matching_with_explanation")
    ]
    log: list[str] = []
    for i, qa in enumerate(matching):
        for qb in matching[i + 1 :]:
            if len(qa.sub_items) != len(qb.sub_items):
                continue
            try:
                ea = extraction.question(qa.id)
                eb = extraction.question(qb.id)
            except KeyError:
                continue
            own = _agreement(qa, ea, version) + _agreement(qb, eb, version)
            crossed = _agreement(qb, ea, version) + _agreement(qa, eb, version)
            total = len(qa.sub_items) + len(qb.sub_items)
            # Strong signal only: crossed agreement beats own by at least a
            # third of the items AND covers at least half of them.
            if crossed - own >= total // 3 and crossed >= total // 2:
                note = (
                    f"suspected answer-table mix-up between questions {qa.id} "
                    f"and {qb.id}: the extracted answers agree with the "
                    f"SIBLING question's key on {crossed}/{total} items but "
                    f"with their own on only {own}/{total}. The sheets' true "
                    "assignment must be confirmed by a human; scores below "
                    "assume the printed titles."
                )
                for qx in (ea, eb):
                    for s in qx.sub_items:
                        if note not in (s.uncertainty_note or ""):  # idempotent on resume
                            s.uncertainty_note = _merge_note(s.uncertainty_note, note)
                        s.confidence = min(s.confidence, 0.4)
                log.append(note)
    return log


def _agreement(q, qx, version: str) -> int:
    accepted = {
        s.id: [a.upper() for a in (s.correct_by_version.get(version) or [])]
        for s in q.sub_items
    }
    n = 0
    for s in qx.sub_items:
        if s.status == "answered" and s.final_answer:
            if s.final_answer.upper() in accepted.get(s.sub_item_id, []):
                n += 1
    return n
