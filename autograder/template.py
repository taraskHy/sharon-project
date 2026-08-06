"""Exam templates: per-exam-family grading configuration.

A template describes HOW an exam family is graded, independently of the
answer key's content:

- the **exam mode** — ``multiple_choice`` (no explanation transcription and
  no judging model calls), ``with_explanation`` (the full pipeline), or
  ``mixed`` (per-question modes);
- the **answer-sheet rule** — ``detected`` (the survey model pass locates
  dedicated answer sheets structurally; the default, used by the
  image-processing exam) or ``fixed_pages`` (this family's answer sheet is a
  known page set, e.g. "the first page is the psychometric-style response
  table"; the survey model pass is skipped entirely and a deterministic
  survey is synthesized). ``fixed_pages`` is template-specific configuration
  — it is never assumed globally.

Templates live next to the answer key as ``<key stem>.template.json`` (like
the variant and alignment sidecars) or are passed explicitly. Two exam
families must never share caches or configuration: every template field
enters the resume/key fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from .schema import (
    AnswerKey,
    AnswerSheetPolicy,
    ExamSurvey,
    PageInfo,
    PageRegion,
)

ExamMode = Literal["multiple_choice", "with_explanation", "mixed"]


class TemplateError(ValueError):
    pass


class ExamTemplate(BaseModel):
    template_id: str = Field(description="Stable identifier, e.g. 'prob-2026a'.")
    name: str = ""
    mode: ExamMode = "with_explanation"
    question_modes: dict[str, Literal["multiple_choice", "with_explanation"]] = Field(
        default_factory=dict,
        description=(
            "Mixed mode: per-question-id mode. Questions not listed use "
            "'multiple_choice' when the exam mode is 'multiple_choice', else "
            "'with_explanation'."
        ),
    )
    answer_sheet_rule: Literal["detected", "fixed_pages"] = "detected"
    answer_sheet_pages: list[int] = Field(
        default_factory=list,
        description="1-based page numbers of the answer sheet when rule=fixed_pages.",
    )
    answer_sheet_description: str = Field(
        default="",
        description=(
            "What the fixed answer sheet looks like (helps extraction orient), "
            "e.g. 'psychometric-style response table: one row per question, "
            "columns א/ב/ג/ד with checkboxes'."
        ),
    )
    booklet_answers_not_graded: bool = Field(
        default=True,
        description=(
            "With rule=fixed_pages: whether markings on the question pages are "
            "excluded from grading (the printed exam instructions usually say "
            "so explicitly, e.g. 'mark here only!')."
        ),
    )
    policy_evidence: str = Field(
        default="",
        description="Quote/paraphrase of the printed instruction backing the rule.",
    )
    notes: str = ""

    def mode_for_question(self, qid: str) -> str:
        if self.mode == "mixed":
            return self.question_modes.get(qid, "with_explanation")
        return self.mode


def template_path(key_path: str | Path) -> Path:
    key_path = Path(key_path)
    return key_path.with_name(key_path.stem + ".template.json")


def load_template(
    key_path: str | Path, explicit: str | Path | None = None
) -> Optional[ExamTemplate]:
    """Load the exam family's template. Returns None when the family has no
    template file (full-pipeline defaults apply, as before templates existed)."""
    path = Path(explicit) if explicit else template_path(key_path)
    if not path.exists():
        return None
    try:
        tpl = ExamTemplate.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError) as e:
        raise TemplateError(f"exam template {path} is invalid: {e}") from e
    if tpl.answer_sheet_rule == "fixed_pages" and not tpl.answer_sheet_pages:
        raise TemplateError(
            f"exam template {path}: rule 'fixed_pages' requires answer_sheet_pages"
        )
    tpl_dict = tpl.model_dump()
    tpl_dict["_path"] = str(path)
    tpl.__dict__["_path"] = str(path)
    return tpl


def template_fingerprint(tpl: ExamTemplate) -> str:
    return hashlib.sha256(
        json.dumps(tpl.model_dump(), sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


def apply_template_to_key(key: AnswerKey, tpl: ExamTemplate) -> list[str]:
    """Enforce the template's modes on the loaded answer key (in place).

    Returns audit notes. Multiple-choice questions get explanation judging
    switched off structurally (``_question_needs_judging`` then skips them, so
    no judging model call can occur); with_explanation questions keep or gain
    the explanation requirement exactly as the key/rubric parsed it.
    """
    notes: list[str] = []
    for q in key.questions:
        mode = tpl.mode_for_question(q.id)
        if mode == "multiple_choice":
            changed = (
                q.type not in ("multiple_choice", "matching")
                or q.explanation_required
                or q.explanation_weight
            )
            if q.type not in ("multiple_choice", "matching"):
                q.type = "multiple_choice"
            q.explanation_required = False
            q.explanation_weight = 0.0
            if changed:
                notes.append(
                    f"question {q.id}: multiple-choice mode enforced by template "
                    f"{tpl.template_id} (no explanation component)"
                )
        elif mode == "with_explanation":
            if q.type == "multiple_choice":
                q.type = "selection_with_explanation"
                notes.append(
                    f"question {q.id}: explanation component enabled by template "
                    f"{tpl.template_id}"
                )
        if tpl.answer_sheet_rule == "fixed_pages":
            pages = ", ".join(str(p) for p in tpl.answer_sheet_pages)
            q.answer_source = (
                f"the dedicated answer sheet on page(s) {pages} "
                f"({tpl.answer_sheet_description or 'per template'}); "
                "markings on question pages are "
                + ("not graded" if tpl.booklet_answers_not_graded else "secondary")
            )
    return notes


def synthesized_survey(
    tpl: ExamTemplate, key: AnswerKey, n_pages: int
) -> ExamSurvey:
    """Deterministic survey for rule=fixed_pages — no survey model call.

    The template's page rule IS the document policy: the listed pages are the
    dedicated answer sheet serving every question; every other page is
    question/instruction material whose student ink is scratch work.
    """
    sheet_pages = [p for p in tpl.answer_sheet_pages if 1 <= p <= n_pages]
    qids = [q.id for q in key.questions]
    pages: list[PageInfo] = []
    for num in range(1, n_pages + 1):
        if num in sheet_pages:
            pages.append(
                PageInfo(
                    page_number=num,
                    content_summary=(
                        tpl.answer_sheet_description
                        or "dedicated answer sheet (template-defined)"
                    ),
                    page_kind="answer_sheet",
                    regions=[
                        PageRegion(
                            kind="answer_table",
                            question_ids=qids,
                            description=(
                                tpl.answer_sheet_description
                                or "answer table serving every question"
                            ),
                        )
                    ],
                    question_ids=qids,
                    is_answer_area=True,
                )
            )
        else:
            pages.append(
                PageInfo(
                    page_number=num,
                    content_summary=(
                        "question booklet page (template rule: student ink here "
                        "is scratch work, not final answers)"
                    ),
                    page_kind="question_or_instructions",
                    question_ids=[],
                )
            )
    policy_bits = [f"template {tpl.template_id}: answer sheet fixed to page(s) "
                   f"{', '.join(str(p) for p in sheet_pages)}"]
    if tpl.policy_evidence:
        policy_bits.append(tpl.policy_evidence)
    return ExamSurvey(
        pages=pages,
        answer_sheet_policy=AnswerSheetPolicy(
            authoritative_pages=sheet_pages,
            booklet_answers_not_graded=tpl.booklet_answers_not_graded,
            policy_source="; ".join(policy_bits),
        ),
        marking_conventions=[],
        student_ink_description=(
            "unknown (survey synthesized from template; extraction reads the "
            "sheet pages directly)"
        ),
        grader_annotations_description="",
        authoritative_answer_locations=[
            f"question {qid}: answer sheet page(s) "
            + ", ".join(str(p) for p in sheet_pages)
            for qid in qids
        ],
        notes=f"survey synthesized deterministically from template {tpl.template_id}",
    )
