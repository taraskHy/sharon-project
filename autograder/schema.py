"""Pydantic models for every stage of the pipeline.

Three families of models:

1. Answer key / rubric   (``AnswerKey``)      — parsed once per exam form.
2. Extraction            (``ExamSurvey``, ``QuestionExtraction``) — what the
   student visibly did, before any grading decision.
3. Results               (``ExamResult``)     — graded output.

Extraction is deliberately separated from grading so that "what marks exist
on the page" is recorded independently of "how many points that is worth",
and so ambiguity can be preserved instead of silently resolved.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------
# Answer key / rubric
# --------------------------------------------------------------------------

QuestionType = Literal[
    "multiple_choice",              # pick one option, no explanation
    "selection_with_explanation",   # pick one option + short justification
    "matching_with_explanation",    # match items to labels + justification each
    "matching",                     # match items to labels, no justification
    "open",                         # free-text (graded by rubric only)
]


class KeySubItem(BaseModel):
    """One gradeable unit: a single MC question or one row of a matching task."""

    id: str = Field(description="Sub-item identifier, e.g. '1'..'20' or a row number.")
    prompt: str = Field(description="Short text of the operation/question being answered.")
    correct_by_version: dict[str, list[str]] = Field(
        description=(
            "Map from exam-version id to the list of accepted answers for this "
            "sub-item, canonicalised to single uppercase Latin letters where the "
            "options are lettered (Hebrew א/ב/ג/ד become A/B/C/D). Multiple "
            "entries in a list mean any of them earns credit. Exams without "
            "versions use the single key 'default'."
        )
    )
    points: float = Field(description="Maximum points for this sub-item.")
    reference_explanation: Optional[str] = Field(
        default=None,
        description=(
            "The answer key's reasoning for the correct answer, used as the "
            "reference when judging student explanations."
        ),
    )


class KeyQuestion(BaseModel):
    id: str = Field(description="Question identifier, e.g. '1', '2', '3'.")
    title: str = Field(description="Short human-readable title of the question.")
    type: QuestionType
    max_points: float = Field(
        description=(
            "Maximum points for the whole question. May be lower than the sum of "
            "sub-item points when the rubric caps the total (e.g. 20 items x 2 "
            "points capped at 36)."
        )
    )
    sub_items: list[KeySubItem]
    explanation_required: bool = Field(
        default=False,
        description=(
            "True when the rubric states that a correct selection earns no "
            "credit without a valid explanation."
        ),
    )
    explanation_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of each sub-item's points allocated to the explanation as "
            "a separate component. 0 means all points sit on the selection and "
            "the explanation (if required) acts as a gate."
        ),
    )
    answer_source: Optional[str] = Field(
        default=None,
        description=(
            "Where the authoritative student answers live, per the exam's own "
            "instructions (e.g. 'the answer table, not markings in the booklet')."
        ),
    )
    grading_notes: Optional[str] = Field(
        default=None,
        description="Free-text rubric notes relevant to grading this question.",
    )


class AnswerKey(BaseModel):
    exam_title: str
    versions: list[str] = Field(
        description="Exam version identifiers, e.g. ['A1','A2','A3'], or ['default'].",
    )
    questions: list[KeyQuestion]
    total_points: float
    general_rules: list[str] = Field(
        default_factory=list,
        description="Exam-level grading rules taken from the cover page / rubric.",
    )

    @field_validator("versions")
    @classmethod
    def _versions_not_empty(cls, v: list[str]) -> list[str]:
        return v or ["default"]

    def question(self, qid: str) -> KeyQuestion:
        for q in self.questions:
            if q.id == qid:
                return q
        raise KeyError(f"question {qid!r} not in answer key")


# --------------------------------------------------------------------------
# Survey (whole-document pass over the student exam)
# --------------------------------------------------------------------------


class PageInfo(BaseModel):
    page_number: int = Field(description="1-based page number in the scan.")
    content_summary: str
    question_ids: list[str] = Field(
        default_factory=list,
        description="Question ids that appear on this page (after resolving any student renumbering).",
    )
    is_answer_area: bool = Field(
        default=False,
        description="True if this page contains an answer table / bubble sheet.",
    )
    answer_area_for_question: Optional[str] = Field(
        default=None,
        description=(
            "If this page holds an answer table: which question it actually "
            "answers, AFTER applying any handwritten corrections by the student "
            "(students sometimes fill tables under the wrong printed title and "
            "renumber them by hand)."
        ),
    )
    has_student_writing: bool = False
    has_grader_annotations: bool = Field(
        default=False,
        description="True if instructor marks (ticks, scores, comments, usually a different ink colour) appear.",
    )


class ConventionNote(BaseModel):
    """A handwritten note by the student that changes how marks must be read."""

    page_number: int
    verbatim_text: str = Field(description="Transcription of the note as written.")
    interpretation: str = Field(
        description="What the note means for interpreting marks elsewhere in the exam."
    )
    scope: str = Field(
        description="Which pages/questions the note applies to, e.g. 'answer table on page 13'."
    )


class ExamSurvey(BaseModel):
    pages: list[PageInfo]
    marking_conventions: list[ConventionNote] = Field(default_factory=list)
    student_ink_description: str = Field(
        description="How the student's own writing looks (colour, style)."
    )
    grader_annotations_description: str = Field(
        description=(
            "Description of instructor annotations present in the scan (colour, "
            "kind, where) so later passes can ignore them. Empty string if none."
        )
    )
    authoritative_answer_locations: list[str] = Field(
        default_factory=list,
        description=(
            "Per the exam's printed instructions and the student's notes: for "
            "each question, where the final answers must be read from."
        ),
    )
    version_hints: Optional[str] = Field(
        default=None,
        description="Any printed or visual hints about which exam version this scan is.",
    )
    notes: Optional[str] = None


# --------------------------------------------------------------------------
# Per-question extraction
# --------------------------------------------------------------------------

AnswerStatus = Literal["answered", "unanswered", "ambiguous"]


class MarkObservation(BaseModel):
    location: str = Field(description="Where the mark is, e.g. 'answer table row 3, option C'.")
    mark_type: str = Field(
        description="circle | x | filled | cross_out | scribble | arrow | text_note | grader_mark | other"
    )
    meaning: str = Field(
        description=(
            "The interpreted meaning under the document's conventions: "
            "selected_final | cancelled | draft | grader_annotation | unclear"
        )
    )


class SubItemExtraction(BaseModel):
    sub_item_id: str
    status: AnswerStatus
    final_answer: Optional[str] = Field(
        default=None,
        description=(
            "The student's final selected answer, canonicalised to the option "
            "letter (Latin A-Z; Hebrew option letters mapped א=A ב=B ג=C ד=D). "
            "null when unanswered or ambiguous."
        ),
    )
    candidate_answers: list[str] = Field(
        default_factory=list,
        description="When ambiguous: the plausible final answers that cannot be decided between.",
    )
    explanation_transcription: Optional[str] = Field(
        default=None,
        description="Near-verbatim transcription of the student's written justification, if any.",
    )
    explanation_legibility: Literal["none", "full", "partial", "illegible"] = "none"
    marks_observed: list[MarkObservation] = Field(default_factory=list)
    interpretation_rationale: str = Field(
        description="How the final answer (or the ambiguity) was decided from the visible marks."
    )
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty_note: Optional[str] = Field(
        default=None,
        description="Set when handwriting or intent could not be determined reliably.",
    )


class QuestionExtraction(BaseModel):
    question_id: str
    source_pages: list[int]
    authoritative_source: str = Field(
        description="Where the final answers were read from and why (e.g. 'answer table p.13 per exam instructions')."
    )
    sub_items: list[SubItemExtraction]
    notes: Optional[str] = None


class ExamExtraction(BaseModel):
    """Container for all per-question extractions (serialised to disk)."""

    questions: list[QuestionExtraction]

    def question(self, qid: str) -> QuestionExtraction:
        for q in self.questions:
            if q.question_id == qid:
                return q
        raise KeyError(f"question {qid!r} not in extraction")


# --------------------------------------------------------------------------
# Explanation judging (LLM output)
# --------------------------------------------------------------------------

ExplanationVerdict = Literal["valid", "partially_valid", "invalid", "missing", "illegible"]


class ExplanationEvaluation(BaseModel):
    sub_item_id: str
    verdict: ExplanationVerdict
    reasoning: str = Field(description="Concise justification for the verdict.")
    explanation_matches_different_answer: Optional[str] = Field(
        default=None,
        description=(
            "If the student's explanation correctly justifies a DIFFERENT option "
            "than the one they selected (suggesting a copying slip), the option "
            "it actually matches; otherwise null."
        ),
    )


class ExplanationJudgement(BaseModel):
    evaluations: list[ExplanationEvaluation]


# --------------------------------------------------------------------------
# Graded results
# --------------------------------------------------------------------------


class SubItemResult(BaseModel):
    question_id: str
    sub_item_id: str
    question_type: QuestionType
    status: AnswerStatus
    student_answer: Optional[str]
    accepted_answers: list[str]
    selection_correct: Optional[bool] = Field(
        description="null when the sub-item is unanswered or ambiguous."
    )
    explanation_transcription: Optional[str]
    explanation_evaluation: Optional[ExplanationEvaluation]
    points_selection: float
    points_explanation: float
    points_total: float
    points_max: float
    reason: str
    needs_review: bool = False
    uncertainty_note: Optional[str] = None


class QuestionResult(BaseModel):
    question_id: str
    question_type: QuestionType
    points_awarded: float
    points_max: float
    sub_results: list[SubItemResult]
    capped: bool = Field(
        default=False,
        description="True when raw sub-item points exceeded the question cap.",
    )
    summary: str


class ReviewItem(BaseModel):
    question_id: str
    sub_item_id: str
    reason: str


class ExamResult(BaseModel):
    exam_file: str
    graded_at: str
    model: str = Field(description="backend:model identity string that graded this exam.")
    backend_info: Optional[dict] = Field(
        default=None,
        description="Exact backend/model/generation configuration used (no secrets).",
    )
    detected_version: Optional[str]
    version_detection: str = Field(
        description="How the exam version was determined (or why it is uncertain)."
    )
    total_awarded: float
    total_max: float
    questions: list[QuestionResult]
    unanswered: list[ReviewItem] = Field(default_factory=list)
    needs_human_review: list[ReviewItem] = Field(default_factory=list)
    mark_interpretations: list[str] = Field(
        default_factory=list,
        description=(
            "Record of how corrected, conflicting, or convention-dependent "
            "markings were interpreted document-wide."
        ),
    )
