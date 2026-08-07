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
    versions_unverified: list[str] = Field(
        default_factory=list,
        description=(
            "Version ids whose answer for this sub-item could not be "
            "verified deterministically (e.g. colour-only encoding with no "
            "text-layer group and no operator override). Grading exams of "
            "these versions flags the sub-item for human review."
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

PageKind = Literal[
    "question_or_instructions",  # printed questions/instructions; any student ink here is normally scratch
    "answer_sheet",              # dedicated sheet the student fills in (tables, explanation lines)
    "mixed",                     # printed question material AND a designated answer area on one page
    "instructor_only",           # grading grid / score summary meant for the instructor
    "other",                     # cover, blank, or unidentifiable
]

RegionKind = Literal[
    "question_text",       # printed question/instructions/options/diagrams
    "answer_table",        # table/bubble grid holding final selections
    "explanation_area",    # designated space for written justifications
    "scratch_work",        # student calculations/drafts outside designated areas
    "instructor_grading",  # instructor-only boxes, score grids, annotations
    "convention_note",     # student note changing how marks must be read
    "other",
]


class PageRegion(BaseModel):
    """A functional region of a page. Locations are descriptive (e.g. 'bottom
    third', 'table under the Question 2 heading') — later passes are visual
    and need orientation, not pixel geometry."""

    kind: RegionKind
    question_ids: list[str] = Field(
        default_factory=list,
        description="Question ids this region belongs to, if determinable.",
    )
    description: str = Field(
        description="Where on the page the region is and what it contains."
    )


class PageInfo(BaseModel):
    page_number: int = Field(description="1-based page number in the scan.")
    content_summary: str
    page_kind: PageKind = Field(
        default="question_or_instructions",
        description=(
            "Classification of the page's role. Detect dedicated answer sheets "
            "from headings/instructions, table layouts, repeated question "
            "identifiers, and position in the document (often near the end) — "
            "never assume a fixed count of them."
        ),
    )
    regions: list[PageRegion] = Field(
        default_factory=list,
        description="Functional regions on this page (answer tables, explanation areas, scratch work, instructor-only areas).",
    )
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
    sheet_condition: Optional[Literal["present", "blank", "damaged", "ambiguous"]] = Field(
        default=None,
        description=(
            "For answer-sheet pages once close-read at full resolution: the "
            "sheet's physical/legibility condition. None before close-read "
            "or for non-sheet pages."
        ),
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


class AnswerSheetPolicy(BaseModel):
    """Where the student's FINAL answers live, per the exam's own printed
    instructions and the document's actual structure. Drives which pages
    extraction reads at full resolution and how conflicts are resolved."""

    authoritative_pages: list[int] = Field(
        default_factory=list,
        description=(
            "Pages of the dedicated answer sheet(s), if any exist. Empty when "
            "the exam expects answers directly on the question pages."
        ),
    )
    booklet_answers_not_graded: bool = Field(
        default=False,
        description=(
            "True when the exam's printed instructions explicitly state that "
            "answers/markings on the question pages (the booklet) are not "
            "checked. This rule is then followed strictly."
        ),
    )
    policy_source: Optional[str] = Field(
        default=None,
        description=(
            "Quote or paraphrase of the printed instruction / evidence the "
            "policy is based on (e.g. cover-page wording, answer-sheet heading)."
        ),
    )


class ExamSurvey(BaseModel):
    pages: list[PageInfo]
    answer_sheet_policy: AnswerSheetPolicy = Field(default_factory=AnswerSheetPolicy)
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
# Exam-variant detection (cover-page marker) and question alignment
# --------------------------------------------------------------------------


class VariantDetection(BaseModel):
    """Model output: what variant marker (e.g. flower) the cover page shows.

    The model NEVER sees answers or answer keys here — only the cover image
    and the marker descriptions. The variant must never be inferred from the
    student's answers or from whichever key scores highest."""

    marker_seen: str = Field(
        description="Short description of the marker actually visible on the page."
    )
    matched_marker: Optional[str] = Field(
        default=None,
        description=(
            "The name of the ONE catalogue marker this matches, or null when "
            "the marker is missing, cropped, illegible, or matches none/"
            "several of the catalogue entries."
        ),
    )
    confident: bool = Field(
        description="True only when the match is visually unambiguous."
    )
    page_region: str = Field(
        description="Where on the page the marker was found, e.g. 'bottom third, center-left'."
    )
    obstruction_note: Optional[str] = Field(
        default=None,
        description=(
            "Anything interfering with detection (ink over the marker, crop, "
            "scan damage). Instructor ink must be ignored, not matched."
        ),
    )


class QuestionAlignmentEntry(BaseModel):
    question_id: str
    printed_to_key: dict[str, str] = Field(
        description=(
            "Map from the sub-item number PRINTED on this variant's form to "
            "the answer key's canonical sub-item id, matched by question "
            "CONTENT. Identity when the variant prints the key's order."
        )
    )
    identical_order: bool = False
    notes: Optional[str] = None


class VariantAlignment(BaseModel):
    """Per-variant mapping of printed sub-item numbering to the key's
    canonical numbering (variants shuffle question order)."""

    variant: str
    questions: list[QuestionAlignmentEntry]
    confident: bool = True
    notes: Optional[str] = None


# --------------------------------------------------------------------------
# Answer-sheet close-read (full-resolution second look at the sheets only)
# --------------------------------------------------------------------------


class SheetPageReading(BaseModel):
    """Full-resolution reading of ONE dedicated answer-sheet page: which
    question(s) it actually serves after student corrections, its condition,
    and its regions. Never decides final answers."""

    page_number: int
    printed_title_question: Optional[str] = Field(
        default=None,
        description="Question id the PRINTED page title claims, if any.",
    )
    serves_questions: list[str] = Field(
        default_factory=list,
        description=(
            "Question id(s) this sheet page ACTUALLY answers, after applying "
            "handwritten corrections (crossed-out printed titles, renumbering, "
            "notes like 'I swapped the tables'). Equal to the printed title "
            "when no correction exists."
        ),
    )
    correction_evidence: Optional[str] = Field(
        default=None,
        description=(
            "Verbatim/near-verbatim evidence when serves_questions differs "
            "from the printed title. null when there is no correction."
        ),
    )
    sheet_condition: Literal["present", "blank", "damaged", "ambiguous"] = Field(
        description=(
            "'present' = usable; 'blank' = the student left it empty; "
            "'damaged' = torn/cut off/unscannable; 'ambiguous' = unreadable."
        )
    )
    regions: list[PageRegion] = Field(default_factory=list)


class SheetCloseRead(BaseModel):
    """Output of the close-read pass over the authoritative sheet pages."""

    pages: list[SheetPageReading]
    marking_conventions: list[ConventionNote] = Field(
        default_factory=list,
        description=(
            "Every handwritten note ON THESE PAGES that changes how marks "
            "must be read (e.g. 'answers marked with X are final'), verbatim "
            "+ interpretation + scope."
        ),
    )
    notes: Optional[str] = None


# --------------------------------------------------------------------------
# Per-question extraction
# --------------------------------------------------------------------------

AnswerStatus = Literal["answered", "unanswered", "ambiguous"]

AnswerOrigin = Literal[
    "answer_sheet",   # read from a dedicated answer sheet / answer table
    "question_page",  # read from student ink on a question/instruction page
    "both",           # both sources exist and agree
    "none",           # nothing was read (unanswered) or origin not determinable
]

AnswerSheetStatus = Literal[
    "present",         # the sheet exists and this question's part is usable
    "missing",         # the survey expected a sheet but it is absent from the scan
    "blank",           # the sheet exists but this question's part was left empty
    "damaged",         # torn/cut-off/unscannable portion
    "ambiguous",       # the sheet's markings cannot be read reliably
    "not_applicable",  # this exam has no dedicated answer sheet for the question
]


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
    answer_origin: AnswerOrigin = Field(
        default="none",
        description=(
            "Where the reported final answer/explanation was read from. When a "
            "dedicated answer sheet exists, finals come from it; question-page "
            "ink is scratch work and may serve only as flagged secondary "
            "evidence when the sheet is missing/blank/damaged/ambiguous."
        ),
    )
    source_page: Optional[int] = Field(
        default=None,
        description="Page number the final answer was read from (provenance).",
    )
    source_region: Optional[str] = Field(
        default=None,
        description=(
            "Region the final answer was read from, e.g. 'answer table row 7' "
            "or 'explanation lines under item 3' (provenance)."
        ),
    )
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
    answer_sheet_status: AnswerSheetStatus = Field(
        default="not_applicable",
        description=(
            "Condition of the dedicated answer sheet for THIS question. "
            "'present' when it exists and was readable; 'missing'/'blank'/"
            "'damaged'/'ambiguous' route question-page evidence to human "
            "review; 'not_applicable' when the exam has no sheet for it."
        ),
    )
    sub_items: list[SubItemExtraction]
    notes: Optional[str] = None


class BandRowExtraction(BaseModel):
    """One answer-table row read from a cropped header+row band image
    (``tablecrop``). The tiny schema keeps per-row calls fast and focused."""

    printed_row_number: str = Field(
        description=(
            "The question number PRINTED in the row's number column, exactly "
            "as printed. Read it from the image — do not assume it."
        )
    )
    row: SubItemExtraction


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
    variant_detection: Optional[dict] = Field(
        default=None,
        description=(
            "Structured record of marker-based variant detection: detected "
            "marker, selected variant, confidence, page/region evidence, and "
            "the source of the marker-to-variant mapping."
        ),
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
