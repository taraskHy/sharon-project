"""Typed decision statuses + raw signal capture.

Two separate failure classes must never be confused:

    OCR uncertainty      "I am not sure WHAT the student wrote."
    grading uncertainty  "I know what they wrote, but not how the rubric applies."

Routing follows from that separation and nothing else:

    OCR unclear                  -> OCR resolution/escalation (never a stronger grader)
    OCR clear + grading clear    -> AUTO
    OCR clear + grading unclear  -> grading escalation / grading RAG (never stronger OCR)

``DecisionSignals`` persists the RAW inputs behind each decision so a future
calibration pass can fit thresholds empirically. Nothing here fits, trains or
thresholds anything: model-reported confidence is recorded as one signal among
many and is explicitly NOT sufficient evidence for AUTO on its own.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

# --------------------------------------------------------------------------
# statuses
# --------------------------------------------------------------------------

OCRStatus = Literal["OCR_OK", "OCR_UNRESOLVED"]
GradeStatus = Literal["GRADE_OK", "GRADE_UNCERTAIN", "GRADE_INVALID", "GRADE_DISAGREEMENT"]

OCR_OK: str = "OCR_OK"
OCR_UNRESOLVED: str = "OCR_UNRESOLVED"
GRADE_OK: str = "GRADE_OK"
GRADE_UNCERTAIN: str = "GRADE_UNCERTAIN"
GRADE_INVALID: str = "GRADE_INVALID"
GRADE_DISAGREEMENT: str = "GRADE_DISAGREEMENT"

Route = Literal["AUTO", "OCR_ESCALATION", "GRADE_ESCALATION", "REVIEW", "PAUSED"]


# --------------------------------------------------------------------------
# raw signals (persisted; never thresholded here)
# --------------------------------------------------------------------------


@dataclass
class MCSignals:
    cv_score: Optional[float] = None            # deterministic ink score of the winner
    cv_margin: Optional[float] = None           # winner - runner-up
    candidate_cells: Optional[int] = None
    blank_metric: Optional[float] = None
    ink_metric: Optional[float] = None
    cv_local_agreement: Optional[bool] = None
    cv_cloud_agreement: Optional[bool] = None
    local_cloud_agreement: Optional[bool] = None
    model_reported_confidence: Optional[str] = None
    resolver_source: Optional[str] = None       # deterministic|local_model|cloud_model|agreement|review


@dataclass
class OCRSignals:
    crop_quality_status: Optional[str] = None   # imagequality.ImageQualityResult.status
    crop_quality_signals: dict[str, Any] = field(default_factory=dict)
    verifier_verdict: Optional[str] = None      # supported | review
    provider_agreement: Optional[bool] = None
    output_chars: Optional[int] = None
    length_ratio: Optional[float] = None        # observed / expected chars, where an expectation exists
    script_anomaly: Optional[str] = None        # e.g. no_hebrew, latin_only
    truncated: Optional[bool] = None
    schema_valid: Optional[bool] = None
    suspicion_signals: list[str] = field(default_factory=list)
    model_reported_confidence: Optional[str] = None


@dataclass
class GradingSignals:
    schema_valid: Optional[bool] = None
    evidence_checked: Optional[int] = None
    evidence_verified: Optional[int] = None
    evidence_fabricated: Optional[int] = None
    evidence_missing: Optional[int] = None
    invariants_ok: Optional[bool] = None
    invariant_problems: list[str] = field(default_factory=list)
    rubric_consistent: Optional[bool] = None
    primary_score: Optional[float] = None
    escalation_score: Optional[float] = None
    score_delta: Optional[float] = None
    primary_escalation_agreement: Optional[bool] = None
    explicit_uncertainty: Optional[bool] = None
    rag_used: Optional[bool] = None
    #: False when an optional-RAG policy wanted context but none was available
    #: (no course/retriever/index). Grading continued without it — recorded,
    #: never a REVIEW cause by itself.
    rag_available: Optional[bool] = None
    model_reported_confidence: Optional[str] = None


@dataclass
class DecisionSignals:
    """Everything raw that fed one question-level decision.

    Persist it verbatim. Calibration (mapping these onto an AUTO/REVIEW
    threshold) is deliberately NOT implemented: the empirical work has not
    been run, and hard-coding a guess here would defeat the purpose.
    """

    item_id: str = ""
    question_id: str = ""
    mc: MCSignals = field(default_factory=MCSignals)
    ocr: OCRSignals = field(default_factory=OCRSignals)
    grading: GradingSignals = field(default_factory=GradingSignals)
    raw: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "DecisionSignals":
        d = dict(d or {})
        return cls(
            item_id=d.get("item_id", ""), question_id=d.get("question_id", ""),
            mc=MCSignals(**(d.get("mc") or {})), ocr=OCRSignals(**(d.get("ocr") or {})),
            grading=GradingSignals(**(d.get("grading") or {})), raw=dict(d.get("raw") or {}))


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------


@dataclass
class RouteDecision:
    route: str
    reason_code: str
    explanation: str = ""
    ocr_status: Optional[str] = None
    grade_status: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


def route_item(*, ocr_status: Optional[str], grade_status: Optional[str] = None,
               ocr_escalation_available: bool = False,
               grade_escalation_available: bool = False,
               ocr_escalation_exhausted: bool = False,
               grade_escalation_exhausted: bool = False,
               paused: bool = False) -> RouteDecision:
    """The single routing rule. OCR trouble is answered with OCR work and
    grading trouble with grading work — never the other way round."""
    if paused:
        return RouteDecision("PAUSED", "BUDGET_PAUSED",
                             "model work paused before this item was decided",
                             ocr_status, grade_status)
    if ocr_status == OCR_UNRESOLVED:
        # Grading difficulty is irrelevant here: we do not know what was written.
        if ocr_escalation_available and not ocr_escalation_exhausted:
            return RouteDecision("OCR_ESCALATION", "OCR_UNRESOLVED",
                                 "the crop/transcription is not trusted; resolve the reading first",
                                 ocr_status, grade_status)
        return RouteDecision("REVIEW", "OCR_UNRESOLVED",
                             "the writing could not be read reliably and no OCR resolution remains",
                             ocr_status, grade_status)
    if grade_status is None or grade_status == GRADE_OK:
        return RouteDecision("AUTO", "AUTO", "reading and rubric application are both settled",
                             ocr_status, grade_status)
    if grade_status == GRADE_DISAGREEMENT:
        # Escalation already ran and disagreed: more grading calls cannot settle it.
        return RouteDecision("REVIEW", "GRADE_DISAGREEMENT",
                             "primary and escalation graders disagree", ocr_status, grade_status)
    if grade_status in (GRADE_UNCERTAIN, GRADE_INVALID):
        # A stronger OCR pass is NOT an option here — the reading is fine.
        if grade_escalation_available and not grade_escalation_exhausted:
            return RouteDecision("GRADE_ESCALATION", grade_status,
                                 "the reading is settled; the rubric application is not",
                                 ocr_status, grade_status)
        return RouteDecision("REVIEW", grade_status,
                             "rubric application unresolved and no grading escalation remains",
                             ocr_status, grade_status)
    return RouteDecision("REVIEW", "GRADE_UNCERTAIN", f"unknown grade status {grade_status!r}",
                         ocr_status, grade_status)


def ocr_status_from(*, suspicious: bool, verifier_verdict: Optional[str] = None,
                    quality_status: Optional[str] = None) -> str:
    """Deterministic mapping of the OCR-side evidence onto a typed status."""
    if quality_status and quality_status != "OK":
        return OCR_UNRESOLVED
    if not suspicious:
        return OCR_OK
    if verifier_verdict == "supported":
        return OCR_OK
    return OCR_UNRESOLVED


def grade_status_from(*, validation_ok: bool, uncertain: bool = False,
                      disagreement: bool = False) -> str:
    """Deterministic mapping of the grading-side evidence onto a typed status."""
    if disagreement:
        return GRADE_DISAGREEMENT
    if not validation_ok:
        return GRADE_INVALID
    if uncertain:
        return GRADE_UNCERTAIN
    return GRADE_OK
