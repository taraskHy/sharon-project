"""Low-review escalation engine.

    AUTO -> suspicious? -> automatic escalation -> resolved? -> AUTO / REVIEW

Human REVIEW is the LAST fallback, never the first response to uncertainty
— but safety is never lowered: only deterministic validation + a second
independent read can turn a suspicious case into AUTO.

OCR path  : primary transcription -> deterministic suspicion signals
            (cheap, local) -> only if suspicious: image-grounded verifier
            (task ocr_verify) -> agreement => AUTO, disagreement => REVIEW.
Grade path: primary grader (task grade_primary, tiny structured output) ->
            deterministic validation (score range, rubric ids, schema,
            MC consistency, explicit uncertainty) -> clean => AUTO; else
            stronger grader (task grade_escalate) -> consistent => AUTO;
            unresolved => REVIEW.

Every decision is recorded so the metrics can pair REVIEW rate with the
safety signals that produced it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .gradingpack import QuestionGradingPack

# ------------------------------------------------------------ OCR stage -----

_PROTOCOL_ARTIFACT = re.compile(r'^\s*\{\s*"transcription"|```|\\n|"transcription"')
_LATIN_TOKEN = re.compile(r"[A-Za-z]{2,}")
_DIGIT_TOKEN = re.compile(r"\d")


@dataclass
class OCRSuspicion:
    suspicious: bool
    signals: list[str] = field(default_factory=list)


def ocr_suspicion(text: str, *, expected_min_chars: int = 3, max_repeat_ratio: float = 0.5,
                  hebrew_expected: bool = True) -> OCRSuspicion:
    """Deterministic, local signals only. Conservative on purpose: normal
    handwriting reads must NOT be flagged (that would waste verifier calls)."""
    s: list[str] = []
    t = (text or "").strip()
    if len(t) < expected_min_chars:
        s.append("empty_or_tiny")
    if _PROTOCOL_ARTIFACT.search(t):
        s.append("protocol_artifact")
    words = t.split()
    if len(words) >= 6:
        uniq = len(set(words)) / len(words)
        if uniq < max_repeat_ratio:
            s.append("repetition")
    if hebrew_expected and t and not re.search(r"[֐-׿]", t) and len(t) > 12:
        s.append("no_hebrew")
    if re.search(r"\[\?\]|\[unreadable\]|לא קריא", t):
        s.append("self_flagged_unreadable")
    # short technical tokens are the known dangerous class -> flag for verify
    if len(_LATIN_TOKEN.findall(t)) >= 1 and len(t) < 40:
        s.append("short_technical_token")
    return OCRSuspicion(bool(s), s)


class OCRVerifyResult(BaseModel):
    verdict: Literal["supported", "review"]
    omissions: list[str] = Field(default_factory=list)
    substitutions: list[str] = Field(default_factory=list)
    additions: list[str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"] = "medium"


OCR_VERIFY_SYSTEM = (
    "You are a strict transcription-fidelity checker for handwritten Hebrew "
    "exam text (may mix English technical tokens). You receive an image of ONE "
    "handwritten line/cell and a proposed transcription. Judge ONLY whether the "
    "transcription faithfully matches the visible handwriting: do NOT solve, "
    "improve, correct terminology or spelling, or infer intent. Report omitted "
    "visible text, added text not visible, and substitutions; attend to short "
    "technical tokens, Latin letters, numbers, operators, negations. Reply with "
    "ONLY the JSON object."
)


@dataclass
class OCRDecision:
    outcome: Literal["auto", "review"]
    transcription: str
    suspicion: OCRSuspicion
    verify: Optional[dict] = None
    reason: str = ""


def escalate_ocr(*, transcription: str, crop_png_b64: str | None, gateway=None,
                 meta: dict | None = None, task: str = "ocr_verify") -> OCRDecision:
    susp = ocr_suspicion(transcription)
    if not susp.suspicious:
        return OCRDecision("auto", transcription, susp, reason="no suspicion signal")
    if gateway is None or crop_png_b64 is None:
        return OCRDecision("review", transcription, susp, reason="suspicious; no verifier available")
    try:
        gateway.route(task)
    except Exception:  # noqa: BLE001
        return OCRDecision("review", transcription, susp, reason="suspicious; ocr_verify not configured")
    try:
        res = gateway.call(task=task, system=OCR_VERIFY_SYSTEM, content_blocks=[
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": crop_png_b64}},
            {"type": "text", "text": "Proposed transcription:\n" + transcription + "\nCheck fidelity now."},
        ], output_model=OCRVerifyResult, meta={**(meta or {}), "stage": "escalation"})
    except Exception as e:  # noqa: BLE001
        return OCRDecision("review", transcription, susp, reason=f"verifier failed: {type(e).__name__}")
    v = res.value
    if v.verdict == "supported" and v.confidence in ("high", "medium") and not (v.omissions or v.substitutions or v.additions):
        return OCRDecision("auto", transcription, susp, v.model_dump(), reason="verifier supports transcription")
    return OCRDecision("review", transcription, susp, v.model_dump(), reason="verifier disagreement")


# ---------------------------------------------------------- grading stage ---


class GradeResult(BaseModel):
    """Routine grader output — deliberately tiny."""

    score: float
    rubric_items_met: list[str] = Field(default_factory=list)
    uncertain: bool = False
    evidence: Optional[str] = Field(default=None, description="<= 200 chars or null")


GRADE_SYSTEM = (
    "You grade ONE student answer against the provided question pack (rubric "
    "and scoring rules are authoritative; course context is supplemental). "
    "Preserve the student's wording as given — never rewrite it. Return the "
    "score (within the stated maximum), the rubric item ids met, uncertain=true "
    "if the transcription or the rubric leaves the score genuinely undecidable, "
    "and at most a 200-character evidence note. Reply with ONLY the JSON object."
)


def grade_prompt(pack: QuestionGradingPack, *, selected: str | None, transcription: str,
                 version: str | None) -> list[dict]:
    correct = None
    if version:
        correct = {sid: v.get(version) for sid, v in pack.correct_by_version.items()}
    return [{"type": "text", "text": (
        pack.to_grader_context() + "\n\n"
        + (f"Correct option(s) for this exam version: {correct}\n" if correct else "")
        + f"Student selected option: {selected or '(none)'}\n"
        + f"Student explanation (verbatim transcription):\n---\n{transcription}\n---\n"
        + f"Allowed rubric item ids: {pack.rubric_item_ids() or '(none)'}. "
        + f"Score range: 0..{pack.max_score:g}.")}]


@dataclass
class GradeValidation:
    ok: bool
    problems: list[str] = field(default_factory=list)


def validate_grade(g: GradeResult, pack: QuestionGradingPack, *, selection_correct: bool | None,
                   selected: str | None) -> GradeValidation:
    p: list[str] = []
    if not (0 <= g.score <= pack.max_score):
        p.append(f"score {g.score} outside 0..{pack.max_score}")
    allowed = set(pack.rubric_item_ids())
    bad = [r for r in g.rubric_items_met if r not in allowed]
    if allowed and bad:
        p.append(f"unknown rubric ids {bad}")
    if g.uncertain:
        p.append("grader reported uncertainty")
    if g.evidence and len(g.evidence) > 220:
        p.append("evidence exceeds length limit")
    # MC consistency: under wrong_choice_zero a wrong selection cannot score > 0
    if pack.grading_policy == "wrong_choice_zero" and selection_correct is False and g.score > 0:
        p.append("nonzero score with wrong choice under wrong_choice_zero")
    if pack.grading_policy == "choice_only" and g.rubric_items_met:
        p.append("rubric items on a choice_only question")
    return GradeValidation(not p, p)


@dataclass
class GradeDecision:
    outcome: Literal["auto", "review"]
    result: Optional[GradeResult]
    stage: Literal["primary", "escalated", "none"]
    problems: list[str] = field(default_factory=list)
    reason: str = ""


def escalate_grade(*, pack: QuestionGradingPack, selected: str | None, transcription: str,
                   version: str | None, selection_correct: bool | None, gateway,
                   meta: dict | None = None, primary_task: str = "grade_primary",
                   escalate_task: str = "grade_escalate", score_tolerance: float = 0.5) -> GradeDecision:
    m = {**(meta or {}), "pack_hash": pack.hash, "question_id": pack.question_id}
    blocks = grade_prompt(pack, selected=selected, transcription=transcription, version=version)
    try:
        primary = gateway.call(task=primary_task, system=GRADE_SYSTEM, content_blocks=blocks,
                               output_model=GradeResult, meta={**m, "stage": "grade"}).value
    except Exception as e:  # noqa: BLE001
        return GradeDecision("review", None, "none", [f"primary failed: {type(e).__name__}"], "primary grader failed")
    v = validate_grade(primary, pack, selection_correct=selection_correct, selected=selected)
    if v.ok:
        return GradeDecision("auto", primary, "primary", reason="primary clean")
    try:
        gateway.route(escalate_task)
    except Exception:  # noqa: BLE001
        return GradeDecision("review", primary, "primary", v.problems, "inconsistent; no escalation model configured")
    try:
        second = gateway.call(task=escalate_task, system=GRADE_SYSTEM, content_blocks=blocks,
                              output_model=GradeResult, meta={**m, "stage": "escalation"}).value
    except Exception as e:  # noqa: BLE001
        return GradeDecision("review", primary, "primary", v.problems + [f"escalation failed: {type(e).__name__}"],
                             "escalation failed")
    v2 = validate_grade(second, pack, selection_correct=selection_correct, selected=selected)
    consistent = (v2.ok and abs(second.score - primary.score) <= score_tolerance
                  and set(second.rubric_items_met) == set(primary.rubric_items_met))
    if v2.ok and (consistent or primary.uncertain and not second.uncertain):
        # second stage clean AND either agrees with primary, or resolves the
        # primary's declared uncertainty with a clean, self-consistent grade
        return GradeDecision("auto", second, "escalated", v.problems, "escalation resolved consistently")
    return GradeDecision("review", second if v2.ok else primary, "escalated", v.problems + v2.problems,
                         "unresolved disagreement after escalation")


# ------------------------------------------------------------- metrics -----


class ReviewMetrics:
    """Pairs REVIEW rate with the safety/escalation signals behind it."""

    def __init__(self):
        self.c = {k: 0 for k in (
            "items", "auto", "review", "escalated", "deterministic_only", "mc_early_exit",
            "explanations_skipped", "cloud_calls_avoided", "cloud_tokens_avoided_est",
            "local_qwen_fallback", "local_qwen_resolved", "cloud_mc_escalation",
            "ocr_escalation", "ocr_review", "grade_escalation", "grade_review")}

    def bump(self, key: str, n: int = 1):
        self.c[key] += n

    def as_dict(self) -> dict:
        n = self.c["items"] or 1
        d = dict(self.c)
        d.update({
            "auto_pct": round(100 * self.c["auto"] / n, 1),
            "review_pct": round(100 * self.c["review"] / n, 1),
            "escalation_pct": round(100 * self.c["escalated"] / n, 1),
            "deterministic_only_pct": round(100 * self.c["deterministic_only"] / n, 1),
            "mc_early_exit_pct": round(100 * self.c["mc_early_exit"] / n, 1),
        })
        return d
