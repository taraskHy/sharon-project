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

from .evidence import CreditedItem, validate_evidence
from .gradingpack import QuestionGradingPack
from .signals import (OCR_UNRESOLVED as OCR_UNRESOLVED_, DecisionSignals, GradingSignals,
                      OCRSignals, grade_status_from, ocr_status_from)

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
    status: str = "OCR_OK"              # signals.OCRStatus
    signals: OCRSignals = field(default_factory=OCRSignals)


def _ocr_signals(text: str, susp: OCRSuspicion, verify: dict | None,
                 quality_status: str | None) -> OCRSignals:
    return OCRSignals(crop_quality_status=quality_status,
                      verifier_verdict=(verify or {}).get("verdict"),
                      output_chars=len(text or ""),
                      script_anomaly="no_hebrew" if "no_hebrew" in susp.signals else None,
                      schema_valid=True, suspicion_signals=list(susp.signals),
                      model_reported_confidence=(verify or {}).get("confidence"))


def escalate_ocr(*, transcription: str, crop_png_b64: str | None, gateway=None,
                 meta: dict | None = None, task: str = "ocr_verify",
                 quality_status: str | None = None) -> OCRDecision:
    """OCR-side escalation ONLY. It is never triggered by grading difficulty:
    the caller reaches it because the READING is in doubt (deterministic
    suspicion signals, or a pre-OCR image-quality verdict)."""
    susp = ocr_suspicion(transcription)
    if quality_status and quality_status != "OK":
        # An unreadable/blank/clipped crop is an imaging problem: no amount of
        # verifier agreement should turn it into AUTO.
        return OCRDecision("review", transcription, susp, reason=f"image quality {quality_status}",
                           status=OCR_UNRESOLVED_,
                           signals=_ocr_signals(transcription, susp, None, quality_status))
    if not susp.suspicious:
        return OCRDecision("auto", transcription, susp, reason="no suspicion signal",
                           status="OCR_OK", signals=_ocr_signals(transcription, susp, None, quality_status))
    if gateway is None or crop_png_b64 is None:
        return OCRDecision("review", transcription, susp, reason="suspicious; no verifier available",
                           status=OCR_UNRESOLVED_,
                           signals=_ocr_signals(transcription, susp, None, quality_status))
    try:
        gateway.route(task)
    except Exception:  # noqa: BLE001
        return OCRDecision("review", transcription, susp, reason="suspicious; ocr_verify not configured",
                           status=OCR_UNRESOLVED_,
                           signals=_ocr_signals(transcription, susp, None, quality_status))
    try:
        res = gateway.call(task=task, system=OCR_VERIFY_SYSTEM, content_blocks=[
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": crop_png_b64}},
            {"type": "text", "text": "Proposed transcription:\n" + transcription + "\nCheck fidelity now."},
        ], output_model=OCRVerifyResult, meta={**(meta or {}), "stage": "escalation"})
    except Exception as e:  # noqa: BLE001
        return OCRDecision("review", transcription, susp, reason=f"verifier failed: {type(e).__name__}",
                           status=OCR_UNRESOLVED_,
                           signals=_ocr_signals(transcription, susp, None, quality_status))
    v = res.value
    vd = v.model_dump()
    sig = _ocr_signals(transcription, susp, vd, quality_status)
    if v.verdict == "supported" and v.confidence in ("high", "medium") and not (v.omissions or v.substitutions or v.additions):
        sig.provider_agreement = True
        return OCRDecision("auto", transcription, susp, vd, reason="verifier supports transcription",
                           status="OCR_OK", signals=sig)
    sig.provider_agreement = False
    return OCRDecision("review", transcription, susp, vd, reason="verifier disagreement",
                       status=OCR_UNRESOLVED_, signals=sig)


# ---------------------------------------------------------- grading stage ---


class RubricItemGrade(BaseModel):
    """One rubric item's verdict WITH the span it rests on."""

    id: str
    met: bool
    student_evidence: Optional[str] = Field(
        default=None,
        description=("A SHORT span copied verbatim from the student transcription that "
                     "supports met=true. null when the item is not met (or when the "
                     "rubric declares this item needs no quoted span)."))


class GradeResult(BaseModel):
    """Routine grader output — deliberately tiny."""

    score: float
    rubric_items: list[RubricItemGrade] = Field(default_factory=list)
    rubric_items_met: list[str] = Field(
        default_factory=list,
        description="Legacy id-only form; prefer rubric_items with evidence.")
    uncertain: bool = False
    evidence: Optional[str] = Field(default=None, description="<= 200 chars or null")

    def credited(self) -> list[CreditedItem]:
        """Every rubric item awarded credit, with its cited span (if any).
        Unions the structured and legacy fields; duplicates are preserved so
        the validator can report double credit."""
        out = [CreditedItem(i.id, i.student_evidence) for i in self.rubric_items if i.met]
        structured = {i.id for i in self.rubric_items}   # the same id in both fields is one claim
        out += [CreditedItem(rid, None) for rid in self.rubric_items_met if rid not in structured]
        return out

    def met_ids(self) -> list[str]:
        seen, out = set(), []
        for c in self.credited():
            if c.id not in seen:
                seen.add(c.id)
                out.append(c.id)
        return out


GRADE_SYSTEM = (
    "You grade ONE student answer against the provided question pack (rubric "
    "and scoring rules are authoritative; course context is supplemental). "
    "Preserve the student's wording as given — never rewrite it. Return the "
    "score (within the stated maximum) and one entry per rubric item: its id, "
    "whether it is met, and — when met — a SHORT span copied VERBATIM from the "
    "student transcription that supports it (copy it exactly; never paraphrase, "
    "translate, correct or invent a span; if no span in the transcription "
    "supports the item, the item is not met). Set uncertain=true if the "
    "transcription or the rubric leaves the score genuinely undecidable. Reply "
    "with ONLY the JSON object."
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
    evidence: Optional[dict] = None       # evidence.EvidenceValidation.as_dict()
    invariants: Optional[dict] = None     # invariants.InvariantReport.as_dict()


def validate_grade(g: GradeResult, pack: QuestionGradingPack, *, selection_correct: bool | None,
                   selected: str | None, transcription: str | None = None) -> GradeValidation:
    """Deterministic validation of ONE grader output.

    ``transcription`` is the FROZEN student transcription; it is read only —
    evidence is never made to match by editing it. Omitting it means cited
    evidence cannot be verified, which is itself a problem when the pack
    requires evidence-grounded credit.
    """
    from .invariants import check_question_invariants

    p: list[str] = []
    if not (0 <= g.score <= pack.max_score):
        p.append(f"score {g.score} outside 0..{pack.max_score}")
    allowed = set(pack.rubric_item_ids())
    bad = [r for r in g.met_ids() if r not in allowed]
    if allowed and bad:
        p.append(f"unknown rubric ids {bad}")
    if g.uncertain:
        p.append("grader reported uncertainty")
    if g.evidence and len(g.evidence) > 220:
        p.append("evidence exceeds length limit")
    # MC consistency: under wrong_choice_zero a wrong selection cannot score > 0
    if pack.grading_policy == "wrong_choice_zero" and selection_correct is False and g.score > 0:
        p.append("nonzero score with wrong choice under wrong_choice_zero")
    if pack.grading_policy == "choice_only" and g.met_ids():
        p.append("rubric items on a choice_only question")

    ev = validate_evidence(credited=g.credited(), transcription=transcription,
                           specs=pack.rubric_specs(), policy=pack.evidence_policy)
    p.extend(ev.problems)
    inv = check_question_invariants(g, pack)
    p.extend(inv.problems)
    return GradeValidation(not p, p, ev.as_dict(), inv.as_dict())


@dataclass
class GradeDecision:
    outcome: Literal["auto", "review"]
    result: Optional[GradeResult]
    stage: str                          # none | primary | primary_rag | escalated
    problems: list[str] = field(default_factory=list)
    reason: str = ""
    status: str = "GRADE_OK"            # signals.GradeStatus
    signals: GradingSignals = field(default_factory=GradingSignals)


def escalate_grade(*, pack: QuestionGradingPack, selected: str | None, transcription: str,
                   version: str | None, selection_correct: bool | None, gateway,
                   meta: dict | None = None, primary_task: str = "grade_primary",
                   escalate_task: str = "grade_escalate", score_tolerance: float = 0.5,
                   rag_attach=None) -> GradeDecision:
    """``rag_attach(pack) -> pack`` is the injected grading-side retrieval. It
    is consulted ONLY where the pack's RAG policy says so:

        RAG_DISABLED / RAG_ALWAYS   never called here (RAG_ALWAYS is already
                                    baked into the pack at build time)
        RAG_ON_UNCERTAIN            after an unclean primary, retry the PRIMARY
                                    grader once with course context
        RAG_ON_ESCALATION           give the escalation grader course context

    Retrieval is grading-side only: the query is built from the question and
    rubric, never from the student's words (see gradingpack.rag_query).
    """
    m = {**(meta or {}), "pack_hash": pack.hash, "question_id": pack.question_id}
    blocks = grade_prompt(pack, selected=selected, transcription=transcription, version=version)
    try:
        primary = gateway.call(task=primary_task, system=GRADE_SYSTEM, content_blocks=blocks,
                               output_model=GradeResult, meta={**m, "stage": "grade"}).value
    except Exception as e:  # noqa: BLE001
        return GradeDecision("review", None, "none", [f"primary failed: {type(e).__name__}"],
                             "primary grader failed", "GRADE_INVALID",
                             GradingSignals(schema_valid=False))
    v = validate_grade(primary, pack, selection_correct=selection_correct, selected=selected,
                       transcription=transcription)
    sig = _grading_signals(primary, v)
    if v.ok:
        return GradeDecision("auto", primary, "primary", reason="primary clean",
                             status="GRADE_OK", signals=sig)
    status = grade_status_from(validation_ok=v.ok, uncertain=primary.uncertain)

    policy = getattr(pack, "rag_policy", "RAG_ALWAYS")
    rag_pack = None
    if policy == "RAG_ON_UNCERTAIN" and rag_attach is not None:
        rag_pack = rag_attach(pack)
        rag_blocks = grade_prompt(rag_pack, selected=selected, transcription=transcription,
                                  version=version)
        try:
            retried = gateway.call(task=primary_task, system=GRADE_SYSTEM, content_blocks=rag_blocks,
                                   output_model=GradeResult,
                                   meta={**m, "pack_hash": rag_pack.hash, "stage": "grade_rag"}).value
        except Exception:  # noqa: BLE001 — a failed retry just leaves us where we were
            retried = None
        if retried is not None:
            v_rag = validate_grade(retried, rag_pack, selection_correct=selection_correct,
                                   selected=selected, transcription=transcription)
            if v_rag.ok:
                sig_rag = _grading_signals(retried, v_rag)
                sig_rag.rag_used = True
                return GradeDecision("auto", retried, "primary_rag", v.problems,
                                     "course context resolved the primary grading",
                                     "GRADE_OK", sig_rag)

    try:
        gateway.route(escalate_task)
    except Exception:  # noqa: BLE001
        return GradeDecision("review", primary, "primary", v.problems,
                             "inconsistent; no escalation model configured", status, sig)
    esc_pack = pack
    if policy == "RAG_ON_ESCALATION" and rag_attach is not None:
        esc_pack = rag_attach(pack)
    elif rag_pack is not None:
        esc_pack = rag_pack
    esc_blocks = (blocks if esc_pack is pack else
                  grade_prompt(esc_pack, selected=selected, transcription=transcription,
                               version=version))
    try:
        second = gateway.call(task=escalate_task, system=GRADE_SYSTEM, content_blocks=esc_blocks,
                              output_model=GradeResult,
                              meta={**m, "pack_hash": esc_pack.hash, "stage": "escalation"}).value
    except Exception as e:  # noqa: BLE001
        return GradeDecision("review", primary, "primary", v.problems + [f"escalation failed: {type(e).__name__}"],
                             "escalation failed", status, sig)
    v2 = validate_grade(second, esc_pack, selection_correct=selection_correct, selected=selected,
                        transcription=transcription)
    consistent = (v2.ok and abs(second.score - primary.score) <= score_tolerance
                  and set(second.met_ids()) == set(primary.met_ids()))
    sig2 = _grading_signals(second, v2)
    sig2.rag_used = esc_pack is not pack
    sig2.primary_score = primary.score
    sig2.escalation_score = second.score
    sig2.score_delta = round(abs(second.score - primary.score), 4)
    sig2.primary_escalation_agreement = bool(consistent)
    if v2.ok and (consistent or primary.uncertain and not second.uncertain):
        # second stage clean AND either agrees with primary, or resolves the
        # primary's declared uncertainty with a clean, self-consistent grade
        return GradeDecision("auto", second, "escalated", v.problems,
                             "escalation resolved consistently", "GRADE_OK", sig2)
    return GradeDecision("review", second if v2.ok else primary, "escalated", v.problems + v2.problems,
                         "unresolved disagreement after escalation", "GRADE_DISAGREEMENT", sig2)


def _grading_signals(g: GradeResult, v: GradeValidation) -> GradingSignals:
    ev = v.evidence or {}
    return GradingSignals(
        schema_valid=True, invariants_ok=bool((v.invariants or {}).get("ok", True)),
        invariant_problems=list((v.invariants or {}).get("problems", [])),
        evidence_checked=ev.get("checked"), evidence_verified=len(ev.get("verified", [])),
        evidence_fabricated=len(ev.get("fabricated", [])), evidence_missing=len(ev.get("missing", [])),
        rubric_consistent=v.ok, primary_score=g.score, explicit_uncertainty=bool(g.uncertain))


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
