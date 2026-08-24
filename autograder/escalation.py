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
from .usage import BudgetExceeded, is_cloud_route

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
    # short technical tokens are the known dangerous class -> flag for verify.
    # Digits/operators count: a short formula ("x=3", "5+3=8") is exactly as
    # substitution-prone as a short Latin token and must not bypass review.
    if (len(_LATIN_TOKEN.findall(t)) >= 1 or _DIGIT_TOKEN.search(t)) and len(t) < 40:
        s.append("short_technical_token")
    return OCRSuspicion(bool(s), s)


class OCRVerifyResult(BaseModel):
    verdict: Literal["supported", "review"]
    omissions: list[str] = Field(default_factory=list)
    substitutions: list[str] = Field(default_factory=list)
    additions: list[str] = Field(default_factory=list)
    # Fail-closed default: an OMITTED self-assessment must never satisfy the
    # AUTO gate (which requires high/medium), so silence defaults to "low".
    confidence: Literal["high", "medium", "low"] = "low"


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
    # Call accountability: whether a verifier request was actually attempted
    # (an attempted-but-failed call must be traced as FAILED, never as a
    # skip with avoided-cost credit) and, on success, the real call
    # metadata (model/cache_hit/usage/latency/cloud) for the trace.
    attempted: bool = False
    error: str = ""
    call_meta: dict = field(default_factory=dict)


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
                 quality_status: str | None = None,
                 extra_suspicion: list[str] | None = None) -> OCRDecision:
    """OCR-side escalation ONLY. It is never triggered by grading difficulty:
    the caller reaches it because the READING is in doubt (deterministic
    suspicion signals, a pre-OCR image-quality verdict, or caller-supplied
    ``extra_suspicion`` such as the OCR model's own partial-legibility
    self-report)."""
    susp = ocr_suspicion(transcription)
    if extra_suspicion:
        susp.signals.extend(extra_suspicion)
        susp.suspicious = bool(susp.signals)
    if quality_status and quality_status != "OK":
        # An unreadable/blank/clipped crop is an imaging problem: no amount of
        # verifier agreement should turn it into AUTO.
        return OCRDecision("review", transcription, susp, reason=f"image quality {quality_status}",
                           status=OCR_UNRESOLVED_,
                           signals=_ocr_signals(transcription, susp, None, quality_status))
    if not susp.suspicious:
        return OCRDecision("auto", transcription, susp, reason="no suspicion signal",
                           status="OCR_OK", signals=_ocr_signals(transcription, susp, None, quality_status))
    if crop_png_b64 is None:
        # Fail-closed: without an evidence crop the verifier must NOT be
        # called over an arbitrary region (evidencecrops.py). The reading
        # stays unverified -> REVIEW, never AUTO.
        return OCRDecision("review", transcription, susp,
                           reason="suspicious; no evidence crop available (crop producer unavailable)",
                           status=OCR_UNRESOLVED_,
                           signals=_ocr_signals(transcription, susp, None, quality_status))
    if gateway is None:
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
    except BudgetExceeded:
        # Budget exhaustion is a job-level PAUSE signal, not a verifier
        # failure: mirror escalate_grade so the caller pauses the run instead
        # of silently degrading the item to REVIEW.
        raise
    except Exception as e:  # noqa: BLE001
        return OCRDecision("review", transcription, susp, reason=f"verifier failed: {type(e).__name__}",
                           status=OCR_UNRESOLVED_, attempted=True, error=type(e).__name__,
                           signals=_ocr_signals(transcription, susp, None, quality_status))
    v = res.value
    vd = v.model_dump()
    sig = _ocr_signals(transcription, susp, vd, quality_status)
    call_meta = {
        "model": res.route.model,
        "cache_hit": res.cache_hit,
        "usage": dict(res.usage or {}),
        "request_id": (res.usage or {}).get("request_id"),
        "latency_s": res.latency_s,
        "cloud": is_cloud_route(res.route.backend, res.route.base_url),
    }
    if v.verdict == "supported" and v.confidence in ("high", "medium") and not (v.omissions or v.substitutions or v.additions):
        sig.provider_agreement = True
        return OCRDecision("auto", transcription, susp, vd, reason="verifier supports transcription",
                           status="OCR_OK", signals=sig, attempted=True, call_meta=call_meta)
    sig.provider_agreement = False
    return OCRDecision("review", transcription, susp, vd, reason="verifier disagreement",
                       status=OCR_UNRESOLVED_, signals=sig, attempted=True, call_meta=call_meta)


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


#: grade-v3 (2026-08-25). ``score`` means EXPLANATION QUALITY ONLY.
#:
#: grade-v2 asked for "the score (within the stated maximum)" and showed the
#: pack's final-score scoring rules, so the model answered with the student's
#: FINAL SUB-ITEM GRADE. With no selection in the prompt that grade is
#: correctly 0 — every candidate returned 0.0 on every case in the 2026-08-24
#: and 2026-08-25 smoke runs, and their own evidence said why ("no letter/answer
#: given"). Production never wanted that number: it maps `score` through
#: reliability._verdict_from_score onto an explanation verdict and computes the
#: final grade itself from the deterministically-resolved selection.
#:
#: v3 makes the requested quantity match the consumed quantity.
GRADE_SYSTEM = (
    "You judge the QUALITY OF ONE WRITTEN EXPLANATION against the supplied "
    "rubric and official solution. That judgement is your ONLY task.\n\n"
    "You are NOT grading the student's overall answer, and you are NOT "
    "computing the student's final score for this question. A separate "
    "deterministic step combines your judgement with whether the student's "
    "multiple-choice / matching selection was correct. That step is not yours.\n\n"
    "Therefore:\n"
    "- Judge ONLY the text of the student explanation shown below.\n"
    "- The student's selected option is deliberately NOT part of this task. Do "
    "not reason about which option was chosen, whether one was chosen, or "
    "whether it was right.\n"
    "- Do NOT require the explanation to name, restate or identify a letter or "
    "option, unless a rubric item explicitly demands it.\n"
    "- Never lower your judgement because no selection appears in this prompt. "
    "Its absence carries no information about the explanation.\n\n"
    "Grade only according to the supplied rubric, and only what the student "
    "actually wrote. Course context, when present, is supplemental reference "
    "material for judging CORRECTNESS — it is never evidence of what the "
    "student wrote: never assume the student wrote something merely because it "
    "appears in the course context. Preserve the student's wording as given — "
    "never rewrite it.\n\n"
    "Report explanation quality in `score`, using EXACTLY one of the three "
    "values stated below the transcription. `score` is the EXPLANATION-QUALITY "
    "value only — it is NOT the student's final score for the question.\n\n"
    "Also return one entry per rubric item: its id, whether it is met, and — "
    "when met — a SHORT span copied VERBATIM from the student transcription "
    "that supports it (copy it exactly; never paraphrase, translate, correct or "
    "invent a span; spans from the course context do not count; if no span in "
    "the transcription supports the item, the item is not met). Set "
    "uncertain=true if the transcription or the rubric leaves the EXPLANATION "
    "QUALITY genuinely undecidable. Reply with ONLY the JSON object."
)


def explanation_scale(max_score: float) -> str:
    """The three explanation-quality values the model may return, spelled out.

    Production quantises ``score`` by RATIO into three verdicts
    (``reliability._verdict_from_score``), so only three values carry meaning.
    Naming them removes the model's incentive to invent intermediate numbers
    that are then silently collapsed.
    """
    return (f"  {0:g}  = invalid: the explanation is wrong, empty of relevant "
            f"content, or does not support the rubric at all\n"
            f"  {max_score / 2:g}  = partially valid: partly correct reasoning, "
            f"or correct but incomplete against the rubric\n"
            f"  {max_score:g}  = valid: correct and sufficient reasoning for this "
            f"question")


def grade_prompt(pack: QuestionGradingPack, *, selected: str | None, transcription: str,
                 version: str | None) -> list[dict]:
    """Build the explanation-quality grading prompt (grade-v3).

    THE SELECTION IS NEVER RENDERED. Under v3 the model judges the explanation
    and nothing else; the selection is resolved deterministically elsewhere and
    combined with this judgement downstream. Showing it — or showing a
    placeholder for its absence — can only bias a judgement that must not
    depend on it. ``selected`` is still accepted so callers and
    ``validate_grade`` keep one signature, and is deliberately unused here.

    (v2 rendered "Student selected option: X" and omitted the line when there
    was none. Omission was right for v2's question but is moot now: v3 never
    asks about the selection at all.)

    THE PACK'S SCORING RULES ARE NOT RENDERED either. They describe how the
    student's FINAL sub-item score is composed — e.g. "explanation weight 0",
    "no credit for an answer without an explanation". Those are downstream
    composition rules; handing them to an explanation judge is what made v2
    models reason about the missing selection and return 0.
    """
    correct = None
    if version:
        correct = {sid: v.get(version) for sid, v in pack.correct_by_version.items()}
    return [{"type": "text", "text": (
        pack.to_grader_context(include_scoring_rules=False) + "\n\n"
        + (f"Correct option(s) for this exam version: {correct}\n" if correct else "")
        + f"Student explanation (verbatim transcription):\n---\n{transcription}\n---\n"
        + f"Allowed rubric item ids: {pack.rubric_item_ids() or '(none)'}.\n"
        + "Return `score` as the EXPLANATION-QUALITY value, using exactly one of:\n"
        + explanation_scale(pack.max_score) + "\n"
        + "`score` is not the student's final score for this question.")}]


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
    rag_chunk_ids: list[str] = field(default_factory=list)   # chunks in the DECIDING request
    rag_chars: int = 0


def _rag_meta(p: QuestionGradingPack) -> dict:
    """Numbers-only RAG accounting attached to every grading call's meta, so
    the ledger can separate RAG-added input from the base context."""
    return {"rag_policy": getattr(p, "rag_policy", None),
            "rag_chars": int(getattr(p, "rag_chars", 0) or 0),
            "rag_chunks": len(getattr(p, "rag_evidence", []) or [])}


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
                               output_model=GradeResult,
                               meta={**m, **_rag_meta(pack), "stage": "grade"}).value
    except BudgetExceeded:
        # Budget exhaustion is a job-level PAUSE signal, not a grading failure:
        # the item must stay pending, not be recorded as an unresolved grade.
        raise
    except Exception as e:  # noqa: BLE001
        return GradeDecision("review", None, "none", [f"primary failed: {type(e).__name__}"],
                             "primary grader failed", "GRADE_INVALID",
                             GradingSignals(schema_valid=False))
    v = validate_grade(primary, pack, selection_correct=selection_correct, selected=selected,
                       transcription=transcription)
    sig = _grading_signals(primary, v)
    if v.ok:
        return GradeDecision("auto", primary, "primary", reason="primary clean",
                             status="GRADE_OK", signals=sig,
                             rag_chunk_ids=[e.chunk_id for e in pack.rag_evidence],
                             rag_chars=pack.rag_chars)
    status = grade_status_from(validation_ok=v.ok, uncertain=primary.uncertain)

    policy = getattr(pack, "rag_policy", "RAG_DISABLED")
    rag_pack = None
    if policy == "RAG_ON_UNCERTAIN" and rag_attach is not None:
        rag_pack = rag_attach(pack)
        if rag_pack is pack or not rag_pack.rag_evidence:
            # Optional RAG is unavailable (no course/retriever/index or the
            # search returned nothing): grading continues WITHOUT it — the
            # absence of retrieval is recorded, never a failure or a REVIEW.
            sig.rag_available = False
            rag_pack = None
    if rag_pack is not None:
        rag_blocks = grade_prompt(rag_pack, selected=selected, transcription=transcription,
                                  version=version)
        try:
            retried = gateway.call(task=primary_task, system=GRADE_SYSTEM, content_blocks=rag_blocks,
                                   output_model=GradeResult,
                                   meta={**m, **_rag_meta(rag_pack), "pack_hash": rag_pack.hash,
                                         "stage": "grade_rag"}).value
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
                                     "GRADE_OK", sig_rag,
                                     rag_chunk_ids=[e.chunk_id for e in rag_pack.rag_evidence],
                                     rag_chars=rag_pack.rag_chars)

    try:
        gateway.route(escalate_task)
    except Exception:  # noqa: BLE001
        return GradeDecision("review", primary, "primary", v.problems,
                             "inconsistent; no escalation model configured", status, sig)
    esc_pack = pack
    if policy == "RAG_ON_ESCALATION" and rag_attach is not None:
        esc_pack = rag_attach(pack)
        if esc_pack is pack or not esc_pack.rag_evidence:
            sig.rag_available = False        # degrade to no-RAG escalation, never REVIEW
            esc_pack = pack
    elif rag_pack is not None:
        esc_pack = rag_pack
    esc_blocks = (blocks if esc_pack is pack else
                  grade_prompt(esc_pack, selected=selected, transcription=transcription,
                               version=version))
    try:
        second = gateway.call(task=escalate_task, system=GRADE_SYSTEM, content_blocks=esc_blocks,
                              output_model=GradeResult,
                              meta={**m, **_rag_meta(esc_pack), "pack_hash": esc_pack.hash,
                                    "stage": "escalation"}).value
    except BudgetExceeded:
        raise
    except Exception as e:  # noqa: BLE001
        return GradeDecision("review", primary, "primary", v.problems + [f"escalation failed: {type(e).__name__}"],
                             "escalation failed", status, sig)
    v2 = validate_grade(second, esc_pack, selection_correct=selection_correct, selected=selected,
                        transcription=transcription)
    consistent = (v2.ok and abs(second.score - primary.score) <= score_tolerance
                  and set(second.met_ids()) == set(primary.met_ids()))
    sig2 = _grading_signals(second, v2)
    sig2.rag_used = esc_pack is not pack
    sig2.rag_available = sig.rag_available
    sig2.primary_score = primary.score
    sig2.escalation_score = second.score
    sig2.score_delta = round(abs(second.score - primary.score), 4)
    sig2.primary_escalation_agreement = bool(consistent)
    esc_rag_ids = [e.chunk_id for e in esc_pack.rag_evidence]
    if v2.ok and (consistent or primary.uncertain and not second.uncertain):
        # second stage clean AND either agrees with primary, or resolves the
        # primary's declared uncertainty with a clean, self-consistent grade
        return GradeDecision("auto", second, "escalated", v.problems,
                             "escalation resolved consistently", "GRADE_OK", sig2,
                             rag_chunk_ids=esc_rag_ids, rag_chars=esc_pack.rag_chars)
    return GradeDecision("review", second if v2.ok else primary, "escalated", v.problems + v2.problems,
                         "unresolved disagreement after escalation", "GRADE_DISAGREEMENT", sig2,
                         rag_chunk_ids=esc_rag_ids, rag_chars=esc_pack.rag_chars)


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
