"""The reliability grading route, wired at the real explanation-grading seam.

Three explicit modes (``--grading-mode``):

``legacy``       the validated ``ExplanationJudgement`` path, unchanged. Default.
``reliability``  this module decides each written answer:

                     MC resolution + grading-policy early exit
                       (BEFORE any OCR or grading work)
                       -> lazy explanation OCR where deferred (gateway task
                          ocr_primary — only items that survived the gate)
                       -> frozen transcription
                       -> typed OCR status (never "fixed" by a stronger grader)
                       -> grade_primary through the gateway
                       -> evidence validation + question invariants
                       -> escalation / grading RAG where the policy allows
                       -> AUTO / REVIEW / PAUSED
                       -> deterministic final score (grade_exam, as always)

``shadow``       BOTH routes execute; the legacy grade stays authoritative and
                 the reliability route only records what it WOULD have decided.

Two invariants hold in every mode:

* the final number is computed by the deterministic scorer in ``grade.py``.
  The grader's own score is recorded as a PROPOSAL and never written to a
  student's result.
* nothing here is provider-specific: every model call goes through the task
  gateway (``ModelGateway``) and the escalation abstractions.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

from .escalation import escalate_grade, escalate_ocr
from .grade import _accepted, _question_needs_judging, normalize_answer
from .gradingpack import QuestionGradingPack
from .policies import MCResolution, decide_before_ocr
from .privacy import anonymous_item_id
from .schema import AnswerKey, ExamExtraction, ExplanationEvaluation, ReviewItem
from .signals import (DecisionSignals, GRADE_DISAGREEMENT, GRADE_INVALID, GRADE_OK,
                      GRADE_UNCERTAIN, OCR_OK, OCR_UNRESOLVED, route_item)
from .trace import DecisionRecord, DecisionTrace, DecisionTraceStore, EarlyExitLedger
from .usage import BudgetExceeded

GRADING_MODES = ("legacy", "reliability", "shadow")


class GradingModeError(ValueError):
    """The requested grading mode cannot run with the given configuration."""


@dataclass
class ReliabilityConfig:
    mode: str = "legacy"
    min_confidence: float = 0.9
    rag_policy: str = "RAG_DISABLED"
    ocr_verify: bool = True          # consult the OCR verifier on a suspicious read
    grade_escalation: bool = True    # consult the escalation grader on an unclean grade
    primary_task: str = "grade_primary"
    escalate_task: str = "grade_escalate"
    ocr_verify_task: str = "ocr_verify"

    def __post_init__(self):
        if self.mode not in GRADING_MODES:
            raise GradingModeError(f"unknown grading mode {self.mode!r} "
                                   f"(expected one of {list(GRADING_MODES)})")


@dataclass
class ItemDecision:
    question_id: str
    sub_item_id: str
    item_id: str
    final_state: str                     # AUTO | REVIEW | PAUSED
    reason_code: str
    evaluation: Optional[ExplanationEvaluation]
    proposed_score: Optional[float] = None
    max_score: Optional[float] = None
    record: Optional[DecisionRecord] = None

    def as_dict(self) -> dict:
        return {"question_id": self.question_id, "sub_item_id": self.sub_item_id,
                "item_id": self.item_id, "final_state": self.final_state,
                "reason_code": self.reason_code, "proposed_score": self.proposed_score,
                "max_score": self.max_score}


@dataclass
class ReliabilityRun:
    mode: str
    evaluations: dict[str, dict[str, ExplanationEvaluation]] = field(default_factory=dict)
    decisions: list[ItemDecision] = field(default_factory=list)
    review_items: list[ReviewItem] = field(default_factory=list)
    records: list[DecisionRecord] = field(default_factory=list)
    paused: bool = False
    pause_reason: str = ""
    #: evidencecrops.collect_crops report (provider, AVAILABLE/UNAVAILABLE,
    #: reason, counts) — set by the caller so the log/GUI can say WHY the
    #: verifier did or did not see a crop.
    evidence_crops: dict = field(default_factory=dict)

    def accounting(self) -> dict:
        ledger = EarlyExitLedger()
        ledger.extend(self.records)
        return ledger.as_dict()

    def by_state(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for d in self.decisions:
            out[d.final_state] = out.get(d.final_state, 0) + 1
        return out


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _sub_pack(pack: QuestionGradingPack, sub_points: float,
              cache: dict[tuple[str, float], QuestionGradingPack]) -> QuestionGradingPack:
    """A per-sub-item view of the question pack.

    The pack's ``max_score`` is the QUESTION maximum; one written answer is
    graded against its own sub-item maximum, or the range invariant would be
    meaningless for multi-item questions."""
    key = (pack.question_id, float(sub_points))
    hit = cache.get(key)
    if hit is not None:
        return hit
    view = copy.deepcopy(pack)
    view.max_score = float(sub_points)
    view.compute_hash()
    cache[key] = view
    return view


def _evaluation(sub_id: str, verdict: str, reasoning: str) -> ExplanationEvaluation:
    return ExplanationEvaluation(sub_item_id=sub_id, verdict=verdict, reasoning=reasoning)


def _verdict_from_score(score: float, max_score: float) -> str:
    """Map the grader's proposal onto the existing explanation verdict, which
    the deterministic scorer then turns into points. The model never supplies
    the number itself."""
    if max_score <= 0:
        return "invalid"
    ratio = score / max_score
    if ratio >= 0.999:
        return "valid"
    if ratio <= 0.001:
        return "invalid"
    return "partially_valid"


def _mc_resolution(se, min_confidence: float) -> MCResolution:
    sel = normalize_answer(se.final_answer)
    state = ("single_mark" if se.status == "answered" and sel
             else "blank" if se.status == "unanswered"
             else "multiple_marks" if se.status == "ambiguous" else "unclear")
    return MCResolution(sel, state, float(se.confidence or 0.0), "deterministic",
                        list(se.candidate_answers or []))


_SKIP_BY_FLAG = {"deterministic_choice_only": "choice_only",
                 "deterministic_zero_wrong_choice": "wrong_choice_zero"}

#: Image-quality verdicts that are FACTS, not thresholds: there is nothing to
#: read. Every other verdict is an uncalibrated threshold and may only advise.
_EVIDENCE_BACKED_QUALITY = ("BLANK", "INVALID")


def _crop_quality(crop_b64: str | None) -> Optional[str]:
    """Deterministic triage of a crop, when the caller supplied one."""
    if not crop_b64:
        return None
    import base64

    from .imagequality import triage_crop

    try:
        raw = base64.b64decode(crop_b64, validate=False)
    except Exception:  # noqa: BLE001 — an undecodable payload IS the finding
        return "INVALID"
    return triage_crop(raw).status


# --------------------------------------------------------------------------
# the route
# --------------------------------------------------------------------------


def run_reliability_judging(*, key: AnswerKey, extraction: ExamExtraction, version: str,
                            config: ReliabilityConfig, gateway=None,
                            packs: dict[str, QuestionGradingPack] | None = None,
                            policies: dict[str, str] | None = None,
                            exam_id: str = "", job_id: str = "",
                            trace_store: DecisionTraceStore | None = None,
                            crops: dict[tuple[str, str], str] | None = None,
                            rag_attach: Callable[[QuestionGradingPack], QuestionGradingPack] | None = None,
                            ocr_fn: Callable | None = None,
                            variant_source: str | None = None,
                            alignment_source: str | None = None,
                            progress: Callable[[str], None] | None = None) -> ReliabilityRun:
    """Decide every written answer. Traces are persisted AS THE ROUTE RUNS."""
    if config.mode == "legacy":
        raise GradingModeError("run_reliability_judging is not used in legacy mode")
    if gateway is None:
        raise GradingModeError(
            f"grading mode {config.mode!r} needs a task gateway (--models-config); "
            "it never calls a provider directly")
    # Run-level refusal for the REQUIRED role: a batch must never degrade a
    # missing/UNSELECTED grade_primary route into one silent REVIEW per item.
    # Optional roles (ocr_verify, grade_escalate, ocr_primary) still degrade
    # gracefully per design; the primary grader is not optional.
    try:
        gateway.route(config.primary_task)
    except Exception as e:  # noqa: BLE001 — any unusable route refuses the run
        raise GradingModeError(
            f"grading mode {config.mode!r} cannot start: required model role "
            f"{config.primary_task!r} is not usable ({e})") from e
    packs = packs or {}
    policies = policies or {}
    crops = crops or {}
    run = ReliabilityRun(mode=config.mode)
    pack_cache: dict[tuple[str, float], QuestionGradingPack] = {}

    grade_available = config.grade_escalation and _route_ok(gateway, config.escalate_task)
    ocr_available = config.ocr_verify and _route_ok(gateway, config.ocr_verify_task)

    for q in key.questions:
        if not _question_needs_judging(q):
            continue                        # same gate as the legacy judge
        try:
            ext_q = extraction.question(q.id)
        except KeyError:
            continue
        if progress:
            progress(f"[{config.mode}] grading question {q.id}")
        run.evaluations.setdefault(q.id, {})
        sub_by_id = {s.id: s for s in q.sub_items}
        pack = packs.get(q.id)
        policy = policies.get(q.id)
        for se in ext_q.sub_items:
            ks = sub_by_id.get(se.sub_item_id)
            decision = _decide_item(
                q=q, ks=ks, se=se, version=version, policy=policy, pack=pack,
                config=config, gateway=gateway, crops=crops, rag_attach=rag_attach,
                ocr_fn=ocr_fn, pack_cache=pack_cache, exam_id=exam_id, job_id=job_id,
                variant_source=variant_source, alignment_source=alignment_source,
                grade_available=grade_available, ocr_available=ocr_available,
                paused=run.paused, pause_reason=run.pause_reason)
            if decision.final_state == "PAUSED" and not run.paused:
                run.paused, run.pause_reason = True, decision.reason_code
            run.decisions.append(decision)
            if decision.record is not None:
                run.records.append(decision.record)
                if trace_store is not None:
                    trace_store.append(decision.record)   # persisted as we go
            if decision.evaluation is not None:
                run.evaluations[q.id][se.sub_item_id] = decision.evaluation
            if decision.final_state in ("REVIEW", "PAUSED"):
                run.review_items.append(ReviewItem(
                    question_id=q.id, sub_item_id=se.sub_item_id,
                    reason=f"[{decision.reason_code}] "
                           + ((decision.record.reason if decision.record else "") or
                              "the reliability route could not settle this item")))
    return run


def _route_ok(gateway, task: str) -> bool:
    try:
        gateway.route(task)
        return True
    except Exception:  # noqa: BLE001 — an unconfigured task is simply unavailable
        return False


def _decide_item(*, q, ks, se, version, policy, pack, config, gateway, crops, rag_attach,
                 ocr_fn=None, pack_cache, exam_id, job_id, variant_source, alignment_source,
                 grade_available, ocr_available, paused, pause_reason) -> ItemDecision:
    sid = se.sub_item_id
    item_id = anonymous_item_id(job_id or "job", exam_id or "exam", q.id, sid)
    sub_points = float(getattr(ks, "points", 0.0) or 0.0)
    t = DecisionTrace(exam_id, q.id, sid, item_id=item_id)
    t.package(variant=version, variant_source=variant_source,
              alignment_source=alignment_source, grading_policy=policy,
              pack_hash=(pack.hash if pack else None),
              rag_policy=(pack.rag_policy if pack else config.rag_policy))
    signals = DecisionSignals(item_id=item_id, question_id=q.id)

    def finish(state: str, code: str, reason: str, evaluation, proposed=None) -> ItemDecision:
        t.signals(signals)
        rec = t.finish(state, code, reason, points_max=sub_points)
        return ItemDecision(q.id, sid, item_id, state, code, evaluation,
                            proposed_score=proposed, max_score=sub_points, record=rec)

    if paused:
        return finish("PAUSED", "BUDGET_PAUSED",
                      f"model work paused before this item ({pause_reason})", None)

    # ---- 1. selection + policy early exit: BEFORE any OCR or grading work ---
    mc = _mc_resolution(se, config.min_confidence)
    signals.mc.cv_score = mc.confidence
    signals.mc.candidate_cells = len(mc.candidates) or (1 if mc.selected else 0)
    signals.mc.resolver_source = mc.source
    t.deterministic(f"selection {mc.selected or '—'} ({mc.state}, confidence {mc.confidence:.2f})")
    accepted = _accepted(ks, version) if ks else []
    if policy and ks is not None:
        gate = decide_before_ocr(policy=policy, mc=mc, accepted=accepted,
                                 points_selection=sub_points, points_max=sub_points,
                                 min_confidence=config.min_confidence)
        t.statuses(mc_route=f"{mc.source}:{mc.state}")
        if gate.action == "score_locally":
            skip = _SKIP_BY_FLAG.get(gate.persist_flag or "", "deterministic_mc")
            for stage in ("ocr_explanation", "grading_rag", "grade_primary", "grade_escalate"):
                t.skipped(stage, skip, detail=gate.reason,
                          avoided={"ocr": 1, "cloud": 1} if stage == "ocr_explanation"
                          else {"grading": 1, "cloud": 1} if stage == "grade_primary" else {})
            return finish("AUTO", "AUTO", gate.reason,
                          _evaluation(sid, "missing",
                                      f"[{gate.persist_flag}] {gate.reason} — explanation not "
                                      "evaluated by policy"))
        if gate.action == "review":
            t.skipped("grade_primary", "deterministic_mc", detail=gate.reason,
                      avoided={"grading": 1, "cloud": 1})
            return finish("REVIEW", "MC_UNRESOLVED", gate.reason,
                          _evaluation(sid, "missing", gate.reason))

    # ---- 1b. lazy explanation OCR: the gate above proved it is needed ------
    #
    # In reliability mode the extraction pass deliberately deferred every
    # gradeable explanation ("deferred"); the transcription happens HERE,
    # per item, through the gateway (task ocr_primary) — so an item the
    # policy gate settled deterministically never pays for OCR at all. The
    # result is written back into the extraction sub-item and becomes the
    # frozen student transcription (created once, then immutable).
    if se.explanation_legibility == "deferred" and not (se.explanation_transcription or "").strip():
        if ocr_fn is None:
            t.skipped("grade_primary", "ocr_unresolved",
                      detail="transcription was deferred but no lazy OCR is available",
                      avoided={"grading": 1, "cloud": 1})
            return finish("REVIEW", "OCR_UNRESOLVED",
                          "the explanation transcription was deferred and no lazy OCR "
                          "route is available to produce it",
                          _evaluation(sid, "illegible", "deferred transcription unavailable"))
        try:
            ocr_res = ocr_fn(q, se)
        except BudgetExceeded as e:
            t.failed("ocr_explanation", f"budget exhausted: {e}", task="ocr_primary")
            return finish("PAUSED", "BUDGET_PAUSED", f"budget exhausted: {e}", None)
        except Exception as e:  # noqa: BLE001 — an OCR-side failure is a REVIEW
            from .extract import OCRPageSelectionError
            t.failed("ocr_explanation", f"{type(e).__name__}: {e}", task="ocr_primary")
            if isinstance(e, OCRPageSelectionError):
                # No provider was contacted: the survey placed the question
                # nowhere, and the whole exam is never silently sent instead.
                return finish("REVIEW", "OCR_UNRESOLVED", str(e),
                              _evaluation(sid, "illegible",
                                          "page selection unavailable — OCR refused rather "
                                          "than sending the whole exam"))
            return finish("REVIEW", "PROVIDER_FAILED",
                          f"lazy explanation OCR failed: {type(e).__name__}",
                          _evaluation(sid, "illegible", "the OCR model could not be reached"))
        t.executed("ocr_explanation", task=ocr_res.get("task"), model=ocr_res.get("model"),
                   cache_hit=ocr_res.get("cache_hit"), usage=ocr_res.get("usage"),
                   request_id=ocr_res.get("request_id"),
                   latency_s=ocr_res.get("latency_s"),
                   cloud=_route_is_cloud(gateway, ocr_res.get("task") or "ocr_primary"))
        se.explanation_transcription = ocr_res.get("transcription")
        se.explanation_legibility = ocr_res.get("legibility") or "none"

    # ---- 2. OCR status: is the reading itself trustworthy? -----------------
    text = (se.explanation_transcription or "").strip()
    legibility = se.explanation_legibility
    crop = crops.get((q.id, sid))
    quality = _crop_quality(crop)
    if not text and legibility in ("illegible", "partial"):
        # Writing exists but could not be read: an OCR problem, never a
        # grading one. It must not be handed to a stronger grader.
        t.statuses(ocr_status=OCR_UNRESOLVED)
        signals.ocr.output_chars = 0
        signals.ocr.primary_legibility = legibility
        t.skipped("grade_primary", "ocr_unresolved",
                  detail=f"unreadable handwriting (legibility {legibility})",
                  avoided={"grading": 1, "cloud": 1})
        return finish("REVIEW", "OCR_UNRESOLVED",
                      f"the student wrote an explanation but it could not be read "
                      f"reliably (legibility: {legibility})",
                      _evaluation(sid, "illegible",
                                  f"unreadable handwriting (legibility {legibility})"))
    if not text:
        # An empty transcription may be trusted as "no writing" only when no
        # image evidence contradicts it: a non-blank crop, or a self-report
        # of a 'full' reading with no text, is an OCR problem to review —
        # never a silent AUTO 'missing' (which would zero the explanation).
        if crop is not None and quality != "BLANK":
            t.statuses(ocr_status=OCR_UNRESOLVED)
            signals.ocr.output_chars = 0
            signals.ocr.crop_quality_status = quality
            signals.ocr.primary_legibility = legibility
            t.skipped("grade_primary", "ocr_unresolved",
                      detail=f"empty transcription over a non-blank crop (quality {quality})",
                      avoided={"grading": 1, "cloud": 1})
            return finish("REVIEW", "OCR_UNRESOLVED",
                          f"the transcription is empty but the answer image is not blank "
                          f"(image quality: {quality}) — possibly unread writing",
                          _evaluation(sid, "illegible",
                                      "empty transcription over a non-blank crop"))
        if legibility == "full":
            t.statuses(ocr_status=OCR_UNRESOLVED)
            signals.ocr.output_chars = 0
            signals.ocr.primary_legibility = legibility
            t.skipped("grade_primary", "ocr_unresolved",
                      detail="contradictory OCR output: legibility 'full' with empty text",
                      avoided={"grading": 1, "cloud": 1})
            return finish("REVIEW", "OCR_UNRESOLVED",
                          "the OCR model reported a fully legible answer but returned an "
                          "empty transcription — contradictory output",
                          _evaluation(sid, "illegible", "contradictory empty OCR output"))
        t.statuses(ocr_status=OCR_OK)
        t.skipped("grade_primary", "explanation_not_required",
                  detail="no written explanation to grade", avoided={"grading": 1, "cloud": 1})
        return finish("AUTO", "AUTO", "no written explanation found for this sub-item",
                      _evaluation(sid, "missing", "no written explanation found for this sub-item"))

    # The OCR model's own structured uncertainty ('partial'/'illegible' with
    # text present) is an OCR-side signal: it must at least flag the item.
    self_declared = ([f"self_declared_{legibility}"]
                     if legibility in ("partial", "illegible") else None)
    try:
        ocr = escalate_ocr(transcription=text, crop_png_b64=crop,
                           gateway=gateway if ocr_available else None,
                           quality_status=quality, task=config.ocr_verify_task,
                           extra_suspicion=self_declared,
                           meta={"item_id": item_id, "question_id": q.id,
                                 "job_id": job_id or None, "exam_id": exam_id or None})
    except BudgetExceeded as e:
        t.failed("ocr_verify", f"budget exhausted: {e}", task=config.ocr_verify_task)
        return finish("PAUSED", "BUDGET_PAUSED", f"budget exhausted: {e}", None)
    signals.ocr = ocr.signals
    signals.ocr.primary_legibility = legibility
    if ocr.verify is not None:
        t.executed("ocr_verify", task=config.ocr_verify_task,
                   model=ocr.call_meta.get("model") or _model_for(gateway, config.ocr_verify_task),
                   cache_hit=ocr.call_meta.get("cache_hit"),
                   usage=ocr.call_meta.get("usage"),
                   request_id=ocr.call_meta.get("request_id"),
                   latency_s=ocr.call_meta.get("latency_s"),
                   cloud=ocr.call_meta.get("cloud", True),
                   reason=ocr.reason)
    elif ocr.attempted:
        # A request was actually made and failed: trace it as FAILED — never
        # as a skip with avoided-cost credit.
        t.failed("ocr_verify", ocr.reason, task=config.ocr_verify_task)
    else:
        t.skipped("ocr_verify",
                  "no_suspicion_signal" if not ocr.suspicion.suspicious else "ocr_unresolved",
                  detail=ocr.reason, avoided={"ocr": 1, "cloud": 1})
    t.statuses(ocr_status=ocr.status)
    advisory_ocr = ""
    if ocr.status == OCR_UNRESOLVED:
        # WHY the reading is doubted decides what may follow (see §5/§9):
        #
        #   evidence-backed  a verifier looked at the crop and disagreed, or the
        #                    crop is empty/undecodable  -> withhold judgement,
        #                    REVIEW, and never hand the text to a stronger grader
        #   heuristic-only   the deterministic suspicion signals fired, but they
        #                    are UNCALIBRATED -> they may flag and route, they may
        #                    NOT cost the student points, so the ordinary grader
        #                    still runs and the item is flagged for a human
        evidence_backed = ocr.verify is not None or (quality in _EVIDENCE_BACKED_QUALITY)
        r = route_item(ocr_status=OCR_UNRESOLVED, grade_status=None,
                       ocr_escalation_available=ocr_available,
                       ocr_escalation_exhausted=ocr.verify is not None,
                       grade_escalation_available=grade_available)
        if evidence_backed:
            # A stronger GRADER is never bought here: the reading is the problem.
            t.skipped("grade_primary", "ocr_unresolved", detail="transcription not trusted",
                      avoided={"grading": 1, "cloud": 1})
            return finish("REVIEW", "OCR_UNRESOLVED",
                          f"{r.explanation} ({', '.join(ocr.suspicion.signals) or ocr.reason})",
                          _evaluation(sid, "illegible", f"transcription not trusted: {ocr.reason}"))
        advisory_ocr = ("unverified reading (" + ", ".join(ocr.suspicion.signals) + "): "
                        "flagged for a human; the score is unaffected by this "
                        "uncalibrated signal")
        t.deterministic(advisory_ocr)

    # ---- 3. grading: the reading is settled, the rubric is not -------------
    if pack is None:
        t.skipped("grade_primary", "no_grading_pack",
                  detail="no grading pack for this question")
        return finish("REVIEW", "GRADE_UNCERTAIN",
                      "no question grading pack is available for this question",
                      _evaluation(sid, "illegible", "no grading pack available"))
    view = _sub_pack(pack, sub_points, pack_cache)
    selection_correct = (mc.selected in accepted) if (mc.selected and accepted) else None
    try:
        decision = escalate_grade(pack=view, selected=mc.selected, transcription=text,
                                  version=version, selection_correct=selection_correct,
                                  gateway=gateway, rag_attach=rag_attach,
                                  meta={"item_id": item_id, "question_id": q.id,
                                        "job_id": job_id or None, "exam_id": exam_id or None},
                                  primary_task=config.primary_task,
                                  escalate_task=config.escalate_task)
    except BudgetExceeded as e:
        t.failed("grade_primary", f"budget exhausted: {e}", task=config.primary_task)
        return finish("PAUSED", "BUDGET_PAUSED", f"budget exhausted: {e}", None)
    except Exception as e:  # noqa: BLE001 — a grader failure is a REVIEW, not a crash
        t.failed("grade_primary", f"{type(e).__name__}: {e}", task=config.primary_task)
        # There is NO cloud fallback: a local grading route that cannot be
        # reached (or answers malformed) parks the item for a human with a
        # typed reason. Nothing is retried on another provider.
        local = _route_is_cloud(gateway, config.primary_task) is False
        code = "LOCAL_GRADER_UNAVAILABLE" if local else "PROVIDER_FAILED"
        why = ("the local grading model could not be reached or did not return "
               "a valid grade" if local else "the grading model could not be reached")
        return finish("REVIEW", code, f"grading failed: {type(e).__name__}",
                      _evaluation(sid, "illegible", why))

    if decision.result is None and decision.stage == "none":
        # escalate_grade absorbed the primary grader's failure into a review
        # decision (provider down / malformed beyond parsing). Type it here:
        # a dead LOCAL grading route is LOCAL_GRADER_UNAVAILABLE (tier-0
        # systemic), never a per-item validation problem — and there is no
        # cloud fallback in either case.
        t.failed("grade_primary", "; ".join(decision.problems) or decision.reason,
                 task=config.primary_task)
        local = _route_is_cloud(gateway, config.primary_task) is False
        code = "LOCAL_GRADER_UNAVAILABLE" if local else "PROVIDER_FAILED"
        why = ("the local grading model could not be reached or did not return "
               "a valid grade" if local else "the grading model could not be reached")
        return finish("REVIEW", code, f"{decision.reason}: "
                      f"{'; '.join(decision.problems)[:200]}",
                      _evaluation(sid, "illegible", why))

    t.executed("grade_primary", task=config.primary_task,
               model=_model_for(gateway, config.primary_task),
               cloud=_route_is_cloud(gateway, config.primary_task))
    if decision.stage == "primary_rag":
        t.executed("grading_rag", task=config.primary_task,
                   model=_model_for(gateway, config.primary_task),
                   cloud=_route_is_cloud(gateway, config.primary_task),
                   reason="retried with course context")
    if decision.stage == "escalated":
        t.executed("grade_escalate", task=config.escalate_task,
                   model=_model_for(gateway, config.escalate_task),
                   cloud=_route_is_cloud(gateway, config.escalate_task))
    else:
        t.skipped("grade_escalate", "no_suspicion_signal", detail=decision.reason,
                  avoided={"grading": 1, "cloud": 1})
    signals.grading = decision.signals
    t.rag(policy=(pack.rag_policy if pack else config.rag_policy),
          used=bool(decision.rag_chunk_ids),
          available=(decision.signals.rag_available
                     if decision.signals.rag_available is not None
                     else (pack.rag_available if pack else None)),
          chunk_ids=list(decision.rag_chunk_ids or []), chars=decision.rag_chars)
    result = decision.result
    proposed = float(result.score) if result is not None else None
    status = (GRADE_OK if decision.outcome == "auto" else
              decision.status if decision.status in (GRADE_INVALID, GRADE_UNCERTAIN,
                                                     GRADE_DISAGREEMENT) else GRADE_UNCERTAIN)
    t.statuses(grade_status=status, proposed_score=proposed,
               evidence=_ev_dict(decision), invariants=_inv_dict(decision),
               escalation={"stage": decision.stage, "outcome": decision.outcome,
                           "score_delta": decision.signals.score_delta,
                           "problems": list(decision.problems)[:5]})
    if decision.outcome == "auto" and result is not None:
        verdict = _verdict_from_score(result.score, view.max_score)
        evaluation = _evaluation(sid, verdict, (result.evidence or decision.reason)[:200])
        reason = (f"{decision.reason}; proposal {result.score:g}/{view.max_score:g} "
                  "(the final number is computed deterministically)")
        if advisory_ocr:
            # Same evaluation, same points — only the review state differs.
            return finish("REVIEW", "OCR_UNRESOLVED", f"{advisory_ocr}. {reason}",
                          evaluation, proposed)
        return finish("AUTO", "AUTO", reason, evaluation, proposed)

    code = _reason_code(decision)
    r = route_item(ocr_status=OCR_OK, grade_status=status,
                   grade_escalation_available=grade_available,
                   grade_escalation_exhausted=decision.stage in ("escalated", "primary_rag"))
    return finish("REVIEW", code,
                  f"{decision.reason}: {'; '.join(decision.problems)[:200] or r.explanation}",
                  _evaluation(sid, "illegible", f"[{code}] {decision.reason}"[:200]), proposed)


def _reason_code(decision) -> str:
    ev = (decision.signals.evidence_fabricated or 0) + (decision.signals.evidence_missing or 0)
    if ev:
        return "EVIDENCE_INVALID"
    return {"GRADE_DISAGREEMENT": "GRADE_DISAGREEMENT",
            "GRADE_INVALID": "GRADE_INVALID"}.get(decision.status, "GRADE_UNCERTAIN")


def _ev_dict(decision) -> dict:
    s = decision.signals
    return {"checked": s.evidence_checked, "verified": s.evidence_verified,
            "fabricated": s.evidence_fabricated, "missing": s.evidence_missing}


def _inv_dict(decision) -> dict:
    return {"ok": decision.signals.invariants_ok,
            "problems": list(decision.signals.invariant_problems or [])}


def _model_for(gateway, task: str) -> Optional[str]:
    try:
        return gateway.route(task).model
    except Exception:  # noqa: BLE001
        return None


def _route_is_cloud(gateway, task: str) -> Optional[bool]:
    """Effective cloud classification of a task's route (None: no route).
    Grading is local in production, so trace rows must record what the route
    actually is instead of assuming cloud."""
    from .usage import is_cloud_route
    try:
        r = gateway.route(task)
    except Exception:  # noqa: BLE001
        return None
    return is_cloud_route(r.backend, r.base_url)


# --------------------------------------------------------------------------
# shadow comparison (§7)
# --------------------------------------------------------------------------


@dataclass
class ShadowItemComparison:
    question_id: str
    sub_item_id: str
    item_id: str
    legacy_points: Optional[float]
    reliability_points: Optional[float]     # deterministic score under the shadow route
    reliability_proposal: Optional[float]   # what the grader proposed (never authoritative)
    points_max: Optional[float]
    score_delta: Optional[float]
    legacy_review: bool
    reliability_state: str
    legacy_reason_code: str
    reliability_reason_code: str
    route_difference: str

    def as_dict(self) -> dict:
        return asdict(self)


def compare_shadow(*, legacy_result, shadow_result, run: ReliabilityRun) -> dict:
    """Compact record for a later strong-PC migration decision.

    It records ONLY what happened. No semantic judgement is made here, and
    nothing in this function can influence the authoritative result.
    """
    from .reviewqueue import classify_reason

    legacy_rows = _rows(legacy_result)
    shadow_rows = _rows(shadow_result)
    legacy_reviews = {(r.question_id, r.sub_item_id): r.reason
                      for r in getattr(legacy_result, "needs_human_review", [])}
    items: list[ShadowItemComparison] = []
    for d in run.decisions:
        k = (d.question_id, d.sub_item_id)
        lrow, srow = legacy_rows.get(k), shadow_rows.get(k)
        lp = lrow.get("points_total") if lrow else None
        sp = srow.get("points_total") if srow else None
        lreview = bool(lrow.get("needs_review")) if lrow else k in legacy_reviews
        lcode = classify_reason(legacy_reviews.get(k, "")) if lreview else "AUTO"
        items.append(ShadowItemComparison(
            question_id=d.question_id, sub_item_id=d.sub_item_id, item_id=d.item_id,
            legacy_points=lp, reliability_points=sp, reliability_proposal=d.proposed_score,
            points_max=(lrow or srow or {}).get("points_max"),
            score_delta=(None if lp is None or sp is None else round(sp - lp, 4)),
            legacy_review=lreview, reliability_state=d.final_state,
            legacy_reason_code=lcode, reliability_reason_code=d.reason_code,
            route_difference=_route_difference(lreview, d.final_state)))
    deltas = [i.score_delta for i in items if i.score_delta is not None]
    n = len(items) or 1
    return {
        "mode": "shadow",
        "authoritative": "legacy",
        "note": ("the legacy grade is authoritative; the reliability figures are a "
                 "recorded proposal and were never applied"),
        "items": [i.as_dict() for i in items],
        "totals": {
            "legacy_total": getattr(legacy_result, "total_awarded", None),
            "reliability_total": getattr(shadow_result, "total_awarded", None),
            "total_delta": (None if shadow_result is None or legacy_result is None else
                            round(shadow_result.total_awarded - legacy_result.total_awarded, 4)),
            "total_max": getattr(legacy_result, "total_max", None),
        },
        "agreement": {
            "items": len(items),
            "exact_score_agreement": round(100 * sum(1 for d in deltas if abs(d) < 1e-9) / n, 1),
            "mean_abs_delta": (round(sum(abs(d) for d in deltas) / len(deltas), 4)
                               if deltas else None),
            "legacy_review_items": sum(1 for i in items if i.legacy_review),
            "reliability_review_items": sum(1 for i in items if i.reliability_state != "AUTO"),
            "review_rate_delta_pct": round(
                100 * (sum(1 for i in items if i.reliability_state != "AUTO")
                       - sum(1 for i in items if i.legacy_review)) / n, 1),
            "route_differences": {r: sum(1 for i in items if i.route_difference == r)
                                  for r in ("same", "reliability_only_review",
                                            "legacy_only_review")},
            "reason_code_differences": sorted({
                f"{i.legacy_reason_code}->{i.reliability_reason_code}" for i in items
                if i.legacy_reason_code != i.reliability_reason_code}),
        },
        "accounting": run.accounting(),
        "states": run.by_state(),
    }


def _route_difference(legacy_review: bool, state: str) -> str:
    rel_review = state != "AUTO"
    if legacy_review == rel_review:
        return "same"
    return "reliability_only_review" if rel_review else "legacy_only_review"


def _rows(result) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for q in (getattr(result, "questions", None) or []):
        for s in q.sub_results:
            out[(q.question_id, s.sub_item_id)] = s.model_dump() if hasattr(s, "model_dump") else dict(s)
    return out


def apply_review_items(result, run: ReliabilityRun) -> None:
    """Attach the reliability route's REVIEW items to an AUTHORITATIVE result.
    Only ever called in ``reliability`` mode — never in shadow."""
    if run.mode != "reliability":
        raise GradingModeError("shadow-mode decisions may not be applied to a result")
    have = {(r.question_id, r.sub_item_id) for r in result.needs_human_review}
    for item in run.review_items:
        if (item.question_id, item.sub_item_id) not in have:
            result.needs_human_review.append(item)
            have.add((item.question_id, item.sub_item_id))
