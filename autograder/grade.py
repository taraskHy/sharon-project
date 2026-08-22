"""Grading: deterministic scoring + LLM judging of written explanations.

All point arithmetic, version detection, caps, and ambiguity policy are plain
Python (testable offline). The only model involvement at this stage is judging
whether a written explanation expresses the key's reasoning.
"""

from __future__ import annotations

import json

from .config import GraderConfig
from .backends import VisionBackend
from .prompts import JUDGE_SYSTEM
from .schema import (
    AnswerKey,
    ExamExtraction,
    ExamResult,
    ExamSurvey,
    ExplanationEvaluation,
    ExplanationJudgement,
    KeyQuestion,
    QuestionExtraction,
    QuestionResult,
    ReviewItem,
    SubItemExtraction,
    SubItemResult,
)

# Hebrew option letters -> Latin canonical form: moved to policies.py (the
# deterministic, no-model module) so autograder.eligibility and the labeling
# app never import the grading pipeline; re-exported here for existing callers.
from .policies import _HEBREW_LETTERS, normalize_answer  # noqa: F401  (re-export)

_VERDICT_FACTOR_KEYS = ("valid", "partially_valid")


class PipelineStateError(RuntimeError):
    """The answer key and the extraction disagree about the exam's structure
    (typically a stale --resume intermediate)."""


def _accepted(sub_key, version: str) -> list[str]:
    answers = sub_key.correct_by_version.get(version)
    if answers is None:
        answers = sub_key.correct_by_version.get("default", [])
    return [a for a in (normalize_answer(x) for x in answers) if a]


# --------------------------------------------------------------------------
# Version detection
# --------------------------------------------------------------------------


class VersionDecision:
    def __init__(self, version: str, description: str, uncertain: bool):
        self.version = version
        self.description = description
        self.uncertain = uncertain


def detect_version(
    key: AnswerKey,
    extraction: ExamExtraction,
    config: GraderConfig,
) -> VersionDecision:
    """Pick the exam version whose key best agrees with the student's answers.

    A human grader does the same when the version isn't printed on the scan:
    the version whose answer set the student's unambiguous answers align with
    is almost certainly the form the student sat.
    """
    versions = key.versions or ["default"]
    if config.version != "auto":
        if config.version not in versions:
            raise ValueError(
                f"--version {config.version!r} is not one of the key's versions {versions}"
            )
        return VersionDecision(config.version, "version supplied by user", False)
    if len(versions) == 1:
        return VersionDecision(versions[0], "answer key defines a single version", False)

    scores = {v: 0 for v in versions}
    counted = 0
    for q in key.questions:
        try:
            ext = extraction.question(q.id)
        except KeyError:
            continue
        sub_by_id = {s.id: s for s in q.sub_items}
        for se in ext.sub_items:
            if se.status != "answered":
                continue
            ans = normalize_answer(se.final_answer)
            ks = sub_by_id.get(se.sub_item_id)
            if ans is None or ks is None:
                continue
            # Only sub-items with an entry for every version can compare
            # versions fairly; a partially-filled map would skew detection
            # towards whichever versions happen to be present.
            if not all(_accepted(ks, v) for v in versions):
                continue
            counted += 1
            for v in versions:
                if ans in _accepted(ks, v):
                    scores[v] += 1

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_v, best_s = ranked[0]
    second_s = ranked[1][1] if len(ranked) > 1 else 0
    description = (
        f"auto-detected from answer agreement over {counted} answered sub-items: "
        + ", ".join(f"{v}={s}" for v, s in ranked)
        + f" -> {best_v}"
    )
    uncertain = (best_s - second_s) < config.version_margin
    if uncertain:
        description += (
            f" (UNCERTAIN: margin {best_s - second_s} below threshold "
            f"{config.version_margin}; human review advised)"
        )
    return VersionDecision(best_v, description, uncertain)


# --------------------------------------------------------------------------
# Explanation judging (LLM)
# --------------------------------------------------------------------------


def _question_needs_judging(q: KeyQuestion) -> bool:
    return (
        q.explanation_required
        or q.explanation_weight > 0
        or q.type in ("selection_with_explanation", "matching_with_explanation", "open")
    )


_POLICIES: dict[str, str] = {}
_POLICY_MIN_CONF = 0.9
_EARLY_EXIT_LOG: list[dict] = []


def set_grading_policies(policies: dict[str, str] | None, min_confidence: float = 0.9) -> None:
    """Install per-question grading policies (question_id -> policy name).
    Empty/None restores today's behavior (every transcribed explanation is
    judged). Policies gate explanation judging BEFORE any model call."""
    global _POLICIES, _POLICY_MIN_CONF
    _POLICIES = dict(policies or {})
    _POLICY_MIN_CONF = min_confidence
    _EARLY_EXIT_LOG.clear()


def early_exit_log() -> list[dict]:
    """Persisted record of every deterministic early exit taken this run."""
    return list(_EARLY_EXIT_LOG)


def _policy_gate(q: KeyQuestion, se: SubItemExtraction, version: str):
    """Return an EarlyExitDecision for this sub-item, or None when no policy
    is installed for the question (-> unchanged legacy path)."""
    policy = _POLICIES.get(q.id)
    if not policy:
        return None
    from .policies import MCResolution, decide_before_ocr

    ks = next((s for s in q.sub_items if s.id == se.sub_item_id), None)
    if ks is None:
        return None
    sel = normalize_answer(se.final_answer)
    state = ("single_mark" if se.status == "answered" and sel
             else "blank" if se.status == "unanswered"
             else "multiple_marks" if se.status == "ambiguous" else "unclear")
    mc = MCResolution(sel, state, float(se.confidence or 0.0), "deterministic",
                      list(se.candidate_answers or []))
    return decide_before_ocr(policy=policy, mc=mc, accepted=_accepted(ks, version),
                             points_selection=float(ks.points), points_max=float(ks.points),
                             min_confidence=_POLICY_MIN_CONF)


def judge_question(
    llm: VisionBackend,
    q: KeyQuestion,
    ext_q: QuestionExtraction,
    version: str,
) -> dict[str, ExplanationEvaluation]:
    """Judge every sub-item that has a transcribed explanation. Sub-items with
    no transcription are marked 'missing' locally without a model call."""
    evaluations: dict[str, ExplanationEvaluation] = {}
    sub_by_id = {s.id: s for s in q.sub_items}
    to_judge: list[SubItemExtraction] = []

    for se in ext_q.sub_items:
        gate = _policy_gate(q, se, version)
        if gate is not None and gate.skip_explanation:
            # Deterministic early exit: the configured policy settles this
            # sub-item from the MC selection alone — NO explanation judging,
            # NO model call, regardless of whether an explanation exists.
            _EARLY_EXIT_LOG.append({"question_id": q.id, "sub_item_id": se.sub_item_id,
                                    "policy": _POLICIES.get(q.id), "flag": gate.persist_flag,
                                    "reason": gate.reason})
            evaluations[se.sub_item_id] = ExplanationEvaluation(
                sub_item_id=se.sub_item_id,
                verdict="missing",
                reasoning=f"[{gate.persist_flag}] {gate.reason} — explanation not evaluated by policy",
            )
            continue
        text = (se.explanation_transcription or "").strip()
        if text:
            to_judge.append(se)
        elif se.explanation_legibility in ("illegible", "partial"):
            # Writing exists but could not be transcribed: this is uncertainty,
            # not absence — it must route to human review, never a silent zero.
            evaluations[se.sub_item_id] = ExplanationEvaluation(
                sub_item_id=se.sub_item_id,
                verdict="illegible",
                reasoning=(
                    "the student wrote an explanation but it could not be read "
                    f"reliably (legibility: {se.explanation_legibility})"
                ),
            )
        else:
            evaluations[se.sub_item_id] = ExplanationEvaluation(
                sub_item_id=se.sub_item_id,
                verdict="missing",
                reasoning="no written explanation found for this sub-item",
            )

    if not to_judge:
        return evaluations

    items_payload = []
    for se in to_judge:
        ks = sub_by_id.get(se.sub_item_id)
        items_payload.append(
            {
                "sub_item_id": se.sub_item_id,
                "question": ks.prompt if ks else "",
                "accepted_answers": _accepted(ks, version) if ks else [],
                "reference_reasoning": (ks.reference_explanation if ks else None) or "(none given in key)",
                "student_selected_answer": normalize_answer(se.final_answer),
                "student_explanation_transcription": se.explanation_transcription,
                "transcription_legibility": se.explanation_legibility,
            }
        )

    blocks = [
        {
            "type": "text",
            "text": (
                f"Question {q.id}: {q.title}\n"
                + (f"Rubric notes: {q.grading_notes}\n" if q.grading_notes else "")
                + "\nSub-items to judge:\n"
                + json.dumps(items_payload, ensure_ascii=False, indent=1)
                + "\n\nReturn one evaluation per sub-item listed above."
            ),
        }
    ]
    judgement = llm.parse(
        system=JUDGE_SYSTEM,
        content_blocks=blocks,
        output_model=ExplanationJudgement,
        max_tokens=16000,
    )
    for ev in judgement.evaluations:
        evaluations[ev.sub_item_id] = ev

    # Anything the judge failed to return gets flagged rather than guessed.
    for se in to_judge:
        if se.sub_item_id not in evaluations:
            evaluations[se.sub_item_id] = ExplanationEvaluation(
                sub_item_id=se.sub_item_id,
                verdict="illegible",
                reasoning="judge did not return an evaluation for this sub-item",
            )
    return evaluations


def judge_all(
    llm: VisionBackend,
    key: AnswerKey,
    extraction: ExamExtraction,
    version: str,
    progress=None,
) -> dict[str, dict[str, ExplanationEvaluation]]:
    out: dict[str, dict[str, ExplanationEvaluation]] = {}
    for q in key.questions:
        if not _question_needs_judging(q):
            continue
        if progress:
            progress(f"judging explanations for question {q.id}")
        try:
            ext_q = extraction.question(q.id)
        except KeyError as e:
            raise PipelineStateError(
                f"the extraction has no data for question {q.id} — likely a stale "
                "--resume intermediate; re-run without --resume"
            ) from e
        out[q.id] = judge_question(llm, q, ext_q, version)
    return out


# --------------------------------------------------------------------------
# Deterministic scoring
# --------------------------------------------------------------------------


def _verdict_factor(verdict: str | None, config: GraderConfig) -> float:
    if verdict == "valid":
        return 1.0
    if verdict == "partially_valid":
        return config.partial_explanation_factor
    return 0.0


def _grade_sub_item(
    q: KeyQuestion,
    se: SubItemExtraction,
    evaluation: ExplanationEvaluation | None,
    version: str,
    config: GraderConfig,
) -> SubItemResult:
    sub_key = next((s for s in q.sub_items if s.id == se.sub_item_id), None)
    if sub_key is None:
        raise PipelineStateError(
            f"extraction reports sub-item {se.sub_item_id!r} that does not exist in "
            f"the key for question {q.id} — likely a stale --resume intermediate; "
            "re-run without --resume"
        )
    accepted = _accepted(sub_key, version)
    max_pts = sub_key.points
    student_answer = normalize_answer(se.final_answer)

    needs_review = bool(se.uncertainty_note) or se.confidence < 0.7
    review_reasons: list[str] = []
    if se.uncertainty_note:
        review_reasons.append(f"extraction uncertainty: {se.uncertainty_note}")
    elif se.confidence < 0.7:
        review_reasons.append(f"low extraction confidence ({se.confidence:.2f})")
    if version in getattr(sub_key, "versions_unverified", []):
        needs_review = True
        review_reasons.append(
            f"the answer key's value for version {version} on this sub-item is "
            "deterministically unverified (colour-only encoding); confirm "
            "against the official key"
        )

    if se.status == "unanswered":
        return SubItemResult(
            question_id=q.id,
            sub_item_id=se.sub_item_id,
            question_type=q.type,
            status="unanswered",
            student_answer=None,
            accepted_answers=accepted,
            selection_correct=None,
            explanation_transcription=se.explanation_transcription,
            explanation_evaluation=evaluation,
            points_selection=0.0,
            points_explanation=0.0,
            points_total=0.0,
            points_max=max_pts,
            reason="; ".join(["no answer was given", *review_reasons]),
            needs_review=needs_review,
            uncertainty_note=se.uncertainty_note,
        )

    # Open (free-text) questions carry no selection: all points ride on the
    # judged explanation.
    if q.type == "open":
        factor = _verdict_factor(evaluation.verdict if evaluation else None, config)
        points = max_pts * factor
        verdict_txt = evaluation.verdict if evaluation else "missing"
        if verdict_txt == "illegible":
            needs_review = True
            review_reasons.append("the written answer could not be read reliably")
        reasons = [
            f"free-text answer judged {verdict_txt}"
            + (f": {evaluation.reasoning}" if evaluation else "")
        ]
        reasons.extend(review_reasons)
        return SubItemResult(
            question_id=q.id,
            sub_item_id=se.sub_item_id,
            question_type=q.type,
            status=se.status,
            student_answer=None,
            accepted_answers=[],
            selection_correct=None,
            explanation_transcription=se.explanation_transcription,
            explanation_evaluation=evaluation,
            points_selection=0.0,
            points_explanation=round(points, 4),
            points_total=round(points, 4),
            points_max=max_pts,
            reason="; ".join(reasons),
            needs_review=needs_review,
            uncertainty_note=se.uncertainty_note,
        )

    if se.status == "ambiguous" or student_answer is None:
        candidates = ", ".join(se.candidate_answers) or "unknown"
        return SubItemResult(
            question_id=q.id,
            sub_item_id=se.sub_item_id,
            question_type=q.type,
            status="ambiguous",
            student_answer=None,
            accepted_answers=accepted,
            selection_correct=None,
            explanation_transcription=se.explanation_transcription,
            explanation_evaluation=evaluation,
            points_selection=0.0,
            points_explanation=0.0,
            points_total=0.0,
            points_max=max_pts,
            reason=(
                "the student's final intention could not be determined "
                f"(candidates: {candidates}); no points awarded pending human review — "
                f"{se.interpretation_rationale}"
            ),
            needs_review=True,
            uncertainty_note=se.uncertainty_note or se.interpretation_rationale,
        )

    if not accepted:
        # The key defines no accepted answers for this sub-item under the
        # graded version. That is a key defect, not a wrong answer — flag it
        # instead of scoring the student down.
        return SubItemResult(
            question_id=q.id,
            sub_item_id=se.sub_item_id,
            question_type=q.type,
            status=se.status,
            student_answer=student_answer,
            accepted_answers=[],
            selection_correct=None,
            explanation_transcription=se.explanation_transcription,
            explanation_evaluation=evaluation,
            points_selection=0.0,
            points_explanation=0.0,
            points_total=0.0,
            points_max=max_pts,
            reason=(
                f"the answer key defines no accepted answers for version "
                f"{version!r} on this sub-item — cannot grade automatically; "
                "human review required"
            ),
            needs_review=True,
            uncertainty_note="answer key incomplete for this sub-item/version",
        )

    selection_correct = student_answer in accepted
    verdict = evaluation.verdict if evaluation else None
    reasons: list[str] = []

    explanation_relevant = _question_needs_judging(q)
    w = q.explanation_weight if explanation_relevant else 0.0

    if w > 0:
        sel_max = max_pts * (1 - w)
        exp_max = max_pts * w
        points_selection = sel_max if selection_correct else 0.0
        points_explanation = exp_max * _verdict_factor(verdict, config)
        reasons.append(
            f"selection '{student_answer}' is "
            + ("correct" if selection_correct else f"incorrect (accepted: {', '.join(accepted)})")
        )
        if evaluation:
            reasons.append(f"explanation judged {verdict}: {evaluation.reasoning}")
    elif explanation_relevant and q.explanation_required:
        factor = _verdict_factor(verdict, config)
        if selection_correct and factor > 0:
            points_selection = max_pts * factor
            reasons.append(
                f"selection '{student_answer}' is correct and the explanation was judged "
                f"{verdict}" + ("" if factor == 1.0 else f" (partial credit x{factor})")
            )
        elif selection_correct:
            points_selection = 0.0
            reasons.append(
                f"selection '{student_answer}' is correct but the rubric awards no credit "
                f"without a valid explanation (explanation judged {verdict or 'missing'})"
            )
            if not (se.explanation_transcription or "").strip():
                # A correct selection zeroed on an EMPTY transcription is
                # indistinguishable from a transcription failure (a live model
                # limitation on Hebrew handwriting) — never a silent zero.
                needs_review = True
                review_reasons.append(
                    "correct selection gated to zero on an empty explanation "
                    "transcription — the explanation may exist on the sheet "
                    "but be untranscribed; verify on the scan"
                )
        else:
            points_selection = 0.0
            reasons.append(
                f"selection '{student_answer}' is incorrect (accepted: {', '.join(accepted)})"
            )
        if evaluation and evaluation.verdict in _VERDICT_FACTOR_KEYS:
            reasons.append(f"explanation assessment: {evaluation.reasoning}")
        points_explanation = 0.0
    else:
        points_selection = max_pts if selection_correct else 0.0
        points_explanation = 0.0
        reasons.append(
            f"selection '{student_answer}' is "
            + ("correct" if selection_correct else f"incorrect (accepted: {', '.join(accepted)})")
        )

    # A correct explanation attached to a wrong selection is a known
    # copying-slip pattern; flag it for a human instead of deciding either way.
    if (
        evaluation
        and evaluation.explanation_matches_different_answer
        and not selection_correct
    ):
        needs_review = True
        review_reasons.append(
            "explanation correctly justifies option "
            f"'{normalize_answer(evaluation.explanation_matches_different_answer)}' "
            "rather than the selected one — possible copying slip; a human may "
            "decide to award credit"
        )
    if verdict == "illegible":
        needs_review = True
        review_reasons.append("explanation could not be read reliably")

    if review_reasons:
        reasons.extend(review_reasons)

    total = round(points_selection + points_explanation, 4)
    return SubItemResult(
        question_id=q.id,
        sub_item_id=se.sub_item_id,
        question_type=q.type,
        status="answered",
        student_answer=student_answer,
        accepted_answers=accepted,
        selection_correct=selection_correct,
        explanation_transcription=se.explanation_transcription,
        explanation_evaluation=evaluation,
        points_selection=round(points_selection, 4),
        points_explanation=round(points_explanation, 4),
        points_total=total,
        points_max=max_pts,
        reason="; ".join(reasons),
        needs_review=needs_review,
        uncertainty_note=se.uncertainty_note,
    )


def grade_exam(
    key: AnswerKey,
    extraction: ExamExtraction,
    judgements: dict[str, dict[str, ExplanationEvaluation]],
    version_decision: VersionDecision,
    config: GraderConfig,
    survey: ExamSurvey | None = None,
    exam_file: str = "",
    graded_at: str = "",
    model: str = "",
) -> ExamResult:
    question_results: list[QuestionResult] = []
    unanswered: list[ReviewItem] = []
    needs_review: list[ReviewItem] = []

    for q in key.questions:
        try:
            ext_q = extraction.question(q.id)
        except KeyError as e:
            raise PipelineStateError(
                f"the extraction has no data for question {q.id} — likely a stale "
                "--resume intermediate; re-run without --resume"
            ) from e
        q_judgements = judgements.get(q.id, {})
        sub_results = [
            _grade_sub_item(q, se, q_judgements.get(se.sub_item_id), version_decision.version, config)
            for se in ext_q.sub_items
        ]
        raw = sum(s.points_total for s in sub_results)
        capped = raw > q.max_points
        awarded = min(raw, q.max_points)

        for s in sub_results:
            if s.status == "unanswered":
                unanswered.append(
                    ReviewItem(question_id=q.id, sub_item_id=s.sub_item_id, reason=s.reason)
                )
            if s.needs_review:
                needs_review.append(
                    ReviewItem(question_id=q.id, sub_item_id=s.sub_item_id, reason=s.reason)
                )

        n_correct = sum(1 for s in sub_results if s.selection_correct)
        summary = (
            f"{n_correct}/{len(sub_results)} sub-items correct; "
            f"{round(awarded, 2)}/{q.max_points} points"
            + (f" (raw {round(raw, 2)} capped at {q.max_points})" if capped else "")
        )
        question_results.append(
            QuestionResult(
                question_id=q.id,
                question_type=q.type,
                points_awarded=round(awarded, 4),
                points_max=q.max_points,
                sub_results=sub_results,
                capped=capped,
                summary=summary,
            )
        )

    if version_decision.uncertain:
        needs_review.insert(
            0,
            ReviewItem(
                question_id="*",
                sub_item_id="*",
                reason=f"exam version detection is uncertain: {version_decision.description}",
            ),
        )

    mark_interpretations: list[str] = []
    if survey:
        for note in survey.marking_conventions:
            mark_interpretations.append(
                f"convention note (p.{note.page_number}): \"{note.verbatim_text}\" -> "
                f"{note.interpretation} [scope: {note.scope}]"
            )
        if survey.grader_annotations_description:
            mark_interpretations.append(
                f"instructor annotations ignored: {survey.grader_annotations_description}"
            )
    for ext_q in extraction.questions:
        mark_interpretations.append(
            f"question {ext_q.question_id}: answers read from {ext_q.authoritative_source}"
        )
        for se in ext_q.sub_items:
            cancelled = [m for m in se.marks_observed if m.meaning == "cancelled"]
            if cancelled or len(se.marks_observed) > 1:
                mark_interpretations.append(
                    f"question {ext_q.question_id} sub-item {se.sub_item_id}: "
                    f"{se.interpretation_rationale}"
                )

    total_awarded = sum(qr.points_awarded for qr in question_results)
    # The per-question caps are the ground truth for the maximum; the key's
    # own printed total may disagree (LLM parse slip or a rubric quirk).
    total_max = sum(q.max_points for q in key.questions)
    if abs(total_max - key.total_points) > 1e-9:
        mark_interpretations.append(
            f"note: the answer key states total_points={key.total_points:g} but the "
            f"question maxima sum to {total_max:g}; using the per-question sum"
        )
    result = ExamResult(
        exam_file=exam_file,
        graded_at=graded_at,
        model=model,
        detected_version=version_decision.version,
        version_detection=version_decision.description,
        total_awarded=round(total_awarded, 4),
        total_max=total_max,
        questions=question_results,
        unanswered=unanswered,
        needs_human_review=needs_review,
        mark_interpretations=mark_interpretations,
    )

    # Deterministic self-check: the arithmetic above is plain Python, so a
    # violation here means a real defect, not a model mistake. It is reported
    # (and routed to a human), never silently corrected.
    from .invariants import check_exam_invariants

    inv = check_exam_invariants(result, key)
    if not inv.ok:
        result.mark_interpretations.append(
            "grade invariants violated: " + "; ".join(inv.problems[:5]))
        result.needs_human_review.insert(0, ReviewItem(
            question_id="*", sub_item_id="*",
            reason=("this exam's scores fail a deterministic consistency check "
                    f"({len(inv.problems)} problem(s): {'; '.join(inv.problems[:3])}) — "
                    "the totals must be verified before the grade is used")))
    return result
