"""Per-question grading policies + the early-exit decision that runs BEFORE
any explanation OCR or model call.

Policies:
- choice_only                       grade the MC selection locally; never OCR
- wrong_choice_zero                 wrong (confidently resolved) MC -> 0, STOP
- explanation_required_if_correct   wrong MC -> configured wrong-answer rule
                                    (skip explanation when permitted);
                                    correct MC -> continue to explanation
- explanation_can_rescue_wrong_choice   never early-exit on a wrong choice
- choice_and_explanation_independent    grade both components by weight

The MC selection must satisfy the resolution policy (state single_mark or
blank, confidence >= threshold) before ANY zero early exit is allowed —
an ambiguous MC never yields an invalid deterministic zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

POLICIES = (
    "choice_only",
    "wrong_choice_zero",
    "explanation_required_if_correct",
    "explanation_can_rescue_wrong_choice",
    "choice_and_explanation_independent",
)
Policy = Literal[
    "choice_only",
    "wrong_choice_zero",
    "explanation_required_if_correct",
    "explanation_can_rescue_wrong_choice",
    "choice_and_explanation_independent",
]

MCState = Literal["single_mark", "multiple_marks", "erased", "blank", "unclear"]


@dataclass
class MCResolution:
    """Outcome of the MC resolution chain (deterministic CV -> local -> cloud)."""

    selected: str | None            # letter or None
    state: MCState
    confidence: float               # 0..1
    source: str = "deterministic"   # deterministic | local_model | cloud_model | agreement | review
    candidates: list[str] = field(default_factory=list)

    def resolved(self, min_confidence: float = 0.9) -> bool:
        return self.state in ("single_mark", "blank") and self.confidence >= min_confidence


@dataclass
class EarlyExitDecision:
    action: Literal["score_locally", "ocr_explanation", "review"]
    reason: str
    score: float | None = None          # set when score_locally
    persist_flag: str | None = None     # e.g. deterministic_zero_wrong_choice
    skip_explanation: bool = False
    selection_correct: bool | None = None


def decide_before_ocr(*, policy: str, mc: MCResolution, accepted: list[str],
                      points_selection: float, points_max: float,
                      wrong_answer_rule: str = "zero",
                      min_confidence: float = 0.9) -> EarlyExitDecision:
    """The policy gate. Runs with NO OCR and NO model call.

    wrong_answer_rule (for explanation_required_if_correct):
      "zero"        -> wrong MC scores 0, skip explanation
      "selection"   -> wrong MC scores 0 for selection, skip explanation
      "process"     -> still process explanation
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown grading policy {policy!r}")
    resolved = mc.resolved(min_confidence)
    if not resolved:
        # An ambiguous selection can never justify a deterministic zero.
        if policy == "choice_only":
            return EarlyExitDecision("review", f"MC unresolved ({mc.state}, conf {mc.confidence:.2f}) under choice_only")
        return EarlyExitDecision("ocr_explanation", f"MC unresolved ({mc.state}); explanation processed, no early exit")
    correct = mc.selected is not None and mc.selected in accepted
    if policy == "choice_only":
        return EarlyExitDecision("score_locally", "choice_only: local MC score",
                                 score=points_selection if correct else 0.0,
                                 persist_flag="deterministic_choice_only", skip_explanation=True,
                                 selection_correct=correct)
    if policy == "wrong_choice_zero":
        if not correct:
            return EarlyExitDecision("score_locally", "wrong_choice_zero: wrong MC -> 0, no OCR",
                                     score=0.0, persist_flag="deterministic_zero_wrong_choice",
                                     skip_explanation=True, selection_correct=False)
        return EarlyExitDecision("ocr_explanation", "wrong_choice_zero: correct MC -> explanation processed",
                                 selection_correct=True)
    if policy == "explanation_required_if_correct":
        if not correct:
            if wrong_answer_rule in ("zero", "selection"):
                return EarlyExitDecision("score_locally", f"wrong MC under rule {wrong_answer_rule}: skip explanation",
                                         score=0.0, persist_flag="deterministic_zero_wrong_choice",
                                         skip_explanation=True, selection_correct=False)
            return EarlyExitDecision("ocr_explanation", "wrong MC but rule=process", selection_correct=False)
        return EarlyExitDecision("ocr_explanation", "correct MC: explanation required", selection_correct=True)
    if policy == "explanation_can_rescue_wrong_choice":
        return EarlyExitDecision("ocr_explanation", "rescue policy: explanation always processed",
                                 selection_correct=correct)
    # choice_and_explanation_independent
    return EarlyExitDecision("ocr_explanation", "independent components: MC scored locally, explanation processed",
                             selection_correct=correct)


def infer_policy_from_key(explanation_required: bool, explanation_weight: float,
                          grading_notes: str | None) -> tuple[str | None, str]:
    """Stage 1 of policy inference: deterministic rules from the key/package.
    Returns (policy or None if ambiguous, evidence)."""
    notes = (grading_notes or "").lower()
    if not explanation_required and explanation_weight <= 0:
        return "choice_only", "key: explanation not required and weight 0"
    for kw in ("wrong answer zero", "wrong choice zero", "תשובה שגויה - אפס", "תשובה שגויה 0",
               "no points if wrong", "0 for wrong"):
        if kw in notes:
            return "wrong_choice_zero", f"grading_notes matches {kw!r}"
    for kw in ("rescue", "explanation can", "credit for explanation even", "גם אם התשובה שגויה"):
        if kw in notes:
            return "explanation_can_rescue_wrong_choice", f"grading_notes matches {kw!r}"
    for kw in ("independent", "separately", "בנפרד"):
        if kw in notes:
            return "choice_and_explanation_independent", f"grading_notes matches {kw!r}"
    if explanation_weight >= 0.99:
        return "explanation_can_rescue_wrong_choice", "explanation carries all points"
    return None, "ambiguous: explanation required without an explicit wrong-choice rule"
