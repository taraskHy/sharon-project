"""The GRADE_PRIMARY benchmark target: the canonical explanation verdict.

The grading model's responsibility is NOT the final numeric sub-item score.
Production resolves the selection deterministically, asks the model to judge
the explanation, and then computes the number itself:

    GradeResult.score            the model's bounded proposal
      -> reliability._verdict_from_score(score, max_score)
             ratio >= 0.999 -> "valid"
             ratio <= 0.001 -> "invalid"
             else           -> "partially_valid"
      -> grade._verdict_factor(verdict, config)
             valid 1.0 | partially_valid 0.5 | everything else 0.0
      -> grade._grade_sub_item, explanation_required and weight == 0 branch
             final = max_points * factor   if selection_correct else 0.0

Both conversions are imported from the production modules, never reimplemented
here: a benchmark-only interpretation that drifts from production would
measure something nobody runs.

The verdict names are the ones the codebase already uses
(``schema.ExplanationVerdict``); no new vocabulary is introduced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..config import GraderConfig
from ..grade import _verdict_factor
from ..reliability import _verdict_from_score

#: The three classes a GRADING model can be held responsible for. `missing`
#: and `illegible` are OCR-side states of the pipeline (no text reached the
#: grader) and are never a grading judgement, so they are not target classes.
CANONICAL_VERDICTS: tuple[str, ...] = ("invalid", "partially_valid", "valid")

#: Verdicts that mean "the grader saw text and judged it", vs the OCR-side
#: states that must never be scored as a grading decision.
OCR_SIDE_VERDICTS: tuple[str, ...] = ("missing", "illegible")


def verdict_from_model_score(score: float, max_score: float) -> str:
    """The production conversion, verbatim. The benchmark scores THIS."""
    return _verdict_from_score(score, max_score)


def factor_for(verdict: str | None, config: GraderConfig | None = None) -> float:
    """The production verdict -> factor table, verbatim."""
    return _verdict_factor(verdict, config or GraderConfig())


def final_score_for(*, selection_correct: bool, verdict: str | None, max_points: float,
                    config: GraderConfig | None = None) -> float:
    """The production deterministic composition for the gating branch
    (``explanation_required`` and ``explanation_weight == 0``)."""
    if not selection_correct:
        return 0.0
    return max_points * factor_for(verdict, config)


# ------------------------------------------------------------- derivation ----

#: why a row is or is not usable as verdict ground truth
DERIVABLE_FULL = "full_credit_implies_valid"
DERIVABLE_PARTIAL = "partial_credit_implies_partially_valid"
DERIVABLE_ZERO = "zero_with_correct_selection_implies_invalid"
UNRESOLVED_ZERO_UNKNOWN_SELECTION = "zero_but_selection_correctness_unknown"
#: audited — a human read the marked option — but the exam VERSION was never
#: confirmed, so there is no key to compare the mark against. Distinct from
#: "nobody has looked yet": no further selection audit can resolve it.
UNRESOLVED_VERSION_UNCONFIRMED = "zero_marked_option_recorded_but_exam_version_unconfirmed"
EXCLUDED_WRONG_SELECTION = "zero_because_selection_wrong_explanation_never_scored"
UNRESOLVED_EMPTY_TRANSCRIPTION = "zero_with_no_transcription_cannot_separate_invalid_from_missing"
UNRESOLVED_UNEXPECTED_SCORE = "score_not_reachable_by_the_frozen_policy"


@dataclass(frozen=True)
class VerdictDerivation:
    """One case's verdict ground truth, or an explicit refusal to invent one."""

    case_id: str
    instructor_final_score: float | None
    selection_correct: bool | None
    derived_explanation_verdict: str | None
    derivable: bool
    derivation_reason: str
    max_points: float = 4.0
    #: what the derived verdict, put back through production, reproduces
    implied_final_score: float | None = None

    def as_row(self) -> dict:
        return {
            "case_id": self.case_id,
            "instructor_final_score": self.instructor_final_score,
            "selection_correct": self.selection_correct,
            "derived_explanation_verdict": self.derived_explanation_verdict,
            "derivation_reason": self.derivation_reason,
            "derivable": self.derivable,
            "max_points": self.max_points,
            "implied_final_score": self.implied_final_score,
        }


def derive_verdict(*, case_id: str, instructor_final_score: float | None,
                   selection_correct: bool | None, max_points: float = 4.0,
                   transcription: str | None = None,
                   config: GraderConfig | None = None) -> VerdictDerivation:
    """Invert the production composition — ONLY where the inversion is unique.

    Uniqueness, for ``final = max * factor(verdict) if selection_correct``:

    * ``final == max``            -> selection was correct AND verdict ``valid``.
      Nothing else reaches full credit, so the selection state is implied and
      does not need to be known independently.
    * ``final == max * partial``  -> selection correct AND ``partially_valid``.
      Same argument.
    * ``final == 0``              -> SIX states produce this. Resolving it needs
      the selection:
        - selection wrong  -> the explanation was never the reason for the 0;
          the case carries no explanation ground truth at all and is EXCLUDED
          (not "invalid" — that would be a fabricated label).
        - selection correct -> the factor was 0, i.e. one of
          ``invalid`` / ``missing`` / ``illegible``. The frozen dataset supplies
          an audited non-empty transcription, so the grader did see text and the
          OCR-side states are excluded, leaving ``invalid`` uniquely. Without a
          transcription we refuse rather than guess.
        - selection unknown -> unresolved. NEVER treated as False.
    """
    cfg = config or GraderConfig()
    partial = max_points * cfg.partial_explanation_factor

    def out(verdict, derivable, reason):
        implied = (final_score_for(selection_correct=True, verdict=verdict,
                                   max_points=max_points, config=cfg)
                   if derivable and verdict is not None else None)
        return VerdictDerivation(case_id=case_id,
                                 instructor_final_score=instructor_final_score,
                                 selection_correct=selection_correct,
                                 derived_explanation_verdict=verdict,
                                 derivable=derivable, derivation_reason=reason,
                                 max_points=max_points, implied_final_score=implied)

    if instructor_final_score is None:
        return out(None, False, "no authoritative instructor score")
    score = float(instructor_final_score)

    if score == max_points:
        return out("valid", True, DERIVABLE_FULL)
    if partial > 0 and score == partial:
        return out("partially_valid", True, DERIVABLE_PARTIAL)
    if score == 0.0:
        if selection_correct is None:
            return out(None, False, UNRESOLVED_ZERO_UNKNOWN_SELECTION)
        if selection_correct is False:
            return out(None, False, EXCLUDED_WRONG_SELECTION)
        if not (transcription or "").strip():
            return out(None, False, UNRESOLVED_EMPTY_TRANSCRIPTION)
        return out("invalid", True, DERIVABLE_ZERO)
    return out(None, False, UNRESOLVED_UNEXPECTED_SCORE)


@dataclass
class DerivationSummary:
    total: int = 0
    derivable: int = 0
    unresolved: int = 0
    by_split: dict = field(default_factory=dict)
    by_verdict: dict = field(default_factory=dict)
    by_reason: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"total": self.total, "derivable": self.derivable,
                "unresolved": self.unresolved, "by_split": self.by_split,
                "by_verdict": self.by_verdict, "by_reason": self.by_reason}


def summarize(derivations: Iterable[tuple[str, VerdictDerivation]]) -> DerivationSummary:
    """``derivations`` is (split, VerdictDerivation) pairs."""
    s = DerivationSummary()
    for split, d in derivations:
        s.total += 1
        bucket = s.by_split.setdefault(split, {"total": 0, "derivable": 0, "unresolved": 0,
                                               "by_verdict": {}})
        bucket["total"] += 1
        if d.derivable:
            s.derivable += 1
            bucket["derivable"] += 1
            s.by_verdict[d.derived_explanation_verdict] = \
                s.by_verdict.get(d.derived_explanation_verdict, 0) + 1
            bucket["by_verdict"][d.derived_explanation_verdict] = \
                bucket["by_verdict"].get(d.derived_explanation_verdict, 0) + 1
        else:
            s.unresolved += 1
            bucket["unresolved"] += 1
        s.by_reason[d.derivation_reason] = s.by_reason.get(d.derivation_reason, 0) + 1
    return s


__all__ = ["CANONICAL_VERDICTS", "OCR_SIDE_VERDICTS", "VerdictDerivation",
           "DerivationSummary", "verdict_from_model_score", "factor_for",
           "final_score_for", "derive_verdict", "summarize",
           "DERIVABLE_FULL", "DERIVABLE_PARTIAL", "DERIVABLE_ZERO",
           "UNRESOLVED_ZERO_UNKNOWN_SELECTION", "UNRESOLVED_VERSION_UNCONFIRMED",
           "EXCLUDED_WRONG_SELECTION",
           "UNRESOLVED_EMPTY_TRANSCRIPTION", "UNRESOLVED_UNEXPECTED_SCORE"]
