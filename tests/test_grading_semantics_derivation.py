"""What a GRADE_PRIMARY score MEANS — derived from source, pinned as tests.

The 2026-08-24 smoke run compared ``GradeResult.score`` directly against the
instructor's final sub-item score and reported MAE. This file establishes,
from the production code paths only, that those are not the same quantity —
and that the model is not even given the input needed to produce the second.

Pipeline for the 67 GRADE_PRIMARY cases (all
``matching_with_explanation``, ``explanation_required=True``,
``explanation_weight=0.0``, ``max_score=4.0``):

    selection          resolved DETERMINISTICALLY (CV / mc_resolve),
                       never by the grading model
    model              -> GradeResult.score, a bounded proposal
    _verdict_from_score-> "valid" | "partially_valid" | "invalid"   (3 levels)
    _verdict_factor    -> 1.0    | 0.5               | 0.0
    _grade_sub_item    -> final = max_points * factor  if selection_correct
                                = 0.0                  otherwise

``reliability._verdict_from_score`` states it outright: *"Map the grader's
proposal onto the existing explanation verdict, which the deterministic
scorer then turns into points. The model never supplies the number itself."*

No provider is contacted anywhere in this file.
"""

from __future__ import annotations

import pytest

from autograder.config import GraderConfig
from autograder.grade import _verdict_factor
from autograder.reliability import _verdict_from_score

MAX = 4.0
CFG = GraderConfig()


def final_score(*, selection_correct: bool, verdict: str | None,
                max_points: float = MAX, config: GraderConfig = CFG) -> float:
    """The deterministic composition of ``_grade_sub_item``'s
    ``explanation_required and explanation_weight == 0`` branch (grade.py:471).

    Reproduced here rather than invoked through the full extraction machinery
    so the arithmetic is visible; ``test_matches_the_production_branch`` keeps
    it honest against the real function's constants.
    """
    if not selection_correct:
        return 0.0
    return max_points * _verdict_factor(verdict, config)


# ------------------------------------------------- 1/2/3: what makes 0, 2, 4 ----


def test_partial_factor_is_one_half():
    """The only reason 2 is reachable at all."""
    assert CFG.partial_explanation_factor == 0.5


@pytest.mark.parametrize("verdict", ["valid", "partially_valid", "invalid", "missing",
                                     "illegible", None])
def test_wrong_selection_is_zero_whatever_the_explanation_says(verdict):
    """Condition for 0, case A: the selection is wrong. The explanation cannot
    rescue it under this policy — no verdict produces credit."""
    assert final_score(selection_correct=False, verdict=verdict) == 0.0


@pytest.mark.parametrize("verdict", ["invalid", "missing", "illegible", None])
def test_correct_selection_gated_to_zero_by_a_failing_explanation(verdict):
    """Condition for 0, case B: selection correct, explanation not credited.
    The explanation is a GATE — factor 0 zeroes the whole sub-item."""
    assert final_score(selection_correct=True, verdict=verdict) == 0.0


def test_exact_condition_for_two():
    """2 <=> correct selection AND a partially valid explanation. Nothing else
    in the space produces 2."""
    assert final_score(selection_correct=True, verdict="partially_valid") == 2.0
    others = {final_score(selection_correct=sc, verdict=v)
              for sc in (True, False)
              for v in ("valid", "invalid", "missing", "illegible", None)}
    assert 2.0 not in others


def test_exact_condition_for_four():
    """4 <=> correct selection AND a fully valid explanation."""
    assert final_score(selection_correct=True, verdict="valid") == 4.0
    others = {final_score(selection_correct=sc, verdict=v)
              for sc in (True, False)
              for v in ("partially_valid", "invalid", "missing", "illegible", None)}
    assert 4.0 not in others


def test_the_reachable_score_set_is_exactly_the_observed_label_set():
    """{0, 2, 4} — the instructor labels contain no value the pipeline cannot
    produce, and the pipeline produces no value the labels do not contain."""
    reachable = {final_score(selection_correct=sc, verdict=v)
                 for sc in (True, False)
                 for v in ("valid", "partially_valid", "invalid", "missing", "illegible", None)}
    assert reachable == {0.0, 2.0, 4.0}


# ------------------------------- 4: the explanation gates, it does not score ----


def test_explanation_earns_no_independent_points_it_multiplies_the_selection():
    """With ``explanation_weight = 0`` the explanation contributes no points of
    its own; it scales the selection's points by the verdict factor."""
    # all points ride on the selection...
    assert final_score(selection_correct=True, verdict="valid") == MAX
    # ...and the explanation can only scale that down, never add to a zero
    assert final_score(selection_correct=False, verdict="valid") == 0.0
    for verdict, factor in (("valid", 1.0), ("partially_valid", 0.5), ("invalid", 0.0)):
        assert final_score(selection_correct=True, verdict=verdict) == MAX * factor


# ------------------------------ 5/6: what the model's number actually means ----


@pytest.mark.parametrize("proposal,expected", [
    (0.0, "invalid"),
    (0.01, "partially_valid"),
    (1.0, "partially_valid"),
    (2.0, "partially_valid"),
    (3.9, "partially_valid"),
    (4.0, "valid"),
])
def test_model_score_is_quantised_to_three_verdict_levels(proposal, expected):
    """GradeResult.score is NOT used as a number. It is a ratio that collapses
    to one of three verdicts, so every distinction the model draws strictly
    inside (0, max) is discarded."""
    assert _verdict_from_score(proposal, MAX) == expected


def test_the_thresholds_are_on_the_ratio_not_the_raw_score():
    """Both cut points are ratio-based (<=0.001 and >=0.999 of max_score), so
    at max_score=4 there are narrow dead zones at each end: a proposal of
    0.004 is still "invalid" and 3.996 is already "valid". Anything grading
    this role must compare RATIOS, never raw numbers."""
    assert _verdict_from_score(0.004, MAX) == "invalid"
    assert _verdict_from_score(0.005, MAX) == "partially_valid"
    assert _verdict_from_score(3.996, MAX) == "valid"
    assert _verdict_from_score(3.99, MAX) == "partially_valid"


def test_the_models_effective_output_space_has_three_values():
    outcomes = {_verdict_from_score(s / 100 * MAX, MAX) for s in range(0, 101)}
    assert outcomes == {"invalid", "partially_valid", "valid"}


def test_a_grader_that_cannot_see_the_selection_can_only_ever_produce_zero():
    """The smoke-run result, derived rather than observed: with the selection
    withheld and the explanation weighted 0, a grader applying the pack's
    stated rule has no basis to award credit — and every parsed grade in the
    2026-08-24 run was indeed exactly 0.0."""
    verdict = _verdict_from_score(0.0, MAX)
    assert verdict == "invalid"
    assert final_score(selection_correct=True, verdict=verdict) == 0.0
    assert final_score(selection_correct=False, verdict=verdict) == 0.0


# --------------------------------- 7/8: invertibility of the instructor label ----


def verdict_from_final_score(score: float, max_points: float = MAX) -> str | None:
    """The unique explanation verdict implied by a final score, or None when
    the score does not determine one."""
    if score == max_points:
        return "valid"
    if score == max_points * CFG.partial_explanation_factor:
        return "partially_valid"
    return None            # 0 is ambiguous — see the test below


def test_score_four_uniquely_identifies_the_hidden_verdict():
    assert verdict_from_final_score(4.0) == "valid"


def test_score_two_uniquely_identifies_the_hidden_verdict():
    assert verdict_from_final_score(2.0) == "partially_valid"


def test_score_zero_does_not_identify_the_hidden_verdict():
    """0 is produced by SIX distinct (selection, verdict) states. Deriving an
    explanation label from a 0 would be inventing ground truth."""
    producing_zero = [(sc, v)
                      for sc in (True, False)
                      for v in ("valid", "partially_valid", "invalid", "missing", "illegible")
                      if final_score(selection_correct=sc, verdict=v) == 0.0]
    assert len(producing_zero) > 1
    assert verdict_from_final_score(0.0) is None


def test_nonzero_scores_also_pin_selection_correctness():
    """A 2 or a 4 additionally proves the selection was correct — the only
    branch that can award credit."""
    for score in (2.0, 4.0):
        assert final_score(selection_correct=False,
                           verdict=verdict_from_final_score(score)) != score


# ------------------------------------------- guard: the branch really is this ----


def test_matches_the_production_branch():
    """Guard against the derivation drifting from grade.py. If the weight
    stops being 0, or the factor table changes, this file must be revisited."""
    from autograder.grade import _VERDICT_FACTOR_KEYS

    assert _VERDICT_FACTOR_KEYS == ("valid", "partially_valid")
    assert _verdict_factor("valid", CFG) == 1.0
    assert _verdict_factor("partially_valid", CFG) == 0.5
    for v in ("invalid", "missing", "illegible", None, "anything_else"):
        assert _verdict_factor(v, CFG) == 0.0


def test_the_dataset_packs_really_are_gate_shaped():
    """The pack rule string ``explanation weight 0`` is emitted by
    gradingpack.py ONLY when explanation_required is True — its presence in a
    pack is proof that the gating branch applies to that question."""
    import inspect

    from autograder import gradingpack

    src = inspect.getsource(gradingpack)
    assert 'if q.explanation_required:' in src
    assert 'rules.append(f"explanation weight {q.explanation_weight:g}")' in src
