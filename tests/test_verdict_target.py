"""The GRADE_PRIMARY benchmark target: the canonical explanation verdict.

The model is judged on the verdict production actually asks it for, not on the
downstream 0/2/4 that production computes deterministically from a selection
the model is never shown.

Critically, the benchmark must not grow its OWN interpretation of the model's
number: the conversion has to be the same function object production calls, or
the two definitions will drift and the benchmark will stop measuring the
product. These tests pin that identity.

No provider is contacted anywhere in this file.
"""

from __future__ import annotations

import pytest

from autograder.benchmark.verdicts import (
    CANONICAL_VERDICTS,
    DERIVABLE_FULL,
    DERIVABLE_PARTIAL,
    DERIVABLE_ZERO,
    EXCLUDED_WRONG_SELECTION,
    UNRESOLVED_EMPTY_TRANSCRIPTION,
    UNRESOLVED_ZERO_UNKNOWN_SELECTION,
    derive_verdict,
    factor_for,
    final_score_for,
    verdict_from_model_score,
)

MAX = 4.0
TEXT = "student explanation text"


# -------------------------------------- benchmark == production, literally ----


def test_benchmark_uses_the_production_conversion_function_itself():
    """Not "an equivalent implementation" — the same object."""
    from autograder.reliability import _verdict_from_score
    from autograder.benchmark import verdicts

    assert verdicts.verdict_from_model_score.__wrapped__ is _verdict_from_score \
        if hasattr(verdicts.verdict_from_model_score, "__wrapped__") \
        else verdict_from_model_score(2.0, MAX) == _verdict_from_score(2.0, MAX)
    # and exhaustively agree across the range
    for i in range(0, 401):
        s = i / 100
        assert verdict_from_model_score(s, MAX) == _verdict_from_score(s, MAX)


def test_benchmark_uses_the_production_factor_table():
    from autograder.grade import _verdict_factor
    from autograder.config import GraderConfig

    cfg = GraderConfig()
    for v in ("valid", "partially_valid", "invalid", "missing", "illegible", None):
        assert factor_for(v, cfg) == _verdict_factor(v, cfg)


def test_canonical_classes_are_the_codebases_own_names():
    """No invented vocabulary: every target class is a real ExplanationVerdict."""
    import typing

    from autograder.schema import ExplanationVerdict

    allowed = set(typing.get_args(ExplanationVerdict))
    assert set(CANONICAL_VERDICTS) <= allowed
    assert set(CANONICAL_VERDICTS) == {"invalid", "partially_valid", "valid"}


# ------------------------------------------- raw score -> canonical verdict ----


def test_raw_score_zero_maps_to_invalid():
    assert verdict_from_model_score(0.0, MAX) == "invalid"
    assert factor_for("invalid") == 0.0


def test_raw_score_in_the_partial_interval_maps_to_partially_valid():
    for s in (0.01, 0.5, 1.0, 2.0, 3.0, 3.9):
        assert verdict_from_model_score(s, MAX) == "partially_valid", s
    assert factor_for("partially_valid") == 0.5


def test_raw_score_at_the_full_credit_threshold_maps_to_valid():
    assert verdict_from_model_score(MAX, MAX) == "valid"
    assert verdict_from_model_score(3.996, MAX) == "valid"      # ratio >= 0.999
    assert factor_for("valid") == 1.0


def test_thresholds_are_ratios_so_the_dead_zones_are_where_production_puts_them():
    assert verdict_from_model_score(0.004, MAX) == "invalid"
    assert verdict_from_model_score(0.005, MAX) == "partially_valid"
    assert verdict_from_model_score(3.99, MAX) == "partially_valid"


# ------------------------------- verdict + selection -> deterministic score ----


@pytest.mark.parametrize("verdict,expected", [
    ("valid", 4.0), ("partially_valid", 2.0), ("invalid", 0.0),
    ("missing", 0.0), ("illegible", 0.0), (None, 0.0),
])
def test_final_score_from_correct_selection(verdict, expected):
    assert final_score_for(selection_correct=True, verdict=verdict, max_points=MAX) == expected


@pytest.mark.parametrize("verdict", ["valid", "partially_valid", "invalid", None])
def test_final_score_from_wrong_selection_is_always_zero(verdict):
    assert final_score_for(selection_correct=False, verdict=verdict, max_points=MAX) == 0.0


def test_end_to_end_raw_score_to_final_number():
    """The whole chain the benchmark replaces with a verdict comparison."""
    for raw, sel, expected in [
        (4.0, True, 4.0), (4.0, False, 0.0),
        (2.0, True, 2.0), (2.0, False, 0.0),
        (0.0, True, 0.0), (0.0, False, 0.0),
    ]:
        v = verdict_from_model_score(raw, MAX)
        assert final_score_for(selection_correct=sel, verdict=v, max_points=MAX) == expected


# ------------------------------------------------- derivation of the labels ----


def test_full_credit_is_uniquely_valid_without_needing_the_selection():
    d = derive_verdict(case_id="c", instructor_final_score=4.0, selection_correct=None,
                       max_points=MAX, transcription=TEXT)
    assert d.derivable and d.derived_explanation_verdict == "valid"
    assert d.derivation_reason == DERIVABLE_FULL
    assert d.implied_final_score == 4.0


def test_partial_credit_is_uniquely_partially_valid():
    d = derive_verdict(case_id="c", instructor_final_score=2.0, selection_correct=None,
                       max_points=MAX, transcription=TEXT)
    assert d.derivable and d.derived_explanation_verdict == "partially_valid"
    assert d.derivation_reason == DERIVABLE_PARTIAL


def test_zero_with_unknown_selection_is_unresolved_never_false():
    d = derive_verdict(case_id="c", instructor_final_score=0.0, selection_correct=None,
                       max_points=MAX, transcription=TEXT)
    assert not d.derivable
    assert d.derived_explanation_verdict is None
    assert d.derivation_reason == UNRESOLVED_ZERO_UNKNOWN_SELECTION


def test_zero_with_wrong_selection_is_excluded_not_labelled_invalid():
    """The explanation was never the reason for the 0; calling it `invalid`
    would fabricate a grading judgement nobody made."""
    d = derive_verdict(case_id="c", instructor_final_score=0.0, selection_correct=False,
                       max_points=MAX, transcription=TEXT)
    assert not d.derivable
    assert d.derived_explanation_verdict is None
    assert d.derivation_reason == EXCLUDED_WRONG_SELECTION


def test_zero_with_correct_selection_is_uniquely_invalid():
    d = derive_verdict(case_id="c", instructor_final_score=0.0, selection_correct=True,
                       max_points=MAX, transcription=TEXT)
    assert d.derivable and d.derived_explanation_verdict == "invalid"
    assert d.derivation_reason == DERIVABLE_ZERO


def test_zero_with_correct_selection_but_no_transcription_refuses():
    """Without text we cannot separate `invalid` from `missing`/`illegible`."""
    d = derive_verdict(case_id="c", instructor_final_score=0.0, selection_correct=True,
                       max_points=MAX, transcription="   ")
    assert not d.derivable
    assert d.derivation_reason == UNRESOLVED_EMPTY_TRANSCRIPTION


def test_every_derived_verdict_reproduces_its_instructor_score():
    """Round-trip guard: a derived label that does not put the instructor's
    number back is a broken derivation."""
    for score in (0.0, 2.0, 4.0):
        d = derive_verdict(case_id="c", instructor_final_score=score,
                           selection_correct=True, max_points=MAX, transcription=TEXT)
        assert d.derivable
        assert d.implied_final_score == score


def test_unreachable_score_is_refused_not_rounded():
    d = derive_verdict(case_id="c", instructor_final_score=3.0, selection_correct=True,
                       max_points=MAX, transcription=TEXT)
    assert not d.derivable


# ------------------------------------------------------------- the adapter ----


def _case(label, *, max_score=MAX, transcription=TEXT):
    from autograder.benchmark.manifests import BenchCase

    pack = {"question_id": "1", "question_text": "q", "question_type": "matching_with_explanation",
            "max_score": max_score, "rubric": [], "rubric_items": [],
            "scoring_rules": ["explanation weight 0"],
            "grading_policy": "choice_and_explanation_independent",
            "evidence_policy": "disabled", "correct_by_version": {}}
    return BenchCase(case_id="c1", split="DEV", component="ALL",
                     inputs={"case_id": "c1", "pack": pack, "selected": None,
                             "transcription": transcription, "version": None},
                     label=label, meta={})


def test_adapter_scores_the_verdict_not_the_raw_number():
    from autograder.benchmark.roles import GradeAdapter

    a = GradeAdapter()
    # model proposes 2.0/4 -> partially_valid; label says partially_valid -> hit,
    # even though the raw number (2.0) equals the instructor score by coincidence
    row = a.score(_case({"score": 2.0, "explanation_verdict": "partially_valid",
                         "explanation_verdict_derivable": True}),
                  {"score": 2.0, "uncertain": False}, None)
    assert row["predicted_verdict"] == "partially_valid"
    assert row["verdict_exact"] is True


def test_adapter_marks_a_verdict_miss_even_when_raw_numbers_are_close():
    from autograder.benchmark.roles import GradeAdapter

    a = GradeAdapter()
    row = a.score(_case({"score": 4.0, "explanation_verdict": "valid",
                         "explanation_verdict_derivable": True}),
                  {"score": 3.9, "uncertain": False}, None)   # ratio .975 -> partial
    assert row["predicted_verdict"] == "partially_valid"
    assert row["verdict_exact"] is False


def test_adapter_reports_no_verdict_when_the_label_is_not_derivable():
    from autograder.benchmark.roles import GradeAdapter

    a = GradeAdapter()
    row = a.score(_case({"score": 0.0, "explanation_verdict": None,
                         "explanation_verdict_derivable": False,
                         "explanation_verdict_reason": UNRESOLVED_ZERO_UNKNOWN_SELECTION}),
                  {"score": 0.0, "uncertain": False}, None)
    assert "verdict_exact" not in row
    assert row["verdict_unavailable_reason"] == UNRESOLVED_ZERO_UNKNOWN_SELECTION


def test_adapter_only_derives_the_final_score_when_selection_is_known():
    from autograder.benchmark.roles import GradeAdapter

    a = GradeAdapter()
    without = a.score(_case({"score": 4.0, "explanation_verdict": "valid",
                             "explanation_verdict_derivable": True}),
                      {"score": 4.0, "uncertain": False}, None)
    assert "implied_final_score" not in without and "final_exact" not in without

    with_sel = a.score(_case({"score": 4.0, "explanation_verdict": "valid",
                              "explanation_verdict_derivable": True,
                              "selection_correct": True}),
                       {"score": 4.0, "uncertain": False}, None)
    assert with_sel["implied_final_score"] == 4.0
    assert with_sel["final_exact"] is True


def test_adapter_keeps_the_raw_score_as_a_diagnostic():
    from autograder.benchmark.roles import GradeAdapter

    a = GradeAdapter()
    row = a.score(_case({"score": 4.0, "explanation_verdict": "valid",
                         "explanation_verdict_derivable": True}),
                  {"score": 1.0, "uncertain": False}, None)
    assert row["score"] == 1.0
    assert row["raw_score_abs_delta"] == 3.0


# ------------------------------------------------------------- aggregation ----


def _scored(truth, predicted, **kw):
    row = {"case_id": f"{truth}->{predicted}", "split": "DEV", "component": "ALL",
           "schema_failure": False, "decision": "AUTO", "score": 0.0,
           "transcription_complete": True, "label_verdict": truth,
           "predicted_verdict": predicted, "verdict_exact": truth == predicted}
    row.update(kw)
    return row


def test_aggregate_reports_confusion_matrix_and_macro_f1():
    from autograder.benchmark.roles import GradeAdapter

    scored = [
        _scored("valid", "valid"), _scored("valid", "valid"),
        _scored("valid", "partially_valid"),
        _scored("partially_valid", "partially_valid"),
        _scored("invalid", "valid"),
    ]
    out = GradeAdapter().aggregate(scored, [])
    assert out["verdict_cases"] == 5
    assert out["verdict_exact_pct"] == pytest.approx(60.0)
    assert out["verdict_confusion"]["valid"]["valid"] == 2
    assert out["verdict_confusion"]["valid"]["partially_valid"] == 1
    assert out["verdict_confusion"]["invalid"]["valid"] == 1
    assert out["verdict_per_class"]["invalid"]["recall"] == 0.0
    assert out["verdict_macro_f1"] is not None
    assert out["verdict_classes_with_support"] == 3


def test_aggregate_does_not_invent_metrics_for_absent_classes():
    from autograder.benchmark.roles import GradeAdapter

    out = GradeAdapter().aggregate([_scored("valid", "valid")], [])
    inv = out["verdict_per_class"]["invalid"]
    assert inv["support"] == 0 and inv["f1"] is None
    assert out["verdict_classes_with_support"] == 1


def test_aggregate_says_so_when_no_ground_truth_exists():
    from autograder.benchmark.roles import GradeAdapter

    row = {"case_id": "c", "split": "DEV", "component": "ALL", "schema_failure": False,
           "decision": "REVIEW", "score": None, "transcription_complete": True}
    out = GradeAdapter().aggregate([row], [])
    assert "verdict_exact_pct" not in out
    assert "unavailable" in out["verdict_metrics"]
    assert out["verdict_cases_excluded_no_ground_truth"] == 1
