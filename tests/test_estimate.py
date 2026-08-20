"""Pre-run cost/query estimator (§7). No provider calls, no network pricing."""

from __future__ import annotations

import pytest

from autograder.backends.mock import MockBackend
from autograder.estimate import (DISCLAIMER, EscalationAssumptions, TokenAssumptions,
                                 estimate_job, load_pricing)
from autograder.gateway import ModelGateway
from tests.test_grade import make_key

CLOUD_CFG = {
    "models": {
        "ocr_primary": {"backend": "openrouter", "model": "vendor/ocr-1"},
        "ocr_verify": {"backend": "openrouter", "model": "vendor/ocr-1"},
        "grade_primary": {"backend": "openrouter", "model": "vendor/grade-small"},
        "grade_escalate": {"backend": "openrouter", "model": "vendor/grade-big"},
        "mc_resolve_cloud": {"backend": "openrouter", "model": "vendor/mc-1"},
        "mc_resolve": {"backend": "openai", "model": "local-vlm",
                       "base_url": "http://localhost:11434/v1"},
    },
    "pricing": {
        "vendor/ocr-1": {"input": 0.15, "output": 0.60},
        "vendor/grade-small": {"input": 0.10, "output": 0.40},
        "vendor/grade-big": {"input": 1.00, "output": 3.00},
        "vendor/mc-1": {"input": 0.05, "output": 0.20},
    },
}


def gw(cfg=None):
    return ModelGateway.from_dict(cfg or CLOUD_CFG, backend_factory=lambda c: MockBackend(config=c))


def mc_only_policies(key):
    return {q.id: "choice_only" for q in key.questions}


def explanation_policies(key):
    return {q.id: "choice_and_explanation_independent" for q in key.questions}


def test_everything_is_labelled_as_an_estimate():
    e = estimate_job(key=make_key(), exams=10, policies=explanation_policies(make_key()))
    d = e.as_dict()
    assert d["kind"] == "ESTIMATE" and d["disclaimer"] == DISCLAIMER
    assert all(k.startswith("estimated_") for k in d
               if k.endswith(("_calls", "_tokens", "_cost")) and k != "cost_unavailable_reason")


def test_deterministic_heavy_exam_needs_almost_no_cloud_work():
    key = make_key()
    e = estimate_job(key=key, exams=25, policies=mc_only_policies(key), gateway=gw(),
                     pricing=load_pricing(CLOUD_CFG))
    assert e.estimated_calls["ocr"] == 0 and e.estimated_calls["grade_primary"] == 0
    assert e.estimated_calls["ocr_verify"] == 0 and e.estimated_calls["grade_escalate"] == 0
    assert e.estimated_calls["mc_resolve_cloud"] > 0          # only ambiguous rows reach a model
    assert e.sub_items_deterministic == e.sub_items_total
    assert e.explanations_skipped > 0
    assert 0 < e.estimated_cost < 1.0


def test_explanation_heavy_exam_costs_far_more_than_the_same_exam_mc_only():
    key = make_key()
    mc = estimate_job(key=key, exams=25, policies=mc_only_policies(key), gateway=gw(),
                      pricing=load_pricing(CLOUD_CFG))
    expl = estimate_job(key=key, exams=25, policies=explanation_policies(key), gateway=gw(),
                        pricing=load_pricing(CLOUD_CFG))
    assert expl.estimated_cloud_calls > 10 * mc.estimated_cloud_calls
    assert expl.estimated_cost > mc.estimated_cost
    assert expl.estimated_calls["ocr"] > 0 and expl.estimated_calls["grade_primary"] > 0


def test_wrong_choice_zero_halves_the_explanation_work():
    key = make_key()
    a = EscalationAssumptions(mc_correct_rate=0.5, blank_answer_rate=0.0)
    full = estimate_job(key=key, exams=10, policies=explanation_policies(key), assumptions=a)
    gated = estimate_job(key=key, exams=10,
                         policies={q.id: "wrong_choice_zero" for q in key.questions}, assumptions=a)
    assert gated.estimated_calls["ocr"] == pytest.approx(full.estimated_calls["ocr"] / 2, rel=0.02)
    assert gated.explanations_skipped > 0


def test_unavailable_pricing_still_reports_calls_and_tokens():
    key = make_key()
    e = estimate_job(key=key, exams=10, policies=explanation_policies(key), gateway=gw(),
                     pricing=None)
    assert e.estimated_cost is None and "no pricing configured" in e.cost_unavailable_reason
    assert e.estimated_cloud_calls > 0 and e.estimated_input_tokens > 0


def test_partial_pricing_refuses_to_guess_a_total():
    key = make_key()
    e = estimate_job(key=key, exams=10, policies=explanation_policies(key), gateway=gw(),
                     pricing={"vendor/ocr-1": {"input": 0.15, "output": 0.6}})
    assert e.estimated_cost is None
    assert "vendor/grade-small" in e.cost_unavailable_reason


def test_local_only_configuration_costs_nothing():
    key = make_key()
    local = {"models": {"mc_resolve": {"backend": "openai", "model": "local-vlm",
                                       "base_url": "http://localhost:11434/v1"}}}
    e = estimate_job(key=key, exams=10, policies=explanation_policies(key), gateway=gw(local),
                     pricing=load_pricing(CLOUD_CFG))
    assert e.estimated_cost == 0.0 and e.estimated_cloud_calls > 0   # calls, but none routed to cloud


def test_estimates_scale_linearly_with_the_number_of_exams():
    key = make_key()
    p = explanation_policies(key)
    one = estimate_job(key=key, exams=1, policies=p)
    fifty = estimate_job(key=key, exams=50, policies=p)
    assert fifty.estimated_input_tokens == pytest.approx(50 * one.estimated_input_tokens, rel=0.01)
    assert fifty.per_exam["cloud_calls"] == pytest.approx(one.per_exam["cloud_calls"], rel=0.01)


def test_cache_hits_reduce_the_estimate():
    key = make_key()
    p = explanation_policies(key)
    cold = estimate_job(key=key, exams=10, policies=p)
    warm = estimate_job(key=key, exams=10, policies=p,
                        assumptions=EscalationAssumptions(cache_hit_rate=0.5))
    assert warm.estimated_cloud_calls == pytest.approx(cold.estimated_cloud_calls / 2, rel=0.02)


def test_assumptions_record_their_source_and_can_come_from_measured_metrics():
    default = EscalationAssumptions()
    assert default.source == "default"
    hist = EscalationAssumptions.from_metrics({"grade_escalation_rate": 0.42,
                                               "local_resolution_success": 80.0})
    assert hist.source == "historical" and hist.grade_escalation_rate == 0.42
    assert hist.mc_local_resolution_rate == 0.8          # percentages normalised
    e = estimate_job(key=make_key(), exams=5, assumptions=hist)
    assert e.assumptions["source"] == "historical"


def test_pricing_is_loaded_from_local_config_only():
    assert load_pricing(CLOUD_CFG)["vendor/ocr-1"] == {"input": 0.15, "output": 0.60}
    assert load_pricing({}) == {}
    assert load_pricing({"pricing": {"bad": "not-a-table"}}) == {}


def test_gateway_exposes_the_local_pricing_table():
    assert gw().pricing_config["vendor/mc-1"]["input"] == 0.05
    assert gw({"models": {"m": {"backend": "mock", "model": "x"}}}).pricing_config is None


def test_token_assumptions_are_explicit_and_overridable():
    key = make_key()
    p = explanation_policies(key)
    base = estimate_job(key=key, exams=5, policies=p)
    big = estimate_job(key=key, exams=5, policies=p, tokens=TokenAssumptions(ocr_input=2400))
    assert big.estimated_input_tokens > base.estimated_input_tokens
    assert base.assumptions["tokens"]["ocr_input"] == 1200
