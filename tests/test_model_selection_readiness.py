"""Model-selection readiness: UNSELECTED role sentinel, run-level refusal,
and the cross-run experiment budget ($10 hard / $8 warn) — offline tests."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from autograder.backends import BackendConfig
from autograder.backends.mock import MockBackend
from autograder.gateway import UNSELECTED, GatewayConfigError, ModelGateway
from autograder.reliability import GradingModeError, ReliabilityConfig, run_reliability_judging
from autograder.schema import AnswerKey
from autograder.usage import (BudgetExceeded, BudgetLimits, BudgetManager, UsageLedger,
                              aggregate_by, predicted_call_cost, run_cost_report)


class Out(BaseModel):
    text: str


# ------------------------------------------------------------ UNSELECTED ----


def _gw_dict(model: str) -> dict:
    return {"models": {"grade_primary": {"backend": "openrouter", "model": model}}}


def test_unselected_config_loads_but_route_refuses_clearly():
    gw = ModelGateway.from_dict(_gw_dict(UNSELECTED),
                                backend_factory=lambda c: MockBackend(config=c))
    assert gw.describe()["grade_primary"]["model"] == UNSELECTED
    with pytest.raises(GatewayConfigError) as err:
        gw.route("grade_primary")
    message = str(err.value)
    assert "UNSELECTED" in message
    assert "candidates.toml" in message


def test_empty_model_still_fails_at_construction():
    with pytest.raises(GatewayConfigError):
        ModelGateway.from_dict(_gw_dict(""),
                               backend_factory=lambda c: MockBackend(config=c))


def test_reliability_mode_refuses_at_run_level_not_per_item():
    """A missing/UNSELECTED grade_primary must refuse the RUN, never degrade
    into one silent REVIEW per item."""
    gw = ModelGateway.from_dict(_gw_dict(UNSELECTED),
                                backend_factory=lambda c: MockBackend(config=c))
    key = AnswerKey(exam_title="t", versions=["default"], questions=[], total_points=0.0)
    with pytest.raises(GradingModeError) as err:
        run_reliability_judging(
            key=key, extraction=None, version="default",
            config=ReliabilityConfig(mode="reliability"), gateway=gw)
    assert "grade_primary" in str(err.value)
    assert "cannot start" in str(err.value)


# ------------------------------------------------- cross-run cost ceiling ----


def _seed_ledger(tmp_path, costs: list[float]) -> UsageLedger:
    ledger = UsageLedger(tmp_path / "ledger" / "usage.jsonl")
    for cost in costs:
        ledger.record({"cloud": True, "cache_hit": False, "backend": "openrouter",
                       "model": "vendor/m", "task": "grade_primary",
                       "reported_cost": cost, "input_tokens": 100,
                       "output_tokens": 50, "total_tokens": 150})
    return ledger


class _Route:
    backend = "openrouter"
    base_url = None
    model = "vendor/m"
    max_tokens = 300


def test_cost_total_enforced_across_manager_instances(tmp_path):
    ledger = _seed_ledger(tmp_path, [6.0, 4.5])  # 10.50 already spent
    # A brand-new manager (fresh process) must still see the persisted spend.
    manager = BudgetManager(BudgetLimits(max_cost_total=10.0), ledger=ledger)
    with pytest.raises(BudgetExceeded) as err:
        manager.check(task="grade_primary", route=_Route(), meta={})
    assert "cost_total" in str(err.value)
    assert manager.paused is True


def test_cost_total_under_ceiling_passes_and_warns_at_soft_fraction(tmp_path):
    ledger = _seed_ledger(tmp_path, [8.5])  # 8.50 of 10.00 -> above 0.8 soft line
    warnings: list[str] = []
    manager = BudgetManager(BudgetLimits(max_cost_total=10.0), ledger=ledger,
                            warn=warnings.append)
    manager.check(task="grade_primary", route=_Route(), meta={})  # no raise
    assert any("cost_total" in w for w in warnings)  # the $8 warning fired


def test_predicted_cost_refuses_the_crossing_call(tmp_path):
    ledger = _seed_ledger(tmp_path, [9.5])
    manager = BudgetManager(BudgetLimits(max_cost_total=10.0), ledger=ledger)
    # 9.50 known + 0.80 predicted would cross 10.00 -> refuse BEFORE the call
    with pytest.raises(BudgetExceeded):
        manager.check(task="grade_primary", route=_Route(), meta={},
                      predicted_cost=0.8)
    # a cheaper predicted call still fits
    manager2 = BudgetManager(BudgetLimits(max_cost_total=10.0), ledger=ledger)
    manager2.check(task="grade_primary", route=_Route(), meta={}, predicted_cost=0.3)


def test_predicted_call_cost_pricing_math():
    route = _Route()
    pricing = {"vendor/m": {"input": 1.0, "output": 2.0}}  # USD per 1M tokens
    system = "x" * 4000  # -> 1000 tokens
    blocks = [{"type": "text", "text": "y" * 400},  # -> 100 tokens
              {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                           "data": "AAAA"}}]  # -> flat 1100 tokens
    cost = predicted_call_cost(route, system, blocks, pricing)
    expected = ((1000 + 100 + 1100) * 1.0 + 300 * 2.0) / 1e6
    assert cost == pytest.approx(expected)
    # unknown model / no pricing table -> 0.0 (known-spend-only enforcement)
    assert predicted_call_cost(route, system, blocks, {"other/m": {"input": 1}}) == 0.0
    assert predicted_call_cost(route, system, blocks, None) == 0.0


def test_gateway_refuses_before_provider_call_when_ceiling_would_cross(tmp_path):
    ledger = _seed_ledger(tmp_path, [9.99])
    counter = {"calls": 0}

    def factory(cfg: BackendConfig):
        def responder(model, system, blocks):
            counter["calls"] += 1
            return Out(text="paid")
        return MockBackend(config=cfg, responder=responder)

    manager = BudgetManager(BudgetLimits(max_cost_total=10.0), ledger=ledger)
    gw = ModelGateway.from_dict(
        {"models": {"grade_primary": {"backend": "openrouter", "model": "vendor/m",
                                      "max_tokens": 300}},
         "pricing": {"vendor/m": {"input": 1000.0, "output": 1000.0}}},
        backend_factory=factory, ledger=ledger, budget=manager)
    with pytest.raises(BudgetExceeded):
        gw.call(task="grade_primary", system="s" * 4000,
                content_blocks=[{"type": "text", "text": "q"}], output_model=Out)
    assert counter["calls"] == 0  # refused BEFORE any provider call


# ------------------------------------------------------- per-run reporting ----


def test_run_cost_report_before_run_after_and_breakdowns(tmp_path):
    ledger = _seed_ledger(tmp_path, [1.0, 0.5])  # the "before" spend
    baseline = len(ledger.entries())
    # the run: two paid calls on different models/tasks, one cache hit, one local
    ledger.record({"cloud": True, "cache_hit": False, "model": "vendor/a",
                   "task": "ocr_primary", "reported_cost": 0.25,
                   "input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
    ledger.record({"cloud": True, "cache_hit": False, "model": "vendor/b",
                   "task": "grade_primary", "reported_cost": 0.75,
                   "input_tokens": 20, "output_tokens": 10, "total_tokens": 30})
    ledger.record({"cloud": True, "cache_hit": True, "model": "vendor/a",
                   "task": "ocr_primary", "reported_cost": 0})
    ledger.record({"cloud": False, "cache_hit": False, "model": "local/q",
                   "task": "mc_resolve", "reported_cost": 0})

    report = run_cost_report(ledger, baseline)
    assert report["cost_before"] == pytest.approx(1.5)
    assert report["run_cost"] == pytest.approx(1.0)
    assert report["cost_after"] == pytest.approx(2.5)
    assert report["run_calls"] == 2
    assert report["run_cache_hits"] == 1
    assert report["by_model"]["vendor/a"]["calls"] == 1
    assert report["by_model"]["vendor/b"]["reported_cost"] == pytest.approx(0.75)
    assert report["by_task"]["grade_primary"]["total_tokens"] == 30
    assert "local/q" not in report["by_model"]  # local rows never count as spend


def test_aggregate_by_groups_cloud_paid_rows_only():
    rows = [
        {"cloud": True, "cache_hit": False, "model": "m1", "task": "t1",
         "reported_cost": 0.1, "input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        {"cloud": True, "cache_hit": False, "model": "m1", "task": "t2",
         "reported_cost": 0.2, "input_tokens": 2, "output_tokens": 2, "total_tokens": 4},
        {"cloud": True, "cache_hit": True, "model": "m1", "task": "t1", "reported_cost": 0},
        {"cloud": False, "model": "m2", "task": "t1", "reported_cost": 0},
    ]
    by_model = aggregate_by(rows, "model")
    assert set(by_model) == {"m1"}
    assert by_model["m1"]["calls"] == 2
    assert by_model["m1"]["reported_cost"] == pytest.approx(0.3)
    by_task = aggregate_by(rows, "task")
    assert set(by_task) == {"t1", "t2"}
