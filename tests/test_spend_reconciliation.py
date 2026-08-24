"""Historical spend reconciliation — offline.

Fixing the accounting code stops FUTURE billed failures from vanishing; it
does not un-spend money already gone. The 2026-08-24 smoke run left
$0.010724375 of provider-billed spend outside the ledger. Until that is
booked, the $8 warning, the $10 ceiling and "how much is left" are all
computed against a number known to be too small — so the reconciliation is a
budget-safety mechanism, not a reporting nicety.

No provider is contacted anywhere in this file.
"""

from __future__ import annotations

import pytest

from autograder.usage import (
    BudgetExceeded,
    BudgetLimits,
    BudgetManager,
    UsageLedger,
    is_reconciliation,
    reconciled_total,
    record_reconciliation,
)


class _Route:
    backend = "openrouter"
    base_url = None
    model = "vendor/m"


def _ledger(tmp_path):
    return UsageLedger(tmp_path / "usage.jsonl")


def _call_row(cost):
    return {"cloud": True, "backend": "openrouter", "cache_hit": False,
            "reported_cost": cost, "exam_id": "e1", "total_tokens": 100,
            "input_tokens": 60, "output_tokens": 40}


def _recon(ledger, amount, **kw):
    return record_reconciliation(ledger, amount_usd=amount,
                                 reason=kw.pop("reason", "pre-fix accounting defect"),
                                 provenance=kw.pop("provenance", "account delta"), **kw)


# ------------------------------------------------------------- the entry ----


def test_reconciliation_is_not_a_provider_call(tmp_path):
    led = _ledger(tmp_path)
    entry = _recon(led, 0.010724375)
    assert entry["entry_kind"] == "reconciliation"
    assert entry["provider_call"] is False
    assert is_reconciliation(entry)


def test_reconciliation_never_invents_token_counts(tmp_path):
    """The truncated 2026-08-24 request recorded usage={}. A reconciliation
    that filled in plausible tokens would be fabricating measurements."""
    led = _ledger(tmp_path)
    entry = _recon(led, 0.010724375)
    for field in ("input_tokens", "output_tokens", "total_tokens", "reasoning_tokens"):
        assert field not in entry
    assert entry["tokens_available"] is False


def test_reconciliation_records_why_and_where_it_came_from(tmp_path):
    led = _ledger(tmp_path)
    entry = _recon(led, 0.5, reason="billed then discarded on finish_reason=length",
                   provenance="GET /api/v1/key before/after delta",
                   occurred_at="2026-08-24 20:56:07",
                   related_case_ids=["e003_q2_r6"], model="anthropic/claude-sonnet-5")
    assert entry["reason"] and entry["provenance"]
    assert entry["occurred_at"] == "2026-08-24 20:56:07"
    assert entry["related_case_ids"] == ["e003_q2_r6"]
    assert entry["cost_source"] == "provider_account_delta"


def test_a_reconciliation_cannot_reduce_spend(tmp_path):
    """This books a known past charge; it is not a spend-adjustment lever."""
    led = _ledger(tmp_path)
    with pytest.raises(ValueError):
        _recon(led, -0.01)
    with pytest.raises(ValueError):
        _recon(led, 0.0)


# ------------------------------------------------------------ aggregates ----


def test_reconciliation_counts_as_money_not_as_a_call(tmp_path):
    led = _ledger(tmp_path)
    led.record(_call_row(0.01))
    _recon(led, 0.5)
    agg = led.aggregate()
    assert agg["cloud_requests"] == 1, "a reconciliation is not a request"
    assert agg["total_tokens"] == 100, "a reconciliation must not move token averages"
    assert agg["reported_cost"] == pytest.approx(0.01), "call-only view stays call-only"
    tot = reconciled_total(led)
    assert tot["total_usd"] == pytest.approx(0.51)
    assert tot["reconciliation_usd"] == pytest.approx(0.5)
    assert tot["provider_call_rows"] == 1 and tot["reconciliation_rows"] == 1


def test_cache_hit_rate_is_not_polluted(tmp_path):
    led = _ledger(tmp_path)
    led.record(dict(_call_row(0.01), cache_hit=True))
    led.record(_call_row(0.02))
    before = led.aggregate()["cache_hit_rate"]
    _recon(led, 5.0)
    assert led.aggregate()["cache_hit_rate"] == before


# --------------------------------------------------------------- budget -----


def test_reconciliation_counts_toward_the_hard_ceiling(tmp_path):
    led = _ledger(tmp_path)
    _recon(led, 0.95)
    budget = BudgetManager(BudgetLimits(max_cost_total=1.0), ledger=led)
    # a call predicted at $0.10 would cross 1.00 once the booked $0.95 counts
    with pytest.raises(BudgetExceeded):
        budget.check(task="grade_primary", route=_Route(), meta={}, predicted_cost=0.10)


def test_without_reconciliation_the_same_call_would_be_allowed(tmp_path):
    """Directly demonstrates the safety gap the booking closes."""
    led = _ledger(tmp_path)
    budget = BudgetManager(BudgetLimits(max_cost_total=1.0), ledger=led)
    budget.check(task="grade_primary", route=_Route(), meta={}, predicted_cost=0.10)  # fine
    _recon(led, 0.95)
    with pytest.raises(BudgetExceeded):
        budget.check(task="grade_primary", route=_Route(), meta={}, predicted_cost=0.10)


def test_reconciliation_triggers_the_soft_warning(tmp_path):
    led = _ledger(tmp_path)
    _recon(led, 8.5)
    warnings = []
    budget = BudgetManager(BudgetLimits(max_cost_total=10.0, soft_fraction=0.8),
                           ledger=led, warn=warnings.append)
    budget.check(task="grade_primary", route=_Route(), meta={}, predicted_cost=0.01)
    assert any("cost_total" in w for w in warnings), warnings


def test_remaining_budget_reflects_reconciled_spend(tmp_path):
    led = _ledger(tmp_path)
    led.record(_call_row(0.019785625))
    _recon(led, 0.010724375)
    budget = BudgetManager(BudgetLimits(max_cost_total=10.0), ledger=led)
    spent = budget._ledger_cost()
    assert spent == pytest.approx(0.030510)
    assert 10.0 - spent == pytest.approx(9.969490)


def test_the_project_ledger_reconciles_to_the_measured_account_delta():
    """The real campaign ledger, as booked on 2026-08-24."""
    from pathlib import Path

    p = Path("evaluation/model_selection/state/gateway_ledger/usage.jsonl")
    if not p.exists():
        pytest.skip("campaign ledger not present in this checkout")
    tot = reconciled_total(UsageLedger(p))
    assert tot["provider_calls_usd"] == pytest.approx(0.019785625, abs=1e-8)
    assert tot["reconciliation_usd"] == pytest.approx(0.010724375, abs=1e-8)
    # measured GET /api/v1/key delta across the smoke run: $0 -> $0.030510
    assert tot["total_usd"] == pytest.approx(0.030510, abs=1e-8)
