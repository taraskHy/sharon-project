"""Failed-but-billed accounting: a provider charge survives every downstream failure.

The 2026-08-24 GRADE_PRIMARY smoke run spent $0.030510 at OpenRouter while the
local ledger recorded $0.019786. The missing $0.010724 was one Sonnet request
that the provider generated (and billed) in full and that we then discarded
because ``finish_reason=length`` made the body unusable. The $10 campaign
ceiling is enforced against that ledger, so an undercount is a SAFETY bug, not
a reporting one.

Rule under test: the provider creates the charge; our ability to parse the
reply is irrelevant to whether we owe money. Every test here is offline
(httpx.MockTransport) — no provider is ever contacted.
"""

from __future__ import annotations

import dataclasses

import httpx
import pytest
from pydantic import BaseModel

from autograder.backends import BackendConfig
from autograder.backends.base import BackendError
from autograder.backends.openrouter import OpenRouterBackend
from autograder.gateway import ModelGateway
from autograder.usage import BudgetExceeded, BudgetLimits, BudgetManager, UsageLedger, reconcile_cost
from autograder.cloudboundary import research_authorization


class Out(BaseModel):
    score: float


PRICING = {"vendor/m": {"input": 2.0, "output": 10.0}}


def _usage(prompt=2400, completion=600, cost=0.010724):
    u = {"prompt_tokens": prompt, "completion_tokens": completion,
         "total_tokens": prompt + completion}
    if cost is not None:
        u["cost"] = cost
    return u


def _body(content, *, finish="stop", usage=None, model="vendor/m"):
    return {"id": "gen-abc", "provider": "TestProv", "model": model,
            "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": usage if usage is not None else _usage()}


def _gw(tmp_path, handler, *, monkeypatch, limits=None, validation_retries=0):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-not-a-real-key")
    ledger = UsageLedger(tmp_path / "usage.jsonl")

    def factory(cfg: BackendConfig):
        # validation_retries lives on the BackendConfig, not on the route
        # (the benchmark runner sets it the same way)
        return OpenRouterBackend(dataclasses.replace(cfg, validation_retries=validation_retries),
                                 transport=httpx.MockTransport(handler))

    # research mode: billed-attempt ACCOUNTING is what is under test, on a
    # cloud-shaped grading route with a mocked transport; the production
    # boundary would refuse the route before transport (test_cloud_boundary).
    gw = ModelGateway.from_dict(
        {"models": {"grade_primary": {"backend": "openrouter", "model": "vendor/m",
                                      "max_tokens": 600, "prompt_version": "grade-v2"}}},
        backend_factory=factory, ledger=ledger, execution_mode="research",
        research_auth=research_authorization("test:billing", tasks=["grade_primary"],
                                             models=["vendor/m"]))
    gw.pricing_config = PRICING
    if limits is not None:
        gw.budget = BudgetManager(limits, ledger=ledger)
    return gw, ledger


def _call(gw, output_model=Out):
    return gw.call(task="grade_primary", system="s",
                   content_blocks=[{"type": "text", "text": "q"}], output_model=output_model,
                   meta={"job_id": "j", "exam_id": "e1"})


def _spent(ledger):
    return round(sum(float(r.get("reported_cost") or 0) for r in ledger.entries()), 8)


# ---------------------------------------------------------------- scenarios ----


def test_1_billable_usage_with_malformed_output_still_charges(tmp_path, monkeypatch):
    """Provider returns valid billable usage + a body that fails validation."""
    def handler(request):
        return httpx.Response(200, json=_body("{not valid json at all", usage=_usage(cost=0.004)))

    gw, ledger = _gw(tmp_path, handler, monkeypatch=monkeypatch)
    with pytest.raises(BackendError):
        _call(gw)
    rows = ledger.entries()
    assert rows, "a billed provider response must never leave the ledger empty"
    assert _spent(ledger) == pytest.approx(0.004 * len(rows))
    assert all(r["billable"] for r in rows)
    assert all(r["inference_reached"] for r in rows)
    assert all(r["parse_ok"] is False for r in rows)


def test_2_finish_reason_length_still_charges(tmp_path, monkeypatch):
    """The exact Sonnet e003 failure: generated, billed, truncated, discarded."""
    def handler(request):
        return httpx.Response(200, json=_body('{"score": 2.0', finish="length",
                                              usage=_usage(cost=0.010724)))

    gw, ledger = _gw(tmp_path, handler, monkeypatch=monkeypatch)
    with pytest.raises(BackendError, match="truncated"):
        _call(gw)
    rows = ledger.entries()
    assert len(rows) == 1, "truncation is terminal: no repair retry, exactly one charge"
    r = rows[0]
    assert r["billable"] is True
    assert r["parse_ok"] is False
    assert r["finish_reason"] == "length"
    assert r["output_tokens"] == 600
    assert r["cost_source"] == "provider"
    assert _spent(ledger) == pytest.approx(0.010724)


def test_3_parser_throws_after_response_still_charges(tmp_path, monkeypatch):
    """An exception raised after the provider replied must not erase the charge."""
    def handler(request):
        return httpx.Response(200, json=_body('{"score": 1.0}', usage=_usage(cost=0.002)))

    gw, ledger = _gw(tmp_path, handler, monkeypatch=monkeypatch)

    class Exploding(BaseModel):
        score: float

        @classmethod
        def model_validate_json(cls, *a, **k):
            raise RuntimeError("parser blew up")

    with pytest.raises(RuntimeError, match="parser blew up"):
        _call(gw, output_model=Exploding)
    assert _spent(ledger) == pytest.approx(0.002)
    assert ledger.entries()[0]["inference_reached"] is True


def test_4_normal_success_records_exactly_one_entry(tmp_path, monkeypatch):
    def handler(request):
        return httpx.Response(200, json=_body('{"score": 4.0}', usage=_usage(cost=0.003)))

    gw, ledger = _gw(tmp_path, handler, monkeypatch=monkeypatch)
    res = _call(gw)
    assert res.value.score == 4.0
    rows = ledger.entries()
    assert len(rows) == 1, "success must not double-count"
    assert rows[0]["outcome"] == "ok"
    assert rows[0]["parse_ok"] is True
    assert _spent(ledger) == pytest.approx(0.003)


def test_5_pre_inference_http_400_costs_nothing(tmp_path, monkeypatch):
    """The Luna failure: rejected on schema validation before any inference."""
    def handler(request):
        return httpx.Response(400, json={"error": {
            "message": "Invalid schema for response_format: additionalProperties",
            "code": 400}})

    gw, ledger = _gw(tmp_path, handler, monkeypatch=monkeypatch)
    with pytest.raises(BackendError, match="400"):
        _call(gw)
    rows = ledger.entries()
    assert len(rows) == 1, "a refusal is still recorded — 'called and refused' != 'never called'"
    r = rows[0]
    assert r["billable"] is False
    assert r["call_attempted"] is True
    assert r["inference_reached"] is False
    assert r["usage_returned"] is False
    assert _spent(ledger) == 0.0


def test_6_budget_ceiling_counts_failed_but_billed_calls(tmp_path, monkeypatch):
    """A run of truncation failures must still trip the hard ceiling."""
    def handler(request):
        return httpx.Response(200, json=_body('{"score": 2.0', finish="length",
                                              usage=_usage(cost=0.60)))

    limits = BudgetLimits(max_cost_total=1.0, soft_fraction=0.8)
    gw, ledger = _gw(tmp_path, handler, monkeypatch=monkeypatch, limits=limits)

    with pytest.raises(BackendError):
        _call(gw)                      # burns $0.60 on a failure
    assert _spent(ledger) == pytest.approx(0.60)

    with pytest.raises(BackendError):
        _call(gw)                      # burns another $0.60 -> $1.20 total
    assert _spent(ledger) == pytest.approx(1.20)

    # the ceiling now sees $1.20 of failed-but-billed spend and refuses
    with pytest.raises(BudgetExceeded):
        _call(gw)


# ------------------------------------------------------- repair round-trips ----


def test_every_repair_attempt_is_billed_separately(tmp_path, monkeypatch):
    """A validation-repair round-trip bills once PER ATTEMPT. Recording only
    the last one (the old ``last_usage`` behaviour) silently lost the rest."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json=_body('{"wrong": 1}', usage=_usage(cost=0.005)))
        return httpx.Response(200, json=_body('{"score": 4.0}', usage=_usage(cost=0.007)))

    gw, ledger = _gw(tmp_path, handler, monkeypatch=monkeypatch, validation_retries=1)
    res = _call(gw)
    assert res.value.score == 4.0
    rows = ledger.entries()
    assert len(rows) == 2, "both the rejected attempt and the accepted one were billed"
    assert [r["parse_ok"] for r in rows] == [False, True]
    assert [r["outcome"] for r in rows] == ["superseded_attempt", "ok"]
    assert _spent(ledger) == pytest.approx(0.012)


# ------------------------------------------------------------ reconciliation ----


def test_provider_cost_is_authoritative_over_local_pricing():
    """OpenRouter's usage.cost already reflects the provider actually routed
    to; a local price table must never overwrite it."""
    usage = {"reported_cost": 0.001, "input_tokens": 1_000_000, "output_tokens": 1_000_000,
             "model": "vendor/m"}
    assert reconcile_cost(usage, None, PRICING) == (0.001, "provider")


def test_local_pricing_fills_in_when_provider_reports_no_cost():
    usage = {"input_tokens": 1_000_000, "output_tokens": 1_000_000, "model": "vendor/m"}
    cost, src = reconcile_cost(usage, None, PRICING)
    assert src == "local_pricing"
    assert cost == pytest.approx(12.0)


def test_nothing_billable_reported_is_zero_not_a_guess():
    assert reconcile_cost({}, None, PRICING) == (0.0, "none")
