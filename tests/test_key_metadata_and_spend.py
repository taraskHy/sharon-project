"""OpenRouter key-metadata support (mocked transport ONLY — never the real
endpoint) and the spend view (local ledger + account-side numbers)."""
from __future__ import annotations

import json

import httpx
import pytest

from autograder.backends.base import BackendConfig
from autograder.backends.openrouter import OpenRouterBackend, fetch_key_metadata, parse_key_metadata
from autograder.spend import budget_status, ledger_summary, spend_view
from autograder.usage import UsageLedger


def test_parse_key_metadata_keeps_numbers_and_drops_anything_key_like():
    d = parse_key_metadata({"data": {"label": "sk-or-v1-abc", "usage": 1.25, "limit": 10, "limit_remaining": 8.75,
                                     "is_free_tier": False, "rate_limit": {"requests": 20, "interval": "10s"},
                                     "key": "sk-or-v1-SECRET"}})
    assert d["ok"] and d["usage"] == 1.25 and d["limit_remaining"] == 8.75
    assert d["label"] == "<redacted>" and "SECRET" not in json.dumps(d)
    assert d["rate_limit"] == {"requests": 20, "interval": "10s"}
    assert parse_key_metadata({}, status_code=401)["detail"].startswith("OpenRouter rejected")
    assert parse_key_metadata({}, status_code=500)["ok"] is False


def test_fetch_key_metadata_with_mock_transport(monkeypatch, no_network):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("Authorization")
        return httpx.Response(200, json={"data": {"usage": 0.5, "limit": 10, "limit_remaining": 9.5,
                                                  "is_free_tier": False, "rate_limit": {"requests": 1, "interval": "1s"}}})

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-TESTKEY")
    be = OpenRouterBackend(BackendConfig(backend="openrouter", model="vendor/model"),
                           transport=httpx.MockTransport(handler))
    meta = be.key_metadata()
    assert seen["path"].endswith("/key") and seen["auth"] == "Bearer sk-or-TESTKEY"
    assert meta == {"ok": True, "status_code": 200, "usage": 0.5, "limit": 10, "limit_remaining": 9.5,
                    "is_free_tier": False, "rate_limit": {"requests": 1, "interval": "1s"},
                    "raw_fields": ["is_free_tier", "limit", "limit_remaining", "rate_limit", "usage"]}
    assert "TESTKEY" not in json.dumps(meta)

    def unreachable(request):
        raise httpx.ConnectError("boom sk-or-TESTKEY")
    client = httpx.Client(base_url="https://openrouter.ai/api/v1", transport=httpx.MockTransport(unreachable))
    err = fetch_key_metadata(client, "sk-or-TESTKEY")
    assert err["ok"] is False and "TESTKEY" not in err["detail"] and "<redacted>" in err["detail"]


def test_spend_view_keeps_ledger_and_account_numbers_separate(tmp_path):
    led = UsageLedger(tmp_path / "usage.jsonl")
    led.record({"task": "ocr_verify", "backend": "openrouter", "model": "a", "cloud": True, "cache_hit": False,
                "input_tokens": 100, "output_tokens": 10, "total_tokens": 110, "reported_cost": 0.01})
    led.record({"task": "grade_primary", "backend": "openrouter", "model": "b", "cloud": True, "cache_hit": False,
                "input_tokens": 200, "output_tokens": 20, "total_tokens": 220, "reported_cost": 0.02})
    led.record({"task": "grade_primary", "backend": "openrouter", "model": "b", "cloud": True, "cache_hit": True})
    s = ledger_summary(led)
    assert s["cloud_calls"] == 2 and s["cache_hits"] == 1 and s["cumulative_cost"] == 0.03
    assert s["by_task"]["ocr_verify"]["calls"] == 1 and s["by_model"]["b"]["reported_cost"] == 0.02
    assert s["input_tokens"] == 300 and s["output_tokens"] == 30
    v = spend_view(led, {"ok": True, "usage": 3.5, "limit": 10, "limit_remaining": 6.5})
    assert v["local_ledger"]["cumulative_cost"] == 0.03
    assert v["openrouter_key"]["usage_usd"] == 3.5 and "OpenRouter-reported" in v["openrouter_key"]["source"]
    assert v["budget"]["state"] == "OK"
    assert spend_view(led)["openrouter_key"] is None


@pytest.mark.parametrize("cost,state", [(0.0, "OK"), (7.99, "OK"), (8.0, "WARNING"), (9.99, "WARNING"),
                                        (10.0, "HARD_STOP"), (12.0, "HARD_STOP")])
def test_budget_status_policy(cost, state):
    b = budget_status(cost)
    assert b["state"] == state and b["warn_usd"] == 8.0 and b["hard_usd"] == 10.0
