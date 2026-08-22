"""Spend truth: the persistent local usage ledger + (later) OpenRouter's own
key usage, side by side.

Two independent numbers, deliberately never merged into one:

* **local experiment ledger** — every gateway call is appended to a
  UsageLedger JSONL (task, model, tokens, reported cost). This is what the
  budget manager enforces against ($8 warning / $10 hard stop for the
  model-selection campaign). It is authoritative for *our* policy and needs
  no account access.
* **OpenRouter-reported key usage** — ``GET /api/v1/key`` (backends.openrouter
  .fetch_key_metadata), fetched only on explicit demand once a credential
  exists. It is the account-side truth and catches spend made outside this
  product. The product must keep working when it is unavailable.

No function here performs a network call; ``key_metadata`` is accepted as an
argument (already fetched, or None).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .usage import UsageLedger, aggregate_by

#: The model-selection campaign policy (docs/model-selection.md §budget,
#: evaluation/model_selection/candidates.toml [budget]).
EXPERIMENT_HARD_STOP_USD = 10.00
EXPERIMENT_WARN_USD = 8.00


def ledger_summary(ledger: UsageLedger | str | Path) -> dict[str, Any]:
    """Cumulative, per-task and per-model spend over ONE persisted ledger
    (cloud, non-cache rows). Pure file read."""
    led = ledger if isinstance(ledger, UsageLedger) else UsageLedger(ledger)
    rows = led.entries()
    by_task = aggregate_by(rows, "task")
    by_model = aggregate_by(rows, "model")
    cloud_calls = sum(g["calls"] for g in by_task.values())
    return {
        "path": str(led.path),
        "exists": led.path.exists(),
        "rows": len(rows),
        "cloud_calls": cloud_calls,
        "cache_hits": sum(1 for e in rows if e.get("cache_hit") and e.get("cloud", e.get("backend") == "openrouter")),
        "input_tokens": sum(g["input_tokens"] for g in by_task.values()),
        "output_tokens": sum(g["output_tokens"] for g in by_task.values()),
        "total_tokens": sum(g["total_tokens"] for g in by_task.values()),
        "cumulative_cost": round(sum(g["reported_cost"] for g in by_task.values()), 6),
        "by_task": by_task,
        "by_model": by_model,
    }


def budget_status(cumulative_cost: float, *, warn_usd: float = EXPERIMENT_WARN_USD,
                  hard_usd: float = EXPERIMENT_HARD_STOP_USD) -> dict[str, Any]:
    """Classify cumulative spend against the campaign policy.
    state: OK | WARNING (>= warn) | HARD_STOP (>= hard)."""
    c = float(cumulative_cost or 0.0)
    state = "HARD_STOP" if c >= hard_usd else ("WARNING" if c >= warn_usd else "OK")
    return {"cumulative_cost": round(c, 6), "warn_usd": warn_usd, "hard_usd": hard_usd,
            "remaining_to_hard_stop": round(max(hard_usd - c, 0.0), 6), "state": state,
            "fraction_of_hard": round(c / hard_usd, 4) if hard_usd else None}


def spend_view(ledger: UsageLedger | str | Path, key_metadata: dict | None = None, *,
               warn_usd: float = EXPERIMENT_WARN_USD,
               hard_usd: float = EXPERIMENT_HARD_STOP_USD) -> dict[str, Any]:
    """What the GUI shows: the local ledger summary + budget state, and —
    when supplied — OpenRouter's own numbers for the key, clearly labelled
    as account-side and never substituted for the local policy figure."""
    local = ledger_summary(ledger)
    out = {
        "local_ledger": local,
        "budget": budget_status(local["cumulative_cost"], warn_usd=warn_usd, hard_usd=hard_usd),
        "openrouter_key": None,
    }
    if key_metadata:
        if key_metadata.get("ok"):
            out["openrouter_key"] = {
                "source": "GET /api/v1/key (OpenRouter-reported)",
                "label": key_metadata.get("label"),
                "usage_usd": key_metadata.get("usage"),
                "limit_usd": key_metadata.get("limit"),
                "limit_remaining_usd": key_metadata.get("limit_remaining"),
                "is_free_tier": key_metadata.get("is_free_tier"),
                "rate_limit": key_metadata.get("rate_limit"),
            }
        else:
            out["openrouter_key"] = {"source": "GET /api/v1/key", "error": key_metadata.get("detail")}
    return out


__all__ = ["EXPERIMENT_HARD_STOP_USD", "EXPERIMENT_WARN_USD", "ledger_summary",
           "budget_status", "spend_view"]
