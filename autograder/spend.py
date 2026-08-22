"""Spend truth: the persistent local usage ledger + (later) OpenRouter's own
key usage, side by side — and the live-call preflight that orders them.

Two independent numbers, deliberately never merged into one:

* **local experiment ledger** — every gateway call is appended to a
  UsageLedger JSONL (task, model, tokens, reported cost). This is what the
  budget manager enforces against ($8 warning / $10 hard stop for the
  model-selection campaign). **It is the number that controls our local
  experiment ceiling**: it needs no account access and is attributable
  per task / per model / per run.
* **OpenRouter-reported key usage** — ``GET /api/v1/key``
  (backends.openrouter.fetch_key_metadata), fetched only on explicit demand
  once a credential exists. It is the account-side truth and catches spend
  made outside this product. The product must keep working when it is
  unavailable; when the two disagree BOTH are shown and neither overwrites
  the other.

Live-call sequence (Part 7 of the pre-API setup; ``campaign_preflight``):

    OPENROUTER_API_KEY exists
        -> GET /api/v1/key (explicit, on demand; the caller passes the fetcher)
        -> record the starting key usage as a checkpoint (state/key_usage_checkpoints.jsonl)
        -> compare with the local ledger (show both; flag disagreement)
        -> budget safe? (local cumulative + predicted < hard stop)
        -> call allowed

No function here performs a network call by itself; ``fetch_key_metadata``
is injected (and is never injected by anything in this repo today).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from .usage import UsageLedger, aggregate_by

#: The model-selection campaign policy (docs/model-selection.md §budget,
#: evaluation/model_selection/candidates.toml [budget]).
EXPERIMENT_HARD_STOP_USD = 10.00
EXPERIMENT_WARN_USD = 8.00

#: Local ledger vs account usage: above this absolute gap (USD) the two are
#: reported as DISAGREEING (spend outside this product, or a ledger gap).
LEDGER_KEY_DISAGREEMENT_USD = 0.05


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
            "fraction_of_hard": round(c / hard_usd, 4) if hard_usd else None,
            "controls_ceiling": "local ledger (this number); account-side usage is informational"}


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
        "disagreement": None,
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
            out["disagreement"] = compare_ledger_with_key(local["cumulative_cost"], key_metadata)
        else:
            out["openrouter_key"] = {"source": "GET /api/v1/key", "error": key_metadata.get("detail")}
    return out


def compare_ledger_with_key(local_cumulative: float, key_metadata: dict,
                            start_checkpoint_usage: float | None = None,
                            start_ledger_cumulative: float | None = None) -> dict[str, Any]:
    """Show both numbers; flag a disagreement; never reconcile by overwriting.
    When a campaign-start checkpoint is known, the account-side spend
    attributable to the campaign is ``usage_now - usage_at_start`` and is
    compared with the LOCAL LEDGER DELTA since that same checkpoint."""
    usage = key_metadata.get("usage")
    out: dict[str, Any] = {"local_ledger_usd": round(float(local_cumulative or 0), 6),
                           "key_usage_usd": usage, "key_usage_at_campaign_start_usd": start_checkpoint_usage,
                           "local_ledger_at_campaign_start_usd": start_ledger_cumulative,
                           "comparable": False, "disagree": None, "note": None}
    if usage is None:
        out["note"] = "OpenRouter did not report usage; local ledger stands alone"
        return out
    if start_checkpoint_usage is None:
        out["note"] = ("no campaign-start checkpoint: the key's lifetime usage is not attributable to this "
                       "campaign; shown for information only")
        return out
    attributable = round(float(usage) - float(start_checkpoint_usage), 6)
    ledger_delta = round(float(local_cumulative or 0) - float(start_ledger_cumulative or 0), 6)
    out.update({"key_usage_attributable_usd": attributable, "local_ledger_delta_usd": ledger_delta,
                "comparable": True, "gap_usd": round(attributable - ledger_delta, 6)})
    out["disagree"] = abs(out["gap_usd"]) > LEDGER_KEY_DISAGREEMENT_USD
    out["note"] = ("ledger and account agree" if not out["disagree"] else
                   "ledger and account DISAGREE — both shown; the local ledger still controls the ceiling; "
                   "investigate spend made outside this product or a ledger gap before continuing")
    return out


def record_key_usage_checkpoint(state_root: Path, key_metadata: dict, *, reason: str = "preflight",
                                ledger_cumulative: float | None = None,
                                now: str | None = None) -> dict[str, Any]:
    """Append a secret-free checkpoint of the OpenRouter-reported usage AND
    the local ledger cumulative at that moment
    (state/key_usage_checkpoints.jsonl). The FIRST checkpoint is the
    campaign start; later ones allow delta attribution."""
    p = Path(state_root) / "key_usage_checkpoints.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"ts": now or time.strftime("%Y-%m-%d %H:%M:%S"), "reason": reason,
           "usage": key_metadata.get("usage"), "limit": key_metadata.get("limit"),
           "limit_remaining": key_metadata.get("limit_remaining"), "ok": bool(key_metadata.get("ok")),
           "ledger_cumulative": ledger_cumulative}
    with p.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def key_usage_checkpoints(state_root: Path) -> list[dict]:
    p = Path(state_root) / "key_usage_checkpoints.jsonl"
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def campaign_preflight(*, credential_present: bool, fetch_key_metadata: Callable[[], dict] | None,
                       ledger: UsageLedger | str | Path, state_root: Path, predicted_cost: float = 0.0,
                       warn_usd: float = EXPERIMENT_WARN_USD, hard_usd: float = EXPERIMENT_HARD_STOP_USD,
                       now: str | None = None) -> dict[str, Any]:
    """The ordered live-call gate. Returns a decision dict with every step's
    outcome; ``allowed`` is True only when all steps pass. Makes a network
    call ONLY through the injected ``fetch_key_metadata`` (explicit, on
    demand) — pass None to run the local-only steps."""
    steps: list[dict] = []
    decision: dict[str, Any] = {"allowed": False, "steps": steps, "reason": None}
    # 1. credential
    steps.append({"step": "credential_present", "ok": bool(credential_present)})
    if not credential_present:
        decision["reason"] = "OpenRouter credential is not configured"
        return decision
    # 2. GET /api/v1/key (explicit)
    meta = None
    if fetch_key_metadata is None:
        steps.append({"step": "key_metadata", "ok": None, "note": "not fetched (no fetcher injected)"})
    else:
        meta = fetch_key_metadata()
        steps.append({"step": "key_metadata", "ok": bool(meta and meta.get("ok")),
                      "usage": (meta or {}).get("usage"), "limit_remaining": (meta or {}).get("limit_remaining"),
                      "detail": (meta or {}).get("detail")})
        if not meta or not meta.get("ok"):
            decision["reason"] = f"OpenRouter key metadata unavailable: {(meta or {}).get('detail')}"
            return decision
    # 3. checkpoint starting usage (key usage + local ledger at the same instant)
    local = ledger_summary(ledger)
    checkpoints = key_usage_checkpoints(state_root)
    start = checkpoints[0]["usage"] if checkpoints else None
    start_ledger = checkpoints[0].get("ledger_cumulative") if checkpoints else None
    if meta is not None:
        record_key_usage_checkpoint(state_root, meta, reason="preflight",
                                    ledger_cumulative=local["cumulative_cost"], now=now)
        if start is None:
            start, start_ledger = meta.get("usage"), local["cumulative_cost"]
        steps.append({"step": "record_starting_key_usage", "ok": True, "campaign_start_usage": start,
                      "campaign_start_ledger": start_ledger})
    # 4. compare with local ledger (both shown; no overwrite)
    cmp = (compare_ledger_with_key(local["cumulative_cost"], meta or {}, start, start_ledger)
           if meta is not None else None)
    steps.append({"step": "compare_ledger_with_key", "ok": (cmp is None) or (not cmp.get("disagree")),
                  "local_ledger_usd": local["cumulative_cost"], "comparison": cmp})
    # 5. budget safe? local ledger controls the ceiling
    projected = local["cumulative_cost"] + float(predicted_cost or 0.0)
    safe = projected < hard_usd
    st = budget_status(local["cumulative_cost"], warn_usd=warn_usd, hard_usd=hard_usd)
    steps.append({"step": "budget_safe", "ok": safe, "local_cumulative_usd": local["cumulative_cost"],
                  "predicted_call_usd": predicted_cost, "projected_usd": round(projected, 6),
                  "hard_stop_usd": hard_usd, "state": st["state"],
                  "warning": st["state"] != "OK"})
    if not safe:
        decision["reason"] = (f"refused BEFORE the provider request: projected cumulative spend "
                              f"${projected:.4f} would reach the ${hard_usd:.2f} hard stop")
        return decision
    if cmp is not None and cmp.get("disagree"):
        decision["reason"] = "ledger and OpenRouter usage disagree — both shown; resolve before continuing"
        decision["allowed"] = False
        return decision
    decision["allowed"] = True
    decision["reason"] = "call allowed" + (" (WARNING: above $8 soft threshold)" if st["state"] == "WARNING" else "")
    return decision


__all__ = ["EXPERIMENT_HARD_STOP_USD", "EXPERIMENT_WARN_USD", "LEDGER_KEY_DISAGREEMENT_USD",
           "ledger_summary", "budget_status", "spend_view", "compare_ledger_with_key",
           "record_key_usage_checkpoint", "key_usage_checkpoints", "campaign_preflight"]
