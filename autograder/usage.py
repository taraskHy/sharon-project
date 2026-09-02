"""OpenRouter/gateway usage ledger + budget manager.

Ledger: one JSONL line per gateway call (cloud AND local AND cache hits),
numbers/ids only — never prompt content, never secrets. Aggregates:
requests/tokens per exam and per question, cache-hit rate, share of exams
graded entirely without cloud calls, share requiring escalation.

Budget: configurable soft/hard thresholds on calls per job, calls per
exam, input/output tokens, reported cost, and optional daily/monthly
caps. Soft -> warning callback; hard -> raise BudgetExceeded so the caller
pauses model-dependent work SAFELY (completed results stay persisted and
resumable). Never downgrades to a different model.
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

CLOUD_BACKENDS = {"openrouter", "gemini", "anthropic"}

_OPENROUTER_MARKER = "openrouter.ai"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "::1", "host.docker.internal"}
_PRIVATE_NET = re.compile(r"^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)")


def _is_local_url(base_url: str | None) -> bool:
    if not base_url:
        # No URL: the backend name decides (provider-default endpoints).
        return True
    host = (urlparse(base_url).hostname or "").lower()
    if host in _LOCAL_HOSTS or host.endswith(".local"):
        return True
    # RFC1918 ranges apply ONLY to literal IPv4 addresses. A DNS name that
    # merely STARTS with a private prefix (10.0.0.1.evil.example) must stay
    # remote, or the cloud boundary could be slipped by hostname dressing.
    parts = host.split(".")
    is_ipv4 = (len(parts) == 4
               and all(p.isdigit() and len(p) <= 3 and int(p) <= 255
                       for p in parts))
    return bool(is_ipv4 and _PRIVATE_NET.match(host))


def effective_provider(backend: str | None, base_url: str | None) -> str:
    """The provider a request will ACTUALLY reach.

    Cloud classification must follow the effective configuration, never the
    nominal backend string alone: an OpenAI-compatible route pointed at
    openrouter.ai IS OpenRouter for budget, ledger, and provider metadata.
    """
    b = (backend or "").lower()
    if _OPENROUTER_MARKER in (base_url or "").lower():
        return "openrouter"
    if b == "ollama_native":
        return "ollama"
    return b


def is_cloud_route(backend: str | None, base_url: str | None) -> bool:
    """True when a route reaches a remote paid/cloud provider.

    Local OpenAI-compatible servers (Ollama, vLLM, LM Studio, ...) are not
    cloud. A NON-local OpenAI-compatible endpoint is treated as cloud even
    when it is not OpenRouter — an unknown remote endpoint must never escape
    accounting by being nominally 'openai'.
    """
    p = effective_provider(backend, base_url)
    if p in CLOUD_BACKENDS:
        return True
    if p == "openai":
        return not _is_local_url(base_url)
    return False


def _row_is_cloud(entry: dict) -> bool:
    """Cloud classification for a persisted ledger row. New rows carry the
    computed 'cloud' flag; rows written before the effective-provider fix
    fall back to the backend-name rule they were classified under."""
    if "cloud" in entry:
        return bool(entry.get("cloud"))
    return entry.get("backend") in CLOUD_BACKENDS


class UsageLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, entry: dict) -> None:
        safe = {k: v for k, v in entry.items()
                if k not in ("api_key", "authorization", "prompt", "content")}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe, ensure_ascii=False) + "\n")

    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    # -- aggregates ------------------------------------------------------------

    def aggregate(self, job_id: str | None = None) -> dict:
        rows = [e for e in self.entries() if job_id is None or e.get("job_id") == job_id]
        # A reconciliation books historical MONEY, not a request: counting it as
        # a call would corrupt calls-per-exam, cache-hit rate and token averages.
        # Its cost is still budgeted (see BudgetManager._ledger_cost).
        cloud = [e for e in rows if _row_is_cloud(e) and not is_reconciliation(e)]
        cloud_calls = [e for e in cloud if not e.get("cache_hit")]
        n_cache = sum(1 for e in cloud if e.get("cache_hit"))
        by_exam: dict[str, dict] = defaultdict(lambda: {"cloud_calls": 0, "tokens": 0, "escalations": 0})
        by_q: dict[str, dict] = defaultdict(lambda: {"cloud_calls": 0, "tokens": 0})
        for e in cloud_calls:
            ex = e.get("exam_id") or "?"
            by_exam[ex]["cloud_calls"] += 1
            by_exam[ex]["tokens"] += int(e.get("total_tokens") or 0)
            if (e.get("task") or "").endswith("escalate") or e.get("stage") == "escalation":
                by_exam[ex]["escalations"] += 1
            q = f"{ex}:{e.get('question_id')}"
            by_q[q]["cloud_calls"] += 1
            by_q[q]["tokens"] += int(e.get("total_tokens") or 0)
        exams_seen = {e.get("exam_id") for e in rows if e.get("exam_id")}
        exams_cloud = {e.get("exam_id") for e in cloud_calls if e.get("exam_id")}
        exams_escalated = {ex for ex, v in by_exam.items() if v["escalations"]}
        n_exams = len(exams_seen)
        written = [e for e in cloud_calls if (e.get("task") or "").startswith(("ocr", "grade"))]
        return {
            "requests_total": len(rows),
            "cloud_requests": len(cloud_calls),
            "cloud_cache_hits": n_cache,
            "cache_hit_rate": round(n_cache / len(cloud), 4) if cloud else None,
            "input_tokens": sum(int(e.get("input_tokens") or 0) for e in cloud_calls),
            "output_tokens": sum(int(e.get("output_tokens") or 0) for e in cloud_calls),
            "reasoning_tokens": sum(int(e.get("reasoning_tokens") or 0) for e in cloud_calls),
            "total_tokens": sum(int(e.get("total_tokens") or 0) for e in cloud_calls),
            "reported_cost": round(sum(float(e.get("reported_cost") or 0) for e in cloud_calls), 6),
            "exams": n_exams,
            "requests_per_exam": round(len(cloud_calls) / n_exams, 3) if n_exams else None,
            "tokens_per_exam": round(sum(int(e.get("total_tokens") or 0) for e in cloud_calls) / n_exams, 1) if n_exams else None,
            "requests_per_written_answer": round(len(written) / len(by_q), 3) if by_q else None,
            "tokens_per_question": round(sum(v["tokens"] for v in by_q.values()) / len(by_q), 1) if by_q else None,
            "pct_exams_fully_local": round(100 * (n_exams - len(exams_cloud)) / n_exams, 1) if n_exams else None,
            "pct_exams_cloud_escalated": round(100 * len(exams_escalated) / n_exams, 1) if n_exams else None,
        }


class BudgetExceeded(RuntimeError):
    """Hard budget reached: caller must pause model-dependent processing."""


@dataclass
class BudgetLimits:
    max_calls_per_job: int | None = None
    max_calls_per_exam: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    max_cost: float | None = None
    # Cross-RUN ceiling: cumulative reported cost summed over the PERSISTED
    # ledger of this state root (all processes/runs that share it), so an
    # experiment campaign keeps one hard total (e.g. 10.00 USD) across many
    # benchmark invocations. The soft_fraction warning applies (0.8 -> warn
    # at 8.00 of 10.00). Requires every run of the campaign to use the SAME
    # state root / ledger file.
    max_cost_total: float | None = None
    max_calls_per_day: int | None = None
    max_calls_per_month: int | None = None
    soft_fraction: float = 0.8

    # config-key aliases accepted from models.toml [budget]
    _ALIASES = {
        "max_input_tokens_per_job": "max_input_tokens",
        "max_output_tokens_per_job": "max_output_tokens",
        "max_cost_per_job": "max_cost",
    }

    @classmethod
    def from_config(cls, section: dict | None) -> "BudgetLimits | None":
        """Build from a models.toml [budget] table. Missing section or
        enabled=false -> None (no budget). 0 / omitted / null = unlimited."""
        if not section or not section.get("enabled", True):
            return None
        kw = {}
        for k, v in section.items():
            if k == "enabled":
                continue
            name = cls._ALIASES.get(k, k)
            if name not in cls.__dataclass_fields__:
                raise ValueError(f"[budget]: unknown key {k!r}")
            if name == "soft_fraction":
                kw[name] = float(v)
            else:
                kw[name] = None if (v is None or float(v) <= 0) else (
                    float(v) if name in ("max_cost", "max_cost_total") else int(v))
        return cls(**kw)

    def effective(self) -> dict:
        """Human-readable limits for the settings UI (unlimited shown as such)."""
        out = {}
        for f in self.__dataclass_fields__:
            if f.startswith("_"):
                continue
            v = getattr(self, f)
            out[f] = v if v is not None else ("—" if f != "soft_fraction" else v)
        return out   # warn at 80% of any hard limit


@dataclass
class BudgetManager:
    limits: BudgetLimits
    ledger: UsageLedger | None = None
    warn: Callable[[str], None] = field(default=lambda msg: None)
    cloud_only: bool = True
    _calls_job: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _calls_exam: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    _in_tok: int = 0
    _out_tok: int = 0
    _cost: float = 0.0
    _warned: set = field(default_factory=set)
    paused: bool = False
    pause_reason: str | None = None

    def _counts_toward(self, route) -> bool:
        # Effective-provider rule: backend="openai" + an OpenRouter base_url is
        # OpenRouter and must never escape budget enforcement by its name.
        return (not self.cloud_only) or is_cloud_route(
            getattr(route, "backend", None), getattr(route, "base_url", None))

    def _check_one(self, name: str, value: float, limit: float | None) -> None:
        """`value` is the amount that WOULD be consumed if this call proceeds
        (for counters: current+1). A limit of N permits exactly N."""
        if limit is None:
            return
        if value > limit:
            self.paused, self.pause_reason = True, f"hard budget {name} reached ({value} > {limit})"
            raise BudgetExceeded(self.pause_reason)
        if value >= limit * self.limits.soft_fraction and name not in self._warned:
            self._warned.add(name)
            self.warn(f"soft budget {name}: {value} of {limit}")

    def _ledger_cost(self) -> float:
        """Cumulative reported cost over the PERSISTED ledger (cloud,
        non-cache rows). Every gateway call is ledger-recorded before the
        next check runs, so this already includes this process's spend —
        never add the in-memory ``_cost`` on top (it would double count)."""
        if self.ledger is None:
            return self._cost
        return sum(float(e.get("reported_cost") or 0) for e in self.ledger.entries()
                   if _row_is_cloud(e) and not e.get("cache_hit"))
        # NOTE: non-billable rows (pre-inference rejections) carry
        # reported_cost 0, so they are counted structurally but add nothing.

    def check(self, *, task: str, route, meta: dict,
              predicted_cost: float = 0.0) -> None:
        """Called BEFORE a non-cached provider call.

        ``predicted_cost`` is a deterministic pre-call estimate of THIS call's
        cost (see ``predicted_call_cost``): with it, the call that WOULD cross
        a cost ceiling is refused, instead of only its successor."""
        if not self._counts_toward(route):
            return
        job, exam = meta.get("job_id") or "?", meta.get("exam_id") or "?"
        L = self.limits
        self._check_one("calls_per_job", self._calls_job[job] + 1, L.max_calls_per_job)
        self._check_one("calls_per_exam", self._calls_exam[exam] + 1, L.max_calls_per_exam)
        self._check_one("input_tokens", self._in_tok, L.max_input_tokens)
        self._check_one("output_tokens", self._out_tok, L.max_output_tokens)
        self._check_one("cost", self._cost + predicted_cost, L.max_cost)
        if L.max_cost_total is not None:
            self._check_one("cost_total", self._ledger_cost() + predicted_cost,
                            L.max_cost_total)
        if self.ledger is not None and (L.max_calls_per_day or L.max_calls_per_month):
            today = time.strftime("%Y-%m-%d")
            month = today[:7]
            rows = [e for e in self.ledger.entries()
                    if _row_is_cloud(e) and not e.get("cache_hit")]
            self._check_one("calls_per_day", sum(1 for e in rows if str(e.get("ts", "")).startswith(today)) + 1,
                            L.max_calls_per_day)
            self._check_one("calls_per_month", sum(1 for e in rows if str(e.get("ts", "")).startswith(month)) + 1,
                            L.max_calls_per_month)

    def charge(self, *, task: str, route, usage: dict, meta: dict) -> None:
        """Called AFTER a provider call succeeded."""
        if not self._counts_toward(route):
            return
        self._calls_job[meta.get("job_id") or "?"] += 1
        self._calls_exam[meta.get("exam_id") or "?"] += 1
        self._in_tok += int(usage.get("input_tokens") or 0)
        self._out_tok += int(usage.get("output_tokens") or 0)
        self._cost += float(usage.get("reported_cost") or 0)

    def snapshot(self) -> dict:
        return {"input_tokens": self._in_tok, "output_tokens": self._out_tok,
                "cost": round(self._cost, 6), "calls_per_job": dict(self._calls_job),
                "paused": self.paused, "pause_reason": self.pause_reason,
                "limits": self.limits.effective()}


# ------------------------------------------------------------------------
# experiment-budget helpers (model-selection campaigns)
# ------------------------------------------------------------------------


def reconcile_cost(usage: dict, route=None, pricing: dict | None = None) -> tuple[float, str]:
    """The charge to record for one provider response, and where it came from.

    Precedence is deliberate and one-way:

    1. ``reported_cost`` — the PROVIDER's own number (OpenRouter returns it in
       ``usage.cost``). Authoritative: it already accounts for the provider
       actually routed to, cache discounts, and promotional pricing. Never
       overwritten by a local calculation.
    2. local ``[pricing]`` x reported tokens — only when the provider reported
       no cost but did report tokens (generic OpenAI-compatible servers).
       Marked ``local_pricing`` so a reconciliation report can separate
       measured spend from estimated spend.
    3. ``0.0`` / ``none`` — nothing billable was reported.

    Returns ``(cost_usd, source)`` where source is
    ``provider`` | ``local_pricing`` | ``none``.
    """
    if usage and usage.get("reported_cost") is not None:
        return float(usage["reported_cost"]), "provider"
    tin = int((usage or {}).get("input_tokens") or 0)
    tout = int((usage or {}).get("output_tokens") or 0)
    if not (tin or tout):
        return 0.0, "none"
    model = (usage or {}).get("model") or getattr(route, "model", None)
    row = (pricing or {}).get(model) if model else None
    if not row:
        return 0.0, "none"
    cost = tin / 1_000_000 * float(row.get("input", 0.0)) +         tout / 1_000_000 * float(row.get("output", 0.0))
    return round(cost, 8), "local_pricing"


def predicted_call_cost(route, system: str, content_blocks: list[dict],
                        pricing: dict | None) -> float:
    """Deterministic, conservative pre-call cost estimate in USD.

    Uses ONLY the local models.toml [pricing] table (USD per 1M tokens, never
    fetched from the network): text sized at chars/4 tokens, a flat 1100
    tokens per image block, and the route's full ``max_tokens`` as output.
    A model absent from the pricing table estimates 0.0 — the budget then
    enforces known (ledger) spend only, so campaigns MUST list every cloud
    candidate in [pricing] for refuse-before-crossing to work."""
    if not pricing:
        return 0.0
    price = pricing.get(getattr(route, "model", "") or "")
    if not isinstance(price, dict):
        return 0.0
    chars = len(system or "")
    images = 0
    for block in content_blocks or []:
        if block.get("type") == "text":
            chars += len(block.get("text") or "")
        elif block.get("type") == "image":
            images += 1
    input_tokens = chars / 4 + images * 1100
    output_tokens = float(getattr(route, "max_tokens", 0) or 0)
    return (input_tokens * float(price.get("input") or 0)
            + output_tokens * float(price.get("output") or 0)) / 1e6


def aggregate_by(rows: list[dict], key: str) -> dict[str, dict]:
    """Cloud non-cache calls grouped by one ledger field (model / task):
    calls, tokens, reported cost. Raw material for experiment run reports."""
    out: dict[str, dict] = {}
    for e in rows:
        if not _row_is_cloud(e) or e.get("cache_hit"):
            continue
        k = str(e.get(key) or "?")
        g = out.setdefault(k, {"calls": 0, "input_tokens": 0, "output_tokens": 0,
                               "total_tokens": 0, "reported_cost": 0.0})
        g["calls"] += 1
        g["input_tokens"] += int(e.get("input_tokens") or 0)
        g["output_tokens"] += int(e.get("output_tokens") or 0)
        g["total_tokens"] += int(e.get("total_tokens") or 0)
        g["reported_cost"] = round(g["reported_cost"] + float(e.get("reported_cost") or 0), 6)
    return out


def run_cost_report(ledger: UsageLedger, baseline_rows: int) -> dict:
    """The mandatory per-benchmark-run accounting: cost before / run cost /
    cost after, plus calls and tokens by model and by task.

    ``baseline_rows`` is ``len(ledger.entries())`` captured BEFORE the run
    started; everything after that offset is attributed to the run."""
    rows = ledger.entries()
    before, run = rows[:baseline_rows], rows[baseline_rows:]

    def _cost(rs: list[dict]) -> float:
        return round(sum(float(e.get("reported_cost") or 0) for e in rs
                         if _row_is_cloud(e) and not e.get("cache_hit")), 6)

    cost_before = _cost(before)
    run_cost = _cost(run)
    return {
        "rows_before": baseline_rows,
        "rows_after": len(rows),
        "cost_before": cost_before,
        "run_cost": run_cost,
        "cost_after": round(cost_before + run_cost, 6),
        "run_calls": sum(1 for e in run if _row_is_cloud(e) and not e.get("cache_hit")),
        "run_cache_hits": sum(1 for e in run if _row_is_cloud(e) and e.get("cache_hit")),
        "by_model": aggregate_by(run, "model"),
        "by_task": aggregate_by(run, "task"),
    }


# ------------------------------------------------------------------------
# historical spend reconciliation
# ------------------------------------------------------------------------

#: Ledger rows are provider calls unless they say otherwise. Rows written
#: before this field existed have no ``entry_kind`` and are provider calls.
ENTRY_KIND_PROVIDER_CALL = "provider_call"
ENTRY_KIND_RECONCILIATION = "reconciliation"


def entry_kind(row: dict) -> str:
    return row.get("entry_kind") or ENTRY_KIND_PROVIDER_CALL


def is_reconciliation(row: dict) -> bool:
    return entry_kind(row) == ENTRY_KIND_RECONCILIATION


def record_reconciliation(ledger: "UsageLedger", *, amount_usd: float, reason: str,
                          provenance: str, occurred_at: str | None = None,
                          model: str | None = None, task: str | None = None,
                          related_case_ids: list[str] | None = None,
                          ts: str | None = None) -> dict:
    """Book money the provider charged that the ledger never recorded.

    The pre-2026-08-24 gateway wrote its ledger row only after ``parse()``
    returned, so a response that was generated, billed, and then discarded
    (truncation, schema failure) left no trace. Fixing the code stops FUTURE
    losses; it does not un-spend the money already gone. Until the historical
    amount is booked, every budget decision — the $8 warning, the $10 ceiling,
    "how much is left" — is computed against a number that is known to be too
    small.

    This entry is NOT a provider call:

    * ``entry_kind = "reconciliation"`` keeps it out of call counts, token
      aggregates and cache statistics;
    * ``reported_cost`` DOES count toward cumulative spend, which is the whole
      point;
    * token fields are omitted entirely when the provider's counts were never
      captured. A reconciliation must never invent tokens to look complete.

    ``amount_usd`` must be positive: this books a known past charge, it is not
    a mechanism for adjusting spend downward.
    """
    if amount_usd <= 0:
        raise ValueError("a reconciliation books a positive past charge; "
                         f"got {amount_usd!r}")
    entry = {
        "ts": ts or time.strftime("%Y-%m-%d %H:%M:%S"),
        "entry_kind": ENTRY_KIND_RECONCILIATION,
        "provider_call": False,
        "cloud": True,               # it is cloud spend, and must be budgeted as such
        "cache_hit": False,
        "task": task,
        "model": model,
        "backend": "openrouter",
        "effective_provider": "openrouter",
        "reported_cost": round(float(amount_usd), 8),
        "cost_source": "provider_account_delta",
        "occurred_at": occurred_at,
        "reason": reason,
        "provenance": provenance,
        "related_case_ids": list(related_case_ids or []),
        # tokens deliberately absent: they were never captured, and a
        # reconciliation must not fabricate them
        "tokens_available": False,
    }
    ledger.record(entry)
    return entry


def reconciled_total(ledger: "UsageLedger") -> dict:
    """Cumulative cloud spend split by where the number came from."""
    rows = ledger.entries()
    calls = sum(float(r.get("reported_cost") or 0) for r in rows
                if _row_is_cloud(r) and not r.get("cache_hit") and not is_reconciliation(r))
    recon = sum(float(r.get("reported_cost") or 0) for r in rows if is_reconciliation(r))
    return {
        "provider_calls_usd": round(calls, 8),
        "reconciliation_usd": round(recon, 8),
        "total_usd": round(calls + recon, 8),
        "provider_call_rows": sum(1 for r in rows
                                  if _row_is_cloud(r) and not r.get("cache_hit")
                                  and not is_reconciliation(r)),
        "reconciliation_rows": sum(1 for r in rows if is_reconciliation(r)),
    }
