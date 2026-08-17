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
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

CLOUD_BACKENDS = {"openrouter", "gemini", "anthropic"}


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
        cloud = [e for e in rows if e.get("backend") in CLOUD_BACKENDS]
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
                kw[name] = None if (v is None or float(v) <= 0) else (float(v) if name == "max_cost" else int(v))
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
        return (not self.cloud_only) or route.backend in CLOUD_BACKENDS

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

    def check(self, *, task: str, route, meta: dict) -> None:
        """Called BEFORE a non-cached provider call."""
        if not self._counts_toward(route):
            return
        job, exam = meta.get("job_id") or "?", meta.get("exam_id") or "?"
        L = self.limits
        self._check_one("calls_per_job", self._calls_job[job] + 1, L.max_calls_per_job)
        self._check_one("calls_per_exam", self._calls_exam[exam] + 1, L.max_calls_per_exam)
        self._check_one("input_tokens", self._in_tok, L.max_input_tokens)
        self._check_one("output_tokens", self._out_tok, L.max_output_tokens)
        self._check_one("cost", self._cost, L.max_cost)
        if self.ledger is not None and (L.max_calls_per_day or L.max_calls_per_month):
            today = time.strftime("%Y-%m-%d")
            month = today[:7]
            rows = [e for e in self.ledger.entries()
                    if e.get("backend") in CLOUD_BACKENDS and not e.get("cache_hit")]
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
