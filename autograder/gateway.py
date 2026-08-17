"""Central provider-independent model gateway.

Application code asks for a TASK, never a provider:

    gw = ModelGateway.from_file("models.toml")
    result = gw.call(task="grade_primary", system=..., content_blocks=...,
                     output_model=GradeResult, meta={"job_id": ..., ...})

Each task maps (in configuration only) to a backend + model + generation
knobs. Backends are the existing ``VisionBackend`` implementations
(openai-compat/Ollama, openrouter, mock, ...) — the gateway adds routing,
a deterministic request cache, a usage ledger, and budget enforcement
around ``parse()``. No model identifier is hardcoded anywhere in
application logic; ``${ENV_VAR}`` references in the config are expanded
at load time.

Task names are open strings, but the pipeline uses these conventionally:
ocr_primary, ocr_verify, grade_primary, grade_escalate, variant_resolve,
mc_resolve, policy_infer, package_inspect.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from .backends import BackendConfig, BackendError, VisionBackend, create_backend

T = TypeVar("T", bound=BaseModel)

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class GatewayConfigError(ValueError):
    """Bad or incomplete gateway configuration (never raised mid-request)."""


def _expand_env(value: Any) -> Any:
    """Expand ${VAR} references. Unset variables expand to '' so a task with
    an unconfigured model fails validation loudly rather than at call time."""
    if isinstance(value, str):
        return _ENV_REF.sub(lambda m: os.environ.get(m.group(1), ""), value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


@dataclass
class TaskRoute:
    """One task's resolved configuration."""

    task: str
    backend: str
    model: str
    base_url: str | None = None
    structured_mode: str = "json_schema"
    max_tokens: int = 800
    temperature: float | None = 0.0
    timeout_s: float = 300.0
    reasoning: dict[str, Any] | None = None       # e.g. {"effort": "low"} (openrouter)
    provider: dict[str, Any] | None = None        # openrouter provider routing
    extra_generation: dict[str, Any] = field(default_factory=dict)
    prompt_version: str = "v1"                    # part of the cache fingerprint
    cacheable: bool = True
    enabled: bool = True

    def to_backend_config(self) -> BackendConfig:
        eg = dict(self.extra_generation)
        if self.reasoning:
            eg["reasoning"] = self.reasoning
        if self.provider:
            eg["provider"] = self.provider
        return BackendConfig(
            backend="openai" if self.backend in ("ollama", "openai") else self.backend,
            model=self.model,
            base_url=self.base_url,
            structured_mode=self.structured_mode,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            timeout_s=self.timeout_s,
            extra_generation=eg,
        )

    def fingerprint_fields(self) -> dict[str, Any]:
        """Everything about the route that changes the meaning of a result."""
        return {
            "task": self.task, "backend": self.backend, "model": self.model,
            "structured_mode": self.structured_mode, "max_tokens": self.max_tokens,
            "temperature": self.temperature, "reasoning": self.reasoning,
            "extra_generation": dict(sorted(self.extra_generation.items())),
            "prompt_version": self.prompt_version,
        }


@dataclass
class CallResult:
    """What gateway.call returns alongside the parsed object."""

    value: Any
    task: str
    route: TaskRoute
    cache_hit: bool
    latency_s: float
    usage: dict[str, Any] = field(default_factory=dict)
    fingerprint: str | None = None
    retries: int = 0


class ModelGateway:
    """Routes tasks to backends; wraps calls with cache, ledger, budget hooks."""

    def __init__(self, routes: dict[str, TaskRoute], *,
                 backend_factory: Callable[[BackendConfig], VisionBackend] | None = None,
                 cache=None, ledger=None, budget=None):
        if not routes:
            raise GatewayConfigError("gateway configuration defines no tasks")
        for name, r in routes.items():
            if r.enabled and not r.model:
                raise GatewayConfigError(
                    f"task {name!r} has no model configured (unset ${{ENV}} reference?)")
            if r.backend in ("ollama", "openai") and not r.base_url:
                raise GatewayConfigError(f"task {name!r} ({r.backend}) needs base_url")
        self.routes = routes
        self._factory = backend_factory or create_backend
        self._backends: dict[str, VisionBackend] = {}
        self.cache = cache        # duck-typed: get(fp) / put(fp, obj, meta)
        self.ledger = ledger      # duck-typed: record(entry: dict)
        self.budget = budget      # duck-typed: check(meta) -> None | raise; charge(entry)

    # -- construction --------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict, **kw) -> "ModelGateway":
        data = _expand_env(data)
        models = data.get("models") or {}
        defaults = data.get("defaults") or {}
        routes = {}
        for task, spec in models.items():
            merged = {**defaults, **(spec or {})}
            allowed = {f for f in TaskRoute.__dataclass_fields__}
            unknown = set(merged) - allowed
            if unknown:
                raise GatewayConfigError(f"task {task!r}: unknown keys {sorted(unknown)}")
            routes[task] = TaskRoute(task=task, **merged)
        return cls(routes, **kw)

    @classmethod
    def from_file(cls, path: str | Path, **kw) -> "ModelGateway":
        p = Path(path)
        text = p.read_text(encoding="utf-8")
        if p.suffix.lower() == ".json":
            return cls.from_dict(json.loads(text), **kw)
        import tomllib
        return cls.from_dict(tomllib.loads(text), **kw)

    # -- routing ---------------------------------------------------------------

    def route(self, task: str) -> TaskRoute:
        try:
            r = self.routes[task]
        except KeyError:
            raise GatewayConfigError(
                f"no route configured for task {task!r} "
                f"(configured: {sorted(self.routes)})") from None
        if not r.enabled:
            raise GatewayConfigError(f"task {task!r} is disabled in configuration")
        return r

    def backend_for(self, task: str) -> VisionBackend:
        r = self.route(task)
        key = f"{r.backend}|{r.model}|{r.base_url}|{json.dumps(r.reasoning, sort_keys=True)}|{json.dumps(r.provider, sort_keys=True)}"
        if key not in self._backends:
            self._backends[key] = self._factory(r.to_backend_config())
        return self._backends[key]

    def describe(self) -> dict[str, dict]:
        """Task -> configuration summary for the settings UI (no secrets)."""
        return {t: {"backend": r.backend, "model": r.model, "enabled": r.enabled,
                    "max_tokens": r.max_tokens, "reasoning": r.reasoning,
                    "cacheable": r.cacheable}
                for t, r in self.routes.items()}

    # -- the call --------------------------------------------------------------

    def call(self, *, task: str, system: str, content_blocks: list[dict],
             output_model: type[T], meta: dict | None = None,
             max_tokens: int | None = None) -> CallResult:
        route = self.route(task)
        meta = dict(meta or {})
        fp = None
        if self.cache is not None and route.cacheable:
            from .requestcache import fingerprint
            fp = fingerprint(route, system, content_blocks, output_model, max_tokens, meta)
            hit = self.cache.get(fp, output_model)
            if hit is not None:
                res = CallResult(value=hit, task=task, route=route, cache_hit=True,
                                 latency_s=0.0, fingerprint=fp)
                self._ledger_record(res, meta)
                return res
        if self.budget is not None:
            self.budget.check(task=task, route=route, meta=meta)
        backend = self.backend_for(task)
        t0 = time.monotonic()
        value = backend.parse(system=system, content_blocks=content_blocks,
                              output_model=output_model, max_tokens=max_tokens)
        dt = time.monotonic() - t0
        usage = dict(getattr(backend, "last_usage", {}) or {})
        res = CallResult(value=value, task=task, route=route, cache_hit=False,
                         latency_s=round(dt, 3), usage=usage, fingerprint=fp)
        if self.cache is not None and route.cacheable and fp:
            self.cache.put(fp, value, {"task": task, "model": route.model})
        self._ledger_record(res, meta)
        if self.budget is not None:
            self.budget.charge(task=task, route=route, usage=usage, meta=meta)
        return res

    def _ledger_record(self, res: CallResult, meta: dict) -> None:
        if self.ledger is None:
            return
        entry = {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "task": res.task, "backend": res.route.backend, "model": res.route.model,
            "cache_hit": res.cache_hit, "latency_s": res.latency_s,
            "retries": res.retries,
            **{k: meta.get(k) for k in ("job_id", "exam_id", "question_id", "stage")},
            **{k: res.usage.get(k) for k in (
                "provider", "request_id", "input_tokens", "cached_input_tokens",
                "output_tokens", "reasoning_tokens", "total_tokens", "reported_cost")},
        }
        self.ledger.record(entry)


__all__ = ["ModelGateway", "TaskRoute", "CallResult", "GatewayConfigError", "BackendError"]
