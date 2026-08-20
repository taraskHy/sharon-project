"""Model / prompt drift protection: full provenance for every model-backed decision.

OpenRouter routing, provider mixes and model revisions change under a stable
slug. A grade produced today must therefore record enough to be re-examined
later: what we asked for, what actually answered, with which prompt, which
question pack, which inputs, which decoding and schema — and the provider's
own generation id, which is the only handle a provider-side investigation
can use.

Secrets never enter a provenance record (a test asserts it): no API keys, no
authorization headers, no base URLs carrying credentials.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

PROVENANCE_VERSION = "prov-v1"

_SECRET_HINTS = ("key", "token", "secret", "authorization", "password", "credential")


def _h(data: Any) -> str:
    if isinstance(data, bytes):
        return hashlib.sha256(data).hexdigest()[:16]
    if not isinstance(data, str):
        data = json.dumps(data, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]


def input_hashes(content_blocks: list[dict]) -> list[str]:
    """One hash per input block, in order: text by content, images by bytes."""
    out = []
    for b in content_blocks or []:
        if b.get("type") == "text":
            out.append("text:" + _h(b.get("text", "")))
        elif b.get("type") == "image":
            out.append("image:" + _h((b.get("source") or {}).get("data", "")))
        else:
            out.append(str(b.get("type")) + ":" + _h(b))
    return out


@dataclass
class DecisionProvenance:
    """Everything needed to reproduce/audit ONE model-backed decision."""

    task: str
    requested_model: str
    backend: str
    actual_provider: Optional[str] = None      # as reported by the provider, if any
    actual_model: Optional[str] = None         # as reported by the provider, if any
    prompt_version: str = "v1"
    prompt_hash: str = ""                      # system prompt
    input_hashes: list[str] = field(default_factory=list)
    pack_hash: Optional[str] = None
    schema_version: Optional[str] = None
    schema_hash: str = ""
    decoding: dict[str, Any] = field(default_factory=dict)     # temperature/max_tokens/...
    reasoning: dict[str, Any] | None = None
    request_id: Optional[str] = None
    generation_id: Optional[str] = None
    cache_hit: bool = False
    latency_s: Optional[float] = None
    ts: str = ""
    version: str = PROVENANCE_VERSION

    def as_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        """Identity of the CONFIGURATION that produced this decision — changes
        whenever the model, prompt, schema or decoding changes."""
        return _h({k: getattr(self, k) for k in (
            "task", "requested_model", "backend", "prompt_version", "prompt_hash",
            "schema_hash", "decoding", "reasoning", "pack_hash")})


def _no_secrets(d: dict) -> dict:
    return {k: v for k, v in (d or {}).items()
            if not any(h in str(k).lower() for h in _SECRET_HINTS)}


def provenance_from_call(result, *, system: str = "", content_blocks: list[dict] | None = None,
                         output_model=None, pack_hash: str | None = None,
                         schema_version: str | None = None) -> DecisionProvenance:
    """Build a provenance record from a ``gateway.CallResult``."""
    route = result.route
    usage = dict(getattr(result, "usage", {}) or {})
    decoding = _no_secrets({"max_tokens": route.max_tokens, "temperature": route.temperature,
                            "structured_mode": route.structured_mode,
                            **_no_secrets(dict(route.extra_generation or {}))})
    schema_hash = _h(output_model.model_json_schema()) if output_model is not None else ""
    return DecisionProvenance(
        task=result.task, requested_model=route.model, backend=route.backend,
        actual_provider=usage.get("provider"), actual_model=usage.get("model") or usage.get("actual_model"),
        prompt_version=route.prompt_version, prompt_hash=_h(system),
        input_hashes=input_hashes(content_blocks or []), pack_hash=pack_hash,
        schema_version=schema_version, schema_hash=schema_hash,
        decoding=decoding, reasoning=route.reasoning,
        request_id=usage.get("request_id"), generation_id=usage.get("generation_id") or usage.get("id"),
        cache_hit=bool(result.cache_hit), latency_s=getattr(result, "latency_s", None),
        ts=time.strftime("%Y-%m-%d %H:%M:%S"))


def drift_between(a: DecisionProvenance, b: DecisionProvenance) -> list[str]:
    """What changed between two decisions' configurations. Used to explain a
    canary regression, and to tell a lecturer why old and new results differ."""
    out = []
    for f in ("requested_model", "actual_model", "actual_provider", "backend",
              "prompt_version", "prompt_hash", "schema_hash", "pack_hash"):
        x, y = getattr(a, f), getattr(b, f)
        if x != y:
            out.append(f"{f}: {x!r} -> {y!r}")
    if a.decoding != b.decoding:
        out.append(f"decoding: {a.decoding} -> {b.decoding}")
    if a.reasoning != b.reasoning:
        out.append(f"reasoning: {a.reasoning} -> {b.reasoning}")
    return out
