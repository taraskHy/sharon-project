"""Provider-independent inference interface.

The grading pipeline talks to a ``VisionBackend`` and never to a provider SDK.
Every backend implements the same contract:

- ``parse(system=..., content_blocks=..., output_model=..., max_tokens=...)``
  sends one structured vision request (text + base64 PNG images, in the
  pipeline's neutral block format) and returns a validated Pydantic object.
- ``health_check()`` verifies the backend is reachable and the model exists.
- ``describe()`` returns the exact backend/model/configuration identity that
  is recorded in results and in the resume fingerprint. It must never contain
  secrets.

Neutral content-block format (produced by ``autograder.ingest``):

    {"type": "text", "text": "..."}
    {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                 "data": "<base64>"}}

Failure policy: a backend must raise ``BackendError`` — it must never fall
back to a different model silently.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BackendError(RuntimeError):
    """Any inference failure: transport, timeout, refusal, truncation,
    malformed output that survived all retries, or configuration problems."""


# Backwards-compatible alias: the pipeline historically raised LLMError.
LLMError = BackendError


@dataclass
class BackendConfig:
    """Configuration for constructing a backend.

    Loaded from CLI flags, an optional TOML file, and environment variables.
    ``api_key_env`` names the environment variable holding the key — the key
    itself is never stored in configuration, results, or fingerprints.
    """

    backend: str = "openai"  # openai | mock | anthropic
    model: str = ""
    base_url: str | None = None  # e.g. http://localhost:11434/v1 for Ollama
    api_key_env: str = "GRADER_API_KEY"
    structured_mode: str = "json_schema"  # json_schema | json_object | prompt
    #: Emit strict-provider-compatible schemas (closed objects, every property
    #: required, optionality expressed as an explicit null). Required by
    #: OpenAI/Azure structured output; harmless for permissive servers. Set
    #: False only for a server that demonstrably rejects the strict form.
    strict_schema: bool = True
    max_tokens: int = 16000
    temperature: float | None = 0.0  # None = provider default
    timeout_s: float = 300.0
    transport_retries: int = 2  # network/5xx/429 retries
    validation_retries: int = 2  # malformed-JSON repair round-trips
    extra_generation: dict[str, Any] = field(default_factory=dict)
    # Batch-evaluation knobs (used by eval-batch, not per-request):
    concurrency: int = 1

    def api_key(self) -> str | None:
        for name in (self.api_key_env, "OPENAI_API_KEY"):
            value = os.environ.get(name)
            if value:
                return value
        return None


@dataclass
class HealthReport:
    ok: bool
    backend: str
    model: str
    detail: str


@dataclass
class BillingEvent:
    """ONE provider HTTP response that may have incurred a charge.

    A charge is created by the PROVIDER, not by our ability to parse what it
    sent back. Every response therefore produces one of these BEFORE any
    parsing/validation runs, so a downstream failure can never erase the
    money. The lifecycle flags are recorded separately and never collapsed:

    ``call_attempted``   an HTTP request left this process
    ``inference_reached``the provider ran the model (200, or an error body
                         that still reports token usage)
    ``usage_returned``   the response carried a usage block we could read
    ``parse_ok``         the body validated against the output schema; set
                         later by the caller, None until then

    ``billable`` is the accounting decision: True when the provider reported
    tokens or a cost. A pre-inference rejection (HTTP 400 invalid schema,
    401, 404) carries no usage and is NOT billable — but it is still
    recorded, so "we called and were refused" is distinguishable from "we
    never called".
    """

    usage: dict[str, Any] = field(default_factory=dict)
    http_status: int | None = None
    call_attempted: bool = True
    inference_reached: bool = False
    usage_returned: bool = False
    parse_ok: bool | None = None
    finish_reason: str | None = None
    attempt: int = 1
    error: str | None = None

    @property
    def billable(self) -> bool:
        u = self.usage or {}
        return bool(
            (u.get("input_tokens") or 0)
            or (u.get("output_tokens") or 0)
            or (u.get("total_tokens") or 0)
            or (u.get("reported_cost") or 0)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "http_status": self.http_status,
            "call_attempted": self.call_attempted,
            "inference_reached": self.inference_reached,
            "usage_returned": self.usage_returned,
            "parse_ok": self.parse_ok,
            "finish_reason": self.finish_reason,
            "attempt": self.attempt,
            "billable": self.billable,
            "error": self.error,
            **dict(self.usage or {}),
        }


class VisionBackend(ABC):
    """Contract every inference backend implements."""

    config: BackendConfig

    #: Every provider response of the CURRENT ``parse()`` call, in order.
    #: Backends that talk to a paid provider must append one entry per HTTP
    #: response (see ``BillingEvent``). Reset at the start of each ``parse()``.
    billing_events: list["BillingEvent"]

    @abstractmethod
    def parse(
        self,
        *,
        system: str,
        content_blocks: list[dict],
        output_model: type[T],
        max_tokens: int | None = None,
    ) -> T:
        """One structured request. Raises BackendError on any failure."""

    @abstractmethod
    def health_check(self) -> HealthReport:
        """Cheap reachability + model-availability check (no full inference)."""

    def describe(self) -> dict[str, Any]:
        """Exact identity of this backend for results and fingerprints."""
        c = self.config
        return {
            "backend": c.backend,
            "model": c.model,
            "base_url": c.base_url,
            "structured_mode": c.structured_mode,
            "max_tokens": c.max_tokens,
            "temperature": c.temperature,
            "extra_generation": dict(sorted(c.extra_generation.items())),
        }

    @property
    def identity(self) -> str:
        """Short human-readable "backend:model" string."""
        return f"{self.config.backend}:{self.config.model}"


def extract_json_object(text: str) -> str:
    """Pull the first JSON object out of a model reply.

    Open models often wrap JSON in markdown fences or add prose around it,
    especially in 'prompt' structured mode. Returns the JSON substring or the
    original text when no braces are found (validation will then fail with a
    clear error).
    """
    s = text.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1 :]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
        s = s.strip()
    start = s.find("{")
    if start == -1:
        return s
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return s[start:]
