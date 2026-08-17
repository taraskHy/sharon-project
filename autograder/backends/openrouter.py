"""Production OpenRouter backend.

Inherits ALL request plumbing from ``OpenAICompatBackend`` (structured
modes, transport retries with Retry-After, validation-repair round-trips,
truncation/refusal errors) and adds only what OpenRouter needs:

- key from ``OPENROUTER_API_KEY`` ONLY (never written, logged, or shown);
- ``HTTP-Referer`` / ``X-Title`` attribution headers;
- provider routing (``provider``: {order, allow_fallbacks, ...}) and
  ``reasoning`` ({effort} or {max_tokens}) request fields, both from config;
- capture of the last response's ``usage`` (prompt/completion/reasoning/
  cached tokens, reported cost) and OpenRouter request ``id`` in
  ``self.last_usage`` for the ledger — numbers and ids only, never content.

Failure policy is unchanged: raise ``BackendError``; never fall back to a
different model silently.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from .base import BackendConfig, BackendError, HealthReport
from .openai_compat import OpenAICompatBackend

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_KEY_ENV = "OPENROUTER_API_KEY"


def _redact(text: str, key: str | None) -> str:
    """Never let the key surface in any error text."""
    if key and key in text:
        return text.replace(key, "<redacted>")
    return text


class OpenRouterBackend(OpenAICompatBackend):
    def __init__(self, config: BackendConfig, transport: httpx.BaseTransport | None = None,
                 app_title: str = "sharon-autograder"):
        key = os.environ.get(OPENROUTER_KEY_ENV)
        if not key:
            raise BackendError(
                f"OpenRouter backend requires the {OPENROUTER_KEY_ENV} environment "
                "variable (set it in the shell; it is never stored on disk)"
            )
        if not config.model:
            raise BackendError("OpenRouter backend requires a model slug in configuration")
        # Force the OpenRouter endpoint; point the generic key lookup at a
        # variable that does not exist so the parent never installs an
        # unrelated key (e.g. OPENAI_API_KEY) — we set the header ourselves.
        cfg = BackendConfig(**{**config.__dict__,
                               "backend": "openrouter",
                               "base_url": config.base_url or OPENROUTER_BASE_URL,
                               "api_key_env": "__OPENROUTER_KEY_HANDLED_EXPLICITLY__"})
        self._key = key
        super().__init__(cfg, transport=transport)
        self.config = cfg
        self._client.headers["Authorization"] = f"Bearer {key}"
        self._client.headers["HTTP-Referer"] = "https://github.com/taraskHy/sharon-project"
        self._client.headers["X-Title"] = app_title
        # request-level routing / reasoning knobs live in extra_generation
        # under reserved keys so the generic payload builder does not have
        # to know about them
        eg = dict(cfg.extra_generation)
        self._provider_routing: dict[str, Any] | None = eg.pop("provider", None)
        self._reasoning: dict[str, Any] | None = eg.pop("reasoning", None)
        self.config.extra_generation = eg
        self.last_usage: dict[str, Any] = {}

    # -- payload additions ---------------------------------------------------

    def _build_payload(self, messages, output_model, max_tokens):
        payload = super()._build_payload(messages, output_model, max_tokens)
        if self._provider_routing:
            payload["provider"] = self._provider_routing
        if self._reasoning:
            payload["reasoning"] = self._reasoning
        # ask OpenRouter to include usage/cost accounting in the response
        payload["usage"] = {"include": True}
        return payload

    def _post_chat(self, payload: dict) -> dict:
        try:
            data = super()._post_chat(payload)
        except BackendError as e:
            raise BackendError(_redact(str(e), self._key)) from None
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        pdetails = usage.get("prompt_tokens_details") or {}
        self.last_usage = {
            "request_id": data.get("id"),
            "provider": data.get("provider"),
            "model": data.get("model") or self.config.model,
            "input_tokens": usage.get("prompt_tokens"),
            "cached_input_tokens": pdetails.get("cached_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "reported_cost": usage.get("cost"),
        }
        return data

    # -- identity / health -----------------------------------------------------

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d["backend"] = "openrouter"
        d["provider_routing"] = self._provider_routing
        d["reasoning"] = self._reasoning
        return d  # never contains the key

    def health_check(self) -> HealthReport:
        # /models on OpenRouter is public and huge; a cheap authenticated
        # probe is GET /key (returns rate-limit/usage info, no content).
        try:
            resp = self._client.get("/key")
        except httpx.HTTPError as e:
            return HealthReport(ok=False, backend="openrouter", model=self.config.model,
                                detail=_redact(f"cannot reach OpenRouter: {e}", self._key))
        if resp.status_code == 401:
            return HealthReport(ok=False, backend="openrouter", model=self.config.model,
                                detail="OpenRouter rejected the API key (401)")
        if resp.status_code != 200:
            return HealthReport(ok=False, backend="openrouter", model=self.config.model,
                                detail=f"GET /key returned HTTP {resp.status_code}")
        return HealthReport(ok=True, backend="openrouter", model=self.config.model,
                            detail="OpenRouter reachable; key accepted")
