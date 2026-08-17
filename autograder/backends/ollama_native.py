"""Ollama native /api/chat backend (VisionBackend contract).

Why it exists: Ollama's OpenAI-compatible endpoint IGNORES the `think`
option on thinking-capable models (verified 2026-08-16 on qwen3.8: 500
reasoning tokens, empty content), while /api/chat honors `think` AND
per-request `num_ctx`. Any local thinking-capable model used as an
auxiliary resolver must go through this transport.

Structured output: the Pydantic JSON schema is passed as /api/chat
`format`; the reply is still validated locally with bounded repair
retries; BackendError on failure (never a silent fallback).
"""

from __future__ import annotations

import base64
import json
import time
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .base import BackendConfig, BackendError, HealthReport, VisionBackend, extract_json_object

T = TypeVar("T", bound=BaseModel)


class OllamaNativeBackend(VisionBackend):
    def __init__(self, config: BackendConfig, transport: httpx.BaseTransport | None = None):
        if not config.model:
            raise BackendError("backend 'ollama_native' requires a model tag")
        base = (config.base_url or "http://localhost:11434").rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3]
        self.config = config
        self._client = httpx.Client(base_url=base, timeout=httpx.Timeout(config.timeout_s, connect=15.0),
                                    transport=transport)
        eg = dict(config.extra_generation)
        self._think = bool(eg.pop("think", False))
        self._options = dict(eg.pop("options", {}) or {})
        for k in ("num_ctx", "repeat_penalty", "top_p", "top_k", "num_predict"):
            if k in eg:
                self._options[k] = eg.pop(k)
        self.last_usage: dict[str, Any] = {}

    def _messages(self, system: str, content_blocks: list[dict]) -> list[dict]:
        texts, images = [], []
        for b in content_blocks:
            if b.get("type") == "text":
                texts.append(b["text"])
            elif b.get("type") == "image":
                images.append(b["source"]["data"])
            else:
                raise BackendError(f"unsupported content block type: {b.get('type')}")
        user: dict = {"role": "user", "content": "\n".join(texts) or "Reply with the JSON object."}
        if images:
            user["images"] = images
        return [{"role": "system", "content": system}, user]

    def parse(self, *, system: str, content_blocks: list[dict], output_model: type[T],
              max_tokens: int | None = None) -> T:
        max_tokens = max_tokens or self.config.max_tokens
        messages = self._messages(system, content_blocks)
        opts = {**self._options, "num_predict": max_tokens}
        if self.config.temperature is not None:
            opts["temperature"] = self.config.temperature
        last_err = ""
        for attempt in range(self.config.validation_retries + 1):
            body = {"model": self.config.model, "stream": False, "think": self._think,
                    "options": opts, "messages": messages, "format": output_model.model_json_schema()}
            data = self._post(body)
            msg = data.get("message") or {}
            raw = msg.get("content") or ""
            self.last_usage = {"input_tokens": data.get("prompt_eval_count"),
                               "output_tokens": data.get("eval_count"),
                               "total_tokens": (data.get("prompt_eval_count") or 0) + (data.get("eval_count") or 0),
                               "thinking_chars": len(msg.get("thinking") or ""),
                               "done_reason": data.get("done_reason"), "provider": "ollama",
                               "model": data.get("model") or self.config.model}
            if data.get("done_reason") == "length" and not raw.strip():
                raise BackendError(f"output truncated at num_predict={max_tokens} with empty content")
            try:
                return output_model.model_validate_json(extract_json_object(raw))
            except ValidationError as e:
                last_err = f"{e.error_count()} error(s), first: {e.errors()[0].get('msg', '?')}"
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Your previous output failed schema validation: {last_err}. "
                                                        "Return ONLY the corrected JSON object."})
        raise BackendError(f"model output failed {output_model.__name__} validation after "
                           f"{self.config.validation_retries + 1} attempt(s): {last_err}")

    def _post(self, body: dict) -> dict:
        last: Exception | None = None
        for attempt in range(self.config.transport_retries + 1):
            try:
                resp = self._client.post("/api/chat", json=body)
            except httpx.TimeoutException as e:
                raise BackendError(f"Ollama request timed out after {self.config.timeout_s:.0f}s") from e
            except httpx.TransportError as e:
                last = e
                time.sleep(min(2 ** attempt, 10))
                continue
            if resp.status_code in (429, 500, 502, 503, 504):
                last = BackendError(f"HTTP {resp.status_code} from Ollama: {resp.text[:200]}")
                time.sleep(min(2 ** attempt * 2, 30))
                continue
            if resp.status_code != 200:
                raise BackendError(f"HTTP {resp.status_code} from Ollama: {resp.text[:300]}")
            try:
                return resp.json()
            except json.JSONDecodeError as e:
                raise BackendError(f"Ollama returned non-JSON: {resp.text[:200]}") from e
        raise BackendError(f"Ollama unreachable after {self.config.transport_retries + 1} attempts: {last}")

    def health_check(self) -> HealthReport:
        try:
            resp = self._client.get("/api/tags")
        except httpx.HTTPError as e:
            return HealthReport(False, "ollama_native", self.config.model, f"cannot reach Ollama: {e}")
        if resp.status_code != 200:
            return HealthReport(False, "ollama_native", self.config.model, f"GET /api/tags -> HTTP {resp.status_code}")
        names = [m.get("name", "") for m in resp.json().get("models", [])]
        ok = self.config.model in names
        return HealthReport(ok, "ollama_native", self.config.model,
                            "model available" if ok else f"model not pulled (have {len(names)})")

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update({"backend": "ollama_native", "think": self._think, "options": dict(sorted(self._options.items()))})
        return d
