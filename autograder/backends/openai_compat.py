"""Backend for any OpenAI-compatible chat-completions server.

One implementation covers every deployment target in scope:

- **Ollama** (local Windows/Linux)          base_url=http://localhost:11434/v1
- **vLLM** (university GPU server)          base_url=http://server:8000/v1
- **Hugging Face TGI**                      base_url=http://server:8080/v1
- **llama.cpp server / LM Studio**          base_url=http://localhost:.../v1
- **Free hosted APIs for open models**      OpenRouter, Groq, Mistral, ...

Structured output is requested according to ``config.structured_mode``:

- ``json_schema``  — ``response_format={"type": "json_schema", ...}``
  (constrained decoding on vLLM, Ollama, TGI, OpenRouter, ...)
- ``json_object``  — ``response_format={"type": "json_object"}`` plus the
  schema embedded in the prompt (servers without full schema support)
- ``prompt``       — no response_format; schema embedded in the prompt only

Whatever the server enforces, the reply is ALWAYS validated locally against
the Pydantic schema; malformed output triggers bounded repair retries and
then a hard ``BackendError``. There is no silent fallback to another model.
"""

from __future__ import annotations

import json
import time
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from .base import (
    BackendConfig,
    BackendError,
    HealthReport,
    VisionBackend,
    extract_json_object,
)

T = TypeVar("T", bound=BaseModel)

_RETRYABLE_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}


def _to_openai_blocks(blocks: list[dict]) -> list[dict]:
    out = []
    for b in blocks:
        if b.get("type") == "text":
            out.append({"type": "text", "text": b["text"]})
        elif b.get("type") == "image":
            src = b["source"]
            if src.get("type") != "base64":
                raise BackendError(f"unsupported image source type: {src.get('type')}")
            out.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{src['media_type']};base64,{src['data']}"},
                }
            )
        else:
            raise BackendError(f"unsupported content block type: {b.get('type')}")
    return out


class OpenAICompatBackend(VisionBackend):
    def __init__(self, config: BackendConfig, transport: httpx.BaseTransport | None = None):
        if not config.base_url:
            raise BackendError(
                "backend 'openai' requires --base-url (e.g. http://localhost:11434/v1 "
                "for Ollama, http://server:8000/v1 for vLLM)"
            )
        if not config.model:
            raise BackendError("backend 'openai' requires --model")
        self.config = config
        headers = {"Content-Type": "application/json"}
        key = config.api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self._client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(config.timeout_s, connect=15.0),
            transport=transport,  # tests inject httpx.MockTransport here
        )

    # -- request plumbing ---------------------------------------------------

    def _post_chat(self, payload: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(self.config.transport_retries + 1):
            try:
                resp = self._client.post("/chat/completions", json=payload)
            except httpx.TimeoutException as e:
                raise BackendError(
                    f"request to {self.config.base_url} timed out after "
                    f"{self.config.timeout_s:.0f}s — a local CPU server may simply be "
                    "slow; raise --timeout, or check that the model is loaded"
                ) from e
            except httpx.TransportError as e:
                last_error = e
                time.sleep(min(2**attempt, 10))
                continue
            if resp.status_code in _RETRYABLE_STATUS:
                retry_after = resp.headers.get("retry-after")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(2**attempt * 2, 30)
                last_error = BackendError(
                    f"HTTP {resp.status_code} from backend: {resp.text[:300]}"
                )
                time.sleep(delay)
                continue
            if resp.status_code != 200:
                raise BackendError(
                    f"HTTP {resp.status_code} from backend: {resp.text[:500]}"
                )
            try:
                return resp.json()
            except json.JSONDecodeError as e:
                raise BackendError(
                    f"backend returned a non-JSON HTTP body: {resp.text[:300]}"
                ) from e
        raise BackendError(
            f"backend unreachable after {self.config.transport_retries + 1} attempts: {last_error}"
        )

    def _build_payload(
        self, messages: list[dict], output_model: type[BaseModel], max_tokens: int
    ) -> dict:
        payload: dict = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        payload.update(self.config.extra_generation)
        if self.config.structured_mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": output_model.__name__,
                    "schema": output_model.model_json_schema(),
                },
            }
        elif self.config.structured_mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        elif self.config.structured_mode != "prompt":
            raise BackendError(
                f"unknown structured_mode {self.config.structured_mode!r} "
                "(expected json_schema | json_object | prompt)"
            )
        return payload

    # -- public API ----------------------------------------------------------

    def parse(
        self,
        *,
        system: str,
        content_blocks: list[dict],
        output_model: type[T],
        max_tokens: int | None = None,
    ) -> T:
        max_tokens = max_tokens or self.config.max_tokens
        schema_note = (
            "Respond with ONLY a single JSON object (no prose, no markdown fences) "
            "that conforms exactly to this JSON Schema:\n"
            + json.dumps(output_model.model_json_schema(), ensure_ascii=False)
        )
        user_content = _to_openai_blocks(content_blocks) + [
            {"type": "text", "text": schema_note}
        ]
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

        last_validation_error = ""
        for attempt in range(self.config.validation_retries + 1):
            data = self._post_chat(self._build_payload(messages, output_model, max_tokens))
            try:
                choice = data["choices"][0]
                raw = choice["message"]["content"] or ""
                finish = choice.get("finish_reason")
            except (KeyError, IndexError, TypeError) as e:
                raise BackendError(
                    f"unexpected chat-completions response shape: {str(data)[:300]}"
                ) from e
            if finish == "length":
                raise BackendError(
                    f"output was truncated at max_tokens={max_tokens} "
                    "(finish_reason=length); raise --max-tokens"
                )
            if finish == "content_filter":
                raise BackendError("the backend refused this request (content_filter)")
            try:
                return output_model.model_validate_json(extract_json_object(raw))
            except ValidationError as e:
                last_validation_error = (
                    f"{e.error_count()} error(s), first: {e.errors()[0].get('msg', '?')} "
                    f"at {'.'.join(str(x) for x in e.errors()[0].get('loc', ()))}"
                )
            except json.JSONDecodeError as e:  # pragma: no cover - validate_json raises ValidationError
                last_validation_error = str(e)
            # Repair round-trip: show the model its own output and the error.
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your previous output failed schema validation: "
                        f"{last_validation_error}. Return ONLY the corrected JSON object, "
                        "nothing else."
                    ),
                }
            )
        raise BackendError(
            f"model output failed {output_model.__name__} validation after "
            f"{self.config.validation_retries + 1} attempt(s): {last_validation_error}"
        )

    def health_check(self) -> HealthReport:
        try:
            resp = self._client.get("/models")
        except httpx.HTTPError as e:
            return HealthReport(
                ok=False,
                backend=self.config.backend,
                model=self.config.model,
                detail=f"cannot reach {self.config.base_url}: {e}",
            )
        if resp.status_code != 200:
            return HealthReport(
                ok=False,
                backend=self.config.backend,
                model=self.config.model,
                detail=f"GET /models returned HTTP {resp.status_code}: {resp.text[:200]}",
            )
        try:
            ids = [m.get("id", "") for m in resp.json().get("data", [])]
        except (json.JSONDecodeError, AttributeError):
            ids = []
        if self.config.model in ids:
            return HealthReport(
                ok=True,
                backend=self.config.backend,
                model=self.config.model,
                detail=f"server reachable; model available ({len(ids)} models listed)",
            )
        return HealthReport(
            ok=bool(ids),
            backend=self.config.backend,
            model=self.config.model,
            detail=(
                f"server reachable but model {self.config.model!r} not in its list "
                f"({', '.join(ids[:10]) or 'empty'}) — check the model name/tag"
            ),
        )
