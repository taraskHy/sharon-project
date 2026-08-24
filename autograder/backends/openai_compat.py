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

from ..strictschema import strict_json_schema
from .base import (
    BackendConfig,
    BackendError,
    BillingEvent,
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
        #: usage of the most recent provider response (ledger reads this)
        self.last_usage: dict = {}
        #: EVERY provider response of the current parse() call. Appended to
        #: before any parsing runs, so a parse failure cannot erase a charge.
        self.billing_events: list[BillingEvent] = []
        self._attempt_no = 0

    # -- request plumbing ---------------------------------------------------

    # -- billing accounting -------------------------------------------------
    #
    # A provider charge exists the moment the provider runs the model. It is
    # recorded HERE, at the HTTP boundary, before any schema validation can
    # fail — never in the success path of parse().

    def _usage_from_response(self, data: dict) -> dict:
        """Normalize a chat-completions ``usage`` block. Subclasses extend
        this with provider-specific fields (cost, request id, ...)."""
        usage = (data or {}).get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        pdetails = usage.get("prompt_tokens_details") or {}
        return {
            "model": (data or {}).get("model") or self.config.model,
            "input_tokens": usage.get("prompt_tokens"),
            "cached_input_tokens": pdetails.get("cached_tokens"),
            "output_tokens": usage.get("completion_tokens"),
            "reasoning_tokens": details.get("reasoning_tokens"),
            "total_tokens": usage.get("total_tokens"),
            # generic OpenAI-compatible servers do not report a price
            "reported_cost": usage.get("cost"),
        }

    def _note_provider_response(self, *, data: dict | None, http_status: int | None,
                                error: str | None = None) -> BillingEvent:
        """Record one provider response. Called for EVERY HTTP reply — 200 or
        not — so the ledger can tell "refused before inference" (no usage,
        not billable) from "ran and billed us, then failed downstream"."""
        usage = self._usage_from_response(data or {})
        has_usage = any(usage.get(k) for k in
                        ("input_tokens", "output_tokens", "total_tokens", "reported_cost"))
        finish = None
        try:
            finish = (data or {})["choices"][0].get("finish_reason")
        except (KeyError, IndexError, TypeError):
            finish = None
        self._attempt_no += 1
        ev = BillingEvent(
            usage=usage if has_usage else {"model": usage.get("model")},
            http_status=http_status,
            call_attempted=True,
            inference_reached=bool(has_usage) or http_status == 200,
            usage_returned=bool(has_usage),
            finish_reason=finish,
            attempt=self._attempt_no,
            error=error,
        )
        self.billing_events.append(ev)
        if has_usage:
            self.last_usage = dict(usage)
        return ev

    @staticmethod
    def _body_json(resp: httpx.Response) -> dict:
        """Best-effort JSON of a response body; {} when it is not JSON.
        Error bodies sometimes still carry a usage block."""
        try:
            return resp.json() if resp.content else {}
        except ValueError:
            return {}

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
                self._note_provider_response(
                    data=self._body_json(resp), http_status=resp.status_code,
                    error=f"retryable HTTP {resp.status_code}")
                time.sleep(delay)
                continue
            if resp.status_code != 200:
                # Recorded before raising: a rejected request that never
                # reached inference carries no usage and is not billable,
                # but "we were refused" must still be auditable.
                self._note_provider_response(
                    data=self._body_json(resp), http_status=resp.status_code,
                    error=f"HTTP {resp.status_code}")
                raise BackendError(
                    f"HTTP {resp.status_code} from backend: {resp.text[:500]}"
                )
            try:
                data = resp.json()
            except json.JSONDecodeError as e:
                self._note_provider_response(data=None, http_status=resp.status_code,
                                             error="non-JSON body")
                raise BackendError(
                    f"backend returned a non-JSON HTTP body: {resp.text[:300]}"
                ) from e
            self._note_provider_response(data=data, http_status=resp.status_code)
            return data
        raise BackendError(
            f"backend unreachable after {self.config.transport_retries + 1} attempts: {last_error}"
        )

    def schema_for(self, output_model: type[BaseModel]) -> dict:
        """The JSON Schema actually sent to the provider.

        Strict providers (OpenAI/Azure, which is what OpenRouter routed
        ``openai/gpt-5.6-luna-pro`` to) validate this BEFORE inference and
        reject anything whose objects are not closed with
        ``additionalProperties: false``. Applied centrally here so every
        output model and every backend inheriting this transport is covered,
        and so the copy embedded in the prompt is identical to the copy in
        ``response_format``.
        """
        schema = output_model.model_json_schema()
        if not getattr(self.config, "strict_schema", True):
            return schema
        return strict_json_schema(schema)

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
                    "schema": self.schema_for(output_model),
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
            + json.dumps(self.schema_for(output_model), ensure_ascii=False)
        )
        user_content = _to_openai_blocks(content_blocks) + [
            {"type": "text", "text": schema_note}
        ]
        messages: list[dict] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]

        # One parse() == one accounting unit. Every provider response below
        # is appended to self.billing_events; the gateway ledgers ALL of them,
        # including the ones whose bodies we then fail to use.
        self.billing_events = []
        self._attempt_no = 0

        last_validation_error = ""
        for attempt in range(self.config.validation_retries + 1):
            data = self._post_chat(self._build_payload(messages, output_model, max_tokens))
            event = self.billing_events[-1] if self.billing_events else None
            try:
                choice = data["choices"][0]
                raw = choice["message"]["content"] or ""
                finish = choice.get("finish_reason")
            except (KeyError, IndexError, TypeError) as e:
                if event is not None:
                    event.parse_ok = False
                raise BackendError(
                    f"unexpected chat-completions response shape: {str(data)[:300]}"
                ) from e
            if finish == "length":
                # The provider generated (and billed) max_tokens of output.
                # The charge stands; only our use of it failed.
                if event is not None:
                    event.parse_ok = False
                raise BackendError(
                    f"output was truncated at max_tokens={max_tokens} "
                    "(finish_reason=length); raise --max-tokens"
                )
            if finish == "content_filter":
                if event is not None:
                    event.parse_ok = False
                raise BackendError("the backend refused this request (content_filter)")
            try:
                value = output_model.model_validate_json(extract_json_object(raw))
                if event is not None:
                    event.parse_ok = True
                return value
            except ValidationError as e:
                if event is not None:
                    event.parse_ok = False
                last_validation_error = (
                    f"{e.error_count()} error(s), first: {e.errors()[0].get('msg', '?')} "
                    f"at {'.'.join(str(x) for x in e.errors()[0].get('loc', ()))}"
                )
            except json.JSONDecodeError as e:  # pragma: no cover - validate_json raises ValidationError
                if event is not None:
                    event.parse_ok = False
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
