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
import uuid
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


#: Payload keys the caller has already resolved. ``extra_generation`` is merged
#: last and must not be able to silently replace any of them — most importantly
#: ``max_tokens``, which is budget-gated and hashed into the run identity.
_RESERVED_PAYLOAD_KEYS = frozenset({"model", "messages", "max_tokens", "response_format"})


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
        #: Optional RawResponseArchive. When set, EVERY provider HTTP reply is
        #: archived verbatim (sanitized) before parsing — including an HTTP 200
        #: that carries no usage block and no provider field, which is exactly
        #: the shape whose raw body was unrecoverable in the prompt-v2 arms.
        self.raw_archive = None
        #: Populated per response so callers can enforce a provider pin.
        #: NOTE: this is the LAST attempt's verdict only. Callers enforcing a
        #: pin must read ``attempt_records`` / ``route_violation`` instead — a
        #: violation on attempt 1 followed by a correct provider on attempt 2
        #: would otherwise be erased here.
        self.last_route_check: dict | None = None
        #: One entry per PHYSICAL HTTP attempt of the current parse(), in
        #: order. Never overwritten; a retry appends.
        self.attempt_records: list[dict] = []
        #: STICKY. The first explicit route violation seen in this parse().
        #: A later, correct attempt never clears it.
        self.route_violation: dict | None = None
        #: Called immediately BEFORE each physical send with
        #: (attempt_id, retry_index, payload). Raising aborts before
        #: transmission — this is where per-attempt budget authorization runs.
        self.pre_send_hook = None
        #: The payload of the request in flight, for route provenance.
        self._payload_in_flight: dict | None = None
        #: Correlation labels the runner sets so an archived row can be joined
        #: back to its case without re-parsing the body.
        self.capture_task: str | None = None
        self.capture_case_id: str | None = None
        #: Stable correlation labels supplied by the caller.
        self.capture_campaign_id: str | None = None
        self.capture_arm_id: str | None = None
        self.capture_logical_request_id: str | None = None

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
            attempt_id=getattr(self, "_current_attempt_id", None),
            retry_index=getattr(self, "_current_retry_index", 0),
        )
        self.billing_events.append(ev)
        if has_usage:
            self.last_usage = dict(usage)
        return ev

    def _archive_raw(self, *, resp, data: dict | None, http_status: int | None,
                     error: str | None = None, attempt_id: str | None = None,
                     retry_index: int = 0) -> None:
        """Archive one raw provider response, then record its route verdict.

        FAILS CLOSED. If the archive write fails, this raises ArchiveFailure:
        the response has already arrived and already been billed, and that
        cannot be undone, but accepting a parsed result whose evidence was
        never recorded — and continuing to spend on further attempts while
        blind — is worse than stopping.
        """
        if self.raw_archive is None:
            return
        from ..rawcapture import ArchiveFailure, build_record
        try:
            finish = None
            try:
                finish = (data or {})["choices"][0].get("finish_reason")
            except (KeyError, IndexError, TypeError):
                finish = None
            usage = self._usage_from_response(data or {}) if data else {}
            rec = build_record(
                payload=self._payload_in_flight,
                http_status=http_status,
                raw_text=(resp.text if resp is not None else None),
                headers=(resp.headers if resp is not None else None),
                parsed_body=data,
                parsed_outcome={
                    "finish_reason": finish,
                    "usage_returned": bool(data and (data or {}).get("usage")),
                    "input_tokens": usage.get("input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "reported_cost": usage.get("reported_cost"),
                },
                attempt=self._attempt_no,
                task=self.capture_task,
                case_id=self.capture_case_id,
                error=error,
                campaign_id=self.capture_campaign_id,
                arm_id=self.capture_arm_id,
                logical_request_id=self.capture_logical_request_id,
                attempt_id=attempt_id,
                retry_index=retry_index,
            )
        except Exception as e:  # noqa: BLE001 — building a record must not mask the response
            raise ArchiveFailure(f"could not build the raw-response record: {e}") from e
        try:
            self.raw_archive.append(rec)
        except Exception as e:  # noqa: BLE001 — FAIL CLOSED
            self._note_archive_failure(attempt_id=attempt_id, retry_index=retry_index, exc=e)
            raise ArchiveFailure(
                f"raw response for attempt {attempt_id} could not be archived: {e}. "
                "The response was already billed; stopping rather than accepting a parsed "
                "result whose evidence was not recorded."
            ) from e
        # route state: append per attempt, and make the FIRST violation sticky
        self.last_route_check = rec.route_check
        self.attempt_records.append({
            "attempt_id": attempt_id, "retry_index": retry_index,
            "http_status": http_status, "case_id": self.capture_case_id,
            "arm_id": self.capture_arm_id, "campaign_id": self.capture_campaign_id,
            "logical_request_id": self.capture_logical_request_id,
            "raw_body_sha256": rec.raw_body_sha256,
            **rec.route_check,
        })
        if rec.route_check.get("violation") and self.route_violation is None:
            self.route_violation = dict(self.attempt_records[-1])

    def _note_archive_failure(self, *, attempt_id, retry_index, exc) -> None:
        """Strongest independent audit record still available once the primary
        archive write has failed: a sibling marker file, then a billing event.
        Both are best-effort; the ArchiveFailure is raised regardless."""
        try:
            import json as _json
            import time as _time
            path = getattr(self.raw_archive, "path", None)
            if path is not None:
                marker = path.with_suffix(path.suffix + ".ARCHIVE_FAILURE")
                with marker.open("a", encoding="utf-8", newline="\n") as fh:
                    fh.write(_json.dumps({
                        "ts": _time.strftime("%Y-%m-%d %H:%M:%S"),
                        "attempt_id": attempt_id, "retry_index": retry_index,
                        "case_id": self.capture_case_id, "arm_id": self.capture_arm_id,
                        "campaign_id": self.capture_campaign_id,
                        "logical_request_id": self.capture_logical_request_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "consequence": "response was billed but NOT archived; arm stopped",
                    }, ensure_ascii=False) + "\n")
        except Exception:  # noqa: BLE001 — the marker is best-effort
            pass
        try:
            if self.billing_events:
                ev = self.billing_events[-1]
                ev.error = ((ev.error or "") + " | ARCHIVE_FAILURE").strip(" |")
                ev.parse_ok = False
        except Exception:  # noqa: BLE001
            pass

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
        self._payload_in_flight = payload
        for attempt in range(self.config.transport_retries + 1):
            # ---- PHYSICAL SEND BOUNDARY -------------------------------------
            # Every physical attempt, including a transport retry, gets its own
            # identity and must be authorized here. A retry does NOT inherit
            # the first attempt's authorization: the hook raises to abort
            # BEFORE transmission if the campaign has no headroom left.
            attempt_id = uuid.uuid4().hex
            self._current_attempt_id, self._current_retry_index = attempt_id, attempt
            if self.pre_send_hook is not None:
                self.pre_send_hook(attempt_id=attempt_id, retry_index=attempt, payload=payload)
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
                _d = self._body_json(resp)
                self._note_provider_response(
                    data=_d, http_status=resp.status_code,
                    error=f"retryable HTTP {resp.status_code}")
                self._archive_raw(resp=resp, data=_d, http_status=resp.status_code,
                                  error=f"retryable HTTP {resp.status_code}",
                                  attempt_id=attempt_id, retry_index=attempt)
                time.sleep(delay)
                continue
            if resp.status_code != 200:
                # Recorded before raising: a rejected request that never
                # reached inference carries no usage and is not billable,
                # but "we were refused" must still be auditable.
                _d = self._body_json(resp)
                self._note_provider_response(
                    data=_d, http_status=resp.status_code,
                    error=f"HTTP {resp.status_code}")
                self._archive_raw(resp=resp, data=_d, http_status=resp.status_code,
                                  error=f"HTTP {resp.status_code}",
                                  attempt_id=attempt_id, retry_index=attempt)
                raise BackendError(
                    f"HTTP {resp.status_code} from backend: {resp.text[:500]}"
                )
            try:
                data = resp.json()
            except json.JSONDecodeError as e:
                self._note_provider_response(data=None, http_status=resp.status_code,
                                             error="non-JSON body")
                self._archive_raw(resp=resp, data=None, http_status=resp.status_code,
                                  error="non-JSON body",
                                  attempt_id=attempt_id, retry_index=attempt)
                raise BackendError(
                    f"backend returned a non-JSON HTTP body: {resp.text[:300]}"
                ) from e
            self._note_provider_response(data=data, http_status=resp.status_code)
            # Archived for EVERY 200, including the historical failure shape:
            # 200 + no usage block + no provider field + filtered content.
            self._archive_raw(resp=resp, data=data, http_status=resp.status_code,
                              attempt_id=attempt_id, retry_index=attempt)
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
        # extra_generation is merged LAST, so it could silently overwrite a key
        # the caller already resolved — including max_tokens, the budgeted output
        # cap. That is the same class of defect as the 2026-09-02 route/adapter
        # max_tokens bug (a run recording one cap and sending another), so it
        # fails closed instead: a route that wants a different cap sets it on the
        # route, where it lands in the config hash and the cost prediction.
        clashes = _RESERVED_PAYLOAD_KEYS & set(self.config.extra_generation)
        if clashes:
            raise BackendError(
                f"extra_generation may not override {sorted(clashes)}: these are "
                "resolved from the route and recorded in the run's config hash. "
                "Set them on the route instead.")
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
        self.attempt_records = []
        self.route_violation = None
        self.last_route_check = None

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
