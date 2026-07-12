"""Optional Anthropic backend — DEVELOPMENT/COMPARISON ONLY.

The finished university system must not require an Anthropic key; this backend
exists so results from open models can be compared against a strong reference
during development. The ``anthropic`` package is an optional dependency
(``pip install -e .[anthropic]``) and is imported lazily — the rest of the
application runs without it installed.
"""

from __future__ import annotations

from typing import TypeVar

import pydantic
from pydantic import BaseModel

from .base import BackendConfig, BackendError, HealthReport, VisionBackend

T = TypeVar("T", bound=BaseModel)

_SAFE_NONSTREAMING_TOKENS = 16000


class AnthropicBackend(VisionBackend):
    def __init__(self, config: BackendConfig):
        try:
            import anthropic
        except ImportError as e:
            raise BackendError(
                "the 'anthropic' package is not installed; this backend is optional "
                "and for development only — install with: pip install -e .[anthropic]"
            ) from e
        self._anthropic = anthropic
        self.config = config
        if not config.model:
            config.model = "claude-opus-4-8"
        # Zero-arg client resolves ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN /
        # an `ant auth login` profile.
        self.client = anthropic.Anthropic()

    def parse(
        self,
        *,
        system: str,
        content_blocks: list[dict],
        output_model: type[T],
        max_tokens: int | None = None,
    ) -> T:
        anthropic = self._anthropic
        max_tokens = max_tokens or self.config.max_tokens
        client = self.client
        if max_tokens > _SAFE_NONSTREAMING_TOKENS:
            client = self.client.with_options(timeout=3600.0)
        try:
            response = client.messages.parse(
                model=self.config.model,
                max_tokens=max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": system,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                thinking={"type": "adaptive"},
                messages=[{"role": "user", "content": content_blocks}],
                output_format=output_model,
            )
        except anthropic.APIStatusError as e:
            raise BackendError(f"API error ({e.status_code}): {e.message}") from e
        except anthropic.APIConnectionError as e:
            raise BackendError(f"network error talking to the Anthropic API: {e}") from e
        except pydantic.ValidationError as e:
            raise BackendError(
                "the model's structured output did not fit the expected schema "
                f"({output_model.__name__}). This is most often output truncation "
                f"at max_tokens={max_tokens}; re-run with a higher --max-tokens. "
                f"Validation detail: {e.error_count()} error(s), first: "
                f"{e.errors()[0].get('msg', '?')}"
            ) from e
        except ValueError as e:
            raise BackendError(
                f"the SDK rejected the request for model {self.config.model!r}: {e}. "
                "Lower --max-tokens or use the default model."
            ) from e

        if response.stop_reason == "refusal":
            raise BackendError("the model refused this request (stop_reason=refusal)")
        if response.stop_reason == "max_tokens":
            raise BackendError(
                "output was truncated (stop_reason=max_tokens); "
                "re-run with a higher --max-tokens"
            )
        parsed = response.parsed_output
        if parsed is None:
            raise BackendError("model response could not be parsed into the expected schema")
        return parsed

    def health_check(self) -> HealthReport:
        try:
            self.client.models.retrieve(self.config.model)
        except Exception as e:  # noqa: BLE001 - report, don't crash a health check
            return HealthReport(
                ok=False,
                backend="anthropic",
                model=self.config.model,
                detail=f"cannot verify model: {e}",
            )
        return HealthReport(
            ok=True, backend="anthropic", model=self.config.model, detail="model available"
        )
