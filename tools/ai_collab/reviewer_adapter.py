"""Reviewer adapters: OpenRouter (priority), direct OpenAI, and a mock.

Both live backends speak the OpenAI-compatible /chat/completions shape via
stdlib urllib (no new dependency). The API key is read from the environment
at call time (OPENROUTER_API_KEY / OPENAI_API_KEY), used only in the request
header, and never persisted, printed, or included in exceptions.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .config import ReviewerCfg
from .errors import AdapterError
from .redaction import redact_text
from .util import read_json

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"


@dataclass
class ReviewerUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": self.cost_usd,
        }


@dataclass
class ReviewerCallResult:
    raw_text: str
    usage: ReviewerUsage = field(default_factory=ReviewerUsage)
    model: str = ""
    provider_meta: dict = field(default_factory=dict)


def resolved_model(cfg: ReviewerCfg) -> str:
    from .config import has_unresolved_env

    if not cfg.model or has_unresolved_env(cfg.model):
        raise AdapterError(
            f"reviewer model is not resolved ({cfg.model!r}); set the referenced "
            "environment variable (e.g. AI_REVIEW_MODEL) or configure "
            "[reviewer].model"
        )
    return cfg.model


class HTTPReviewer:
    """OpenAI-compatible chat-completions client (urllib, stdlib only)."""

    url: str = ""
    key_env: str = ""
    extra_body: dict = {}
    extra_headers: dict = {}

    def __init__(self, cfg: ReviewerCfg):
        self.cfg = cfg

    def _api_key(self) -> str:
        key = os.environ.get(self.key_env, "")
        if not key:
            raise AdapterError(
                f"environment variable {self.key_env} is not set; the reviewer "
                "backend cannot authenticate (the key is read from the "
                "environment only, never from config)"
            )
        return key

    def call(self, system: str, user: str) -> ReviewerCallResult:
        model = resolved_model(self.cfg)
        body: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_output_tokens,
        }
        if self.cfg.force_json:
            body["response_format"] = {"type": "json_object"}
        body.update(self.extra_body)

        url = self.cfg.api_base or self.url
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            method="POST",
        )
        request.add_header("Content-Type", "application/json")
        request.add_header("Authorization", f"Bearer {self._api_key()}")
        for name, value in self.extra_headers.items():
            request.add_header(name, value)

        try:
            with urllib.request.urlopen(
                request, timeout=self.cfg.request_timeout_seconds
            ) as response:
                payload = json.loads(response.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
            except OSError:
                pass
            detail, _ = redact_text(detail)
            raise AdapterError(
                f"reviewer HTTP error {exc.code} from {url}: {detail}"
            ) from None
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AdapterError(f"reviewer request failed: {exc}") from None

        try:
            content = payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            raise AdapterError(
                f"reviewer response missing choices/message: {str(payload)[:300]}"
            ) from None

        usage_raw = payload.get("usage") or {}
        usage = ReviewerUsage(
            input_tokens=usage_raw.get("prompt_tokens"),
            output_tokens=usage_raw.get("completion_tokens"),
            cost_usd=usage_raw.get("cost"),  # OpenRouter reports this with usage.include
        )
        return ReviewerCallResult(
            raw_text=content,
            usage=usage,
            model=payload.get("model", model),
            provider_meta={
                "id": payload.get("id"),
                "provider": payload.get("provider"),
            },
        )


class OpenRouterReviewer(HTTPReviewer):
    url = OPENROUTER_URL
    key_env = "OPENROUTER_API_KEY"
    extra_body = {"usage": {"include": True}}
    extra_headers = {
        "HTTP-Referer": "https://localhost/ai-collab",
        "X-Title": "ai-collab harness",
    }


class OpenAIReviewer(HTTPReviewer):
    url = OPENAI_URL
    key_env = "OPENAI_API_KEY"


class MockReviewer:
    """Deterministic scripted reviewer. Entries: review dicts or raw strings."""

    def __init__(self, scripted: list, start_index: int = 0):
        self.scripted = scripted
        self.index = start_index
        self.requests: list[tuple[str, str]] = []

    @classmethod
    def from_script_file(
        cls, script_path: Path, start_index: int = 0
    ) -> "MockReviewer":
        data = read_json(Path(script_path))
        return cls(list(data.get("reviews", [])), start_index=start_index)

    def call(self, system: str, user: str) -> ReviewerCallResult:
        if self.index >= len(self.scripted):
            raise AdapterError("mock reviewer script exhausted")
        entry = self.scripted[self.index]
        self.index += 1
        self.requests.append((system, user))
        raw = entry if isinstance(entry, str) else json.dumps(entry, ensure_ascii=False)
        return ReviewerCallResult(
            raw_text=raw,
            usage=ReviewerUsage(input_tokens=None, output_tokens=None, cost_usd=0.0),
            model="mock",
        )


def make_reviewer(cfg: ReviewerCfg, start_index: int = 0):
    if cfg.backend == "openrouter":
        return OpenRouterReviewer(cfg)
    if cfg.backend == "openai":
        return OpenAIReviewer(cfg)
    if cfg.backend == "mock":
        if not cfg.mock_script:
            raise AdapterError(
                "[reviewer].backend='mock' requires [reviewer].mock_script "
                "(JSON file with a 'reviews' list)"
            )
        return MockReviewer.from_script_file(
            Path(cfg.mock_script), start_index=start_index
        )
    raise AdapterError(f"unknown reviewer backend: {cfg.backend}")
