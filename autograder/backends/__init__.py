"""Backend factory. The pipeline depends only on this package's interface."""

from __future__ import annotations

from pathlib import Path

from .base import BackendConfig, BackendError, HealthReport, LLMError, VisionBackend

__all__ = [
    "BackendConfig",
    "BackendError",
    "HealthReport",
    "LLMError",
    "VisionBackend",
    "create_backend",
]


def create_backend(config: BackendConfig) -> VisionBackend:
    if config.backend == "openai":
        from .openai_compat import OpenAICompatBackend

        return OpenAICompatBackend(config)
    if config.backend == "mock":
        from .mock import MockBackend, make_fixture_mock

        # For CLI use, --model may name a fixtures directory.
        if config.model and Path(config.model).is_dir():
            return make_fixture_mock(config.model, config)
        return MockBackend(config=config)
    if config.backend == "anthropic":
        from .anthropic_backend import AnthropicBackend

        return AnthropicBackend(config)
    if config.backend == "openrouter":
        from .openrouter import OpenRouterBackend

        return OpenRouterBackend(config)
    if config.backend == "ollama_native":
        from .ollama_native import OllamaNativeBackend

        return OllamaNativeBackend(config)
    raise BackendError(
        f"unknown backend {config.backend!r} (expected: openai | mock | anthropic | openrouter | ollama_native)"
    )
