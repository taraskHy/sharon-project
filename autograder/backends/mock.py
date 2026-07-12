"""Mock backend for offline tests and plumbing checks.

Two modes:

- **Programmatic** (tests): construct with a ``responder`` callable or a
  ``responses`` list consumed in order. Every request is recorded in
  ``self.calls`` so tests can assert on exactly what would reach a model
  (e.g. that no filename or grade leaks into the input).

- **Fixture directory** (CLI ``--backend mock --model <fixtures_dir>``): each
  request for output model ``Foo`` is answered from ``<fixtures_dir>/Foo.json``.
  This lets the full CLI pipeline run end-to-end with no network and no key.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

from .base import BackendConfig, BackendError, HealthReport, VisionBackend

T = TypeVar("T", bound=BaseModel)


@dataclass
class RecordedCall:
    system: str
    content_blocks: list[dict]
    output_model: str
    max_tokens: int

    def all_text(self) -> str:
        """Every piece of text that would reach the model, concatenated."""
        parts = [self.system]
        for b in self.content_blocks:
            if b.get("type") == "text":
                parts.append(b.get("text", ""))
        return "\n".join(parts)


@dataclass
class MockBackend(VisionBackend):
    config: BackendConfig = field(default_factory=lambda: BackendConfig(backend="mock", model="mock"))
    responder: Callable[[type[BaseModel], str, list[dict]], BaseModel] | None = None
    responses: list[BaseModel] = field(default_factory=list)
    fixtures_dir: Path | None = None
    calls: list[RecordedCall] = field(default_factory=list)

    def parse(
        self,
        *,
        system: str,
        content_blocks: list[dict],
        output_model: type[T],
        max_tokens: int | None = None,
    ) -> T:
        self.calls.append(
            RecordedCall(
                system=system,
                content_blocks=content_blocks,
                output_model=output_model.__name__,
                max_tokens=max_tokens or self.config.max_tokens,
            )
        )
        if self.responder is not None:
            result = self.responder(output_model, system, content_blocks)
            if not isinstance(result, output_model):
                raise BackendError(
                    f"mock responder returned {type(result).__name__}, "
                    f"expected {output_model.__name__}"
                )
            return result
        if self.responses:
            result = self.responses.pop(0)
            if not isinstance(result, output_model):
                raise BackendError(
                    f"mock response queue returned {type(result).__name__}, "
                    f"expected {output_model.__name__}"
                )
            return result
        if self.fixtures_dir is not None:
            fixture = self.fixtures_dir / f"{output_model.__name__}.json"
            if not fixture.exists():
                raise BackendError(
                    f"mock backend has no fixture {fixture} for {output_model.__name__}"
                )
            try:
                return output_model.model_validate_json(fixture.read_text(encoding="utf-8"))
            except ValidationError as e:
                raise BackendError(f"fixture {fixture} is invalid: {e}") from e
        raise BackendError(
            f"mock backend has no response configured for {output_model.__name__}"
        )

    def health_check(self) -> HealthReport:
        detail = "mock backend (no inference)"
        if self.fixtures_dir is not None:
            n = len(list(self.fixtures_dir.glob("*.json"))) if self.fixtures_dir.is_dir() else 0
            detail = f"mock backend with {n} fixture(s) in {self.fixtures_dir}"
        return HealthReport(ok=True, backend="mock", model=self.config.model, detail=detail)

    def describe(self) -> dict:
        d = super().describe()
        if self.fixtures_dir is not None:
            d["fixtures_dir"] = str(self.fixtures_dir)
        return d


def make_fixture_mock(fixtures_dir: str | Path, config: BackendConfig | None = None) -> MockBackend:
    cfg = config or BackendConfig(backend="mock", model=f"fixtures:{fixtures_dir}")
    return MockBackend(config=cfg, fixtures_dir=Path(fixtures_dir))


def dump_fixture(obj: BaseModel, fixtures_dir: str | Path) -> Path:
    """Helper for building fixture directories from real or synthetic runs."""
    path = Path(fixtures_dir) / f"{type(obj).__name__}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path
