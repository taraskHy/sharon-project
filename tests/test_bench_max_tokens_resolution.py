"""A benchmark run must SEND the decoding configuration it RECORDS.

Discovered by the OCR Stage-1b arm (2026-09-02). The run's config hash recorded
``max_tokens: 1000`` — the value derived for that experiment and declared as a
candidate override precisely so that Gemini's mandatory reasoning could not
truncate the transcription — while the provider was actually sent 600, the
``OcrPrimaryAdapter`` default. Three of the eight cases died with
"output was truncated at max_tokens=600", i.e. to the exact failure the
configured cap existed to prevent, and the dry-run cost prediction (which does
read the route) had priced a request that was never made.

``build_route`` seeds the chain with the adapter's own default and then applies
models.toml, a declared candidate override and an explicit ``--max-tokens`` in
that order, so ``route.max_tokens`` is the resolved value. The adapter's
``Request`` only ever carries the unresolved default.
"""
from __future__ import annotations

import inspect

import pytest

from autograder.benchmark.registry import load_registry
from autograder.benchmark.roles import adapter_for
from autograder.benchmark.runner import RunSpec, build_route

REGISTRY = "evaluation/model_selection/candidates.toml"
GEMINI = "google/gemini-3.7-flash"


def _spec(**kw):
    kw.setdefault("candidate", GEMINI)
    return RunSpec(role="ocr_primary", split="dev",
                   backend="openrouter", registry_path=REGISTRY, **kw)


@pytest.fixture(scope="module")
def registry():
    return load_registry(REGISTRY)


def test_the_runner_sends_the_route_max_tokens_not_the_adapter_default():
    """The regression itself, pinned at the call site."""
    src = inspect.getsource(__import__("autograder.benchmark.runner",
                                       fromlist=["x"]).run_benchmark)
    assert "max_tokens=route.max_tokens" in src
    assert "max_tokens=request.max_tokens" not in src, \
        "the adapter default must not override the resolved route configuration"


def test_declared_candidate_override_reaches_the_route(registry):
    """The Stage-1b configuration, through the real precedence chain."""
    ad = adapter_for("ocr_primary")
    route = build_route(_spec(), GEMINI, ad.prompt_version, ad.default_max_tokens,
                        registry=registry)
    assert route.max_tokens == 1000, "the candidate override must win over the adapter default"
    assert route.reasoning == {"effort": "low"}
    assert route.max_tokens != ad.default_max_tokens, \
        "this test is meaningless if the override happens to equal the default"


def test_explicit_spec_max_tokens_beats_the_candidate_override(registry):
    ad = adapter_for("ocr_primary")
    route = build_route(_spec(max_tokens=321), GEMINI, ad.prompt_version,
                        ad.default_max_tokens, registry=registry)
    assert route.max_tokens == 321


def test_a_candidate_with_no_override_keeps_the_resolved_default(registry):
    """Sonnet has no ocr_primary override, so it must be unaffected by Gemini's."""
    ad = adapter_for("ocr_primary")
    route = build_route(_spec(candidate="anthropic/claude-sonnet-5"),
                        "anthropic/claude-sonnet-5", ad.prompt_version,
                        ad.default_max_tokens, registry=registry)
    assert route.reasoning != {"effort": "low"}
    assert route.max_tokens is not None


def test_route_max_tokens_is_in_the_fingerprint(registry):
    """A run must not be able to change its cap without changing its identity."""
    ad = adapter_for("ocr_primary")
    route = build_route(_spec(), GEMINI, ad.prompt_version, ad.default_max_tokens,
                        registry=registry)
    fp = route.fingerprint_fields()
    assert fp["max_tokens"] == 1000
    assert fp["reasoning"] == {"effort": "low"}


def test_backend_config_carries_the_route_cap(registry):
    ad = adapter_for("ocr_primary")
    route = build_route(_spec(), GEMINI, ad.prompt_version, ad.default_max_tokens,
                        registry=registry)
    assert route.to_backend_config().max_tokens == 1000
