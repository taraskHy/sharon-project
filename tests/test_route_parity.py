"""Benchmark/production route parity — offline.

A benchmark that decodes differently from production measures a model nobody
will ever run.

On 2026-08-24 ``build_route`` constructed its own knobs from scratch and
passed ``reasoning=spec.reasoning``, which the CLI never sets. The production
route ``[models.grade_primary] reasoning = {effort = "none"}`` therefore never
reached a single benchmark request. Consequences measured in that run:

- gemini-3.7-flash burned 1106 of 1262 output tokens (88%) on reasoning;
- claude-sonnet-5 lost case e003_q2_r6 entirely to ``finish_reason=length``,
  because reasoning tokens count against the same 600-token cap.

These tests pin every decoding parameter that must cross from models.toml
into a benchmark run. No provider is contacted.
"""

from __future__ import annotations

import textwrap

import pytest

from autograder.benchmark.runner import (
    ROUTE_PARITY_FIELDS,
    RunSpec,
    build_route,
    production_route_defaults,
)

MODELS_TOML = """\
[defaults]
timeout_s = 300.0

[models.grade_primary]
backend = "openrouter"
model = "UNSELECTED"
max_tokens = 600
temperature = 0.0
reasoning = {effort = "none"}
prompt_version = "grade-v2"

[models.ocr_primary]
backend = "openrouter"
model = "UNSELECTED"
max_tokens = 1200
reasoning = {effort = "low"}

[pricing]
"vendor/m" = {input = 1.0, output = 2.0}
"""


@pytest.fixture()
def models_toml(tmp_path):
    p = tmp_path / "models.toml"
    p.write_text(textwrap.dedent(MODELS_TOML), encoding="utf-8")
    return p


def _route(models_toml, role="grade_primary", **spec_kw):
    spec = RunSpec(role=role, split="dev", models_config=models_toml, **spec_kw)
    return build_route(spec, "vendor/m", "grade-v2", 600)


# ----------------------------------------------------------- the root cause ----


def test_configured_reasoning_reaches_the_route(models_toml):
    """The regression that cost the 2026-08-24 smoke run its comparability."""
    assert _route(models_toml).reasoning == {"effort": "none"}


def test_configured_reasoning_reaches_the_provider_payload(models_toml):
    """Parity must survive all the way into the backend configuration —
    a route field nobody forwards is not propagation."""
    cfg = _route(models_toml).to_backend_config()
    assert cfg.extra_generation["reasoning"] == {"effort": "none"}


def test_reasoning_absent_from_config_stays_absent(tmp_path):
    """No models.toml -> no invented reasoning policy."""
    spec = RunSpec(role="grade_primary", split="dev", models_config=None)
    assert build_route(spec, "vendor/m", "grade-v2", 600).reasoning is None


def test_a_different_role_gets_its_own_reasoning(models_toml):
    assert _route(models_toml, role="ocr_primary").reasoning == {"effort": "low"}


# --------------------------------------------------------------- parameters ----


def test_max_tokens_comes_from_production(models_toml):
    assert _route(models_toml).max_tokens == 600


def test_temperature_comes_from_production(models_toml):
    assert _route(models_toml).temperature == 0.0


def test_timeout_inherits_from_defaults_table(models_toml):
    assert _route(models_toml).timeout_s == 300.0


def test_structured_mode_is_json_schema(models_toml):
    assert _route(models_toml).structured_mode == "json_schema"


def test_prompt_version_comes_from_the_request_not_the_config(models_toml):
    """Prompt provenance belongs to the adapter's built request; models.toml
    must never be able to relabel it."""
    assert _route(models_toml).prompt_version == "grade-v2"


def test_response_schema_is_the_strict_one(models_toml):
    """Schema parity: the benchmark sends exactly what production sends."""
    import httpx

    from autograder.backends.openrouter import OpenRouterBackend
    from autograder.escalation import GradeResult
    from autograder.strictschema import schema_violations

    import os

    os.environ.setdefault("OPENROUTER_API_KEY", "sk-or-v1-test-not-a-real-key")
    cfg = _route(models_toml).to_backend_config()
    backend = OpenRouterBackend(
        cfg, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    payload = backend._build_payload([{"role": "user", "content": "x"}], GradeResult, 600)
    assert schema_violations(payload["response_format"]["json_schema"]["schema"]) == []
    assert payload["reasoning"] == {"effort": "none"}
    assert payload["max_tokens"] == 600
    assert payload["temperature"] == 0.0


# ---------------------------------------------------------------- precedence ----


def test_explicit_spec_value_overrides_production(models_toml):
    assert _route(models_toml, max_tokens=4096).max_tokens == 4096
    assert _route(models_toml, reasoning={"effort": "high"}).reasoning == {"effort": "high"}


def test_candidate_slug_is_never_taken_from_models_toml(models_toml):
    """models.toml says UNSELECTED; the benchmark supplies the candidate."""
    assert _route(models_toml).model == "vendor/m"


def test_parity_fields_do_not_include_the_model(models_toml):
    assert "model" not in ROUTE_PARITY_FIELDS
    assert "backend" not in ROUTE_PARITY_FIELDS


# -------------------------------------------------------------- provenance ----


def test_route_parameters_are_inside_the_run_fingerprint(models_toml):
    """A decoding change must produce a DIFFERENT run identity, never a silent
    overwrite of an existing run directory."""
    fp = _route(models_toml).fingerprint_fields()
    assert fp["reasoning"] == {"effort": "none"}
    for field in ("max_tokens", "temperature", "structured_mode", "prompt_version"):
        assert field in fp
    changed = _route(models_toml, reasoning={"effort": "high"}).fingerprint_fields()
    assert changed != fp


def test_production_defaults_reads_only_decoding_knobs(models_toml):
    got = production_route_defaults(models_toml, "grade_primary")
    assert "model" not in got and "backend" not in got
    assert got["reasoning"] == {"effort": "none"}
    assert set(got) <= set(ROUTE_PARITY_FIELDS)


def test_missing_models_config_is_not_an_error(tmp_path):
    assert production_route_defaults(tmp_path / "nope.toml", "grade_primary") == {}
    assert production_route_defaults(None, "grade_primary") == {}
