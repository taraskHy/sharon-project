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


# ------------------------------------------- declared candidate asymmetry ----
#
# Providers do not offer identical inference controls. google/gemini-3.7-flash
# publishes reasoning.mandatory=true and rejects the role's
# reasoning={"effort":"none"} with a pre-inference HTTP 400. Benchmarking it in
# a state it cannot run in measures nothing; dropping it discards a deployable
# model. The asymmetry is DECLARED in candidates.toml so it reaches the run
# fingerprint and the cost prediction.


def _real_registry():
    from autograder.benchmark.registry import load_registry

    return load_registry()


def test_gemini_gets_its_lowest_supported_effort_not_none(models_toml):
    reg = _real_registry()
    r = build_route(RunSpec(role="grade_primary", split="dev", models_config=models_toml),
                    "google/gemini-3.7-flash", "grade-v3", 600, registry=reg)
    assert r.reasoning == {"effort": "low"}


def test_the_other_candidates_keep_reasoning_disabled(models_toml):
    reg = _real_registry()
    for slug in ("openai/gpt-5.6-luna-pro", "anthropic/claude-sonnet-5"):
        r = build_route(RunSpec(role="grade_primary", split="dev", models_config=models_toml),
                        slug, "grade-v3", 600, registry=reg)
        assert r.reasoning == {"effort": "none"}, slug
        assert r.max_tokens == 600, slug


def test_geminis_token_cap_is_raised_so_reasoning_cannot_truncate_the_answer(models_toml):
    reg = _real_registry()
    r = build_route(RunSpec(role="grade_primary", split="dev", models_config=models_toml),
                    "google/gemini-3.7-flash", "grade-v3", 600, registry=reg)
    assert r.max_tokens == 1200
    # measured worst-case reasoning (489) + a full structured answer (250) must fit
    assert r.max_tokens >= 489 + 250


def test_the_override_is_inside_the_run_fingerprint(models_toml):
    """A configuration change must yield a DIFFERENT run identity."""
    reg = _real_registry()
    spec = RunSpec(role="grade_primary", split="dev", models_config=models_toml)
    with_over = build_route(spec, "google/gemini-3.7-flash", "grade-v3", 600,
                            registry=reg).fingerprint_fields()
    without = build_route(spec, "google/gemini-3.7-flash", "grade-v3", 600,
                          registry=None).fingerprint_fields()
    assert with_over != without
    assert with_over["reasoning"] == {"effort": "low"} and with_over["max_tokens"] == 1200


def test_an_explicit_spec_value_still_beats_a_candidate_override(models_toml):
    reg = _real_registry()
    r = build_route(RunSpec(role="grade_primary", split="dev", models_config=models_toml,
                            max_tokens=4096),
                    "google/gemini-3.7-flash", "grade-v3", 600, registry=reg)
    assert r.max_tokens == 4096


def test_an_override_may_not_smuggle_in_arbitrary_route_fields(models_toml):
    """Only decoding knobs are overridable — never the model, backend or the
    prompt version."""
    from autograder.benchmark.registry import RoleCandidates

    reg = _real_registry()
    reg.roles["grade_primary"] = RoleCandidates(
        role="grade_primary", status="UNSELECTED", gateway_task="grade_primary",
        env_slug=None, candidates=["vendor/m"],
        candidate_overrides={"vendor/m": {"model": "someone/else"}})
    with pytest.raises(ValueError, match="candidate_overrides"):
        build_route(RunSpec(role="grade_primary", split="dev", models_config=models_toml),
                    "vendor/m", "grade-v3", 600, registry=reg)


def test_a_candidate_without_an_override_is_untouched(models_toml):
    reg = _real_registry()
    r = build_route(RunSpec(role="grade_primary", split="dev", models_config=models_toml),
                    "anthropic/claude-sonnet-5", "grade-v3", 600, registry=reg)
    assert (r.reasoning, r.max_tokens) == ({"effort": "none"}, 600)


def test_the_declared_override_records_why():
    reg = _real_registry()
    over = reg.for_role("grade_primary").overrides_for("google/gemini-3.7-flash")
    assert over.get("why"), "a candidate asymmetry must carry its justification"
    assert "mandatory" in over["why"].lower()
