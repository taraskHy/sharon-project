"""End-to-end proof that the resolved route ``max_tokens`` reaches the wire.

Stage-1b recorded ``max_tokens: 1000`` in its config hash and its dry-run cost
prediction, while the provider was sent 600 — the ``OcrPrimaryAdapter``
default. Three of eight observations were lost to truncation at the value
nobody had configured. ``tests/test_bench_max_tokens_resolution.py`` pins the
resolution chain and the call site; this module goes one layer further and
asserts on the **serialized HTTP body**, because a plan artifact that says 1000
while the request says 600 is exactly the failure mode that shipped.

Scope, stated precisely after an adversarial review corrected an earlier
overclaim in this docstring: most tests here drive the real *adapter -> route ->
OpenRouterBackend -> wire* path against an ``httpx.MockTransport``, and pin the
runner's call-site expression by source inspection.
``test_gateway_forwards_the_route_cap_to_the_wire`` additionally goes through a
real ``ModelGateway``. ``run_benchmark`` itself is not executed by any test in
this file. No network, no provider spend.
"""
from __future__ import annotations

import hashlib
import json

import httpx
import pytest

from autograder.backends import BackendConfig
from autograder.backends.openrouter import OpenRouterBackend
from autograder.benchmark.registry import load_registry
from autograder.benchmark.roles import adapter_for
from autograder.benchmark.runner import RunSpec, build_route

REGISTRY = "evaluation/model_selection/candidates.toml"
GEMINI = "google/gemini-3.7-flash"
SONNET = "anthropic/claude-sonnet-5"
EXPECTED_MAX_TOKENS = 1000


@pytest.fixture(scope="module")
def registry():
    return load_registry(REGISTRY)


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    """OpenRouterBackend refuses to construct without a key; this never leaves
    the process and no request is ever sent anywhere."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-not-real")


def _spec(**kw):
    kw.setdefault("candidate", GEMINI)
    return RunSpec(role="ocr_primary", split="dev", backend="openrouter",
                   registry_path=REGISTRY, **kw)


def _route(registry, candidate=GEMINI, **kw):
    ad = adapter_for("ocr_primary")
    return build_route(_spec(candidate=candidate, **kw), candidate,
                       ad.prompt_version, ad.default_max_tokens, registry=registry)


def _capture(route, *, max_tokens):
    """Drive the real backend and return the serialized request body."""
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"transcription": "x"}'},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
                      "cost": 0.0},
        })

    backend = OpenRouterBackend(route.to_backend_config(),
                               transport=httpx.MockTransport(handler))
    from autograder.benchmark.roles import BenchTranscription
    backend.parse(system="s",
                  content_blocks=[{"type": "image", "source": {
                      "type": "base64", "media_type": "image/png", "data": "x"}}],
                  output_model=BenchTranscription, max_tokens=max_tokens)
    return seen


# 1 -- candidate route max_tokens overrides the adapter default --------------
def test_candidate_override_beats_adapter_default(registry):
    ad = adapter_for("ocr_primary")
    route = _route(registry)
    assert route.max_tokens == EXPECTED_MAX_TOKENS
    assert ad.default_max_tokens == 600, "the defaults must differ or this proves nothing"
    assert route.max_tokens != ad.default_max_tokens


# 2 -- the SERIALIZED OpenRouter request carries exactly 1000 ---------------
def test_serialized_openrouter_payload_receives_exactly_1000(registry):
    route = _route(registry)
    seen = _capture(route, max_tokens=route.max_tokens)
    assert seen["body"]["max_tokens"] == EXPECTED_MAX_TOKENS, seen["body"]
    assert seen["body"]["model"] == GEMINI
    assert seen["body"]["reasoning"] == {"effort": "low"}
    assert "openrouter.ai" in seen["url"]


def test_the_stage1b_bug_would_be_caught_here(registry):
    """Passing the adapter default (what Stage-1b actually did) must NOT
    serialize as 1000 — the assertion above has to be able to fail."""
    route = _route(registry)
    ad = adapter_for("ocr_primary")
    seen = _capture(route, max_tokens=ad.default_max_tokens)
    assert seen["body"]["max_tokens"] == 600
    assert seen["body"]["max_tokens"] != EXPECTED_MAX_TOKENS


def test_runner_call_site_passes_the_route_value(registry):
    """The exact expression the runner uses, serialized end to end."""
    import inspect

    from autograder.benchmark import runner as runner_mod
    src = inspect.getsource(runner_mod.run_benchmark)
    assert "max_tokens=route.max_tokens" in src
    assert "max_tokens=request.max_tokens" not in src
    route = _route(registry)
    assert _capture(route, max_tokens=route.max_tokens)["body"]["max_tokens"] == EXPECTED_MAX_TOKENS


# 3 -- run.json / plan.json record the same effective value ------------------
def test_committed_stage1b_run_json_recorded_1000_while_the_wire_had_600():
    """The historical evidence, kept as a regression witness."""
    from pathlib import Path
    d = Path("evaluation/model_selection/runs/ocr_primary/"
             "dev__smoke__all__google-gemini-3.7-flash__45297cdd83")
    if not d.exists():
        pytest.skip("stage-1b run not present on this machine")
    cfg = json.loads((d / "run.json").read_text(encoding="utf-8"))["config"]
    assert cfg["route"]["max_tokens"] == EXPECTED_MAX_TOKENS
    errors = [json.loads(l).get("error") or ""
              for l in (d / "outputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert any("max_tokens=600" in e for e in errors), \
        "stage-1b's provider errors are the proof the wire value disagreed with run.json"


# 4 -- the cost prediction uses the same value -------------------------------
def test_cost_prediction_uses_the_route_value(registry):
    from autograder.usage import predicted_call_cost
    route = _route(registry)
    pricing = {GEMINI: {"input": 0.75, "output": 3.75}}
    blocks = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "x"}}]
    at_1000 = predicted_call_cost(route, "s", blocks, pricing)
    cheaper = build_route(_spec(max_tokens=600), GEMINI, "m2-strict-v1", 600, registry=registry)
    at_600 = predicted_call_cost(cheaper, "s", blocks, pricing)
    assert at_1000 > at_600, "a bigger output cap must predict a bigger worst-case cost"
    assert round(at_1000 - at_600, 8) == round(400 * 3.75 / 1e6, 8)


# 5 -- a route without an override keeps its declared default ----------------
def test_a_candidate_without_an_override_is_unaffected(registry):
    route = _route(registry, candidate=SONNET)
    assert route.reasoning != {"effort": "low"}
    seen = _capture(route, max_tokens=route.max_tokens)
    assert seen["body"]["max_tokens"] == route.max_tokens
    assert seen["body"]["max_tokens"] != EXPECTED_MAX_TOKENS, \
        "sonnet must not inherit gemini's declared asymmetry"


# 6 -- prompt text and schema bytes are unchanged ----------------------------
def test_prompt_and_schema_bytes_are_untouched_by_the_fix():
    from pathlib import Path

    from autograder.benchmark.roles import BenchTranscription, _load_historical_prompts
    contract = json.loads(Path(
        "evaluation/model_selection/runs/ocr_primary/"
        "OCR_SMOKE_STAGE1_CONTRACT_EXEC_2026-09-02.json").read_text(encoding="utf-8"))
    now = {k: hashlib.sha256(v.encode()).hexdigest()
           for k, v in sorted(_load_historical_prompts().items())}
    assert now == contract["prompt_sha256_by_category"]
    base = json.loads(Path(
        "evaluation/model_selection/runs/ocr_primary/"
        "OCR_SMOKE_STAGE1_BASELINE_2026-09-02.json").read_text(encoding="utf-8"))
    schema = hashlib.sha256(json.dumps(BenchTranscription.model_json_schema(),
                                       sort_keys=True).encode()).hexdigest()
    assert schema == base["schema_sha256"]


def test_the_serialized_payload_carries_one_image_and_only_the_schema_text(registry):
    """Contract check on the WIRE, not on a plan artifact.

    The Stage-1 contract artifact recorded "zero text blocks", which was true of
    the adapter's ``Request.content_blocks``. The serialized request is not the
    same object: ``OpenAICompatBackend._build_payload`` appends one structured-
    output instruction block carrying the BenchTranscription JSON Schema. That
    is the "minimal structured transcription schema" the campaign allows, and it
    is generated from the output model — it cannot carry case data — but it is
    invisible to ``check_cloud_call``, which scans the adapter-level blocks. So
    it is asserted here, on the bytes that actually leave.
    """
    route = _route(registry)
    seen = _capture(route, max_tokens=route.max_tokens)
    content = seen["body"]["messages"][-1]["content"]
    kinds = [c.get("type") for c in content]
    assert kinds.count("image_url") == 1, kinds
    texts = [c["text"] for c in content if c.get("type") == "text"]
    assert len(texts) == 1, "exactly one text block: the schema instruction"
    only = texts[0]
    assert only.startswith("Respond with ONLY a single JSON object")
    assert "BenchTranscription" in only and '"transcription"' in only
    # nothing case-specific and nothing grading-shaped may ride along
    for banned in ("rubric", "official", "solution", "score", "grade", "verdict",
                   "accepted answer", "instructor", "policy"):
        assert banned not in only.lower(), f"schema text carries {banned!r}"


# ---------------------------------------------------------------------------
# Gaps closed after adversarial review (2026-09-02). The tests above drive the
# adapter -> backend -> wire path directly; an independent verifier pointed out
# that neither the runner nor the ModelGateway was ever executed, so "end to
# end" was overclaimed, and that `extra_generation` is merged into the payload
# AFTER max_tokens and could silently replace it.
# ---------------------------------------------------------------------------

def test_gateway_forwards_the_route_cap_to_the_wire(tmp_path, registry):
    """Through the REAL ModelGateway, not just the backend."""
    from autograder.benchmark.roles import BenchTranscription
    from autograder.gateway import ModelGateway

    route = _route(registry)
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"transcription": "x"}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0}})

    gw = ModelGateway(
        {route.task: route},
        backend_factory=lambda cfg: OpenRouterBackend(cfg, transport=httpx.MockTransport(handler)),
        execution_mode="research",
        research_auth=__import__("autograder.cloudboundary", fromlist=["x"]).research_authorization(
            "test:e2e", tasks=[route.task], models=[route.model]))
    gw.call(task=route.task,
            system=__import__("autograder.benchmark.roles", fromlist=["x"])
            ._load_historical_prompts()["handwritten_cell"],
            content_blocks=[{"type": "image", "source": {"type": "base64",
                                                         "media_type": "image/png", "data": "x"}}],
            output_model=BenchTranscription, max_tokens=route.max_tokens,
            meta={"job_id": "t"})
    assert seen["body"]["max_tokens"] == EXPECTED_MAX_TOKENS
    assert seen["body"]["reasoning"] == {"effort": "low"}


def test_extra_generation_cannot_silently_replace_the_resolved_cap(registry):
    """`payload.update(extra_generation)` runs after max_tokens is set. A config
    that names max_tokens there must fail closed, not quietly win."""
    import dataclasses

    from autograder.backends import BackendError
    from autograder.benchmark.roles import BenchTranscription

    route = _route(registry)
    cfg = dataclasses.replace(route.to_backend_config(),
                              extra_generation={"max_tokens": 42})
    backend = OpenRouterBackend(cfg, transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={})))
    with pytest.raises(BackendError, match="extra_generation may not override"):
        backend.parse(system="s", content_blocks=[{"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": "x"}}],
            output_model=BenchTranscription, max_tokens=EXPECTED_MAX_TOKENS)


def test_benign_extra_generation_still_works(registry):
    """The guard must not break the keys routes legitimately set."""
    import dataclasses

    from autograder.benchmark.roles import BenchTranscription

    route = _route(registry)
    cfg = dataclasses.replace(route.to_backend_config(),
                              extra_generation={"top_p": 0.9})
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"transcription": "x"}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2, "cost": 0.0}})

    OpenRouterBackend(cfg, transport=httpx.MockTransport(handler)).parse(
        system="s", content_blocks=[{"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": "x"}}],
        output_model=BenchTranscription, max_tokens=EXPECTED_MAX_TOKENS)
    assert seen["body"]["max_tokens"] == EXPECTED_MAX_TOKENS
    assert seen["body"]["top_p"] == 0.9


def test_length_is_checked_before_content_filter():
    """Mechanism behind a Stage-1c finding: a completion carries exactly one
    finish_reason, and the backend tests `length` first. A response truncated by
    the cap can therefore never surface as content_filter — so raising the cap
    can UNMASK content_filter on a case that previously reported truncation."""
    import inspect

    from autograder.backends import openai_compat
    src = inspect.getsource(openai_compat.OpenAICompatBackend.parse)
    assert src.index('finish == "length"') < src.index('finish == "content_filter"')
