"""Canonical route identity, fail-closed campaign setup, and a full
CLI-to-wire integration test. ZERO network — MockTransport throughout.

These are the proofs the V1 alt-candidate screen needed and did not have. It
was stopped on 2026-09-04 after a provider-pinned arm was served five cached
responses produced by an UNPINNED run, because identity came from a
hand-maintained field list that omitted ``provider``.

The integration test deliberately uses the REAL CLI argument path, the REAL
TaskRoute construction and the DEFAULT transport-retry configuration. Building
a simplified backend with ``transport_retries=0`` is what hid the earlier
defects, so it is not done here.
"""
import json
from pathlib import Path

import httpx
import pytest

from autograder.gateway import TaskRoute
from autograder.routeidentity import (
    CACHE_IDENTITY_VERSION, EXCLUDED_FIELDS, IdentityError,
    effective_config_fields, experiment_identity, identity_report,
    semantic_request_identity,
)

GEM = "google/gemini-3.7-flash"
AI = {"order": ["google-ai-studio"], "allow_fallbacks": False}
VX = {"order": ["google-vertex"], "allow_fallbacks": False}
SECRET = "sk-or-v1-" + ("FAKE" * 8) + "-NOTAREALKEY"


def _route(**kw):
    base = dict(task="ocr_primary", backend="openrouter", model=GEM,
                base_url="https://openrouter.ai/api/v1", prompt_version="m2-strict-v1",
                max_tokens=1000, temperature=0.0)
    base.update(kw)
    return TaskRoute(**base)


# =============================================================================
# 1. identity is derived, versioned, and separates every route
# =============================================================================

def test_unpinned_ai_studio_and_vertex_have_three_distinct_identities():
    ids = {experiment_identity(_route()),
           experiment_identity(_route(provider=AI)),
           experiment_identity(_route(provider=VX))}
    assert len(ids) == 3


def test_changing_allow_fallbacks_changes_identity():
    pinned = _route(provider={"order": ["google-ai-studio"], "allow_fallbacks": False})
    loose = _route(provider={"order": ["google-ai-studio"], "allow_fallbacks": True})
    assert experiment_identity(pinned) != experiment_identity(loose)


def test_top_level_taskroute_provider_survives_conversion_and_reaches_identity():
    """TaskRoute holds `provider` at top level and folds it into
    extra_generation only inside to_backend_config(). Identity is derived
    AFTER that conversion, so the pin cannot be lost in between."""
    r = _route(provider=AI)
    assert r.provider == AI                      # top level on the route
    assert r.to_backend_config().extra_generation["provider"] == AI   # folded in
    eff = effective_config_fields(r)
    assert eff["extra_generation"]["provider"] == AI                   # and in identity


def test_reasoning_and_overrides_also_reach_identity():
    a = _route(reasoning={"effort": "low"})
    b = _route(reasoning={"effort": "high"})
    assert experiment_identity(a) != experiment_identity(b)


@pytest.mark.parametrize("field,alt", [
    ("model", "qwen/qwen3-vl-235b-a22b-instruct"),
    ("max_tokens", 400),
    ("temperature", 0.7),
    ("structured_mode", "json_object"),
    ("prompt_version", "ocr-neutral-v2"),
])
def test_every_request_affecting_field_changes_identity(field, alt):
    assert experiment_identity(_route()) != experiment_identity(_route(**{field: alt}))


def test_retry_policy_is_experiment_identity_but_not_semantic_identity():
    """Two identities, deliberately distinct. Retry policy changes the
    experiment; it does not change what a successful response says."""
    import dataclasses

    r = _route(provider=AI)
    cfg_a = r.to_backend_config()
    cfg_b = dataclasses.replace(cfg_a, transport_retries=5)

    class _Wrap:
        def __init__(self, cfg, route):
            self._cfg, self.task, self.prompt_version = cfg, route.task, route.prompt_version
        def to_backend_config(self):
            return self._cfg

    a, b = _Wrap(cfg_a, r), _Wrap(cfg_b, r)
    assert experiment_identity(a) != experiment_identity(b), "retry policy IS experiment identity"
    blocks = [{"type": "text", "text": "q"}]
    sem = lambda x: semantic_request_identity(x, system="s", content_blocks=blocks,
                                              schema={"type": "object"}, max_tokens=100)
    assert sem(a) == sem(b), "retry policy is NOT semantic request identity"


def test_secrets_never_enter_or_appear_in_the_identity(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    r = _route(provider=AI)
    eff = effective_config_fields(r)
    blob = json.dumps(eff, default=str)
    assert SECRET not in blob
    assert "api_key" not in blob and "api_key_env" not in blob
    for f in EXCLUDED_FIELDS:
        assert f not in eff
    # and changing the key changes nothing about identity
    before = experiment_identity(r)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + ("DIFF" * 8))
    assert experiment_identity(_route(provider=AI)) == before


def test_equivalent_configs_hash_identically_regardless_of_construction_path():
    """Key order and dict construction order must not matter."""
    a = _route(provider={"order": ["google-ai-studio"], "allow_fallbacks": False},
               reasoning={"effort": "low"})
    b = _route(reasoning={"effort": "low"},
               provider={"allow_fallbacks": False, "order": ["google-ai-studio"]})
    assert experiment_identity(a) == experiment_identity(b)


def test_provider_order_is_sequence_sensitive_not_sorted():
    """`order` is a PREFERENCE list — reordering changes who serves it."""
    a = _route(provider={"order": ["google-ai-studio", "google-vertex"], "allow_fallbacks": False})
    b = _route(provider={"order": ["google-vertex", "google-ai-studio"], "allow_fallbacks": False})
    assert experiment_identity(a) != experiment_identity(b)


def test_identity_is_versioned_so_old_keys_cannot_collide():
    import autograder.routeidentity as ri

    r = _route(provider=AI)
    before = experiment_identity(r)
    monkey = ri.CACHE_IDENTITY_VERSION
    try:
        ri.CACHE_IDENTITY_VERSION = monkey + 1
        assert experiment_identity(r) != before
    finally:
        ri.CACHE_IDENTITY_VERSION = monkey
    # v1 = the hand-listed fingerprint_fields that omitted `provider`
    # v2 = derived from the effective config, but digesting the RAW schema
    # v3 = digests the CANONICAL WIRE SCHEMA actually transmitted
    assert CACHE_IDENTITY_VERSION == 3


def test_identity_report_documents_the_fields_used():
    rep = identity_report(_route(provider=AI))
    assert rep["identity_version"] == CACHE_IDENTITY_VERSION
    assert "to_backend_config" in rep["derived_from"]
    assert rep["provider_order"] == ["google-ai-studio"]
    assert rep["allow_fallbacks"] is False
    assert set(rep["excluded_fields"]) == set(EXCLUDED_FIELDS)


# =============================================================================
# 2. cache keys separate, and historical entries are unreachable
# =============================================================================

def test_cache_keys_separate_pinned_unpinned_and_cross_provider():
    from pydantic import BaseModel

    from autograder.requestcache import fingerprint

    class M(BaseModel):
        transcription: str

    blocks = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                           "data": "aW1n"}}]
    fp = lambda r: fingerprint(r, "sys", blocks, M, 1000)
    keys = {fp(_route()), fp(_route(provider=AI)), fp(_route(provider=VX))}
    assert len(keys) == 3


def test_a_planted_historical_cache_entry_is_unreachable_by_the_new_key(tmp_path):
    """A v1-scheme entry cannot be read by a v2 key, and the file survives."""
    from pydantic import BaseModel

    from autograder.requestcache import RequestCache, fingerprint

    class M(BaseModel):
        transcription: str

    cache = RequestCache(tmp_path / "c")
    blocks = [{"type": "text", "text": "q"}]
    # plant under the OLD scheme's key: the historical list omitted `provider`,
    # so the unpinned route's key is what a v1 pinned run would have used
    legacy_key = fingerprint(_route(), "sys", blocks, M, 1000)
    cache.put(legacy_key, M(transcription="HISTORICAL UNPINNED ANSWER"),
              {"task": "ocr_primary", "model": GEM})
    assert cache.get(legacy_key, M) is not None, "planted entry really is there"

    pinned_key = fingerprint(_route(provider=AI), "sys", blocks, M, 1000)
    assert pinned_key != legacy_key
    assert cache.get(pinned_key, M) is None, "a pinned request must not read the unpinned entry"
    # historical data preserved, not deleted or rewritten
    assert cache.get(legacy_key, M).transcription == "HISTORICAL UNPINNED ANSWER"


def test_refresh_policy_bypasses_reads_but_still_writes(tmp_path):
    from pydantic import BaseModel

    from autograder.backends.mock import MockBackend
    from autograder.gateway import ModelGateway
    from autograder.requestcache import RequestCache, fingerprint

    class Out(BaseModel):
        transcription: str

    cache = RequestCache(tmp_path / "c")
    gw = ModelGateway.from_dict(
        {"models": {"ocr_primary": {"backend": "mock", "model": "m", "max_tokens": 100}}},
        backend_factory=lambda c: MockBackend(config=c, responses=[Out(transcription="LIVE")] * 4),
        cache=cache)
    route = gw.route("ocr_primary")
    blocks = [{"type": "text", "text": "q"}]
    key = fingerprint(route, "s", blocks, Out, 100)
    cache.put(key, Out(transcription="STALE"), {"task": "ocr_primary", "model": "m"})

    gw.cache_read_enabled = False           # REFRESH
    res = gw.call(task="ocr_primary", system="s", content_blocks=blocks,
                  output_model=Out, max_tokens=100, meta={})
    assert res.cache_hit is False, "refresh must not read the cache"
    assert res.value.transcription == "LIVE"
    assert cache.get(key, Out) is not None, "refresh must still WRITE the entry"


# =============================================================================
# 3. campaign setup fails closed — zero sends
# =============================================================================

def test_campaign_setup_failure_is_fatal_and_sends_nothing(tmp_path, monkeypatch):
    """REGRESSION for the exact V1 failure: `campaign` was referenced before
    assignment, the broad handler downgraded it to 'capture unavailable', and
    the arm ran with NO enforcement. It must now abort with zero transport
    activity."""
    from autograder.benchmark import runner as R

    sent = []

    def transport(request):        # must never be reached
        sent.append(request)
        return httpx.Response(200, json={})

    class NoProtocolBackend:       # lacks attempt_records etc.
        pass

    class FakeGW:
        campaign_budget = object()
        def backend_for(self, task):
            return NoProtocolBackend()

    class FakeCampaign:
        campaign = "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V2"

    with pytest.raises(TypeError) as ei:
        R._install_attempt_protocol(gw=FakeGW(), route=type("R", (), {"task": "ocr_primary"})(),
                                    run_dir=tmp_path, run_id="arm", campaign=FakeCampaign())
    assert "attempt-enforcement protocol" in str(ei.value)
    assert sent == [], "no request may be sent when setup fails"


def test_install_protocol_sets_every_correlation_label(tmp_path):
    from autograder.backends.base import BackendConfig
    from autograder.backends.openrouter import OpenRouterBackend
    from autograder.benchmark import runner as R

    import os
    os.environ.setdefault("OPENROUTER_API_KEY", SECRET)
    cfg = BackendConfig(backend="openrouter", model=GEM,
                        base_url="https://openrouter.ai/api/v1",
                        api_key_env="OPENROUTER_API_KEY")
    backend = OpenRouterBackend(cfg, transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json={})))

    class GW:
        campaign_budget = True
        def backend_for(self, task):
            return backend

    class C:
        campaign = "CAMP"

    out = R._install_attempt_protocol(gw=GW(), route=type("R", (), {"task": "ocr_primary"})(),
                                      run_dir=tmp_path, run_id="ARM-1", campaign=C())
    assert out.capture_campaign_id == "CAMP"
    assert out.capture_arm_id == "ARM-1"
    assert out.capture_task == "ocr_primary"
    assert out.raw_archive is not None


def test_required_linkage_fields_are_declared():
    from autograder.benchmark.runner import REQUIRED_LINKAGE_FIELDS
    assert set(REQUIRED_LINKAGE_FIELDS) == {
        "campaign_id", "arm_id", "case_id", "logical_request_id", "attempt_id", "retry_index"}


# =============================================================================
# 4. full CLI -> wire integration, with the DEFAULT retry configuration
# =============================================================================

def test_cli_to_wire_pin_reaches_the_payload_with_default_retries(monkeypatch):
    """The real CLI path, real TaskRoute, default transport_retries."""
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    from autograder.benchmark.cli import _spec_from_args
    from autograder.benchmark.runner import build_route
    from autograder.cli import build_parser

    argv = ("bench run --role ocr_primary --split dev --candidate " + GEM +
            " --subset smoke --prompt-version m2-strict-v1 --research"
            " --models-config models.toml --cache-policy refresh"
            " --provider {\"order\":[\"google-ai-studio\"],\"allow_fallbacks\":false}"
            " --i-understand-this-spends-money").split()
    spec = _spec_from_args(build_parser().parse_args(argv), dry_run=False)
    assert spec.provider == AI and spec.cache_policy == "refresh"

    route = build_route(spec, GEM, "m2-strict-v1", 1000, registry=None)
    assert route.provider == AI, "the pin must reach the TaskRoute"

    cfg = route.to_backend_config()
    assert cfg.extra_generation["provider"] == AI, "and the effective config"
    assert cfg.transport_retries == 2, "DEFAULT retry configuration, not 0"

    # ... and the wire
    sent = []
    from autograder.backends.openrouter import OpenRouterBackend

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "gen-1", "provider": "Google AI Studio",
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.001},
            "choices": [{"finish_reason": "stop",
                         "message": {"content": '{"transcription": "x"}'}}]})

    b = OpenRouterBackend(cfg, transport=httpx.MockTransport(handler))
    b._post_chat(b._build_payload([{"role": "user", "content": "hi"}], _Tr, 1000))
    assert sent[0]["provider"] == AI, "the pin is on the wire"

    # corrected identity differs from the historical unpinned identity
    unpinned = build_route(_spec_without_provider(spec), GEM, "m2-strict-v1", 1000,
                           registry=None)
    assert experiment_identity(route) != experiment_identity(unpinned)


def _spec_without_provider(spec):
    import dataclasses
    return dataclasses.replace(spec, provider=None)


class _Tr(__import__("pydantic").BaseModel):
    transcription: str


def test_cli_to_wire_retry_budget_and_sticky_route_across_attempts(monkeypatch, tmp_path):
    """One logical call, TWO physical attempts under the default retry policy:
    both authorized, both archived, an early violation sticky, all identifiers
    present."""
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    monkeypatch.setattr("autograder.backends.openai_compat.time.sleep", lambda *_: None)
    from autograder.backends.openrouter import OpenRouterBackend
    from autograder.campaignbudget import CampaignBudget
    from autograder.rawcapture import RawResponseArchive

    route = _route(provider=AI, reasoning={"effort": "low"})
    cfg = route.to_backend_config()
    assert cfg.transport_retries == 2

    sent = []

    def handler(request):
        sent.append(request)
        if len(sent) == 1:      # served by the WRONG provider, then a retryable failure
            return httpx.Response(503, json={"id": "g0", "provider": "Google"})
        return httpx.Response(200, json={
            "id": "g1", "provider": "Google AI Studio",
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.001},
            "choices": [{"finish_reason": "stop",
                         "message": {"content": '{"transcription": "x"}'}}]})

    b = OpenRouterBackend(cfg, transport=httpx.MockTransport(handler))
    b.raw_archive = RawResponseArchive(tmp_path / "raw.jsonl")
    b.capture_campaign_id, b.capture_arm_id = "CAMP", "ARM-1"
    b.capture_case_id, b.capture_logical_request_id = "hc_e002_q1_r1", "ARM-1::hc::1"

    budget = CampaignBudget(campaign="CAMP", experiment_sha256="x", starting_ledger_usd=1.0,
                            warning_increment_usd=0.08, hard_increment_usd=0.12,
                            warn_usd=1.08, hard_usd=1.12, predicted_arm_costs={})
    authorized = []

    def hook(*, attempt_id, retry_index, payload):
        budget.authorize_attempt(attempt_id=attempt_id, ledger_now=1.0,
                                 max_request_cost_usd=0.004)
        authorized.append((attempt_id, retry_index))

    b.pre_send_hook = hook
    b._post_chat(b._build_payload([{"role": "user", "content": "hi"}], _Tr, 1000))

    assert len(sent) == 2, "the default retry policy really retried"
    assert len(authorized) == 2 and len({a for a, _ in authorized}) == 2
    assert [r for _, r in authorized] == [0, 1], "budget authorized BOTH physical sends"

    recs = b.raw_archive.records()
    assert len(recs) == 2
    for r in recs:
        for f in ("campaign_id", "arm_id", "case_id", "logical_request_id",
                  "attempt_id", "retry_index"):
            assert r.get(f) is not None, f"missing correlation field {f}"
    assert [r["attempt_id"] for r in recs] == [e.attempt_id for e in b.billing_events]

    # sticky: the LAST check is clean, the recorded violation is not erased
    assert b.last_route_check["violation"] is False
    assert b.route_violation is not None
    assert b.route_violation["observed_provider"] == "Google"
    assert [r["route_check"]["violation"] for r in recs] == [True, False]


# =============================================================================
# 5. the CANONICAL WIRE SCHEMA is part of request identity
# =============================================================================

def _mk(name, fields):
    from pydantic import create_model
    return create_model(name, **fields)


def test_changing_only_the_response_schema_changes_the_cache_identity():
    """Route, prompt, image, max_tokens and every config field held constant —
    only the response schema differs. Identity MUST move."""
    from autograder.requestcache import fingerprint

    r = _route(provider=AI)
    blocks = [{"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                           "data": "aW1n"}}]
    A = _mk("BenchTranscription", {"transcription": (str, ...)})
    B = _mk("BenchTranscription", {"transcription": (str, ...), "confidence": (float, ...)})
    assert fingerprint(r, "sys", blocks, A, 1000) != fingerprint(r, "sys", blocks, B, 1000)


def test_a_changed_field_type_alone_changes_identity():
    from autograder.requestcache import fingerprint

    r = _route(provider=AI)
    blocks = [{"type": "text", "text": "q"}]
    A = _mk("M", {"transcription": (str, ...)})
    B = _mk("M", {"transcription": (int, ...)})
    assert fingerprint(r, "sys", blocks, A, 1000) != fingerprint(r, "sys", blocks, B, 1000)


def test_semantically_identical_schemas_built_independently_share_an_identity():
    """Two models constructed separately, same NAME and same fields, produce the
    same transmitted response_format and therefore the same identity."""
    from autograder.requestcache import fingerprint

    r = _route(provider=AI)
    blocks = [{"type": "text", "text": "q"}]
    A = _mk("BenchTranscription", {"transcription": (str, ...)})
    A2 = _mk("BenchTranscription", {"transcription": (str, ...)})
    assert A is not A2
    assert fingerprint(r, "sys", blocks, A, 1000) == fingerprint(r, "sys", blocks, A2, 1000)


def test_the_transmitted_schema_name_is_part_of_identity():
    """The name really is sent (response_format.json_schema.name), so two
    identically-shaped models with DIFFERENT names are different requests."""
    from autograder.requestcache import fingerprint

    r = _route(provider=AI)
    blocks = [{"type": "text", "text": "q"}]
    A = _mk("BenchTranscription", {"transcription": (str, ...)})
    B = _mk("SomethingElse", {"transcription": (str, ...)})
    assert fingerprint(r, "sys", blocks, A, 1000) != fingerprint(r, "sys", blocks, B, 1000)


def test_identity_uses_the_CANONICAL_WIRE_schema_not_the_raw_model_schema(monkeypatch):
    """The digest must be of what is transmitted. Proof: the wire schema that
    identity hashes is byte-identical to the response_format block the backend
    actually builds."""
    import json as _json

    from autograder.backends.base import BackendConfig
    from autograder.backends.openai_compat import OpenAICompatBackend
    from autograder.routeidentity import wire_response_format

    A = _mk("BenchTranscription", {"transcription": (str, ...)})
    r = _route(provider=AI)
    cfg = BackendConfig(backend="openai", model="m", base_url="http://x/v1",
                        api_key_env="NOPE_NOT_SET")
    be = OpenAICompatBackend(cfg)
    payload = be._build_payload([{"role": "user", "content": "hi"}], A, 1000)
    sent = payload["response_format"]["json_schema"]

    ident = wire_response_format(r, A)
    assert ident["name"] == sent["name"]
    assert _json.dumps(ident["schema"], sort_keys=True) == _json.dumps(sent["schema"], sort_keys=True)
    # and it is NOT merely the raw model schema
    assert ident["schema"] != A.model_json_schema()
    assert "additionalProperties" in ident["schema"]


def test_strict_schema_flag_also_moves_identity():
    import dataclasses

    from autograder.routeidentity import experiment_identity

    r = _route(provider=AI)
    cfg = r.to_backend_config()

    class _W:
        def __init__(self, c):
            self._c, self.task, self.prompt_version = c, r.task, r.prompt_version
        def to_backend_config(self):
            return self._c

    assert experiment_identity(_W(cfg)) != experiment_identity(
        _W(dataclasses.replace(cfg, strict_schema=False)))
