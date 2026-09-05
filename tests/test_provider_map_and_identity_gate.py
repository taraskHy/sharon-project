"""Canonical provider matching, base_url canonicalisation, runtime-derived
identities, and the mandatory pre-send identity gate. ZERO network.

Every mechanism here exists because of a specific V3 failure:

* the campaign was halted on arm 2 by a route violation that was not real —
  ``google-vertex`` reports as ``Google``, and the check compared a slug to a
  display name after stripping punctuation;
* arm 1 passed only because ``google-ai-studio`` and ``Google AI Studio``
  normalise identically by coincidence;
* both arms ran under an identity that differed from the frozen one by
  ``base_url``, and execution continued anyway.
"""
import json

import httpx
import pytest

from autograder.gateway import TaskRoute
from autograder.providermap import (
    COMPLIANT, UNKNOWN, UNKNOWN_AMBIGUOUS, UNKNOWN_UNRECOGNISED,
    UNKNOWN_UNVERIFIED_SLUG, UNVERIFIED, VERIFIED, VIOLATION, ProviderEntry,
    load_provider_map, match_provider, source_digest,
)
from autograder.routeidentity import (
    CACHE_IDENTITY_VERSION, IdentityMismatch, assert_identity_matches,
    canonical_base_url, effective_config_fields, experiment_identity,
    identities_from_argv,
)

GEM = "google/gemini-3.7-flash"
QWEN = "qwen/qwen3-vl-235b-a22b-instruct"
SECRET = "sk-or-v1-" + ("FAKE" * 8) + "-NOTAREALKEY"


@pytest.fixture(scope="module")
def pmap():
    return load_provider_map()


# =============================================================================
# 1. canonical provider mapping, sourced from the preserved artifact
# =============================================================================

def test_mapping_comes_from_a_preserved_artifact_with_a_recorded_digest():
    d = source_digest()
    assert d["artifact"].endswith("OCR_PROVIDER_ROUTE_FORENSICS_2026-09-03.json")
    assert len(d["artifact_sha256"]) == 64
    assert d["recorded_content_sha256"] == \
        "efdc242f95aa311088ade2ea065cb32e3dfc3b30b5cb80524dc904ed9c15bb61"
    assert d["captured_at"] == "2026-09-04 00:18:25"


def test_the_two_google_slugs_are_verified_from_the_artifact(pmap):
    assert pmap["google-ai-studio"].display_names == ("Google AI Studio",)
    assert pmap["google-ai-studio"].status == VERIFIED
    assert pmap["google-vertex"].display_names == ("Google",)
    assert pmap["google-vertex"].status == VERIFIED


def test_alibaba_is_declared_but_explicitly_UNVERIFIED(pmap):
    """No preserved artifact records this slug's display name. It must not be
    guessed, and it must not be silently treated as compliant."""
    e = pmap["alibaba"]
    assert e.status == UNVERIFIED
    assert e.display_names == ()
    assert "never persisted" in e.evidence


# ---- the exact V3 false positive ---------------------------------------------

def test_V3_false_positive_shape_is_now_COMPLIANT(pmap):
    r = match_provider(requested_slug="google-vertex", observed_provider="Google", pmap=pmap)
    assert r["result"] == COMPLIANT
    assert r["expected_display_names"] == ["Google"]
    assert r["observed_provider"] == "Google"
    assert r["requested_provider_slug"] == "google-vertex"


def test_ai_studio_pin_is_compliant_for_the_right_reason(pmap):
    r = match_provider(requested_slug="google-ai-studio",
                       observed_provider="Google AI Studio", pmap=pmap)
    assert r["result"] == COMPLIANT
    assert r["expected_display_names"] == ["Google AI Studio"]


@pytest.mark.parametrize("slug,observed", [
    ("google-vertex", "Google AI Studio"),
    ("google-ai-studio", "Google"),
])
def test_a_genuinely_wrong_provider_is_a_violation(slug, observed, pmap):
    r = match_provider(requested_slug=slug, observed_provider=observed, pmap=pmap)
    assert r["result"] == VIOLATION
    assert observed in r["detail"]


def test_two_different_slugs_are_never_equated_by_text_similarity(pmap):
    """'google-vertex' and 'google-ai-studio' both start with 'google'. Neither
    naive prefixing nor normalisation may make them interchangeable."""
    assert match_provider(requested_slug="google-vertex",
                          observed_provider="Google AI Studio", pmap=pmap)["result"] == VIOLATION


def test_missing_provider_stays_UNKNOWN(pmap):
    for obs in (None, "", "   "):
        assert match_provider(requested_slug="google-vertex",
                              observed_provider=obs, pmap=pmap)["result"] == UNKNOWN


def test_unrecognised_display_value_is_UNKNOWN_not_compliant(pmap):
    r = match_provider(requested_slug="google-vertex",
                       observed_provider="Some New Provider", pmap=pmap)
    assert r["result"] == UNKNOWN_UNRECOGNISED
    assert "NOT silently compliant" in r["detail"]


def test_an_ambiguous_display_name_is_UNKNOWN_AMBIGUOUS():
    amb = {
        "slug-a": ProviderEntry("slug-a", ("Shared Name",), VERIFIED, "test"),
        "slug-b": ProviderEntry("slug-b", ("Shared Name",), VERIFIED, "test"),
    }
    r = match_provider(requested_slug="slug-a", observed_provider="Shared Name", pmap=amb)
    assert r["result"] == COMPLIANT           # it IS one of slug-a's names
    r2 = match_provider(requested_slug="slug-b", observed_provider="Shared Name", pmap=amb)
    assert r2["result"] == COMPLIANT
    other = {**amb, "slug-c": ProviderEntry("slug-c", ("Other",), VERIFIED, "t")}
    r3 = match_provider(requested_slug="slug-c", observed_provider="Shared Name", pmap=other)
    assert r3["result"] == UNKNOWN_AMBIGUOUS
    assert "more than one slug" in r3["detail"]


def test_unverified_slug_can_never_be_confirmed_or_refuted(pmap):
    r = match_provider(requested_slug="alibaba", observed_provider="Alibaba", pmap=pmap)
    assert r["result"] == UNKNOWN_UNVERIFIED_SLUG
    assert r["slug_mapping_status"] == UNVERIFIED
    # crucially: not compliant, and not a violation either
    assert r["result"] not in (COMPLIANT, VIOLATION)


def test_case_and_whitespace_normalisation_applies_only_after_mapping(pmap):
    for obs in ("google", "  Google  ", "GOOGLE"):
        assert match_provider(requested_slug="google-vertex",
                              observed_provider=obs, pmap=pmap)["result"] == COMPLIANT


def test_provider_is_never_inferred_from_the_model_slug(pmap):
    """A Google model served by an unnamed provider stays UNKNOWN."""
    assert match_provider(requested_slug="google-vertex", observed_provider=None,
                          pmap=pmap)["result"] == UNKNOWN


def test_check_route_uses_the_canonical_map():
    from autograder.rawcapture import EXPLICIT, check_route, requested_route_of

    req = requested_route_of({"provider": {"order": ["google-vertex"], "allow_fallbacks": False}})
    v = check_route(req, "Google", EXPLICIT)
    assert v["violation"] is False and v["result"] == COMPLIANT
    v2 = check_route(req, "Google AI Studio", EXPLICIT)
    assert v2["violation"] is True and v2["result"] == VIOLATION


# =============================================================================
# 2. base_url canonicalisation
# =============================================================================

def _route(**kw):
    base = dict(task="ocr_primary", backend="openrouter", model=GEM,
                structured_mode="json_schema", max_tokens=1000, temperature=0.0,
                reasoning={"effort": "low"}, transport_retries=0,
                provider={"order": ["google-ai-studio"], "allow_fallbacks": False},
                prompt_version="m2-strict-v1")
    base.update(kw)
    return TaskRoute(**base)


def test_none_and_the_explicit_default_hash_identically():
    assert experiment_identity(_route(base_url=None)) == \
        experiment_identity(_route(base_url="https://openrouter.ai/api/v1"))


def test_a_trailing_slash_is_not_a_different_endpoint():
    assert experiment_identity(_route(base_url="https://openrouter.ai/api/v1/")) == \
        experiment_identity(_route(base_url=None))


def test_a_genuinely_different_endpoint_changes_identity():
    assert experiment_identity(_route(base_url="https://proxy.example/v1")) != \
        experiment_identity(_route(base_url=None))


def test_canonical_base_url_only_defaults_where_a_default_is_known():
    assert canonical_base_url("openrouter", None) == "https://openrouter.ai/api/v1"
    assert canonical_base_url("openai", None) is None      # no default asserted
    assert canonical_base_url("openrouter", "http://x/v1/") == "http://x/v1"


def test_identity_version_was_incremented_for_the_canonicalisation():
    assert CACHE_IDENTITY_VERSION == 4


# =============================================================================
# 3. runtime-derived identity == hand construction, when equivalent
# =============================================================================

V4_ARGV = {
    "gemini_pinned_ai_studio": (
        "bench run --role ocr_primary --split dev --candidate " + GEM +
        " --subset smoke --prompt-version m2-strict-v1 --research"
        " --models-config models.toml --cache-policy refresh --transport-retries 0"
        " --provider {\"order\":[\"google-ai-studio\"],\"allow_fallbacks\":false}"
        " --i-understand-this-spends-money").split(),
    "gemini_pinned_vertex": (
        "bench run --role ocr_primary --split dev --candidate " + GEM +
        " --subset smoke --prompt-version m2-strict-v1 --research"
        " --models-config models.toml --cache-policy refresh --transport-retries 0"
        " --provider {\"order\":[\"google-vertex\"],\"allow_fallbacks\":false}"
        " --i-understand-this-spends-money").split(),
    "qwen3_vl_235b_pinned_alibaba": (
        "bench run --role ocr_primary --split dev --candidate " + QWEN +
        " --subset smoke --prompt-version m2-strict-v1 --research"
        " --models-config models.toml --cache-policy refresh --transport-retries 0"
        " --provider {\"order\":[\"alibaba\"],\"allow_fallbacks\":false}"
        " --i-understand-this-spends-money").split(),
}


def test_cli_and_hand_construction_agree_when_semantically_equivalent(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    cli = identities_from_argv(V4_ARGV["gemini_pinned_ai_studio"])["experiment_identity"]
    hand = experiment_identity(_route(base_url="https://openrouter.ai/api/v1"))
    assert cli == hand, "the hand-built freeze route must agree with the CLI path"


@pytest.mark.parametrize("arm", list(V4_ARGV))
def test_each_frozen_command_reproduces_its_frozen_identity(arm, monkeypatch):
    from pathlib import Path

    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    v4p = Path("evaluation/model_selection/experiments/"
               "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V4_2026-09-05.json")
    if not v4p.exists():
        pytest.skip("V4 not frozen yet")
    v4 = json.loads(v4p.read_text(encoding="utf-8"))
    frozen = next(a for a in v4["candidates"] if a["arm_id"] == arm)
    got = identities_from_argv(V4_ARGV[arm])["experiment_identity"]
    assert got == frozen["experiment_identity"]
    assert frozen["cli_argv"] == V4_ARGV[arm], "the frozen command must be the tested one"


def test_secrets_never_reach_a_runtime_identity(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    r = identities_from_argv(V4_ARGV["gemini_pinned_vertex"])
    blob = json.dumps(r["effective_config"], default=str)
    assert SECRET not in blob and "api_key" not in blob
    before = r["experiment_identity"]
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-" + ("DIFF" * 8))
    assert identities_from_argv(V4_ARGV["gemini_pinned_vertex"])["experiment_identity"] == before


# =============================================================================
# 4. identity mismatch = zero cache reads, zero sends
# =============================================================================

def test_matching_identity_passes_the_gate():
    r = _route(base_url=None)
    out = assert_identity_matches(route=r, frozen_experiment_identity=experiment_identity(r),
                                  arm_id="arm-1")
    assert out["match"] is True


@pytest.mark.parametrize("differing", [
    {"base_url": "https://proxy.example/v1"},
    {"provider": {"order": ["google-vertex"], "allow_fallbacks": False}},
    {"transport_retries": 2},
    {"max_tokens": 400},
    {"reasoning": {"effort": "high"}},
])
def test_any_differing_field_raises_with_zero_sends(differing):
    """Each field individually. The gate runs before any transport exists, so a
    mismatch cannot produce a request by construction — asserted explicitly."""
    frozen = experiment_identity(_route(base_url=None))
    sent = []

    def transport(request):            # must never be reached
        sent.append(request)
        return httpx.Response(200, json={})

    _ = httpx.MockTransport(transport)
    with pytest.raises(IdentityMismatch) as ei:
        assert_identity_matches(route=_route(**differing),
                                frozen_experiment_identity=frozen, arm_id="arm-1")
    assert "ZERO cache reads and ZERO requests" in str(ei.value)
    assert sent == [], "no request may be sent when identities disagree"


def test_the_mismatch_message_carries_a_secret_free_field_level_diff(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)
    frozen = experiment_identity(_route(base_url=None))
    with pytest.raises(IdentityMismatch) as ei:
        assert_identity_matches(route=_route(base_url="https://proxy.example/v1"),
                                frozen_experiment_identity=frozen, arm_id="arm-x")
    msg = str(ei.value)
    assert "proxy.example" in msg, "the differing value must be visible"
    assert SECRET not in msg and "api_key" not in msg


def test_the_gate_is_reachable_from_the_cli():
    from autograder.benchmark.cli import _spec_from_args
    from autograder.cli import build_parser

    argv = V4_ARGV["gemini_pinned_vertex"] + ["--expect-identity", "deadbeef"]
    spec = _spec_from_args(build_parser().parse_args(argv), dry_run=False)
    assert spec.expect_identity == "deadbeef"


def test_the_V3_identity_mismatch_would_now_be_a_hard_stop():
    """V3 froze base_url='https://openrouter.ai/api/v1' and ran with None. Under
    canonicalisation those now AGREE, so that specific mismatch disappears — and
    had it not, the gate would have stopped the arm instead of continuing."""
    assert experiment_identity(_route(base_url=None)) == \
        experiment_identity(_route(base_url="https://openrouter.ai/api/v1"))
    frozen = experiment_identity(_route(base_url=None))
    with pytest.raises(IdentityMismatch):
        assert_identity_matches(route=_route(max_tokens=999),
                                frozen_experiment_identity=frozen, arm_id="v3-arm-1")
