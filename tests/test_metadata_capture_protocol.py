"""OCR_PROVIDER_METADATA_CAPTURE_PROTOCOL_V1. ZERO network.

Every acceptance threshold is a module constant frozen before capture, so these
tests exercise :func:`evaluate_acceptance` as a pure function over synthetic
snapshots. That is the whole point: V4 was blocked because "capture a catalogue"
was prose, and prose cannot be run against a malformed catalogue to see what it
would decide.
"""
import copy
import json
from pathlib import Path

import pytest

from autograder.metadatacapture import (
    ACCEPTED, ALLOWED_RESPONSE_HEADERS, ENDPOINT_MODEL_ENDPOINTS, ENDPOINT_MODELS,
    ENDPOINT_PROVIDERS, FAILED, FOLLOW_REDIRECTS, FORBIDDEN_REQUEST_HEADERS,
    FROZEN_CAMPAIGN_MAXIMUM_USD, FROZEN_PRICES, HTTP_METHOD, PRESERVED_MAPPING,
    PROTOCOL_ID, REQUIRED_ARMS, REQUIRED_PROVIDER_SLUGS, conservative_arm_cost,
    evaluate_acceptance, protocol_document, safe_headers,
)

PROTOCOL_ARTIFACT = Path("evaluation/model_selection/experiments/"
                         "OCR_PROVIDER_METADATA_CAPTURE_PROTOCOL_V1_2026-09-06.json")
GEM = "google/gemini-3.7-flash"
QWEN = "qwen/qwen3-vl-235b-a22b-instruct"


def _good_snapshot():
    """A synthetic snapshot that must be ACCEPTED."""
    req = {"http_status": 200, "headers": {"content-type": "application/json"},
           "raw_body_sha256": "0" * 64, "archived": True, "parse_error": None}
    return {
        "requests": {"providers": dict(req), "models": dict(req),
                     f"endpoints:{GEM}": dict(req), f"endpoints:{QWEN}": dict(req)},
        "providers": [
            {"slug": "google-ai-studio", "name": "Google AI Studio"},
            {"slug": "google-vertex", "name": "Google"},
            {"slug": "alibaba", "name": "Alibaba"},
            {"slug": "someone-else", "name": "Someone Else"},
        ],
        "model_endpoints": {
            GEM: {"canonical_slug": GEM,
                  "endpoints": [{"provider_name": "Google"},
                                {"provider_name": "Google AI Studio"}],
                  "capabilities": {"image_input": True, "structured_outputs": True}},
            QWEN: {"canonical_slug": QWEN,
                   "endpoints": [{"provider_name": "Alibaba"}],
                   "capabilities": {"image_input": True, "structured_outputs": True}},
        },
        "prices": {
            GEM: {"input_per_m": 0.75, "output_per_m": 3.75},
            QWEN: {"input_per_m": 0.21, "output_per_m": 1.90},
        },
    }


# ---- the protocol is frozen and self-consistent -------------------------------

def test_protocol_artifact_self_hash_verifies():
    import hashlib

    d = json.loads(PROTOCOL_ARTIFACT.read_text(encoding="utf-8"))
    body = json.dumps({k: v for k, v in d.items() if k != "content_sha256"},
                      ensure_ascii=False, indent=1, sort_keys=True, default=str)
    assert hashlib.sha256(body.encode()).hexdigest() == d["content_sha256"]
    assert d["protocol"] == PROTOCOL_ID
    assert d["requests_performed_at_freeze_time"] == 0


def test_the_artifact_matches_the_executable_protocol():
    """Prose and code must not drift: the artifact IS protocol_document()."""
    d = json.loads(PROTOCOL_ARTIFACT.read_text(encoding="utf-8"))
    live = protocol_document()
    for k, v in live.items():
        assert d[k] == v, f"artifact/code disagree on {k}"


def test_protocol_freezes_method_body_and_safety():
    assert HTTP_METHOD == "GET"
    assert protocol_document()["request_body"] is None
    assert FOLLOW_REDIRECTS is False
    assert set(FORBIDDEN_REQUEST_HEADERS) >= {"authorization", "x-api-key", "cookie"}
    assert "STOP rather than sending it" in protocol_document()["unauthenticated"]


def test_endpoints_are_the_three_public_metadata_urls():
    assert ENDPOINT_PROVIDERS == "https://openrouter.ai/api/v1/providers"
    assert ENDPOINT_MODELS == "https://openrouter.ai/api/v1/models"
    assert ENDPOINT_MODEL_ENDPOINTS.endswith("/models/{slug}/endpoints")


def test_allowed_headers_exclude_anything_credential_bearing():
    assert not (set(ALLOWED_RESPONSE_HEADERS) & set(FORBIDDEN_REQUEST_HEADERS))
    kept = safe_headers({"content-type": "application/json", "authorization": "Bearer x",
                         "cookie": "a=b", "x-request-id": "r1", "x-secret": "s"})
    assert kept == {"content-type": "application/json", "x-request-id": "r1"}


# ---- the happy path ----------------------------------------------------------

def test_a_wellformed_catalogue_is_accepted():
    r = evaluate_acceptance(_good_snapshot())
    assert r["result"] == ACCEPTED, [x for x in r["reasons"] if not x["ok"]]
    assert r["resolved_mapping"] == {"google-ai-studio": "Google AI Studio",
                                     "google-vertex": "Google", "alibaba": "Alibaba"}
    assert r["conservative_campaign_maximum_usd"] <= FROZEN_CAMPAIGN_MAXIMUM_USD


def test_the_conservative_maximum_reproduces_the_frozen_figure():
    r = evaluate_acceptance(_good_snapshot())
    assert r["conservative_campaign_maximum_usd"] == pytest.approx(0.096896, abs=1e-6)
    assert r["remaining_headroom_usd"] == pytest.approx(0.00850375, abs=1e-6)


# ---- malformed / ambiguous catalogues ----------------------------------------

def _expect_fail(mutate, code):
    s = _good_snapshot()
    mutate(s)
    r = evaluate_acceptance(s)
    assert r["result"] == FAILED
    assert any(x["code"] == code for x in r["reasons"] if not x["ok"]), \
        [x for x in r["reasons"] if not x["ok"]]
    return r


def test_absent_alibaba_slug_fails():
    _expect_fail(lambda s: s["providers"].remove(
        next(p for p in s["providers"] if p["slug"] == "alibaba")), "SLUG_ABSENT")


def test_duplicated_slug_fails():
    _expect_fail(lambda s: s["providers"].append({"slug": "alibaba", "name": "Alibaba Cloud"}),
                 "SLUG_NOT_UNIQUE")


def test_empty_display_name_fails():
    def m(s):
        for p in s["providers"]:
            if p["slug"] == "alibaba":
                p["name"] = "   "
    _expect_fail(m, "EMPTY_DISPLAY_NAME")


def test_reverse_ambiguity_is_reported_explicitly():
    _expect_fail(lambda s: s["providers"].append({"slug": "other-slug", "name": "Alibaba"}),
                 "REVERSE_AMBIGUOUS")


def test_a_changed_google_mapping_fails_rather_than_updating():
    def m(s):
        for p in s["providers"]:
            if p["slug"] == "google-vertex":
                p["name"] = "Google Cloud Vertex"
    _expect_fail(m, "PRESERVED_MAPPING_CHANGED")


def test_a_renamed_alibaba_is_accepted_only_because_it_has_no_preserved_mapping():
    """An UNVERIFIED slug has nothing to contradict — it just has to resolve
    uniquely. That is the difference between 'unknown' and 'changed'."""
    s = _good_snapshot()
    # a REAL rename appears in both the catalogue and the endpoint listing
    for p in s["providers"]:
        if p["slug"] == "alibaba":
            p["name"] = "Alibaba Cloud Model Studio"
    s["model_endpoints"][QWEN]["endpoints"] = [{"provider_name": "Alibaba Cloud Model Studio"}]
    r = evaluate_acceptance(s)
    assert r["result"] == ACCEPTED, [x for x in r["reasons"] if not x["ok"]]
    assert r["resolved_mapping"]["alibaba"] == "Alibaba Cloud Model Studio"


def test_a_rename_visible_in_only_one_source_fails_the_cross_check():
    """Catalogue and endpoint listing must agree. A name that moves in one place
    but not the other means the arm's route cannot be confirmed."""
    s = _good_snapshot()
    for p in s["providers"]:
        if p["slug"] == "alibaba":
            p["name"] = "Alibaba Cloud Model Studio"
    r = evaluate_acceptance(s)          # endpoints still say "Alibaba"
    assert r["result"] == FAILED
    assert any(x["code"] == "ARM_ROUTE_UNAVAILABLE" for x in r["reasons"] if not x["ok"])


# ---- endpoint availability ----------------------------------------------------

def test_missing_arm_route_fails():
    def m(s):
        s["model_endpoints"][QWEN]["endpoints"] = [{"provider_name": "DeepInfra"}]
    _expect_fail(m, "ARM_ROUTE_UNAVAILABLE")


def test_a_gemini_route_lost_fails():
    def m(s):
        s["model_endpoints"][GEM]["endpoints"] = [{"provider_name": "Google AI Studio"}]
    _expect_fail(m, "ARM_ROUTE_UNAVAILABLE")


def test_alias_redirect_fails():
    def m(s):
        s["model_endpoints"][QWEN]["canonical_slug"] = "qwen/qwen3-vl-235b-a22b-instruct-0925"
    _expect_fail(m, "MODEL_ALIAS_REDIRECT")


def test_lost_image_input_or_structured_output_fails():
    _expect_fail(lambda s: s["model_endpoints"][GEM]["capabilities"].update(
        {"image_input": False}), "IMAGE_INPUT_LOST")
    _expect_fail(lambda s: s["model_endpoints"][QWEN]["capabilities"].update(
        {"structured_outputs": False}), "STRUCTURED_OUTPUT_LOST")


# ---- pricing ------------------------------------------------------------------

def test_a_price_increase_that_breaches_the_frozen_maximum_is_rejected():
    def m(s):
        s["prices"][GEM] = {"input_per_m": 0.75, "output_per_m": 8.00}
    r = _expect_fail(m, "CAMPAIGN_MAXIMUM_EXCEEDED")
    assert r["conservative_campaign_maximum_usd"] > FROZEN_CAMPAIGN_MAXIMUM_USD


def test_a_lower_live_price_does_NOT_lower_the_conservative_maximum():
    """A promotion must not shrink a committed ceiling."""
    s = _good_snapshot()
    s["prices"][GEM] = {"input_per_m": 0.10, "output_per_m": 0.50}
    r = evaluate_acceptance(s)
    assert r["result"] == ACCEPTED
    eff = r["effective_prices_per_m"][GEM]
    assert eff["input_per_m"] == FROZEN_PRICES[GEM]["input_per_m"]
    assert eff["output_per_m"] == FROZEN_PRICES[GEM]["output_per_m"]
    assert eff["retained_frozen_because_live_is_lower"] is True
    assert r["conservative_campaign_maximum_usd"] == pytest.approx(0.096896, abs=1e-6)


def test_missing_price_fields_fail():
    _expect_fail(lambda s: s["prices"].pop(QWEN), "PRICE_FIELDS_MISSING")


def test_conservative_arm_cost_uses_max_tokens_not_an_expectation():
    c = conservative_arm_cost(GEM, 0.75, 3.75)
    assert c == pytest.approx(0.038328, abs=1e-6)


# ---- transport / archival -----------------------------------------------------

@pytest.mark.parametrize("mutate,code", [
    (lambda s: s["requests"]["providers"].update({"http_status": 500}), "HTTP_NOT_200"),
    (lambda s: s["requests"]["models"].update({"headers": {"content-type": "text/html"}}),
     "BAD_CONTENT_TYPE"),
    (lambda s: s["requests"]["providers"].update({"archived": False}), "ARCHIVE_FAILURE"),
    (lambda s: s["requests"]["models"].update({"parse_error": "Expecting value"}),
     "PARSE_FAILURE"),
])
def test_transport_and_archival_failures_reject(mutate, code):
    _expect_fail(mutate, code)


def test_failure_never_silently_drops_qwen():
    doc = protocol_document()
    assert "do NOT drop Qwen" in doc["failure_action"]
    assert any(a["arm_id"] == "qwen3_vl_235b_pinned_alibaba" for a in REQUIRED_ARMS)
    assert "alibaba" in REQUIRED_PROVIDER_SLUGS


def test_gates_and_output_assumptions_are_never_weakened_to_fit():
    doc = protocol_document()
    assert "may not be relaxed to make a budget fit" in doc["gates_never_weakened"]
    assert doc["price_rules"]["effective_price"].startswith("max(frozen, live)")
