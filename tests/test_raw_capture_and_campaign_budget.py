"""Raw-response preservation, campaign budget and route enforcement.

ZERO network. Every provider interaction here is an httpx.MockTransport.

The three mechanisms under test all exist because of a specific, documented
failure:

* the prompt-v2 arms lost 27 filtered response bodies, so the question "which
  provider filtered this crop?" became permanently unanswerable;
* cumulative ceilings computed per arm would hand each sequential arm a fresh
  increment;
* a pinned arm silently served by a different provider is not the arm we froze.
"""
import json
from pathlib import Path

import httpx
import pytest

from autograder.backends.base import BackendConfig
from autograder.backends.openrouter import OpenRouterBackend
from autograder.campaignbudget import (
    CampaignBudget, CampaignBudgetError, CampaignBudgetExceeded,
    create_campaign_budget, load_campaign_budget,
)
from autograder.rawcapture import (
    EXPLICIT, FORBIDDEN_HEADERS, MAX_BODY_CHARS, SAFE_HEADERS, UNKNOWN,
    RawResponseArchive, build_record, check_route, observed_provider_of,
    redact_secrets, requested_route_of, safe_headers,
)

#: A SYNTHETIC, NON-FUNCTIONING credential used only to prove that redaction
#: works. It is not a real key and has never been valid anywhere.
SECRET = "sk-or-v1-" + ("FAKE" * 8) + "-NOTAREALKEY"


# =============================================================================
# 1. Raw response preservation
# =============================================================================

#: The exact historical shape: HTTP 200, no usage block, no provider field,
#: filtered content. This is what the prompt-v2 arms threw away 27 times.
FILTERED_200 = {
    "id": "gen-1788399067-ten6XN1Jm2z60W5Kk97z",
    "model": "google/gemini-3.7-flash",
    "choices": [{"finish_reason": "content_filter", "message": {"role": "assistant", "content": None}}],
}


def _backend(handler, *, provider=None, monkeypatch=None):
    eg = {"reasoning": {"effort": "low"}}
    if provider is not None:
        eg["provider"] = provider
    cfg = BackendConfig(backend="openrouter", model="google/gemini-3.7-flash",
                        base_url="https://openrouter.ai/api/v1", api_key_env="OPENROUTER_API_KEY",
                        extra_generation=eg, transport_retries=0)
    return OpenRouterBackend(cfg, transport=httpx.MockTransport(handler))


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)


def test_filtered_200_with_no_usage_and_no_provider_is_still_archived(tmp_path):
    def handler(request):
        return httpx.Response(200, json=FILTERED_200,
                              headers={"content-type": "application/json",
                                       "x-request-id": "req-abc123"})
    b = _backend(handler, provider={"order": ["google-ai-studio"], "allow_fallbacks": False})
    b.raw_archive = RawResponseArchive(tmp_path / "raw.jsonl")
    b.capture_task, b.capture_case_id = "ocr_primary", "hc_e002_q1_r1"

    b._post_chat({"model": "google/gemini-3.7-flash",
                  "provider": {"order": ["google-ai-studio"], "allow_fallbacks": False},
                  "messages": []})

    recs = b.raw_archive.records()
    assert len(recs) == 1
    r = recs[0]
    # the failure shape is faithfully recorded ...
    assert r["http_status"] == 200
    assert r["parsed_outcome"]["finish_reason"] == "content_filter"
    assert r["parsed_outcome"]["usage_returned"] is False
    # ... and the raw body survives, which is the whole point
    assert r["raw_body"] is not None
    assert json.loads(r["raw_body"])["choices"][0]["finish_reason"] == "content_filter"
    assert r["raw_body_sha256"] and len(r["raw_body_sha256"]) == 64
    assert r["raw_body_bytes"] > 0
    assert r["content_type"] == "application/json"
    assert r["ts"] and r["case_id"] == "hc_e002_q1_r1" and r["task"] == "ocr_primary"
    assert r["requested_model"] == "google/gemini-3.7-flash"


def test_attribution_fields_stay_distinct_and_are_never_inferred(tmp_path):
    """A body that names no provider must yield observed=None / UNKNOWN — never
    the requested route echoed back as if it had been observed."""
    def handler(request):
        return httpx.Response(200, json=FILTERED_200)
    b = _backend(handler, provider={"order": ["google-ai-studio"], "allow_fallbacks": False})
    b.raw_archive = RawResponseArchive(tmp_path / "raw.jsonl")
    b._post_chat({"model": "m", "provider": {"order": ["google-ai-studio"],
                                             "allow_fallbacks": False}, "messages": []})
    r = b.raw_archive.records()[0]
    assert r["requested_provider"] == "google-ai-studio"
    assert r["requested_provider_order"] == ["google-ai-studio"]
    assert r["allow_fallbacks"] is False
    assert r["route_pinned"] is True
    assert r["observed_provider"] is None, "must not be inferred from the request"
    assert r["provider_attribution_status"] == UNKNOWN


def test_explicit_provider_is_recorded_as_observed(tmp_path):
    body = dict(FILTERED_200, provider="Google AI Studio",
                usage={"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001})
    b = _backend(lambda r: httpx.Response(200, json=body),
                 provider={"order": ["google-ai-studio"], "allow_fallbacks": False})
    b.raw_archive = RawResponseArchive(tmp_path / "raw.jsonl")
    b._post_chat({"model": "m", "provider": {"order": ["google-ai-studio"],
                                             "allow_fallbacks": False}, "messages": []})
    r = b.raw_archive.records()[0]
    assert r["observed_provider"] == "Google AI Studio"
    assert r["provider_attribution_status"] == EXPLICIT
    assert r["route_check"]["violation"] is False


def test_parsed_outcome_is_stored_separately_from_the_raw_body(tmp_path):
    body = dict(FILTERED_200, usage={"prompt_tokens": 100, "completion_tokens": 20, "cost": 0.002})
    b = _backend(lambda r: httpx.Response(200, json=body))
    b.raw_archive = RawResponseArchive(tmp_path / "raw.jsonl")
    b._post_chat({"model": "m", "messages": []})
    r = b.raw_archive.records()[0]
    assert set(r["parsed_outcome"]) >= {"finish_reason", "usage_returned", "input_tokens",
                                        "output_tokens", "reported_cost"}
    assert "parsed_outcome" not in json.loads(r["raw_body"]), "raw body must be untouched"
    assert r["parsed_outcome"]["input_tokens"] == 100


@pytest.mark.parametrize("status", [400, 429, 500])
def test_non_200_responses_are_archived_too(tmp_path, status):
    from autograder.backends.base import BackendError
    b = _backend(lambda r: httpx.Response(status, json={"error": {"message": "nope"}}))
    b.raw_archive = RawResponseArchive(tmp_path / "raw.jsonl")
    with pytest.raises(BackendError):
        b._post_chat({"model": "m", "messages": []})
    recs = b.raw_archive.records()
    assert recs and recs[0]["http_status"] == status
    assert recs[0]["raw_body"] is not None


def test_non_json_body_is_archived(tmp_path):
    from autograder.backends.base import BackendError
    b = _backend(lambda r: httpx.Response(200, text="<html>gateway error</html>"))
    b.raw_archive = RawResponseArchive(tmp_path / "raw.jsonl")
    with pytest.raises(BackendError):
        b._post_chat({"model": "m", "messages": []})
    r = b.raw_archive.records()[0]
    assert "<html>" in r["raw_body"]
    assert r["provider_attribution_status"] == UNKNOWN


def test_archive_is_append_only(tmp_path):
    b = _backend(lambda r: httpx.Response(200, json=FILTERED_200))
    b.raw_archive = RawResponseArchive(tmp_path / "raw.jsonl")
    for _ in range(3):
        b._post_chat({"model": "m", "messages": []})
    assert len(b.raw_archive.records()) == 3


def test_archive_write_failure_fails_closed(tmp_path):
    """CORRECTED CLAIM. An earlier version of this suite asserted that an
    archiving failure "never breaks the request" and returned the parsed body
    anyway. That is the wrong guarantee: the response was already billed, and
    accepting it without its evidence is exactly the blindness this module
    exists to prevent. It now FAILS CLOSED."""
    from autograder.rawcapture import ArchiveFailure

    class Exploding:
        path = None
        def append(self, rec):
            raise OSError("disk full")
    b = _backend(lambda r: httpx.Response(200, json=dict(FILTERED_200, usage={"prompt_tokens": 1})))
    b.raw_archive = Exploding()
    with pytest.raises(ArchiveFailure) as ei:
        b._post_chat({"model": "m", "messages": []})
    assert "already billed" in str(ei.value)


def test_archive_failure_leaves_an_independent_audit_record(tmp_path):
    """The strongest record still available once the primary write failed: a
    sibling marker file plus a tainted billing event."""
    from autograder.rawcapture import ArchiveFailure, RawResponseArchive

    arch = RawResponseArchive(tmp_path / "raw.jsonl")
    def boom(rec):
        raise OSError("disk full")
    arch.append = boom
    b = _backend(lambda r: httpx.Response(200, json=dict(FILTERED_200, usage={"prompt_tokens": 1})))
    b.raw_archive = arch
    b.capture_case_id, b.capture_arm_id = "hc_e002_q1_r1", "arm-1"
    with pytest.raises(ArchiveFailure):
        b._post_chat({"model": "m", "messages": []})

    marker = tmp_path / "raw.jsonl.ARCHIVE_FAILURE"
    assert marker.exists(), "an independent failure record must survive"
    rec = json.loads(marker.read_text(encoding="utf-8").splitlines()[0])
    assert rec["case_id"] == "hc_e002_q1_r1" and rec["arm_id"] == "arm-1"
    assert rec["attempt_id"] and "disk full" in rec["error"]
    assert "billed but NOT archived" in rec["consequence"]
    # the billing event is tainted so the ledger row cannot read as clean
    assert b.billing_events[-1].parse_ok is False
    assert "ARCHIVE_FAILURE" in (b.billing_events[-1].error or "")


# ---- secret redaction --------------------------------------------------------

def test_no_secrets_are_retained_in_an_archived_record(tmp_path):
    """The Authorization header the client really sends must never reach the
    archive, and a body echoing a key must be redacted."""
    def handler(request):
        assert request.headers["authorization"] == f"Bearer {SECRET}"   # really sent
        return httpx.Response(200, json={**FILTERED_200, "echo": f"Bearer {SECRET}"},
                              headers={"content-type": "application/json",
                                       "authorization": f"Bearer {SECRET}",
                                       "set-cookie": "session=abc",
                                       "x-api-key": SECRET,
                                       "x-request-id": "req-1"})
    b = _backend(handler)
    b.raw_archive = RawResponseArchive(tmp_path / "raw.jsonl")
    b._post_chat({"model": "m", "messages": []})
    blob = (tmp_path / "raw.jsonl").read_text(encoding="utf-8")
    assert SECRET not in blob
    assert "sk-or-v1-" not in blob
    assert "set-cookie" not in blob.lower()
    assert "authorization" not in blob.lower()
    r = json.loads(blob)
    assert r["headers"].get("x-request-id") == "req-1"      # safe header kept
    assert "[REDACTED]" in r["raw_body"]


def test_header_allowlist_drops_everything_unrecognised():
    got = safe_headers({"content-type": "application/json", "x-request-id": "r1",
                        "authorization": "Bearer x", "cookie": "a=b",
                        "x-secret-internal": "leak-me"})
    assert got == {"content-type": "application/json", "x-request-id": "r1"}
    assert not (SAFE_HEADERS & FORBIDDEN_HEADERS), "allowlist and denylist must not overlap"


@pytest.mark.parametrize("text", [
    f"Bearer {SECRET}", SECRET, 'api_key: "supersecretvalue"', '"api-key": "abcdefghij"',
])
def test_redact_secrets_catches_common_shapes(text):
    assert "[REDACTED]" in redact_secrets(text)


def test_oversized_body_is_truncated_but_hash_describes_the_whole(tmp_path):
    big = "x" * (MAX_BODY_CHARS + 500)
    rec = build_record(payload={"model": "m"}, http_status=200, raw_text=big, headers={})
    d = rec.to_json()
    assert d["raw_body_truncated"] is True
    assert len(d["raw_body"]) == MAX_BODY_CHARS
    import hashlib
    assert d["raw_body_sha256"] == hashlib.sha256(big.encode()).hexdigest()
    assert d["raw_body_bytes"] == len(big.encode())


# =============================================================================
# 2. Campaign-wide budget enforcement
# =============================================================================

L0 = 0.703232
ARMS = {"gemini_pinned_ai_studio": 0.038328, "gemini_pinned_vertex": 0.038328,
        "qwen3_vl_235b_pinned_alibaba": 0.020240}


def _budget(tmp_path, name="cb.json"):
    return create_campaign_budget(
        campaign="OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1", experiment_sha256="39305a62",
        starting_ledger_usd=L0, warning_increment_usd=0.08, hard_increment_usd=0.12,
        predicted_arm_costs=ARMS, path=tmp_path / name)


def test_thresholds_are_absolute_and_derived_from_L0(tmp_path):
    b = _budget(tmp_path)
    assert b.starting_ledger_usd == L0
    assert b.warn_usd == pytest.approx(L0 + 0.08)
    assert b.hard_usd == pytest.approx(L0 + 0.12)
    assert b.predicted_campaign_worst_case_usd == pytest.approx(0.096896)


def test_sequential_arms_cannot_each_receive_a_fresh_budget(tmp_path):
    """THE regression this module exists for. Three arms, one envelope: by the
    third arm the remaining allowance must reflect what the first two spent."""
    b = _budget(tmp_path)
    reloaded = [load_campaign_budget(tmp_path / "cb.json") for _ in range(3)]
    assert {r.hard_usd for r in reloaded} == {b.hard_usd}, "every arm sees the same ceiling"

    ledger = L0
    for arm, spend in (("arm1", 0.05), ("arm2", 0.05)):
        load_campaign_budget(tmp_path / "cb.json").check(
            ledger_now=ledger, max_request_cost_usd=0.004)
        ledger += spend                                    # the arm ran and spent
    assert ledger == pytest.approx(L0 + 0.10)

    # arm 3 now has only $0.02 of headroom, NOT a fresh $0.12
    third = load_campaign_budget(tmp_path / "cb.json")
    assert third.remaining_usd(ledger) == pytest.approx(0.02, abs=1e-6)
    with pytest.raises(CampaignBudgetExceeded):
        third.check(ledger_now=ledger, max_request_cost_usd=0.03)


def test_the_naive_per_arm_recompute_would_have_over_spent():
    """Documents the bug: recomputing ledger_now + increment per arm hands the
    third arm a ceiling far above the campaign authorization."""
    increment, ledger = 0.12, L0
    naive = []
    for _ in range(3):
        naive.append(ledger + increment)       # the WRONG way
        ledger += 0.05
    assert naive[2] > naive[0], "each arm would get a higher ceiling than the last"
    assert naive[2] - (L0 + increment) == pytest.approx(0.10)
    fixed = [L0 + increment] * 3               # the RIGHT way
    assert len(set(fixed)) == 1


def test_reservation_uses_the_worst_case_not_the_expectation(tmp_path):
    b = _budget(tmp_path)
    at_limit = b.hard_usd - 0.002
    b.check(ledger_now=at_limit, max_request_cost_usd=0.001)          # fits
    with pytest.raises(CampaignBudgetExceeded):
        b.check(ledger_now=at_limit, max_request_cost_usd=0.005)      # worst case does not


def test_warning_state_transitions(tmp_path):
    b = _budget(tmp_path)
    assert b.warning_state(L0) == "OK"
    assert b.warning_state(L0 + 0.09) == "WARNING"
    assert b.warning_state(L0 + 0.13) == "HARD"


def test_manifest_is_persisted_with_every_required_field(tmp_path):
    _budget(tmp_path)
    d = json.loads((tmp_path / "cb.json").read_text(encoding="utf-8"))
    for k in ("campaign", "experiment_sha256", "starting_ledger_usd_L0", "warning_increment_usd",
              "hard_increment_usd", "warn_usd_absolute", "hard_usd_absolute",
              "predicted_arm_costs_usd", "predicted_campaign_worst_case_usd", "content_sha256"):
        assert k in d, f"missing {k}"


def test_manifest_cannot_be_widened_mid_campaign(tmp_path):
    _budget(tmp_path)
    d = json.loads((tmp_path / "cb.json").read_text(encoding="utf-8"))
    d["hard_usd_absolute"] = 99.0
    (tmp_path / "cb.json").write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(CampaignBudgetError):
        load_campaign_budget(tmp_path / "cb.json")


def test_L0_is_immutable_a_campaign_gets_exactly_one(tmp_path):
    _budget(tmp_path)
    with pytest.raises(CampaignBudgetError):
        _budget(tmp_path)          # second create against the same path


def test_inconsistent_manifest_is_refused(tmp_path):
    from autograder.campaignbudget import _seal
    d = json.loads((_budget(tmp_path)).path.read_text(encoding="utf-8"))
    d["starting_ledger_usd_L0"] = 0.0        # thresholds no longer L0 + increment
    (tmp_path / "cb.json").write_text(
        json.dumps(_seal({k: v for k, v in d.items() if k != "content_sha256"})), encoding="utf-8")
    with pytest.raises(CampaignBudgetError):
        load_campaign_budget(tmp_path / "cb.json")


# =============================================================================
# 3. Route enforcement
# =============================================================================

def test_a_pin_requires_one_provider_and_no_fallbacks():
    assert requested_route_of({"provider": {"order": ["google-vertex"],
                                            "allow_fallbacks": False}})["route_pinned"] is True
    assert requested_route_of({"provider": {"order": ["google-vertex"],
                                            "allow_fallbacks": True}})["route_pinned"] is False
    assert requested_route_of({"provider": {"order": ["a", "b"],
                                            "allow_fallbacks": False}})["route_pinned"] is False
    assert requested_route_of({})["route_pinned"] is False


def test_an_explicitly_different_provider_is_a_route_violation():
    req = requested_route_of({"provider": {"order": ["google-ai-studio"], "allow_fallbacks": False}})
    v = check_route(req, "Google", EXPLICIT)
    assert v["violation"] is True
    assert "google-ai-studio" in v["detail"] and "Google" in v["detail"]


def test_matching_provider_is_not_a_violation_across_spellings():
    req = requested_route_of({"provider": {"order": ["google-ai-studio"], "allow_fallbacks": False}})
    assert check_route(req, "Google AI Studio", EXPLICIT)["violation"] is False


def test_unknown_attribution_is_not_a_violation():
    """Absence of evidence is not evidence of a breach — otherwise every
    historical filtered response would look like a route violation."""
    req = requested_route_of({"provider": {"order": ["google-vertex"], "allow_fallbacks": False}})
    v = check_route(req, None, UNKNOWN)
    assert v["violation"] is False
    assert "cannot be confirmed" in v["detail"]


def test_unpinned_route_is_never_a_violation():
    v = check_route(requested_route_of({}), "Anyone", EXPLICIT)
    assert v["violation"] is False
    assert "not pinned" in v["detail"]


def test_backend_exposes_the_route_check_for_the_runner_to_act_on(tmp_path):
    body = dict(FILTERED_200, provider="Google",       # served by Vertex ...
                usage={"prompt_tokens": 5, "cost": 0.0001})
    b = _backend(body and (lambda r: httpx.Response(200, json=body)),
                 provider={"order": ["google-ai-studio"], "allow_fallbacks": False})
    b.raw_archive = RawResponseArchive(tmp_path / "raw.jsonl")
    b._post_chat({"model": "m", "provider": {"order": ["google-ai-studio"],
                                             "allow_fallbacks": False}, "messages": []})
    assert b.last_route_check["violation"] is True      # ... but we pinned AI Studio
    assert b.raw_archive.records()[0]["route_check"]["violation"] is True


def test_the_frozen_screen_pins_every_arm():
    screen = json.loads(Path("evaluation/model_selection/experiments/"
                             "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1_2026-09-03.json")
                        .read_text(encoding="utf-8"))
    for arm in screen["candidates"]:
        pr = arm["provider_routing"]
        assert pr["order"] == [arm["provider_pin"]], "order must be exactly the frozen provider"
        assert pr["allow_fallbacks"] is False
        assert requested_route_of({"provider": pr})["route_pinned"] is True
