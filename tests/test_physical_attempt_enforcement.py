"""Enforcement at the PHYSICAL HTTP attempt, including transport retries.

ZERO network — every provider interaction is an httpx.MockTransport.

This file exists because the earlier suite did not prove what it claimed. Its
backends were built with ``transport_retries=0``, so *no* test ever drove a
retry, and three properties were asserted only for the single-attempt case:

* the campaign budget was checked once per LOGICAL call, while ``_post_chat``
  can send up to ``transport_retries + 1`` times — a retry inherited the first
  attempt's authorization;
* ``last_route_check`` was a single mutable field, so a violation on attempt 1
  was erased by a clean attempt 2;
* an archive write failure was swallowed and the parsed body returned as a
  success.

Scenarios A–F below drive the real retry path.
"""
import json

import httpx
import pytest

from autograder.backends.base import BackendConfig, BackendError
from autograder.backends.openrouter import OpenRouterBackend
from autograder.campaignbudget import CampaignBudget, CampaignBudgetExceeded
from autograder.rawcapture import ArchiveFailure, EXPLICIT, UNKNOWN, RawResponseArchive

SECRET = "sk-or-v1-" + ("FAKE" * 8) + "-NOTAREALKEY"
OK_BODY = {"id": "gen-ok", "model": "google/gemini-3.7-flash",
           "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "cost": 0.004},
           "choices": [{"finish_reason": "stop",
                        "message": {"content": '{"transcription": "שלום"}'}}]}
PIN = {"order": ["google-ai-studio"], "allow_fallbacks": False}


@pytest.fixture(autouse=True)
def _key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SECRET)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    """Transport retries back off; tests must not actually wait."""
    monkeypatch.setattr("autograder.backends.openai_compat.time.sleep", lambda *_: None)


def _backend(handler, *, retries=2, provider=PIN):
    eg = {"reasoning": {"effort": "low"}}
    if provider is not None:
        eg["provider"] = provider
    cfg = BackendConfig(backend="openrouter", model="google/gemini-3.7-flash",
                        base_url="https://openrouter.ai/api/v1",
                        api_key_env="OPENROUTER_API_KEY",
                        extra_generation=eg, transport_retries=retries)
    return OpenRouterBackend(cfg, transport=httpx.MockTransport(handler))


def _budget(hard_increment=0.12, L0=1.0):
    return CampaignBudget(campaign="C", experiment_sha256="x", starting_ledger_usd=L0,
                          warning_increment_usd=0.08, hard_increment_usd=hard_increment,
                          warn_usd=L0 + 0.08, hard_usd=L0 + hard_increment,
                          predicted_arm_costs={})


class Recorder:
    """Stands in for the runner's pre-send hook, recording each authorization."""

    def __init__(self, budget, ledger_now, worst):
        self.budget, self.ledger_now, self.worst = budget, ledger_now, worst
        self.authorized = []

    def __call__(self, *, attempt_id, retry_index, payload):
        self.budget.authorize_attempt(attempt_id=attempt_id, ledger_now=self.ledger_now,
                                      max_request_cost_usd=self.worst)
        self.authorized.append((attempt_id, retry_index))


# =============================================================================
# A. retryable failure then success
# =============================================================================

def test_A_retry_then_success_two_attempts_two_authorizations_two_archives(tmp_path):
    sent = []

    def handler(request):
        sent.append(request)
        if len(sent) == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(200, json=OK_BODY)

    b = _backend(handler)
    b.raw_archive = RawResponseArchive(tmp_path / "raw.jsonl")
    b.capture_case_id, b.capture_arm_id = "hc_e002_q1_r1", "arm-1"
    b.capture_logical_request_id = "arm-1::hc_e002_q1_r1::1"
    hook = Recorder(_budget(), ledger_now=1.0, worst=0.004)
    b.pre_send_hook = hook

    data = b._post_chat({"model": "m", "provider": PIN, "messages": []})
    assert data["id"] == "gen-ok"
    assert len(sent) == 2, "the transport really did retry"

    # two DISTINCT attempt ids, authorized before each send
    ids = [a for a, _ in hook.authorized]
    assert len(hook.authorized) == 2
    assert len(set(ids)) == 2, "each physical attempt needs its own identity"
    assert [r for _, r in hook.authorized] == [0, 1]

    # both raw responses retained and linked to those same ids
    recs = b.raw_archive.records()
    assert len(recs) == 2
    assert [r["http_status"] for r in recs] == [429, 200]
    assert [r["attempt_id"] for r in recs] == ids
    assert [r["retry_index"] for r in recs] == [0, 1]
    assert all(r["logical_request_id"] == "arm-1::hc_e002_q1_r1::1" for r in recs)
    assert all(r["raw_body"] for r in recs)


def test_A_billing_events_carry_the_same_attempt_ids(tmp_path):
    sent = []

    def handler(request):
        sent.append(request)
        return (httpx.Response(500, json={"error": "boom"}) if len(sent) == 1
                else httpx.Response(200, json=OK_BODY))

    b = _backend(handler)
    b.raw_archive = RawResponseArchive(tmp_path / "raw.jsonl")
    b.pre_send_hook = Recorder(_budget(), 1.0, 0.004)
    b._post_chat({"model": "m", "provider": PIN, "messages": []})

    ev_ids = [e.attempt_id for e in b.billing_events]
    raw_ids = [r["attempt_id"] for r in b.raw_archive.records()]
    assert ev_ids == raw_ids, "ledger rows and archive rows must join on attempt_id"
    assert [e.retry_index for e in b.billing_events] == [0, 1]
    # and the id survives into the ledger row payload
    assert all(e.as_dict()["attempt_id"] for e in b.billing_events)


# =============================================================================
# B. the retry would cross the hard limit -> refused BEFORE transmission
# =============================================================================

def test_B_retry_refused_before_transmission_when_budget_is_exhausted(tmp_path):
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    b = _backend(handler)
    b.raw_archive = RawResponseArchive(tmp_path / "raw.jsonl")
    # headroom for exactly ONE attempt of 0.07 against a 0.12 increment
    hook = Recorder(_budget(), ledger_now=1.0, worst=0.07)
    b.pre_send_hook = hook

    with pytest.raises(CampaignBudgetExceeded):
        b._post_chat({"model": "m", "provider": PIN, "messages": []})

    assert len(sent) == 1, "the transport must observe NO second request"
    assert len(hook.authorized) == 1
    assert hook.budget.reserved_usd == pytest.approx(0.07)


def test_B_retry_does_not_receive_a_fresh_allowance(tmp_path):
    """The retry is checked against ledger + OUTSTANDING RESERVATIONS, so it
    cannot be authorized against a balance that has not caught up."""
    b = _budget()
    b.authorize_attempt(attempt_id="a0", ledger_now=1.0, max_request_cost_usd=0.07)
    assert b.remaining_usd(1.0) == pytest.approx(0.05)
    with pytest.raises(CampaignBudgetExceeded):
        b.authorize_attempt(attempt_id="a1", ledger_now=1.0, max_request_cost_usd=0.07)


def test_B_reservations_are_not_double_counted_for_the_same_attempt():
    b = _budget()
    b.authorize_attempt(attempt_id="a0", ledger_now=1.0, max_request_cost_usd=0.05)
    b.authorize_attempt(attempt_id="a0", ledger_now=1.0, max_request_cost_usd=0.05)
    assert b.reserved_usd == pytest.approx(0.05), "re-authorizing must replace, not stack"


def test_B_settle_releases_reservations_once_the_ledger_has_them():
    b = _budget()
    b.authorize_attempt(attempt_id="a0", ledger_now=1.0, max_request_cost_usd=0.05)
    released = b.settle_all()
    assert released == pytest.approx(0.05) and b.reserved_usd == 0.0
    # ledger has now grown by the real (smaller) cost; headroom reflects that
    assert b.remaining_usd(1.02) == pytest.approx(0.10)


def test_B_unrelated_concurrent_spend_only_reduces_headroom():
    b = _budget()
    assert b.remaining_usd(1.0) == pytest.approx(0.12)
    assert b.remaining_usd(1.09) == pytest.approx(0.03)   # someone else spent
    with pytest.raises(CampaignBudgetExceeded):
        b.check(ledger_now=1.09, max_request_cost_usd=0.05)


# =============================================================================
# C. wrong provider on attempt 1, correct on attempt 2 -> still tainted
# =============================================================================

def test_C_earlier_route_violation_is_not_erased_by_a_later_correct_attempt(tmp_path):
    sent = []

    def handler(request):
        sent.append(request)
        if len(sent) == 1:                      # served by the WRONG provider, then 500
            return httpx.Response(500, json={**OK_BODY, "provider": "Google"})
        return httpx.Response(200, json={**OK_BODY, "provider": "Google AI Studio"})

    b = _backend(handler)
    b.raw_archive = RawResponseArchive(tmp_path / "raw.jsonl")
    b.pre_send_hook = Recorder(_budget(), 1.0, 0.004)
    b._post_chat({"model": "m", "provider": PIN, "messages": []})

    assert len(sent) == 2
    # the LAST check looks clean ...
    assert b.last_route_check["violation"] is False
    # ... but the sticky record and the per-attempt list still show the breach
    assert b.route_violation is not None
    assert b.route_violation["observed_provider"] == "Google"
    assert b.route_violation["retry_index"] == 0
    assert [r["violation"] for r in b.attempt_records] == [True, False]
    # and it is durable in the archive, not just in memory
    recs = b.raw_archive.records()
    assert recs[0]["route_check"]["violation"] is True
    assert recs[1]["route_check"]["violation"] is False


def test_C_a_tainted_arm_cannot_be_reported_as_clean(tmp_path):
    """The runner reads route_violation, not last_route_check — so a call that
    SUCCEEDED at the transport layer is still stopped."""
    sent = []

    def handler(request):
        sent.append(request)
        return (httpx.Response(500, json={**OK_BODY, "provider": "Google"}) if len(sent) == 1
                else httpx.Response(200, json={**OK_BODY, "provider": "Google AI Studio"}))

    b = _backend(handler)
    b.raw_archive = RawResponseArchive(tmp_path / "raw.jsonl")
    b.pre_send_hook = Recorder(_budget(), 1.0, 0.004)
    b._post_chat({"model": "m", "provider": PIN, "messages": []})
    ok_for_runner = b.route_violation is None
    assert ok_for_runner is False, "the arm must not be marked successful"


# =============================================================================
# D. UNKNOWN then correct -> UNKNOWN retained, no invented violation
# =============================================================================

def test_D_unknown_attribution_is_retained_and_invents_no_violation(tmp_path):
    sent = []

    def handler(request):
        sent.append(request)
        if len(sent) == 1:
            return httpx.Response(429, json={"error": {"message": "rate limited"}})
        return httpx.Response(200, json={**OK_BODY, "provider": "Google AI Studio"})

    b = _backend(handler)
    b.raw_archive = RawResponseArchive(tmp_path / "raw.jsonl")
    b.pre_send_hook = Recorder(_budget(), 1.0, 0.004)
    b._post_chat({"model": "m", "provider": PIN, "messages": []})

    recs = b.raw_archive.records()
    assert recs[0]["provider_attribution_status"] == UNKNOWN
    assert recs[0]["observed_provider"] is None
    assert recs[0]["route_check"]["violation"] is False, "UNKNOWN is not a breach"
    assert recs[1]["provider_attribution_status"] == EXPLICIT
    assert b.route_violation is None
    assert [r["provider_attribution_status"] for r in b.attempt_records] == [UNKNOWN, EXPLICIT]


# =============================================================================
# E. archive write failure -> fail closed, no further attempt
# =============================================================================

def test_E_archive_failure_stops_everything_and_rejects_the_parsed_output(tmp_path):
    sent = []

    def handler(request):
        sent.append(request)
        return httpx.Response(429, json={"error": {"message": "rate limited"}})

    b = _backend(handler)
    arch = RawResponseArchive(tmp_path / "raw.jsonl")
    arch.append = lambda rec: (_ for _ in ()).throw(OSError("disk full"))
    b.raw_archive = arch
    b.pre_send_hook = Recorder(_budget(), 1.0, 0.004)

    with pytest.raises(ArchiveFailure):
        b._post_chat({"model": "m", "provider": PIN, "messages": []})

    assert len(sent) == 1, "no retry may be sent after an archive failure"
    assert (tmp_path / "raw.jsonl.ARCHIVE_FAILURE").exists()


def test_E_a_successful_parse_is_not_returned_when_archiving_failed(tmp_path):
    b = _backend(lambda r: httpx.Response(200, json=OK_BODY), retries=0)
    arch = RawResponseArchive(tmp_path / "raw.jsonl")
    arch.append = lambda rec: (_ for _ in ()).throw(OSError("disk full"))
    b.raw_archive = arch
    with pytest.raises(ArchiveFailure):
        b.parse(system="s", content_blocks=[{"type": "text", "text": "t"}],
                output_model=_Transcription, max_tokens=100)


class _Transcription(__import__("pydantic").BaseModel):
    transcription: str


# =============================================================================
# F. correlation joins by identifier, never by order
# =============================================================================

def test_F_archive_route_and_ledger_rows_join_on_attempt_id(tmp_path):
    sent = []

    def handler(request):
        sent.append(request)
        return (httpx.Response(503, json={"error": "unavailable"}) if len(sent) == 1
                else httpx.Response(200, json={**OK_BODY, "provider": "Google AI Studio"}))

    b = _backend(handler)
    b.raw_archive = RawResponseArchive(tmp_path / "raw.jsonl")
    b.capture_campaign_id = "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1"
    b.capture_arm_id = "dev__smoke__all__google-gemini-3.7-flash__abc"
    b.capture_case_id = "hc_e002_q1_r1"
    b.capture_logical_request_id = "arm::case::1"
    b.pre_send_hook = Recorder(_budget(), 1.0, 0.004)
    b._post_chat({"model": "m", "provider": PIN, "messages": []})

    raw = {r["attempt_id"]: r for r in b.raw_archive.records()}
    route = {r["attempt_id"]: r for r in b.attempt_records}
    ledger = {e.attempt_id: e.as_dict() for e in b.billing_events}

    assert set(raw) == set(route) == set(ledger), "all three keyed by the same ids"
    assert len(raw) == 2 and all(k for k in raw)
    for aid in raw:
        assert raw[aid]["campaign_id"] == "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V1"
        assert raw[aid]["arm_id"] == b.capture_arm_id
        assert raw[aid]["logical_request_id"] == "arm::case::1"
        assert raw[aid]["case_hash"] and len(raw[aid]["case_hash"]) == 16
        assert raw[aid]["ledger_entry_id"] == aid
        assert route[aid]["attempt_id"] == aid
        assert ledger[aid]["retry_index"] == raw[aid]["retry_index"]


def test_F_join_does_not_depend_on_order_or_timestamps(tmp_path):
    sent = []

    def handler(request):
        sent.append(request)
        return (httpx.Response(502, json={"e": 1}) if len(sent) == 1
                else httpx.Response(200, json=OK_BODY))

    b = _backend(handler)
    b.raw_archive = RawResponseArchive(tmp_path / "raw.jsonl")
    b.pre_send_hook = Recorder(_budget(), 1.0, 0.004)
    b._post_chat({"model": "m", "provider": PIN, "messages": []})

    recs = list(reversed(b.raw_archive.records()))     # shuffle the order
    evs = {e.attempt_id: e for e in b.billing_events}
    for r in recs:                                      # still joins correctly
        assert evs[r["attempt_id"]].http_status == r["http_status"]
    # timestamps are not a key: they can legitimately collide within a second
    assert len({r["attempt_id"] for r in recs}) == 2


# =============================================================================
# G. route identity — the defect that stopped the live campaign on 2026-09-04
# =============================================================================

def test_G_provider_must_enter_the_route_fingerprint():
    """REGRESSION, currently FAILING BY DESIGN when unfixed.

    TaskRoute.fingerprint_fields() omitted `provider`, so a provider-pinned arm
    resolved the SAME config_hash and the SAME request-cache fingerprint as the
    unpinned configuration. In the live campaign this made a 'pinned' arm reuse
    5 cached responses produced by an earlier UNPINNED run, and gave the arm no
    distinct run identity.

    Two pins that name DIFFERENT providers must never share a fingerprint, and
    a pinned route must never share one with an unpinned route.
    """
    from autograder.gateway import TaskRoute
    base = dict(task="ocr_primary", backend="openrouter", model="google/gemini-3.7-flash")
    ai = TaskRoute(**base, provider={"order": ["google-ai-studio"], "allow_fallbacks": False})
    vx = TaskRoute(**base, provider={"order": ["google-vertex"], "allow_fallbacks": False})
    auto = TaskRoute(**base)

    from autograder.routeidentity import effective_config_fields, experiment_identity

    # The fix is STRUCTURAL: identity is derived from the effective backend
    # configuration, so the pin cannot escape by being absent from a list.
    eff = effective_config_fields(ai)
    assert eff["extra_generation"]["provider"]["order"] == ["google-ai-studio"]
    assert eff["extra_generation"]["provider"]["allow_fallbacks"] is False

    ids = {experiment_identity(r) for r in (ai, vx, auto)}
    assert len(ids) == 3, "unpinned, ai-studio-pinned and vertex-pinned must all differ"

    fb = TaskRoute(**base, provider={"order": ["google-ai-studio"], "allow_fallbacks": True})
    assert experiment_identity(fb) != experiment_identity(ai), \
        "changing allow_fallbacks must change route identity"


def test_G_request_cache_fingerprint_separates_pinned_routes():
    """The cache key is derived from the route fingerprint, so the same defect
    let an unpinned cached response satisfy a pinned request."""
    from autograder.gateway import TaskRoute
    from autograder.requestcache import fingerprint
    from pydantic import BaseModel

    class M(BaseModel):
        transcription: str

    base = dict(task="ocr_primary", backend="openrouter", model="google/gemini-3.7-flash")
    blocks = [{"type": "text", "text": "same"}]
    fp = lambda r: fingerprint(r, "sys", blocks, M, 1000, {})
    ai = TaskRoute(**base, provider={"order": ["google-ai-studio"], "allow_fallbacks": False})
    vx = TaskRoute(**base, provider={"order": ["google-vertex"], "allow_fallbacks": False})
    auto = TaskRoute(**base)
    assert fp(ai) != fp(vx), "ai-studio and vertex must not share a cache key"
    assert fp(ai) != fp(auto), "a pinned arm must not replay an unpinned cached response"
