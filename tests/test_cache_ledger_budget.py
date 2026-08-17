"""Request cache, usage ledger, budget manager — offline tests."""

from __future__ import annotations

import base64
import json

import pytest
from pydantic import BaseModel

from autograder.backends import BackendConfig
from autograder.backends.mock import MockBackend
from autograder.gateway import ModelGateway
from autograder.requestcache import RequestCache, fingerprint
from autograder.usage import BudgetExceeded, BudgetLimits, BudgetManager, UsageLedger


class Out(BaseModel):
    text: str


def _img(payload: bytes):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                        "data": base64.b64encode(payload).decode()}}


def _gw(tmp_path, responses, **kw):
    counter = {"calls": 0}

    def factory(cfg: BackendConfig):
        def responder(model, system, blocks):
            counter["calls"] += 1
            return responses.pop(0)
        return MockBackend(config=cfg, responder=responder)

    gw = ModelGateway.from_dict(
        {"models": {"grade_primary": {"backend": "openrouter", "model": "vendor/m", "prompt_version": "p1"}}},
        backend_factory=factory, cache=RequestCache(tmp_path / "cache"), **kw)
    return gw, counter


# -------------------------------------------------------------------- cache ----


def test_identical_request_zero_second_provider_call(tmp_path):
    gw, counter = _gw(tmp_path, [Out(text="a"), Out(text="SHOULD-NOT-BE-USED")])
    args = dict(task="grade_primary", system="s", content_blocks=[{"type": "text", "text": "q"}, _img(b"img1")],
                output_model=Out)
    r1 = gw.call(**args)
    r2 = gw.call(**args)
    assert r1.cache_hit is False and r2.cache_hit is True
    assert r2.value.text == "a" and counter["calls"] == 1
    assert gw.cache.stats()["hits"] == 1


def test_changed_model_prompt_image_or_pack_invalidates(tmp_path):
    base_route = ModelGateway.from_dict(
        {"models": {"t": {"backend": "mock", "model": "m1", "prompt_version": "p1", "max_tokens": 100}}},
        backend_factory=lambda c: MockBackend(config=c)).route("t")
    blocks = [{"type": "text", "text": "q"}, _img(b"img1")]
    fp0 = fingerprint(base_route, "sys", blocks, Out, None, {"pack_hash": "P1"})
    # same everything -> same fp
    assert fingerprint(base_route, "sys", blocks, Out, None, {"pack_hash": "P1"}) == fp0
    # model
    r_model = ModelGateway.from_dict(
        {"models": {"t": {"backend": "mock", "model": "m2", "prompt_version": "p1", "max_tokens": 100}}},
        backend_factory=lambda c: MockBackend(config=c)).route("t")
    assert fingerprint(r_model, "sys", blocks, Out, None, {"pack_hash": "P1"}) != fp0
    # prompt text / prompt version
    assert fingerprint(base_route, "sys2", blocks, Out, None, {"pack_hash": "P1"}) != fp0
    r_pv = ModelGateway.from_dict(
        {"models": {"t": {"backend": "mock", "model": "m1", "prompt_version": "p2", "max_tokens": 100}}},
        backend_factory=lambda c: MockBackend(config=c)).route("t")
    assert fingerprint(r_pv, "sys", blocks, Out, None, {"pack_hash": "P1"}) != fp0
    # image
    assert fingerprint(base_route, "sys", [{"type": "text", "text": "q"}, _img(b"img2")], Out, None, {"pack_hash": "P1"}) != fp0
    # question pack
    assert fingerprint(base_route, "sys", blocks, Out, None, {"pack_hash": "P2"}) != fp0
    # decoding config
    r_tok = ModelGateway.from_dict(
        {"models": {"t": {"backend": "mock", "model": "m1", "prompt_version": "p1", "max_tokens": 200}}},
        backend_factory=lambda c: MockBackend(config=c)).route("t")
    assert fingerprint(r_tok, "sys", blocks, Out, None, {"pack_hash": "P1"}) != fp0


def test_failures_never_enter_cache(tmp_path):
    from autograder.backends import BackendError

    class Boom(MockBackend):
        def parse(self, **kw):
            raise BackendError("transient")

    gw = ModelGateway.from_dict({"models": {"t": {"backend": "openrouter", "model": "m"}}},
                                backend_factory=lambda c: Boom(config=c),
                                cache=RequestCache(tmp_path / "c"))
    with pytest.raises(BackendError):
        gw.call(task="t", system="s", content_blocks=[{"type": "text", "text": "x"}], output_model=Out)
    assert not list((tmp_path / "c").rglob("*.json"))


# ------------------------------------------------------------------- ledger ----


def test_ledger_records_and_aggregates(tmp_path):
    led = UsageLedger(tmp_path / "ledger.jsonl")
    led.record({"ts": "2026-08-17 10:00:00", "task": "grade_primary", "backend": "openrouter", "model": "m",
                "cache_hit": False, "job_id": "j", "exam_id": "e1", "question_id": "1", "stage": "grade",
                "input_tokens": 100, "output_tokens": 20, "total_tokens": 120, "reported_cost": 0.001,
                "api_key": "MUST-NOT-PERSIST"})
    led.record({"ts": "2026-08-17 10:00:05", "task": "grade_primary", "backend": "openrouter", "model": "m",
                "cache_hit": True, "job_id": "j", "exam_id": "e1", "question_id": "1", "stage": "grade"})
    led.record({"ts": "2026-08-17 10:00:09", "task": "mc_resolve", "backend": "ollama", "model": "q",
                "cache_hit": False, "job_id": "j", "exam_id": "e2", "question_id": "2", "stage": "mc"})
    led.record({"ts": "2026-08-17 10:00:12", "task": "grade_escalate", "backend": "openrouter", "model": "m2",
                "cache_hit": False, "job_id": "j", "exam_id": "e3", "question_id": "1", "stage": "escalation",
                "input_tokens": 300, "output_tokens": 50, "total_tokens": 350, "reported_cost": 0.01})
    raw = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    assert "MUST-NOT-PERSIST" not in raw
    a = led.aggregate("j")
    assert a["cloud_requests"] == 2 and a["cloud_cache_hits"] == 1
    assert a["cache_hit_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert a["total_tokens"] == 470 and a["reported_cost"] == pytest.approx(0.011)
    assert a["exams"] == 3
    assert a["pct_exams_fully_local"] == pytest.approx(100 * 1 / 3, abs=0.1)   # e2 local only
    assert a["pct_exams_cloud_escalated"] == pytest.approx(100 * 1 / 3, abs=0.1)  # e3


# ------------------------------------------------------------------- budget ----


def _route(backend="openrouter"):
    from autograder.gateway import TaskRoute
    return TaskRoute(task="t", backend=backend, model="m")


def test_budget_soft_then_hard(tmp_path):
    warns = []
    bm = BudgetManager(BudgetLimits(max_calls_per_job=5), warn=warns.append)
    meta = {"job_id": "j", "exam_id": "e"}
    for i in range(5):                                           # exactly 5 allowed
        bm.check(task="t", route=_route(), meta=meta)
        bm.charge(task="t", route=_route(), usage={}, meta=meta)
    assert warns and "soft budget calls_per_job" in warns[0]   # warned at 80% (4/5)
    with pytest.raises(BudgetExceeded):
        bm.check(task="t", route=_route(), meta=meta)            # 6th call = hard
    assert bm.paused and "calls_per_job" in bm.pause_reason


def test_budget_ignores_local_and_paused_gateway_keeps_results(tmp_path):
    bm = BudgetManager(BudgetLimits(max_calls_per_job=1))
    meta = {"job_id": "j", "exam_id": "e"}
    bm.check(task="mc_resolve", route=_route("ollama"), meta=meta)   # local: not counted
    bm.charge(task="mc_resolve", route=_route("ollama"), usage={}, meta=meta)
    bm.check(task="t", route=_route(), meta=meta)
    bm.charge(task="t", route=_route(), usage={"input_tokens": 5}, meta=meta)
    with pytest.raises(BudgetExceeded):
        bm.check(task="t", route=_route(), meta=meta)
    # gateway integration: first call succeeds & persists in cache; second raises; cached result still served
    gw, counter = _gw(tmp_path, [Out(text="kept")],
                      budget=BudgetManager(BudgetLimits(max_calls_per_job=1)))
    args = dict(task="grade_primary", system="s", content_blocks=[{"type": "text", "text": "q"}], output_model=Out,
                meta={"job_id": "j", "exam_id": "e"})
    assert gw.call(**args).value.text == "kept"
    assert gw.call(**args).cache_hit is True                     # cache hit needs no budget
    with pytest.raises(BudgetExceeded):
        gw.call(task="grade_primary", system="s", content_blocks=[{"type": "text", "text": "OTHER"}],
                output_model=Out, meta={"job_id": "j", "exam_id": "e"})
    assert counter["calls"] == 1                                 # never silently downgraded/retried


# ------------------------------------------------- [budget] config wiring ----


def test_budget_limits_from_config_semantics():
    L = BudgetLimits.from_config({"enabled": True, "max_calls_per_job": 5, "max_input_tokens_per_job": 0,
                                  "max_output_tokens_per_job": 1000, "max_cost_per_job": 0.5,
                                  "max_calls_per_day": 0, "soft_fraction": 0.5})
    assert L.max_calls_per_job == 5 and L.max_input_tokens is None          # 0 = unlimited
    assert L.max_output_tokens == 1000 and L.max_cost == 0.5 and L.max_calls_per_day is None
    assert L.soft_fraction == 0.5
    assert BudgetLimits.from_config({"enabled": False, "max_calls_per_job": 1}) is None
    assert BudgetLimits.from_config(None) is None
    with pytest.raises(ValueError):
        BudgetLimits.from_config({"bogus_limit": 3})
    eff = L.effective()
    assert eff["max_input_tokens"] == "—" and eff["max_calls_per_job"] == 5


def test_setup_from_config_wires_budget_and_ui_summary(tmp_path, monkeypatch):
    from autograder.orchestrator import setup_from_config
    from autograder.reviewui import settings_summary
    cfg = tmp_path / "models.toml"
    cfg.write_text('[models.grade_primary]\nbackend="openrouter"\nmodel="vendor/m"\n'
                   '[budget]\nenabled=true\nmax_calls_per_job=2\nmax_cost_per_job=0\nsoft_fraction=0.5\n',
                   encoding="utf-8")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-SECRET")
    calls = {"n": 0}

    def factory(c):
        def responder(model, system, blocks):
            calls["n"] += 1
            return Out(text="ok")
        return MockBackend(config=c, responder=responder)

    rt = setup_from_config(cfg, tmp_path / "state", backend_factory=factory)
    assert rt.gateway.budget is rt.budget and rt.budget.limits.max_calls_per_job == 2
    assert rt.budget.limits.max_cost is None
    meta = {"job_id": "j", "exam_id": "e"}
    rt.gateway.call(task="grade_primary", system="s", content_blocks=[{"type": "text", "text": "a"}], output_model=Out, meta=meta)
    rt.gateway.call(task="grade_primary", system="s", content_blocks=[{"type": "text", "text": "b"}], output_model=Out, meta=meta)
    with pytest.raises(BudgetExceeded):
        rt.gateway.call(task="grade_primary", system="s", content_blocks=[{"type": "text", "text": "c"}], output_model=Out, meta=meta)
    assert calls["n"] == 2 and rt.warnings   # soft warning fired at 50%
    s = settings_summary(gateway=rt.gateway, ledger=rt.ledger, budget=rt.budget, cache=rt.cache,
                         openrouter_key_present=True)
    assert s["budget"]["limits"]["max_calls_per_job"] == 2 and s["budget"]["paused"] is True
    assert "SECRET" not in json.dumps(s)
