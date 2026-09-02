"""Model gateway + OpenRouter backend — offline tests (MockTransport only).

No network, no credits: every OpenRouter "call" hits an httpx.MockTransport.
"""

from __future__ import annotations

import base64
import json

import httpx
import pytest
from pydantic import BaseModel

from autograder.backends import BackendConfig, BackendError
from autograder.backends.mock import MockBackend
from autograder.backends.openrouter import OpenRouterBackend
from autograder.gateway import GatewayConfigError, ModelGateway, TaskRoute
from autograder.cloudboundary import research_authorization


class Out(BaseModel):
    text: str


PNG = base64.b64encode(b"\x89PNG\r\n\x1a\nfakepng").decode()
IMG = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}}


# ---------------------------------------------------------------- gateway ----


def test_gateway_routes_task_to_configured_backend(monkeypatch):
    seen = []

    def factory(cfg: BackendConfig):
        seen.append(cfg)
        return MockBackend(config=cfg, responses=[Out(text="ok")])

    gw = ModelGateway.from_dict({
        "models": {"grade_primary": {"backend": "mock", "model": "m-grade"},
                   "mc_resolve": {"backend": "mock", "model": "m-mc"}}},
        backend_factory=factory)
    res = gw.call(task="grade_primary", system="s", content_blocks=[{"type": "text", "text": "t"}],
                  output_model=Out)
    assert res.value.text == "ok" and res.cache_hit is False
    assert seen[-1].model == "m-grade"
    assert gw.route("mc_resolve").model == "m-mc"


def test_gateway_env_expansion_and_missing_model_fails_loudly(monkeypatch):
    monkeypatch.setenv("GRADE_PRIMARY_MODEL", "vendor/some-model")
    monkeypatch.delenv("OCR_PRIMARY_MODEL", raising=False)
    gw = ModelGateway.from_dict({"models": {"grade_primary": {"backend": "mock", "model": "${GRADE_PRIMARY_MODEL}"}}},
                                backend_factory=lambda c: MockBackend(config=c))
    assert gw.route("grade_primary").model == "vendor/some-model"
    with pytest.raises(GatewayConfigError):
        ModelGateway.from_dict({"models": {"ocr_primary": {"backend": "mock", "model": "${OCR_PRIMARY_MODEL}"}}},
                               backend_factory=lambda c: MockBackend(config=c))


def test_gateway_unknown_task_and_unknown_key():
    gw = ModelGateway.from_dict({"models": {"a": {"backend": "mock", "model": "m"}}},
                                backend_factory=lambda c: MockBackend(config=c))
    with pytest.raises(GatewayConfigError):
        gw.route("nope")
    with pytest.raises(GatewayConfigError):
        ModelGateway.from_dict({"models": {"a": {"backend": "mock", "model": "m", "bogus": 1}}})


def test_gateway_openai_requires_base_url_and_ollama_routes_native():
    with pytest.raises(GatewayConfigError):
        ModelGateway.from_dict({"models": {"mc_resolve": {"backend": "openai", "model": "q"}}})
    seen = {}

    def factory(cfg):
        seen["backend"] = cfg.backend
        seen["eg"] = cfg.extra_generation
        return MockBackend(config=cfg, responses=[Out(text="ok")])

    gw = ModelGateway.from_dict({"models": {"mc_resolve": {"backend": "ollama", "model": "q",
                                                            "extra_generation": {"think": False}}}},
                                backend_factory=factory)
    gw.call(task="mc_resolve", system="s", content_blocks=[{"type": "text", "text": "t"}], output_model=Out)
    assert seen["backend"] == "ollama_native" and seen["eg"]["think"] is False


def test_ollama_native_backend_sends_think_false_and_format(monkeypatch):
    from autograder.backends.ollama_native import OllamaNativeBackend
    captured = {}

    def handler(req: httpx.Request):
        captured["path"] = req.url.path
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json={"model": "q", "done_reason": "stop", "prompt_eval_count": 12,
                                         "eval_count": 5, "message": {"role": "assistant",
                                                                      "content": '{"text": "hi"}',
                                                                      "thinking": ""}})

    be = OllamaNativeBackend(BackendConfig(backend="ollama_native", model="q", max_tokens=99,
                                           extra_generation={"think": False, "options": {"num_ctx": 8192}}),
                             transport=httpx.MockTransport(handler))
    out = be.parse(system="SYS", content_blocks=[{"type": "text", "text": "hello"}, IMG], output_model=Out)
    assert out.text == "hi" and captured["path"] == "/api/chat"
    b = captured["body"]
    assert b["think"] is False and b["options"]["num_ctx"] == 8192 and b["options"]["num_predict"] == 99
    assert b["format"]["type"] == "object" and b["messages"][1]["images"]
    assert be.last_usage["input_tokens"] == 12 and be.last_usage["thinking_chars"] == 0


def test_gateway_never_hardcodes_models():
    import inspect
    from autograder import gateway
    src = inspect.getsource(gateway)
    for banned in ("claude-", "gemini-", "gpt-", "openrouter.ai/api/v1/chat"):
        assert banned not in src


# ------------------------------------------------------------- openrouter ----


def _or_backend(monkeypatch, handler, **cfg):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-TESTKEY-123")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    c = BackendConfig(backend="openrouter", model="vendor/model", max_tokens=200,
                      transport_retries=2, validation_retries=1, **cfg)
    return OpenRouterBackend(c, transport=httpx.MockTransport(handler))


def _ok_body(text='{"text": "hi"}', **usage):
    u = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "cost": 0.0001,
         "completion_tokens_details": {"reasoning_tokens": 0},
         "prompt_tokens_details": {"cached_tokens": 4}}
    u.update(usage)
    return {"id": "gen-abc", "provider": "SomeProvider", "model": "vendor/model",
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}], "usage": u}


def test_openrouter_requires_env_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(BackendError):
        OpenRouterBackend(BackendConfig(backend="openrouter", model="x"))


def test_openrouter_text_image_serialization_and_headers(monkeypatch):
    captured = {}

    def handler(req: httpx.Request):
        captured["headers"] = dict(req.headers)
        captured["body"] = json.loads(req.content)
        return httpx.Response(200, json=_ok_body())

    be = _or_backend(monkeypatch, handler,
                     extra_generation={"reasoning": {"effort": "low"},
                                       "provider": {"order": ["A", "B"]}})
    out = be.parse(system="SYS", content_blocks=[{"type": "text", "text": "hello"}, IMG],
                   output_model=Out)
    assert out.text == "hi"
    b = captured["body"]
    assert b["model"] == "vendor/model" and b["reasoning"] == {"effort": "low"}
    assert b["provider"] == {"order": ["A", "B"]} and b["usage"] == {"include": True}
    user = b["messages"][1]["content"]
    assert user[0] == {"type": "text", "text": "hello"}
    assert user[1]["type"] == "image_url" and user[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert b["response_format"]["type"] == "json_schema"
    h = captured["headers"]
    assert h["authorization"] == "Bearer sk-or-TESTKEY-123"
    assert "x-title" in h and "http-referer" in h
    # usage captured, key never in describe()
    assert be.last_usage["input_tokens"] == 10 and be.last_usage["cached_input_tokens"] == 4
    assert be.last_usage["reported_cost"] == 0.0001 and be.last_usage["request_id"] == "gen-abc"
    assert "TESTKEY" not in json.dumps(be.describe())


def test_openrouter_429_retry_after_then_success(monkeypatch):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, text="slow down")
        return httpx.Response(200, json=_ok_body())

    monkeypatch.setattr("time.sleep", lambda s: None)
    be = _or_backend(monkeypatch, handler)
    assert be.parse(system="s", content_blocks=[{"type": "text", "text": "x"}], output_model=Out).text == "hi"
    assert calls["n"] == 2


def test_openrouter_retries_exhausted_raise_without_key_leak(monkeypatch):
    def handler(req):
        return httpx.Response(503, text="unavailable sk-or-TESTKEY-123")

    monkeypatch.setattr("time.sleep", lambda s: None)
    be = _or_backend(monkeypatch, handler)
    with pytest.raises(BackendError) as ei:
        be.parse(system="s", content_blocks=[{"type": "text", "text": "x"}], output_model=Out)
    assert "TESTKEY" not in str(ei.value)


def test_openrouter_timeout_is_backend_error(monkeypatch):
    def handler(req):
        raise httpx.ReadTimeout("slow")

    be = _or_backend(monkeypatch, handler)
    with pytest.raises(BackendError):
        be.parse(system="s", content_blocks=[{"type": "text", "text": "x"}], output_model=Out)


def test_openrouter_truncation_and_validation_repair(monkeypatch):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, json=_ok_body(text='{"wrong": 1}'))
        return httpx.Response(200, json=_ok_body())

    be = _or_backend(monkeypatch, handler)
    assert be.parse(system="s", content_blocks=[{"type": "text", "text": "x"}], output_model=Out).text == "hi"
    assert calls["n"] == 2  # one repair round-trip

    def trunc(req):
        body = _ok_body()
        body["choices"][0]["finish_reason"] = "length"
        return httpx.Response(200, json=body)

    be2 = _or_backend(monkeypatch, trunc)
    with pytest.raises(BackendError):
        be2.parse(system="s", content_blocks=[{"type": "text", "text": "x"}], output_model=Out)


def test_openrouter_via_gateway_records_usage(monkeypatch):
    def handler(req):
        return httpx.Response(200, json=_ok_body())

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-TESTKEY-123")
    entries = []

    class Ledger:
        def record(self, e):
            entries.append(e)

    def factory(cfg):
        return OpenRouterBackend(cfg, transport=httpx.MockTransport(handler))

    gw = ModelGateway.from_dict({"models": {"grade_primary": {"backend": "openrouter", "model": "vendor/model"}}},
                                backend_factory=factory, ledger=Ledger(),
                                execution_mode="research",  # usage-recording vehicle
                                research_auth=research_authorization(
                                    "test:gateway-usage", tasks=["grade_primary"],
                                    models=["vendor/model"]))
    res = gw.call(task="grade_primary", system="s", content_blocks=[{"type": "text", "text": "x"}],
                  output_model=Out, meta={"job_id": "j1", "exam_id": "e1", "question_id": "3", "stage": "grade"})
    assert res.usage["total_tokens"] == 15
    assert entries[0]["job_id"] == "j1" and entries[0]["total_tokens"] == 15 and entries[0]["cache_hit"] is False
    assert "TESTKEY" not in json.dumps(entries)
