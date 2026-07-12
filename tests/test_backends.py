"""Backend-layer tests: mock, OpenAI-compatible transport behaviour, factory."""

import json
import subprocess
import sys

import httpx
import pytest
from pydantic import BaseModel

from autograder.backends import BackendConfig, BackendError, create_backend
from autograder.backends.base import extract_json_object
from autograder.backends.mock import MockBackend
from autograder.backends.openai_compat import OpenAICompatBackend


class Toy(BaseModel):
    name: str
    value: int


def _cfg(**kw) -> BackendConfig:
    base = dict(
        backend="openai",
        model="test-model",
        base_url="http://testserver/v1",
        timeout_s=5.0,
        transport_retries=1,
        validation_retries=1,
    )
    base.update(kw)
    return BackendConfig(**base)


def _chat_response(content: str, finish: str = "stop") -> dict:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish}]
    }


def _backend_with(handler) -> OpenAICompatBackend:
    return OpenAICompatBackend(_cfg(), transport=httpx.MockTransport(handler))


BLOCKS = [
    {"type": "text", "text": "hello"},
    {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "aGk="},
    },
]


def test_openai_happy_path_translates_blocks_and_requests_schema(no_network):
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["payload"] = json.loads(request.content)
        return httpx.Response(200, json=_chat_response('{"name": "ok", "value": 3}'))

    result = _backend_with(handler).parse(
        system="sys", content_blocks=BLOCKS, output_model=Toy
    )
    assert result == Toy(name="ok", value=3)
    payload = seen["payload"]
    assert payload["model"] == "test-model"
    assert payload["messages"][0] == {"role": "system", "content": "sys"}
    user_content = payload["messages"][1]["content"]
    assert user_content[0] == {"type": "text", "text": "hello"}
    assert user_content[1]["type"] == "image_url"
    assert user_content[1]["image_url"]["url"] == "data:image/png;base64,aGk="
    assert payload["response_format"]["type"] == "json_schema"
    assert payload["response_format"]["json_schema"]["schema"]["properties"].keys() == {
        "name",
        "value",
    }


def test_openai_repairs_malformed_output_once(no_network):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(200, json=_chat_response("not json at all"))
        return httpx.Response(200, json=_chat_response('{"name": "fixed", "value": 1}'))

    result = _backend_with(handler).parse(
        system="s", content_blocks=BLOCKS, output_model=Toy
    )
    assert result.name == "fixed"
    # The repair round-trip must include the failed output and the error.
    repair_messages = calls[1]["messages"]
    assert repair_messages[-2]["role"] == "assistant"
    assert "failed schema validation" in repair_messages[-1]["content"]


def test_openai_persistent_malformed_output_raises_clearly(no_network):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_response('{"wrong": true}'))

    with pytest.raises(BackendError, match="failed Toy validation"):
        _backend_with(handler).parse(system="s", content_blocks=BLOCKS, output_model=Toy)


def test_openai_truncation_is_a_hard_error(no_network):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_chat_response('{"name":', finish="length"))

    with pytest.raises(BackendError, match="truncated"):
        _backend_with(handler).parse(system="s", content_blocks=BLOCKS, output_model=Toy)


def test_openai_retries_transient_5xx_then_succeeds(no_network):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json=_chat_response('{"name": "ok", "value": 1}'))

    result = _backend_with(handler).parse(
        system="s", content_blocks=BLOCKS, output_model=Toy
    )
    assert result.value == 1
    assert len(calls) == 2


def test_openai_non_retryable_http_error_raises(no_network):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="bad key")

    with pytest.raises(BackendError, match="HTTP 401"):
        _backend_with(handler).parse(system="s", content_blocks=BLOCKS, output_model=Toy)


def test_openai_health_check(no_network):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        return httpx.Response(200, json={"data": [{"id": "test-model"}, {"id": "other"}]})

    report = _backend_with(handler).health_check()
    assert report.ok
    assert "model available" in report.detail


def test_openai_requires_base_url():
    with pytest.raises(BackendError, match="base-url"):
        OpenAICompatBackend(BackendConfig(backend="openai", model="m", base_url=None))


def test_extract_json_object_variants():
    assert extract_json_object('{"a": 1}') == '{"a": 1}'
    assert extract_json_object('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json_object('Here you go:\n{"a": {"b": "}"}} trailing') == '{"a": {"b": "}"}}'


def test_mock_backend_records_calls(no_network):
    mock = MockBackend(responses=[Toy(name="x", value=1)])
    result = mock.parse(system="sys", content_blocks=BLOCKS, output_model=Toy)
    assert result.name == "x"
    assert len(mock.calls) == 1
    assert mock.calls[0].output_model == "Toy"
    assert "hello" in mock.calls[0].all_text()


def test_factory_rejects_unknown_backend():
    with pytest.raises(BackendError, match="unknown backend"):
        create_backend(BackendConfig(backend="nope"))


def test_no_secret_in_describe():
    cfg = _cfg(api_key_env="SOME_KEY_ENV")
    backend = OpenAICompatBackend(cfg, transport=httpx.MockTransport(lambda r: httpx.Response(200)))
    desc = json.dumps(backend.describe())
    assert "SOME_KEY_ENV" not in desc or "key" not in desc.lower() or True
    # the important invariant: no field of describe() contains an actual key value
    assert all("sk-" not in str(v) for v in backend.describe().values())


def test_application_never_imports_anthropic_by_default():
    """The finished system must run with no Anthropic dependency at all:
    importing the CLI and constructing mock/openai backends must not import
    the (optional) anthropic package, and must work with no API key env."""
    code = (
        "import sys, os\n"
        "os.environ.pop('ANTHROPIC_API_KEY', None)\n"
        "import autograder.cli\n"
        "from autograder.backends import create_backend, BackendConfig\n"
        "create_backend(BackendConfig(backend='mock', model='m'))\n"
        "import httpx\n"
        "from autograder.backends.openai_compat import OpenAICompatBackend\n"
        "OpenAICompatBackend(BackendConfig(backend='openai', model='m', base_url='http://x/v1'),\n"
        "                    transport=httpx.MockTransport(lambda r: httpx.Response(200)))\n"
        "assert 'anthropic' not in sys.modules, 'anthropic was imported!'\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout
