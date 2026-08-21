"""The OpenRouter/cloud gateway boundary (effective-provider classification).

The audit found two bypasses: (1) the legacy direct-backend path could carry
OpenRouter traffic with no privacy scan/cache/budget/ledger; (2) budget
classification keyed on the backend NAME, so backend="openai" pointed at
openrouter.ai escaped enforcement. These tests pin the fixes. No network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from autograder.backends import BackendConfig, BackendError
from autograder.backends.mock import MockBackend
from autograder.cli import build_parser, guard_direct_cloud_backend, main
from autograder.escalation import GradeResult
from autograder.gateway import ModelGateway
from autograder.key_parser import save_answer_key
from autograder.privacy import PrivacyError
from autograder.requestcache import RequestCache
from autograder.usage import (BudgetExceeded, BudgetLimits, BudgetManager, UsageLedger,
                              effective_provider, is_cloud_route)
from tests.test_grade import make_key

OPENROUTER_URL = "https://openrouter.ai/api/v1"
LOCAL_URL = "http://localhost:11434/v1"
REMOTE_OPENAI_URL = "https://api.groq.com/openai/v1"


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------


def test_effective_provider_sees_through_the_backend_name():
    assert effective_provider("openrouter", None) == "openrouter"
    assert effective_provider("openai", OPENROUTER_URL) == "openrouter"
    assert effective_provider("openai", "https://OpenRouter.ai/api/v1") == "openrouter"
    assert effective_provider("openai", LOCAL_URL) == "openai"
    assert effective_provider("openai", REMOTE_OPENAI_URL) == "openai"
    assert effective_provider("ollama_native", LOCAL_URL) == "ollama"


def test_cloud_classification_by_effective_configuration():
    assert is_cloud_route("openrouter", None)
    assert is_cloud_route("openai", OPENROUTER_URL)          # the audit's bypass
    assert is_cloud_route("anthropic", None)
    assert not is_cloud_route("openai", LOCAL_URL)           # local Ollama/vLLM
    assert not is_cloud_route("openai", "http://192.168.1.20:8000/v1")
    assert not is_cloud_route("ollama_native", LOCAL_URL)
    assert not is_cloud_route("mock", None)
    # unknown REMOTE openai-compatible endpoints stay inside accounting
    assert is_cloud_route("openai", REMOTE_OPENAI_URL)


# --------------------------------------------------------------------------
# budget applies to both OpenRouter forms
# --------------------------------------------------------------------------


@dataclass
class _Route:
    backend: str
    base_url: str | None = None


def _manager() -> BudgetManager:
    return BudgetManager(BudgetLimits(max_calls_per_job=1), ledger=None, warn=lambda m: None)


@pytest.mark.parametrize("route", [_Route("openrouter"), _Route("openai", OPENROUTER_URL)])
def test_budget_counts_both_openrouter_forms(route):
    bm = _manager()
    meta = {"job_id": "j", "exam_id": "e"}
    bm.check(task="grade_primary", route=route, meta=meta)
    bm.charge(task="grade_primary", route=route, usage={}, meta=meta)
    with pytest.raises(BudgetExceeded):
        bm.check(task="grade_primary", route=route, meta=meta)


def test_budget_ignores_local_openai_compatible_routes():
    bm = _manager()
    meta = {"job_id": "j", "exam_id": "e"}
    for _ in range(5):  # far beyond the 1-call job limit: local calls are free
        bm.check(task="mc_resolve", route=_Route("openai", LOCAL_URL), meta=meta)
        bm.charge(task="mc_resolve", route=_Route("openai", LOCAL_URL), usage={}, meta=meta)


# --------------------------------------------------------------------------
# gateway: privacy + ledger + budget wrap both OpenRouter forms identically
# --------------------------------------------------------------------------


def _gateway(tmp_path, route_spec: dict):
    """A gateway whose single 'grade_primary' route LOOKS cloud but is served
    by a mock backend (no network)."""
    def factory(cfg):
        return MockBackend(config=BackendConfig(backend="mock", model="m"),
                           responder=lambda model, system, blocks: GradeResult(score=1.0))

    gw = ModelGateway.from_dict({"models": {"grade_primary": route_spec}},
                                backend_factory=factory,
                                cache=RequestCache(tmp_path / "cache"),
                                ledger=UsageLedger(tmp_path / "usage.jsonl"))
    gw.budget = BudgetManager(BudgetLimits(max_calls_per_job=1), ledger=None,
                              warn=lambda m: None)
    return gw


@pytest.mark.parametrize("route_spec", [
    {"backend": "openrouter", "model": "vendor/model"},
    {"backend": "openai", "base_url": OPENROUTER_URL, "model": "vendor/model"},
])
def test_gateway_wraps_both_openrouter_forms(tmp_path, route_spec):
    gw = _gateway(tmp_path, route_spec)

    # privacy: a forbidden identity key aborts BEFORE any provider work
    with pytest.raises(PrivacyError):
        gw.call(task="grade_primary", system="s",
                content_blocks=[{"type": "text", "text": "hi", "student_name": "Dana"}],
                output_model=GradeResult, meta={"job_id": "j", "exam_id": "e"})

    res = gw.call(task="grade_primary", system="s",
                  content_blocks=[{"type": "text", "text": "grade this"}],
                  output_model=GradeResult, meta={"job_id": "j", "exam_id": "e"})
    assert res.value.score == 1.0 and not res.cache_hit

    # ledger: the entry is classified by the EFFECTIVE provider
    rows = gw.ledger.entries()
    assert rows and rows[-1]["effective_provider"] == "openrouter"
    assert rows[-1]["cloud"] is True

    # cache: the identical request is served locally, no budget consumed
    res2 = gw.call(task="grade_primary", system="s",
                   content_blocks=[{"type": "text", "text": "grade this"}],
                   output_model=GradeResult, meta={"job_id": "j", "exam_id": "e"})
    assert res2.cache_hit

    # budget: the 1-call job limit now blocks the next NEW request
    with pytest.raises(BudgetExceeded):
        gw.call(task="grade_primary", system="s",
                content_blocks=[{"type": "text", "text": "a different request"}],
                output_model=GradeResult, meta={"job_id": "j", "exam_id": "e"})


def test_genuine_local_openai_route_is_not_reclassified(tmp_path):
    gw = _gateway(tmp_path, {"backend": "openai", "base_url": LOCAL_URL, "model": "qwen"})
    gw.call(task="grade_primary", system="s",
            content_blocks=[{"type": "text", "text": "grade this"}],
            output_model=GradeResult, meta={"job_id": "j", "exam_id": "e"})
    row = gw.ledger.entries()[-1]
    assert row["effective_provider"] == "openai" and row["cloud"] is False


# --------------------------------------------------------------------------
# the legacy direct-backend path refuses OpenRouter entirely
# --------------------------------------------------------------------------


def test_guard_rejects_explicit_openrouter_backend():
    with pytest.raises(BackendError, match="task gateway"):
        guard_direct_cloud_backend(BackendConfig(backend="openrouter", model="vendor/m"))


def test_guard_rejects_openai_compatible_openrouter_url():
    with pytest.raises(BackendError, match="task gateway"):
        guard_direct_cloud_backend(
            BackendConfig(backend="openai", model="vendor/m", base_url=OPENROUTER_URL))


def test_guard_allows_local_openai_compatible_backends():
    guard_direct_cloud_backend(
        BackendConfig(backend="openai", model="qwen", base_url=LOCAL_URL))
    guard_direct_cloud_backend(BackendConfig(backend="mock", model="m"))


def _key_json(tmp_path):
    key_path = tmp_path / "answer_key.json"
    save_answer_key(make_key(), key_path)
    return key_path


def test_cli_grade_refuses_openrouter_via_flags(tmp_path):
    rc = main(["grade", "--exam", str(tmp_path / "missing.pdf"),
               "--key", str(_key_json(tmp_path)), "--out", str(tmp_path / "out"),
               "--backend", "openai", "--base-url", OPENROUTER_URL])
    assert rc == 2   # refused before any backend construction or file access


def test_cli_refuses_openrouter_via_config_toml(tmp_path):
    cfg = tmp_path / "grader.toml"
    cfg.write_text('[backend]\nbackend = "openrouter"\nmodel = "vendor/m"\n',
                   encoding="utf-8")
    rc = main(["parse-key", "--key", str(_key_json(tmp_path)),
               "--out", str(tmp_path / "out"), "--config", str(cfg)])
    assert rc == 2
