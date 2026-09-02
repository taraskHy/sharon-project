"""Disguised remote routes vs the production cloud boundary (reverification).

Extends tests/test_cloud_boundary.py with the hostname-dressing matrix: DNS
names that merely LOOK local (10.0.0.1.evil.example, localhost.evil.example)
must classify as REMOTE, and every remote grading route must be refused
BEFORE any request serialization or network access. Includes the regression
pin for the RFC1918-prefix fix in usage._is_local_url. No network call.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autograder.cloudboundary import (CloudBoundaryError, check_cloud_call,
                                      is_remote_route)  # noqa: E402
from autograder.usage import _is_local_url  # noqa: E402

GRADING_TASKS = ("grade_primary", "grade_escalate", "grading_rag",
                 "mc_resolve_cloud", "variant_resolve_cloud",
                 "align_resolve_cloud", "totally_new_role")

DISGUISED_REMOTE_URLS = (
    # hostname dressing around private-range prefixes (the regression)
    "http://10.0.0.1.evil.example:11434/v1",
    "http://192.168.1.1.evil.example/v1",
    "http://172.16.0.1.attacker.example/v1",
    # localhost dressing
    "http://localhost.evil.example:11434/v1",
    "http://127.0.0.1.evil.example/v1",
    # plainly remote endpoints under innocuous ports/paths
    "https://api.evil.example:11434/v1",
    "https://openrouter.ai/api/v1",
    "https://api.anthropic.com/v1",
)

GENUINELY_LOCAL_URLS = (
    "http://localhost:11434/v1",
    "http://127.0.0.1:11434/v1",
    "http://[::1]:11434/v1",
    "http://10.0.0.7:11434/v1",
    "http://192.168.1.20:11434/v1",
    "http://172.31.255.1:11434/v1",
    "http://gpu-box.local:11434/v1",
)


@pytest.mark.parametrize("url", DISGUISED_REMOTE_URLS)
def test_dressed_hostnames_classify_as_remote(url):
    assert not _is_local_url(url), url
    assert is_remote_route("ollama", url), url          # backend name is no cover
    assert is_remote_route("openai", url), url


@pytest.mark.parametrize("url", GENUINELY_LOCAL_URLS)
def test_genuinely_local_urls_stay_local(url):
    assert _is_local_url(url), url
    assert not is_remote_route("ollama", url), url


@pytest.mark.parametrize("task", GRADING_TASKS)
@pytest.mark.parametrize("backend,url", [
    ("ollama", "http://10.0.0.1.evil.example:11434/v1"),   # 'ollama' + remote URL
    ("ollama_native", "http://localhost.evil.example:11434"),
    ("openai", "https://api.evil.example/v1"),             # OpenAI-compat remote
    ("openai", "https://openrouter.ai/api/v1"),            # OpenRouter, other name
    ("anthropic", None),                                   # Anthropic endpoint
    ("openrouter", None),
])
def test_every_disguised_remote_grading_route_is_refused(task, backend, url):
    with pytest.raises(CloudBoundaryError) as e:
        check_cloud_call(task=task, backend=backend, base_url=url,
                         execution_mode="production")
    assert e.value.code == "CLOUD_TASK_FORBIDDEN"


def test_refusal_happens_before_serialization_or_network(tmp_path):
    """A gateway with NO backend for the route still refuses at the boundary:
    the check runs before any backend/serialization work (and the shared
    no_network fixture would kill any socket attempt anyway)."""
    from autograder.gateway import ModelGateway
    gw = ModelGateway.from_dict(
        {"models": {"grade_primary": {
            "backend": "openai", "base_url": "http://10.0.0.1.evil.example/v1",
            "model": "x"}}})
    from pydantic import BaseModel

    class Out(BaseModel):
        ok: bool

    with pytest.raises(CloudBoundaryError):
        gw.call(task="grade_primary", system="s",
                content_blocks=[{"type": "text", "text": "t"}],
                output_model=Out, meta={"job_id": "j"})


def test_ipv4_lookalike_edge_cases_fail_closed():
    # not four octets, octet > 255, non-numeric — none may count as private
    for host in ("http://10.1.2/v1", "http://10.1.2.3.4/v1",
                 "http://10.1.2.999/v1", "http://10.a.2.3/v1"):
        assert not _is_local_url(host), host
