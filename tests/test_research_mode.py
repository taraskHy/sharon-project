"""Cloud grading survives ONLY as an explicit research act.

The historical cloud-grader benchmarks (grade-v3/v4 on Sonnet/Gemini) stay
reproducible — but behind `bench ... --research`, never as a production route
and never by default. No provider is contacted anywhere in this file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autograder.benchmark.runner import RunSpec, run_benchmark

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_credential(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


def _spec(tmp_path, **kw):
    d = dict(role="grade_primary", split="DEV", candidate="google/gemini-3.7-flash",
             backend="openrouter", dry_run=False, skip_key_preflight=True,
             state_root=tmp_path / "state", runs_root=tmp_path / "runs",
             subset="dev_verdict")
    d.update(kw)
    return RunSpec(**d)


def test_a_live_cloud_grading_benchmark_without_research_mode_is_refused(tmp_path):
    """§13-8/22: the refusal happens BEFORE any gateway, credential check, or
    provider work, and it names the missing flag."""
    with pytest.raises(RuntimeError, match="--research"):
        run_benchmark(_spec(tmp_path))


def test_the_gate_is_role_independent():
    """§13-9: grade_escalate (and every other role) shares the same gate — a
    single is_remote_route check on the resolved route, not a per-role list.
    Pinned at the source level because the escalate DATASET is not built in
    this clone (building it needs harvested runs)."""
    import inspect

    from autograder.benchmark import runner
    src = inspect.getsource(runner.run_benchmark)
    assert "is_remote_route" in src and "spec.research" in src
    assert "role ==" not in src.split("is_remote_route")[1][:400],         "the research gate must not special-case roles"


def test_research_mode_reaches_the_ordinary_cloud_readiness_gate(tmp_path):
    """With --research the boundary opens — and the run then fails on the
    NEXT gate (no credential), proving the flag is what was missing and that
    nothing fired before the credential check."""
    from autograder.cloudcheck import CloudNotReady
    with pytest.raises(CloudNotReady, match="credential"):
        run_benchmark(_spec(tmp_path, research=True))


def test_dry_runs_stay_flagless_and_call_free(tmp_path):
    res = run_benchmark(_spec(tmp_path, dry_run=True))
    assert res.dry_run and res.cases_selected > 0


def test_the_research_gateway_is_marked_research_and_the_default_is_production(tmp_path):
    from autograder.benchmark.registry import load_registry
    from autograder.benchmark.runner import build_gateway, build_route

    registry = load_registry()
    spec = _spec(tmp_path, research=True)
    route = build_route(spec, spec.candidate, "grade-v4-charitable", 600, registry=registry)
    gw = build_gateway(spec, route, registry, lambda m: None)
    assert gw.execution_mode == "research"
    spec2 = _spec(tmp_path)
    gw2 = build_gateway(spec2, route, registry, lambda m: None)
    assert gw2.execution_mode == "production"


def test_the_bench_cli_exposes_the_research_flag():
    from autograder.cli import build_parser
    p = build_parser()
    ns = p.parse_args(["bench", "run", "--role", "grade_primary", "--split", "dev",
                       "--candidate", "x/y", "--research"])
    assert ns.research is True
    ns2 = p.parse_args(["bench", "run", "--role", "grade_primary", "--split", "dev",
                        "--candidate", "x/y"])
    assert ns2.research is False


def test_local_benchmark_runs_need_no_research_flag(tmp_path):
    """A LOCAL grading benchmark is production-shaped, not research: the
    remote gate does not fire for a local backend. (It fails later on the
    unreachable local server in a real run; here the check of interest is
    that no --research refusal occurs.)"""
    spec = _spec(tmp_path, backend="ollama", base_url="http://localhost:11434/v1",
                 candidate="qwen3-vl:8b-instruct", limit=1)
    try:
        run_benchmark(spec)
    except RuntimeError as e:
        assert "--research" not in str(e)
    except Exception:
        pass  # any later failure (dead local server, etc.) is out of scope here
