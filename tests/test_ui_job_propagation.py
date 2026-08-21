"""UI → job → grade-subprocess propagation of the grading configuration.

The audit found the web UI's jobs always ran pure legacy defaults because
grading mode / models config / RAG policy were never forwarded. These tests
pin the propagation contract at the exact subprocess command line, and prove
the produced flags parse in the real CLI parser (no webui/CLI drift).
Execution of the modes at the seam is covered by test_grading_modes.py.
"""

from __future__ import annotations

import json
from pathlib import Path

from autograder import jobs
from autograder.cli import build_parser
from tests.conftest import make_pdf


def _job(tmp_path: Path, grading_args: dict | None, **kw) -> Path:
    exam = make_pdf(tmp_path / "student.pdf")
    key = make_pdf(tmp_path / "key.pdf")
    return jobs.create_job(key=key, exams=[exam], job_root=tmp_path / "jobs",
                           backend_args={"--backend": "mock", "--model": "m"},
                           grading_args=grading_args or {}, mask=False, **kw)


RELIABILITY_ARGS = {
    "--max-tokens": 8000,
    "--grading-mode": "reliability",
    "--models-config": "models.toml",
    "--rag-policy": "RAG_ON_UNCERTAIN",
    "--course": "algo101",
}


def test_reliability_configuration_reaches_the_subprocess_command(tmp_path):
    job_dir = _job(tmp_path, RELIABILITY_ARGS, grading_mode="reliability")
    job = jobs.load_job(job_dir)
    assert job["grading_mode"] == "reliability"          # informational field
    cmd = jobs._grade_command(job, job_dir, job["exams"][0])
    for flag, value in RELIABILITY_ARGS.items():
        i = cmd.index(flag)
        assert cmd[i + 1] == str(value)

    # the flags the UI emits must PARSE in the real CLI parser (no drift)
    args = build_parser().parse_args(cmd[3:])
    assert args.grading_mode == "reliability"
    assert args.models_config == "models.toml"
    assert args.rag_policy == "RAG_ON_UNCERTAIN"
    assert args.course == "algo101"


def test_legacy_job_command_is_unchanged(tmp_path):
    job_dir = _job(tmp_path, {"--max-tokens": 8000})
    job = jobs.load_job(job_dir)
    assert job["grading_mode"] == "legacy"
    cmd = jobs._grade_command(job, job_dir, job["exams"][0])
    for flag in ("--grading-mode", "--models-config", "--rag-policy", "--course"):
        assert flag not in cmd
    args = build_parser().parse_args(cmd[3:])
    assert args.grading_mode == "legacy"                 # parser default
    assert args.rag_policy == "RAG_DISABLED"             # parser default


def test_old_persisted_jobs_without_the_field_still_run_legacy(tmp_path):
    """A job.json written before the propagation change (no grading_mode key,
    no mode flags) must behave exactly as before."""
    job_dir = _job(tmp_path, {"--max-tokens": 8000})
    p = job_dir / "job.json"
    old = json.loads(p.read_text(encoding="utf-8"))
    del old["grading_mode"]
    p.write_text(json.dumps(old), encoding="utf-8")
    job = jobs.load_job(job_dir)
    cmd = jobs._grade_command(job, job_dir, job["exams"][0])
    assert "--grading-mode" not in cmd
    assert build_parser().parse_args(cmd[3:]).grading_mode == "legacy"


def test_shadow_configuration_propagates_like_reliability(tmp_path):
    job_dir = _job(tmp_path, {"--grading-mode": "shadow",
                              "--models-config": "models.toml"},
                   grading_mode="shadow")
    job = jobs.load_job(job_dir)
    cmd = jobs._grade_command(job, job_dir, job["exams"][0])
    args = build_parser().parse_args(cmd[3:])
    assert args.grading_mode == "shadow"
    # shadow's non-authoritativeness itself is pinned at the seam:
    # test_grading_modes.test_shadow_runs_both_and_legacy_stays_authoritative


def test_package_blocker_is_detected_from_the_log_tail():
    assert jobs._package_setup_blocked("...\nautograder.preflight.PackageSetupRequired: 2 blocking")
    assert not jobs._package_setup_blocked("BackendError: connection refused")
    assert not jobs._package_setup_blocked("")
