"""Shared builders for the tools/ai_collab test suite (offline, mock-only)."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tools.ai_collab.config import CollabConfig


def _git(repo: Path, *args: str) -> None:
    proc = subprocess.run(
        ["git", "-C", str(repo)] + list(args),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert proc.returncode == 0, f"git {args} failed: {proc.stderr}"


def make_repo(tmp_path: Path, branch: str = "feature/collab") -> Path:
    """A tiny git repo with one commit, checked out on a feature branch."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "collab-test@example.com")
    _git(repo, "config", "user.name", "collab-test")
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "app.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "init")
    if branch:
        _git(repo, "checkout", "-b", branch)
    return repo


def base_cfg(mode: str = "semi_auto", max_rounds: int = 3) -> CollabConfig:
    """Config for orchestrator tests: mock reviewer backend, no test commands."""
    cfg = CollabConfig()
    cfg.run.mode = mode
    cfg.run.max_rounds = max_rounds
    cfg.reviewer.backend = "mock"  # fingerprint model = "mock"; adapters injected
    cfg.tests.commands = []
    return cfg


def make_task(tmp_path: Path, text: str = "Improve app.py politely.\n") -> Path:
    task = tmp_path / "task.md"
    task.write_text(text, encoding="utf-8")
    return task


def handoff(
    task_id: str,
    round_no: int,
    status: str = "READY_FOR_REVIEW",
    summary: str = "implemented the change",
    files: list[str] | None = None,
) -> dict:
    return {
        "task_id": task_id,
        "round": round_no,
        "status": status,
        "summary": summary,
        "files_changed": files or [],
        "tests": {"commands": [], "passed": 0, "failed": 0},
        "architecture_changes": [],
        "known_gaps": [],
        "questions_for_reviewer": [],
    }


def finding(
    fid: str = "F1",
    severity: str = "high",
    issue: str = "add() lacks input validation",
    requested_change: str = "validate the inputs",
    file: str = "app.py",
) -> dict:
    return {
        "id": fid,
        "severity": severity,
        "category": "correctness",
        "file": file,
        "line_or_symbol": "add",
        "issue": issue,
        "evidence": "diff hunk",
        "requested_change": requested_change,
    }


def review(
    verdict: str = "APPROVED",
    findings: list[dict] | None = None,
    summary: str = "looks correct",
    **extra,
) -> dict:
    doc = {
        "verdict": verdict,
        "summary": summary,
        "findings": findings or [],
        "approved_scope": [],
        "tests_requested": [],
        "context_requests": [],
    }
    doc.update(extra)
    return doc


def mutate_marker(marker: str):
    """A Claude-mock side effect appending a distinctive line to app.py."""

    def _mutate(repo: Path) -> None:
        path = repo / "app.py"
        path.write_text(
            path.read_text(encoding="utf-8") + f"\n# {marker}\n", encoding="utf-8"
        )

    return _mutate
