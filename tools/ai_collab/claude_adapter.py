"""Claude (implementer) adapters.

``claude_code`` — headless invocation of the locally installed Claude Code
CLI, verified against the installed version (2.1.215):

    claude -p --output-format json --permission-mode <mode> \
           [--allowedTools <t1,t2,...>] [--model <m>] [--max-budget-usd <x>]

The prompt is fed via STDIN (avoids Windows command-line length limits); the
handoff channel is a FILE the prompt instructs Claude to write, which is more
deterministic than parsing prose. The 2.1.x CLI has no --max-turns flag;
bounding is via --max-budget-usd (optional) plus a subprocess timeout.

``manual`` — no child process: the orchestrator writes the prompt +
instructions and pauses; the user drives their own interactive Claude Code
session and saves the handoff file, then runs `continue`.

``mock`` — deterministic scripted adapter for tests / offline rehearsal.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .config import ClaudeCfg
from .errors import AdapterError
from .util import read_json, write_json_atomic


@dataclass
class ClaudeInvocation:
    prompt_text: str
    handoff_path: Path
    round_no: int
    attempt_no: int


@dataclass
class ClaudeResult:
    status: str  # "completed" | "pending" | "error"
    raw_output: str = ""
    stderr: str = ""
    returncode: int | None = None
    error: str = ""


class ClaudeCodeAdapter:
    def __init__(self, cfg: ClaudeCfg, repo: Path):
        self.cfg = cfg
        self.repo = Path(repo)

    def build_command(self) -> list[str]:
        exe = shutil.which(self.cfg.executable) or self.cfg.executable
        cmd = [exe, "-p", "--output-format", "json"]
        if self.cfg.permission_mode:
            cmd += ["--permission-mode", self.cfg.permission_mode]
        if self.cfg.allowed_tools:
            cmd += ["--allowedTools", ",".join(self.cfg.allowed_tools)]
        if self.cfg.model:
            cmd += ["--model", self.cfg.model]
        if self.cfg.max_budget_usd and self.cfg.max_budget_usd > 0:
            cmd += ["--max-budget-usd", str(self.cfg.max_budget_usd)]
        cmd += list(self.cfg.extra_args)
        if sys.platform == "win32" and exe.lower().endswith((".cmd", ".bat")):
            cmd = ["cmd", "/c"] + cmd
        return cmd

    def run_round(self, inv: ClaudeInvocation) -> ClaudeResult:
        cmd = self.build_command()
        try:
            proc = subprocess.run(
                cmd,
                input=inv.prompt_text,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(self.repo),
                timeout=self.cfg.timeout_seconds,
            )
        except FileNotFoundError as exc:
            return ClaudeResult(
                status="error",
                error=f"claude executable not found ({self.cfg.executable}): {exc}",
            )
        except subprocess.TimeoutExpired:
            return ClaudeResult(
                status="error",
                error=f"claude timed out after {self.cfg.timeout_seconds}s",
            )
        status = "completed" if proc.returncode == 0 else "error"
        return ClaudeResult(
            status=status,
            raw_output=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
            error="" if proc.returncode == 0 else f"claude exited rc={proc.returncode}",
        )


class ManualClaudeAdapter:
    def __init__(self, cfg: ClaudeCfg, repo: Path):
        self.cfg = cfg
        self.repo = Path(repo)

    def run_round(self, inv: ClaudeInvocation) -> ClaudeResult:
        return ClaudeResult(status="pending")


@dataclass
class MockAction:
    """One scripted Claude attempt.

    ``handoff``       dict written as JSON to the handoff path
    ``handoff_text``  raw text written instead (to simulate malformed output)
    ``mutate``        optional callable(repo: Path) simulating edits
    ``skip_handoff``  simulate Claude forgetting to write the handoff
    ``raw``           text returned as adapter stdout
    """

    handoff: dict | None = None
    handoff_text: str | None = None
    mutate: object = None
    skip_handoff: bool = False
    raw: str = "(mock claude output)"


class MockClaudeAdapter:
    def __init__(self, actions: list[MockAction], repo: Path, start_index: int = 0):
        self.actions = actions
        self.repo = Path(repo)
        self.index = start_index
        self.calls: list[ClaudeInvocation] = []
        self.prompts: list[str] = []

    @classmethod
    def from_script_file(
        cls, script_path: Path, repo: Path, start_index: int = 0
    ) -> "MockClaudeAdapter":
        data = read_json(Path(script_path))
        actions = []
        for entry in data.get("handoffs", []):
            if isinstance(entry, dict):
                actions.append(MockAction(handoff=entry))
            else:
                actions.append(MockAction(handoff_text=str(entry)))
        return cls(actions, repo, start_index=start_index)

    def run_round(self, inv: ClaudeInvocation) -> ClaudeResult:
        if self.index >= len(self.actions):
            return ClaudeResult(status="error", error="mock claude script exhausted")
        action = self.actions[self.index]
        self.index += 1
        self.calls.append(inv)
        self.prompts.append(inv.prompt_text)
        if callable(action.mutate):
            action.mutate(self.repo)
        if not action.skip_handoff:
            if action.handoff_text is not None:
                inv.handoff_path.parent.mkdir(parents=True, exist_ok=True)
                inv.handoff_path.write_text(action.handoff_text, encoding="utf-8")
            elif action.handoff is not None:
                write_json_atomic(inv.handoff_path, action.handoff)
        return ClaudeResult(status="completed", raw_output=action.raw, returncode=0)


def make_claude_adapter(cfg: ClaudeCfg, repo: Path, start_index: int = 0):
    if cfg.mode == "claude_code":
        return ClaudeCodeAdapter(cfg, repo)
    if cfg.mode == "manual":
        return ManualClaudeAdapter(cfg, repo)
    if cfg.mode == "mock":
        if not cfg.mock_script:
            raise AdapterError(
                "[claude].mode='mock' requires [claude].mock_script "
                "(JSON file with a 'handoffs' list)"
            )
        script = Path(cfg.mock_script)
        if not script.is_absolute():
            script = Path(repo) / script
        return MockClaudeAdapter.from_script_file(script, repo, start_index=start_index)
    raise AdapterError(f"unknown claude mode: {cfg.mode}")


def extract_handoff_fallback(raw_stdout: str):
    """Fallback when the handoff file is missing: look inside the CLI's JSON
    envelope (``--output-format json`` puts the final text in ``result``)."""
    from .schemas import parse_json_lenient

    envelope = parse_json_lenient(raw_stdout)
    if isinstance(envelope, dict):
        result_text = envelope.get("result")
        if isinstance(result_text, str):
            candidate = parse_json_lenient(result_text)
            if isinstance(candidate, dict) and "status" in candidate:
                return candidate
    return None
