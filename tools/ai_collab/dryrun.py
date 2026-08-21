"""--dry-run: validate config + repo, show the planned loop and payload sizes.

Guarantees: zero AI/API calls, zero code changes, zero run artifacts.
Everything here is read-only (git reads, file reads, PATH lookups).
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from . import git_ops, prompts
from .claude_adapter import ClaudeCodeAdapter
from .config import CollabConfig, has_unresolved_env
from .errors import GitError, GitSafetyError
from .orchestrator import default_task_id
from .payload import DiffBundle, build_reviewer_payload
from .util import read_text

_KEY_ENVS = {"openrouter": "OPENROUTER_API_KEY", "openai": "OPENAI_API_KEY"}


def _mode_plan(mode: str, max_rounds: int) -> list[str]:
    lines = []
    for round_no in range(1, max_rounds + 1):
        kind = "implements task" if round_no == 1 else "applies reviewer findings"
        lines.append(f"round {round_no}: Claude {kind} -> handoff + orchestrator diff capture")
        if mode == "manual":
            lines.append("         PAUSE: user approval before the reviewer call")
        lines.append(
            f"review {round_no}: budget check -> cache check -> reviewer verdict"
        )
        lines.append("         APPROVED -> STOP | BLOCKED -> STOP")
        if round_no < max_rounds:
            if mode in ("manual", "semi_auto"):
                lines.append(
                    "         CHANGES_REQUIRED -> PAUSE: user approval before fixes"
                )
            else:
                lines.append("         CHANGES_REQUIRED -> next round (auto_bounded)")
        else:
            lines.append("         CHANGES_REQUIRED -> MAX_ROUNDS -> STOP")
    return lines


def build_dry_run_report(
    repo_root: Path,
    cfg: CollabConfig,
    config_warnings: list[str],
    task_path: Path,
    task_id: str | None = None,
) -> tuple[str, bool]:
    lines: list[str] = []
    ok = True
    add = lines.append

    add("AI-COLLAB DRY RUN (no API calls, no code changes, no artifacts written)")
    add("=" * 74)

    # --- config ---------------------------------------------------------
    add("")
    add("[config]")
    add(f"  source: {cfg.source_path or '(defaults only; no config.toml found)'}")
    add(f"  run.mode = {cfg.run.mode}   max_rounds = {cfg.run.max_rounds}   "
        f"on_test_failure = {cfg.run.on_test_failure}")
    add(f"  claude.mode = {cfg.claude.mode}")
    add(f"  reviewer.backend = {cfg.reviewer.backend}   model = {cfg.reviewer.model}")
    if has_unresolved_env(cfg.reviewer.model):
        add("    WARNING: reviewer model has an unresolved ${ENV} reference; "
            "a live run would fail at the first reviewer call")
    add(f"  budget: calls<={cfg.budget.max_reviewer_calls} "
        f"in_tokens<={cfg.budget.max_input_tokens} "
        f"out_tokens<={cfg.budget.max_output_tokens} "
        f"cost<=${cfg.budget.max_cost_usd}")
    add(f"  payload caps: total {cfg.payload.max_total_chars} chars, "
        f"diff {cfg.payload.max_diff_chars}, per-file {cfg.payload.max_file_chars}")
    add(f"  tests.commands = {cfg.tests.commands or '(none configured)'}")
    add(f"  graphify.enabled = {cfg.graphify.enabled}")
    for warning in config_warnings:
        add(f"  WARNING: {warning}")

    # --- environment ----------------------------------------------------
    add("")
    add("[environment]")
    key_env = _KEY_ENVS.get(cfg.reviewer.backend)
    if key_env:
        add(f"  {key_env}: {'set' if os.environ.get(key_env) else 'NOT SET'}"
            "  (value never read during dry-run, never printed)")
    else:
        add("  reviewer backend is mock: no API key needed")

    # --- claude invocation preview -------------------------------------
    add("")
    add("[claude invocation]")
    if cfg.claude.mode == "claude_code":
        resolved = shutil.which(cfg.claude.executable)
        add(f"  executable: {cfg.claude.executable} -> "
            f"{resolved or 'NOT FOUND on PATH'}")
        if not resolved:
            add("    WARNING: a live run would fail to launch Claude Code")
        preview = ClaudeCodeAdapter(cfg.claude, repo_root).build_command()
        add("  command: " + " ".join(preview))
        add("  prompt: fed via STDIN; handoff: JSON file written by Claude;")
        add(f"  timeout: {cfg.claude.timeout_seconds}s; cwd: {repo_root}")
    elif cfg.claude.mode == "manual":
        add("  manual mode: orchestrator writes prompt + MANUAL_INSTRUCTIONS.md "
            "and pauses; you drive your own Claude Code session")
    else:
        add(f"  mock mode (script: {cfg.claude.mock_script or 'MISSING'})")

    # --- repository -----------------------------------------------------
    add("")
    add("[repository]")
    add(f"  root: {repo_root}")
    try:
        git_ops.ensure_repo(repo_root)
        branch = git_ops.current_branch(repo_root)
        head = git_ops.head_commit(repo_root)
        dirty = git_ops.status_porcelain(repo_root)
        add(f"  branch: {branch or '(detached HEAD)'}")
        add(f"  head:   {head[:12]}")
        add(f"  dirty files: {len(dirty)}")
        try:
            git_ops.check_start_preconditions(
                repo_root,
                cfg.git.protected_branches,
                cfg.git.require_clean,
                allow_dirty=False,
            )
            add("  start check: PASS (a run could start here)")
        except GitSafetyError as exc:
            add(f"  start check: WOULD REFUSE -> {exc}")
            ok = False
    except GitError as exc:
        add(f"  ERROR: {exc}")
        ok = False

    # --- task -----------------------------------------------------------
    add("")
    add("[task]")
    if task_path.is_file():
        task_text = read_text(task_path)
        tid = task_id or default_task_id(task_path, task_text)
        add(f"  file: {task_path}")
        add(f"  task id: {tid}")
        add(f"  size: {len(task_text)} chars")
    else:
        add(f"  ERROR: task file not found: {task_path}")
        task_text = ""
        ok = False

    # --- planned loop ---------------------------------------------------
    add("")
    add(f"[planned loop] mode={cfg.run.mode}, hard cap {cfg.run.max_rounds} rounds")
    for line in _mode_plan(cfg.run.mode, cfg.run.max_rounds):
        add("  " + line)
    add("  additional stop states: BLOCKED, TEST_FAILURE, BUDGET_EXHAUSTED,")
    add("  USER_APPROVAL_REQUIRED, ERROR — no implicit retries beyond "
        "1 bounded test fix and 1 malformed-reply retry")

    # --- payload estimate ----------------------------------------------
    add("")
    add("[reviewer payload estimate] (diff is empty until Claude runs)")
    if task_text:
        context_path = repo_root / cfg.payload.context_file
        context_text = (
            read_text(context_path) if context_path.is_file() else "(missing)"
        )
        system = prompts.reviewer_system_prompt(cfg.policy.block_on)
        payload = build_reviewer_payload(
            cfg,
            system,
            task_text,
            context_text,
            handoff_obj=None,
            bundle=DiffBundle(),
            test_output="",
        )
        add(f"  system prompt: {len(system)} chars")
        for name, size in payload.section_sizes.items():
            add(f"  section {name}: {size} chars")
        add(f"  estimated input tokens per review call (before diff): "
            f"~{payload.est_input_tokens}")

    # --- guarantees ------------------------------------------------------
    add("")
    add("[dry-run guarantees]")
    add("  Claude child processes launched: 0")
    add("  OpenRouter API calls: 0")
    add("  OpenAI API calls: 0")
    add("  code edits / git mutations / run artifacts written: 0")
    add("")
    add("verdict: " + ("READY - a real run could start" if ok else
                       "NOT READY - fix the items marked ERROR/WOULD REFUSE"))
    return "\n".join(lines), ok
