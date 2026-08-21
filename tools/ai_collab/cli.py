"""CLI for the collaboration harness.

    python -m tools.ai_collab start <task.md> [--dry-run] [options]
    python -m tools.ai_collab status <task-id>
    python -m tools.ai_collab continue <task-id>
    python -m tools.ai_collab approve <task-id> [--note ...]
    python -m tools.ai_collab stop <task-id> [--note ...]
    python -m tools.ai_collab list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import git_ops
from .artifacts import RunPaths, RunRecord
from .config import load_config
from .dryrun import build_dry_run_report
from .errors import CollabError, GitError
from .orchestrator import Orchestrator
from .states import (
    AWAITING_CLAUDE,
    AWAITING_FIX_APPROVAL,
    AWAITING_REVIEW_APPROVAL,
    MODES,
    PAUSE_STATES,
    TERMINAL_STATES,
    USER_APPROVAL_REQUIRED,
)
from .util import read_json

_PENDING_HINTS = {
    AWAITING_CLAUDE: (
        "manual Claude round in progress: follow MANUAL_INSTRUCTIONS.md in the "
        "current round directory, then run `continue <task-id>`"
    ),
    AWAITING_REVIEW_APPROVAL: (
        "captured diff is ready: run `approve <task-id>` to send it to the "
        "reviewer, or `stop <task-id>`"
    ),
    AWAITING_FIX_APPROVAL: (
        "reviewer requires changes: run `approve <task-id>` to let Claude apply "
        "the findings, or `stop <task-id>`"
    ),
    USER_APPROVAL_REQUIRED: (
        "Claude flagged an issue that needs your decision (see the last handoff "
        "summary): `approve <task-id> --note ...` to continue, or `stop <task-id>`"
    ),
}


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", help="repository root (default: cwd's git toplevel)")
    parser.add_argument("--config", help="path to a config TOML")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tools.ai_collab",
        description="Bounded Claude<->reviewer collaboration harness (developer tooling)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_start = sub.add_parser("start", help="start a new run from a task file")
    p_start.add_argument("task_file")
    p_start.add_argument("--task-id", dest="task_id")
    p_start.add_argument("--mode", choices=MODES)
    p_start.add_argument(
        "--claude-mode", dest="claude_mode", choices=("claude_code", "manual", "mock")
    )
    p_start.add_argument("--dry-run", action="store_true", dest="dry_run")
    p_start.add_argument("--allow-dirty", action="store_true", dest="allow_dirty")
    p_start.add_argument("--create-branch", dest="create_branch")
    _add_common(p_start)

    for name, help_text in (
        ("status", "show the state of a run"),
        ("continue", "resume a paused/interrupted run"),
        ("approve", "pass the current user gate and continue"),
        ("stop", "stop a run now"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("task_id")
        if name in ("approve", "stop"):
            p.add_argument("--note", default="")
        _add_common(p)

    p_list = sub.add_parser("list", help="list runs")
    _add_common(p_list)
    return parser


def _repo_root(args) -> Path:
    base = Path(args.repo).resolve() if getattr(args, "repo", None) else Path.cwd()
    try:
        return git_ops.repo_toplevel(base)
    except (GitError, OSError):
        return base


def _load(args):
    repo_root = _repo_root(args)
    config_path = Path(args.config) if getattr(args, "config", None) else None
    cfg, warnings = load_config(config_path, repo_root)
    if getattr(args, "mode", None):
        cfg.run.mode = args.mode
    if getattr(args, "claude_mode", None):
        cfg.claude.mode = args.claude_mode
    return repo_root, cfg, warnings


def _print_status(repo_root: Path, task_id: str) -> None:
    paths = RunPaths(repo_root, task_id)
    run = RunRecord.load(paths.run_json)
    data = run.data
    print(f"task:        {data['task_id']}")
    print(f"state:       {data['state']}")
    print(f"mode:        {data['mode']}   round {data['current_round']}/{data['max_rounds']}")
    print(f"branch:      {data['baseline'].get('branch')}   base {str(data['baseline'].get('base_commit'))[:12]}")
    if data.get("last_verdict"):
        print(f"last verdict: {data['last_verdict']}")
    if data.get("stop_reason"):
        print(f"stop reason: {data['stop_reason']}")
    budget = data.get("budget", {})
    print(
        "budget used: "
        f"{budget.get('reviewer_calls', 0)} reviewer calls, "
        f"~{budget.get('input_tokens', 0)} in / {budget.get('output_tokens', 0)} out tokens, "
        f"${budget.get('cost_usd', 0.0):.4f} reported cost"
    )
    if paths.final_json.is_file():
        final = read_json(paths.final_json)
        print(f"final state: {final.get('final_state')} ({final.get('stop_reason')})")
    hint = _PENDING_HINTS.get(data["state"])
    if hint:
        print(f"next action: {hint}")
    print(f"artifacts:   {paths.run_dir}")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "start":
            repo_root, cfg, warnings = _load(args)
            task_path = Path(args.task_file)
            if args.dry_run:
                report, ok = build_dry_run_report(
                    repo_root, cfg, warnings, task_path, args.task_id
                )
                print(report)
                return 0 if ok else 2
            for warning in warnings:
                print(f"warning: {warning}", file=sys.stderr)
            orch = Orchestrator.start(
                repo_root,
                cfg,
                task_path,
                task_id=args.task_id,
                allow_dirty=args.allow_dirty,
                create_branch_name=args.create_branch,
            )
            state = orch.advance()
            print(f"run {orch.run.task_id} -> {state}")
            _print_status(repo_root, orch.run.task_id)
            return 0

        if args.cmd == "list":
            repo_root, _cfg, _warnings = _load(args)
            runs_dir = RunPaths(repo_root, "x").runs_dir
            found = False
            if runs_dir.is_dir():
                for run_dir in sorted(runs_dir.iterdir()):
                    run_json = run_dir / "run.json"
                    if run_json.is_file():
                        data = read_json(run_json)
                        print(
                            f"{data.get('task_id'):40s} {data.get('state'):24s} "
                            f"updated {data.get('updated_at')}"
                        )
                        found = True
            if not found:
                print("(no runs)")
            return 0

        repo_root, cfg, warnings = _load(args)
        for warning in warnings:
            print(f"warning: {warning}", file=sys.stderr)

        if args.cmd == "status":
            _print_status(repo_root, args.task_id)
            return 0
        if args.cmd == "continue":
            orch = Orchestrator.load(repo_root, cfg, args.task_id)
            state = orch.advance()
            print(f"run {args.task_id} -> {state}")
            _print_status(repo_root, args.task_id)
            return 0
        if args.cmd == "approve":
            orch = Orchestrator.load(repo_root, cfg, args.task_id)
            before = orch.run.state
            state = orch.advance(approve=True, note=args.note)
            if before == state and state in PAUSE_STATES:
                print(f"nothing to approve in state {state}")
            print(f"run {args.task_id} -> {state}")
            _print_status(repo_root, args.task_id)
            return 0
        if args.cmd == "stop":
            orch = Orchestrator.load(repo_root, cfg, args.task_id)
            if orch.run.state in TERMINAL_STATES:
                print(f"run already finished: {orch.run.state}")
            else:
                final = orch.stop_by_user(note=args.note)
                print(f"run {args.task_id} stopped -> final state {final}")
            _print_status(repo_root, args.task_id)
            return 0
        raise CollabError(f"unknown command {args.cmd}")
    except CollabError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
