"""Optional, targeted Graphify context for the reviewer.

Disabled by default. When enabled, runs only the configured, allow-listed
``graphify query|path|explain`` commands and concatenates their output —
never the whole graph. Failures are recorded as notes, never fatal: source
remains authoritative and the reviewer payload simply omits the section.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from .config import GraphifyCfg

_ALLOWED_SUBCOMMANDS = {"query", "path", "explain"}
_MAX_CMD_OUTPUT_CHARS = 8000


def collect_graphify_context(repo: Path, cfg: GraphifyCfg) -> str:
    if not cfg.enabled or not cfg.commands:
        return ""
    parts: list[str] = [
        "Graphify static-AST notes (navigation aid only; the graph cannot see "
        "dependency injection, runtime config, or dynamic registration — "
        "source code is authoritative):\n"
    ]
    for command in cfg.commands:
        label = " ".join(command)
        if len(command) < 2 or command[1] not in _ALLOWED_SUBCOMMANDS:
            parts.append(f"$ {label}\n[skipped: only {_ALLOWED_SUBCOMMANDS} allowed]\n")
            continue
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(repo),
                timeout=cfg.timeout_seconds,
                shell=False,
            )
            output = proc.stdout if proc.returncode == 0 else (
                f"[graphify failed rc={proc.returncode}] {proc.stderr[:300]}"
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            output = f"[graphify unavailable: {exc}]"
        parts.append(f"$ {label}\n{output[:_MAX_CMD_OUTPUT_CHARS]}\n")
    return "\n".join(parts)
