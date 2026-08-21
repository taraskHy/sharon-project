"""Configuration loading for the collaboration harness.

TOML file (default ``tools/ai_collab/config.toml``, template
``config.example.toml``) overlaid onto dataclass defaults. String values may
reference environment variables as ``${VAR}``; expansion happens in memory at
load time and unresolved references are reported as warnings (they only become
hard errors when the value is actually needed for a live call).

API keys are NEVER part of the config. They are read from the environment
(``OPENROUTER_API_KEY`` / ``OPENAI_API_KEY``) at call time by the reviewer
adapter and are never persisted or printed.
"""

from __future__ import annotations

import dataclasses
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .errors import ConfigError
from .states import MODES

_ENV_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

DEFAULT_CONFIG_RELPATH = Path("tools/ai_collab/config.toml")


@dataclass
class RunCfg:
    mode: str = "semi_auto"  # manual | semi_auto | auto_bounded
    max_rounds: int = 3
    on_test_failure: str = "stop"  # stop | review
    test_fix_attempts: int = 1


@dataclass
class ClaudeCfg:
    mode: str = "claude_code"  # claude_code | manual | mock
    executable: str = "claude"
    permission_mode: str = "acceptEdits"
    allowed_tools: list[str] = field(
        default_factory=lambda: [
            "Edit",
            "Write",
            "Read",
            "Glob",
            "Grep",
            "Bash(git *)",
            "Bash(python*)",
            "Bash(.venv*)",
        ]
    )
    model: str = ""  # empty -> Claude Code session default
    max_budget_usd: float = 0.0  # 0 -> flag not passed
    timeout_seconds: int = 3600
    extra_args: list[str] = field(default_factory=list)
    handoff_filename: str = "claude_handoff.json"
    mock_script: str = ""  # JSON file for the file-driven mock adapter


@dataclass
class ReviewerCfg:
    backend: str = "openrouter"  # openrouter | openai | mock
    model: str = "${AI_REVIEW_MODEL}"
    temperature: float = 0.0
    max_output_tokens: int = 4000
    request_timeout_seconds: int = 180
    max_retries: int = 1  # extra attempts after malformed reviewer output
    allow_context_followup: bool = True
    max_context_files: int = 5
    force_json: bool = True
    api_base: str = ""  # optional endpoint override
    mock_script: str = ""  # JSON file for the file-driven mock reviewer


@dataclass
class PolicyCfg:
    # Severities that veto an APPROVED verdict (orchestrator-side backstop).
    block_on: list[str] = field(default_factory=lambda: ["critical", "high"])


@dataclass
class BudgetCfg:
    # A limit <= 0 disables that particular limit.
    max_reviewer_calls: int = 8
    max_input_tokens: int = 400_000
    max_output_tokens: int = 40_000
    max_cost_usd: float = 5.0


@dataclass
class PayloadCfg:
    max_total_chars: int = 240_000
    max_diff_chars: int = 120_000
    max_file_chars: int = 20_000
    max_test_output_chars: int = 20_000
    max_handoff_chars: int = 20_000
    max_context_chars: int = 30_000
    max_untracked_file_chars: int = 10_000
    context_file: str = "tools/ai_collab/context/project_context.md"


@dataclass
class GitCfg:
    protected_branches: list[str] = field(default_factory=lambda: ["main", "master"])
    require_clean: bool = True


@dataclass
class TestsCfg:
    # Each command is an argv list run from the repo root by the orchestrator.
    commands: list[list[str]] = field(default_factory=list)
    timeout_seconds: int = 1800


@dataclass
class GraphifyCfg:
    enabled: bool = False
    commands: list[list[str]] = field(default_factory=list)
    timeout_seconds: int = 120


@dataclass
class RedactionCfg:
    extra_deny_globs: list[str] = field(default_factory=list)
    extra_patterns: list[str] = field(default_factory=list)


@dataclass
class CollabConfig:
    run: RunCfg = field(default_factory=RunCfg)
    claude: ClaudeCfg = field(default_factory=ClaudeCfg)
    reviewer: ReviewerCfg = field(default_factory=ReviewerCfg)
    policy: PolicyCfg = field(default_factory=PolicyCfg)
    budget: BudgetCfg = field(default_factory=BudgetCfg)
    payload: PayloadCfg = field(default_factory=PayloadCfg)
    git: GitCfg = field(default_factory=GitCfg)
    tests: TestsCfg = field(default_factory=TestsCfg)
    graphify: GraphifyCfg = field(default_factory=GraphifyCfg)
    redaction: RedactionCfg = field(default_factory=RedactionCfg)
    source_path: str = ""  # config file actually loaded ("" -> defaults only)
    raw: dict = field(default_factory=dict)  # as-read TOML (unexpanded, no secrets)


_SECTIONS = {
    "run": RunCfg,
    "claude": ClaudeCfg,
    "reviewer": ReviewerCfg,
    "policy": PolicyCfg,
    "budget": BudgetCfg,
    "payload": PayloadCfg,
    "git": GitCfg,
    "tests": TestsCfg,
    "graphify": GraphifyCfg,
    "redaction": RedactionCfg,
}

# Fields on which ${VAR} expansion is performed at load time.
_ENV_EXPAND_FIELDS = {
    ("reviewer", "model"),
    ("reviewer", "api_base"),
    ("claude", "executable"),
    ("claude", "model"),
}


def expand_env(value: str) -> tuple[str, list[str]]:
    """Expand ``${VAR}`` references; return (expanded, missing_var_names)."""
    missing: list[str] = []

    def _sub(match: re.Match) -> str:
        name = match.group(1)
        val = os.environ.get(name)
        if val is None:
            missing.append(name)
            return match.group(0)  # leave the reference in place
        return val

    return _ENV_RE.sub(_sub, value), missing


def has_unresolved_env(value: str) -> bool:
    return bool(_ENV_RE.search(value))


def _coerce(section: str, key: str, default_value, value):
    """Loose type checking with int->float promotion; raise ConfigError otherwise."""
    if isinstance(default_value, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"[{section}].{key}: expected a boolean, got {value!r}")
        return value
    if isinstance(default_value, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"[{section}].{key}: expected a number, got {value!r}")
        return float(value)
    if isinstance(default_value, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"[{section}].{key}: expected an integer, got {value!r}")
        return value
    if isinstance(default_value, str):
        if not isinstance(value, str):
            raise ConfigError(f"[{section}].{key}: expected a string, got {value!r}")
        return value
    if isinstance(default_value, list):
        if not isinstance(value, list):
            raise ConfigError(f"[{section}].{key}: expected a list, got {value!r}")
        return value
    return value


def load_config(
    path: Path | None, repo_root: Path
) -> tuple[CollabConfig, list[str]]:
    """Load config from *path* (or the default location, or pure defaults).

    Returns ``(config, warnings)``. Structural problems raise ConfigError;
    unresolved ``${VAR}`` references and unknown keys are warnings.
    """
    warnings: list[str] = []
    cfg = CollabConfig()

    resolved_path: Path | None = None
    if path is not None:
        resolved_path = Path(path)
        if not resolved_path.is_file():
            raise ConfigError(f"config file not found: {resolved_path}")
    else:
        candidate = repo_root / DEFAULT_CONFIG_RELPATH
        if candidate.is_file():
            resolved_path = candidate

    raw: dict = {}
    if resolved_path is not None:
        try:
            with open(resolved_path, "rb") as fh:
                raw = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"invalid TOML in {resolved_path}: {exc}") from exc
        cfg.source_path = str(resolved_path)
        cfg.raw = raw

    for section_name, data in raw.items():
        if section_name not in _SECTIONS:
            warnings.append(f"unknown config section [{section_name}] ignored")
            continue
        if not isinstance(data, dict):
            raise ConfigError(f"[{section_name}] must be a table")
        target = getattr(cfg, section_name)
        for key, value in data.items():
            if not hasattr(target, key):
                warnings.append(f"unknown key [{section_name}].{key} ignored")
                continue
            default_value = getattr(_SECTIONS[section_name](), key)
            setattr(target, key, _coerce(section_name, key, default_value, value))

    # Environment expansion (in memory only).
    for section_name, key in _ENV_EXPAND_FIELDS:
        target = getattr(cfg, section_name)
        value = getattr(target, key)
        if isinstance(value, str) and value:
            expanded, missing = expand_env(value)
            setattr(target, key, expanded)
            for var in missing:
                warnings.append(
                    f"[{section_name}].{key} references ${{{var}}} which is not set"
                )

    _validate(cfg)
    return cfg, warnings


def _validate(cfg: CollabConfig) -> None:
    problems: list[str] = []
    if cfg.run.mode not in MODES:
        problems.append(f"[run].mode must be one of {MODES}, got {cfg.run.mode!r}")
    if cfg.run.max_rounds < 1:
        problems.append("[run].max_rounds must be >= 1")
    if cfg.run.on_test_failure not in ("stop", "review"):
        problems.append("[run].on_test_failure must be 'stop' or 'review'")
    if cfg.run.test_fix_attempts < 0:
        problems.append("[run].test_fix_attempts must be >= 0")
    if cfg.claude.mode not in ("claude_code", "manual", "mock"):
        problems.append("[claude].mode must be claude_code | manual | mock")
    if cfg.reviewer.backend not in ("openrouter", "openai", "mock"):
        problems.append("[reviewer].backend must be openrouter | openai | mock")
    if cfg.reviewer.max_retries < 0:
        problems.append("[reviewer].max_retries must be >= 0")
    for sev in cfg.policy.block_on:
        if sev not in ("critical", "high", "medium", "low"):
            problems.append(f"[policy].block_on contains unknown severity {sev!r}")
    for i, command in enumerate(cfg.tests.commands):
        if not isinstance(command, list) or not all(
            isinstance(part, str) for part in command
        ) or not command:
            problems.append(
                f"[tests].commands[{i}] must be a non-empty list of strings"
            )
    for i, command in enumerate(cfg.graphify.commands):
        if (
            not isinstance(command, list)
            or not command
            or not all(isinstance(part, str) for part in command)
            or command[0] != "graphify"
        ):
            problems.append(
                f"[graphify].commands[{i}] must be an argv list starting with 'graphify'"
            )
    if problems:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(problems))


def config_snapshot(cfg: CollabConfig) -> dict:
    """A JSON-safe snapshot for run.json: effective values, env refs unexpanded.

    Uses the raw TOML for fields that referenced the environment so that no
    expanded environment value is ever persisted.
    """
    snap: dict = {}
    for section_name in _SECTIONS:
        section = getattr(cfg, section_name)
        entry = dataclasses.asdict(section)
        raw_section = cfg.raw.get(section_name, {})
        for section2, key in _ENV_EXPAND_FIELDS:
            if section2 != section_name:
                continue
            if key in raw_section:
                entry[key] = raw_section[key]  # keep the as-written (${VAR}) form
            else:
                entry[key] = getattr(_SECTIONS[section_name](), key)  # unexpanded default
        snap[section_name] = entry
    snap["source_path"] = cfg.source_path
    return snap


def generation_config(cfg: ReviewerCfg) -> dict:
    """The generation parameters that participate in the cache fingerprint."""
    return {
        "temperature": cfg.temperature,
        "max_output_tokens": cfg.max_output_tokens,
        "force_json": cfg.force_json,
    }
