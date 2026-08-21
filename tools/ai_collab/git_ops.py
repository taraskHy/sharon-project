"""Read-only git capture + safety checks.

Every orchestrator-side git call goes through ``run_git``, which enforces a
read-only subcommand allow-list. The single mutating operation the harness
supports (creating an explicitly named review branch) requires the user to
pass ``--create-branch`` on the CLI and uses its own function.

The orchestrator — not Claude's self-report — is the source of truth for
base commit, head commit, changed files and the diff (spec sections 9/10).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .errors import GitError, GitSafetyError

_READONLY_SUBCOMMANDS = frozenset(
    {"rev-parse", "status", "diff", "log", "ls-files", "show", "branch"}
)

_MAX_UNTRACKED_READ_BYTES = 300_000


def run_git(repo: Path, args: list[str], check: bool = True) -> str:
    if not args or args[0] not in _READONLY_SUBCOMMANDS:
        raise GitError(
            f"refusing non-allow-listed git subcommand: {args[:1]} "
            f"(allowed: {sorted(_READONLY_SUBCOMMANDS)})"
        )
    proc = subprocess.run(
        ["git", "-C", str(repo)] + args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and proc.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed (rc={proc.returncode}): "
            f"{proc.stderr.strip()[:500]}"
        )
    return proc.stdout


def ensure_repo(repo: Path) -> None:
    try:
        out = run_git(repo, ["rev-parse", "--is-inside-work-tree"])
    except (GitError, OSError) as exc:
        raise GitError(f"not a usable git repository: {repo} ({exc})") from exc
    if out.strip() != "true":
        raise GitError(f"not inside a git work tree: {repo}")


def repo_toplevel(repo: Path) -> Path:
    return Path(run_git(repo, ["rev-parse", "--show-toplevel"]).strip())


def current_branch(repo: Path) -> str:
    return run_git(repo, ["branch", "--show-current"]).strip()


def head_commit(repo: Path) -> str:
    return run_git(repo, ["rev-parse", "HEAD"]).strip()


def status_porcelain(repo: Path) -> list[str]:
    out = run_git(repo, ["status", "--porcelain"])
    return [line for line in out.splitlines() if line.strip()]


def is_dirty(repo: Path) -> bool:
    return bool(status_porcelain(repo))


def untracked_files(repo: Path) -> list[str]:
    out = run_git(repo, ["ls-files", "--others", "--exclude-standard"])
    return [line.strip() for line in out.splitlines() if line.strip()]


def commit_exists(repo: Path, sha: str) -> bool:
    try:
        run_git(repo, ["rev-parse", "--verify", "--quiet", sha + "^{commit}"])
        return True
    except GitError:
        return False


def diff_from(repo: Path, base: str) -> str:
    """Committed AND uncommitted tracked changes relative to *base*."""
    return run_git(repo, ["diff", "--no-color", base, "--"])


def name_status_from(repo: Path, base: str) -> list[tuple[str, str]]:
    out = run_git(repo, ["diff", "--no-color", "--name-status", base, "--"])
    result: list[tuple[str, str]] = []
    for line in out.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) >= 2:
            # Renames/copies (R100 old new) -> report status + final path.
            result.append((parts[0], parts[-1]))
    return result


@dataclass
class ChangeSet:
    """Orchestrator-captured ground truth for one review round."""

    base: str
    head: str
    branch: str
    name_status: list[tuple[str, str]] = field(default_factory=list)
    diff_text: str = ""
    untracked: list[str] = field(default_factory=list)
    untracked_contents: dict[str, str] = field(default_factory=dict)

    @property
    def changed_paths(self) -> list[str]:
        return [path for _status, path in self.name_status] + list(self.untracked)


def _read_untracked(repo: Path, relpath: str) -> str:
    full = repo / relpath
    try:
        data = full.read_bytes()
    except OSError as exc:
        return f"[unreadable untracked file: {exc}]"
    if b"\x00" in data[:8192]:
        return f"[binary untracked file omitted: {len(data)} bytes]"
    if len(data) > _MAX_UNTRACKED_READ_BYTES:
        data = data[:_MAX_UNTRACKED_READ_BYTES]
    return data.decode("utf-8", errors="replace")


def capture_change_set(repo: Path, base: str) -> ChangeSet:
    """Capture everything the reviewer needs, independent of Claude's report."""
    if not commit_exists(repo, base):
        raise GitError(f"base commit does not exist: {base}")
    change = ChangeSet(
        base=base,
        head=head_commit(repo),
        branch=current_branch(repo),
        name_status=name_status_from(repo, base),
        diff_text=diff_from(repo, base),
        untracked=untracked_files(repo),
    )
    for relpath in change.untracked:
        change.untracked_contents[relpath] = _read_untracked(repo, relpath)
    return change


def check_start_preconditions(
    repo: Path,
    protected_branches: list[str],
    require_clean: bool,
    allow_dirty: bool,
) -> dict:
    """Validate branch + working tree before a run. Returns a baseline dict."""
    ensure_repo(repo)
    branch = current_branch(repo)
    if not branch:
        raise GitSafetyError(
            "detached HEAD: check out a feature branch before starting a run"
        )
    if branch in protected_branches:
        raise GitSafetyError(
            f"current branch {branch!r} is protected; switch to a feature branch "
            "or pass --create-branch <name> to create one explicitly"
        )
    dirty = status_porcelain(repo)
    if dirty and require_clean and not allow_dirty:
        raise GitSafetyError(
            "working tree has uncommitted changes; commit/stash them or rerun "
            "with --allow-dirty to explicitly accept reviewing on top of them:\n  "
            + "\n  ".join(dirty[:20])
        )
    return {
        "branch": branch,
        "base_commit": head_commit(repo),
        "dirty_at_start": bool(dirty),
        "dirty_files_at_start": dirty[:200],
        "allow_dirty": bool(allow_dirty),
    }


def create_branch(repo: Path, name: str) -> None:
    """Explicitly user-approved branch creation (the only mutating git op)."""
    if not name or name.strip() != name or " " in name:
        raise GitSafetyError(f"invalid branch name: {name!r}")
    proc = subprocess.run(
        ["git", "-C", str(repo), "checkout", "-b", name],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        raise GitError(f"could not create branch {name!r}: {proc.stderr.strip()[:300]}")
