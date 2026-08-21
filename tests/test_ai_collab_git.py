"""Git capture + safety tests for tools/ai_collab (temp repos, offline)."""

import subprocess

import pytest

from _ai_collab_helpers import _git, make_repo
from tools.ai_collab import git_ops
from tools.ai_collab.errors import GitError, GitSafetyError


def test_run_git_rejects_mutating_subcommands(tmp_path):
    repo = make_repo(tmp_path)
    for args in (["checkout", "-b", "x"], ["commit", "-m", "x"], ["push"],
                 ["reset", "--hard"], ["merge", "x"]):
        with pytest.raises(GitError):
            git_ops.run_git(repo, args)


def test_capture_change_set_is_exact(tmp_path):
    repo = make_repo(tmp_path)
    base = git_ops.head_commit(repo)

    app = repo / "app.py"
    app.write_text(app.read_text(encoding="utf-8") + "\n# round-one edit\n",
                   encoding="utf-8")
    (repo / "notes.txt").write_text(
        "untracked scratch notes\n", encoding="utf-8", newline="\n"
    )

    change = git_ops.capture_change_set(repo, base)
    assert change.base == base
    assert ("M", "app.py") in change.name_status
    assert "+# round-one edit" in change.diff_text
    assert change.untracked == ["notes.txt"]
    assert change.untracked_contents["notes.txt"] == "untracked scratch notes\n"

    # A commit must not hide anything: committed + new working-tree edits both show.
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "wip")
    app.write_text(app.read_text(encoding="utf-8") + "# round-two edit\n",
                   encoding="utf-8")
    change2 = git_ops.capture_change_set(repo, base)
    assert "+# round-one edit" in change2.diff_text
    assert "+# round-two edit" in change2.diff_text
    assert change2.head != base


def test_capture_rejects_missing_base(tmp_path):
    repo = make_repo(tmp_path)
    with pytest.raises(GitError):
        git_ops.capture_change_set(repo, "0" * 40)


def test_start_preconditions_dirty_tree(tmp_path):
    repo = make_repo(tmp_path)
    (repo / "app.py").write_text("changed\n", encoding="utf-8")
    with pytest.raises(GitSafetyError):
        git_ops.check_start_preconditions(repo, ["main"], True, allow_dirty=False)
    baseline = git_ops.check_start_preconditions(
        repo, ["main"], True, allow_dirty=True
    )
    assert baseline["dirty_at_start"] is True
    assert baseline["allow_dirty"] is True


def test_start_preconditions_protected_branch(tmp_path):
    repo = make_repo(tmp_path, branch="")  # stay on the default branch
    subprocess.run(
        ["git", "-C", str(repo), "branch", "-m", "main"],
        check=True, capture_output=True,
    )
    with pytest.raises(GitSafetyError):
        git_ops.check_start_preconditions(
            repo, ["main", "master"], True, allow_dirty=False
        )
    git_ops.create_branch(repo, "feature/review-x")
    baseline = git_ops.check_start_preconditions(
        repo, ["main", "master"], True, allow_dirty=False
    )
    assert baseline["branch"] == "feature/review-x"
