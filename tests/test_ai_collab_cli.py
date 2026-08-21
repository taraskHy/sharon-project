"""CLI-level tests: dry-run guarantees and the file-driven mock end-to-end."""

import json

from _ai_collab_helpers import finding, handoff, make_repo, make_task, review
from tools.ai_collab.cli import main
from tools.ai_collab.util import read_json


def _write_cli_config(repo, handoffs, reviews, mode="semi_auto", max_rounds=2):
    base = repo / "tools" / "ai_collab"
    base.mkdir(parents=True, exist_ok=True)
    handoffs_path = base / "mock_handoffs.json"
    reviews_path = base / "mock_reviews.json"
    handoffs_path.write_text(
        json.dumps({"handoffs": handoffs}), encoding="utf-8"
    )
    reviews_path.write_text(json.dumps({"reviews": reviews}), encoding="utf-8")
    config = f"""
[run]
mode = "{mode}"
max_rounds = {max_rounds}

[claude]
mode = "mock"
mock_script = "{handoffs_path.as_posix()}"

[reviewer]
backend = "mock"
model = "mock-model"
mock_script = "{reviews_path.as_posix()}"
"""
    (base / "config.toml").write_text(config, encoding="utf-8")
    # keep the temp repo's working tree clean for the start precondition
    (repo / ".gitignore").write_text("tools/\n", encoding="utf-8")
    import subprocess

    subprocess.run(
        ["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True
    )
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-m", "cli config"],
        check=True,
        capture_output=True,
    )


def test_cli_dry_run_makes_zero_calls_and_writes_nothing(
    tmp_path, capsys, no_network
):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    rc = main(["start", str(task), "--dry-run", "--repo", str(repo)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "OpenRouter API calls: 0" in out
    assert "OpenAI API calls: 0" in out
    assert "Claude child processes launched: 0" in out
    assert "READY" in out
    # zero artifacts: the runs directory was never created
    assert not (repo / "tools" / "ai_collab" / "runs").exists()


def test_cli_dry_run_flags_unsafe_repo(tmp_path, capsys, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    (repo / "app.py").write_text("dirty\n", encoding="utf-8")
    rc = main(["start", str(task), "--dry-run", "--repo", str(repo)])
    out = capsys.readouterr().out
    assert rc == 2
    assert "WOULD REFUSE" in out
    assert "NOT READY" in out


def test_cli_full_mock_run_and_status(tmp_path, capsys, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    _write_cli_config(
        repo,
        handoffs=[handoff("demo1", 1)],
        reviews=[review("APPROVED")],
    )
    rc = main(["start", str(task), "--task-id", "demo1", "--repo", str(repo)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "-> APPROVED" in out

    rc = main(["status", "demo1", "--repo", str(repo)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "state:       APPROVED" in out

    rc = main(["list", "--repo", str(repo)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "demo1" in out

    run_dir = repo / "tools" / "ai_collab" / "runs" / "demo1"
    assert (run_dir / "final.json").is_file()
    assert (run_dir / "round_01" / "review.json").is_file()
    assert (run_dir / "round_01" / "diff.patch").is_file()


def test_cli_stop_records_changes_required(tmp_path, capsys, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    _write_cli_config(
        repo,
        handoffs=[handoff("demo2", 1)],
        reviews=[review("CHANGES_REQUIRED", [finding()])],
    )
    rc = main(["start", str(task), "--task-id", "demo2", "--repo", str(repo)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "AWAITING_FIX_APPROVAL" in out

    rc = main(["stop", "demo2", "--note", "enough", "--repo", str(repo)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "CHANGES_REQUIRED" in out
    final = read_json(repo / "tools" / "ai_collab" / "runs" / "demo2" / "final.json")
    assert final["final_state"] == "CHANGES_REQUIRED"
    assert "stopped_by_user" in final["stop_reason"]
