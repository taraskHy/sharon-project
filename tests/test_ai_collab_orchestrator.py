"""End-to-end orchestrator scenarios with mock adapters (offline, no network).

Covers the spec's required matrix: one-round approval, changes-then-approval,
max rounds, budget exhaustion, test failure, malformed handoff, malformed
reviewer output, reviewer blocked, dirty tree, request caching, secret
redaction, exact diff capture, and resume after interruption.
"""

import sys

import pytest

from _ai_collab_helpers import (
    base_cfg,
    finding,
    handoff,
    make_repo,
    make_task,
    mutate_marker,
    review,
)
from tools.ai_collab.claude_adapter import MockAction, MockClaudeAdapter
from tools.ai_collab.errors import GitSafetyError
from tools.ai_collab.orchestrator import Orchestrator
from tools.ai_collab.reviewer_adapter import MockReviewer
from tools.ai_collab.states import (
    APPROVED,
    AWAITING_CLAUDE,
    AWAITING_FIX_APPROVAL,
    AWAITING_REVIEW_APPROVAL,
    BLOCKED,
    BUDGET_EXHAUSTED,
    CHANGES_REQUIRED,
    ERROR,
    MAX_ROUNDS,
    STOPPED,
    TEST_FAILURE,
    USER_APPROVAL_REQUIRED,
)
from tools.ai_collab.util import read_json


def start(repo, cfg, task, claude, reviewer, tid="t1", **kw):
    return Orchestrator.start(
        repo, cfg, task, task_id=tid, claude_adapter=claude, reviewer=reviewer, **kw
    )


def test_one_round_approval(tmp_path, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    cfg = base_cfg()
    claude = MockClaudeAdapter(
        [MockAction(handoff=handoff("t1", 1), mutate=mutate_marker("round-1"))], repo
    )
    reviewer = MockReviewer([review("APPROVED")])
    orch = start(repo, cfg, task, claude, reviewer)

    assert orch.advance() == APPROVED

    # artifacts
    assert orch.paths.final_json.is_file()
    final = read_json(orch.paths.final_json)
    assert final["final_state"] == APPROVED
    assert final["stop_reason"] == "reviewer_approved"
    diff = orch.paths.diff_patch(1).read_text(encoding="utf-8")
    assert "+# round-1" in diff
    stored = read_json(orch.paths.review_json(1))
    assert stored["effective_verdict"] == APPROVED
    # the reviewer saw the REAL diff, not a self-report
    assert "+# round-1" in reviewer.requests[0][1]
    assert orch.run.data["budget"]["reviewer_calls"] == 1
    assert orch.paths.audit_jsonl.is_file()


def test_changes_required_then_approval_auto(tmp_path, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    cfg = base_cfg(mode="auto_bounded")
    the_finding = finding(issue="guard against None inputs")
    claude = MockClaudeAdapter(
        [
            MockAction(handoff=handoff("t1", 1), mutate=mutate_marker("round-1")),
            MockAction(handoff=handoff("t1", 2), mutate=mutate_marker("round-2-fix")),
        ],
        repo,
    )
    reviewer = MockReviewer(
        [review("CHANGES_REQUIRED", [the_finding]), review("APPROVED")]
    )
    orch = start(repo, cfg, task, claude, reviewer)

    assert orch.advance() == APPROVED
    assert orch.run.current_round == 2
    # the fix round prompt carried the structured findings, scope-limited
    fix_prompt = claude.prompts[1]
    assert "guard against None inputs" in fix_prompt
    assert "SCOPE FOR THIS ROUND" in fix_prompt
    # reviewer round 2 saw the cumulative real diff
    assert "+# round-2-fix" in reviewer.requests[1][1]
    assert orch.run.data["budget"]["reviewer_calls"] == 2


def test_semi_auto_pauses_before_fixes_and_resumes(tmp_path, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    cfg = base_cfg(mode="semi_auto")
    claude = MockClaudeAdapter(
        [MockAction(handoff=handoff("t1", 1), mutate=mutate_marker("round-1"))], repo
    )
    reviewer = MockReviewer([review("CHANGES_REQUIRED", [finding()])])
    orch = start(repo, cfg, task, claude, reviewer)
    assert orch.advance() == AWAITING_FIX_APPROVAL

    # resume in a NEW orchestrator (fresh process simulation)
    claude2 = MockClaudeAdapter(
        [MockAction(handoff=handoff("t1", 2), mutate=mutate_marker("round-2"))], repo
    )
    reviewer2 = MockReviewer([review("APPROVED")])
    orch2 = Orchestrator.load(
        repo, cfg, "t1", claude_adapter=claude2, reviewer=reviewer2
    )
    assert orch2.run.state == AWAITING_FIX_APPROVAL
    assert orch2.advance(approve=True, note="go ahead") == APPROVED
    assert orch2.run.data["approvals"][0]["gate"] == AWAITING_FIX_APPROVAL


def test_manual_mode_gates(tmp_path, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    cfg = base_cfg(mode="manual")
    cfg.claude.mode = "manual"  # factory builds the manual adapter
    reviewer = MockReviewer([review("CHANGES_REQUIRED", [finding()])])
    orch = Orchestrator.start(
        repo, cfg, task, task_id="t1", reviewer=reviewer
    )
    assert orch.advance() == AWAITING_CLAUDE
    assert orch.paths.manual_instructions(1).is_file()
    assert orch.paths.claude_prompt(1, 1).is_file()

    # the user drives their own Claude session: repo edit + handoff file
    mutate_marker("manual-edit")(repo)
    from tools.ai_collab.util import write_json_atomic

    write_json_atomic(orch.paths.handoff(1), handoff("t1", 1))
    assert orch.advance() == AWAITING_REVIEW_APPROVAL  # manual gate pre-review
    assert orch.advance() == AWAITING_REVIEW_APPROVAL  # continue without approve holds
    assert orch.advance(approve=True) == AWAITING_FIX_APPROVAL
    assert "+# manual-edit" in reviewer.requests[0][1]


def test_max_rounds_stop(tmp_path, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    cfg = base_cfg(mode="auto_bounded", max_rounds=2)
    claude = MockClaudeAdapter(
        [
            MockAction(handoff=handoff("t1", 1), mutate=mutate_marker("r1")),
            MockAction(handoff=handoff("t1", 2), mutate=mutate_marker("r2")),
        ],
        repo,
    )
    reviewer = MockReviewer(
        [
            review("CHANGES_REQUIRED", [finding("F1")]),
            review("CHANGES_REQUIRED", [finding("F2")]),
        ]
    )
    orch = start(repo, cfg, task, claude, reviewer)
    assert orch.advance() == MAX_ROUNDS
    final = read_json(orch.paths.final_json)
    assert final["final_state"] == MAX_ROUNDS
    assert final["last_verdict"] == "CHANGES_REQUIRED"
    assert orch.run.data["budget"]["reviewer_calls"] == 2


def test_budget_exhaustion_stops_before_call(tmp_path, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    cfg = base_cfg(mode="auto_bounded", max_rounds=3)
    cfg.budget.max_reviewer_calls = 1
    claude = MockClaudeAdapter(
        [
            MockAction(handoff=handoff("t1", 1), mutate=mutate_marker("r1")),
            MockAction(handoff=handoff("t1", 2), mutate=mutate_marker("r2")),
        ],
        repo,
    )
    reviewer = MockReviewer([review("CHANGES_REQUIRED", [finding()])])
    orch = start(repo, cfg, task, claude, reviewer)
    assert orch.advance() == BUDGET_EXHAUSTED
    assert "max_reviewer_calls" in orch.run.data["stop_reason"]
    assert len(reviewer.requests) == 1  # the second call was never made


def test_test_failure_after_bounded_fix(tmp_path, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    cfg = base_cfg()
    cfg.tests.commands = [[sys.executable, "-c", "import sys; sys.exit(1)"]]
    claude = MockClaudeAdapter(
        [
            MockAction(handoff=handoff("t1", 1), mutate=mutate_marker("r1")),
            MockAction(handoff=handoff("t1", 1)),  # bounded fix attempt, no fix
        ],
        repo,
    )
    reviewer = MockReviewer([])
    orch = start(repo, cfg, task, claude, reviewer)
    assert orch.advance() == TEST_FAILURE
    assert len(claude.calls) == 2  # exactly one bounded fix attempt
    assert reviewer.requests == []  # reviewer never called
    assert orch.paths.tests_txt(1).is_file()
    assert "test_fix" in claude.prompts[1] or "FAILING TEST OUTPUT" in claude.prompts[1]


def test_test_failure_then_pass(tmp_path, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    cfg = base_cfg()
    check = "import sys, os; sys.exit(0 if os.path.exists('ok.marker') else 1)"
    cfg.tests.commands = [[sys.executable, "-c", check]]

    def create_marker(repo_path):
        (repo_path / "ok.marker").write_text("ok\n", encoding="utf-8")

    claude = MockClaudeAdapter(
        [
            MockAction(handoff=handoff("t1", 1), mutate=mutate_marker("r1")),
            MockAction(handoff=handoff("t1", 1), mutate=create_marker),
        ],
        repo,
    )
    reviewer = MockReviewer([review("APPROVED")])
    orch = start(repo, cfg, task, claude, reviewer)
    assert orch.advance() == APPROVED
    rec = orch.run.round_rec(1)
    assert rec["tests"]["fix_attempts_used"] == 1
    assert rec["tests"]["passed"] is True


def test_malformed_handoff_errors(tmp_path, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    cfg = base_cfg()
    claude = MockClaudeAdapter(
        [MockAction(handoff_text="this is definitely not json")], repo
    )
    orch = start(repo, cfg, task, claude, MockReviewer([]))
    assert orch.advance() == ERROR
    assert orch.run.data["stop_reason"].startswith("malformed_claude_handoff")


def test_missing_handoff_errors(tmp_path, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    cfg = base_cfg()
    claude = MockClaudeAdapter([MockAction(skip_handoff=True, raw="no json")], repo)
    orch = start(repo, cfg, task, claude, MockReviewer([]))
    assert orch.advance() == ERROR
    assert "malformed_claude_handoff" in orch.run.data["stop_reason"]


def test_malformed_reviewer_retry_then_success(tmp_path, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    cfg = base_cfg()
    claude = MockClaudeAdapter(
        [MockAction(handoff=handoff("t1", 1), mutate=mutate_marker("r1"))], repo
    )
    reviewer = MockReviewer(["utter garbage, no json", review("APPROVED")])
    orch = start(repo, cfg, task, claude, reviewer)
    assert orch.advance() == APPROVED
    assert orch.run.data["budget"]["reviewer_calls"] == 2
    meta = read_json(orch.paths.reviewer_request_meta(1))
    kinds = [c["kind"] for c in meta["calls"]]
    assert kinds == ["initial", "retry_1"]
    # the retry carried a strict-format reminder
    assert "STRICT FORMAT REMINDER" in reviewer.requests[1][1]


def test_malformed_reviewer_exhausts_retries(tmp_path, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    cfg = base_cfg()
    claude = MockClaudeAdapter(
        [MockAction(handoff=handoff("t1", 1), mutate=mutate_marker("r1"))], repo
    )
    reviewer = MockReviewer(["garbage one", "garbage two"])
    orch = start(repo, cfg, task, claude, reviewer)
    assert orch.advance() == ERROR
    assert orch.run.data["stop_reason"].startswith("malformed_reviewer_output")
    assert orch.run.data["budget"]["reviewer_calls"] == 2


def test_reviewer_blocked(tmp_path, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    cfg = base_cfg()
    claude = MockClaudeAdapter(
        [MockAction(handoff=handoff("t1", 1), mutate=mutate_marker("r1"))], repo
    )
    reviewer = MockReviewer([review("BLOCKED", summary="diff unreviewable")])
    orch = start(repo, cfg, task, claude, reviewer)
    assert orch.advance() == BLOCKED
    assert read_json(orch.paths.final_json)["stop_reason"] == "reviewer_blocked"


def test_policy_downgrades_approval_with_blocking_findings(tmp_path, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    cfg = base_cfg(mode="semi_auto")
    claude = MockClaudeAdapter(
        [MockAction(handoff=handoff("t1", 1), mutate=mutate_marker("r1"))], repo
    )
    # reviewer says APPROVED but reports a high-severity finding
    reviewer = MockReviewer([review("APPROVED", [finding(severity="high")])])
    orch = start(repo, cfg, task, claude, reviewer)
    assert orch.advance() == AWAITING_FIX_APPROVAL
    stored = read_json(orch.paths.review_json(1))
    assert stored["effective_verdict"] == "CHANGES_REQUIRED"
    assert "downgraded" in stored["policy_note"]


def test_user_approval_required_then_continue(tmp_path, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    cfg = base_cfg()
    claude = MockClaudeAdapter(
        [
            MockAction(
                handoff=handoff(
                    "t1", 1, status="USER_APPROVAL_REQUIRED",
                    summary="found an unrelated broken invariant",
                )
            ),
            MockAction(handoff=handoff("t1", 1), mutate=mutate_marker("approved-work")),
        ],
        repo,
    )
    reviewer = MockReviewer([review("APPROVED")])
    orch = start(repo, cfg, task, claude, reviewer)
    assert orch.advance() == USER_APPROVAL_REQUIRED
    assert orch.advance() == USER_APPROVAL_REQUIRED  # plain continue holds
    assert orch.advance(approve=True, note="scope approved") == APPROVED
    assert "explicitly approved" in claude.prompts[1]
    assert "scope approved" in claude.prompts[1]


def test_user_approval_required_then_stop(tmp_path, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    cfg = base_cfg()
    claude = MockClaudeAdapter(
        [MockAction(handoff=handoff("t1", 1, status="USER_APPROVAL_REQUIRED"))], repo
    )
    orch = start(repo, cfg, task, claude, MockReviewer([]))
    assert orch.advance() == USER_APPROVAL_REQUIRED
    final_state = orch.stop_by_user(note="not now")
    assert final_state == USER_APPROVAL_REQUIRED
    final = read_json(orch.paths.final_json)
    assert final["final_state"] == USER_APPROVAL_REQUIRED
    assert final["machine_state"] == STOPPED


def test_stop_with_pending_changes_required(tmp_path, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    cfg = base_cfg(mode="semi_auto")
    claude = MockClaudeAdapter(
        [MockAction(handoff=handoff("t1", 1), mutate=mutate_marker("r1"))], repo
    )
    reviewer = MockReviewer([review("CHANGES_REQUIRED", [finding()])])
    orch = start(repo, cfg, task, claude, reviewer)
    assert orch.advance() == AWAITING_FIX_APPROVAL
    assert orch.stop_by_user() == CHANGES_REQUIRED
    assert read_json(orch.paths.final_json)["final_state"] == CHANGES_REQUIRED


def test_dirty_tree_refused_unless_allowed(tmp_path, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    cfg = base_cfg()
    (repo / "app.py").write_text("uncommitted local edit\n", encoding="utf-8")
    with pytest.raises(GitSafetyError):
        start(repo, cfg, task, MockClaudeAdapter([], repo), MockReviewer([]))
    orch = start(
        repo, cfg, task,
        MockClaudeAdapter([], repo), MockReviewer([]),
        allow_dirty=True,
    )
    assert orch.run.data["baseline"]["dirty_at_start"] is True


def test_review_cache_reused_across_runs(tmp_path, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    cfg = base_cfg()
    claude1 = MockClaudeAdapter(
        [MockAction(handoff=handoff("t1", 1), mutate=mutate_marker("same-change"))],
        repo,
    )
    reviewer1 = MockReviewer([review("APPROVED")])
    orch1 = start(repo, cfg, task, claude1, reviewer1, tid="t1")
    assert orch1.advance() == APPROVED
    assert len(reviewer1.requests) == 1

    # Second run: identical task text, identical diff -> cache hit, zero calls.
    claude2 = MockClaudeAdapter([MockAction(handoff=handoff("t2", 1))], repo)
    reviewer2 = MockReviewer([])  # would raise if ever called
    orch2 = start(
        repo, cfg, task, claude2, reviewer2, tid="t2", allow_dirty=True
    )
    assert orch2.advance() == APPROVED
    assert reviewer2.requests == []
    assert orch2.run.data["budget"]["reviewer_calls"] == 0
    assert read_json(orch2.paths.review_json(1))["cached"] is True


def test_context_followup_flow(tmp_path, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    cfg = base_cfg()
    claude = MockClaudeAdapter(
        [MockAction(handoff=handoff("t1", 1), mutate=mutate_marker("r1"))], repo
    )
    first = review(
        "CHANGES_REQUIRED", [finding()], context_requests=["app.py"]
    )
    reviewer = MockReviewer([first, review("APPROVED")])
    orch = start(repo, cfg, task, claude, reviewer)
    assert orch.advance() == APPROVED  # follow-up verdict replaced the first
    assert len(reviewer.requests) == 2
    assert "FILE: app.py" in reviewer.requests[1][1]
    assert orch.run.data["budget"]["reviewer_calls"] == 2
    assert orch.run.round_rec(1)["review"]["followup_used"] is True


def test_resume_after_crash_mid_claude(tmp_path, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    cfg = base_cfg()

    class CrashingClaude(MockClaudeAdapter):
        def run_round(self, inv):
            super().run_round(inv)  # writes handoff + mutates
            raise RuntimeError("simulated crash before ingest")

    claude = CrashingClaude(
        [MockAction(handoff=handoff("t1", 1), mutate=mutate_marker("r1"))], repo
    )
    orch = start(repo, cfg, task, claude, MockReviewer([]))
    with pytest.raises(RuntimeError):
        orch.advance()
    assert orch.run.state == "CLAUDE_RUNNING"  # persisted mid-flight state

    # Fresh orchestrator: must recover from artifacts WITHOUT re-running Claude.
    silent_claude = MockClaudeAdapter([], repo)  # errors if ever called
    reviewer = MockReviewer([review("APPROVED")])
    orch2 = Orchestrator.load(
        repo, cfg, "t1", claude_adapter=silent_claude, reviewer=reviewer
    )
    assert orch2.advance() == APPROVED
    assert silent_claude.calls == []
    assert len(reviewer.requests) == 1


def test_secret_redaction_end_to_end(tmp_path, no_network):
    repo = make_repo(tmp_path)
    task = make_task(tmp_path)
    cfg = base_cfg()
    secret = "sk-or-abcdefghij1234567890"

    def leak_secret(repo_path):
        app = repo_path / "app.py"
        app.write_text(
            app.read_text(encoding="utf-8") + f'\nAPI_KEY = "{secret}"\n',
            encoding="utf-8",
        )
        (repo_path / ".env").write_text(
            f"OPENROUTER_API_KEY={secret}\n", encoding="utf-8"
        )

    claude = MockClaudeAdapter(
        [MockAction(handoff=handoff("t1", 1), mutate=leak_secret)], repo
    )
    reviewer = MockReviewer([review("APPROVED")])
    orch = start(repo, cfg, task, claude, reviewer)
    assert orch.advance() == APPROVED

    sent = reviewer.requests[0][1]
    assert secret not in sent
    assert "[REDACTED:openrouter_key]" in sent
    assert "EXCLUDED" in sent  # .env dropped from the untracked section
    # persisted artifacts are scrubbed too
    for path in (orch.paths.diff_patch(1), orch.paths.reviewer_request(1)):
        assert secret not in path.read_text(encoding="utf-8")
    changed = read_json(orch.paths.changed_files(1))
    assert ".env" in changed["excluded_secret_paths"]
