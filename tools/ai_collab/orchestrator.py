"""The bounded collaboration loop (spec sections 6-8, 12-14, 19-21).

One Orchestrator instance drives one run. All durable state lives in
run.json + round artifacts, so a run can be resumed after interruption by
constructing the orchestrator again (``Orchestrator.load``) and calling
``advance``. ``advance`` processes steps until the run reaches a pause state
(user gate) or a terminal state; it never loops past ``max_rounds`` and every
reviewer call is budget-checked first.
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
from pathlib import Path

from . import audit, git_ops, prompts, schemas
from .artifacts import RunPaths, RunRecord
from .budget import BudgetTracker
from .cache import ReviewCache
from .claude_adapter import (
    ClaudeInvocation,
    ManualClaudeAdapter,
    extract_handoff_fallback,
    make_claude_adapter,
)
from .config import CollabConfig, config_snapshot, generation_config
from .errors import AdapterError, CollabError, ConfigError
from .graphify_ctx import collect_graphify_context
from .payload import (
    DiffBundle,
    build_reviewer_payload,
    load_requested_files,
    prepare_diff_bundle,
)
from .redaction import redact_text
from .reviewer_adapter import make_reviewer, resolved_model
from .states import (
    APPROVED,
    AWAITING_CLAUDE,
    AWAITING_FIX_APPROVAL,
    AWAITING_REVIEW_APPROVAL,
    BLOCKED,
    BUDGET_EXHAUSTED,
    CHANGES_REQUIRED,
    CLAUDE_RUNNING,
    CREATED,
    ERROR,
    H_BLOCKED,
    H_USER_APPROVAL,
    MAX_ROUNDS,
    PAUSE_STATES,
    REVIEWING,
    STOPPED,
    TERMINAL_STATES,
    TEST_FAILURE,
    USER_APPROVAL_REQUIRED,
    V_APPROVED,
    V_BLOCKED,
    V_CHANGES_REQUIRED,
)
from .util import (
    est_tokens,
    now_iso,
    read_json,
    read_text,
    sha256_text,
    slugify,
    truncate_tail,
    write_json_atomic,
)

_STEP_GUARD = 200


def default_task_id(task_path: Path, task_text: str) -> str:
    return f"{slugify(task_path.stem)}-{sha256_text(task_text)[:8]}"


class Orchestrator:
    def __init__(
        self,
        repo_root: Path,
        cfg: CollabConfig,
        run: RunRecord,
        paths: RunPaths,
        claude_adapter=None,
        reviewer=None,
    ):
        self.repo = Path(repo_root)
        self.cfg = cfg
        self.run = run
        self.paths = paths
        self._claude = claude_adapter
        self._reviewer = reviewer
        self.cache = ReviewCache(paths.cache_dir)
        self.tracker = BudgetTracker(cfg.budget, run.budget_state)

    # ------------------------------------------------------------------ setup
    @classmethod
    def start(
        cls,
        repo_root: Path,
        cfg: CollabConfig,
        task_path: Path,
        task_id: str | None = None,
        allow_dirty: bool = False,
        create_branch_name: str | None = None,
        claude_adapter=None,
        reviewer=None,
    ) -> "Orchestrator":
        repo_root = Path(repo_root)
        task_path = Path(task_path)
        if not task_path.is_file():
            raise ConfigError(f"task file not found: {task_path}")
        task_text = read_text(task_path)

        if create_branch_name:
            git_ops.ensure_repo(repo_root)
            git_ops.create_branch(repo_root, create_branch_name)
        baseline = git_ops.check_start_preconditions(
            repo_root,
            cfg.git.protected_branches,
            cfg.git.require_clean,
            allow_dirty,
        )

        tid = task_id or default_task_id(task_path, task_text)
        paths = RunPaths(repo_root, tid)
        if paths.run_dir.exists():
            raise ConfigError(
                f"run '{tid}' already exists at {paths.run_dir}; use "
                f"`continue {tid}` or pick a different --task-id"
            )
        paths.init_run_dir()
        shutil.copyfile(task_path, paths.task_md)

        run = RunRecord.new(
            paths.run_json,
            tid,
            mode=cfg.run.mode,
            max_rounds=cfg.run.max_rounds,
            baseline=baseline,
            config_snapshot=config_snapshot(cfg),
        )
        run.save()
        audit.log_event(
            paths.audit_jsonl,
            "run_started",
            task_id=tid,
            mode=cfg.run.mode,
            max_rounds=cfg.run.max_rounds,
            branch=baseline["branch"],
            base_commit=baseline["base_commit"],
            dirty_at_start=baseline["dirty_at_start"],
        )
        return cls(repo_root, cfg, run, paths, claude_adapter, reviewer)

    @classmethod
    def load(
        cls,
        repo_root: Path,
        cfg: CollabConfig,
        task_id: str,
        claude_adapter=None,
        reviewer=None,
    ) -> "Orchestrator":
        paths = RunPaths(repo_root, task_id)
        run = RunRecord.load(paths.run_json)
        return cls(repo_root, cfg, run, paths, claude_adapter, reviewer)

    # ------------------------------------------------------------- adapters
    @property
    def claude(self):
        if self._claude is None:
            self._claude = make_claude_adapter(
                self.cfg.claude,
                self.repo,
                start_index=self.run.counter("claude_attempts_completed"),
            )
        return self._claude

    @property
    def reviewer(self):
        if self._reviewer is None:
            self._reviewer = make_reviewer(
                self.cfg.reviewer,
                start_index=self.run.counter("reviewer_adapter_calls"),
            )
        return self._reviewer

    def _save_budget(self) -> None:
        self.run.set_budget_state(self.tracker.state)

    def _audit(self, event: str, **fields) -> None:
        audit.log_event(self.paths.audit_jsonl, event, **fields)

    # ------------------------------------------------------------ public API
    def advance(self, approve: bool = False, note: str = "") -> str:
        """Process until the next pause/terminal state. ``approve`` passes the
        current user gate (if any); it is consumed by exactly one gate."""
        if self.run.state in TERMINAL_STATES:
            return self.run.state
        if self.run.state in PAUSE_STATES:
            if not self._resume_from_pause(approve, note):
                return self.run.state
        guard = 0
        while (
            self.run.state not in TERMINAL_STATES
            and self.run.state not in PAUSE_STATES
        ):
            guard += 1
            if guard > _STEP_GUARD:
                raise CollabError("state machine failed to settle (guard tripped)")
            self._step()
        return self.run.state

    def stop_by_user(self, note: str = "") -> str:
        state = self.run.state
        if state in TERMINAL_STATES:
            return state
        if state == USER_APPROVAL_REQUIRED:
            final_state = USER_APPROVAL_REQUIRED
        elif self.run.data.get("last_verdict") == V_CHANGES_REQUIRED:
            final_state = CHANGES_REQUIRED
        else:
            final_state = STOPPED
        self._audit("user_stopped", at_state=state, note=note)
        reason = "stopped_by_user" + (f": {note}" if note else "")
        # Keep the machine state terminal (STOPPED) while final.json records
        # the spec stop state (e.g. USER_APPROVAL_REQUIRED / CHANGES_REQUIRED).
        self._finalize(STOPPED, reason, final_state_override=final_state)
        return final_state

    # ------------------------------------------------------------ state steps
    def _step(self) -> None:
        state = self.run.state
        if state == CREATED:
            self._begin_round(1, kind="implement")
        elif state == CLAUDE_RUNNING:
            self._run_claude_attempt()
        elif state == REVIEWING:
            self._do_review()
        else:  # pragma: no cover - defended by advance()
            raise CollabError(f"unexpected state in _step: {state}")

    def _resume_from_pause(self, approve: bool, note: str) -> bool:
        state = self.run.state
        if state == AWAITING_CLAUDE:
            handoff_path = self.paths.handoff(self.run.current_round)
            if not handoff_path.exists():
                return False  # still waiting for the manual handoff
            attempt = self.run.current_attempt()
            if attempt is not None:
                attempt["adapter_status"] = "completed_manual"
                attempt["finished_at"] = now_iso()
            self.run.bump_counter("claude_attempts_completed")
            self.run.save()
            self._audit(
                "claude_manual_completed", round=self.run.current_round
            )
            self._ingest_handoff()
            return True
        if not approve:
            return False
        self.run.add_approval(gate=state, note=note)
        self._audit("approval", gate=state, note=note)
        if state == AWAITING_REVIEW_APPROVAL:
            self.run.state = REVIEWING
            self.run.save()
            return True
        if state == AWAITING_FIX_APPROVAL:
            self._begin_round(self.run.current_round + 1, kind="fix")
            return True
        if state == USER_APPROVAL_REQUIRED:
            self._begin_attempt(kind="approved_continue", approval_note=note)
            return True
        return False  # pragma: no cover

    # ------------------------------------------------------------ Claude side
    def _begin_round(self, round_no: int, kind: str) -> None:
        self.run.data["current_round"] = round_no
        self.run.round_rec(round_no)
        self._audit("round_started", round=round_no, kind=kind)
        self._begin_attempt(kind=kind)

    def _begin_attempt(
        self,
        kind: str,
        approval_note: str = "",
        test_output_tail: str = "",
    ) -> None:
        round_no = self.run.current_round
        rec = self.run.round_rec(round_no)
        attempt_no = len(rec["claude_attempts"]) + 1
        handoff_path = self.paths.handoff(round_no)
        if handoff_path.exists():  # archive stale handoff from a prior attempt
            handoff_path.replace(
                self.paths.round_dir(round_no)
                / f"claude_handoff_superseded_a{attempt_no - 1}.json"
            )

        task_text = read_text(self.paths.task_md)
        abs_handoff = str(handoff_path.resolve())
        task_id = self.run.task_id
        max_rounds = self.run.max_rounds
        if kind == "implement":
            prompt = prompts.claude_implement_prompt(
                task_id, round_no, max_rounds, abs_handoff, task_text
            )
        elif kind == "fix":
            last = self.run.data.get("last_review") or {}
            prompt = prompts.claude_fix_prompt(
                task_id,
                round_no,
                max_rounds,
                abs_handoff,
                task_text,
                findings=last.get("findings", []),
                tests_requested=last.get("tests_requested", []),
                review_summary=last.get("summary", ""),
            )
        elif kind == "test_fix":
            prompt = prompts.claude_test_fix_prompt(
                task_id,
                round_no,
                max_rounds,
                abs_handoff,
                task_text,
                test_output_tail,
            )
        elif kind == "approved_continue":
            prompt = prompts.claude_approved_continue_prompt(
                task_id,
                round_no,
                max_rounds,
                abs_handoff,
                task_text,
                approval_note,
                self.run.data.get("last_handoff_summary", ""),
            )
        else:  # pragma: no cover
            raise CollabError(f"unknown attempt kind: {kind}")

        prompt_path = self.paths.claude_prompt(round_no, attempt_no)
        prompt_path.write_text(prompt, encoding="utf-8")
        rec["claude_attempts"].append(
            {
                "attempt": attempt_no,
                "kind": kind,
                "started_at": now_iso(),
                "adapter_status": "prepared",
                "prompt_file": prompt_path.name,
            }
        )
        self.run.state = CLAUDE_RUNNING
        self.run.save()
        self._audit(
            "claude_attempt_prepared", round=round_no, attempt=attempt_no, kind=kind
        )

    def _run_claude_attempt(self) -> None:
        round_no = self.run.current_round
        rec = self.run.round_rec(round_no)
        attempt = rec["claude_attempts"][-1]
        handoff_path = self.paths.handoff(round_no)

        # Crash recovery: the adapter ran to completion but the result was
        # never ingested. Do not re-run Claude; use the artifacts.
        if attempt["adapter_status"] == "launched" and handoff_path.exists():
            attempt["adapter_status"] = "completed_recovered"
            attempt["finished_at"] = now_iso()
            self.run.bump_counter("claude_attempts_completed")
            self.run.save()
            self._audit(
                "claude_recovered_from_artifacts",
                round=round_no,
                attempt=attempt["attempt"],
            )
            self._ingest_handoff()
            return

        adapter = self.claude
        if isinstance(adapter, ManualClaudeAdapter):
            instructions = (
                f"# Manual Claude round - task {self.run.task_id}, "
                f"round {round_no}, attempt {attempt['attempt']}\n\n"
                f"1. Open your interactive Claude Code session in this repository.\n"
                f"2. Give it the prompt stored at:\n"
                f"   {self.paths.claude_prompt(round_no, attempt['attempt'])}\n"
                f"3. Make sure it ends by writing the handoff JSON to:\n"
                f"   {handoff_path}\n"
                f"4. Then run: python -m tools.ai_collab continue {self.run.task_id}\n"
            )
            self.paths.manual_instructions(round_no).write_text(
                instructions, encoding="utf-8"
            )
            attempt["adapter_status"] = "manual_pending"
            self.run.state = AWAITING_CLAUDE
            self.run.save()
            self._audit(
                "awaiting_manual_claude", round=round_no, attempt=attempt["attempt"]
            )
            return

        attempt["adapter_status"] = "launched"
        self.run.save()
        self._audit(
            "claude_attempt_started", round=round_no, attempt=attempt["attempt"]
        )
        invocation = ClaudeInvocation(
            prompt_text=read_text(
                self.paths.claude_prompt(round_no, attempt["attempt"])
            ),
            handoff_path=handoff_path,
            round_no=round_no,
            attempt_no=attempt["attempt"],
        )
        result = adapter.run_round(invocation)

        output = result.raw_output or ""
        if result.stderr:
            output += "\n--- stderr ---\n" + result.stderr
        redacted_output, _ = redact_text(output)
        self.paths.claude_output(round_no, attempt["attempt"]).write_text(
            redacted_output, encoding="utf-8"
        )
        attempt["finished_at"] = now_iso()
        attempt["adapter_status"] = result.status
        attempt["returncode"] = result.returncode
        if result.status == "error":
            self.run.save()
            self._audit(
                "claude_attempt_error",
                round=round_no,
                attempt=attempt["attempt"],
                error=result.error,
            )
            self._finalize(ERROR, f"claude_adapter_error: {result.error}")
            return
        self.run.bump_counter("claude_attempts_completed")
        self.run.save()
        self._audit(
            "claude_attempt_finished",
            round=round_no,
            attempt=attempt["attempt"],
            returncode=result.returncode,
        )
        self._ingest_handoff(raw_stdout=result.raw_output)

    def _ingest_handoff(self, raw_stdout: str = "") -> None:
        round_no = self.run.current_round
        rec = self.run.round_rec(round_no)
        attempt = rec["claude_attempts"][-1]
        handoff_path = self.paths.handoff(round_no)

        obj = None
        source = "file"
        if handoff_path.exists():
            text = read_text(handoff_path)
            obj = schemas.parse_json_lenient(text)
        if obj is None and raw_stdout:
            obj = extract_handoff_fallback(raw_stdout)
            if obj is not None:
                source = "stdout_envelope"
                write_json_atomic(handoff_path, obj)
        if obj is None:
            self._finalize(
                ERROR,
                "malformed_claude_handoff: no parseable handoff JSON was produced",
            )
            return
        problems = schemas.validate_handoff(obj)
        if problems:
            self.run.add_note(
                f"round {round_no} handoff invalid: " + "; ".join(problems)
            )
            self.run.save()
            self._finalize(
                ERROR, "malformed_claude_handoff: " + "; ".join(problems[:5])
            )
            return

        handoff = schemas.normalize_handoff(obj)
        attempt["handoff_status"] = handoff["status"]
        self.run.data["last_handoff_summary"] = handoff.get("summary", "")
        self._audit(
            "handoff_ingested",
            round=round_no,
            status=handoff["status"],
            source=source,
        )
        if handoff["status"] == H_BLOCKED:
            self.run.save()
            self._finalize(
                BLOCKED, "claude_blocked: " + handoff.get("summary", "")[:300]
            )
            return
        if handoff["status"] == H_USER_APPROVAL:
            self.run.state = USER_APPROVAL_REQUIRED
            self.run.save()
            self._audit(
                "paused_user_approval_required",
                round=round_no,
                summary=handoff.get("summary", "")[:300],
            )
            return
        self.run.save()
        self._post_claude_capture(handoff)

    # -------------------------------------------------- capture + tests
    def _post_claude_capture(self, handoff: dict) -> None:
        round_no = self.run.current_round
        rec = self.run.round_rec(round_no)
        base = self.run.data["baseline"]["base_commit"]

        change = git_ops.capture_change_set(self.repo, base)
        bundle = prepare_diff_bundle(change, self.cfg)
        diff_artifact = bundle.diff_text
        if bundle.untracked_text:
            diff_artifact += "\n=== UNTRACKED FILES ===\n" + bundle.untracked_text
        self.paths.diff_patch(round_no).write_text(diff_artifact, encoding="utf-8")
        write_json_atomic(self.paths.changed_files(round_no), bundle.changed_files)
        write_json_atomic(
            self.paths.round_dir(round_no) / "bundle.json",
            dataclasses.asdict(bundle),
        )

        reported = set(handoff.get("files_changed") or [])
        actual = {path for _status, path in change.name_status} | set(change.untracked)
        unreported = sorted(actual - reported)[:20]
        if unreported:
            self.run.add_note(
                f"round {round_no}: changed but missing from handoff report: "
                + ", ".join(unreported)
            )
        self._audit(
            "change_captured",
            round=round_no,
            base=change.base,
            head=change.head,
            files=len(actual),
            diff_chars=len(bundle.diff_text),
            excluded_secret_paths=bundle.changed_files["excluded_secret_paths"],
        )

        if self.cfg.tests.commands:
            passed, output = self._run_tests(round_no)
            rec["tests"]["passed"] = passed
            if not passed:
                used = int(rec["tests"].get("fix_attempts_used", 0))
                if used < self.cfg.run.test_fix_attempts:
                    rec["tests"]["fix_attempts_used"] = used + 1
                    self.run.save()
                    self._audit(
                        "test_fix_attempt", round=round_no, attempt=used + 1
                    )
                    tail, _ = truncate_tail(
                        output, self.cfg.payload.max_test_output_chars
                    )
                    self._begin_attempt(kind="test_fix", test_output_tail=tail)
                    return
                if self.cfg.run.on_test_failure == "stop":
                    self.run.save()
                    self._finalize(
                        TEST_FAILURE,
                        "tests still failing after the bounded fix attempt",
                    )
                    return
                self.run.add_note(
                    f"round {round_no}: proceeding to review with failing tests "
                    "(on_test_failure=review)"
                )
        else:
            rec["tests"]["passed"] = None

        if self.run.mode == "manual":
            self.run.state = AWAITING_REVIEW_APPROVAL
            self.run.save()
            self._audit("paused_review_approval", round=round_no)
        else:
            self.run.state = REVIEWING
            self.run.save()

    def _run_tests(self, round_no: int) -> tuple[bool, str]:
        rec = self.run.round_rec(round_no)
        parts: list[str] = []
        codes: list[int] = []
        for command in self.cfg.tests.commands:
            try:
                proc = subprocess.run(
                    command,
                    cwd=str(self.repo),
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=self.cfg.tests.timeout_seconds,
                )
                code = proc.returncode
                out = proc.stdout + ("\n" + proc.stderr if proc.stderr else "")
            except subprocess.TimeoutExpired:
                code, out = -1, f"[test command timed out after {self.cfg.tests.timeout_seconds}s]"
            except OSError as exc:
                code, out = -2, f"[test command failed to start: {exc}]"
            parts.append(f"$ {' '.join(command)}\n(exit {code})\n{out}\n")
            codes.append(code)
        combined, _ = redact_text("\n".join(parts), self.cfg.redaction.extra_patterns)
        self.paths.tests_txt(round_no).write_text(combined, encoding="utf-8")
        rec["tests"]["commands"] = [" ".join(c) for c in self.cfg.tests.commands]
        rec["tests"]["exit_codes"] = codes
        self._audit("tests_run", round=round_no, exit_codes=codes)
        return all(code == 0 for code in codes), combined

    # ------------------------------------------------------------- review side
    def _do_review(self) -> None:
        round_no = self.run.current_round
        rec = self.run.round_rec(round_no)
        review_rec = rec["review"]

        # Crash recovery: a completed review.json means the verdict was
        # reached; just re-apply the transition.
        review_json_path = self.paths.review_json(round_no)
        if review_json_path.exists():
            stored = read_json(review_json_path)
            review = stored.get("review", {})
            self.run.data["last_review"] = {
                "round": round_no,
                "summary": review.get("summary", ""),
                "findings": review.get("findings", []),
                "tests_requested": review.get("tests_requested", []),
            }
            self._audit("review_reloaded", round=round_no)
            self._handle_verdict(stored.get("effective_verdict", V_BLOCKED))
            return

        try:
            self._review_round(round_no, rec, review_rec)
        except AdapterError as exc:
            self.run.save()
            self._finalize(ERROR, f"reviewer_adapter_error: {exc}")

    def _review_round(self, round_no: int, rec: dict, review_rec: dict) -> None:
        cfg = self.cfg
        task_text = read_text(self.paths.task_md)
        context_path = self.repo / cfg.payload.context_file
        context_text = (
            read_text(context_path)
            if context_path.is_file()
            else "(project reviewer context file is missing)"
        )
        handoff = schemas.normalize_handoff(
            schemas.parse_json_lenient(read_text(self.paths.handoff(round_no))) or {}
        )
        bundle_data = read_json(self.paths.round_dir(round_no) / "bundle.json")
        bundle = DiffBundle(**bundle_data)
        tests_path = self.paths.tests_txt(round_no)
        test_output = read_text(tests_path) if tests_path.is_file() else ""

        graphify_text = ""
        if cfg.graphify.enabled:
            graphify_path = self.paths.graphify_txt(round_no)
            if graphify_path.is_file():
                graphify_text = read_text(graphify_path)
            else:
                graphify_text = collect_graphify_context(self.repo, cfg.graphify)
                if graphify_text:
                    graphify_path.write_text(graphify_text, encoding="utf-8")

        if cfg.reviewer.backend == "mock":
            model = "mock"
        else:
            model = resolved_model(cfg.reviewer)  # raises AdapterError if unset

        system = prompts.reviewer_system_prompt(cfg.policy.block_on)
        payload = build_reviewer_payload(
            cfg,
            system,
            task_text,
            context_text,
            handoff,
            bundle,
            test_output,
            graphify_text,
        )
        fp_parts = {
            "prompt_version": prompts.PROMPT_VERSION,
            "model": model,
            "generation": generation_config(cfg.reviewer),
            "task_sha256": payload.hashes["task_sha256"],
            "diff_sha256": payload.hashes["diff_sha256"],
            "context_sha256": payload.hashes["context_sha256"],
            "tests_sha256": payload.hashes["tests_sha256"],
        }
        fingerprint = ReviewCache.fingerprint(fp_parts)
        meta = {
            "fingerprint": fingerprint,
            "fingerprint_parts": fp_parts,
            "est_input_tokens": payload.est_input_tokens,
            "section_sizes": payload.section_sizes,
            "truncations": payload.truncations,
            "redaction_counts": payload.redaction_counts,
            "hashes": payload.hashes,
            "cached": False,
            "calls": [],
        }
        self.paths.reviewer_request(round_no).write_text(
            payload.system + "\n\n========== USER MESSAGE ==========\n\n" + payload.user,
            encoding="utf-8",
        )

        review = None
        raw_text = ""
        cached_entry = self.cache.get(fingerprint)
        if cached_entry is not None:
            review = schemas.normalize_review(cached_entry["review"])
            raw_text = "(served from local review cache)"
            meta["cached"] = True
            review_rec["cached"] = True
            self._audit("reviewer_cache_hit", round=round_no, fingerprint=fingerprint)
        else:
            review, raw_text, fail_reason = self._call_reviewer_with_retries(
                payload, meta
            )
            if review is None:
                write_json_atomic(self.paths.reviewer_request_meta(round_no), meta)
                self.run.save()
                if fail_reason.startswith("budget:"):
                    self._finalize(BUDGET_EXHAUSTED, fail_reason[len("budget:") :])
                else:
                    self._finalize(ERROR, "malformed_reviewer_output: " + fail_reason)
                return
            self.cache.put(
                fingerprint,
                review,
                {"model": model, "round": round_no, "task_id": self.run.task_id},
            )

        # Optional bounded context follow-up (one per round).
        followup_requests = review.get("context_requests") or []
        if (
            followup_requests
            and cfg.reviewer.allow_context_followup
            and not review_rec.get("followup_used")
        ):
            review_rec["followup_used"] = True
            review = self._context_followup(
                round_no,
                review,
                followup_requests,
                system,
                task_text,
                context_text,
                handoff,
                bundle,
                test_output,
                graphify_text,
                fp_parts,
                meta,
            )

        effective, policy_note = self._apply_policy(review)
        self.paths.review_raw(round_no).write_text(raw_text, encoding="utf-8")
        write_json_atomic(
            self.paths.review_json(round_no),
            {
                "review": review,
                "effective_verdict": effective,
                "policy_note": policy_note,
                "cached": meta["cached"],
            },
        )
        write_json_atomic(self.paths.reviewer_request_meta(round_no), meta)

        review_rec.update(
            {
                "verdict": review["verdict"],
                "effective_verdict": effective,
                "findings": len(review.get("findings", [])),
                "fingerprint": fingerprint,
                "cached": meta["cached"],
                "policy_note": policy_note,
            }
        )
        self.run.data["last_review"] = {
            "round": round_no,
            "summary": review.get("summary", ""),
            "findings": review.get("findings", []),
            "tests_requested": review.get("tests_requested", []),
        }
        self.run.save()
        self._audit(
            "review_completed",
            round=round_no,
            verdict=review["verdict"],
            effective_verdict=effective,
            findings=len(review.get("findings", [])),
            cached=meta["cached"],
        )
        self._handle_verdict(effective)

    def _call_reviewer_with_retries(
        self, payload, meta: dict
    ) -> tuple[dict | None, str, str]:
        cfg = self.cfg
        attempts_allowed = 1 + max(0, cfg.reviewer.max_retries)
        user = payload.user
        raw_text = ""
        problems: list[str] = ["reviewer was never called"]
        for attempt_index in range(attempts_allowed):
            est = est_tokens(payload.system) + est_tokens(user)
            reason = self.tracker.check_next_call(est)
            if reason:
                meta["calls"].append(
                    {"ts": now_iso(), "kind": "skipped_budget", "reason": reason}
                )
                self._audit("budget_stop", reason=reason)
                return None, raw_text, "budget:" + reason
            result = self.reviewer.call(payload.system, user)
            self.run.bump_counter("reviewer_adapter_calls")
            input_tokens = (
                result.usage.input_tokens
                if isinstance(result.usage.input_tokens, int)
                else est
            )
            output_tokens = (
                result.usage.output_tokens
                if isinstance(result.usage.output_tokens, int)
                else est_tokens(result.raw_text)
            )
            self.tracker.record_call(input_tokens, output_tokens, result.usage.cost_usd)
            self._save_budget()
            self.run.save()
            call_meta = {
                "ts": now_iso(),
                "kind": "initial" if attempt_index == 0 else f"retry_{attempt_index}",
                "model": result.model,
                "usage": result.usage.to_dict(),
                "est_input_tokens": est,
            }
            meta["calls"].append(call_meta)
            self._audit(
                "reviewer_call",
                kind=call_meta["kind"],
                model=result.model,
                usage=result.usage.to_dict(),
                est_input_tokens=est,
            )
            raw_text = result.raw_text
            obj = schemas.parse_json_lenient(raw_text)
            problems = (
                schemas.validate_review(obj)
                if obj is not None
                else ["no JSON object found in reviewer reply"]
            )
            if not problems:
                return schemas.normalize_review(obj), raw_text, ""
            self._audit(
                "reviewer_reply_malformed",
                attempt=attempt_index,
                problems=problems[:5],
            )
            user = payload.user + prompts.reviewer_retry_note(problems)
        return None, raw_text, "; ".join(problems[:5])

    def _context_followup(
        self,
        round_no: int,
        review: dict,
        requests: list[str],
        system: str,
        task_text: str,
        context_text: str,
        handoff: dict,
        bundle: DiffBundle,
        test_output: str,
        graphify_text: str,
        fp_parts: dict,
        meta: dict,
    ) -> dict:
        """One bounded follow-up with requested source files. Never fatal:
        on any problem the original review stands."""
        files, notes = load_requested_files(self.repo, requests, self.cfg)
        for note in notes:
            self.run.add_note(f"round {round_no} followup: {note}")
        self._audit(
            "context_followup",
            round=round_no,
            requested=requests,
            served=sorted(files),
            notes=notes,
        )
        if not files:
            return review

        followup_payload = build_reviewer_payload(
            self.cfg,
            system,
            task_text,
            context_text,
            handoff,
            bundle,
            test_output,
            graphify_text,
            requested_files=files,
            followup=True,
        )
        followup_parts = dict(
            fp_parts,
            followup_files_sha256=sha256_text(
                "\n".join(f"{p}:{sha256_text(c)}" for p, c in sorted(files.items()))
            ),
        )
        fingerprint = ReviewCache.fingerprint(followup_parts)
        self.paths.round_dir(round_no).joinpath(
            "reviewer_request_followup.txt"
        ).write_text(
            followup_payload.system
            + "\n\n========== USER MESSAGE ==========\n\n"
            + followup_payload.user,
            encoding="utf-8",
        )

        cached_entry = self.cache.get(fingerprint)
        if cached_entry is not None:
            meta["calls"].append(
                {"ts": now_iso(), "kind": "followup_cache_hit", "fingerprint": fingerprint}
            )
            self._audit("reviewer_cache_hit", round=round_no, fingerprint=fingerprint)
            return schemas.normalize_review(cached_entry["review"])

        est = est_tokens(followup_payload.system) + est_tokens(followup_payload.user)
        reason = self.tracker.check_next_call(est)
        if reason:
            self.run.add_note(
                f"round {round_no}: follow-up skipped (budget: {reason}); "
                "original review stands"
            )
            meta["calls"].append(
                {"ts": now_iso(), "kind": "followup_skipped_budget", "reason": reason}
            )
            return review
        try:
            result = self.reviewer.call(followup_payload.system, followup_payload.user)
        except AdapterError as exc:
            self.run.add_note(
                f"round {round_no}: follow-up failed ({exc}); original review stands"
            )
            return review
        self.run.bump_counter("reviewer_adapter_calls")
        input_tokens = (
            result.usage.input_tokens
            if isinstance(result.usage.input_tokens, int)
            else est
        )
        output_tokens = (
            result.usage.output_tokens
            if isinstance(result.usage.output_tokens, int)
            else est_tokens(result.raw_text)
        )
        self.tracker.record_call(input_tokens, output_tokens, result.usage.cost_usd)
        self._save_budget()
        self.run.save()
        meta["calls"].append(
            {
                "ts": now_iso(),
                "kind": "followup",
                "model": result.model,
                "usage": result.usage.to_dict(),
                "est_input_tokens": est,
            }
        )
        self._audit(
            "reviewer_call",
            kind="followup",
            model=result.model,
            usage=result.usage.to_dict(),
            est_input_tokens=est,
        )
        obj = schemas.parse_json_lenient(result.raw_text)
        problems = (
            schemas.validate_review(obj)
            if obj is not None
            else ["no JSON object found"]
        )
        if problems:
            self.run.add_note(
                f"round {round_no}: follow-up reply malformed; original review stands"
            )
            return review
        final_review = schemas.normalize_review(obj)
        final_review["context_requests"] = []  # follow-ups are exhausted this round
        self.cache.put(
            fingerprint,
            final_review,
            {"model": result.model, "round": round_no, "task_id": self.run.task_id},
        )
        return final_review

    def _apply_policy(self, review: dict) -> tuple[str, str]:
        verdict = review["verdict"]
        if verdict == V_APPROVED:
            blocking = [
                f.get("id", "?")
                for f in review.get("findings", [])
                if f.get("severity") in self.cfg.policy.block_on
            ]
            if blocking:
                return (
                    V_CHANGES_REQUIRED,
                    "APPROVED downgraded by severity policy; blocking findings: "
                    + ", ".join(blocking),
                )
        return verdict, ""

    def _handle_verdict(self, effective: str) -> None:
        round_no = self.run.current_round
        if effective == V_APPROVED:
            self.run.data["last_verdict"] = V_APPROVED
            self._finalize(APPROVED, "reviewer_approved")
            return
        if effective == V_BLOCKED:
            self.run.data["last_verdict"] = V_BLOCKED
            self._finalize(BLOCKED, "reviewer_blocked")
            return
        self.run.data["last_verdict"] = V_CHANGES_REQUIRED
        if round_no >= self.run.max_rounds:
            self._finalize(
                MAX_ROUNDS,
                f"max_rounds ({self.run.max_rounds}) reached with changes still required",
            )
            return
        over = self.tracker.post_call_exceeded()
        if over:
            self._finalize(BUDGET_EXHAUSTED, over)
            return
        if self.run.mode == "auto_bounded":
            self._begin_round(round_no + 1, kind="fix")
        else:
            self.run.state = AWAITING_FIX_APPROVAL
            self.run.save()
            self._audit("paused_fix_approval", round=round_no)

    # ------------------------------------------------------------- terminal
    def _finalize(
        self, state: str, reason: str, final_state_override: str | None = None
    ) -> None:
        self.run.state = state
        self.run.data["stop_reason"] = reason
        self._save_budget()
        self.run.save()
        final = {
            "task_id": self.run.task_id,
            "final_state": final_state_override or state,
            "machine_state": state,
            "stop_reason": reason,
            "last_verdict": self.run.data.get("last_verdict"),
            "rounds": len(self.run.data.get("rounds", [])),
            "current_round": self.run.current_round,
            "budget": self.run.data.get("budget"),
            "finished_at": now_iso(),
        }
        write_json_atomic(self.paths.final_json, final)
        self._audit(
            "finalized",
            state=state,
            final_state=final["final_state"],
            reason=reason,
        )
