"""Run directory layout and the persisted run record (run.json).

    tools/ai_collab/runs/<task-id>/
        task.md                     immutable task snapshot
        run.json                    state machine + budget + round metadata
        audit.jsonl                 append-only audit log
        final.json                  written exactly once, at a terminal state
        round_01/
            claude_prompt_a1.md     prompt for attempt 1 (a2 = bounded fix, ...)
            claude_output_a1.txt    raw adapter stdout/stderr
            claude_handoff.json     structured handoff (latest attempt wins)
            MANUAL_INSTRUCTIONS.md  only in manual Claude mode
            diff.patch              redacted, size-capped orchestrator diff
            changed_files.json      orchestrator-captured change list
            tests.txt               orchestrator-run test output
            graphify.txt            optional architecture notes
            reviewer_request.txt    redacted payload as sent (system + user)
            reviewer_request_meta.json  sizes, hashes, fingerprint, usage, cache
            review.json             validated review + effective verdict
            review_raw.txt          raw reviewer reply text

Everything under runs/ and cache/ is gitignored (root .gitignore and a
self-ignoring .gitignore inside each directory).
"""

from __future__ import annotations

from pathlib import Path

from .budget import BudgetState
from .cache import ensure_ignored_dir
from .errors import ConfigError
from .util import now_iso, read_json, write_json_atomic


class RunPaths:
    def __init__(self, repo_root: Path, task_id: str):
        self.repo_root = Path(repo_root)
        self.base = self.repo_root / "tools" / "ai_collab"
        self.runs_dir = self.base / "runs"
        self.cache_dir = self.base / "cache"
        self.run_dir = self.runs_dir / task_id
        self.run_json = self.run_dir / "run.json"
        self.task_md = self.run_dir / "task.md"
        self.audit_jsonl = self.run_dir / "audit.jsonl"
        self.final_json = self.run_dir / "final.json"

    def init_run_dir(self) -> None:
        ensure_ignored_dir(self.runs_dir)
        self.run_dir.mkdir(parents=True, exist_ok=False)

    def round_dir(self, round_no: int) -> Path:
        path = self.run_dir / f"round_{round_no:02d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    # per-round files -------------------------------------------------------
    def claude_prompt(self, round_no: int, attempt: int) -> Path:
        return self.round_dir(round_no) / f"claude_prompt_a{attempt}.md"

    def claude_output(self, round_no: int, attempt: int) -> Path:
        return self.round_dir(round_no) / f"claude_output_a{attempt}.txt"

    def handoff(self, round_no: int) -> Path:
        return self.round_dir(round_no) / "claude_handoff.json"

    def manual_instructions(self, round_no: int) -> Path:
        return self.round_dir(round_no) / "MANUAL_INSTRUCTIONS.md"

    def diff_patch(self, round_no: int) -> Path:
        return self.round_dir(round_no) / "diff.patch"

    def changed_files(self, round_no: int) -> Path:
        return self.round_dir(round_no) / "changed_files.json"

    def tests_txt(self, round_no: int) -> Path:
        return self.round_dir(round_no) / "tests.txt"

    def graphify_txt(self, round_no: int) -> Path:
        return self.round_dir(round_no) / "graphify.txt"

    def reviewer_request(self, round_no: int) -> Path:
        return self.round_dir(round_no) / "reviewer_request.txt"

    def reviewer_request_meta(self, round_no: int) -> Path:
        return self.round_dir(round_no) / "reviewer_request_meta.json"

    def review_json(self, round_no: int) -> Path:
        return self.round_dir(round_no) / "review.json"

    def review_raw(self, round_no: int) -> Path:
        return self.round_dir(round_no) / "review_raw.txt"


class RunRecord:
    """Thin wrapper over the run.json dict with atomic persistence."""

    def __init__(self, data: dict, path: Path):
        self.data = data
        self.path = path

    # -- lifecycle ----------------------------------------------------------
    @classmethod
    def new(
        cls,
        path: Path,
        task_id: str,
        mode: str,
        max_rounds: int,
        baseline: dict,
        config_snapshot: dict,
    ) -> "RunRecord":
        data = {
            "task_id": task_id,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "mode": mode,
            "max_rounds": max_rounds,
            "state": "CREATED",
            "stop_reason": None,
            "current_round": 0,
            "last_verdict": None,
            "baseline": baseline,
            "rounds": [],
            "budget": BudgetState().to_dict(),
            "counters": {"claude_attempts_completed": 0, "reviewer_adapter_calls": 0},
            "approvals": [],
            "notes": [],
            "config_snapshot": config_snapshot,
        }
        return cls(data, path)

    @classmethod
    def load(cls, path: Path) -> "RunRecord":
        if not path.is_file():
            raise ConfigError(f"no run found at {path}")
        return cls(read_json(path), path)

    def save(self) -> None:
        self.data["updated_at"] = now_iso()
        write_json_atomic(self.path, self.data)

    # -- accessors ----------------------------------------------------------
    @property
    def task_id(self) -> str:
        return self.data["task_id"]

    @property
    def state(self) -> str:
        return self.data["state"]

    @state.setter
    def state(self, value: str) -> None:
        self.data["state"] = value

    @property
    def mode(self) -> str:
        return self.data["mode"]

    @property
    def max_rounds(self) -> int:
        return int(self.data["max_rounds"])

    @property
    def current_round(self) -> int:
        return int(self.data["current_round"])

    @property
    def budget_state(self) -> BudgetState:
        return BudgetState.from_dict(self.data.get("budget"))

    def set_budget_state(self, state: BudgetState) -> None:
        self.data["budget"] = state.to_dict()

    def counter(self, name: str) -> int:
        return int(self.data.setdefault("counters", {}).get(name, 0))

    def bump_counter(self, name: str) -> None:
        counters = self.data.setdefault("counters", {})
        counters[name] = int(counters.get(name, 0)) + 1

    def add_note(self, note: str) -> None:
        self.data.setdefault("notes", []).append({"ts": now_iso(), "note": note})

    def add_approval(self, gate: str, note: str = "") -> None:
        self.data.setdefault("approvals", []).append(
            {"ts": now_iso(), "gate": gate, "note": note}
        )

    def round_rec(self, round_no: int) -> dict:
        for rec in self.data["rounds"]:
            if rec.get("round") == round_no:
                return rec
        rec = {
            "round": round_no,
            "claude_attempts": [],
            "tests": {"fix_attempts_used": 0},
            "review": {},
        }
        self.data["rounds"].append(rec)
        return rec

    def current_attempt(self) -> dict | None:
        if self.current_round < 1:
            return None
        attempts = self.round_rec(self.current_round)["claude_attempts"]
        return attempts[-1] if attempts else None
