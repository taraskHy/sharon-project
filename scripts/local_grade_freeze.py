"""Freeze (and later verify) the LOCAL grade_primary experiment.

The production grader is selected by running the SAME frozen verdict
benchmark the cloud baselines used — same dataset, same prompt
(grade-v4-charitable), same GradeResult schema, same production verdict
conversion — against LOCAL candidates only. This module writes ONE portable
freeze record binding every hash the strong-PC run must match, and verifies
it back before any execution:

    python scripts/local_grade_freeze.py            # write/refresh the freeze
    python scripts/local_grade_freeze.py --verify   # exit 0 iff everything matches

ZERO inference, zero network: everything is derived from files already in the
repository. HELD_OUT is never read. Paths inside the record are repo-relative
(portable); no machine-specific absolute path may enter it (test-pinned).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

#: The ACTIVE experiment record — the output-contract phase (2026-08-28):
#: structural DEV smoke + CALIBRATION quality under grade-v4-charitable-local.
FREEZE_PATH = REPO / "evaluation" / "model_selection" / "experiments" / \
    "LOCAL_GRADE_CONTRACT_FREEZE_2026-08-28.json"
#: The COMPLETED FullDev phase (grade-v4-charitable, grade-bench-v2). Its
#: record is immutable history; verify only its self-consistency, never
#: "refresh" it against live code.
HISTORICAL_FREEZE_PATH = REPO / "evaluation" / "model_selection" / "experiments" / \
    "LOCAL_GRADE_PRIMARY_FREEZE_2026-08-27.json"
AUDIT_PATH = REPO / "evaluation" / "model_selection" / "runs" / "grade_primary" / \
    "CALIBRATION_AUDIT_2026-08-26.json"
SUBSETS = REPO / "evaluation" / "model_selection" / "subsets"
SMOKE = REPO / "evaluation" / "model_selection" / "smoke" / "grade_primary_smoke.json"
RUNS_ROOT_REL = "evaluation/model_selection/runs/local_grade_primary"

#: Writers per split (mirrors the frozen dataset's WRITER_SPLIT_A). HELD_OUT
#: writers are listed ONLY so the freeze can prove none of them appears.
HELD_OUT_WRITERS = ("e005", "e006")


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO),
                                   text=True).strip()


def build_freeze() -> dict:
    """Assemble the freeze record from the repository's frozen artifacts."""
    from autograder.benchmark.manifests import load_manifest
    from autograder.benchmark.registry import load_registry
    from autograder.benchmark.roles import GradeAdapter
    from autograder.escalation import GRADE_VALIDATION_VERSION, GradeResult, grade_system_for

    manifest = load_manifest("grade_primary")
    ds_dir = REPO / "evaluation" / "model_selection" / "datasets" / "grade_primary"

    prompt_version = "grade-v4-charitable-local"
    system = grade_system_for(prompt_version)
    schema = json.dumps(GradeResult.model_json_schema(), sort_keys=True)

    dev = _load(SUBSETS / "grade_primary__dev_verdict.json")
    cal = _load(SUBSETS / "grade_primary__calibration_verdict_v4.json")
    smoke = _load(SMOKE)

    def _population(sub: dict, expect_split: str) -> dict:
        cases = sub["cases"]
        for c in cases:
            assert c["split"] == expect_split, (c["case_id"], c["split"])
            assert c["writer"] not in HELD_OUT_WRITERS, c["case_id"]
        dist: dict[str, int] = {}
        for c in cases:
            v = c.get("verdict")
            if v:
                dist[v] = dist.get(v, 0) + 1
        return {
            "split": expect_split,
            "case_ids": [c["case_id"] for c in cases],           # fixed order
            "verdicts": {c["case_id"]: c.get("verdict") for c in cases},
            "class_distribution": dist,
            "selection_sha256": sub["selection_sha256"],
        }

    # The completed human audit: any case decided C has an evidence/
    # transcription problem serious enough that the previous benchmark
    # comparison is unreliable for it. It is NOT silently a clean hard
    # target; the recompute policy excludes it from strict metrics until the
    # artifact is repaired.
    audit = _load(AUDIT_PATH)
    audit_decisions = {c["case_id"]: c.get("human_decision") for c in audit["cases"]}
    evidence_review = sorted(cid for cid, d in audit_decisions.items() if d == "C")

    # instructor-derived verdicts for the two smoke cases (structural smoke
    # still records its targets, it just never makes a quality claim)
    by_id = {c.case_id: c for c in manifest.cases}
    smoke_verdicts = {c["case_id"]: by_id[c["case_id"]].label.get("explanation_verdict")
                      for c in smoke["cases"]}

    registry = load_registry()
    local_candidates = list(registry.for_role("grade_primary_local").candidates)

    cal_ids = [c["case_id"] for c in cal["cases"]]
    audit_probe = _load(AUDIT_PATH)
    c_decided = sorted(c["case_id"] for c in audit_probe["cases"]
                       if c.get("human_decision") == "C")
    strict_ids = [cid for cid in cal_ids if cid not in c_decided]

    doc = {
        "experiment": "local_grade_primary_output_contract",
        "created_at": "2026-08-28",
        "purpose": ("verify the LOCAL grader output contract structurally on the frozen "
                    "two-case DEV smoke, then evaluate grading quality on the frozen "
                    "CALIBRATION population. DEV was consumed for development by the "
                    "completed FullDev phase and never again supports a quality claim; "
                    "cloud-grader runs are research baselines and select nothing"),
        "git_commit": _git_head(),
        "target": "canonical explanation verdict via the production conversion "
                  "(reliability._verdict_from_score; imported by benchmark/verdicts.py)",
        "ground_truth": ("actual instructor-assigned grades (final_labels.json, "
                         "ground_truth_source=original_instructor_grade) + actual selection "
                         "correctness + frozen production scoring policy. A/B/C/D human-audit "
                         "decisions are diagnostic flags only and NEVER targets; no previous "
                         "model output (Qwen/Gemini/Sonnet/Luna), model vote, or subjective "
                         "ranking defines a target; no target reaches a model request"),
        "prompt_version": prompt_version,
        "prompt_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
        "schema_name": "GradeResult",
        "schema_sha256": hashlib.sha256(schema.encode("utf-8")).hexdigest(),
        "adapter_version": GradeAdapter.adapter_version,
        "validation_version": GRADE_VALIDATION_VERSION,
        "max_tokens": GradeAdapter.default_max_tokens,
        "rag_policy": "RAG_DISABLED",
        "candidates": local_candidates,
        "candidates_note": ("qwen3.8:27b-q4_K_M was DROPPED by owner decision on 2026-08-28 "
                            "after the FullDev report and may not run again; "
                            "qwen3-vl:8b-instruct is a development candidate, not a winner; "
                            "qwen3-vl:30b-a3b-instruct runs only if the structural smoke and "
                            "the machine preflight pass"),
        "route_requirement": {
            "backend": "local only (ollama / local OpenAI-compatible); remote URLs "
                       "refused by cloudboundary.is_remote_route and the bench "
                       "--research gate",
            "cloud_grading_calls": 0,
            "cloud_grading_cost_usd": 0.0,
        },
        "dataset": {
            "name": "grade_primary",
            "status": manifest.status,
            "manifest": "evaluation/model_selection/datasets/grade_primary/manifest.json",
            "inputs_sha256": manifest.hashes["inputs_sha256"],
            "labels_sha256": manifest.hashes["labels_sha256"],
            "final_labels_sha256": _sha256_file(ds_dir / "final_labels.json"),
            "manifest_sha256": _sha256_file(ds_dir / "manifest.json"),
        },
        "populations": {
            "smoke": {
                "split": "DEV",
                "case_ids": [c["case_id"] for c in smoke["cases"]],
                "verdicts": smoke_verdicts,
                "selection_sha256": smoke["selection_sha256"],
                "note": ("frozen pre-registered DEV smoke — STRUCTURAL check only "
                         "(schema, rubric_items population, exact evidence placement, "
                         "invalid/zero routing, latency); NO quality claim is ever made "
                         "from these two cases"),
            },
            "dev_verdict": {**_population(dev, "DEV"),
                            "note": ("COMPLETED population of the FullDev phase "
                                     "(2026-08-27 freeze); retained for the record — this "
                                     "experiment never reruns it and never claims quality "
                                     "from DEV")},
            "calibration_verdict_v4": {**_population(cal, "CALIBRATION"),
                                       "note": "the QUALITY population of this experiment"},
        },
        "strict_metrics": {
            "calibration_strict_case_ids": strict_ids,
            "excluded": {cid: ("committed human-audit decision C: evidence/transcription "
                               "issue — diagnostic row only; the actual instructor grade "
                               "and the raw output are preserved, the target is NOT "
                               "replaced by the C decision") for cid in c_decided},
            "rule": ("strict model-quality denominators use calibration_strict_case_ids "
                     "only; excluded cases still run for diagnostic completeness"),
        },
        "held_out": {
            "writers": list(HELD_OUT_WRITERS),
            "policy": "NEVER read or executed during selection; one final confirmation "
                      "run later via `bench final-eval --confirm-held-out`",
        },
        "limitations": {
            "invalid_class": "NOT MEASURED — zero derivable invalid cases exist in "
                             "DEV or CALIBRATION; no production-readiness claim for "
                             "invalid explanations without new authoritative data",
        },
        "human_audit": {
            "artifact": "evaluation/model_selection/runs/grade_primary/CALIBRATION_AUDIT_2026-08-26.json",
            "decisions": audit_decisions,
            "evidence_review_required": evidence_review,
            "strict_metrics_policy": ("cases decided C are excluded from strict-accuracy "
                                      "denominators until their evidence/transcription is "
                                      "repaired (scripts/calibration_audit_recompute.py); "
                                      "they still run, so the raw outputs exist either way"),
        },
        "completed_history": {
            "fulldev_freeze": "evaluation/model_selection/experiments/LOCAL_GRADE_PRIMARY_FREEZE_2026-08-27.json",
            "fulldev_results": "evaluation/model_selection/runs/local_grade_primary/FULLDEV_2026-08-28.md",
            "fulldev_audit": "evaluation/model_selection/runs/local_grade_primary/FULLDEV_AUDIT_2026-08-28.json",
            "note": ("the FullDev phase ran grade-v4-charitable / grade-bench-v2 / "
                     "grade-validation-v1 semantics; its artifacts are immutable and are "
                     "never re-scored under this experiment's rules"),
        },
        "runs_root": RUNS_ROOT_REL,
        "candidate_registry": "evaluation/model_selection/candidates.toml [roles.grade_primary_local]",
    }
    payload = json.dumps({k: v for k, v in doc.items() if k != "experiment_sha256"},
                         ensure_ascii=False, sort_keys=True)
    doc["experiment_sha256"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return doc


def verify_freeze(freeze_path: Path = FREEZE_PATH) -> list[str]:
    """Return the list of mismatches between the freeze record and the live
    repository (empty = everything matches). ZERO inference, zero network."""
    problems: list[str] = []
    if not freeze_path.exists():
        return [f"freeze record missing: {freeze_path}"]
    frozen = _load(freeze_path)
    live = build_freeze()
    for key in ("prompt_version", "prompt_sha256", "schema_sha256", "adapter_version",
                "validation_version", "max_tokens", "rag_policy", "candidates"):
        if frozen.get(key) != live.get(key):
            problems.append(f"{key}: frozen {frozen.get(key)!r} != live {live.get(key)!r}")
    if frozen.get("strict_metrics") != live.get("strict_metrics"):
        problems.append("strict_metrics changed (calibration strict ids / audit-C exclusions)")
    if any(frozen["dataset"].get(k) != live["dataset"].get(k)
           for k in ("inputs_sha256", "labels_sha256", "final_labels_sha256", "manifest_sha256")):
        # A post-freeze dataset move is acceptable ONLY when the manifest's
        # revision chain explains the walk from the frozen hashes to the live
        # ones with recorded, owner-confirmed repairs (e.g. the 2026-09-01
        # confirmed_row_transposition). Anything unexplained is drift.
        ds = REPO / "evaluation" / "model_selection" / "datasets" / "grade_primary"
        man = json.loads((ds / "manifest.json").read_text(encoding="utf-8"))
        cur_i = frozen["dataset"].get("inputs_sha256")
        cur_l = frozen["dataset"].get("labels_sha256")
        explained = []
        for rev in man.get("revisions", []):
            if (rev.get("previous_inputs_sha256") == cur_i
                    and rev.get("previous_labels_sha256") == cur_l
                    and rev.get("owner_confirmed") is True):
                cur_i, cur_l = rev["inputs_sha256"], rev["labels_sha256"]
                explained.append(rev["kind"])
        if (cur_i, cur_l) != (live["dataset"].get("inputs_sha256"),
                              live["dataset"].get("labels_sha256")):
            problems.append("dataset hashes changed since the freeze with NO owner-confirmed "
                            "revision chain explaining the walk")
        # final_labels must never move at all (instructor grades are immutable)
        if frozen["dataset"].get("final_labels_sha256") != live["dataset"].get("final_labels_sha256"):
            problems.append("final_labels_sha256 changed — instructor grades must never move")
    for pop in ("smoke", "dev_verdict", "calibration_verdict_v4"):
        f, l = frozen["populations"][pop], live["populations"][pop]
        if f["case_ids"] != l["case_ids"]:
            problems.append(f"populations.{pop}.case_ids changed")
        if f["selection_sha256"] != l["selection_sha256"]:
            problems.append(f"populations.{pop}.selection_sha256 changed")
    if frozen["human_audit"]["decisions"] != live["human_audit"]["decisions"]:
        problems.append("human_audit.decisions differ from the saved audit artifact")
    for pop in frozen["populations"].values():
        for cid in pop["case_ids"]:
            if cid.split("_")[0] in HELD_OUT_WRITERS:
                problems.append(f"HELD_OUT writer leaked into a frozen population: {cid}")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true",
                    help="verify the existing freeze against the live repo (no write)")
    args = ap.parse_args(argv)
    if args.verify:
        problems = verify_freeze()
        if problems:
            print("FREEZE MISMATCH:")
            for p in problems:
                print(" -", p)
            return 2
        print(f"freeze verified: {FREEZE_PATH.name} matches the live repository")
        return 0
    doc = build_freeze()
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FREEZE_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                           encoding="utf-8")
    print(f"wrote {FREEZE_PATH}")
    print(f"experiment_sha256 {doc['experiment_sha256']}")
    print(f"DEV {len(doc['populations']['dev_verdict']['case_ids'])} · "
          f"CALIBRATION {len(doc['populations']['calibration_verdict_v4']['case_ids'])} · "
          f"smoke {len(doc['populations']['smoke']['case_ids'])} · "
          f"evidence-review {doc['human_audit']['evidence_review_required']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
