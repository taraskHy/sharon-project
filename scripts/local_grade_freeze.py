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

FREEZE_PATH = REPO / "evaluation" / "model_selection" / "experiments" / \
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
    from autograder.benchmark.roles import GradeAdapter
    from autograder.escalation import GradeResult, grade_system_for

    manifest = load_manifest("grade_primary")
    ds_dir = REPO / "evaluation" / "model_selection" / "datasets" / "grade_primary"

    prompt_version = "grade-v4-charitable"
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

    doc = {
        "experiment": "local_grade_primary",
        "created_at": "2026-08-27",
        "purpose": ("select the PRODUCTION local grading model on the frozen verdict "
                    "benchmark; cloud-grader runs (grade-v3/v4, Sonnet/Gemini) are "
                    "research baselines and do not select this model"),
        "git_commit": _git_head(),
        "target": "canonical explanation verdict via the production conversion "
                  "(reliability._verdict_from_score; imported by benchmark/verdicts.py)",
        "prompt_version": prompt_version,
        "prompt_sha256": hashlib.sha256(system.encode("utf-8")).hexdigest(),
        "schema_name": "GradeResult",
        "schema_sha256": hashlib.sha256(schema.encode("utf-8")).hexdigest(),
        "adapter_version": GradeAdapter.adapter_version,
        "max_tokens": GradeAdapter.default_max_tokens,
        "rag_policy": "RAG_DISABLED",
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
                "selection_sha256": smoke["selection_sha256"],
                "note": "frozen pre-registered DEV smoke; first live execution of any candidate",
            },
            "dev_verdict": _population(dev, "DEV"),
            "calibration_verdict_v4": _population(cal, "CALIBRATION"),
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
                "max_tokens", "rag_policy"):
        if frozen.get(key) != live.get(key):
            problems.append(f"{key}: frozen {frozen.get(key)!r} != live {live.get(key)!r}")
    for key in ("inputs_sha256", "labels_sha256", "final_labels_sha256", "manifest_sha256"):
        if frozen["dataset"].get(key) != live["dataset"].get(key):
            problems.append(f"dataset.{key} changed since the freeze")
    for pop in ("smoke", "dev_verdict", "calibration_verdict_v4"):
        f, l = frozen["populations"][pop], live["populations"][pop]
        if f["case_ids"] != l["case_ids"]:
            problems.append(f"populations.{pop}.case_ids changed")
        if f["selection_sha256"] != l["selection_sha256"]:
            problems.append(f"populations.{pop}.selection_sha256 changed")
    if frozen["human_audit"]["decisions"] != live["human_audit"]["decisions"]:
        problems.append("human_audit.decisions differ from the saved audit artifact")
    for cid in frozen["populations"]["dev_verdict"]["case_ids"] + \
            frozen["populations"]["calibration_verdict_v4"]["case_ids"]:
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
