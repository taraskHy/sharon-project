"""Strong-PC preflight for the local grade_primary experiment — ZERO inference.

    python scripts/local_grade_preflight.py [--json]

Verifies, without a single model call, embedding, download, or provider
request:

  1. the freeze record matches the live repository (dataset hashes, frozen
     case lists, prompt/schema hashes, audit decisions);
  2. the production cloud boundary blocks cloud grading and allows OCR only;
  3. models.toml (when present) routes grading locally;
  4. no HELD_OUT writer appears in any frozen population;
  5. which local backends/models are INSTALLED (metadata listing only — the
     Ollama API/CLI list never loads a model).

Exit codes: 0 ready · 2 freeze mismatch (execution must refuse) · 3 boundary
problem. Model discovery being empty does NOT fail preflight — it reports
which candidates are missing so the operator can pull them deliberately
(this script never downloads anything).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.local_grade_freeze import FREEZE_PATH, RUNS_ROOT_REL, verify_freeze  # noqa: E402


def boundary_report() -> dict:
    """Prove in-process that the production boundary is armed. No network."""
    from autograder.cloudboundary import (CLOUD_OCR_ALLOWLIST, CloudBoundaryError,
                                          check_cloud_call)
    out = {"cloud_ocr_allowlist": sorted(CLOUD_OCR_ALLOWLIST)}
    try:
        check_cloud_call(task="grade_primary", backend="openrouter", base_url=None,
                         execution_mode="production")
        out["cloud_grading_blocked"] = False
    except CloudBoundaryError:
        out["cloud_grading_blocked"] = True
    try:
        check_cloud_call(task="grade_primary", backend="ollama",
                         base_url="http://localhost:11434/v1",
                         execution_mode="production")
        out["local_grading_allowed"] = True
    except CloudBoundaryError:
        out["local_grading_allowed"] = False
    return out


def models_toml_report() -> dict:
    """Grading routes in the machine's models.toml must be local. TOML parse
    only; no gateway, no backend construction."""
    from autograder.readiness import role_status
    cfg = REPO / "models.toml"
    rs = role_status(cfg if cfg.exists() else None)
    tasks = rs.get("tasks") or {}
    grade = {t: tasks.get(t) or {} for t in ("grade_primary", "grade_escalate")}
    return {
        "config": rs.get("config"),
        "using_example": bool(rs.get("using_example_as_template")),
        "grade_routes": {t: {"backend": v.get("backend"), "status": v.get("status"),
                             "cloud": v.get("cloud")} for t, v in grade.items()},
        "grade_routes_local": all(not v.get("cloud") for v in grade.values()),
    }


def installed_local_models() -> dict:
    """List installed Ollama models — METADATA ONLY (`ollama list` prints the
    manifest table; no model is loaded, nothing is downloaded)."""
    try:
        raw = subprocess.run(["ollama", "list"], capture_output=True, text=True,
                             timeout=20)
    except Exception as e:  # noqa: BLE001 — absence of Ollama is a report, not a crash
        return {"available": False, "error": f"{type(e).__name__}: {e}", "models": []}
    if raw.returncode != 0:
        return {"available": False, "error": (raw.stderr or raw.stdout).strip()[:200],
                "models": []}
    models = []
    for line in raw.stdout.splitlines()[1:]:
        parts = line.split()
        if parts:
            models.append(parts[0])
    return {"available": True, "models": models}


def candidate_report(installed: list[str]) -> dict:
    """Frozen local candidates vs what this machine actually has. Discovery is
    dynamic — the registry never hardcodes machine paths."""
    import tomllib
    reg = tomllib.loads((REPO / "evaluation" / "model_selection" /
                         "candidates.toml").read_text(encoding="utf-8"))
    sec = (reg.get("roles") or {}).get("grade_primary_local") or {}
    cands = list(sec.get("candidates") or [])
    have = set(installed)
    return {"status": sec.get("status"), "candidates": cands,
            "installed": [c for c in cands if c in have],
            "missing": [c for c in cands if c not in have]}


def run(json_out: bool = False) -> int:
    freeze_problems = verify_freeze()
    boundary = boundary_report()
    cfg = models_toml_report()
    local = installed_local_models()
    cands = candidate_report(local.get("models") or [])
    frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8")) if FREEZE_PATH.exists() else {}
    report = {
        "freeze": {"path": str(FREEZE_PATH.relative_to(REPO)).replace("\\", "/"),
                   "ok": not freeze_problems, "problems": freeze_problems,
                   "experiment_sha256": frozen.get("experiment_sha256"),
                   "git_commit_at_freeze": frozen.get("git_commit")},
        "populations": {k: len(v.get("case_ids") or [])
                        for k, v in (frozen.get("populations") or {}).items()},
        "evidence_review_required": (frozen.get("human_audit") or {}).get(
            "evidence_review_required"),
        "held_out": "not selectable (smoke/dev populations only; final-eval is a "
                    "separate confirmed path)",
        "boundary": boundary,
        "models_toml": cfg,
        "local_models": local,
        "candidates": cands,
        "runs_root": RUNS_ROOT_REL,
        "inference_calls_made": 0,
        "provider_calls_made": 0,
    }
    if json_out:
        print(json.dumps(report, ensure_ascii=False, indent=1))
    else:
        print(f"freeze: {'OK' if report['freeze']['ok'] else 'MISMATCH'}"
              + (f" ({len(freeze_problems)} problems)" if freeze_problems else ""))
        for p in freeze_problems:
            print("  -", p)
        print(f"boundary: cloud grading blocked = {boundary['cloud_grading_blocked']}, "
              f"local grading allowed = {boundary['local_grading_allowed']}, "
              f"cloud allowlist = {boundary['cloud_ocr_allowlist']}")
        print(f"models.toml grade routes local = {cfg['grade_routes_local']} ({cfg['config']})")
        print(f"local models: {local.get('models') or local.get('error')}")
        print(f"candidates installed = {cands['installed']}, missing = {cands['missing']}")
        print("inference calls made = 0")
    if freeze_problems:
        return 2
    if not (boundary["cloud_grading_blocked"] and boundary["local_grading_allowed"]):
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    return run(json_out=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
