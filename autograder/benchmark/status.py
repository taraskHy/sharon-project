"""Per-role dataset readiness (Part 4F vocabulary):

    READY                    can be run AND scored now (inputs + labels complete)
    PARTIALLY_READY          runnable; some cases lack labels -> partial scoring
    NEEDS_OWNER_LABELS       inputs built; evaluation labels must come from the owner
    PENDING_OTHER_EXPERIMENT dataset is derived from another role's results (not run yet)
    NOT_AVAILABLE            cannot be built from existing audited/local artifacts (why)

Nothing is called READY unless `bench dry-run` would select cases AND the
adapter can score every one of them against an evaluation-side label.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .manifests import (DEFAULT_BENCH_ROOT, DEFAULT_DATASETS_ROOT, ROLES, BenchmarkIntegrityError,
                        BenchmarkManifest, BenchmarkNotBuilt, load_manifest)

#: label key that makes a case scorable, per role
SCORABLE_LABEL = {
    "ocr_primary": "reference",
    "ocr_verify": "expected_verdict",
    "grade_primary": "score",
    "grade_escalate": "score",
    "mc_resolve_cloud": "answer",
    "variant_resolve": "variants",
    "align_resolve": "mapping",
}

NOT_AVAILABLE_REASONS = {
    "grade_primary": "grading inputs not built (run `bench build-grading`)",
    "grade_escalate": "PENDING_PRIMARY_RESULTS: escalation cases are harvested from grade_primary runs",
    "mc_resolve_cloud": "no audited ambiguous-MC band crops exported yet (run `bench build-mc`)",
    "variant_resolve": "no audited cover/marker image set exported yet (run `bench build-variant`)",
    "align_resolve": "no operator-verified alignment cases exported yet (run `bench build-align`)",
}


def role_dataset_status(role: str, manifest: BenchmarkManifest | None = None, *,
                        bench_root: Path = DEFAULT_BENCH_ROOT,
                        datasets_root: Path = DEFAULT_DATASETS_ROOT) -> dict[str, Any]:
    out: dict[str, Any] = {"role": role, "status": None, "cases": 0, "labeled": 0, "detail": "",
                           "owner_action": None}
    try:
        m = manifest or load_manifest(role, bench_root=bench_root, datasets_root=datasets_root)
    except BenchmarkIntegrityError as e:
        out.update({"status": "NOT_AVAILABLE", "detail": f"INTEGRITY ERROR: {e}"})
        return out
    except BenchmarkNotBuilt as e:
        out.update({"status": "NOT_AVAILABLE", "detail": str(e)})
        return out
    if m.status == "NOT_BUILT" or not m.cases:
        if role == "grade_escalate":
            out.update({"status": "PENDING_OTHER_EXPERIMENT", "detail": NOT_AVAILABLE_REASONS[role]})
        else:
            out.update({"status": "NOT_AVAILABLE", "detail": NOT_AVAILABLE_REASONS.get(role, "not built")})
        return out
    key = SCORABLE_LABEL[role]
    labeled = sum(1 for c in m.cases if c.label.get(key) is not None)
    out.update({"cases": len(m.cases), "labeled": labeled, "counts": m.counts(), "hashes": m.hashes,
                "manifest_status": m.status})
    if role in ("grade_primary", "grade_escalate"):
        # human labels are owner work: count them explicitly
        pending = len(m.cases) - labeled
        if labeled == 0:
            out.update({"status": "NEEDS_OWNER_LABELS",
                        "detail": f"{len(m.cases)} grading cases built; 0 owner-labeled",
                        "owner_action": f"label {pending} grading case(s): shared tool `python -m labeling_app serve` "
                                        f"(friends via Cloudflare Tunnel; then `bench import-final-labels`) or the local "
                                        f"owner tool `python -m streamlit run scripts/grade_label_ui.py`"})
        elif pending:
            out.update({"status": "PARTIALLY_READY",
                        "detail": f"{labeled}/{len(m.cases)} owner-labeled; unlabeled cases are run but "
                                  "report decision metrics only",
                        "owner_action": f"label the remaining {pending} grading case(s)"})
        else:
            out.update({"status": "READY", "detail": f"{labeled}/{len(m.cases)} owner-labeled"})
        return out
    if labeled == len(m.cases):
        out.update({"status": "READY", "detail": f"{len(m.cases)} cases, all labeled"})
    elif labeled:
        out.update({"status": "PARTIALLY_READY", "detail": f"{labeled}/{len(m.cases)} labeled"})
    else:
        out.update({"status": "NOT_AVAILABLE", "detail": "cases exist but carry no evaluation labels"})
    return out


def all_role_statuses(*, bench_root: Path = DEFAULT_BENCH_ROOT,
                      datasets_root: Path = DEFAULT_DATASETS_ROOT) -> dict[str, dict]:
    return {r: role_dataset_status(r, bench_root=bench_root, datasets_root=datasets_root) for r in ROLES}


__all__ = ["role_dataset_status", "all_role_statuses", "SCORABLE_LABEL"]
