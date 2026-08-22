"""Run reports and cross-candidate comparison (read-only over run dirs).

REAL and SYNTHETIC verifier components are always presented separately; the
COMBINED block is labelled secondary. No winner is chosen: the comparison is
a table, the decision is the owner's (and stays UNSELECTED until models.toml
is edited by hand).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .manifests import REPO_ROOT
from .runner import DEFAULT_RUNS_ROOT


def load_run(run_dir: Path) -> dict[str, Any]:
    d = Path(run_dir)
    out: dict[str, Any] = {"run_dir": str(d)}
    for name in ("run.json", "metrics.json", "usage.json", "plan.json"):
        p = d / name
        if p.exists():
            out[name[:-5]] = json.loads(p.read_text(encoding="utf-8"))
    return out


def list_runs(role: str, runs_root: Path = DEFAULT_RUNS_ROOT, split: str | None = None) -> list[dict]:
    root = Path(runs_root) / role
    if not root.is_dir():
        return []
    rows = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        r = load_run(d)
        run = r.get("run", {})
        cfg = run.get("config", {})
        if split and cfg.get("split") != split.upper():
            continue
        rows.append({"run_id": d.name, "split": cfg.get("split"), "component": cfg.get("component"),
                     "candidate": cfg.get("candidate"), "mode": run.get("last_mode"),
                     "cases_done": run.get("cases_done"), "cases_failed": run.get("cases_failed"),
                     "stopped_reason": run.get("stopped_reason"), "has_metrics": "metrics" in r,
                     "metrics": r.get("metrics"), "usage": r.get("usage"), "plan": r.get("plan")})
    return rows


def _headline(role: str, m: dict | None) -> dict[str, Any]:
    """The per-role headline numbers for a comparison table."""
    if not m:
        return {}
    if role == "ocr_verify":
        real, syn = m.get("REAL") or {}, m.get("SYNTHETIC") or {}
        return {"REAL_FAR_pct": real.get("false_accept_rate_pct"), "REAL_FRR_pct": real.get("false_reject_rate_pct"),
                "REAL_precision_pct": real.get("supported_precision_pct"), "REAL_review_pct": real.get("review_rate_pct"),
                "SYN_FAR_pct": syn.get("false_accept_rate_pct"),
                "SYN_numeric_FAR_pct": (syn.get("numeric_math") or {}).get("false_accept_rate_pct"),
                "schema_failures": (real.get("schema_failures") or 0) + (syn.get("schema_failures") or 0),
                "cost": (m.get("usage") or {}).get("reported_cost"),
                "latency_median_s": (m.get("usage") or {}).get("latency_median_s")}
    if role == "ocr_primary":
        o = m.get("overall") or {}
        return {"mean_cer": o.get("mean_cer"), "median_cer": o.get("median_cer"),
                "usable_le_0.25_pct": o.get("usable_le_0.25_pct"), "usable_le_0.50_pct": o.get("usable_le_0.50_pct"),
                "number_sign_errors": o.get("number_sign_formula_errors"), "schema_failures": m.get("schema_failures"),
                "cost": (m.get("usage") or {}).get("reported_cost"),
                "latency_median_s": (m.get("usage") or {}).get("latency_median_s")}
    if role in ("grade_primary", "grade_escalate"):
        return {"exact_score_pct": m.get("exact_score_pct"), "mean_abs_error": m.get("mean_abs_score_error"),
                "harmful_up": m.get("harmful_upgrades"), "harmful_down": m.get("harmful_downgrades"),
                "auto_pct": m.get("auto_rate_pct"), "review_pct": m.get("review_rate_pct"),
                "evidence_failures": m.get("evidence_validation_failures"), "schema_failures": m.get("schema_failures"),
                "cost": (m.get("usage") or {}).get("reported_cost")}
    return {"exact_correct_pct": m.get("exact_correct_pct"), "unsafe_automatic": m.get("unsafe_automatic"),
            "abstention_pct": m.get("abstention_pct"), "schema_failures": m.get("schema_failures"),
            "cost": (m.get("usage") or {}).get("reported_cost")}


def compare(role: str, split: str, runs_root: Path = DEFAULT_RUNS_ROOT, component: str | None = None) -> dict:
    rows = []
    for r in list_runs(role, runs_root, split):
        if component and r.get("component") not in (component, None):
            continue
        rows.append({"candidate": r["candidate"], "run_id": r["run_id"], "component": r.get("component"),
                     "mode": r["mode"], "cases_done": r["cases_done"], **_headline(role, r.get("metrics"))})
    return {"role": role, "split": split.upper(), "component": component, "runs": rows,
            "note": ("comparison only — no winner is selected here; REAL and SYNTHETIC verifier "
                     "components are separate columns, COMBINED is secondary")}


def format_table(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return "(no runs)"
    cols = columns or sorted({k for r in rows for k in r}, key=lambda k: (k != "candidate", k))
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    line = " | ".join(c.ljust(widths[c]) for c in cols)
    sep = "-+-".join("-" * widths[c] for c in cols)
    body = "\n".join(" | ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols) for r in rows)
    return f"{line}\n{sep}\n{body}"


def historical_ocr_metrics() -> dict:
    """OCR_PRIMARY historical outputs (hebrew_bench_v2/outputs/<config>/run1)
    re-scored against the AUDITED references — read-only, no model calls
    (scripts/refaudit.preview_metrics)."""
    from .manifests import DEFAULT_BENCH_ROOT, _load_refaudit
    ra = _load_refaudit()
    store = ra.AuditStore(DEFAULT_BENCH_ROOT)
    return ra.preview_metrics(store)


__all__ = ["load_run", "list_runs", "compare", "format_table", "historical_ocr_metrics"]
