"""Zero-key product readiness check (dry run) — `autograder readiness`.

Answers "is the product ready to run, and what exactly is still missing?"
WITHOUT a single model or network call: every section is computed from
files, configuration and the environment's *presence* of a credential.

Sections (Part 12 of the pre-API setup):
    course_store, exam_packages, benchmarks, model_roles, local_models,
    rag_index, openrouter, budget, verifier_crops, gui, held_out, blockers

Cloud-dependent operations elsewhere fail with the same one-sentence
explanations this report shows (cloudcheck.py): "grade_primary model is not
selected", "OpenRouter credential is not configured".
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CLOUD_TASKS = ("ocr_primary", "ocr_verify", "grade_primary", "grade_escalate", "mc_resolve_cloud",
               "variant_resolve_cloud", "align_resolve_cloud", "policy_infer_cloud")
_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _safe(fn, default):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 — readiness reports problems, it does not crash on them
        return {**default, "error": f"{type(e).__name__}: {e}"} if isinstance(default, dict) else default


# ----------------------------------------------------------------------------
# model roles (parsed from models.toml WITHOUT constructing backends)
# ----------------------------------------------------------------------------

def role_status(models_config: Path | None) -> dict[str, Any]:
    """Per gateway task: backend, how the model is expressed, and a status:
    SELECTED_LOCAL | CONFIGURED_CLOUD | UNSELECTED | DISABLED | ABSENT.
    The ${ENV} slug is only checked for presence; its value is not echoed
    unless it is a plain model slug (never a credential)."""
    cfg_path = Path(models_config) if models_config else REPO_ROOT / "models.toml"
    using_example = False
    if not cfg_path.exists():
        example = REPO_ROOT / "models.example.toml"
        if not example.exists():
            return {"config": str(cfg_path), "exists": False, "tasks": {}, "note": "no models.toml and no example"}
        cfg_path, using_example = example, True
    data = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
    models = data.get("models") or {}
    tasks: dict[str, dict] = {}
    for task, sec in models.items():
        backend = str(sec.get("backend", ""))
        raw = str(sec.get("model", "") or "")
        enabled = bool(sec.get("enabled", True))
        env_refs = _ENV_REF.findall(raw)
        env_set = all(bool(os.environ.get(e)) for e in env_refs) if env_refs else None
        cloud = backend in ("openrouter", "anthropic") or (backend == "openai" and "openrouter" in str(sec.get("base_url", "")))
        if not enabled:
            status = "DISABLED"
        elif raw == "UNSELECTED" or (env_refs and not env_set):
            status = "UNSELECTED"
        elif cloud:
            status = "CONFIGURED_CLOUD"
        else:
            status = "SELECTED_LOCAL"
        shown = raw if not env_refs else ("${" + env_refs[0] + "}" + (" (set)" if env_set else " (unset)"))
        tasks[task] = {"backend": backend, "model": shown, "status": status, "cloud": cloud,
                       "base_url": sec.get("base_url")}
    for task in CLOUD_TASKS:
        tasks.setdefault(task, {"backend": None, "model": None, "status": "ABSENT", "cloud": True})
    return {"config": str(cfg_path), "exists": True, "using_example_as_template": using_example,
            "tasks": tasks,
            "unselected": sorted(t for t, v in tasks.items() if v["status"] in ("UNSELECTED", "ABSENT")),
            "budget_section": data.get("budget") or {}, "pricing_table": bool(data.get("pricing"))}


# ----------------------------------------------------------------------------
# sections
# ----------------------------------------------------------------------------

def _course_store() -> dict:
    from . import courses
    out = {"root": str(courses.courses_root()), "courses": []}
    for c in courses.list_courses():
        cid = c.get("course_id") or c.get("id")
        st = _safe(lambda: courses.index_status(cid), {"course_id": cid})
        out["courses"].append({"course_id": cid, "name": c.get("name"), "sources": st.get("n_sources"),
                               "indexed": st.get("indexed"), "stale": st.get("stale"),
                               "chunks": st.get("n_chunks"), "embed_model": st.get("embed_model")})
    out["count"] = len(out["courses"])
    return out


def _exam_packages() -> dict:
    from .reviewui import package_dirs
    out = {"dirs": [], "packages": []}
    for d in package_dirs():
        out["dirs"].append(str(d))
        if not d.is_dir():
            continue
        for tpl in sorted(d.glob("*.template.json")):
            stem = tpl.name[: -len(".template.json")]
            key_json = d / f"{stem}.json"
            key_pdf = d / f"{stem}.pdf"
            entry = {"dir": str(d), "template": tpl.name, "key_json": key_json.exists(), "key_pdf": key_pdf.exists(),
                     "variants": (d / f"{stem}.variants.json").exists(),
                     "alignment": (d / f"{stem}.alignment.json").exists(), "preflight": None}
            if key_json.exists():
                def _pf():
                    from .key_parser import load_answer_key
                    from .preflight import alignment_from_discovery, preflight_package
                    k = load_answer_key(key_json)
                    al = None
                    ap = d / f"{stem}.alignment.json"
                    if ap.exists():
                        al = json.loads(ap.read_text(encoding="utf-8"))
                    rep = preflight_package(key=k, variants=list(k.versions),
                                            alignment=alignment_from_discovery(al, list(k.versions), k))
                    return {"status": rep.status, "summary": rep.summary(),
                            "blocking": len(getattr(rep, "blocking", []) or [])}
                entry["preflight"] = _safe(_pf, {"status": "UNKNOWN"})
            else:
                entry["preflight"] = {"status": "KEY_NOT_PARSED",
                                      "summary": "answer key not parsed yet (PDF only) — parse it in Exam setup"}
            out["packages"].append(entry)
    out["count"] = len(out["packages"])
    return out


def _benchmarks(bench_root: Path | None, datasets_root: Path | None) -> dict:
    from .benchmark.manifests import DEFAULT_BENCH_ROOT, DEFAULT_DATASETS_ROOT, all_manifest_summaries
    summ = all_manifest_summaries(bench_root=bench_root or DEFAULT_BENCH_ROOT,
                                  datasets_root=datasets_root or DEFAULT_DATASETS_ROOT)
    return {role: {"status": s.get("status"), "cases": s.get("cases"), "counts": s.get("counts"),
                   "hashes": s.get("hashes"), "error": s.get("error")} for role, s in summ.items()}


def _local_models(roles: dict) -> dict:
    tasks = {t: v for t, v in (roles.get("tasks") or {}).items() if v.get("status") == "SELECTED_LOCAL"}
    return {"configured_local_routes": tasks,
            "ollama_host_env": os.environ.get("OLLAMA_HOST", "(default http://localhost:11434)"),
            "probed": False,
            "note": "zero-network readiness: local model availability is NOT probed here; run "
                    "`autograder doctor` to probe Ollama when you want a live check"}


def _openrouter() -> dict:
    from .cloudcheck import openrouter_credential_present
    return {"configured": "YES" if openrouter_credential_present() else "NO",
            "credential_env": "OPENROUTER_API_KEY (presence only; value never read, stored or shown)",
            "key_metadata_endpoint": "GET /api/v1/key — supported (backends.openrouter.fetch_key_metadata), NOT called",
            "calls_made_by_this_check": 0}


def _budget(roles: dict, state_root: Path | None) -> dict:
    from .benchmark.registry import load_registry
    from .benchmark.runner import DEFAULT_STATE_ROOT
    from .spend import EXPERIMENT_HARD_STOP_USD, EXPERIMENT_WARN_USD, budget_status, ledger_summary
    reg = _safe(lambda: load_registry().summary(), {})
    root = Path(state_root) if state_root else DEFAULT_STATE_ROOT
    led = ledger_summary(root / "gateway_ledger" / "usage.jsonl")
    warn = (reg.get("budget") or {}).get("warn_usd") or EXPERIMENT_WARN_USD
    hard = (reg.get("budget") or {}).get("experiment_total_usd") or EXPERIMENT_HARD_STOP_USD
    return {"policy": {"warning_usd": warn, "hard_stop_usd": hard,
                       "source": "evaluation/model_selection/candidates.toml [budget]"},
            "models_toml_budget": roles.get("budget_section") or {},
            "campaign_ledger": {"path": led["path"], "exists": led["exists"], "cloud_calls": led["cloud_calls"],
                                "cumulative_cost": led["cumulative_cost"], "by_task": led["by_task"],
                                "by_model": led["by_model"]},
            "status": budget_status(led["cumulative_cost"], warn_usd=warn, hard_usd=hard)}


def _verifier_crops() -> dict:
    from .evidencecrops import production_crop_provider
    return production_crop_provider().describe()


def _gui() -> dict:
    return {"streamlit_installed": importlib.util.find_spec("streamlit") is not None,
            "app": str(REPO_ROOT / "autograder" / "webui.py"),
            "app_exists": (REPO_ROOT / "autograder" / "webui.py").exists(),
            "screens": ["Dashboard", "Exam setup", "Grading progress", "Review queue", "Results / export",
                        "Advanced / diagnostics"],
            "launch": "python -m autograder ui  (or: streamlit run autograder/webui.py)"}


def _held_out(bench_root: Path | None) -> dict:
    from .benchmark.manifests import DEFAULT_BENCH_ROOT, load_manifest
    from .benchmark.runner import HELD_OUT_LOG, held_out_executions
    ex = held_out_executions(HELD_OUT_LOG)
    split = _safe(lambda: load_manifest("ocr_verify", bench_root=bench_root or DEFAULT_BENCH_ROOT).split_assignment, {})
    return {"log": str(HELD_OUT_LOG), "executions": len(ex), "untouched": len(ex) == 0,
            "writer_split": split,
            "rule": "HELD_OUT needs --split held_out --confirm-held-out; every execution is logged permanently; "
                    "once inspected and used to change the system it must be demoted to DEV"}


def readiness_report(*, models_config: Path | None = None, bench_root: Path | None = None,
                     datasets_root: Path | None = None, state_root: Path | None = None) -> dict[str, Any]:
    roles = _safe(lambda: role_status(models_config), {"tasks": {}, "unselected": []})
    rep: dict[str, Any] = {
        "mode": "zero-key dry run (no model or network calls)",
        "course_store": _safe(_course_store, {"courses": [], "count": 0}),
        "exam_packages": _safe(_exam_packages, {"packages": [], "count": 0}),
        "benchmarks": _safe(lambda: _benchmarks(bench_root, datasets_root), {}),
        "model_roles": roles,
        "local_models": _safe(lambda: _local_models(roles), {}),
        "rag_index": None,
        "openrouter": _openrouter(),
        "budget": _safe(lambda: _budget(roles, state_root), {}),
        "verifier_crops": _safe(_verifier_crops, {}),
        "gui": _gui(),
        "held_out": _safe(lambda: _held_out(bench_root), {}),
        "network_calls": 0,
    }
    rep["rag_index"] = {"courses": rep["course_store"].get("courses", []),
                        "embed_model_default": "bge-m3 via local Ollama (never cloud)",
                        "indexed_courses": sum(1 for c in rep["course_store"].get("courses", []) if c.get("indexed"))}
    blockers: list[str] = []
    if rep["openrouter"]["configured"] == "NO":
        blockers.append("OpenRouter credential is not configured (OPENROUTER_API_KEY unset) — cloud roles cannot run")
    unsel = [t for t in roles.get("unselected", []) if t in CLOUD_TASKS]
    if unsel:
        blockers.append(f"{len(unsel)} cloud role(s) UNSELECTED: {', '.join(unsel)} — run benchmarks, then set models.toml")
    if roles.get("using_example_as_template"):
        blockers.append("models.toml missing — copy models.example.toml to models.toml to enable reliability mode")
    if (rep["verifier_crops"] or {}).get("status") == "UNAVAILABLE":
        blockers.append("production verifier crops UNAVAILABLE — ocr_verify runs fail-closed (REVIEW on suspicion)")
    for role, b in (rep["benchmarks"] or {}).items():
        if b.get("status") == "INTEGRITY_ERROR":
            blockers.append(f"benchmark {role}: INTEGRITY ERROR — {b.get('error')}")
    if not rep["gui"]["streamlit_installed"]:
        blockers.append("streamlit not installed — GUI cannot start")
    rep["blockers"] = blockers
    rep["ready_for_zero_key_dry_run"] = all(
        b.get("status") != "INTEGRITY_ERROR" for b in (rep["benchmarks"] or {}).values()) and rep["gui"]["streamlit_installed"]
    return rep


def format_readiness(rep: dict) -> str:
    lines = [f"READINESS — {rep['mode']}", ""]
    cs = rep["course_store"]; lines.append(f"course store      : {cs.get('count', 0)} course(s) at {cs.get('root')}")
    for c in cs.get("courses", []):
        lines.append(f"                    - {c['course_id']}: {c.get('sources')} source(s), indexed={c.get('indexed')}, stale={c.get('stale')}")
    ep = rep["exam_packages"]; lines.append(f"exam packages     : {ep.get('count', 0)} found in {ep.get('dirs')}")
    for p in ep.get("packages", []):
        pf = p.get("preflight") or {}
        lines.append(f"                    - {p['template']}: {pf.get('status')} — {pf.get('summary', '')}")
    lines.append("benchmarks        :")
    for role, b in (rep["benchmarks"] or {}).items():
        lines.append(f"                    - {role:<17} {b.get('status'):<20} cases={b.get('cases')} counts={b.get('counts')}")
    mr = rep["model_roles"]; lines.append(f"model roles       : {mr.get('config')}" + (" (EXAMPLE used as template)" if mr.get("using_example_as_template") else ""))
    for t, v in (mr.get("tasks") or {}).items():
        lines.append(f"                    - {t:<22} {v.get('status'):<16} {v.get('backend') or '-':<10} {v.get('model') or '-'}")
    lm = rep["local_models"]; lines.append(f"local models      : {len(lm.get('configured_local_routes') or {})} local route(s); probed={lm.get('probed')}")
    ri = rep["rag_index"]; lines.append(f"RAG index         : {ri.get('indexed_courses')} indexed course(s); embed {ri.get('embed_model_default')}")
    orr = rep["openrouter"]; lines.append(f"OpenRouter        : configured: {orr['configured']} — {orr['key_metadata_endpoint']}")
    bd = rep["budget"]; st = bd.get("status") or {}
    lines.append(f"budget            : warn ${bd.get('policy', {}).get('warning_usd')} / hard ${bd.get('policy', {}).get('hard_stop_usd')}; "
                 f"campaign ledger spent ${st.get('cumulative_cost')} ({st.get('state')})")
    vc = rep["verifier_crops"]; lines.append(f"verifier crops    : {vc.get('status')} — {vc.get('reason') or ''}")
    g = rep["gui"]; lines.append(f"GUI               : streamlit={'yes' if g['streamlit_installed'] else 'NO'}; screens: {', '.join(g['screens'])}")
    ho = rep["held_out"]; lines.append(f"held-out          : untouched={ho.get('untouched')} executions={ho.get('executions')} writers={ho.get('writer_split')}")
    lines.append(f"network calls     : {rep['network_calls']}")
    lines.append("")
    lines.append("BLOCKERS before live cloud use:" if rep["blockers"] else "no blockers")
    for b in rep["blockers"]:
        lines.append(f"  - {b}")
    lines.append(f"ready for zero-key dry run: {'YES' if rep['ready_for_zero_key_dry_run'] else 'NO'}")
    return "\n".join(lines)


def cmd_readiness(args) -> int:
    rep = readiness_report(models_config=Path(args.models_config) if args.models_config else None,
                           bench_root=Path(args.bench_root) if args.bench_root else None,
                           datasets_root=Path(args.datasets_root) if args.datasets_root else None,
                           state_root=Path(args.state_root) if args.state_root else None)
    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=1, default=str))
    else:
        print(format_readiness(rep))
    return 0


def add_readiness_command(sub) -> None:
    p = sub.add_parser("readiness", help="Zero-key product dry run: what is ready, what blocks live use (no calls)")
    p.add_argument("--models-config", default=None, help="models.toml (default: repo models.toml, else the example)")
    p.add_argument("--bench-root", default=None)
    p.add_argument("--datasets-root", default=None)
    p.add_argument("--state-root", default=None, help="campaign state root (ledger) to report spend from")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_readiness)


__all__ = ["readiness_report", "format_readiness", "role_status", "add_readiness_command", "CLOUD_TASKS"]
