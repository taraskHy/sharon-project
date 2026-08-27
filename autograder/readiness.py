"""Zero-key product readiness check (dry run) — `autograder readiness`.

Answers "is the product ready to run, and what exactly is still missing?"
WITHOUT a single model or network call: every section is computed from
files, configuration and the environment's *presence* of a credential.

Headline (Part 11 of the pre-API setup):

    PRE-API SETUP COMPLETE: YES/NO      (API KEY = NOT INSTALLED never blocks it)
    READY FOR API KEY: YES/NO           (+ blockers)
    OWNER ACTION REQUIRED: ...          (owner work such as grading labels)

Categories: REFERENCE GROUND TRUTH, OCR_PRIMARY DATASET, OCR_VERIFY DATASET,
GRADE_PRIMARY DATASET, GRADE_ESCALATE DATASET, MC_RESOLVE DATASET,
VARIANT_RESOLVE DATASET, ALIGN_RESOLVE DATASET, MODEL CONFIG, BUDGET,
COURSE STORE, PRODUCTION VERIFIER, GUI, HELD_OUT PROTECTION, API KEY.

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
#: The tasks a PRODUCTION configuration may route to the cloud — the mirror
#: of cloudboundary.CLOUD_OCR_ALLOWLIST. Since the 2026-08 re-architecture
#: grading is local-only; the cloud grade/aux roles below are research-only.
CLOUD_TASKS = ("ocr_primary", "ocr_verify")
#: Production roles that must be LOCAL (grading never leaves the machine).
PRODUCTION_LOCAL_TASKS = ("grade_primary", "grade_escalate")
#: Roles that exist only for the historical research benchmarks; a production
#: request on them is refused by the cloud boundary.
RESEARCH_ONLY_CLOUD_TASKS = ("mc_resolve_cloud", "variant_resolve_cloud",
                             "align_resolve_cloud", "policy_infer_cloud")
#: roles whose benchmark must be READY (with a frozen smoke subset) before
#: the first live experiment — the rest may be owner-pending
FIRST_EXPERIMENT_ROLES = ("ocr_primary", "ocr_verify")
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
    env_slugs: list[str] = []
    for task, sec in models.items():
        backend = str(sec.get("backend", ""))
        raw = str(sec.get("model", "") or "")
        enabled = bool(sec.get("enabled", True))
        env_refs = _ENV_REF.findall(raw)
        env_set = all(bool(os.environ.get(e)) for e in env_refs) if env_refs else None
        cloud = backend in ("openrouter", "anthropic") or (backend == "openai" and "openrouter" in str(sec.get("base_url", "")))
        if env_refs and cloud:
            env_slugs.append(task)
        if not enabled:
            status = "DISABLED"
        elif raw == "UNSELECTED" or (env_refs and not env_set):
            status = "UNSELECTED"
        elif cloud:
            status = "CONFIGURED_CLOUD"
        else:
            status = "SELECTED_LOCAL"
        shown = raw if not env_refs else ("${" + env_refs[0] + "}" + (" (set)" if env_set else " (unset)"))
        # The production boundary: a cloud route on a non-OCR task can never
        # execute in production (cloudboundary.py) — surface that here so the
        # GUI shows DISABLED IN PRODUCTION instead of a live-looking cloud row.
        blocked = bool(cloud) and task not in CLOUD_TASKS
        if blocked and status == "CONFIGURED_CLOUD":
            status = "BLOCKED_IN_PRODUCTION"
        tasks[task] = {"backend": backend, "model": shown, "status": status, "cloud": cloud,
                       "blocked_in_production": blocked, "base_url": sec.get("base_url")}
    for task in CLOUD_TASKS:
        tasks.setdefault(task, {"backend": None, "model": None, "status": "ABSENT", "cloud": True,
                                "blocked_in_production": False})
    for task in PRODUCTION_LOCAL_TASKS:
        tasks.setdefault(task, {"backend": None, "model": None, "status": "ABSENT", "cloud": False,
                                "blocked_in_production": False})
    pricing = data.get("pricing") or {}
    priced = [k for k, v in pricing.items() if isinstance(v, dict) and (v.get("input") or v.get("output"))]
    return {"config": str(cfg_path), "exists": True, "using_example_as_template": using_example,
            "tasks": tasks,
            "unselected": sorted(t for t, v in tasks.items() if v["status"] in ("UNSELECTED", "ABSENT")),
            "configured_cloud": sorted(t for t, v in tasks.items() if v["status"] == "CONFIGURED_CLOUD"),
            # cloud routes the boundary will refuse at call time — they cannot
            # run, but a hardcoded cloud grading slug in models.toml is still
            # worth surfacing as a config smell rather than silently "OK"
            "blocked_in_production": sorted(t for t, v in tasks.items()
                                            if v.get("blocked_in_production")),
            "env_slug_cloud_tasks": env_slugs,
            "budget_section": data.get("budget") or {}, "pricing_table": bool(pricing),
            "pricing_entries": sorted(pricing), "pricing_priced": priced}


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
                               "chunks": st.get("n_chunks"), "embed_model": st.get("embed_model"),
                               "built": st.get("built")})
    out["count"] = len(out["courses"])
    out["env"] = {"GRADER_COURSES_DIR": os.environ.get("GRADER_COURSES_DIR")}
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
                entry["preflight"] = _safe(_pf, {"status": "UNKNOWN", "summary": "package check unavailable"})
                if entry["preflight"].get("error"):
                    entry["preflight"]["summary"] = f"package check unavailable: {entry['preflight']['error']}"
            else:
                entry["preflight"] = {"status": "KEY_NOT_PARSED",
                                      "summary": "answer key not parsed yet (PDF only) — parse it in Exam setup"}
            out["packages"].append(entry)
    out["count"] = len(out["packages"])
    out["env"] = {"GRADER_PACKAGE_DIRS": os.environ.get("GRADER_PACKAGE_DIRS")}
    return out


def _benchmarks(bench_root: Path | None, datasets_root: Path | None) -> dict:
    from .benchmark.manifests import DEFAULT_BENCH_ROOT, DEFAULT_DATASETS_ROOT, all_manifest_summaries
    summ = all_manifest_summaries(bench_root=bench_root or DEFAULT_BENCH_ROOT,
                                  datasets_root=datasets_root or DEFAULT_DATASETS_ROOT)
    return {role: {"status": s.get("status"), "cases": s.get("cases"), "counts": s.get("counts"),
                   "hashes": s.get("hashes"), "error": s.get("error")} for role, s in summ.items()}


def _dataset_statuses(bench_root: Path | None, datasets_root: Path | None) -> dict:
    from .benchmark.manifests import DEFAULT_BENCH_ROOT, DEFAULT_DATASETS_ROOT, load_manifest
    from .benchmark.smoke import smoke_status
    from .benchmark.status import all_role_statuses
    br, dr = bench_root or DEFAULT_BENCH_ROOT, datasets_root or DEFAULT_DATASETS_ROOT
    st = all_role_statuses(bench_root=br, datasets_root=dr)
    for role in FIRST_EXPERIMENT_ROLES:
        try:
            m = load_manifest(role, bench_root=br, datasets_root=dr)
        except Exception:  # noqa: BLE001
            m = None
        st[role]["smoke"] = _safe(lambda: smoke_status(role, m), {"frozen": False})
    return st


def _reference_truth(bench_root: Path | None) -> dict:
    from .benchmark.manifests import DEFAULT_BENCH_ROOT, load_manifest, reference_breakdown
    m = load_manifest("ocr_primary", bench_root=bench_root or DEFAULT_BENCH_ROOT)
    b = reference_breakdown(m)
    b.pop("rows", None)
    return b


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
            "live_preflight": "credential -> GET /api/v1/key (explicit) -> checkpoint -> compare with ledger -> budget safe? -> call",
            "calls_made_by_this_check": 0}


def _budget(roles: dict, state_root: Path | None) -> dict:
    from .benchmark.registry import load_registry
    from .benchmark.runner import DEFAULT_STATE_ROOT
    from .spend import (EXPERIMENT_HARD_STOP_USD, EXPERIMENT_WARN_USD, budget_status, key_usage_checkpoints,
                        ledger_summary)
    reg = _safe(lambda: load_registry().summary(), {})
    root = Path(state_root) if state_root else DEFAULT_STATE_ROOT
    led = ledger_summary(root / "gateway_ledger" / "usage.jsonl")
    warn = (reg.get("budget") or {}).get("warn_usd") or EXPERIMENT_WARN_USD
    hard = (reg.get("budget") or {}).get("experiment_total_usd") or EXPERIMENT_HARD_STOP_USD
    mb = roles.get("budget_section") or {}
    return {"policy": {"warning_usd": warn, "hard_stop_usd": hard,
                       "source": "evaluation/model_selection/candidates.toml [budget]"},
            "models_toml_budget": mb,
            "models_toml_enforces_campaign_ceiling": (float(mb.get("max_cost_total") or 0) == float(hard)
                                                     and float(mb.get("soft_fraction") or 0) == round(warn / hard, 4)),
            "campaign_state_root": str(root),
            "campaign_ledger": {"path": led["path"], "exists": led["exists"], "cloud_calls": led["cloud_calls"],
                                "cumulative_cost": led["cumulative_cost"], "by_task": led["by_task"],
                                "by_model": led["by_model"]},
            "key_usage_checkpoints": len(key_usage_checkpoints(root)),
            "status": budget_status(led["cumulative_cost"], warn_usd=warn, hard_usd=hard)}


def _verifier_crops() -> dict:
    from .evidencecrops import production_crop_provider
    d = production_crop_provider().describe()
    d["benchmark"] = "READY (frozen REAL + SYNTHETIC_NEAR_MISS)"
    d["production"] = ("SAFE BUT UNAVAILABLE — calibrated explanation crop geometry does not exist; "
                       "suspicious OCR -> no trusted crop -> REVIEW; no verifier provider call; "
                       "no invented coordinates; no full-page sends")
    return d


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
            "protection": ["HELD_OUT cannot be dry-run or previewed (bench dry-run / inspect refuse)",
                           "only `bench final-eval --confirm-held-out --i-understand-this-spends-money` executes it (live)",
                           "every execution is appended permanently with git commit / config / model / prompt / schema hashes",
                           "inspected held-out results are demoted to DEV and can never again be treated as unseen"]}


# ----------------------------------------------------------------------------
# the report
# ----------------------------------------------------------------------------

def _cat(name: str, status: str, ok: bool, detail: str = "", owner_action: str | None = None) -> dict:
    return {"category": name, "status": status, "ok": ok, "detail": detail, "owner_action": owner_action}


def readiness_report(*, models_config: Path | None = None, bench_root: Path | None = None,
                     datasets_root: Path | None = None, state_root: Path | None = None) -> dict[str, Any]:
    roles = _safe(lambda: role_status(models_config), {"tasks": {}, "unselected": []})
    rep: dict[str, Any] = {
        "mode": "zero-key dry run (no model or network calls)",
        "reference_truth": _safe(lambda: _reference_truth(bench_root), {}),
        "datasets": _safe(lambda: _dataset_statuses(bench_root, datasets_root), {}),
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

    # ---- categories ---------------------------------------------------------
    cats: list[dict] = []
    rt = rep["reference_truth"] or {}
    ok_ref = bool(rt.get("all_valid_for_strict_scoring"))
    hand, other = rt.get("handwritten_manual_audit") or {}, rt.get("other_categories_text_layer") or {}
    cats.append(_cat("REFERENCE GROUND TRUTH", "OK" if ok_ref else "INVALID", ok_ref,
                     f"{rt.get('total')} items: {hand.get('count')} manually audited "
                     f"({hand.get('confirmed')} confirmed / {hand.get('corrected')} corrected / "
                     f"{hand.get('ambiguous')} ambiguous) + {other.get('count')} text-layer (mechanical); "
                     f"invalid: {rt.get('invalid_items')}" if rt else f"unavailable: {rt.get('error')}"))
    ds = rep["datasets"] or {}
    for role, label in (("ocr_primary", "OCR_PRIMARY DATASET"), ("ocr_verify", "OCR_VERIFY DATASET"),
                        ("grade_primary", "GRADE_PRIMARY DATASET"), ("grade_escalate", "GRADE_ESCALATE DATASET"),
                        ("mc_resolve_cloud", "MC_RESOLVE DATASET"), ("variant_resolve", "VARIANT_RESOLVE DATASET"),
                        ("align_resolve", "ALIGN_RESOLVE DATASET")):
        d = ds.get(role) or {}
        status = d.get("status") or "UNKNOWN"
        detail = d.get("detail") or ""
        smoke = d.get("smoke")
        if smoke is not None:
            detail += (f"; smoke frozen ({smoke.get('cases')} cases, {str(smoke.get('selection_sha256'))[:12]}…)"
                       if smoke.get("frozen") and smoke.get("valid") else "; SMOKE SUBSET MISSING/INVALID")
        ok = status in ("READY",) and (smoke is None or (smoke.get("frozen") and smoke.get("valid")))
        if role not in FIRST_EXPERIMENT_ROLES:
            # owner-pending / pending / not-available states are explicit, not failures
            ok = status in ("READY", "PARTIALLY_READY", "NEEDS_OWNER_LABELS", "PENDING_OTHER_EXPERIMENT", "NOT_AVAILABLE")
        cats.append(_cat(label, status, ok, detail, d.get("owner_action")))
    # model config
    mc_ok = (roles.get("exists") and not roles.get("using_example_as_template")
             and not roles.get("configured_cloud") and not roles.get("env_slug_cloud_tasks"))
    cats.append(_cat("MODEL CONFIG", "OK" if mc_ok else ("MISSING models.toml" if roles.get("using_example_as_template") else "CHECK"),
                     bool(mc_ok),
                     f"{roles.get('config')}; cloud roles UNSELECTED: {len([t for t in roles.get('unselected', []) if t in CLOUD_TASKS])}"
                     f"; configured cloud (should be 0 before selection): {roles.get('configured_cloud')}"
                     f"; ${{ENV}} cloud slugs (should be none): {roles.get('env_slug_cloud_tasks')}"
                     f"; pricing entries priced: {len(roles.get('pricing_priced') or [])}"))
    bd = rep["budget"] or {}
    b_ok = bool(bd.get("models_toml_enforces_campaign_ceiling")) and (bd.get("status") or {}).get("state") != "HARD_STOP"
    cats.append(_cat("BUDGET", "OK" if b_ok else "CHECK", b_ok,
                     f"warning ${bd.get('policy', {}).get('warning_usd')} / hard stop ${bd.get('policy', {}).get('hard_stop_usd')}; "
                     f"models.toml enforces ceiling: {bd.get('models_toml_enforces_campaign_ceiling')}; "
                     f"campaign spent ${(bd.get('status') or {}).get('cumulative_cost')} ({(bd.get('status') or {}).get('state')}); "
                     f"state root {bd.get('campaign_state_root')}"))
    cs = rep["course_store"]
    cats.append(_cat("COURSE STORE", f"{cs.get('count', 0)} course(s), {rep['rag_index']['indexed_courses']} indexed", True,
                     f"{cs.get('root')}" + (f"; stale: {[c['course_id'] for c in cs.get('courses', []) if c.get('stale')]}" if cs.get("courses") else "")))
    vc = rep["verifier_crops"] or {}
    cats.append(_cat("PRODUCTION VERIFIER", "BENCHMARK READY / PRODUCTION SAFE BUT UNAVAILABLE", True,
                     vc.get("production", "")))
    g = rep["gui"]
    cats.append(_cat("GUI", "OK" if g["streamlit_installed"] and g["app_exists"] else "MISSING",
                     bool(g["streamlit_installed"] and g["app_exists"]), ", ".join(g["screens"])))
    ho = rep["held_out"] or {}
    cats.append(_cat("HELD_OUT PROTECTION", "OK (untouched)" if ho.get("untouched") else f"EXECUTED x{ho.get('executions')}",
                     bool(ho.get("untouched")), "; ".join(ho.get("protection", []))))
    key_present = rep["openrouter"]["configured"] == "YES"
    cats.append(_cat("API KEY", "INSTALLED" if key_present else "NOT INSTALLED", True,
                     "intentionally absent until the owner decides; never blocks setup completeness"))
    rep["categories"] = cats

    blockers = [f"{c['category']}: {c['status']} — {c['detail']}" for c in cats
                if not c["ok"] and c["category"] != "API KEY"]
    owner_actions = [f"{c['category']}: {c['owner_action']}" for c in cats if c.get("owner_action")]
    if not roles.get("pricing_priced"):
        owner_actions.append("MODEL CONFIG: populate/verify [pricing] in models.toml from the live OpenRouter "
                             "model page immediately before the first paid run (estimator + predicted-cost refusal)")
    for b in (rep["benchmarks"] or {}).values():
        if b.get("status") == "INTEGRITY_ERROR":
            blockers.append(f"benchmark INTEGRITY ERROR — {b.get('error')}")
    rep["blockers"] = blockers
    rep["owner_actions"] = owner_actions
    rep["pre_api_setup_complete"] = not blockers
    rep["ready_for_api_key"] = not blockers
    rep["ready_for_zero_key_dry_run"] = all(
        b.get("status") != "INTEGRITY_ERROR" for b in (rep["benchmarks"] or {}).values()) and rep["gui"]["streamlit_installed"]
    return rep


def format_readiness(rep: dict) -> str:
    lines = [f"READINESS — {rep['mode']}", ""]
    lines.append(f"PRE-API SETUP COMPLETE: {'YES' if rep.get('pre_api_setup_complete') else 'NO'}")
    lines.append(f"READY FOR API KEY: {'YES' if rep.get('ready_for_api_key') else 'NO'}")
    lines.append("")
    w = max(len(c["category"]) for c in rep.get("categories", [])) if rep.get("categories") else 20
    for c in rep.get("categories", []):
        mark = "✓" if c["ok"] else "✗"
        lines.append(f"  {mark} {c['category']:<{w}}  {c['status']}")
        if c.get("detail"):
            lines.append(f"      {c['detail']}")
    lines.append("")
    if rep.get("blockers"):
        lines.append("BLOCKERS:")
        for b in rep["blockers"]:
            lines.append(f"  - {b}")
    else:
        lines.append("BLOCKERS: none")
    if rep.get("owner_actions"):
        lines.append("OWNER ACTION REQUIRED:")
        for a in rep["owner_actions"]:
            lines.append(f"  - {a}")
    lines.append("")
    cs = rep["course_store"]; lines.append(f"course store      : {cs.get('count', 0)} course(s) at {cs.get('root')}")
    for c in cs.get("courses", []):
        lines.append(f"                    - {c['course_id']}: {c.get('sources')} source(s), indexed={c.get('indexed')}, "
                     f"chunks={c.get('chunks')}, stale={c.get('stale')}, embed={c.get('embed_model')}")
    ep = rep["exam_packages"]; lines.append(f"exam packages     : {ep.get('count', 0)} found in {ep.get('dirs')}")
    for p in ep.get("packages", []):
        pf = p.get("preflight") or {}
        lines.append(f"                    - {p['template']}: {pf.get('status')} — {pf.get('summary', '')}")
    mr = rep["model_roles"]; lines.append(f"model roles       : {mr.get('config')}" + (" (EXAMPLE used as template)" if mr.get("using_example_as_template") else ""))
    for t, v in (mr.get("tasks") or {}).items():
        lines.append(f"                    - {t:<22} {v.get('status'):<16} {v.get('backend') or '-':<10} {v.get('model') or '-'}")
    lm = rep["local_models"]; lines.append(f"local models      : {len(lm.get('configured_local_routes') or {})} local route(s); probed={lm.get('probed')}")
    orr = rep["openrouter"]; lines.append(f"OpenRouter        : configured: {orr['configured']} — {orr['key_metadata_endpoint']}")
    lines.append(f"                    live preflight: {orr['live_preflight']}")
    bd = rep["budget"]; st = bd.get("status") or {}
    lines.append(f"budget            : warn ${bd.get('policy', {}).get('warning_usd')} / hard ${bd.get('policy', {}).get('hard_stop_usd')}; "
                 f"campaign ledger spent ${st.get('cumulative_cost')} ({st.get('state')}); state root {bd.get('campaign_state_root')}")
    vc = rep["verifier_crops"]; lines.append(f"verifier          : benchmark {vc.get('benchmark')}; production {vc.get('production')}")
    ho = rep["held_out"]; lines.append(f"held-out          : untouched={ho.get('untouched')} executions={ho.get('executions')} writers={ho.get('writer_split')}")
    lines.append(f"network calls     : {rep['network_calls']}")
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


__all__ = ["readiness_report", "format_readiness", "role_status", "add_readiness_command", "CLOUD_TASKS",
           "FIRST_EXPERIMENT_ROLES"]
