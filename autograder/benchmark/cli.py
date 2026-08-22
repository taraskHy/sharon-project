"""`autograder bench ...` — the model-selection benchmark commands.

    bench list                                  roles, dataset status, candidates, held-out log
    bench inspect  --role R [--split S] [--component C] [--case-id ID]
                                                counts; DEV request preview (CALIBRATION/HELD_OUT previews refused)
    bench dry-run  --role R --split S --candidate SLUG [...]   plan + predicted cost, ZERO calls
    bench run      --role R --split S --candidate SLUG [...]   live run through ModelGateway (needs a credential)
    bench report   --run-dir DIR | --role R --split S [--historical]
    bench compare  --role R --split S [--component C]
    bench held-out-log

Safety defaults: `run` refuses UNSELECTED roles and unlisted slugs, refuses
HELD_OUT without --confirm-held-out (and logs it permanently), never repairs
malformed output silently, and shares one campaign state root so the $8
warning / $10 hard stop apply across every run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .manifests import (DEFAULT_BENCH_ROOT, DEFAULT_DATASETS_ROOT, ROLES, SPLITS, BenchmarkIntegrityError,
                        BenchmarkNotBuilt, all_manifest_summaries, load_manifest)
from .registry import DEFAULT_REGISTRY_PATH, load_registry
from .report import compare, format_table, historical_ocr_metrics, list_runs, load_run
from .roles import adapter_for
from .runner import (DEFAULT_RUNS_ROOT, DEFAULT_STATE_ROOT, HELD_OUT_LOG, HeldOutRefused, LeakageError,
                     RunSpec, UnselectedCandidate, held_out_executions, run_benchmark)


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _emit(obj, as_json: bool) -> None:
    if as_json:
        print(json.dumps(obj, ensure_ascii=False, indent=1, default=str))
    else:
        print(obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False, indent=1, default=str))


def cmd_bench_list(args) -> int:
    reg = load_registry(args.registry)
    mans = all_manifest_summaries(bench_root=Path(args.bench_root), datasets_root=Path(args.datasets_root))
    out = {"registry": reg.summary(), "benchmarks": mans,
           "held_out_executions": len(held_out_executions(Path(args.held_out_log))),
           "state_root": str(args.state_root), "runs_root": str(args.runs_root)}
    if args.json:
        _emit(out, True)
        return 0
    print(f"candidate registry: {reg.path} (v{reg.version}, {reg.updated}); budget warn ${reg.warn_usd} / hard ${reg.experiment_total_usd}")
    rows = []
    for role in ROLES:
        m = mans.get(role, {})
        rc = reg.roles.get(role)
        counts = m.get("counts") or {}
        rows.append({"role": role, "status": m.get("status"), "cases": m.get("cases", 0),
                     "DEV": sum((counts.get("DEV") or {}).values()),
                     "CALIBRATION": sum((counts.get("CALIBRATION") or {}).values()),
                     "HELD_OUT": sum((counts.get("HELD_OUT") or {}).values()),
                     "selection": rc.status if rc else "?", "candidates": len(rc.candidates) if rc else 0})
    print(format_table(rows, ["role", "status", "cases", "DEV", "CALIBRATION", "HELD_OUT", "selection", "candidates"]))
    print(f"held-out executions logged: {out['held_out_executions']} ({args.held_out_log})")
    return 0


def cmd_bench_inspect(args) -> int:
    m = load_manifest(args.role, bench_root=Path(args.bench_root), datasets_root=Path(args.datasets_root))
    summ = m.summary()
    if args.case_id or args.preview:
        split = (args.split or "DEV").upper()
        if split != "DEV" and not args.confirm_held_out:
            _log(f"refusing to preview {split} cases: only DEV may be inspected while developing "
                 "(pass --confirm-held-out to override deliberately; the override is logged)")
            return 2
        cases = m.by_split(split, args.component)
        if args.case_id:
            cases = [c for c in cases if c.case_id == args.case_id]
        if not cases:
            _log("no matching case")
            return 2
        c = cases[0]
        adapter = adapter_for(args.role)
        from .runner import files_root_for
        req = adapter.build_request(dict(c.inputs), files_root_for(m, Path(args.bench_root)))
        summ["preview"] = {"case_id": c.case_id, "split": c.split, "component": c.component,
                           "model_visible_inputs": c.inputs, "request_text": req.text_for_inspection(),
                           "provenance": req.provenance(),
                           "label_fields_withheld": sorted(c.label.keys())}
    _emit(summ, True)
    return 0


def _spec_from_args(args, dry_run: bool) -> RunSpec:
    return RunSpec(
        role=args.role, split=args.split, candidate=args.candidate, component=args.component,
        backend=args.backend, base_url=args.base_url,
        models_config=Path(args.models_config) if args.models_config else None,
        registry_path=Path(args.registry), bench_root=Path(args.bench_root),
        datasets_root=Path(args.datasets_root), state_root=Path(args.state_root),
        runs_root=Path(args.runs_root), held_out_log=Path(args.held_out_log),
        limit=args.limit, dry_run=dry_run, confirm_held_out=bool(args.confirm_held_out),
        retry_failed=bool(getattr(args, "retry_failed", False)), allow_unlisted=bool(args.allow_unlisted),
        note=args.note or "", max_tokens=args.max_tokens)


def _run(args, dry_run: bool) -> int:
    spec = _spec_from_args(args, dry_run)
    try:
        res = run_benchmark(spec, progress=_log)
    except HeldOutRefused as e:
        _log(f"REFUSED: {e}")
        return 3
    except UnselectedCandidate as e:
        _log(f"REFUSED: {e}")
        return 3
    except LeakageError as e:
        _log(f"LEAKAGE: {e}")
        return 4
    except (BenchmarkIntegrityError, BenchmarkNotBuilt) as e:
        _log(f"BENCHMARK: {e}")
        return 5
    except Exception as e:  # noqa: BLE001 — cloud readiness must read as one sentence
        from ..cloudcheck import CloudNotReady, explain_cloud_error
        if isinstance(e, CloudNotReady):
            _log(f"NOT READY: {e}")
            return 6
        friendly = explain_cloud_error(e)
        if friendly is not None:
            _log(f"NOT READY: {friendly}")
            return 6
        raise
    out = {"run_id": res.run_id, "run_dir": str(res.run_dir), "mode": "dry_run" if res.dry_run else "live",
           "role": res.role, "split": res.split, "component": res.component, "candidate": res.candidate,
           "cases_selected": res.cases_selected, "cases_done": res.cases_done, "cases_failed": res.cases_failed,
           "skipped_resume": res.cases_skipped_resume, "stopped_reason": res.stopped_reason,
           "predicted_cost": res.predicted_cost, "metrics": res.metrics, "usage": res.usage,
           "warnings": res.warnings, "provider_calls_made": 0 if res.dry_run else None}
    _emit(out, True)
    return 0


def cmd_bench_dry_run(args) -> int:
    return _run(args, dry_run=True)


def cmd_bench_run(args) -> int:
    if not args.i_understand_this_spends_money:
        _log("REFUSED: a live benchmark run spends OpenRouter budget. Re-run with "
             "--i-understand-this-spends-money (the $8 warning / $10 hard stop still apply).")
        return 3
    return _run(args, dry_run=False)


def cmd_bench_report(args) -> int:
    if args.historical:
        _emit(historical_ocr_metrics(), True)
        return 0
    if args.run_dir:
        _emit(load_run(Path(args.run_dir)), True)
        return 0
    rows = list_runs(args.role, Path(args.runs_root), args.split)
    if args.json:
        _emit(rows, True)
    else:
        print(format_table([{k: v for k, v in r.items() if k not in ("metrics", "usage", "plan")} for r in rows]))
    return 0


def cmd_bench_compare(args) -> int:
    out = compare(args.role, args.split, Path(args.runs_root), args.component)
    if args.json:
        _emit(out, True)
    else:
        print(format_table(out["runs"]))
        print(out["note"])
    return 0


def cmd_bench_held_out_log(args) -> int:
    rows = held_out_executions(Path(args.held_out_log))
    _emit({"path": str(args.held_out_log), "executions": rows}, True)
    return 0


def add_bench_commands(sub) -> None:
    b = sub.add_parser("bench", help="Model-selection benchmarks (frozen datasets; gateway-routed)")
    bs = b.add_subparsers(dest="bench_command", required=True)

    def common(p, *, needs_role=True, needs_split=False):
        if needs_role:
            p.add_argument("--role", required=True, choices=ROLES)
        if needs_split:
            p.add_argument("--split", required=True, type=str.lower,
                           choices=[s.lower() for s in SPLITS],
                           help="DEV (develop) | CALIBRATION (select) | HELD_OUT (needs --confirm-held-out)")
        p.add_argument("--component", default=None, help="ocr_verify: REAL | SYNTHETIC (default: both, reported separately)")
        p.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
        p.add_argument("--bench-root", default=str(DEFAULT_BENCH_ROOT))
        p.add_argument("--datasets-root", default=str(DEFAULT_DATASETS_ROOT))
        p.add_argument("--state-root", default=str(DEFAULT_STATE_ROOT), help="shared campaign cache/ledger/budget root")
        p.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
        p.add_argument("--held-out-log", default=str(HELD_OUT_LOG))
        p.add_argument("--json", action="store_true")

    p = bs.add_parser("list", help="roles, dataset status, candidates"); common(p, needs_role=False)
    p.set_defaults(func=cmd_bench_list)

    p = bs.add_parser("inspect", help="manifest summary; DEV request preview"); common(p)
    p.add_argument("--split", default="dev", type=str.lower, choices=[s.lower() for s in SPLITS])
    p.add_argument("--case-id", default=None)
    p.add_argument("--preview", action="store_true", help="render the first case's model-visible request")
    p.add_argument("--confirm-held-out", action="store_true")
    p.set_defaults(func=cmd_bench_inspect)

    for name, fn, help_ in (("dry-run", cmd_bench_dry_run, "plan + predicted cost; ZERO provider calls"),
                            ("run", cmd_bench_run, "LIVE run through ModelGateway (spends budget)")):
        p = bs.add_parser(name, help=help_); common(p, needs_split=True)
        p.add_argument("--candidate", default=None, help="model slug (must be registered in candidates.toml)")
        p.add_argument("--backend", default="openrouter", help="openrouter | openai (compat endpoint) | mock")
        p.add_argument("--base-url", default=None)
        p.add_argument("--models-config", default=None, help="models.toml for [pricing] (estimator) only")
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--max-tokens", type=int, default=None)
        p.add_argument("--note", default="")
        p.add_argument("--confirm-held-out", action="store_true")
        p.add_argument("--allow-unlisted", action="store_true")
        if name == "run":
            p.add_argument("--retry-failed", action="store_true", help="explicitly re-attempt failed cases (recorded)")
            p.add_argument("--i-understand-this-spends-money", action="store_true")
        p.set_defaults(func=fn)

    p = bs.add_parser("report", help="one run or all runs of a role"); common(p, needs_role=False)
    p.add_argument("--role", default=None, choices=ROLES)
    p.add_argument("--split", default=None, type=str.lower, choices=[s.lower() for s in SPLITS])
    p.add_argument("--run-dir", default=None)
    p.add_argument("--historical", action="store_true", help="ocr_primary: historical outputs re-scored on audited refs")
    p.set_defaults(func=cmd_bench_report)

    p = bs.add_parser("compare", help="candidates side by side (no winner chosen)"); common(p, needs_split=True)
    p.set_defaults(func=cmd_bench_compare)

    p = bs.add_parser("held-out-log", help="permanent log of HELD_OUT executions"); common(p, needs_role=False)
    p.set_defaults(func=cmd_bench_held_out_log)


__all__ = ["add_bench_commands"]
