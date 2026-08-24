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
        if split == "HELD_OUT":
            _log("HELD_OUT is reserved for final evaluation and cannot be previewed/dry-run.")
            return 3
        if split != "DEV":
            _log(f"refusing to preview {split} cases: only DEV may be inspected while developing")
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


def _spec_from_args(args, dry_run: bool, *, final_evaluation: bool = False) -> RunSpec:
    return RunSpec(
        role=args.role, split=args.split, candidate=args.candidate, component=args.component,
        subset=getattr(args, "subset", None), final_evaluation=final_evaluation,
        backend=args.backend, base_url=args.base_url,
        models_config=Path(args.models_config) if args.models_config else None,
        registry_path=Path(args.registry), bench_root=Path(args.bench_root),
        datasets_root=Path(args.datasets_root), state_root=Path(args.state_root),
        runs_root=Path(args.runs_root), held_out_log=Path(args.held_out_log),
        limit=args.limit, dry_run=dry_run, confirm_held_out=bool(args.confirm_held_out),
        retry_failed=bool(getattr(args, "retry_failed", False)), allow_unlisted=bool(args.allow_unlisted),
        note=args.note or "", max_tokens=args.max_tokens)


def _run(args, dry_run: bool, *, final_evaluation: bool = False) -> int:
    spec = _spec_from_args(args, dry_run, final_evaluation=final_evaluation)
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
    if args.split == "held_out":
        _log("REFUSED: HELD_OUT is reserved for final evaluation; use `bench final-eval` "
             "(explicitly confirmed, permanently logged).")
        return 3
    if not args.i_understand_this_spends_money:
        _log("REFUSED: a live benchmark run spends OpenRouter budget. Re-run with "
             "--i-understand-this-spends-money (the $8 warning / $10 hard stop still apply).")
        return 3
    return _run(args, dry_run=False)


def cmd_bench_final_eval(args) -> int:
    """The ONLY path that may execute HELD_OUT: live, confirmed, logged."""
    if not args.confirm_held_out:
        _log("REFUSED: final evaluation on HELD_OUT requires --confirm-held-out. Once executed and "
             "inspected, the held-out split can never again be treated as unseen.")
        return 3
    if not args.i_understand_this_spends_money:
        _log("REFUSED: final evaluation spends OpenRouter budget; pass --i-understand-this-spends-money.")
        return 3
    args.split = "held_out"
    return _run(args, dry_run=False, final_evaluation=True)


def cmd_bench_smoke(args) -> int:
    from .smoke import SMOKE_RULES, freeze_smoke, load_smoke, propose_smoke, smoke_status
    m = load_manifest(args.role, bench_root=Path(args.bench_root), datasets_root=Path(args.datasets_root))
    root = Path(args.smoke_root)
    if args.smoke_command == "propose":
        _emit(propose_smoke(args.role, m), True)
        return 0
    if args.smoke_command == "freeze":
        try:
            d = freeze_smoke(args.role, m, root,
                             allow_unfilled=getattr(args, "allow_unfilled", False))
        except Exception as e:  # noqa: BLE001
            _log(f"REFUSED: {e}")
            return 3
        _emit({"frozen": str(root / f"{args.role}_smoke.json"), "cases": len(d["cases"]),
               "selection_sha256": d["selection_sha256"], "unfilled_slots": d["unfilled_slots"]}, True)
        return 0
    if args.smoke_command == "show":
        st = smoke_status(args.role, m, root)
        if st.get("frozen") and st.get("valid"):
            st["cases_detail"] = load_smoke(args.role, m, root)["cases"]
        _emit(st, True)
        return 0
    _log("unknown smoke command")
    return 2


def cmd_bench_build(args) -> int:
    """Dataset builders — local deterministic processing only, NO model calls.
    Each refuses to overwrite an existing frozen dataset."""
    from . import datasets as ds
    root = Path(args.datasets_root)
    try:
        if args.build_command == "build-grading":
            man = ds.build_grading_dataset(root / "grade_primary",
                                           key_json=Path(args.key_json) if args.key_json else None,
                                           bench_root=Path(args.bench_root))
            _emit({"built": str(root / "grade_primary"), "cases": man["cases"],
                   "excluded_cells": man.get("excluded_cells"), "inputs_sha256": man["inputs_sha256"],
                   "labels_sha256": man["labels_sha256"],
                   "owner_action": "label the cases with: python -m streamlit run scripts/grade_label_ui.py"}, True)
        elif args.build_command == "build-mc":
            man = ds.build_mc_dataset(root / "mc_resolve_cloud")
            _emit({"built": str(root / "mc_resolve_cloud"), "cases": man["cases"],
                   "ambiguous_rows_per_exam": man["extra"]["ambiguous_rows_per_exam"],
                   "inputs_sha256": man["inputs_sha256"], "labels_sha256": man["labels_sha256"]}, True)
        elif args.build_command == "build-variant":
            man = ds.build_variant_dataset(root / "variant_resolve")
            _emit({"built": str(root / "variant_resolve"), "cases": man["cases"],
                   "inputs_sha256": man["inputs_sha256"], "labels_sha256": man["labels_sha256"]}, True)
        elif args.build_command == "build-escalation":
            if not args.from_run:
                _log("REFUSED: --from-run <grade_primary run dir> is required (escalation cases are harvested, "
                     "never chosen by hand); status stays PENDING_PRIMARY_RESULTS")
                return 3
            man = ds.build_escalation_dataset(root / "grade_escalate", from_run_dir=Path(args.from_run),
                                              grade_dataset_dir=root / "grade_primary")
            _emit({"built": str(root / "grade_escalate"), "cases": man["cases"]}, True)
        elif args.build_command == "build-align":
            _log("NOT AVAILABLE: the printed variant booklets (test/003_70.pdf A2, test/002_76.pdf A3) are image "
                 "scans without a text layer, so the model-visible printed (id, text) list needs an OCR pass; "
                 "synthesizing printed text from canonical text is solved by the deterministic stage (margin 0.75) "
                 "and would not exercise the model role. The operator-verified mapping "
                 "(sample_data/Exam_solution.alignment.json) is the label once printed texts exist.")
            return 5
        else:
            _log("unknown build command")
            return 2
    except (ds.DatasetExists, ds.DatasetBuildError) as e:
        _log(f"REFUSED: {e}")
        return 3
    return 0


def cmd_bench_repair_grading_evidence(args) -> int:
    """Re-freeze ONLY the label-side evidence inventory of the frozen
    grade_primary dataset from the upstream line records (one image per
    recorded line). Inputs stay byte-identical; the revision is recorded."""
    from . import datasets as ds
    d = Path(args.datasets_root) / "grade_primary"
    try:
        res = ds.repair_grading_evidence(d, bench_root=Path(args.bench_root), dry_run=bool(args.dry_run))
    except ds.DatasetBuildError as e:
        _log(f"REFUSED: {e}")
        return 3
    _emit(res, True)
    return 0


def cmd_bench_evidence_repairs(args) -> int:
    """Status + integrity of the manual GRADE_PRIMARY evidence repairs (read-only)."""
    from .evidence_repairs import verify_repairs
    d = Path(args.datasets_root) / args.role
    if not (d / "manifest.json").exists():
        _log(f"REFUSED: no frozen dataset at {d}")
        return 3
    rep = verify_repairs(d)
    if not args.json:
        print(f"{rep['repaired']} of {rep['expected']} expected line repair(s) recorded "
              f"({rep['by_disposition']}); remaining: {rep['remaining'] or 'none'}")
        for p in rep["problems"]:
            print(f"  PROBLEM {p['line_id']}: {p['problem']}")
        print(f"frozen OCR benchmark: " + ", ".join(f"{k}={v[:12]}…" for k, v in rep["frozen_bench_sha256"].items()))
        print("repair tool: .venv\\Scripts\\python.exe -m streamlit run scripts\\evidence_repair_ui.py "
              "-- --browser.gatherUsageStats false")
        return 0 if not rep["problems"] else 1
    _emit(rep, True)
    return 0 if not rep["problems"] else 1


def cmd_bench_apply_evidence_repairs(args) -> int:
    """Fold completed human repairs into the grading dataset (records a manifest revision)."""
    from .datasets import DatasetBuildError
    from .evidence_repairs import RepairError, apply_repairs
    d = Path(args.datasets_root) / args.role
    try:
        out = apply_repairs(d, dry_run=bool(args.dry_run), allow_partial=bool(args.allow_partial))
    except (RepairError, DatasetBuildError) as e:
        _log(f"REFUSED: {e}")
        return 3
    _emit(out, True)
    if out["written"]:
        _log(f"[repairs] model input CHANGED: inputs_sha256 {out['previous_inputs_sha256'][:12]}… -> "
             f"{out['inputs_sha256'][:12]}… (recorded in the manifest revisions)")
        _log("[repairs] rebuild the labeling bundle to serve the repaired evidence: "
             "python -m labeling_app build-bundle --replace")
    return 0


def cmd_bench_missing_transcriptions(args) -> int:
    """Which grading cases are NOT measurable for model accuracy because a
    recorded student line has no audited transcription — and exactly which line
    that is. Read-only; the values come from the dataset's own evidence records,
    nothing is OCR'd or inferred."""
    m = load_manifest(args.role, bench_root=Path(args.bench_root), datasets_root=Path(args.datasets_root))
    rows = []
    for c in sorted(m.cases, key=lambda c: c.case_id):
        missing = list(c.label.get("lines_without_audited_transcription") or [])
        if not missing:
            continue
        lines = {e.get("sample_id"): e for e in (c.label.get("evidence_lines") or []) if e.get("sample_id")}
        rows.append({
            "case_id": c.case_id, "split": c.split,
            "line_count": c.label.get("line_count"),
            "lines_transcribed": (c.label.get("line_count") or 0) - len(missing),
            "audited": f"{(c.label.get('line_count') or 0) - len(missing)}/{c.label.get('line_count')}",
            "missing": [{"sample_id": s,
                         "image": (lines.get(s) or {}).get("image"),
                         "transcription_status": (lines.get(s) or {}).get("transcription_status"),
                         "line_index": (lines.get(s) or {}).get("index")} for s in missing],
            "line_inventory_source": c.label.get("line_inventory_source"),
            "has_ground_truth_score": c.label.get("score") is not None,
        })
    out = {"role": args.role, "cases_total": len(m.cases), "cases_incomplete": len(rows),
           "cases_complete": len(m.cases) - len(rows),
           "note": ("the grading model reads the TRANSCRIPTION (autograder/escalation.grade_prompt builds a "
                    "text-only request), so a line without an audited transcription is invisible to it: these "
                    "cases keep their ground-truth score but are excluded from accuracy metrics"),
           "cases": rows}
    if args.json:
        _emit(out, True)
        return 0
    print(f"{out['cases_incomplete']} of {out['cases_total']} {args.role} case(s) have an incomplete transcription")
    for r in rows:
        print(f"    {r['case_id']}")
        for mrow in r["missing"]:
            print(f"      missing: {mrow['sample_id']}")
            print(f"      image:   {mrow['image']}")
            print(f"      status:  {mrow['transcription_status']}")
        print(f"      line_count: {r['line_count']}")
        print(f"      audited: {r['audited']}")
    print(out["note"])
    return 0


def cmd_bench_import_final_labels(args) -> int:
    """FINAL labels from the shared labeling app (labeling_app export) ->
    datasets/<role>/final_labels.json. Only agreement/adjudicated rows."""
    from .finallabels import import_final_labels
    d = Path(args.datasets_root) / args.role
    if not (d / "manifest.json").exists():
        _log(f"REFUSED: no frozen dataset at {d}")
        return 3
    try:
        res = import_final_labels(Path(args.export), d)
    except (ValueError, FileNotFoundError) as e:
        _log(f"REFUSED: {e}")
        return 3
    _emit(res, True)
    return 0


def cmd_bench_owner_labels(args) -> int:
    from .ownerlabels import OwnerLabelStore
    d = Path(args.datasets_root) / args.role
    m = load_manifest(args.role, bench_root=Path(args.bench_root), datasets_root=Path(args.datasets_root))
    store = OwnerLabelStore(d)
    summ = store.summary([c.case_id for c in m.cases])
    summ["role"] = args.role
    _emit(summ, True)
    return 0


def cmd_bench_references(args) -> int:
    from .manifests import reference_breakdown
    m = load_manifest("ocr_primary", bench_root=Path(args.bench_root), datasets_root=Path(args.datasets_root))
    b = reference_breakdown(m)
    if args.json:
        _emit(b, True)
        return 0
    h, o = b["handwritten_manual_audit"], b["other_categories_text_layer"]
    print(f"{b['total']} total")
    print(f"├── handwritten (manual audit required): {h['count']}")
    print(f"│   ├── confirmed: {h['confirmed']}")
    print(f"│   ├── corrected: {h['corrected']}")
    print(f"│   ├── ambiguous: {h['ambiguous']}")
    print(f"│   └── by category: {h['by_category']}")
    print(f"└── other categories (text layer, no manual audit): {o['count']}")
    print(f"    ├── by category: {o['by_category']}")
    print(f"    └── why trustworthy: {o['why_trustworthy']}")
    print(f"by provenance class: {b['by_provenance_class']}")
    print(f"all valid for strict scoring: {b['all_valid_for_strict_scoring']}; invalid items: {b['invalid_items']}")
    print(f"frozen: {b['frozen']}")
    return 0


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

    p = bs.add_parser("inspect", help="manifest summary; DEV request preview (HELD_OUT never previewable)"); common(p)
    p.add_argument("--split", default="dev", type=str.lower, choices=[s.lower() for s in SPLITS])
    p.add_argument("--case-id", default=None)
    p.add_argument("--preview", action="store_true", help="render the first case's model-visible request")
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
        p.add_argument("--subset", default=None, choices=["smoke"],
                       help="smoke = the frozen pre-registered DEV smoke subset (first live execution)")
        p.add_argument("--allow-unlisted", action="store_true")
        if name == "run":
            p.add_argument("--retry-failed", action="store_true", help="explicitly re-attempt failed cases (recorded)")
            p.add_argument("--i-understand-this-spends-money", action="store_true")
        p.set_defaults(func=fn, confirm_held_out=False)

    p = bs.add_parser("final-eval", help="FINAL evaluation on HELD_OUT (live only; confirmed; permanently logged)")
    common(p)
    p.add_argument("--candidate", required=True)
    p.add_argument("--backend", default="openrouter")
    p.add_argument("--base-url", default=None)
    p.add_argument("--models-config", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--note", default="")
    p.add_argument("--confirm-held-out", action="store_true")
    p.add_argument("--allow-unlisted", action="store_true")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--i-understand-this-spends-money", action="store_true")
    p.set_defaults(func=cmd_bench_final_eval, subset=None)

    p = bs.add_parser("smoke", help="pre-registered DEV smoke subsets: propose | freeze | show")
    p.add_argument("smoke_command", choices=["propose", "freeze", "show"])
    common(p)
    from .smoke import DEFAULT_SMOKE_ROOT
    p.add_argument("--smoke-root", default=str(DEFAULT_SMOKE_ROOT))
    p.add_argument("--allow-unfilled", action="store_true",
                   help="freeze even though some slot could not be filled "
                        "(the gap is recorded in the frozen file)")
    p.set_defaults(func=cmd_bench_smoke)

    p = bs.add_parser("references", help="ocr_primary: the explicit 129-item reference provenance breakdown")
    common(p, needs_role=False)
    p.set_defaults(func=cmd_bench_references)

    for name, help_ in (("build-grading", "GRADE_PRIMARY inputs from audited cell transcriptions + the frozen key (no labels)"),
                        ("build-mc", "MC_RESOLVE from prob scans: deterministic band crops of ambiguous rows + audited answers"),
                        ("build-variant", "VARIANT_RESOLVE marker-region crops + audited variant ids"),
                        ("build-escalation", "GRADE_ESCALATE harvested from a grade_primary run (--from-run)"),
                        ("build-align", "ALIGN_RESOLVE (reports why it is not available yet)")):
        p = bs.add_parser(name, help=help_); common(p, needs_role=False)
        p.add_argument("--key-json", default=None, help="build-grading: frozen answer-key JSON (default: auto-detect)")
        p.add_argument("--from-run", default=None, help="build-escalation: grade_primary run directory")
        p.set_defaults(func=cmd_bench_build, build_command=name)

    p = bs.add_parser("repair-grading-evidence",
                      help="grade_primary: re-freeze the label-side evidence inventory from the upstream line "
                           "records (inputs untouched; revision recorded)")
    common(p, needs_role=False)
    p.add_argument("--dry-run", action="store_true", help="report what would change without writing")
    p.set_defaults(func=cmd_bench_repair_grading_evidence)

    p = bs.add_parser("owner-labels", help="grading roles: how many cases the owner still has to label")
    common(p)
    p.set_defaults(func=cmd_bench_owner_labels)

    p = bs.add_parser("evidence-repairs",
                      help="status + integrity of the manual GRADE_PRIMARY evidence repairs (human line "
                           "transcriptions stored outside the frozen OCR benchmark)")
    common(p)
    p.set_defaults(func=cmd_bench_evidence_repairs)

    p = bs.add_parser("apply-evidence-repairs",
                      help="fold completed human line repairs into the grading dataset (changes the model input; "
                           "records a manifest revision; never touches hebrew_bench_v2)")
    common(p)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-partial", action="store_true",
                   help="apply the repairs recorded so far instead of requiring all of them")
    p.set_defaults(func=cmd_bench_apply_evidence_repairs)

    p = bs.add_parser("missing-transcriptions",
                      help="grading roles: cases excluded from accuracy because a recorded line has no audited "
                           "transcription, with the exact missing line ids")
    common(p)
    p.set_defaults(func=cmd_bench_missing_transcriptions)

    p = bs.add_parser("import-final-labels", help="import the shared labeling app's FINAL export into a grading dataset")
    common(p)
    p.add_argument("--export", required=True, help="final_labels.json exported by `python -m labeling_app export`")
    p.set_defaults(func=cmd_bench_import_final_labels)

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
