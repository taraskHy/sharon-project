"""`python -m labeling_app ...`

    build-bundle  [--dataset DIR] [--out DIR] [--replace]
                  anonymized frozen bundle (offline, uses the repo dataset). --replace rebuilds IN PLACE:
                  the old bundle's evidence fingerprints are registered in labels.db first (so every
                  existing label records what it was actually made against), the old bundle is moved
                  to bundle.previous-<stamp>/ (never deleted), opaque item ids stay stable, and the
                  labels whose evidence changed are reported as STALE (re-review), nothing else is touched.
    serve         [--host 127.0.0.1] [--port 8787] [--data-dir DIR] [--bundle DIR] [--admin-key KEY]
    export        [--data-dir DIR] [--out FILE]    final_labels.json (FINAL labels only)
    backup        [--data-dir DIR] [--copy-to DIR]
    status        [--data-dir DIR]
    evidence-report [--data-dir DIR] [--bundle DIR]   preserved / stale / unknown labels per case (exact)
    set-provenance --grader G --source S [--asserted-by WHO] [--items CASE...] [--dry-run]
                                                     record HOW existing scores were derived
                                                     (label_source/entered_by/source_ref); never edits a score

Data directory (the ONLY place live state lives; keep it OUT of OneDrive):
    %LOCALAPPDATA%\\autograder\\labeling\\   (override: LABELING_DATA_DIR or --data-dir)
        labels.db            live SQLite database (WAL: labels.db-wal / labels.db-shm while running)
        bundle/              anonymized item bundle
        exports/             final_labels.json written by export
        backups/<stamp>/     labels.db snapshot + final_labels.json + manifest
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def default_data_dir() -> Path:
    env = os.environ.get("LABELING_DATA_DIR")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / "autograder" / "labeling"


def _register_bundle_evidence(bundle_dir: Path, data_dir: Path) -> dict | None:
    """Register a bundle's evidence fingerprints in labels.db (what the labels
    on record were made against). Returns the sync summary, or None when
    there is no database yet (nothing to preserve)."""
    from .app import Bundle, LabelDB
    if not (data_dir / "labels.db").exists():
        return None
    b = Bundle(bundle_dir)
    db = LabelDB(data_dir / "labels.db")
    try:
        db.load_items(b.items)
        return db.sync_evidence(b.fingerprints)
    finally:
        db.close()


def _print_evidence_report(rep: dict, id_map: dict[str, str], *, file=None) -> None:
    """Human-readable stale/preserved accounting, joined with dataset case ids.
    ``file`` is resolved at CALL time (a def-time ``sys.stderr`` default binds
    whatever stream was current when the module was first imported)."""
    file = file or sys.stderr
    cid = lambda i: id_map.get(i, i)  # noqa: E731
    print(f"[evidence] labels preserved : {rep['labels_preserved']} (nothing deleted)", file=file)
    print(f"[evidence] labels fresh     : {rep['labels_fresh']}", file=file)
    print(f"[evidence] labels stale     : {rep['labels_stale']} (evidence changed after the label; re-review required)", file=file)
    print(f"[evidence] labels unknown   : {rep['labels_unknown_evidence']} (no fingerprint on record)", file=file)
    print(f"[evidence] FINALs stale     : {rep['finals_stale']} / {rep['finals_total']}", file=file)
    by_src = rep.get("labels_by_source") or {}
    if by_src:
        print("[evidence] labels by provenance: "
              + ", ".join(f"{k}={v}" for k, v in sorted(by_src.items()) if v), file=file)
    auth_rep = rep.get("authoritative_labels_on_repaired_evidence") or []
    if auth_rep:
        print(f"[evidence] authoritative labels on REPAIRED evidence: {len(auth_rep)} "
              f"(score came from the complete original grading -> still valid, NOT re-review work)", file=file)
        for r in auth_rep:
            print(f"[evidence]   VALID (repaired evidence)  case {cid(r['item_id'])}  grader {r['grader']}  "
                  f"source {r['label_source']}", file=file)
    changed = rep.get("items_evidence_changed") or []
    print(f"[evidence] items whose evidence changed: {len(changed)}"
          + (": " + ", ".join(sorted(cid(r['item_id']) for r in changed)) if changed else ""), file=file)
    for r in rep.get("stale_labels") or []:
        print(f"[evidence]   STALE label  case {cid(r['item_id'])}  grader {r['grader']}  status {r['status']}  "
              f"rev {r['revision']}  (evidence changed {r.get('evidence_changed_at')})", file=file)
    for r in rep.get("stale_finals") or []:
        print(f"[evidence]   STALE FINAL  case {cid(r['item_id'])}  source {r['source']}  finalized {r['finalized_at']}", file=file)


def cmd_build_bundle(args) -> int:
    from .bundle import Bundle, build_bundle, previous_bundle_info
    out = Path(args.out) if args.out else default_data_dir() / "bundle"
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    previous = previous_bundle_info(out)
    if previous is not None and not args.replace:
        print(f"{out} already holds a bundle; re-run with --replace to rebuild in place (old bundle kept aside, "
              "item ids stable, labels preserved) or pass --out for a fresh directory", file=sys.stderr)
        return 2
    before = None
    if previous is not None:
        # 1) register what the EXISTING labels were made against (the old bundle)
        before = _register_bundle_evidence(out, data_dir)
    meta = build_bundle(Path(args.dataset), out, evaluation_root=REPO_ROOT / "evaluation", replace=bool(args.replace))
    report = {"bundle": str(out), **meta}
    if previous is not None:
        # 2) register the NEW bundle: labels whose evidence changed become stale (nothing else moves)
        after = _register_bundle_evidence(out, data_dir)
        if after is not None:
            report["evidence_sync"] = {"registered_old_bundle": (before or {}).get("registered"),
                                       "changed_items": after["changed"], "report": after["report"]}
            _print_evidence_report(after["report"], Bundle(out).id_map)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


def _join_case_ids(rep: dict, id_map: dict[str, str]) -> dict:
    """Attach dataset case ids to a report WITHOUT hiding anything.

    An item the current bundle does not know keeps its item id in
    ``affected_case_ids`` and is named in ``items_not_in_current_bundle``,
    instead of silently becoming ``case_id: null`` and vanishing from the
    affected list — which understated the blast radius exactly when the bundle
    and the database had drifted apart, i.e. when it mattered most."""
    unmapped: set[str] = set()
    for key in ("stale_labels", "unknown_evidence_labels", "stale_finals", "items_evidence_changed",
                "authoritative_labels_on_repaired_evidence"):
        for row in rep.get(key, []):
            case = id_map.get(row["item_id"])
            row["case_id"] = case
            if case is None:
                unmapped.add(row["item_id"])
                row["case_id_unavailable"] = ("this item is not in the current bundle — the bundle was rebuilt "
                                              "with a different id salt, or is stale/missing")
    rep["affected_case_ids"] = sorted({r.get("case_id") or r["item_id"]
                                       for r in rep.get("items_evidence_changed", [])})
    rep["items_not_in_current_bundle"] = sorted(unmapped)
    return rep


def cmd_evidence_report(args) -> int:
    data_dir, bundle, db = _open(args, register=False)      # reporting never registers a bundle
    rep = _join_case_ids(db.evidence_report(), bundle.id_map)
    _print_evidence_report(rep, bundle.id_map)
    print(json.dumps({"data_dir": str(data_dir), "db": str(db.path), **rep}, ensure_ascii=False, indent=1))
    return 0


def cmd_verify_provenance(args) -> int:
    """READ-ONLY verification of the stored label provenance. Opens ONLY the
    database (no bundle, no dataset, no eligibility recompute) so it works while
    the server runs and whatever state the bundle is in."""
    from .db import LabelDB
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    db_path = data_dir / "labels.db"
    if not db_path.exists():
        print(f"no database at {db_path}", file=sys.stderr)
        return 2
    rep = LabelDB(db_path).verify_provenance()
    print(f"[provenance] labels: {rep['labels_total']} — by source: "
          + ", ".join(f"{k}={v}" for k, v in rep["labels_by_source"].items() if v), file=sys.stderr)
    print(f"[provenance] provenance events cross-checked: {rep['provenance_events_checked']}; "
          f"scores changed since provenance was recorded: {len(rep['scores_changed_since_provenance_recorded'])}",
          file=sys.stderr)
    for g, d in sorted(rep["per_grader"].items()):
        print(f"[provenance]   {g}: {d['labels']} label(s) {d['by_source']} entered_by={d['entered_by']} "
              f"asserted_by={d['asserted_by']} revisions={d['revisions']} statuses={d['statuses']}", file=sys.stderr)
    if rep["authoritative_labels_on_repaired_evidence"]:
        print(f"[provenance] authoritative labels on repaired evidence: "
              f"{len(rep['authoritative_labels_on_repaired_evidence'])} (valid; the repair stays recorded)",
              file=sys.stderr)
    print(f"[provenance] stale labels: {len(rep['stale_labels'])}", file=sys.stderr)
    print(json.dumps({"db": str(db_path), **rep}, ensure_ascii=False, indent=1))
    return 0 if rep["scores_unchanged"] else 1


def cmd_set_provenance(args) -> int:
    """Record how an existing grader's scores were DERIVED. Never edits a score."""
    data_dir, bundle, db = _open(args)
    refs: dict[str, str] = {}
    if not args.no_source_ref:
        # source_ref = the authoritative origin of the score: the original graded
        # exam file (kept private: its name carries the instructor total) + case id
        for iid, pv in (bundle.private_provenance or {}).items():
            src = pv.get("source_file")
            case = pv.get("case_id") or bundle.id_map.get(iid, iid)
            if src:
                refs[iid] = f"{src}#{case}"
    item_ids = None
    if args.items:
        by_case = {v: k for k, v in bundle.id_map.items()}
        item_ids = [by_case.get(x, x) for x in args.items]
    out = db.set_label_provenance(grader=args.grader, label_source=args.source,
                                  entered_by=args.entered_by or args.grader,
                                  asserted_by=args.asserted_by, source_refs=refs,
                                  item_ids=item_ids, actor=args.asserted_by or "system",
                                  dry_run=args.dry_run)
    cid = lambda i: bundle.id_map.get(i, i)  # noqa: E731
    for row in out["applied"]:
        row["case_id"] = cid(row["item_id"])
    for row in out["skipped"]:
        row["case_id"] = cid(row["item_id"])
    tag = "DRY RUN — nothing written" if args.dry_run else "applied"
    print(f"[provenance] {tag}: {out['applied_count']} label(s) of grader {args.grader!r} "
          f"-> {args.source}", file=sys.stderr)
    print(f"[provenance] asserted_by={out['asserted_by'] or '(unset)'} "
          f"entered_by={out['entered_by']} — an ASSERTION recorded as such, not machine-verified",
          file=sys.stderr)
    print(f"[provenance] scores modified: {out['scores_modified']} (this command never edits a score)",
          file=sys.stderr)
    if out["skipped"]:
        print(f"[provenance] skipped {out['skipped_count']}:", file=sys.stderr)
        for r in out["skipped"]:
            print(f"[provenance]   {r['case_id']}: {r['reason']}", file=sys.stderr)
    if not args.dry_run:
        out["evidence_report"] = db.evidence_report()
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


def _dataset_dir(args) -> Path | None:
    """The dataset directory used to (re)compute label eligibility. Explicit
    --dataset wins; otherwise the repo default when it exists."""
    if getattr(args, "dataset", None):
        return Path(args.dataset)
    default = REPO_ROOT / "evaluation" / "model_selection" / "datasets" / "grade_primary"
    return default if (default / "cases_inputs.jsonl").exists() else None


def cmd_serve(args) -> int:
    import uvicorn
    from .app import create_app
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    bundle = Path(args.bundle) if args.bundle else data_dir / "bundle"
    if not (bundle / "items.json").exists():
        print(f"no bundle at {bundle}; run: python -m labeling_app build-bundle", file=sys.stderr)
        return 2
    app = create_app(data_dir=data_dir, bundle_dir=bundle, admin_key=args.admin_key,
                     backup_copy_to=Path(args.backup_copy_to) if args.backup_copy_to else None,
                     dataset_dir=_dataset_dir(args))
    rec = app.state.eligibility_recompute
    if args.dataset and not rec["applied"]:
        # an EXPLICIT dataset that cannot be applied is an operator error, never a silent no-op
        print(f"[labeling] ERROR: --dataset given but the eligibility recompute was not applied: "
              f"{rec['reason']}", file=sys.stderr)
        return 2
    if not app.state.bundle.eligibility_known():
        print("[labeling] WARNING: eligibility is UNKNOWN for some items (old bundle, no usable dataset"
              + (f"; recompute skipped: {rec['reason']}" if not rec["applied"] else "")
              + ") — policy-decided items cannot be blocked; pass --dataset", file=sys.stderr)
    ineligible = app.state.bundle.ineligible_item_ids()
    sync = app.state.evidence_sync
    if sync.get("changed"):
        print(f"[labeling] evidence : {len(sync['changed'])} item(s) changed evidence since the labels on record — "
              "their labels are STALE and will be re-served to their graders", file=sys.stderr)
    _print_evidence_report(sync["report"], app.state.bundle.id_map)
    print(f"[labeling] data dir : {data_dir}", file=sys.stderr)
    print(f"[labeling] bundle   : {bundle} ({len(app.state.bundle.items)} items"
          + (f", {len(ineligible)} ineligible/policy-decided" if ineligible else "") + ")", file=sys.stderr)
    print(f"[labeling] grader   : http://{args.host}:{args.port}/", file=sys.stderr)
    print(f"[labeling] admin    : http://{args.host}:{args.port}/admin"
          + (f"?key={args.admin_key}" if args.admin_key else "  (no admin key set)"), file=sys.stderr)
    print("[labeling] tunnel   : cloudflared tunnel --url http://127.0.0.1:%d   (run separately when you want it public)"
          % args.port, file=sys.stderr)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def _open(args, *, register: bool = True):
    """Open (data_dir, bundle, db).

    ``register=False`` for REPORTING commands. Registering a bundle is a WRITE:
    ``load_items`` inserts every item id the bundle carries, and the sync calls
    rewrite eligibility and evidence fingerprints. A bundle whose id salt differs
    from the one the database was built against therefore inserts a whole second
    set of item rows — which is how 67 orphan items once appeared in the live
    database, and why a report could then show ``case_id: null`` for items the
    new bundle does not know. A command that only reads must only read."""
    from .app import Bundle, LabelDB
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    bundle = Bundle(Path(args.bundle) if getattr(args, "bundle", None) else data_dir / "bundle")
    bundle.verify_evidence()
    ds = _dataset_dir(args)
    rec = {"applied": False, "reason": "no dataset directory found"}
    if ds is not None:
        rec = bundle.apply_dataset_eligibility(ds)
    if getattr(args, "dataset", None) and not rec["applied"]:
        raise SystemExit(f"--dataset given but the eligibility recompute was not applied: {rec['reason']}")
    db = LabelDB(data_dir / "labels.db")
    if not register:
        return data_dir, bundle, db
    db.load_items(bundle.items)
    # fail-safe: unknown eligibility never flips flags (a status/export run can
    # never erase the running server's enforcement)
    db.sync_eligibility([i["item_id"] for i in bundle.items], bundle.ineligible_item_ids(),
                        eligibility_known=bundle.eligibility_known())
    db.sync_evidence(bundle.fingerprints)
    return data_dir, bundle, db


def cmd_export(args) -> int:
    """Write final_labels.json. Reads the database; never registers the bundle.

    Registering here once reproduced the orphan-item incident in full: a bundle
    whose id salt differed from the database's inserted 67 phantom items AND
    retired all 67 real ones to eligible=0. `export` is step two of the
    label-import sequence, so it is exactly the command that must not do that."""
    from .export import write_export
    data_dir, bundle, db = _open(args, register=False)
    out = Path(args.out) if args.out else data_dir / "exports" / "final_labels.json"
    data = write_export(db, bundle, out)
    print(json.dumps({"written": str(out), "final_count": data["final_count"], "content_sha256": data["content_sha256"]}, indent=1))
    return 0


def cmd_backup(args) -> int:
    """Snapshot labels.db FIRST (read-only online backup; works while the server
    runs) and export FINAL labels only if the bundle loads. A missing or corrupt
    bundle never prevents the database backup."""
    from .backup import make_backup
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    db_path = data_dir / "labels.db"
    if not db_path.exists():
        print(f"no database at {db_path} — nothing to back up", file=sys.stderr)
        return 2
    bundle_dir = Path(args.bundle) if getattr(args, "bundle", None) else data_dir / "bundle"
    out = make_backup(db_path, None, data_dir, bundle_dir=bundle_dir,
                      copy_to=Path(args.copy_to) if args.copy_to else None)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if out["export"].get("status") != "written":
        print(f"[labeling] note: final_labels.json export skipped — {out['export'].get('reason')}; "
              "the database snapshot succeeded", file=sys.stderr)
    return 0


def cmd_status(args) -> int:
    data_dir, bundle, db = _open(args, register=False)      # reporting never registers a bundle
    rep = _join_case_ids(db.evidence_report(), bundle.id_map)
    print(json.dumps({"data_dir": str(data_dir), "db": str(db.path), "bundle_items": len(bundle.items),
                      "summary": db.summary(), "evidence": rep}, ensure_ascii=False, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    from .db import LABEL_SOURCES
    p = argparse.ArgumentParser(prog="labeling_app", description="shared human grading-label tool (no AI)")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-bundle"); b.add_argument("--dataset", default=str(REPO_ROOT / "evaluation" / "model_selection" / "datasets" / "grade_primary")); b.add_argument("--out", default=None)
    b.add_argument("--data-dir", default=None, help="labels.db location (default data dir); used by --replace to register fingerprints")
    b.add_argument("--replace", action="store_true", help="rebuild in place: old bundle kept aside, item ids stable, stale labels reported")
    b.set_defaults(func=cmd_build_bundle)
    s = sub.add_parser("serve"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8787)
    s.add_argument("--data-dir", default=None); s.add_argument("--bundle", default=None); s.add_argument("--admin-key", default=os.environ.get("LABELING_ADMIN_KEY"))
    s.add_argument("--backup-copy-to", default=None); s.add_argument("--dataset", default=None,
                   help="dataset dir for eligibility recompute (default: the repo grade_primary)")
    s.set_defaults(func=cmd_serve)
    e = sub.add_parser("export"); e.add_argument("--data-dir", default=None); e.add_argument("--bundle", default=None); e.add_argument("--out", default=None); e.add_argument("--dataset", default=None); e.set_defaults(func=cmd_export)
    k = sub.add_parser("backup"); k.add_argument("--data-dir", default=None); k.add_argument("--bundle", default=None); k.add_argument("--copy-to", default=None); k.add_argument("--dataset", default=None); k.set_defaults(func=cmd_backup)
    t = sub.add_parser("status"); t.add_argument("--data-dir", default=None); t.add_argument("--bundle", default=None); t.add_argument("--dataset", default=None); t.set_defaults(func=cmd_status)
    v = sub.add_parser("evidence-report"); v.add_argument("--data-dir", default=None); v.add_argument("--bundle", default=None); v.add_argument("--dataset", default=None); v.set_defaults(func=cmd_evidence_report)
    vp = sub.add_parser("verify-provenance", help="READ-ONLY: what provenance the stored labels carry, and proof "
                                                  "from the audit trail that recording it changed no score")
    vp.add_argument("--data-dir", default=None)
    vp.set_defaults(func=cmd_verify_provenance)
    g = sub.add_parser("set-provenance", help="record HOW an existing grader's scores were derived (never edits a score)")
    g.add_argument("--grader", required=True)
    g.add_argument("--source", required=True, choices=list(LABEL_SOURCES))
    g.add_argument("--entered-by", default=None, help="who keyed the score in (default: the grader)")
    g.add_argument("--asserted-by", default="", help="who ASSERTS this provenance (e.g. owner) — stored, not verified")
    g.add_argument("--items", nargs="*", default=None, help="case ids to limit to (default: all of the grader's labels)")
    g.add_argument("--no-source-ref", action="store_true", help="do not record the private source file reference")
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--data-dir", default=None); g.add_argument("--bundle", default=None); g.add_argument("--dataset", default=None)
    g.set_defaults(func=cmd_set_provenance)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
