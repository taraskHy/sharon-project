"""`python -m labeling_app ...`

    build-bundle  [--dataset DIR] [--out DIR]      anonymized frozen bundle (offline, uses the repo dataset)
    serve         [--host 127.0.0.1] [--port 8787] [--data-dir DIR] [--bundle DIR] [--admin-key KEY]
    export        [--data-dir DIR] [--out FILE]    final_labels.json (FINAL labels only)
    backup        [--data-dir DIR] [--copy-to DIR]
    status        [--data-dir DIR]

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


def cmd_build_bundle(args) -> int:
    from .bundle import build_bundle
    out = Path(args.out) if args.out else default_data_dir() / "bundle"
    meta = build_bundle(Path(args.dataset), out, evaluation_root=REPO_ROOT / "evaluation")
    print(json.dumps({"bundle": str(out), **meta}, ensure_ascii=False, indent=1))
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


def _open(args):
    from .app import Bundle, LabelDB
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    bundle = Bundle(Path(args.bundle) if getattr(args, "bundle", None) else data_dir / "bundle")
    ds = _dataset_dir(args)
    rec = {"applied": False, "reason": "no dataset directory found"}
    if ds is not None:
        rec = bundle.apply_dataset_eligibility(ds)
    if getattr(args, "dataset", None) and not rec["applied"]:
        raise SystemExit(f"--dataset given but the eligibility recompute was not applied: {rec['reason']}")
    db = LabelDB(data_dir / "labels.db")
    db.load_items(bundle.items)
    # fail-safe: unknown eligibility never flips flags (a status/export run can
    # never erase the running server's enforcement)
    db.sync_eligibility([i["item_id"] for i in bundle.items], bundle.ineligible_item_ids(),
                        eligibility_known=bundle.eligibility_known())
    return data_dir, bundle, db


def cmd_export(args) -> int:
    from .export import write_export
    data_dir, bundle, db = _open(args)
    out = Path(args.out) if args.out else data_dir / "exports" / "final_labels.json"
    data = write_export(db, bundle, out)
    print(json.dumps({"written": str(out), "final_count": data["final_count"], "content_sha256": data["content_sha256"]}, indent=1))
    return 0


def cmd_backup(args) -> int:
    from .backup import make_backup
    data_dir, bundle, db = _open(args)
    out = make_backup(db, bundle, data_dir, copy_to=Path(args.copy_to) if args.copy_to else None)
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return 0


def cmd_status(args) -> int:
    data_dir, bundle, db = _open(args)
    print(json.dumps({"data_dir": str(data_dir), "db": str(db.path), "bundle_items": len(bundle.items),
                      "summary": db.summary()}, ensure_ascii=False, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="labeling_app", description="shared human grading-label tool (no AI)")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-bundle"); b.add_argument("--dataset", default=str(REPO_ROOT / "evaluation" / "model_selection" / "datasets" / "grade_primary")); b.add_argument("--out", default=None); b.set_defaults(func=cmd_build_bundle)
    s = sub.add_parser("serve"); s.add_argument("--host", default="127.0.0.1"); s.add_argument("--port", type=int, default=8787)
    s.add_argument("--data-dir", default=None); s.add_argument("--bundle", default=None); s.add_argument("--admin-key", default=os.environ.get("LABELING_ADMIN_KEY"))
    s.add_argument("--backup-copy-to", default=None); s.add_argument("--dataset", default=None,
                   help="dataset dir for eligibility recompute (default: the repo grade_primary)")
    s.set_defaults(func=cmd_serve)
    e = sub.add_parser("export"); e.add_argument("--data-dir", default=None); e.add_argument("--bundle", default=None); e.add_argument("--out", default=None); e.add_argument("--dataset", default=None); e.set_defaults(func=cmd_export)
    k = sub.add_parser("backup"); k.add_argument("--data-dir", default=None); k.add_argument("--bundle", default=None); k.add_argument("--copy-to", default=None); k.add_argument("--dataset", default=None); k.set_defaults(func=cmd_backup)
    t = sub.add_parser("status"); t.add_argument("--data-dir", default=None); t.add_argument("--bundle", default=None); t.add_argument("--dataset", default=None); t.set_defaults(func=cmd_status)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
