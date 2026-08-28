"""review46_app CLI.

    build   [--out DIR] [--replace]          the 46-case blind review bundle
    serve   [--host 127.0.0.1] [--port 8790] [--data-dir DIR] [--bundle DIR]
            [--admin-key KEY] [--invite-token TOKEN]
    status  [--data-dir DIR]
    export  [--data-dir DIR]
    backup  [--data-dir DIR] [--copy-to DIR]

Data directory (live state; keep OUT of OneDrive/Git):
    %LOCALAPPDATA%\\autograder\\review46\\   (override: REVIEW46_DATA_DIR or --data-dir)
        labels.db        live SQLite (WAL)     admin_key.txt   generated once
        bundle/          blind review bundle   exports/  backups/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import PORT_DEFAULT, default_data_dir


def cmd_build(args) -> int:
    from .build import build_review_bundle
    out = Path(args.out) if args.out else default_data_dir() / "bundle"
    try:
        doc = build_review_bundle(out, replace=bool(args.replace))
    except FileExistsError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(json.dumps(doc, indent=1))
    return 0


def cmd_serve(args) -> int:
    import uvicorn
    from .app import create_app
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    app = create_app(data_dir=data_dir,
                     bundle_dir=Path(args.bundle) if args.bundle else None,
                     admin_key=args.admin_key or None,
                     invite_token=args.invite_token or None)
    key_file = data_dir / "admin_key.txt"
    print(f"reviewer URL : http://{args.host}:{args.port}/")
    print(f"admin URL    : http://{args.host}:{args.port}/admin?key=<see {key_file}>")
    print(f"data dir     : {data_dir}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_status(args) -> int:
    from labeling_app.db import LabelDB
    from . import assert_not_live_review_db
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    p = data_dir / "labels.db"
    if not p.exists():
        print("no database yet")
        return 0
    assert_not_live_review_db(p)
    db = LabelDB(p)
    print(json.dumps(db.summary(), indent=1, default=str))
    db.close()
    return 0


def cmd_export(args) -> int:
    print("export runs through the live server: GET /api/admin/export "
          "(writes <data>/exports/campaign_results.json)")
    return 0


def cmd_backup(args) -> int:
    from labeling_app.backup import make_backup
    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    out = make_backup(data_dir / "labels.db", None, data_dir,
                      copy_to=Path(args.copy_to) if args.copy_to else None,
                      bundle_dir=data_dir / "bundle")
    print(json.dumps(out, indent=1, default=str))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="review46_app", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build")
    p.add_argument("--out", default=None)
    p.add_argument("--replace", action="store_true")
    p.set_defaults(fn=cmd_build)
    p = sub.add_parser("serve")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=PORT_DEFAULT)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--bundle", default=None)
    p.add_argument("--admin-key", default=None)
    p.add_argument("--invite-token", default=None)
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("status")
    p.add_argument("--data-dir", default=None)
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("export")
    p.add_argument("--data-dir", default=None)
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("backup")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--copy-to", default=None)
    p.set_defaults(fn=cmd_backup)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
