"""Render the interactive flow map from flow-spec.json + scan.json.

Validates the curated architecture spec against the static scan (so drift is
loud, not silent), merges module drill-down detail, and injects everything
into the bundled HTML template.

Outputs two files:
  flowmap.html        - artifact-ready fragment (no <html>/<head>/<body>;
                        the Claude Artifact publisher adds the skeleton)
  flowmap.local.html  - standalone page for double-click viewing in a browser

Usage:
    python render_flowmap.py [--spec docs/flow-spec.json]
                             [--scan flow-out/scan.json]
                             [--out-dir flow-out] [--root .]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent.parent / "assets" / "template.html"

VALID_COMPONENT_KINDS = {"ui", "stage", "service", "store", "external", "human"}
VALID_FLOW_KINDS = {"call", "data", "subprocess", "di", "http", "human"}


def fail(msg: str) -> "SystemExit":
    print(f"error: {msg}", file=sys.stderr)
    return SystemExit(2)


def validate(spec: dict, scan: dict) -> list[str]:
    warnings: list[str] = []
    comp_ids = set()
    group_ids = {g.get("id") for g in spec.get("groups", [])}
    scan_mods = set(scan.get("modules", {}))

    for c in spec.get("components", []):
        cid = c.get("id")
        if not cid or not c.get("name"):
            raise fail(f"component missing id/name: {c}")
        if cid in comp_ids:
            raise fail(f"duplicate component id: {cid}")
        comp_ids.add(cid)
        if c.get("kind") and c["kind"] not in VALID_COMPONENT_KINDS:
            warnings.append(f"component {cid}: unknown kind '{c['kind']}'")
        if group_ids and c.get("group") not in group_ids:
            warnings.append(f"component {cid}: group '{c.get('group')}' not in groups")
        for m in c.get("modules", []):
            if scan_mods and m not in scan_mods:
                warnings.append(f"component {cid}: module '{m}' not found by scan (stale spec?)")

    for f in spec.get("flows", []):
        if f.get("from") not in comp_ids or f.get("to") not in comp_ids:
            raise fail(f"flow references unknown component: {f.get('from')} -> {f.get('to')}")
        if f.get("kind") and f["kind"] not in VALID_FLOW_KINDS:
            warnings.append(f"flow {f['from']}->{f['to']}: unknown kind '{f['kind']}'")
        if not f.get("evidence"):
            warnings.append(f"flow {f['from']}->{f['to']}: no evidence (file:line) recorded")

    # completeness: scanned modules no component claims — candidates for the spec
    claimed = {m for c in spec.get("components", []) for m in c.get("modules", [])}
    orphans = sorted(m for m in scan_mods - claimed if not m.endswith("__init__"))
    if orphans:
        warnings.append(
            f"{len(orphans)} scanned modules not covered by any component "
            f"(fine if intentional): {', '.join(orphans[:12])}"
            + (" ..." if len(orphans) > 12 else "")
        )
    return warnings


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default="docs/flow-spec.json")
    ap.add_argument("--scan", default="flow-out/scan.json")
    ap.add_argument("--out-dir", default="flow-out")
    ap.add_argument("--root", default=".")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    spec_path = root / args.spec
    scan_path = root / args.scan
    if not spec_path.exists():
        raise fail(f"spec not found: {spec_path} (author it per references/flow-spec.md)")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    scan = json.loads(scan_path.read_text(encoding="utf-8")) if scan_path.exists() else {}
    if not scan:
        print(f"note: no scan at {scan_path}; drill-down detail will be empty", file=sys.stderr)

    for w in validate(spec, scan):
        print(f"warn: {w}", file=sys.stderr)

    spec.setdefault("generated", date.today().isoformat())

    # drill-down detail only for modules the spec references
    referenced = {m for c in spec.get("components", []) for m in c.get("modules", [])}
    module_details = {}
    for m in referenced:
        info = scan.get("modules", {}).get(m)
        if info:
            module_details[m] = {
                "path": info["path"],
                "loc": info["loc"],
                "defs": info["defs"][:10],
                "entry_hints": info.get("entry_hints", []),
            }

    payload = json.dumps(
        {"spec": spec, "moduleDetails": module_details},
        ensure_ascii=False, separators=(",", ":"),
    ).replace("</", "<\\/")

    html = TEMPLATE.read_text(encoding="utf-8")
    html = html.replace("__FLOWMAP_TITLE__", spec.get("title", "Flow map"))
    html = html.replace("__FLOWMAP_DATA_JSON__", payload)

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    frag = out_dir / "flowmap.html"
    frag.write_text(html, encoding="utf-8")
    local = out_dir / "flowmap.local.html"
    local.write_text(
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "</head>\n<body>\n" + html + "\n</body>\n</html>\n",
        encoding="utf-8",
    )
    n_c = len(spec.get("components", []))
    n_f = len(spec.get("flows", []))
    print(f"rendered {n_c} components, {n_f} flows")
    print(f"  artifact fragment: {frag}")
    print(f"  local page:        {local}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
