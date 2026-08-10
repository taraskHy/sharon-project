"""gemini_protocol_clean_v1 — DETERMINISTIC protocol-artifact cleanup arm.

NOT a model. Re-extracts, from each frozen record's byte-for-byte `raw`
provider response, the transcription the provider declared it was
returning ({"transcription": ...} envelope, per the runner's prompt
contract), using the single shared parser
`m2_bench_run.parse_declared_envelope`. Allowed operations only:
code-fence stripping, envelope parsing (complete or truncated), quoting-
artifact removal, JSON string unescaping. NO spelling/vocabulary/grammar
changes, NO word completion, NO course material, NO references.

Fail-closed: a record whose raw carries no confidently-parseable declared
envelope keeps its existing extracted transcription unchanged.

Output: outputs/<config-id>/run1/<item>.json with the original raw
preserved byte-for-byte, the source transcription, the cleaned
transcription, and the exact ops applied.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "evaluation" / "hebrew_bench_v2"

spec = importlib.util.spec_from_file_location("mbr", REPO / "scripts" / "m2_bench_run.py")
mbr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mbr)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-config", default="gemini3_flash")
    ap.add_argument("--config-id", default="gemini_protocol_clean_v1")
    args = ap.parse_args()

    items = json.loads((BENCH / "items.json").read_text(encoding="utf-8"))["items"]
    src_run = BENCH / "outputs" / args.source_config / "run1"
    outdir = BENCH / "outputs" / args.config_id
    run_dir = outdir / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)

    parser_src = inspect.getsource(mbr.parse_declared_envelope)
    (outdir / "config.json").write_text(json.dumps({
        "config_id": args.config_id,
        "source_config": args.source_config,
        "deterministic": True,
        "parser": "m2_bench_run.parse_declared_envelope",
        "parser_sha256": hashlib.sha256(parser_src.encode()).hexdigest()[:16],
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=1), encoding="utf-8")

    n = changed = failed_closed = 0
    op_counts: dict[str, int] = {}
    for it in items:
        if it["category"] not in ("handwritten_line", "handwritten_cell"):
            continue
        p = src_run / f"{it['id']}.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("error") or not (rec.get("transcription") or "").strip():
            continue
        n += 1
        raw = rec.get("raw") or ""
        source_t = rec["transcription"]
        if raw:
            text, ops = mbr.parse_declared_envelope(raw)
            if text is not None and text.strip():
                cleaned = text
            else:
                cleaned = source_t
                ops = ops + ["fail_closed_keep_existing"]
                failed_closed += 1
        else:
            cleaned, ops = source_t, ["no_raw_fail_closed"]
            failed_closed += 1
        for op in ops:
            op_counts[op] = op_counts.get(op, 0) + 1
        if cleaned != source_t:
            changed += 1
        (run_dir / f"{it['id']}.json").write_text(json.dumps({
            "item": it["id"],
            "raw": raw,                       # byte-for-byte provider response
            "source_transcription": source_t,  # what was previously scored
            "transcription": cleaned,          # deterministic extraction
            "ops": ops,
            "changed": cleaned != source_t,
            "error": None,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{args.config_id}: {n} items, {changed} changed, {failed_closed} fail-closed")
    print("ops:", json.dumps(op_counts, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
