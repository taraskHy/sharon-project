"""Score a PyLaia decode file against the pilot annotations (val by
default). Line-level and cell-level CER/WER, usable-rate (cell CER <=
0.25, the campaign definition), and a confidence-abstention curve.

Runs under .venv (no torch). Reads the pilot ANNOTATIONS (the owner's
labels for these writers) — never the e002 benchmark GT and never
internal_test unless --split internal_test is passed together with
--allow-internal-test (single final report; see htr_pilot_gates.md).

Decode-file format (pylaia-htr-decode-ctc output, confidence enabled):
    <sample_id> <conf> <text...>     or      <sample_id> <text...>

    .venv/Scripts/python.exe scripts/htr_pilot_eval.py decodes/val_trial01.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.hebrew_bench_eval import lev, normalize, word_align
from scripts.htr_annotation_lib import (
    UNREADABLE_TOKEN, load_all_annotations, load_samples,
)

USABLE_CER = 0.25
TAUS = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]


def parse_decode(path: Path) -> dict[str, tuple[float | None, str]]:
    out: dict[str, tuple[float | None, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        sid, rest = parts[0], (parts[1] if len(parts) > 1 else "")
        conf = None
        toks = rest.split(maxsplit=1)
        if toks and toks[0].replace(".", "", 1).isdigit():
            conf = float(toks[0])
            rest = toks[1] if len(toks) > 1 else ""
        if "__aug" in sid:
            continue
        out[sid] = (conf, rest.strip())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("decode_file")
    ap.add_argument("--root", default="evaluation/htr_pilot")
    ap.add_argument("--split", default="val")
    ap.add_argument("--allow-internal-test", action="store_true")
    args = ap.parse_args()
    if args.split == "internal_test" and not args.allow_internal_test:
        print("REFUSING: internal_test eval without --allow-internal-test")
        return 2

    root = Path(args.root)
    samples = load_samples(root, args.split)
    ann = load_all_annotations(root, args.split)
    hyps = parse_decode(Path(args.decode_file))

    line_cers, cells = [], defaultdict(list)
    n_gt_lines = 0
    for s in samples:
        rec = ann.get(s["sample_id"])
        if rec is None or rec["status"] != "ok" or \
                UNREADABLE_TOKEN in rec["transcription"]:
            continue  # only owner-verified clean lines are scoreable
        n_gt_lines += 1
        conf, hyp_raw = hyps.get(s["sample_id"], (None, ""))
        hyp, ref = normalize(hyp_raw), normalize(rec["transcription"])
        line_cers.append(lev(hyp, ref) / max(len(ref), 1))
        key = (s["writer"], s["question"], s["row"])
        cells[key].append((s["line_index"], hyp, ref, conf))

    if not line_cers:
        print("no scoreable lines (annotate this split first, or decode file empty)")
        return 3

    cell_rows = []
    for key, items in cells.items():
        items.sort()
        hyp = " ".join(h for _i, h, _r, _c in items).strip()
        ref = " ".join(r for _i, _h, r, _c in items).strip()
        confs = [c for _i, _h, _r, c in items if c is not None]
        cer = lev(hyp, ref) / max(len(ref), 1)
        gw, hw = ref.split(), hyp.split()
        subs, dels, ins = word_align(gw, hw)
        cell_rows.append({
            "cell": f"{key[0]}_q{key[1]}_r{key[2]}", "cer": cer,
            "wer": (subs + dels + ins) / max(len(gw), 1),
            "conf": min(confs) if confs else None,
        })

    mean_line_cer = sum(line_cers) / len(line_cers)
    mean_cell_cer = sum(r["cer"] for r in cell_rows) / len(cell_rows)
    usable = sum(r["cer"] <= USABLE_CER for r in cell_rows)
    report = {
        "decode_file": args.decode_file, "split": args.split,
        "scoreable_lines": n_gt_lines, "cells": len(cell_rows),
        "mean_line_cer": round(mean_line_cer, 4),
        "mean_cell_cer": round(mean_cell_cer, 4),
        "mean_cell_wer": round(sum(r["wer"] for r in cell_rows) / len(cell_rows), 4),
        "usable_cells": usable,
        "usable_rate": round(usable / len(cell_rows), 4),
    }
    if any(r["conf"] is not None for r in cell_rows):
        curve = []
        for tau in TAUS:
            acc = [r for r in cell_rows if (r["conf"] or 0) >= tau]
            if acc:
                curve.append({
                    "tau": tau, "coverage": round(len(acc) / len(cell_rows), 3),
                    "cer_on_accepted": round(sum(r["cer"] for r in acc) / len(acc), 4),
                    "usable_on_accepted_rate": round(
                        sum(r["cer"] <= USABLE_CER for r in acc) / len(acc), 4),
                })
            else:
                curve.append({"tau": tau, "coverage": 0.0})
        report["confidence_abstention_curve"] = curve
    print(json.dumps(report, indent=1))
    worst = sorted(cell_rows, key=lambda r: -r["cer"])[:5]
    print("worst cells:", ", ".join(f"{r['cell']}({r['cer']:.2f})" for r in worst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
