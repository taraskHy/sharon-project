"""Signal-2 post-hoc analysis: join frozen verifier outputs with the
preserved/silent labels. Written and committed BEFORE any verifier output
existed — thresholds T1-T4 are fixed a priori in m2_verify_run.py.

Cell aggregation (a priori): verdict = "review" if ANY constituent item
says review (or has a runtime error); confidence = min over items
(high > medium > low); issue lists are unions.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "evaluation" / "hebrew_bench_v2"
CONF_ORDER = {"high": 2, "medium": 1, "low": 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-config", default="gemini3_flash_verify",
                    help="verifier arm to analyze (mechanical only — "
                         "thresholds/aggregation are frozen)")
    args = ap.parse_args()
    VDIR = BENCH / "outputs" / args.verify_config / "run1"
    labels = {}
    for l in (REPO / "evaluation" / "m2_grading" / "gemini3_flash.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        labels[r["cell"]] = ("preserved" if r["agree"]
                             else "abstain" if r["verdict_ocr"] == "unintelligible"
                             else "silent")
    items = json.loads((BENCH / "items.json").read_text(encoding="utf-8"))["items"]
    cellmap = defaultdict(list)
    for it in items:
        iid = it["id"]
        if iid.startswith("hl_"):
            cellmap[iid.split("__")[0].replace("hl_", "")].append(iid)
        elif iid.startswith("hc_"):
            cellmap[iid.replace("hc_", "")].append(iid)

    cells = {}
    missing_items = []
    for cell in labels:
        agg = {"verdict": "supported", "confidence": "high",
               "omissions": [], "substitutions": [], "additions": [],
               "uncertain_regions": [], "items": []}
        complete = True
        for iid in sorted(cellmap[cell]):
            p = VDIR / f"{iid}.json"
            if not p.exists():
                complete = False
                missing_items.append(iid)
                continue
            rec = json.loads(p.read_text(encoding="utf-8"))
            vj = rec.get("verdict_json") or {}
            agg["items"].append(iid)
            if rec.get("error") or vj.get("verdict") == "review":
                agg["verdict"] = "review"
            c = vj.get("confidence", "low")
            if CONF_ORDER.get(c, 0) < CONF_ORDER.get(agg["confidence"], 2):
                agg["confidence"] = c
            for k in ("omissions", "substitutions", "additions", "uncertain_regions"):
                agg[k].extend(vj.get(k) or [])
        if complete:
            cells[cell] = agg
    print(f"cells with complete verifier coverage: {len(cells)}/{len(labels)}"
          + (f" | missing items: {missing_items}" if missing_items else ""))

    def flagged(agg, T):
        if T == "T1":
            return agg["verdict"] == "review"
        if T == "T2":
            return agg["verdict"] == "review" or agg["confidence"] == "low"
        if T == "T3":
            return agg["verdict"] == "review" or agg["confidence"] in ("low", "medium")
        if T == "T4":
            return bool(agg["omissions"] or agg["substitutions"] or agg["additions"])
        raise ValueError(T)

    silent_total = sum(1 for c in cells if labels[c] == "silent")
    pres_total = sum(1 for c in cells if labels[c] == "preserved")
    print(f"labels among covered: silent {silent_total}, preserved {pres_total}, "
          f"abstain {sum(1 for c in cells if labels[c] == 'abstain')}")
    for T in ("T1", "T2", "T3", "T4"):
        review = {c for c in cells if flagged(cells[c], T)}
        auto = set(cells) - review
        caught = sum(1 for c in review if labels[c] == "silent")
        missed = sum(1 for c in auto if labels[c] == "silent")
        flag_pres = sum(1 for c in review if labels[c] == "preserved")
        print(f"{T}: caught-silent {caught}/{silent_total} | silent-auto-pass {missed} | "
              f"flagged-preserved {flag_pres}/{pres_total} | review {len(review)}/{len(cells)} | "
              f"auto {len(auto)} | silent-among-auto {missed}/{len(auto) or 1}")

    print("\nknown upward-credit fidelity failures:")
    for cell in ("e003_q1_r3", "e006_q1_r2"):
        if cell in cells:
            a = cells[cell]
            print(f"  {cell}: verdict={a['verdict']} conf={a['confidence']} "
                  f"omissions={a['omissions']} substitutions={a['substitutions']}")
        else:
            print(f"  {cell}: NOT covered")

    print("\nper-cell detail:")
    for c in sorted(cells):
        a = cells[c]
        print(f"  {c:<14} {labels[c]:<10} {a['verdict']:<10} conf={a['confidence']} "
              f"om={len(a['omissions'])} sub={len(a['substitutions'])} add={len(a['additions'])}")
    suffix = ("" if args.verify_config == "gemini3_flash_verify"
              else f"_{args.verify_config}")
    out_name = f"signal2_verifier{suffix}.json"
    (REPO / "evaluation" / "m2_grading" / out_name).write_text(
        json.dumps({"cells": cells, "labels": {c: labels[c] for c in cells}},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nwrote {out_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
