"""Evaluate Mission-2 benchmark outputs against hebrew_bench_v2 references.

The ONLY component that reads references.json, strictly post-inference.
Reuses the July campaign's normalization/CER/WER definitions (imported from
hebrew_bench_eval.py) so numbers are comparable across campaigns.

Per config: overall + per-category mean CER, WER, usable rate (CER<=0.25),
omission/hallucination rates, RTL-reversal flags (a hypothesis whose
REVERSED text matches the reference far better than the forward text),
hard-cell honesty, association pair accuracy, latency stats. Appends one
row per (config, category) to evaluation/m2_bench_results.csv.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "evaluation" / "hebrew_bench_v2"
CSV_PATH = REPO / "evaluation" / "m2_bench_results.csv"

spec = importlib.util.spec_from_file_location(
    "hb_eval", REPO / "scripts" / "hebrew_bench_eval.py"
)
hb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hb)
normalize, lev, word_align = hb.normalize, hb.lev, hb.word_align

UNREADABLE_MARKERS = ["[unreadable]", "[?]", "לא קריא", "unreadable"]


def parse_pairs(text: str) -> dict[str, str]:
    """Extract option-letter -> value pairs from a hypothesis string."""
    pairs = {}
    for m in re.finditer(r"([אבגד])['\"]?\s*[:=\-)]*\s*([0-9]+(?:[./][0-9]+)*)", text or ""):
        pairs.setdefault(m.group(1), m.group(2))
    for m in re.finditer(r"([0-9]+(?:[./][0-9]+)*)\s*[:=\-(]*\s*['\"]?([אבגד])", text or ""):
        pairs.setdefault(m.group(2), m.group(1))
    return pairs


def main() -> int:
    config_id = sys.argv[1]
    refs = json.loads((BENCH / "references.json").read_text(encoding="utf-8"))
    items = {
        i["id"]: i
        for i in json.loads((BENCH / "items.json").read_text(encoding="utf-8"))["items"]
    }
    outdir = BENCH / "outputs" / config_id
    cfg = json.loads((outdir / "config.json").read_text(encoding="utf-8"))
    rundirs = sorted(d for d in outdir.iterdir() if d.is_dir() and d.name.startswith("run"))

    per_cat = defaultdict(lambda: {
        "cer": [], "usable": 0, "n": 0, "gtw": 0, "dels": 0, "hypw": 0,
        "ins": 0, "werr": 0, "rev_flags": 0, "hard_ok": 0, "hard_halluc": 0,
        "hard_n": 0, "assoc_ok": 0, "assoc_total": 0, "lat": [], "errors": 0,
    })
    details = []
    for rundir in rundirs:
        for f in sorted(rundir.glob("*.json")):
            rec = json.loads(f.read_text(encoding="utf-8"))
            iid = rec["item"]
            item = items.get(iid)
            ref = refs.get(iid)
            if item is None or ref is None:
                continue
            cat = item["category"]
            S = per_cat[cat]
            S["lat"].append(rec.get("latency_s") or 0)
            if rec.get("error"):
                S["errors"] += 1
            hyp_raw = rec.get("transcription") or ""

            if cat == "option_row_association":
                gt_pairs = ref.get("pairs") or {}
                hyp_pairs = parse_pairs(hyp_raw)
                ok = sum(1 for k, v in gt_pairs.items() if hyp_pairs.get(k) == v)
                S["assoc_ok"] += ok
                S["assoc_total"] += len(gt_pairs)
                details.append(f"{rundir.name} {iid} pairs {ok}/{len(gt_pairs)} hyp={hyp_raw[:60]!r}")
                continue

            hyp, gt_text = normalize(hyp_raw), normalize(ref["text"])
            flagged = any(m in hyp_raw.lower() for m in UNREADABLE_MARKERS)
            cer = lev(hyp, gt_text) / max(len(gt_text), 1)
            if item.get("hard"):
                S["hard_n"] += 1
                if flagged or cer <= 0.35:
                    S["hard_ok"] += 1
                elif hyp:
                    S["hard_halluc"] += 1
                details.append(f"{rundir.name} {iid} [hard] flagged={flagged} cer={cer:.2f}")
                continue
            rev_cer = lev(hyp[::-1], gt_text) / max(len(gt_text), 1)
            reversed_flag = hyp and rev_cer + 0.2 < cer
            S["cer"].append(cer)
            S["n"] += 1
            S["usable"] += cer <= 0.25
            S["rev_flags"] += bool(reversed_flag)
            gw, hw = gt_text.split(), hyp.split()
            subs, dels, ins = word_align(gw, hw)
            S["gtw"] += len(gw)
            S["dels"] += dels
            S["hypw"] += len(hw)
            S["ins"] += ins
            S["werr"] += subs + dels + ins
            details.append(
                f"{rundir.name} {iid} cer={cer:.2f}"
                + (" REVERSED" if reversed_flag else "")
                + f" hyp={hyp_raw[:60]!r}"
            )

    rows = []
    for cat, S in sorted(per_cat.items()):
        lat = sorted(S["lat"])
        row = {
            "config_id": config_id, "backend": cfg.get("backend"),
            "model": cfg.get("model"), "preproc": cfg.get("preproc"),
            "category": cat, "items": S["n"] or (S["assoc_total"] and len(lat)) or S["hard_n"],
            "mean_cer": round(sum(S["cer"]) / len(S["cer"]), 4) if S["cer"] else None,
            "usable_rate": round(S["usable"] / S["n"], 4) if S["n"] else None,
            "wer": round(S["werr"] / S["gtw"], 4) if S["gtw"] else None,
            "omission_rate": round(S["dels"] / S["gtw"], 4) if S["gtw"] else None,
            "halluc_rate": round(S["ins"] / S["hypw"], 4) if S["hypw"] else None,
            "reversed_flags": S["rev_flags"],
            "hard_flag_ok": S["hard_ok"] or None,
            "hard_halluc": S["hard_halluc"] or None,
            "hard_n": S["hard_n"] or None,
            "assoc_pair_acc": round(S["assoc_ok"] / S["assoc_total"], 4) if S["assoc_total"] else None,
            "errors": S["errors"],
            "median_latency_s": round(lat[len(lat) // 2], 2) if lat else None,
        }
        rows.append(row)

    exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if not exists:
            w.writeheader()
        w.writerows(rows)
    (outdir / "eval_detail.txt").write_text("\n".join(details), encoding="utf-8")
    for row in rows:
        print(json.dumps(row, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
