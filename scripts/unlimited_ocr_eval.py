"""Post-inference evaluator for the Unlimited-OCR arm (Phases 7/8/10).

STRICTLY post-hoc: reads references.json only AFTER raw predictions are
persisted by scripts/unlimited_ocr_run.py. Reuses the campaign's
normalization/CER/WER definitions (hebrew_bench_eval.py) so numbers are
comparable with every other arm.

Per item: CER, WER, omission, empty/error flags. Aggregates: mean/median
CER and WER, usable at CER<=0.25 and <=0.50, p25/p75, min/max, subgroup
splits (strict/hard, per-writer, category). Paired same-sample
comparisons against existing arms' persisted run1 outputs (never re-runs
them). Writes evaluation/unlimited_ocr/<label>_eval.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "evaluation" / "hebrew_bench_v2"
OUT = REPO / "evaluation" / "unlimited_ocr"

spec = importlib.util.spec_from_file_location(
    "hb_eval", REPO / "scripts" / "hebrew_bench_eval.py")
hb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hb)
normalize, lev, word_align = hb.normalize, hb.lev, hb.word_align


def item_metrics(hyp_raw: str, ref_text: str) -> dict:
    hyp, gt = normalize(hyp_raw or ""), normalize(ref_text)
    cer = lev(hyp, gt) / max(len(gt), 1)
    gw, hw = gt.split(), hyp.split()
    subs, dels, ins = word_align(gw, hw)
    return {
        "cer": round(cer, 4),
        "wer": round((subs + dels + ins) / max(len(gw), 1), 4),
        "omission": round(dels / max(len(gw), 1), 4),
        "hyp_words": len(hw),
        "empty": not hyp.strip(),
    }


def load_pred(config: str, iid: str) -> dict | None:
    f = BENCH / "outputs" / config / "run1" / f"{iid}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def agg(cers: list[float]) -> dict:
    if not cers:
        return {"n": 0}
    s = sorted(cers)
    return {
        "n": len(s),
        "mean": round(statistics.mean(s), 4),
        "median": round(statistics.median(s), 4),
        "p25": round(s[max(0, len(s) // 4 - (len(s) % 4 == 0))], 4) if len(s) >= 4 else None,
        "p75": round(s[min(len(s) - 1, (3 * len(s)) // 4)], 4) if len(s) >= 4 else None,
        "min": round(s[0], 4),
        "max": round(s[-1], 4),
        "usable_le_025": sum(c <= 0.25 for c in s),
        "usable_le_050": sum(c <= 0.50 for c in s),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-id", default="unlimited_ocr_gundam_eager")
    ap.add_argument("--items", required=True, help="comma list, the exact frozen set")
    ap.add_argument("--label", required=True, help="artifact label, e.g. smoke5")
    ap.add_argument("--compare", default="qwen8b_strict_contrast,mlkit_ink_rtl_a1,gemini3_flash")
    args = ap.parse_args()

    ids = [i for i in args.items.split(",") if i]
    refs = json.loads((BENCH / "references.json").read_text(encoding="utf-8"))
    items_meta = {
        i["id"]: i
        for i in json.loads((BENCH / "items.json").read_text(encoding="utf-8"))["items"]
    }

    per_item, errors = [], 0
    for iid in ids:
        rec = load_pred(args.config_id, iid)
        meta = items_meta[iid]
        ref = refs[iid]
        row = {"item": iid, "writer": iid.split("_")[1],
               "category": meta["category"], "hard": bool(meta.get("hard"))}
        if rec is None:
            row.update({"error": "MISSING PREDICTION FILE"})
            errors += 1
        else:
            row.update({
                "raw": rec.get("raw"),
                "transcription": rec.get("transcription"),
                "latency_s": rec.get("latency_s"),
                "error": rec.get("error"),
            })
            if rec.get("error"):
                errors += 1
            row.update(item_metrics(rec.get("transcription") or "", ref["text"]))
        per_item.append(row)

    scored = [r for r in per_item if "cer" in r]
    strict = [r for r in scored if not r["hard"]]
    hard = [r for r in scored if r["hard"]]
    by_writer = {}
    for r in scored:
        by_writer.setdefault(r["writer"], []).append(r["cer"])
    result = {
        "config_id": args.config_id,
        "label": args.label,
        "items_requested": ids,
        "errors": errors,
        "all": agg([r["cer"] for r in scored]),
        "all_wer_mean": round(statistics.mean([r["wer"] for r in scored]), 4) if scored else None,
        "all_wer_median": round(statistics.median([r["wer"] for r in scored]), 4) if scored else None,
        "omission_mean": round(statistics.mean([r["omission"] for r in scored]), 4) if scored else None,
        "empty_outputs": sum(r.get("empty", False) for r in scored),
        "strict": agg([r["cer"] for r in strict]),
        "hard": agg([r["cer"] for r in hard]),
        "by_writer": {w: agg(c) for w, c in sorted(by_writer.items())},
        "by_category": {},
        "latency": agg([r["latency_s"] for r in scored if r.get("latency_s") is not None]),
        "per_item": per_item,
    }
    cats = {}
    for r in scored:
        cats.setdefault(r["category"], []).append(r["cer"])
    result["by_category"] = {c: agg(v) for c, v in sorted(cats.items())}

    comparisons = {}
    for other in [c for c in args.compare.split(",") if c]:
        pairs = []
        for r in per_item:
            if "cer" not in r:
                continue
            orec = load_pred(other, r["item"])
            if orec is None or orec.get("error"):
                continue
            om = item_metrics(orec.get("transcription") or "", refs[r["item"]]["text"])
            pairs.append({"item": r["item"], "ours": r["cer"], "theirs": om["cer"]})
        if pairs:
            ours = [p["ours"] for p in pairs]
            theirs = [p["theirs"] for p in pairs]
            comparisons[other] = {
                "paired_n": len(pairs),
                "ours_mean_cer": round(statistics.mean(ours), 4),
                "theirs_mean_cer": round(statistics.mean(theirs), 4),
                "ours_median_cer": round(statistics.median(ours), 4),
                "theirs_median_cer": round(statistics.median(theirs), 4),
                "wins": sum(p["ours"] < p["theirs"] for p in pairs),
                "ties": sum(p["ours"] == p["theirs"] for p in pairs),
                "losses": sum(p["ours"] > p["theirs"] for p in pairs),
                "pairs": pairs,
            }
        else:
            comparisons[other] = {"paired_n": 0}
    result["comparisons"] = comparisons

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / f"{args.label}_eval.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    slim = {k: v for k, v in result.items() if k not in ("per_item",)}
    slim["comparisons"] = {k: {kk: vv for kk, vv in v.items() if kk != "pairs"}
                           for k, v in comparisons.items()}
    print(json.dumps(slim, ensure_ascii=False, indent=1))
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
