"""Extended post-inference analysis for a Mission-2 benchmark arm.

Reads persisted predictions + references (strictly post-inference) and
produces per-writer tables, CER percentiles, best/worst lists, latency
distribution and a machine-readable artifact:
outputs/<config>/full_analysis.json. Deterministic; no model calls.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "evaluation" / "hebrew_bench_v2"

spec = importlib.util.spec_from_file_location("hb", REPO / "scripts" / "hebrew_bench_eval.py")
hb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hb)


def main() -> int:
    config = sys.argv[1]
    refs = json.loads((BENCH / "references.json").read_text(encoding="utf-8"))
    meta = {i["id"]: i for i in json.loads((BENCH / "items.json").read_text(encoding="utf-8"))["items"]}
    outdir = BENCH / "outputs" / config / "run1"

    rows = []
    for f in sorted(outdir.glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        iid = rec["item"]
        m = meta.get(iid)
        ref = refs.get(iid)
        if m is None or ref is None:
            continue
        hyp_raw = rec.get("transcription") or ""
        hyp, gt = hb.normalize(hyp_raw), hb.normalize(ref["text"])
        cer = hb.lev(hyp, gt) / max(len(gt), 1)
        gw, hw = gt.split(), hyp.split()
        s, d, i = hb.word_align(gw, hw)
        rows.append({
            "item": iid, "category": m["category"], "writer": m.get("writer"),
            "hard": m.get("hard", False),
            "cer": round(cer, 3),
            "wer": round((s + d + i) / max(len(gw), 1), 3),
            "omit": round(d / max(len(gw), 1), 3),
            "empty": not hyp.strip(),
            "error": rec.get("error"),
            "latency_s": rec.get("latency_s"),
            "hyp_preview": hyp_raw[:100],
        })

    analysis = {"config": config, "n": len(rows)}
    hw_strict = [r for r in rows if r["category"] in ("handwritten_line", "handwritten_cell") and not r["hard"]]
    hw_hard = [r for r in rows if r["category"] in ("handwritten_line", "handwritten_cell") and r["hard"]]

    def summarize(name, subset):
        if not subset:
            return None
        cers = sorted(r["cer"] for r in subset)
        n = len(cers)
        wers = [r["wer"] for r in subset]
        lats = sorted(r["latency_s"] or 0 for r in subset)
        out = {
            "n": n,
            "cer_mean": round(statistics.mean(cers), 3),
            "cer_median": round(statistics.median(cers), 3),
            "cer_p25": round(cers[n // 4], 3), "cer_p75": round(cers[min(n - 1, 3 * n // 4)], 3),
            "cer_min": cers[0], "cer_max": cers[-1],
            "wer_mean": round(statistics.mean(wers), 3),
            "wer_median": round(statistics.median(wers), 3),
            "omission_mean": round(statistics.mean(r["omit"] for r in subset), 3),
            "usable_025": sum(c <= .25 for c in cers), "usable_050": sum(c <= .50 for c in cers),
            "empty_outputs": sum(r["empty"] for r in subset),
            "inference_errors": sum(bool(r["error"]) for r in subset),
            "latency_median_s": lats[n // 2] if lats else None,
            "latency_p90_s": lats[min(n - 1, int(n * .9))] if lats else None,
        }
        print(f"{name}: {json.dumps(out)}")
        return out

    analysis["handwritten_strict"] = summarize("HANDWRITTEN strict", hw_strict)
    analysis["handwritten_hard"] = summarize("HANDWRITTEN hard", hw_hard)
    for cat in ("printed_rtl", "mixed_he_en", "formula_printed"):
        analysis[cat] = summarize(cat, [r for r in rows if r["category"] == cat])

    by_writer = defaultdict(list)
    for r in hw_strict:
        by_writer[r["writer"]].append(r["cer"])
    analysis["handwritten_by_writer"] = {
        w: {"n": len(v), "cer_mean": round(statistics.mean(v), 3),
            "cer_median": round(statistics.median(v), 3)}
        for w, v in sorted(by_writer.items())
    }
    print("by writer:", json.dumps(analysis["handwritten_by_writer"]))

    ranked = sorted(hw_strict, key=lambda r: r["cer"])
    analysis["best10"] = ranked[:10]
    analysis["worst10"] = ranked[-10:]
    print("\nBEST 10 (strict handwriting):")
    for r in ranked[:10]:
        print(f"  {r['cer']} {r['item']} :: {r['hyp_preview'][:70]}")
    print("WORST 10:")
    for r in ranked[-10:]:
        print(f"  {r['cer']} {r['item']} :: {r['hyp_preview'][:70]}")

    (BENCH / "outputs" / config / "full_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("\nwrote full_analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
