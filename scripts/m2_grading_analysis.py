"""Analyze grading-decision-preservation ledgers (Mission 2).

Terminology: the comparison baseline is the REFERENCE-SIDE FIXED-GRADER
DECISION (the same fixed judge reading the owner-verified transcription).
It is NOT ground truth and NOT a human grade. Metrics measure decision
preservation under OCR corruption, not objective grading accuracy.

Deterministic; reads only the persisted JSONL ledgers.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
GDIR = REPO / "evaluation" / "m2_grading"
ARMS = ["qwen8b_strict_contrast", "mlkit_ink_rtl_a1", "gemini3_flash"]
BANDS = [(0.0, 0.25), (0.25, 0.50), (0.50, 99.0)]


def load(arm):
    p = GDIR / f"{arm}.jsonl"
    if not p.exists():
        return {}
    recs = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            recs[r["cell"]] = r
        except json.JSONDecodeError:
            pass
    return recs


def summarize(name, recs):
    n = len(recs)
    if not n:
        return None
    agree = sum(r["agree"] for r in recs.values())
    dirs = Counter(r["direction"] for r in recs.values())
    conf = Counter((r["verdict_ref"], r["verdict_ocr"]) for r in recs.values())
    out = {
        "arm": name, "n": n,
        "decision_agreement_rate": round(agree / n, 3),
        "flip_rate": round(1 - agree / n, 3),
        "directions": dict(dirs),
        "confusion_ref_vs_ocr": {f"{a}->{b}": c for (a, b), c in sorted(conf.items())},
        "by_cer_band": {},
    }
    for lo, hi in BANDS:
        band = [r for r in recs.values() if lo <= r["ocr_cell_cer"] < hi]
        if band:
            out["by_cer_band"][f"{lo}-{hi if hi < 99 else 'inf'}"] = {
                "n": len(band),
                "agreement": round(sum(r["agree"] for r in band) / len(band), 3),
                "mean_cer": round(statistics.mean(r["ocr_cell_cer"] for r in band), 3),
            }
    print(json.dumps(out, ensure_ascii=False, indent=1))
    return out


def main() -> int:
    data = {arm: load(arm) for arm in ARMS}
    report = {"_terminology": (
        "Baseline = reference-side fixed-grader decision (same fixed judge on "
        "the owner-verified transcription). NOT ground truth, NOT human "
        "grades. Metrics measure OCR-induced decision preservation only."
    )}
    for arm in ARMS:
        report[arm] = summarize(arm, data[arm])

    # cross-model comparisons on IDENTICAL cell subsets only
    report["common_subsets"] = {}
    for a, b in [("qwen8b_strict_contrast", "mlkit_ink_rtl_a1"),
                 ("qwen8b_strict_contrast", "gemini3_flash"),
                 ("mlkit_ink_rtl_a1", "gemini3_flash")]:
        common = sorted(set(data[a]) & set(data[b]))
        if not common:
            continue
        entry = {"n": len(common)}
        for arm in (a, b):
            sub = [data[arm][c] for c in common]
            entry[arm] = {
                "agreement": round(sum(r["agree"] for r in sub) / len(sub), 3),
                "mean_cell_cer": round(statistics.mean(r["ocr_cell_cer"] for r in sub), 3),
            }
        report["common_subsets"][f"{a} vs {b}"] = entry
        print(f"\nCOMMON {a} vs {b} (n={len(common)}):")
        print(json.dumps(entry, indent=1))

    # danger analysis: flips at low CER, preservation at high CER
    for arm in ARMS:
        recs = data[arm]
        low_flips = [c for c, r in recs.items() if r["ocr_cell_cer"] <= 0.25 and not r["agree"]]
        high_pres = [c for c, r in recs.items() if r["ocr_cell_cer"] > 0.50 and r["agree"]]
        report.setdefault("danger_analysis", {})[arm] = {
            "low_cer_flips": low_flips,
            "high_cer_preserved_n": len(high_pres),
            "high_cer_preserved_examples": high_pres[:5],
        }
    print("\ndanger analysis:", json.dumps(report["danger_analysis"], ensure_ascii=False, indent=1))

    (GDIR / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print("\nwrote evaluation/m2_grading/analysis.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
