"""Paired raw-vs-RAG-suggested CER for a RAG arm (post-inference only).

Reuses the canonical CER definition (hebrew_bench_eval.normalize/lev, as in
m2_bench_eval). Reads each RAG record's raw_text and suggested_text and
joins references AFTER the records exist. Writes <arm>/paired_eval.json.
Usage: python scripts/m2_rag_paired.py <rag_config_id>
"""

from __future__ import annotations

import importlib.util
import json
import statistics as st
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "evaluation" / "hebrew_bench_v2"
spec = importlib.util.spec_from_file_location("hb_eval", REPO / "scripts" / "hebrew_bench_eval.py")
hb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hb)


def cer_of(ref: str, hyp: str) -> float:
    h, g = hb.normalize(hyp), hb.normalize(ref)
    return hb.lev(h, g) / max(len(g), 1)


def main() -> int:
    cfg = sys.argv[1]
    refs = json.loads((BENCH / "references.json").read_text(encoding="utf-8"))
    run = BENCH / "outputs" / cfg / "run1"
    rows = []
    for p in sorted(run.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        ref = refs.get(rec["item"])
        if not ref or not ref.get("text"):
            continue
        rows.append({"item": rec["item"],
                     "changed": rec["suggested_text"] != rec["raw_text"],
                     "cer_raw": cer_of(ref["text"], rec["raw_text"]),
                     "cer_sug": cer_of(ref["text"], rec["suggested_text"]),
                     "edits": len(rec.get("edits", [])),
                     "semantic_change_risk": rec.get("semantic_change_risk"),
                     "needs_review": rec.get("needs_review"),
                     "error": rec.get("error")})
    n = len(rows)
    ch = [r for r in rows if r["changed"]]
    imp = [r for r in ch if r["cer_sug"] < r["cer_raw"] - 1e-9]
    wor = [r for r in ch if r["cer_sug"] > r["cer_raw"] + 1e-9]
    summary = {
        "config_id": cfg, "n": n,
        "mean_cer_raw": round(st.mean(r["cer_raw"] for r in rows), 4),
        "mean_cer_suggested": round(st.mean(r["cer_sug"] for r in rows), 4),
        "median_cer_raw": round(st.median(r["cer_raw"] for r in rows), 4),
        "median_cer_suggested": round(st.median(r["cer_sug"] for r in rows), 4),
        "usable_025_raw": sum(1 for r in rows if r["cer_raw"] <= 0.25),
        "usable_025_suggested": sum(1 for r in rows if r["cer_sug"] <= 0.25),
        "usable_050_raw": sum(1 for r in rows if r["cer_raw"] <= 0.50),
        "usable_050_suggested": sum(1 for r in rows if r["cer_sug"] <= 0.50),
        "texts_changed": len(ch), "improved": len(imp), "worsened": len(wor),
        "neutral": len(ch) - len(imp) - len(wor),
        "repair_errors": sum(1 for r in rows if r["error"]),
        "semantic_risk_flags": sum(1 for r in rows if r["semantic_change_risk"]),
        "needs_review": sum(1 for r in rows if r["needs_review"]),
        "changed_items": sorted(({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
                                 for r in ch), key=lambda r: r["cer_sug"] - r["cer_raw"]),
    }
    out = BENCH / "outputs" / cfg / "paired_eval.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in summary.items():
        if k != "changed_items":
            print(f"{k}: {v}")
    for r in summary["changed_items"]:
        print(f"  {r['item']}: {r['cer_raw']:.3f} -> {r['cer_sug']:.3f} edits={r['edits']} "
              f"sem_risk={r['semantic_change_risk']} review={r['needs_review']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
