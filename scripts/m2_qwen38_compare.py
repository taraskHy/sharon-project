"""qwen38_27b_q4km vs canonical baselines on IDENTICAL items.

Post-inference only. Uses the canonical CER definition (m2_bench_eval:
hebrew_bench_eval.normalize/lev over reference text) and the PERSISTED
per-item records of gemini_protocol_clean_v1 and mlkit_ink_rtl_a1 (top-1),
never summary statistics. Writes evaluation/hebrew_bench_v2/outputs/
qwen38_27b_q4km/comparison.json.
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


def load_arm(config_id: str, text_key: str = "transcription") -> dict[str, str]:
    run = BENCH / "outputs" / config_id / "run1"
    out = {}
    for p in run.glob("*.json"):
        r = json.loads(p.read_text(encoding="utf-8"))
        t = r.get(text_key)
        if r.get("error") or not (t or "").strip():
            continue
        out[r["item"]] = t
    return out


def summarize(name: str, ids: list[str], texts: dict[str, str], refs: dict) -> dict:
    cers = [cer_of(refs[i]["text"], texts[i]) for i in ids]
    return {"arm": name, "n": len(ids),
            "mean_cer": round(st.mean(cers), 4), "median_cer": round(st.median(cers), 4),
            "usable_025": sum(1 for c in cers if c <= 0.25),
            "usable_050": sum(1 for c in cers if c <= 0.50),
            "per_item": {i: round(c, 4) for i, c in zip(ids, cers)}}


def main() -> int:
    refs = json.loads((BENCH / "references.json").read_text(encoding="utf-8"))
    items = {i["id"]: i for i in json.loads((BENCH / "items.json").read_text(encoding="utf-8"))["items"]}
    q = load_arm("qwen38_27b_q4km")
    gem = load_arm("gemini_protocol_clean_v1")
    ml = load_arm("mlkit_ink_rtl_a1")
    gate = [g["item"] for g in json.loads(
        (BENCH / "outputs" / "gemini3_flash" / "final_gate20_paired.json").read_text(encoding="utf-8"))]

    def scored(ids):
        return [i for i in ids if i in refs and refs[i].get("text")
                and items.get(i, {}).get("category") in ("handwritten_line", "handwritten_cell")]

    report = {"note": "identical-item comparisons; canonical CER; references post-inference"}
    # full handwritten qwen38 coverage
    hw = scored([i for i in q])
    report["qwen38_handwritten_all"] = summarize("qwen38_27b_q4km", hw, q, refs)
    # vs Gemini on identical items
    both_g = scored([i for i in q if i in gem])
    report["vs_gemini"] = {"n": len(both_g),
                           "qwen38": summarize("qwen38_27b_q4km", both_g, q, refs),
                           "gemini_protocol_clean": summarize("gemini_protocol_clean_v1", both_g, gem, refs)}
    w = sum(1 for i in both_g if cer_of(refs[i]["text"], q[i]) < cer_of(refs[i]["text"], gem[i]) - 1e-9)
    t = sum(1 for i in both_g if abs(cer_of(refs[i]["text"], q[i]) - cer_of(refs[i]["text"], gem[i])) < 1e-9)
    report["vs_gemini"]["qwen38_WTL"] = f"{w}/{t}/{len(both_g)-w-t}"
    # vs ML Kit on identical items
    both_m = scored([i for i in q if i in ml])
    report["vs_mlkit"] = {"n": len(both_m),
                          "qwen38": summarize("qwen38_27b_q4km", both_m, q, refs),
                          "mlkit": summarize("mlkit_ink_rtl_a1", both_m, ml, refs)}
    w = sum(1 for i in both_m if cer_of(refs[i]["text"], q[i]) < cer_of(refs[i]["text"], ml[i]) - 1e-9)
    t = sum(1 for i in both_m if abs(cer_of(refs[i]["text"], q[i]) - cer_of(refs[i]["text"], ml[i])) < 1e-9)
    report["vs_mlkit"]["qwen38_WTL"] = f"{w}/{t}/{len(both_m)-w-t}"
    # canonical gate-20 (three-way where qwen38 covers)
    g20 = scored([i for i in gate if i in q and i in gem and i in ml])
    report["gate20"] = {"n": len(g20), "missing_from_qwen38": [i for i in gate if i not in q],
                        "qwen38": summarize("qwen38_27b_q4km", g20, q, refs),
                        "gemini_protocol_clean": summarize("gemini_protocol_clean_v1", g20, gem, refs),
                        "mlkit": summarize("mlkit_ink_rtl_a1", g20, ml, refs)}
    out = BENCH / "outputs" / "qwen38_27b_q4km" / "comparison.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    for k in ("qwen38_handwritten_all", "vs_gemini", "vs_mlkit", "gate20"):
        v = report[k]
        if "arm" in v:
            print(f"{k}: n={v['n']} mean={v['mean_cer']} median={v['median_cer']} "
                  f"usable25={v['usable_025']} usable50={v['usable_050']}")
        else:
            print(f"{k}: n={v['n']}" + (f" WTL={v['qwen38_WTL']}" if "qwen38_WTL" in v else ""))
            for arm_key, arm in v.items():
                if isinstance(arm, dict) and "arm" in arm:
                    print(f"   {arm['arm']}: mean={arm['mean_cer']} median={arm['median_cer']} "
                          f"usable25={arm['usable_025']} usable50={arm['usable_050']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
