"""Score saved candidate transcriptions against owner ground truth.

The ONLY ground-truth reader of the assisted-annotation benchmark, run
strictly AFTER raw candidate outputs exist on disk
(evaluation/htr_candidates/outputs/<config>/<sample_id>.json).

Tripwire: any candidate whose sample_id is among the overfit-test ids is
counted `contaminated` and excluded from every metric (the CRNN
checkpoint trained on those exact lines).

Writes evaluation/annotation_candidate_results.csv (one row per sample ×
config) + evaluation/htr_candidates/eval_summary.json, and evaluates the
pre-registered decision gate of evaluation/htr_candidates/PROTOCOL.md.
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.hebrew_bench_eval import lev, normalize, word_align  # noqa: E402
from scripts.htr_annotation_lib import (  # noqa: E402
    UNREADABLE_TOKEN, load_all_annotations,
)

ROOT = Path("evaluation/htr_pilot")
OUT = Path("evaluation/htr_candidates")
CSV_PATH = Path("evaluation/annotation_candidate_results.csv")
CONFIGS = ("qwen_line", "qwen_line_cell", "crnn_overfit")

# Pre-registered gate (PROTOCOL.md): a conservative subset must reach
GATE = {"min_coverage": 0.20, "min_helpful_rate": 0.60, "max_major_rate": 0.10}


def edit_class(cer: float) -> str:
    if cer == 0:
        return "no_edit"
    if cer <= 0.15:
        return "minor"
    if cer <= 0.40:
        return "moderate"
    return "major"


def agreement(a: str, b: str) -> float:
    na, nb = normalize(a or ""), normalize(b or "")
    if not na and not nb:
        return 1.0
    return 1.0 - lev(na, nb) / max(len(na), len(nb), 1)


def load_candidates(config: str, ids: list[str]) -> dict[str, dict]:
    d = OUT / "outputs" / config
    out = {}
    for sid in ids:
        p = d / f"{sid}.json"
        if p.exists():
            out[sid] = json.loads(p.read_text(encoding="utf-8"))
    return out


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    gtw = sum(r["ref_words"] for r in rows)
    hyw = sum(r["hyp_words"] for r in rows)
    cls = {c: sum(r["edit_class"] == c for r in rows) / n
           for c in ("no_edit", "minor", "moderate", "major")}
    return {
        "n": n,
        "mean_cer": round(sum(r["cer"] for r in rows) / n, 4),
        "median_cer": round(sorted(r["cer"] for r in rows)[n // 2], 4),
        "mean_wer": round(sum(r["wer"] for r in rows) / n, 4),
        "exact_rate": round(sum(r["exact"] for r in rows) / n, 4),
        "helpful_rate(no_edit+minor)": round(cls["no_edit"] + cls["minor"], 4),
        **{f"{c}_rate": round(v, 4) for c, v in cls.items()},
        "omission_rate": round(sum(r["dels"] for r in rows) / max(gtw, 1), 4),
        "insertion_rate": round(sum(r["ins"] for r in rows) / max(hyw, 1), 4),
        "halluc_line_rate(ins>=1)": round(sum(r["ins"] >= 1 for r in rows) / n, 4),
        "errors": sum(1 for r in rows if r["error"]),
    }


def main() -> int:
    sel = json.loads((OUT / "bench_ids.json").read_text(encoding="utf-8"))
    ids = sel["ids"]
    overfit = set(sel["excluded_overfit_ids"])
    ann = load_all_annotations(ROOT, "train")

    cands = {c: load_candidates(c, ids) for c in CONFIGS}
    contaminated = {c: sorted(set(cands[c]) & overfit) for c in CONFIGS}

    all_rows = []
    for sid in ids:
        rec = ann[sid]
        assert rec["status"] == "ok" and rec["human_verified"]
        ref_raw = rec["transcription"]
        has_span = UNREADABLE_TOKEN in ref_raw
        ref = normalize(ref_raw)
        ab = agreement(
            (cands["qwen_line"].get(sid) or {}).get("candidate") or "",
            (cands["qwen_line_cell"].get(sid) or {}).get("candidate") or "")
        for config in CONFIGS:
            c = cands[config].get(sid)
            if c is None or sid in overfit:
                continue
            hyp_raw = c.get("candidate") or ""
            hyp = normalize(hyp_raw)
            cer = lev(hyp, ref) / max(len(ref), 1)
            gw, hw = ref.split(), hyp.split()
            subs, dels, ins = word_align(gw, hw)
            all_rows.append({
                "sample_id": sid, "config": config, "writer": rec["writer"],
                "has_span": has_span,
                "cer": round(cer, 4),
                "wer": round((subs + dels + ins) / max(len(gw), 1), 4),
                "exact": hyp == ref, "edit_class": edit_class(cer),
                "ref_words": len(gw), "hyp_words": len(hw),
                "subs": subs, "dels": dels, "ins": ins,
                "agreement_ab": round(ab, 4),
                "confidence": c.get("confidence"),
                "latency_s": c.get("latency_s"),
                "error": c.get("error") or "",
                "candidate": hyp_raw,
            })

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)

    summary: dict = {"at": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "bench_n": len(ids),
                     "contaminated_excluded": contaminated,
                     "configs": {}, "agreement_analysis": {},
                     "gate": GATE, "gate_search": [], "decision": None}
    for config in CONFIGS:
        rows = [r for r in all_rows if r["config"] == config]
        summary["configs"][config] = {
            "all": summarize(rows),
            "clean_gt": summarize([r for r in rows if not r["has_span"]]),
            "span_gt": summarize([r for r in rows if r["has_span"]]),
        }

    # does A<->B agreement predict correctness of A?
    a_rows = [r for r in all_rows if r["config"] == "qwen_line"]
    for lo, hi, tag in ((0.9, 1.01, ">=0.9"), (0.7, 0.9, "0.7-0.9"),
                        (0.0, 0.7, "<0.7")):
        rows = [r for r in a_rows if lo <= r["agreement_ab"] < hi]
        if rows:
            summary["agreement_analysis"][tag] = summarize(rows)

    # pre-registered gate search over model-visible signals only
    best = None
    for config in ("qwen_line", "qwen_line_cell"):
        rows = [r for r in all_rows if r["config"] == config]
        for tau in (0.99, 0.95, 0.9, 0.8, 0.7):
            sub = [r for r in rows if r["agreement_ab"] >= tau]
            if not sub:
                continue
            s = summarize(sub)
            entry = {"signal": f"agreement>={tau}", "candidate": config,
                     "coverage": round(len(sub) / len(rows), 4),
                     "helpful_rate": s["helpful_rate(no_edit+minor)"],
                     "major_rate": s["major_rate"], "n": len(sub)}
            entry["passes_gate"] = (
                entry["coverage"] >= GATE["min_coverage"]
                and entry["helpful_rate"] >= GATE["min_helpful_rate"]
                and entry["major_rate"] <= GATE["max_major_rate"])
            summary["gate_search"].append(entry)
            if entry["passes_gate"] and (best is None
                                         or entry["coverage"] > best["coverage"]):
                best = entry
    crnn_rows = [r for r in all_rows if r["config"] == "crnn_overfit"]
    for tau in (0.9, 0.8, 0.7):
        sub = [r for r in crnn_rows if (r["confidence"] or 0) >= tau]
        if not sub:
            continue
        s = summarize(sub)
        entry = {"signal": f"crnn_conf>={tau}", "candidate": "crnn_overfit",
                 "coverage": round(len(sub) / max(len(crnn_rows), 1), 4),
                 "helpful_rate": s["helpful_rate(no_edit+minor)"],
                 "major_rate": s["major_rate"], "n": len(sub)}
        entry["passes_gate"] = (
            entry["coverage"] >= GATE["min_coverage"]
            and entry["helpful_rate"] >= GATE["min_helpful_rate"]
            and entry["major_rate"] <= GATE["max_major_rate"])
        summary["gate_search"].append(entry)
        if entry["passes_gate"] and (best is None
                                     or entry["coverage"] > best["coverage"]):
            best = entry

    summary["decision"] = (
        {"verdict": "ALLOW_CONSERVATIVE_DISPLAY", "subset": best}
        if best else {"verdict": "REJECT_PREFILL", "subset": None})
    (OUT / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    slim = json.loads(json.dumps(summary))
    print(json.dumps(slim, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
