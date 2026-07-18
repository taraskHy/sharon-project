"""Claude Vision candidate benchmark — evaluation side (the ONLY GT reader).

Run strictly AFTER all raw outputs exist under
evaluation/claude_candidates/outputs/. Scores both passes against the
owner's hidden labels, evaluates the owner's pre-registered acceptance
gate (evaluation/claude_candidates/PROTOCOL.md — not to be weakened),
and writes:

    evaluation/claude_annotation_candidate_results.csv   per line x pass
    evaluation/claude_candidates/eval_summary.json       metrics + verdict
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
from scripts.htr_annotation_lib import load_all_annotations, normalize_text  # noqa: E402

ROOT = Path("evaluation/htr_pilot")
OUT = Path("evaluation/claude_candidates")
CSV_PATH = Path("evaluation/claude_annotation_candidate_results.csv")
CONFIGS = ("claude_line", "claude_line_cell")

# Owner's acceptance gate, verbatim operationalization (PROTOCOL.md):
GATE = {
    "min_exact_rate": 0.40,          # normalized exact-match definition
    "max_median_cer": 0.10,
    "max_major_halluc_line_rate": 0.05,  # >= 2 inserted words on a line
    "agreement_taus": (1.0, 0.98, 0.95, 0.9, 0.85, 0.8),
    "agreement_subset_max_cer": 0.10,
    "agreement_subset_min_n": 5,
}

# Time-saved model constants (PROTOCOL.md; estimate, not measurement)
T_SCRATCH_MIN, T_PER_CHAR = 15.0, 0.55
T_REVIEW_EXACT, T_FIX_BASE, T_FIX_PER_EDIT, T_WASTED = 6.0, 8.0, 1.2, 5.0


def strict_norm(s: str) -> str:
    return " ".join(normalize_text(s or "").split())


def agreement(a: str, b: str) -> float:
    na, nb = normalize(a or ""), normalize(b or "")
    if not na and not nb:
        return 1.0
    return 1.0 - lev(na, nb) / max(len(na), len(nb), 1)


def line_metrics(hyp_raw: str, ref_raw: str) -> dict:
    hyp, ref = normalize(hyp_raw or ""), normalize(ref_raw)
    cer = lev(hyp, ref) / max(len(ref), 1)
    gw, hw = ref.split(), hyp.split()
    subs, dels, ins = word_align(gw, hw)
    exact_strict = strict_norm(hyp_raw) == strict_norm(ref_raw) and bool(hyp_raw)
    char_edits = lev(hyp, ref)
    t_scratch = max(T_SCRATCH_MIN, T_PER_CHAR * len(strict_norm(ref_raw)))
    if exact_strict:
        t_assisted = T_REVIEW_EXACT
    elif cer <= 0.15:
        t_assisted = T_FIX_BASE + T_FIX_PER_EDIT * char_edits
    else:
        t_assisted = t_scratch + T_WASTED
    return {
        "cer": round(cer, 4),
        "wer": round((subs + dels + ins) / max(len(gw), 1), 4),
        "exact_strict": exact_strict,
        "exact_norm": hyp == ref and bool(ref),
        "edit_class": ("no_edit" if exact_strict else
                       "minor" if cer <= 0.15 else
                       "moderate" if cer <= 0.40 else "major"),
        "subs": subs, "dels": dels, "ins": ins,
        "major_halluc": ins >= 2,
        "ref_words": len(gw), "hyp_words": len(hw),
        "char_edits": char_edits,
        "t_scratch_s": round(t_scratch, 1),
        "t_assisted_s": round(t_assisted, 1),
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    cers = sorted(r["cer"] for r in rows)
    gtw = sum(r["ref_words"] for r in rows)
    hyw = sum(r["hyp_words"] for r in rows)
    return {
        "n": n,
        "mean_cer": round(sum(cers) / n, 4),
        "median_cer": round(cers[n // 2] if n % 2 else
                            (cers[n // 2 - 1] + cers[n // 2]) / 2, 4),
        "mean_wer": round(sum(r["wer"] for r in rows) / n, 4),
        "exact_strict_rate": round(sum(r["exact_strict"] for r in rows) / n, 4),
        "exact_norm_rate": round(sum(r["exact_norm"] for r in rows) / n, 4),
        "no_edit_rate": round(sum(r["edit_class"] == "no_edit" for r in rows) / n, 4),
        "minor_rate": round(sum(r["edit_class"] == "minor" for r in rows) / n, 4),
        "moderate_rate": round(sum(r["edit_class"] == "moderate" for r in rows) / n, 4),
        "major_rate": round(sum(r["edit_class"] == "major" for r in rows) / n, 4),
        "omission_rate": round(sum(r["dels"] for r in rows) / max(gtw, 1), 4),
        "insertion_rate": round(sum(r["ins"] for r in rows) / max(hyw, 1), 4),
        "major_halluc_line_rate": round(sum(r["major_halluc"] for r in rows) / n, 4),
        "errors": sum(1 for r in rows if r.get("error")),
        "est_time_scratch_s": round(sum(r["t_scratch_s"] for r in rows), 1),
        "est_time_assisted_s": round(sum(r["t_assisted_s"] for r in rows), 1),
        "est_time_saved_s": round(sum(r["t_scratch_s"] - r["t_assisted_s"]
                                      for r in rows), 1),
    }


def main() -> int:
    sel = json.loads((OUT / "claude_bench_ids.json").read_text(encoding="utf-8"))
    ids = sel["ids"]
    ann = load_all_annotations(ROOT, "train")

    cands: dict[str, dict[str, dict]] = {}
    for config in CONFIGS:
        d = OUT / "outputs" / config
        cands[config] = {}
        for sid in ids:
            p = d / f"{sid}.json"
            if p.exists():
                cands[config][sid] = json.loads(p.read_text(encoding="utf-8"))
    missing = {c: [s for s in ids if s not in cands[c]] for c in CONFIGS}
    if any(missing.values()):
        print(f"REFUSING: outputs incomplete (raw-before-eval rule): "
              f"{ {c: len(v) for c, v in missing.items()} } missing")
        return 2

    all_rows = []
    for sid in ids:
        rec = ann[sid]
        assert rec["status"] == "ok" and rec["human_verified"]
        ref_raw = rec["transcription"]
        ab = agreement(cands["claude_line"][sid].get("candidate") or "",
                       cands["claude_line_cell"][sid].get("candidate") or "")
        for config in CONFIGS:
            c = cands[config][sid]
            m = line_metrics(c.get("candidate") or "", ref_raw)
            all_rows.append({
                "sample_id": sid, "config": config, "writer": rec["writer"],
                **m, "agreement_ab": round(ab, 4),
                "stop_reason": c.get("stop_reason"),
                "latency_s": c.get("latency_s"),
                "error": c.get("error") or "",
                "candidate": c.get("candidate") or "",
            })

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)

    summary: dict = {"at": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "bench_n": len(ids), "gate": GATE,
                     "configs": {}, "agreement_subsets": {},
                     "per_config_gate": {}, "verdict": None}
    verdicts = {}
    for config in CONFIGS:
        rows = [r for r in all_rows if r["config"] == config]
        s = summarize(rows)
        summary["configs"][config] = s
        subsets = []
        agree_ok = False
        for tau in GATE["agreement_taus"]:
            sub = [r for r in rows if r["agreement_ab"] >= tau]
            if len(sub) >= GATE["agreement_subset_min_n"]:
                sub_cer = sum(r["cer"] for r in sub) / len(sub)
                entry = {"tau": tau, "n": len(sub),
                         "mean_cer": round(sub_cer, 4),
                         "passes": sub_cer <= GATE["agreement_subset_max_cer"]}
                subsets.append(entry)
                agree_ok = agree_ok or entry["passes"]
        summary["agreement_subsets"][config] = subsets
        checks = {
            "exact_match_ge_40pct": s["exact_norm_rate"] >= GATE["min_exact_rate"],
            "median_cer_le_0.10": s["median_cer"] <= GATE["max_median_cer"],
            "major_halluc_le_5pct_lines":
                s["major_halluc_line_rate"] <= GATE["max_major_halluc_line_rate"],
            "agreement_subset_cer_le_0.10": agree_ok,
        }
        checks["all"] = all(checks.values())
        summary["per_config_gate"][config] = checks
        verdicts[config] = checks["all"]

    summary["verdict"] = "ACCEPT" if any(verdicts.values()) else "REJECT"
    (OUT / "eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\nVERDICT: {summary['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
