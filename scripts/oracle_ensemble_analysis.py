"""Oracle-ensemble analysis over the retained transcription outputs.

Pre-registered in NEXT_SESSION_HANDOFF.md: uses ONLY already-saved expert
outputs (run1 of every config, a fixed choice made before looking at any
number) plus the hidden GT, strictly post-hoc. No new inference, no
training, no MoE building — this measures whether an ensemble COULD help
and whether GT-free selection rules capture any of it.

Per verified strict cell:
- oracle = the expert output with the lowest CER (upper bound of any
  selection rule over these experts);
- per-expert CER; pairwise error correlation (Pearson over per-cell CER);
- GT-free inter-expert agreement (normalized similarity of outputs);
- cells where NO expert has meaningful text (min CER > 0.5 / > 0.25);
- GT-free selection rules evaluated post-hoc: medoid consensus (output
  closest to the other outputs) and agreement-gated abstention (accept a
  cell only if some pair of experts agrees >= tau). Selection NEVER looks
  at Hebrew fluency or GT — only inter-output agreement.

Outputs: evaluation/oracle_ensemble_analysis.md (+ per-cell CSV holding
only metric numbers, never GT text).

    .venv/Scripts/python.exe scripts/oracle_ensemble_analysis.py
"""

from __future__ import annotations

import csv
import json
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.hebrew_bench_eval import lev, normalize

BENCH = Path("evaluation/hebrew_bench")
OUT_MD = Path("evaluation/oracle_ensemble_analysis.md")
OUT_CSV = Path("evaluation/oracle_ensemble_percell.csv")

EXPERTS = [
    "it1_baseline_8b", "it2_strict_prompt", "it3_q8_quant", "it4_contrast",
    "it5_moe30b", "it6_hdd_words", "it7_surya",
    "isol0_orig_e002", "isol1_blueonly", "isol2_tsub", "isol3_tsub_lines",
    "isol4_hdd_blueonly", "isol5_hdd_tsub", "isol6_hdd_tsub_lines",
]
USABLE_CER = 0.25
MEANINGLESS_CER = 0.50
TAUS = [0.5, 0.6, 0.7, 0.8, 0.9]


def cer(hyp: str, ref: str) -> float:
    return lev(hyp, ref) / max(len(ref), 1)


def sim(a: str, b: str) -> float:
    """GT-free normalized similarity between two outputs."""
    if not a and not b:
        return 1.0
    return 1.0 - lev(a, b) / max(len(a), len(b), 1)


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    sy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def main() -> int:
    gt_path = BENCH / "verified_ground_truth.json"
    if not gt_path.exists():
        print("REFUSING: no human-verified ground truth")
        return 2
    gt = json.loads(gt_path.read_text(encoding="utf-8"))["cells"]
    strict = sorted(k for k, v in gt.items() if v["type"] == "strict"
                    and v.get("human_verified"))
    hard = sorted(k for k, v in gt.items() if v["type"] == "hard"
                  and v.get("human_verified"))

    # run1 output per expert per cell (missing file -> empty output).
    hyp: dict[str, dict[str, str]] = {}
    for e in EXPERTS:
        hyp[e] = {}
        for cid in strict + hard:
            f = BENCH / "outputs" / e / "run1" / f"{cid}.json"
            if f.exists():
                rec = json.loads(f.read_text(encoding="utf-8"))
                hyp[e][cid] = normalize(rec.get("transcription") or "")
            else:
                hyp[e][cid] = ""

    ref = {cid: normalize(gt[cid]["text"]) for cid in strict}
    C = {e: {cid: cer(hyp[e][cid], ref[cid]) for cid in strict} for e in EXPERTS}

    # --- per-expert + oracle ------------------------------------------------
    expert_rows = []
    for e in EXPERTS:
        cs = [C[e][cid] for cid in strict]
        expert_rows.append({
            "expert": e,
            "mean_cer": sum(cs) / len(cs),
            "usable": sum(c <= USABLE_CER for c in cs),
        })
    oracle_pick = {cid: min(EXPERTS, key=lambda e: C[e][cid]) for cid in strict}
    oracle_cer = {cid: C[oracle_pick[cid]][cid] for cid in strict}
    oracle_mean = sum(oracle_cer.values()) / len(strict)
    oracle_usable = sum(c <= USABLE_CER for c in oracle_cer.values())
    no_meaning_50 = [cid for cid in strict if oracle_cer[cid] > MEANINGLESS_CER]
    best_single = min(expert_rows, key=lambda r: r["mean_cer"])

    # --- pairwise error correlation (GT-based) ------------------------------
    pair_corr = {}
    for a, b in combinations(EXPERTS, 2):
        pair_corr[(a, b)] = pearson([C[a][c] for c in strict],
                                    [C[b][c] for c in strict])
    finite = [v for v in pair_corr.values() if v == v]
    mean_corr = sum(finite) / len(finite)

    # --- GT-free selection rules --------------------------------------------
    def medoid(cid: str) -> str:
        outs = {e: hyp[e][cid] for e in EXPERTS}
        return min(EXPERTS, key=lambda e: sum(
            lev(outs[e], outs[o]) / max(len(outs[e]), len(outs[o]), 1)
            for o in EXPERTS if o != e))

    med_pick = {cid: medoid(cid) for cid in strict}
    med_cer = [C[med_pick[cid]][cid] for cid in strict]
    med_mean = sum(med_cer) / len(med_cer)
    med_usable = sum(c <= USABLE_CER for c in med_cer)

    max_pair_sim = {}
    for cid in strict:
        max_pair_sim[cid] = max(sim(hyp[a][cid], hyp[b][cid])
                                for a, b in combinations(EXPERTS, 2))
    gate_rows = []
    for tau in TAUS:
        acc = [cid for cid in strict if max_pair_sim[cid] >= tau]
        if acc:
            cs = [C[med_pick[cid]][cid] for cid in acc]
            gate_rows.append({"tau": tau, "coverage": len(acc),
                              "cer_on_accepted": sum(cs) / len(cs),
                              "usable_on_accepted": sum(c <= USABLE_CER for c in cs)})
        else:
            gate_rows.append({"tau": tau, "coverage": 0,
                              "cer_on_accepted": float("nan"),
                              "usable_on_accepted": 0})

    # --- hard cells: honest abstention counts (any expert flagging) ---------
    markers = ["[unreadable]", "[?]", "לא קריא", "unreadable"]
    hard_flags = {cid: sum(any(m in hyp[e][cid] for m in markers)
                           for e in EXPERTS) for cid in hard}

    # --- write CSV (numbers only) -------------------------------------------
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell"] + EXPERTS + ["oracle_cer", "oracle_expert",
                                         "medoid_expert", "max_pair_sim"])
        for cid in strict:
            w.writerow([cid] + [round(C[e][cid], 4) for e in EXPERTS]
                       + [round(oracle_cer[cid], 4), oracle_pick[cid],
                          med_pick[cid], round(max_pair_sim[cid], 4)])

    # --- report ---------------------------------------------------------------
    L = []
    L.append("# Oracle-ensemble analysis (post-hoc, run1 of every config)\n")
    L.append(f"Experts: {len(EXPERTS)} configs; strict cells: {len(strict)}; "
             f"hard cells: {len(hard)}. Fixed rule: run1 output per config; "
             "no GT-based selection is proposed for deployment — this is an "
             "upper-bound measurement.\n")
    L.append("## Per-expert vs oracle (strict cells)\n")
    L.append("| expert | mean CER | usable |")
    L.append("|---|---|---|")
    for r in sorted(expert_rows, key=lambda r: r["mean_cer"]):
        L.append(f"| {r['expert']} | {r['mean_cer']:.3f} | {r['usable']}/{len(strict)} |")
    L.append(f"| **oracle (lowest-CER expert per cell)** | **{oracle_mean:.3f}** "
             f"| **{oracle_usable}/{len(strict)}** |")
    L.append("")
    L.append(f"- Best single expert: {best_single['expert']} "
             f"(CER {best_single['mean_cer']:.3f}).")
    L.append(f"- Oracle improvement over best single: "
             f"{best_single['mean_cer'] - oracle_mean:.3f} CER "
             f"({(best_single['mean_cer'] - oracle_mean) / best_single['mean_cer']:.0%} rel).")
    L.append(f"- Cells where NO expert reaches CER <= {MEANINGLESS_CER}: "
             f"{len(no_meaning_50)}/{len(strict)} ({', '.join(no_meaning_50) or 'none'}).")
    L.append(f"- Cells where NO expert reaches usable (CER <= {USABLE_CER}): "
             f"{len(strict) - oracle_usable}/{len(strict)}.")
    L.append(f"- Mean pairwise error correlation (Pearson over per-cell CER): "
             f"{mean_corr:.3f} (1 = experts fail identically).\n")
    L.append("## GT-free selection rules (evaluated post-hoc)\n")
    L.append(f"- Medoid consensus (closest-to-others output): CER {med_mean:.3f}, "
             f"usable {med_usable}/{len(strict)}.")
    L.append("- Agreement-gated abstention (accept cell iff any expert pair "
             "agrees >= tau; medoid on accepted):\n")
    L.append("| tau | coverage | CER on accepted | usable on accepted |")
    L.append("|---|---|---|---|")
    for g in gate_rows:
        cer_s = "-" if g["cer_on_accepted"] != g["cer_on_accepted"] else f"{g['cer_on_accepted']:.3f}"
        L.append(f"| {g['tau']} | {g['coverage']}/{len(strict)} | {cer_s} "
                 f"| {g['usable_on_accepted']} |")
    L.append("")
    L.append("## Hard cells (honest-abstention counts)\n")
    L.append("| cell | experts flagging unreadable (of "
             f"{len(EXPERTS)}) |")
    L.append("|---|---|")
    for cid in hard:
        L.append(f"| {cid} | {hard_flags[cid]} |")
    L.append("")
    L.append("## Decision (per the pre-registered gate)\n")
    L.append("DECISION_TO_FILL\n")
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:40]))
    print(f"... -> {OUT_MD}, {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
