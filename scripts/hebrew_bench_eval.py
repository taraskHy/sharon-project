"""Evaluate transcription outputs against the HIDDEN ground truth.

The ONLY component allowed to read ground_truth.json, strictly after
inference. Computes per the campaign definitions:

- normalized CER (char Levenshtein / len(gt)) over STRICT cells;
- WER, omission rate (deletions/gt words), hallucinated-word rate
  (insertions/hyp words) from word-level alignment;
- usable-transcription rate (cell usable iff CER <= 0.25);
- flagged-not-guessed behaviour on HARD cells;
- stability: mean pairwise CER between repeated runs' outputs.

Appends one row per config to evaluation/hebrew_transcription_results.csv.
"""

from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BENCH = Path("evaluation/hebrew_bench")
CSV_PATH = Path("evaluation/hebrew_transcription_results.csv")
UNREADABLE_MARKERS = ["[unreadable]", "[?]", "לא קריא", "unreadable"]


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")  # strip niqqud
    s = re.sub(r"[\"'`()\[\]{}.,;:!?~ـ—–-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s


def lev(a, b) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def word_align(gt_words, hyp_words):
    """DP alignment; returns (substitutions, deletions, insertions)."""
    m, n = len(gt_words), len(hyp_words)
    D = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        D[i][0] = i
    for j in range(n + 1):
        D[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1,
                          D[i - 1][j - 1] + (gt_words[i - 1] != hyp_words[j - 1]))
    subs = dels = ins = 0
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + (gt_words[i - 1] != hyp_words[j - 1]):
            subs += gt_words[i - 1] != hyp_words[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return subs, dels, ins


def main() -> int:
    config_id = sys.argv[1]
    gt_path = BENCH / "verified_ground_truth.json"
    if not gt_path.exists():
        print(
            "REFUSING TO EVALUATE: no HUMAN-VERIFIED ground truth exists yet.\n"
            "The AI's own readings (candidate_annotations.json) are not "
            "authoritative — the system under evaluation cannot annotate its "
            "own benchmark. Fill evaluation/hebrew_bench/human_annotation/"
            "annotation_template.csv (human_transcription + "
            "human_verified=true), then build verified_ground_truth.json "
            "with scripts/build_verified_gt.py."
        )
        return 2
    gt = json.loads(gt_path.read_text(encoding="utf-8"))["cells"]
    unverified = [k for k, v in gt.items() if not v.get("human_verified")]
    if unverified:
        print(f"REFUSING TO EVALUATE: {len(unverified)} cells lack human_verified=true: {unverified}")
        return 2
    outdir = BENCH / "outputs" / config_id
    cfg = json.loads((outdir / "config.json").read_text(encoding="utf-8"))
    runs = sorted(d for d in outdir.iterdir() if d.is_dir() and d.name.startswith("run"))

    per_run = []
    detail_lines = []
    for rundir in runs:
        cer_sum = cer_n = 0.0
        usable = 0
        gt_words_total = del_total = 0
        hyp_words_total = ins_total = 0
        wer_err = 0
        hard_flag_ok = hard_halluc = hard_n = 0
        cells = {}
        for f in sorted(rundir.glob("*.json")):
            rec = json.loads(f.read_text(encoding="utf-8"))
            cid = rec["cell"]
            hyp_raw = rec.get("transcription") or ""
            cells[cid] = hyp_raw
            g = gt.get(cid)
            if g is None:
                continue
            hyp, ref = normalize(hyp_raw), normalize(g["text"])
            flagged = any(m in (hyp_raw or "").lower() for m in UNREADABLE_MARKERS)
            if g["type"] == "hard":
                hard_n += 1
                cer = lev(hyp, ref) / max(len(ref), 1)
                if flagged or cer <= 0.35:
                    hard_flag_ok += 1
                elif hyp:
                    hard_halluc += 1
                detail_lines.append(f"{rundir.name} {cid} [hard] flagged={flagged} cer={cer:.2f} hyp={hyp_raw[:70]!r}")
                continue
            cer = lev(hyp, ref) / max(len(ref), 1)
            cer_sum += cer
            cer_n += 1
            usable += cer <= 0.25
            gw, hw = ref.split(), hyp.split()
            subs, dels, ins = word_align(gw, hw)
            gt_words_total += len(gw)
            hyp_words_total += len(hw)
            del_total += dels
            ins_total += ins
            wer_err += subs + dels + ins
            detail_lines.append(f"{rundir.name} {cid} cer={cer:.2f} hyp={hyp_raw[:70]!r}")
        per_run.append({
            "cer": cer_sum / max(cer_n, 1),
            "usable_rate": usable / max(cer_n, 1),
            "wer": wer_err / max(gt_words_total, 1),
            "omission_rate": del_total / max(gt_words_total, 1),
            "halluc_rate": ins_total / max(hyp_words_total, 1),
            "hard_flag_ok": hard_flag_ok, "hard_halluc": hard_halluc, "hard_n": hard_n,
            "cells": cells,
        })

    # stability: mean pairwise CER between runs' outputs on strict cells
    stab = []
    strict_ids = [k for k, v in gt.items() if v["type"] == "strict"]
    for i in range(len(per_run)):
        for j in range(i + 1, len(per_run)):
            for cid in strict_ids:
                a = normalize(per_run[i]["cells"].get(cid, ""))
                b = normalize(per_run[j]["cells"].get(cid, ""))
                stab.append(lev(a, b) / max(len(a), len(b), 1))
    stability_cer = sum(stab) / max(len(stab), 1)

    def avg(key):
        return sum(r[key] for r in per_run) / max(len(per_run), 1)

    row = {
        "config_id": config_id, "model": cfg["model"], "prompt": cfg["prompt"],
        "preproc": cfg["preproc"], "runs": len(per_run),
        "mean_cer": round(avg("cer"), 4), "usable_rate": round(avg("usable_rate"), 4),
        "wer": round(avg("wer"), 4), "omission_rate": round(avg("omission_rate"), 4),
        "halluc_rate": round(avg("halluc_rate"), 4),
        "stability_pairwise_cer": round(stability_cer, 4),
        "hard_flag_ok": sum(r["hard_flag_ok"] for r in per_run),
        "hard_halluc": sum(r["hard_halluc"] for r in per_run),
        "hard_total": sum(r["hard_n"] for r in per_run),
        "wall_s": cfg.get("total_wall_s"),
    }
    exists = CSV_PATH.exists()
    with CSV_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(json.dumps(row, indent=1))
    (outdir / "eval_detail.txt").write_text("\n".join(detail_lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
