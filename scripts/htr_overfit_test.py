"""Pre-registered real-data overfit test: can the CRNN+CTC memorize ~20
owner-verified training lines?

PRE-REGISTERED SUCCESS GATE (fixed before running, not to be weakened):
    PASS iff normalized mean line CER <= 0.05
         AND at least 18/20 lines have CER <= 0.10.

Selection is deterministic and saved BEFORE training: the first 20
eligible train-split lines in sorted sample_id order, where eligible =
owner-verified status 'ok' with no partial [לא קריא] span. Flagged,
skipped, blank, unreadable, draft and unannotated lines are excluded.
Val/internal_test data are not touched (val annotations do not exist yet;
the rig's "val" list is the SAME 20 training lines so the trainer's
per-epoch metric IS memorization CER — this rig is for the overfit test
only and is never validated as a real package).

Artifacts under evaluation/htr_overfit_test/:
    selected_ids.json   the 20 ids (written before training)
    pkg/                self-contained rig package (copied line crops)
    ws/                 prepare workspace + model + decodes + train_log.txt
    report.md           full report per the owner's checklist

    .venv-train/Scripts/python.exe scripts/htr_overfit_test.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.hebrew_bench_eval import lev, normalize, word_align
from scripts.htr_annotation_lib import UNREADABLE_TOKEN, load_all_annotations, load_samples

ROOT = Path("evaluation/htr_pilot")
OUT = Path("evaluation/htr_overfit_test")
N_LINES = 20
GATE_MEAN_CER = 0.05
GATE_PER_LINE_CER = 0.10
GATE_MIN_LINES_OK = 18


def run(cmd, log_path: Path | None = None) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd))
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True,
                       encoding="utf-8")
    if log_path:
        log_path.write_text(p.stdout + ("\n--- STDERR ---\n" + p.stderr
                                        if p.stderr.strip() else ""),
                            encoding="utf-8")
    return p


def cer_raw(hyp: str, ref: str) -> float:
    return lev(hyp, ref) / max(len(ref), 1)


def main() -> int:
    t0 = time.monotonic()
    if OUT.exists():
        # OneDrive-synced dirs intermittently deny deletion; contents are
        # rewritten deterministically below, so best-effort cleanup is fine.
        shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True, exist_ok=True)

    # --- deterministic selection, saved before training ---------------------
    samples = {s["sample_id"]: s for s in load_samples(ROOT, "train")}
    ann = load_all_annotations(ROOT, "train")
    eligible = [sid for sid, r in sorted(ann.items())
                if r["status"] == "ok" and r["human_verified"]
                and UNREADABLE_TOKEN not in r["transcription"]]
    picked = eligible[:N_LINES]
    (OUT / "selected_ids.json").write_text(json.dumps({
        "picked": picked, "rule": "first N eligible in sorted sample_id order",
        "n_eligible": len(eligible), "gate": {
            "mean_cer_max": GATE_MEAN_CER,
            "per_line_cer_max": GATE_PER_LINE_CER,
            "min_lines_within": GATE_MIN_LINES_OK},
        "at": time.strftime("%F %T")}, indent=1), encoding="utf-8")
    print(f"selected {len(picked)} of {len(eligible)} eligible lines")

    # --- rig package (train == val == the 20 lines) -------------------------
    pkg = OUT / "pkg"
    for split in ("train", "val", "internal_test"):
        (pkg / "splits").mkdir(parents=True, exist_ok=True)
        (pkg / "annotations" / split).mkdir(parents=True, exist_ok=True)
        recs = []
        if split != "internal_test":
            for sid in picked:
                s = json.loads(json.dumps(samples[sid]))
                s["split"] = split
                recs.append(s)
                rec = json.loads(json.dumps(ann[sid]))
                rec["split"] = split
                (pkg / "annotations" / split / f"{sid}.json").write_text(
                    json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        (pkg / "splits" / f"{split}.json").write_text(
            json.dumps(recs, ensure_ascii=False), encoding="utf-8")
    for sid in picked:
        for rel in samples[sid]["images"].values():
            dst = pkg / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copyfile(ROOT / rel, dst)

    # --- prepare (no augmentation: pure memorization) ------------------------
    ws = OUT / "ws"
    p = run([sys.executable, "-X", "utf8", "scripts/htr_train_prepare.py",
             "--root", pkg, "--out", ws, "--aug", "0"])
    assert p.returncode == 0, p.stdout + p.stderr
    prep = json.loads((ws / "prepare_summary.json").read_text(encoding="utf-8"))

    # --- train ---------------------------------------------------------------
    p = run([sys.executable, "-X", "utf8", "scripts/htr_pilot_train.py",
             "--workspace", ws, "train", "--epochs", "900",
             "--batch-size", "8", "--patience", "150", "--min-epochs", "500"],
            log_path=ws / "train_log.txt")
    assert p.returncode == 0, (p.stdout + p.stderr)[-2000:]
    epochs = [ln for ln in p.stdout.splitlines() if ln.startswith("epoch")]
    best_line = [ln for ln in p.stdout.splitlines() if ln.startswith("best val CER")]
    trainer_best = float(best_line[0].split()[3]) if best_line else float("nan")

    # --- decode with the RELOADED checkpoint ---------------------------------
    p = run([sys.executable, "-X", "utf8", "scripts/htr_pilot_train.py",
             "--workspace", ws, "decode", "--split", "val",
             "--out", "decodes/overfit.txt"])
    assert p.returncode == 0, p.stdout + p.stderr
    hyps = {}
    for line in (ws / "decodes/overfit.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) >= 2:
            hyps[parts[0]] = (float(parts[1]), parts[2] if len(parts) > 2 else "")

    # --- per-line metrics -----------------------------------------------------
    rows = []
    for sid in picked:
        gt = ann[sid]["transcription"]
        conf, pred = hyps.get(sid, (0.0, ""))
        n_gt, n_pred = normalize(gt), normalize(pred)
        rows.append({
            "sid": sid, "gt": gt, "pred": pred, "conf": conf,
            "cer_raw": cer_raw(pred, gt),
            "cer_norm": cer_raw(n_pred, n_gt),
            "cer_norm_reversed": cer_raw(n_pred[::-1], n_gt),
            "wer": (lambda s, d, i, n: (s + d + i) / max(n, 1))(
                *word_align(n_gt.split(), n_pred.split()), len(n_gt.split())),
        })
    mean = lambda k: sum(r[k] for r in rows) / len(rows)  # noqa: E731
    n_within = sum(r["cer_norm"] <= GATE_PER_LINE_CER for r in rows)
    gate_pass = mean("cer_norm") <= GATE_MEAN_CER and n_within >= GATE_MIN_LINES_OK

    # --- diagnostics ----------------------------------------------------------
    charset = sorted({c for sid in picked for c in ann[sid]["transcription"]})
    syms = {ln.split(" ", 1)[0] for ln in
            (ws / "syms.txt").read_text(encoding="utf-8").splitlines()}
    missing_chars = [c for c in charset if c != " " and c not in syms]
    widths, len_fail = [], []
    text_table = {ln.split(" ", 1)[0]: ln.split(" ")[1:] for ln in
                  (ws / "text/train.txt").read_text(encoding="utf-8").splitlines()}
    for sid in picked:
        img = cv2.imread(str(ws / "imgs/train" / f"{sid}.png"), cv2.IMREAD_GRAYSCALE)
        widths.append(img.shape[1])
        if img.shape[1] // 4 < len(text_table[sid]):  # trainer subsamples W/4
            len_fail.append(sid)
    empties = [r["sid"] for r in rows if not r["pred"]]
    ckpt_delta = abs(trainer_best - mean("cer_norm"))

    # --- report ---------------------------------------------------------------
    L = [
        "# Real-data overfit test — CRNN+CTC on 20 verified training lines\n",
        f"Date: {time.strftime('%F %T')}. Pre-registered gate: mean normalized "
        f"line CER <= {GATE_MEAN_CER} AND >= {GATE_MIN_LINES_OK}/{N_LINES} lines "
        f"with CER <= {GATE_PER_LINE_CER}.\n",
        f"## VERDICT: **{'PASS' if gate_pass else 'FAIL'}** — mean CER "
        f"{mean('cer_norm'):.4f} (raw {mean('cer_raw'):.4f}), "
        f"{n_within}/{N_LINES} lines within {GATE_PER_LINE_CER}.\n",
        "## Setup",
        f"- {len(picked)} lines picked deterministically from {len(eligible)} "
        "eligible (selected_ids.json written before training).",
        "- No augmentation; train == val == the 20 lines (memorization rig).",
        f"- prepare: {json.dumps(prep['splits']['train'])}",
        f"- epochs run: {len(epochs)}; trainer best (in-memory) CER "
        f"{trainer_best:.4f}.\n",
        "## Loss / memorization-CER by epoch (every 25th + last)",
        "```",
        *[ln for i, ln in enumerate(epochs)
          if i < 3 or i % 25 == 0 or i >= len(epochs) - 3],
        "```\n",
        "## Aggregates",
        f"- final training CER: normalized {mean('cer_norm'):.4f}, raw "
        f"{mean('cer_raw'):.4f}; WER {mean('wer'):.4f}.",
        f"- RTL order: mean CER of REVERSED predictions {mean('cer_norm_reversed'):.3f} "
        f"vs {mean('cer_norm'):.4f} correct-order — reversed must be far worse.",
        f"- mean confidence {mean('conf'):.4f} "
        f"(min {min(r['conf'] for r in rows):.4f}).",
        f"- checkpoint save/reload: trainer-best {trainer_best:.4f} vs reloaded-"
        f"decode {mean('cer_norm'):.4f} (|delta| {ckpt_delta:.4f}).",
        f"- vocabulary: {len(charset)} distinct label chars; missing from syms: "
        f"{missing_chars or 'none'}; lines skipped for unknown chars: "
        f"{prep['splits']['train'].get('skipped_unknown_char_lines', 0)}.",
        f"- image widths after h=128 resize: min {min(widths)}, max {max(widths)} px; "
        f"CTC length violations (frames < symbols): {len_fail or 'none'}.",
        f"- empty decodes: {empties or 'none'}.\n",
        "## Per-line: prediction vs ground truth",
        "| line | CER | conf | text (GT then PRED) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        L.append(f"| {r['sid']} | {r['cer_norm']:.3f} | {r['conf']:.3f} | "
                 f"GT: {r['gt']} |")
        L.append(f"| | | | PR: {r['pred']} |")
    L.append(f"\nTotal wall time {time.monotonic() - t0:.0f}s.\n")
    (OUT / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:28]))
    print(f"-> {OUT / 'report.md'}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
