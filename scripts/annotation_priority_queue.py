"""Deterministic annotation priority queues for UNTOUCHED train samples.

Uses ONLY model-visible, non-ground-truth signals: image statistics of
the line crop, split-file geometry, the CRNN candidate decode
(confidence + decoded text — candidate text, never a label), and, where
Phase-2 queue candidates exist, A<->B agreement. No annotation
transcription is ever read; annotation files are consulted only to
learn WHICH samples are untouched.

    prep    write untouched line images + list into the CRNN decode ws
    build   compute signals + ranks -> evaluation/annotation_priority_queue.csv

Between the two, decode with the overfit checkpoint:
    .venv-train/Scripts/python.exe scripts/htr_pilot_train.py \
        --workspace evaluation/htr_candidates/crnn_ws decode \
        --split untouched --out decodes/untouched.txt --device cpu

Queues (fixed weights, ties broken by sample_id — fully deterministic):
- easy_rank    annotate these first for cheap verified lines
- info_rank    active-learning value: hard-for-model but well-segmented
- recrop_flag  segmentation-repair queue (inspect geometry before typing)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.htr_annotation_lib import load_all_annotations, load_samples  # noqa: E402
from scripts.htr_train_prepare import load_line_gray  # noqa: E402

ROOT = Path("evaluation/htr_pilot")
CRNN_WS = Path("evaluation/htr_candidates/crnn_ws")
CSV_PATH = Path("evaluation/annotation_priority_queue.csv")
LATIN_DIGIT = re.compile(r"[A-Za-z0-9]")


def untouched_samples() -> list[dict]:
    samples = load_samples(ROOT, "train")
    ann = load_all_annotations(ROOT, "train")
    return [s for s in samples if s["sample_id"] not in ann]


def image_signals(path: Path) -> dict:
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape
    lo, hi = np.percentile(gray, 5), np.percentile(gray, 95)
    contrast = float(hi - lo)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    _t, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if contrast < 20:  # near-blank crop: Otsu on noise is meaningless
        ink[:] = 0
    ink_frac = float(ink.mean() / 255.0)
    rows = (ink > 0).mean(axis=1)
    edge_touch = float(rows[:3].max() if h >= 3 else 0) + \
        float(rows[-3:].max() if h >= 3 else 0)
    # text bands: runs of rows with >4% ink — >1 band suggests a merged crop
    band_rows = rows > 0.04
    n_bands = int(np.diff(band_rows.astype(int), prepend=0).clip(min=0).sum())
    # strike-through proxy: longest horizontal ink run relative to width
    longest_run = 0
    if ink_frac > 0.001:
        core = ink[int(h * 0.25):int(h * 0.75)]
        for r in core[:: max(1, core.shape[0] // 16)]:
            runs = np.diff(np.flatnonzero(np.diff(
                np.concatenate(([0], (r > 0).astype(int), [0])))).reshape(-1, 2),
                axis=1)
            if len(runs):
                longest_run = max(longest_run, int(runs.max()))
    return {
        "width": w, "height": h, "contrast": round(contrast, 1),
        "blur_var": round(blur, 1), "ink_frac": round(ink_frac, 4),
        "edge_touch": round(edge_touch, 3), "n_bands": n_bands,
        "strike_run_frac": round(longest_run / max(w, 1), 3),
    }


def cmd_prep(args) -> int:
    samples = untouched_samples()
    (CRNN_WS / "imgs" / "untouched").mkdir(parents=True, exist_ok=True)
    ids = []
    for s in samples:
        sid = s["sample_id"]
        cv2.imwrite(str(CRNN_WS / "imgs" / "untouched" / f"{sid}.png"),
                    load_line_gray(ROOT / s["images"]["line"]))
        ids.append(sid)
    (CRNN_WS / "lists" / "untouched.txt").write_text(
        "\n".join(ids) + "\n", encoding="utf-8")
    print(f"{len(ids)} untouched lines -> {CRNN_WS}")
    return 0


def cmd_build(args) -> int:
    samples = untouched_samples()
    dec_path = CRNN_WS / "decodes" / "untouched.txt"
    decodes: dict[str, tuple[float, str]] = {}
    if dec_path.exists():
        for line in dec_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                sid, conf, text = (line.split(" ", 2) + ["", ""])[:3]
                decodes[sid] = (float(conf), text.strip())

    rows = []
    for s in samples:
        sid = s["sample_id"]
        sig = image_signals(ROOT / s["images"]["line"])
        conf, text = decodes.get(sid, (None, ""))
        mixed = bool(LATIN_DIGIT.search(text)) if text else False
        blankish = sig["ink_frac"] < 0.004 or s.get("expected_blank", False)
        # segmentation-quality score in [0,1]; 1 = clean
        seg = 1.0
        seg -= 0.35 * min(sig["edge_touch"], 1.0)
        seg -= 0.30 * (sig["n_bands"] > 1)
        seg -= 0.20 * (sig["height"] > 320 or sig["height"] < 40)
        seg -= 0.15 * (sig["strike_run_frac"] > 0.35)
        seg = max(seg, 0.0)
        conf_v = conf if conf is not None else 0.0
        norm_w = min(sig["width"] / 2000.0, 1.0)
        clarity = min(sig["contrast"] / 120.0, 1.0)
        easy = (0.35 * conf_v + 0.25 * seg + 0.15 * clarity
                + 0.15 * (1.0 - norm_w) + 0.10 * (not mixed))
        info = (0.40 * (1.0 - conf_v) + 0.25 * seg + 0.20 * norm_w
                + 0.15 * mixed)
        if blankish:  # blanks are one-click confirms: easy, zero info value
            easy, info = easy + 0.5, 0.0
        recrop = seg < 0.55
        rows.append({
            "sample_id": sid, "writer": s["writer"],
            "question": s["question"], "row": s["row"],
            "line_index": s["line_index"], "expected_blank": s["expected_blank"],
            **sig, "crnn_conf": None if conf is None else round(conf, 4),
            "mixed_script_in_decode": mixed, "blankish": blankish,
            "seg_quality": round(seg, 3),
            "candidate_agreement": None,  # filled only if Phase-2 queue ran
            "easy_score": round(easy, 4), "info_score": round(info, 4),
            "recrop_flag": recrop,
        })

    rows_easy = sorted(rows, key=lambda r: (-r["easy_score"], r["sample_id"]))
    for i, r in enumerate(rows_easy, 1):
        r["easy_rank"] = i
    rows_info = sorted(rows, key=lambda r: (-r["info_score"], r["sample_id"]))
    for i, r in enumerate(rows_info, 1):
        r["info_rank"] = i
    recrop_rows = sorted([r for r in rows if r["recrop_flag"]],
                         key=lambda r: (r["seg_quality"], r["sample_id"]))
    for i, r in enumerate(recrop_rows, 1):
        r["recrop_rank"] = i
    for r in rows:
        r.setdefault("recrop_rank", "")

    rows.sort(key=lambda r: r["easy_rank"])
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({
        "untouched": len(rows), "recrop_queue": len(recrop_rows),
        "blankish": sum(r["blankish"] for r in rows),
        "with_crnn_conf": sum(r["crnn_conf"] is not None for r in rows),
        "csv": str(CSV_PATH), "at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prep")
    sub.add_parser("build")
    args = ap.parse_args()
    return {"prep": cmd_prep, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
