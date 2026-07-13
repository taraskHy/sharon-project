"""Deterministic text-line segmentation for the Hebrew HTR benchmark.

Pure numpy projection-profile splitting (no ML, no new dependencies): each
explanation-cell crop is binarized (percentile threshold), rows with ink
density above a floor are merged into line bands (min-gap merging), bands
get vertical padding, and each line is saved as its own PNG for inspection
under evaluation/hebrew_bench/segments/<cell_id>/line<N>.png.

Deterministic by construction: same input bytes -> same segments. The
ground truth plays no role here (segmentation is chosen before any
evaluation, on image statistics only).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fitz
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.hebrew_bench_run import _encode_png_gray  # reuse the PNG writer

BENCH = Path("evaluation/hebrew_bench")
SEGROOT = BENCH / "segments"

MIN_LINE_HEIGHT_FRAC = 0.08   # of crop height — reject speck bands
MERGE_GAP_FRAC = 0.035        # gaps smaller than this merge adjacent bands
PAD_FRAC = 0.02


def to_gray(png: bytes) -> np.ndarray:
    pix = fitz.Pixmap(png)
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return arr.mean(axis=2)


def segment_cell(png: bytes) -> list[np.ndarray]:
    gray = to_gray(png)
    h, w = gray.shape
    # Binarize: ink = darker than halfway between the 5th/95th percentiles.
    lo, hi = np.percentile(gray, 5), np.percentile(gray, 95)
    ink = gray < (lo + 0.45 * (hi - lo))
    # Drop border table-rule artifacts: clear 2% frames.
    fr_h, fr_w = int(h * 0.02) + 1, int(w * 0.02) + 1
    ink[:fr_h, :] = ink[-fr_h:, :] = False
    ink[:, :fr_w] = ink[:, -fr_w:] = False
    profile = ink.sum(axis=1) / w
    rows = profile > 0.01
    # Merge bands separated by small gaps.
    bands = []
    start = None
    gap = 0
    max_gap = int(h * MERGE_GAP_FRAC)
    for y in range(h):
        if rows[y]:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                bands.append((start, y - gap))
                start = None
    if start is not None:
        bands.append((start, h - 1))
    pad = int(h * PAD_FRAC) + 1
    out = []
    for y0, y1 in bands:
        if (y1 - y0) < h * MIN_LINE_HEIGHT_FRAC:
            continue
        a, b = max(0, y0 - pad), min(h, y1 + pad)
        out.append(gray[a:b, :])
    return out or [gray]  # never return nothing: fall back to the full cell


def main() -> int:
    manifest = json.loads((BENCH / "crops_manifest.json").read_text(encoding="utf-8"))
    report = {}
    for m in manifest:
        cid = m["id"]
        segs = segment_cell(Path(m["file"]).read_bytes())
        outdir = SEGROOT / cid
        outdir.mkdir(parents=True, exist_ok=True)
        for i, seg in enumerate(segs, 1):
            (outdir / f"line{i}.png").write_bytes(_encode_png_gray(seg))
        report[cid] = len(segs)
    (SEGROOT / "segmentation_report.json").write_text(
        json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
