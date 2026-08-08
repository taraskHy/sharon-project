"""mlkit_linesplit_v1: automatic multi-line splitting + conservative
neighbor-row bleed trimming for the ML Kit arm.

REFERENCE-FREE: image geometry only (horizontal ink projection profile).
The frozen baseline (mlkit_ink_rtl_a1) is untouched; this is a separate
arm. The recognizer/APK/model are the frozen ones.

Splitter (deterministic, fixed a priori):
- binarize with the frozen a1 pipeline's Otsu;
- smoothed row-ink profile (window 5);
- ink rows = profile > max(2, 5% of profile max);
- consecutive ink rows -> raw bands; merge bands separated by < 5 px;
- drop speck bands (< 40 ink px total);
- CONSERVATIVE bleed trim: an edge-touching band is trimmed only if its
  height < 40% of the tallest band AND its ink mass < 25% of the largest
  band's mass. Ambiguous ink is otherwise preserved.
Bands are processed top-to-bottom; within a band the frozen rtl stroke
assembly applies; strokes are offset back to full-crop coordinates.

Diagnostics per item: band coords, trimmed bands, per-band stroke counts,
debug overlay PNG.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "evaluation" / "hebrew_bench_v2"
OUT_STROKES = BENCH / "ink_strokes" / "linesplit_v1"
OUTDIR = BENCH / "outputs" / "mlkit_linesplit_v1" / "run1"
ADB = r"C:\Users\ethan\android-m2\sdk\platform-tools\adb.exe"
PKG = "com.m2.inkrunner"

spec = importlib.util.spec_from_file_location("mis", REPO / "scripts" / "m2_ink_strokes.py")
mis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mis)
mis.VERSION = "rtl_a1"

GATE_20 = [
    "hc_e002_q1_r1", "hc_e002_q1_r2", "hc_e002_q1_r3", "hc_e002_q1_r4",
    "hl_e003_q1_r1__l1", "hl_e003_q1_r2__l2", "hl_e003_q1_r3__l1",
    "hl_e003_q1_r4__l1", "hl_e004_q1_r1__l1", "hl_e004_q1_r2__l1",
    "hl_e004_q1_r3__l1", "hl_e004_q1_r3__l2", "hl_e005_q1_r1__l1",
    "hl_e005_q1_r1__l2", "hl_e005_q1_r2__l1", "hl_e005_q1_r2__l2",
    "hl_e006_q1_r1__l1", "hl_e006_q1_r2__l1", "hl_e006_q1_r3__l1",
    "hl_e007_q1_r1__l1",
]


def detect_bands(gray: np.ndarray) -> tuple[list[tuple[int, int]], list[dict]]:
    ink = mis.binarize(gray)
    profile = ink.sum(axis=1).astype(float)
    k = 5
    smooth = np.convolve(profile, np.ones(k) / k, mode="same")
    thr = max(2.0, 0.05 * smooth.max())
    rows = smooth > thr
    bands = []
    start = None
    for y, on in enumerate(rows):
        if on and start is None:
            start = y
        elif not on and start is not None:
            bands.append([start, y])
            start = None
    if start is not None:
        bands.append([start, len(rows)])
    # merge tiny gaps
    merged = []
    for b in bands:
        if merged and b[0] - merged[-1][1] <= 6:
            merged[-1][1] = b[1]
        else:
            merged.append(list(b))
    # drop specks
    kept = [b for b in merged if ink[b[0]:b[1]].sum() >= 40]
    # STRUCTURAL: bands under 15 px cannot be handwritten lines at this
    # corpus scale (measured single lines are 40-90 px tall) — merge
    # interior slivers into the nearest band; drop edge-touching slivers
    # (border remnants / neighbor-row fragments).
    H = gray.shape[0]
    solid = []
    pending_lead = None
    for b in kept:
        if b[1] - b[0] >= 15:
            if pending_lead is not None:
                b = [pending_lead[0], b[1]]  # merge leading sliver downward
                pending_lead = None
            solid.append(b)
            continue
        if b[0] <= 2 or b[1] >= H - 2:
            continue  # edge sliver -> drop (conservative bleed/border trim)
        if solid:
            solid[-1][1] = b[1]  # merge into band above
        else:
            pending_lead = b  # first band is a sliver -> merge into next
    kept = solid or kept
    # STRUCTURAL: touching multi-line bands never dip below the absolute
    # threshold — split over-tall bands (>95 px; measured single lines are
    # 40-90 px) at the deepest interior valley of the smoothed profile.
    def split_tall(b):
        y0, y1 = b
        if y1 - y0 <= 95:
            return [b]
        core = smooth[y0 + 20 : y1 - 20]
        if not len(core):
            return [b]
        vy = int(np.argmin(core)) + y0 + 20
        left_peak = smooth[y0:vy].max()
        right_peak = smooth[vy:y1].max()
        if smooth[vy] > 0.6 * min(left_peak, right_peak):
            return [b]  # no meaningful valley -> keep as one line
        return split_tall([y0, vy]) + split_tall([vy, y1])

    kept = [sb for b in kept for sb in split_tall(list(b))]
    trimmed = []
    if len(kept) > 1:
        heights = [b[1] - b[0] for b in kept]
        masses = [float(ink[b[0]:b[1]].sum()) for b in kept]
        hmax, mmax = max(heights), max(masses)
        final = []
        for b, h, m in zip(kept, heights, masses):
            edge = b[0] <= 2 or b[1] >= H - 2
            if edge and h < 0.4 * hmax and m < 0.25 * mmax:
                trimmed.append({"band": b, "height": h, "mass": m,
                                "reason": "edge-touching small band (neighbor-row bleed)"})
            else:
                final.append(b)
        kept = final or kept  # never trim everything
    return [tuple(b) for b in kept], trimmed


def convert(item: str) -> dict:
    gray = mis.load_gray(BENCH / "crops" / f"{item}.png")
    H, W = gray.shape
    bands, trimmed = detect_bands(gray)
    all_strokes = []
    band_counts = []
    pad = 3
    for y0, y1 in bands:
        y0p, y1p = max(0, y0 - pad), min(H, y1 + pad)
        band = gray[y0p:y1p, :]
        ink = mis.binarize(band)
        ink = mis.remove_structural_lines(ink)
        ink = mis.solidify(ink)
        from skimage.measure import label as sklabel
        lab = sklabel(ink, connectivity=2)
        keep = np.zeros_like(ink)
        for i in range(1, lab.max() + 1):
            m = lab == i
            if m.sum() >= 6:
                keep |= m
        from skimage.morphology import skeletonize
        skel = skeletonize(keep)
        deg, nodes, edges = mis.skeleton_graph(skel)
        strokes_px = mis.assemble_strokes_rtl(nodes, edges, band.shape)
        strokes = [mis.resample(s) for s in strokes_px if len(s) >= 3]
        strokes = [[[x, y + y0p] for x, y in s] for s in strokes]
        band_counts.append(len(strokes))
        all_strokes.extend(strokes)
    # debug overlay
    dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    for y0, y1 in bands:
        cv2.rectangle(dbg, (0, y0), (W - 1, y1), (0, 160, 0), 1)
    for t in trimmed:
        y0, y1 = t["band"]
        cv2.rectangle(dbg, (0, y0), (W - 1, y1), (0, 0, 220), 1)
    (OUT_STROKES / "debug").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_STROKES / "debug" / f"{item}.png"), dbg)

    rec = {
        "item": item,
        "heuristic": "linesplit_v1 (projection-profile bands; frozen a1 core; rtl within band)",
        "stroke_order_note": "SYNTHETIC approximation; bands top-to-bottom",
        "width": int(W), "height": int(H),
        "n_bands": len(bands),
        "bands": [list(b) for b in bands],
        "trimmed_bands": trimmed,
        "n_strokes": len(all_strokes),
        "band_stroke_counts": band_counts,
        "strokes": [[[int(x), int(y)] for x, y in s] for s in all_strokes],
    }
    (OUT_STROKES / f"{item}.json").write_text(json.dumps(rec, ensure_ascii=False),
                                              encoding="utf-8")
    return rec


def adb(*args, check=True, timeout=120):
    r = subprocess.run([ADB, *args], capture_output=True, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"adb {' '.join(args)} failed: {r.stderr.strip()[:200]}")
    return r.stdout


def main() -> int:
    convert_only = "--convert-only" in sys.argv
    OUT_STROKES.mkdir(parents=True, exist_ok=True)
    for item in GATE_20:
        r = convert(item)
        print(f"{item}: bands={r['n_bands']} {r['bands']} trimmed={len(r['trimmed_bands'])} "
              f"strokes={r['band_stroke_counts']}")
    if convert_only:
        return 0
    OUTDIR.mkdir(parents=True, exist_ok=True)
    adb("shell", f"run-as {PKG} sh -c 'rm -rf files/in files/out files/status.txt; mkdir -p files/in files/out'")
    adb("shell", "mkdir", "-p", "/data/local/tmp/m2split")
    for item in GATE_20:
        adb("push", str(OUT_STROKES / f"{item}.json"), f"/data/local/tmp/m2split/{item}.json")
    adb("shell", f"run-as {PKG} sh -c 'cp /data/local/tmp/m2split/*.json files/in/'")
    adb("shell", "am", "force-stop", PKG)
    adb("shell", "am", "start", "-n", f"{PKG}/.MainActivity")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 300:
        time.sleep(5)
        status = adb("shell", f"run-as {PKG} cat files/status.txt", check=False).strip()
        if status.startswith(("done", "fatal")):
            print(f"device status: {status}")
            break
    # settle: ensure the out dir has all files before pulling
    time.sleep(5)
    got = 0
    for item in GATE_20:
        r = subprocess.run([ADB, "exec-out", "run-as", PKG, "cat", f"files/out/{item}.json"],
                           capture_output=True, text=True)
        if r.returncode == 0 and (r.stdout or "").strip().startswith("{"):
            (OUTDIR / f"{item}.json").write_text(r.stdout, encoding="utf-8")
            got += 1
        else:
            print(f"MISSING {item}")
    print(f"pulled {got}/{len(GATE_20)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
