"""ML Kit ORACLE-SEGMENTATION diagnostic (Mission 2).

DIAGNOSTIC ONLY — never a production pipeline. Question: how much of ML
Kit's poor handwriting result is preprocessing/segmentation vs the
recognizer itself?

Manual interventions below were chosen by VISUAL inspection of the crops
(documented per item; fractional coordinates). The reference transcription
was NOT consulted for any intervention — it is used only for post-hoc CER
scoring. Interventions are limited to: erasing obvious non-text junk
(strike-out scribbles, neighbor-row bleed, marginal annotations, cancelled
struck lines) and declaring the true line bands. The handwriting itself is
never altered. Stroke generation reuses the FROZEN a1 converter functions
per line band (bands processed top-to-bottom, rtl ordering within a band),
and recognition uses the unchanged frozen APK/model on the emulator.

Outputs: evaluation/hebrew_bench_v2/ink_strokes/oracle_seg/<item>.json
         evaluation/hebrew_bench_v2/outputs/mlkit_ink_oracle_seg/run1/
Frozen artifacts are never modified.
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
OUT_STROKES = BENCH / "ink_strokes" / "oracle_seg"
OUTDIR = BENCH / "outputs" / "mlkit_ink_oracle_seg" / "run1"
ADB = r"C:\Users\ethan\android-m2\sdk\platform-tools\adb.exe"
PKG = "com.m2.inkrunner"

spec = importlib.util.spec_from_file_location("mis", REPO / "scripts" / "m2_ink_strokes.py")
mis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mis)
mis.VERSION = "rtl_a1"  # frozen a1 pipeline behavior for binarize+clean

# --------------------------------------------------------------------------
# MANUAL INTERVENTIONS (visual inspection; fractions of width/height).
# erase: rectangles [x0, y0, x1, y1] set to white.  bands: y-ranges of the
# TRUE handwritten lines, processed top-to-bottom.  Items with erase=[] and
# bands=[[0,1]] are pass-through CONTROLS.
# --------------------------------------------------------------------------
INTERVENTIONS: dict[str, dict] = {
    # struck-through cell: line 1 entirely cancelled by the student (heavy
    # strike scribbles); diagonal slashes intrude into line 2's right side;
    # printed border remnants at edges. Keep only the readable second line.
    "hc_e002_q1_r1": {
        "erase": [[0.0, 0.0, 1.0, 0.52], [0.72, 0.52, 1.0, 1.0]],
        "bands": [[0.52, 1.0]],
        "note": "line1 = cancelled struck text (removed whole); right-corner "
                "diagonal strikes removed; keeps readable tail line only",
    },
    # two written lines; scribbled-out word in line 2 left-center
    "hl_e006_q1_r1__l1": {
        "erase": [[0.27, 0.50, 0.42, 1.0]],
        "bands": [[0.0, 0.50], [0.50, 1.0]],
        "note": "2-line crop split; struck word in line2 removed",
    },
    # two clean lines
    "hl_e007_q1_r1__l1": {
        "erase": [],
        "bands": [[0.0, 0.52], [0.52, 1.0]],
        "note": "2-line crop split only",
    },
    "hl_e003_q1_r1__l1": {
        "erase": [],
        "bands": [[0.0, 0.50], [0.50, 1.0]],
        "note": "2-line crop split only",
    },
    # 2 lines + heavy scribble blob + next-row bleed at bottom
    "hl_e003_q1_r2__l2": {
        "erase": [[0.36, 0.42, 0.55, 0.78], [0.0, 0.85, 1.0, 1.0]],
        "bands": [[0.0, 0.44], [0.44, 0.85]],
        "note": "scribbled-out word removed; bottom next-row bleed removed; "
                "2 lines split",
    },
    # previous-row bleed at top + 2 lines
    "hl_e003_q1_r3__l1": {
        "erase": [[0.0, 0.0, 1.0, 0.17]],
        "bands": [[0.17, 0.60], [0.60, 1.0]],
        "note": "top bleed removed; 2 lines split",
    },
    "hl_e003_q1_r4__l1": {
        "erase": [],
        "bands": [[0.0, 0.50], [0.50, 1.0]],
        "note": "2-line crop split only",
    },
    # 2 lines; struck word in line 1 right-of-center
    "hl_e006_q1_r2__l1": {
        "erase": [[0.57, 0.0, 0.69, 0.55]],
        "bands": [[0.0, 0.55], [0.55, 1.0]],
        "note": "struck word in line1 removed; 2 lines split",
    },
    # single line + tiny interlinear annotation top-right
    "hl_e005_q1_r2__l2": {
        "erase": [[0.74, 0.0, 1.0, 0.34]],
        "bands": [[0.0, 1.0]],
        "note": "top-right interlinear annotation removed",
    },
    "hl_e005_q1_r1__l2": {
        "erase": [[0.86, 0.0, 1.0, 0.36]],
        "bands": [[0.0, 1.0]],
        "note": "top-right fragment removed",
    },
    # CONTROLS — no intervention
    "hl_e004_q1_r1__l1": {"erase": [], "bands": [[0.0, 1.0]], "note": "control"},
    "hl_e004_q1_r3__l1": {"erase": [], "bands": [[0.0, 1.0]], "note": "control"},
    "hl_e005_q1_r2__l1": {"erase": [], "bands": [[0.0, 1.0]], "note": "control"},
    "hl_e006_q1_r3__l1": {"erase": [], "bands": [[0.0, 1.0]], "note": "control"},
}


def convert_oracle(item: str, spec_: dict) -> dict:
    gray = mis.load_gray(BENCH / "crops" / f"{item}.png")
    H, W = gray.shape
    cleaned = gray.copy()
    for x0, y0, x1, y1 in spec_["erase"]:
        cleaned[int(y0 * H):int(y1 * H), int(x0 * W):int(x1 * W)] = 255
    (OUT_STROKES / "cleaned").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT_STROKES / "cleaned" / f"{item}.png"), cleaned)

    all_strokes = []
    band_stroke_counts = []
    for y0f, y1f in spec_["bands"]:
        y0, y1 = int(y0f * H), int(y1f * H)
        band = cleaned[y0:y1, :]
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
        # shift back to full-crop coordinates
        strokes = [[[x, y + y0] for x, y in s] for s in strokes]
        band_stroke_counts.append(len(strokes))
        all_strokes.extend(strokes)
    rec = {
        "item": item,
        "heuristic": "oracle_seg (manual bands + junk erasure; frozen a1 core)",
        "stroke_order_note": "SYNTHETIC approximation; bands top-to-bottom",
        "intervention": spec_,
        "width": int(W), "height": int(H),
        "n_strokes": len(all_strokes),
        "band_stroke_counts": band_stroke_counts,
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
    OUT_STROKES.mkdir(parents=True, exist_ok=True)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for item, spec_ in INTERVENTIONS.items():
        r = convert_oracle(item, spec_)
        print(f"{item}: bands={r['band_stroke_counts']} total={r['n_strokes']} "
              f"({spec_['note'][:50]})")

    # device round-trip with the FROZEN app (clear scratch dirs first;
    # frozen outputs already persisted on disk)
    adb("shell", f"run-as {PKG} sh -c 'rm -rf files/in files/out files/status.txt; mkdir -p files/in files/out'")
    adb("shell", "mkdir", "-p", "/data/local/tmp/m2oracle")
    for item in INTERVENTIONS:
        adb("push", str(OUT_STROKES / f"{item}.json"), f"/data/local/tmp/m2oracle/{item}.json")
    adb("shell", f"run-as {PKG} sh -c 'cp /data/local/tmp/m2oracle/*.json files/in/'")
    adb("shell", "am", "force-stop", PKG)
    adb("shell", "am", "start", "-n", f"{PKG}/.MainActivity")
    t0 = time.monotonic()
    while time.monotonic() - t0 < 300:
        time.sleep(5)
        status = adb("shell", f"run-as {PKG} cat files/status.txt", check=False).strip()
        if status.startswith(("done", "fatal")):
            print(f"device status: {status}")
            break
    for item in INTERVENTIONS:
        r = subprocess.run([ADB, "exec-out", "run-as", PKG, "cat", f"files/out/{item}.json"],
                           capture_output=True, text=True)
        if r.returncode == 0 and (r.stdout or "").strip().startswith("{"):
            (OUTDIR / f"{item}.json").write_text(r.stdout, encoding="utf-8")
            print(f"pulled {item}")
        else:
            print(f"MISSING {item}: {(r.stderr or r.stdout or '')[:80]}")
    print("oracle run complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
