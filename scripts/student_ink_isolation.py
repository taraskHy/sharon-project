"""Student-ink isolation for the 16 verified exam-002 explanation cells.

Hypothesis under test: recognizers fail partly because crops still contain
printed table structure; a registered blank-template subtraction plus
blue-ink isolation may yield a cleaner handwriting image.

No blank answer-sheet template exists in the repo (the born-digital
Exam_solution.pdf holds only the question booklet, not the answer tables),
so the blank template is SYNTHESIZED: the per-pixel median of N other dev
exams' scans of the same page, each red-masked and ECC-registered onto exam
002's own page frame. Printed structure is common to every exam and
survives the median; per-student handwriting occupies different pixels per
exam and drops out. Held-out exams are never touched (test/ = the 41
sanctioned dev exams).

Outputs per cell (evaluation/student_ink_isolation_artifacts/):
  A original crop        (the existing bench crop, red-masked at source)
  B aligned template crop
  C absolute-difference image (inverted for viewing)
  D final student-ink mask
  E1 blue-ink-only image      (colour separation alone, no template)
  E2 template-subtracted image (blue OR dark-where-template-clean)
  F line-segment crops from E2

All thresholds are frozen from image/colour statistics BEFORE any
recognition and are never tuned against transcription metrics:

- T_BLUE = 25: blue-dominance b - max(r,g). Measured on exam-002 p11 at
  2200 px: paper pixels P99 = 12, P99.9 = 38 (JPEG fringing on print
  edges); blue ink core is far above. 25 sits above paper noise on the
  20-30 plateau of the dark-pixel dominance curve.
- T_dark: per-page Otsu on the grey histogram (measured 160 on e002 p11),
  clamped to [100, 200] with a percentile fallback — same family as the
  repo's existing 5/95-percentile binarisation.
- Red exclusion: the production rule from autograder.masking
  (r - max(g,b) > 50 and r > 70), dilated 2 px.
- Template print mask dilated 2 px to absorb residual registration error.
- T_TEXTURE = 200 on a 9 px box-blurred template: dithered print fill (the
  table header's speckle band) medians into a texture too light for the
  per-pixel dark test, but its local mean (P50 = 188 measured on the q2
  template) sits well below paper (page-level local-mean P1 = 207). Any
  locally non-white template zone is treated as print; blue strokes there
  are still kept by the blue term.
- Despeckle: components < 8 px (stroke width at this zoom is 3-5 px; dust
  specks are 1-4 px).
- Final mask dilated 1 px to keep anti-aliased stroke edges; only ORIGINAL
  scan pixels are copied — nothing is reconstructed, sharpened or redrawn.

Line segmentation (F) reuses the constants of scripts/segment_lines.py
(row-profile > 0.01 of width, merge gap 3.5 %, min height 8 %, pad 2 %)
computed on the ink MASK instead of a grey-percentile binarisation.

Usage:  .venv/Scripts/python.exe scripts/student_ink_isolation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import fitz
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ART = Path("evaluation/student_ink_isolation_artifacts")
BENCH = Path("evaluation/hebrew_bench")

ZOOM_WIDTH = 2200  # matches gen_hebrew_bench.py so crops align with the bench

# Geometry copied verbatim from gen_hebrew_bench.py (e002_q1) and
# add_e002_q2_bench.py (e002_q2) — the recipes that produced the bench crops.
SHEETS = {
    "e002_q1": {
        "pdf": "test/002_76.pdf", "page": 11,
        "x": (0.06, 0.645),
        "rows": {
            1: (0.355, 0.434), 2: (0.426, 0.497), 3: (0.488, 0.562),
            4: (0.553, 0.625), 5: (0.616, 0.687), 6: (0.678, 0.747),
            7: (0.738, 0.810), 8: (0.801, 0.880),
        },
    },
    "e002_q2": {
        "pdf": "test/002_76.pdf", "page": 12,
        "x": (0.06, 0.645),
        "rows": {
            1: (0.353, 0.432), 2: (0.424, 0.495), 3: (0.487, 0.556),
            4: (0.548, 0.620), 5: (0.612, 0.681), 6: (0.673, 0.750),
            7: (0.742, 0.815), 8: (0.806, 0.882),
        },
    },
}

# Donor pool for the synthetic blank template: dev exams only, numeric order,
# excluding exam 002 itself (the template must be independent of the page it
# cleans). Registration keeps the first N_KEEP donors whose ECC correlation
# passes CC_MIN; the rest are skipped and logged.
DONORS = [f"test/{i:03d}_{g}.pdf" for i, g in [
    (3, 70), (4, 58), (5, 48), (6, 86), (7, 48), (8, 52), (9, 94), (10, 52),
    (11, 98), (12, 70), (13, 80), (14, 58), (15, 59), (16, 65), (17, 30),
    (18, 92), (19, 44), (20, 60), (21, 86), (22, 76), (23, 62), (24, 54),
    (25, 42), (26, 88), (27, 80), (28, 52), (29, 78), (30, 74), (31, 48),
    (32, 86), (33, 76), (34, 86), (35, 48), (36, 75), (37, 38), (38, 32),
    (39, 76), (40, 44), (41, 68), (42, 86),
]]
N_KEEP = 16
CC_MIN = 0.55

T_BLUE = 25          # blue dominance threshold, from measured page statistics
RED_DOMINANCE = 50   # = autograder.masking.RED_DOMINANCE
MIN_RED_VALUE = 70   # = autograder.masking.MIN_RED_VALUE
OTSU_CLAMP = (100, 200)
TMPL_DILATE = 2      # px, registration slack on the template print mask
T_TEXTURE = 200      # local-mean threshold marking dithered print fill
TEXTURE_BLUR = 9     # px box blur for the local mean
RED_DILATE = 2       # px, red pen halo
INK_DILATE = 1       # px, keep anti-aliased stroke edges
MIN_SPECK = 8        # px, connected components smaller than this are dust

# Line segmentation — constants mirrored from scripts/segment_lines.py.
LINE_PROFILE_MIN = 0.01
MERGE_GAP_FRAC = 0.035
MIN_LINE_HEIGHT_FRAC = 0.08
PAD_FRAC = 0.02


def render_page(pdf: str, page1: int, width: int = ZOOM_WIDTH) -> np.ndarray:
    doc = fitz.open(pdf)
    page = doc[page1 - 1]
    zoom = width / page.rect.width
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, pix.n)[:, :, :3].copy()
    doc.close()
    return arr


def page_count(pdf: str) -> int:
    doc = fitz.open(pdf)
    n = len(doc)
    doc.close()
    return n


def red_mask(arr: np.ndarray) -> np.ndarray:
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    return (r - np.maximum(g, b) > RED_DOMINANCE) & (r > MIN_RED_VALUE)


def whiten_red(arr: np.ndarray) -> np.ndarray:
    out = arr.copy()
    out[red_mask(arr)] = 255
    return out


def gray_of(arr: np.ndarray) -> np.ndarray:
    return arr.mean(axis=2)


def dark_threshold(gray: np.ndarray) -> float:
    t, _ = cv2.threshold(gray.astype(np.uint8), 0, 255,
                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if not (OTSU_CLAMP[0] <= t <= OTSU_CLAMP[1]):
        lo, hi = np.percentile(gray, 5), np.percentile(gray, 95)
        t = lo + 0.45 * (hi - lo)
    return float(t)


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=px).astype(bool)


def despeckle(mask: np.ndarray, min_area: int) -> np.ndarray:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = True
    return out


def ecc_align(ref_s: np.ndarray, mov_s: np.ndarray,
              iters: int) -> tuple[float, np.ndarray]:
    """ECC affine between two same-scale inverted-gray images; phase-corr
    init. Returns (cc, 2x3 warp at that scale)."""
    if mov_s.shape != ref_s.shape:
        mov_s = cv2.resize(mov_s, (ref_s.shape[1], ref_s.shape[0]))
    (dx, dy), _ = cv2.phaseCorrelate(mov_s, ref_s)
    warp = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, 1e-6)
    cc, warp = cv2.findTransformECC(ref_s, mov_s, warp, cv2.MOTION_AFFINE, crit, None, 5)
    return float(cc), warp.copy()


def register(ref_inv: np.ndarray, mov_inv: np.ndarray) -> tuple[float, np.ndarray]:
    """Half-scale ECC; returns (cc, 2x3 warp at FULL resolution)."""
    s = 0.5
    ref_s = cv2.resize(ref_inv, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    mov_s = cv2.resize(mov_inv, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cc, warp = ecc_align(ref_s, mov_s, iters=200)
    warp[:, 2] /= s
    return cc, warp


def inv_gray(rgb: np.ndarray) -> np.ndarray:
    return (255 - gray_of(rgb)).astype(np.float32)


def find_sheet_page(ref_inv: np.ndarray, donor: str,
                    hint: int | None) -> tuple[int, float, np.ndarray, np.ndarray] | None:
    """Locate the donor page showing the same printed sheet as the reference.

    Donor booklets differ in page count and scan order, so the sheet's page
    number is not constant. Try the hinted page at half scale first; else
    score every page with a cheap quarter-scale ECC and refine the winner.
    Returns (page1, cc, warp, red-masked RGB at full zoom) or None.
    """
    n = page_count(donor)

    def full_register(p: int):
        mov_rgb = whiten_red(render_page(donor, p))
        cc, warp = register(ref_inv, inv_gray(mov_rgb))
        return cc, warp, mov_rgb

    if hint is not None and 1 <= hint <= n:
        try:
            cc, warp, mov_rgb = full_register(hint)
            if cc >= CC_MIN:
                return hint, cc, warp, mov_rgb
        except cv2.error:
            pass

    ref_q = cv2.resize(ref_inv, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    scores = []
    for p in range(1, n + 1):
        try:
            mov_q = inv_gray(whiten_red(render_page(donor, p, width=550)))
            cc_q, _ = ecc_align(ref_q, mov_q, iters=80)
        except cv2.error:
            cc_q = -1.0
        scores.append((cc_q, p))
    cc_q, best = max(scores)
    if cc_q < 0.45:
        return None
    try:
        cc, warp, mov_rgb = full_register(best)
    except cv2.error:
        return None
    if cc < CC_MIN:
        return None
    return best, cc, warp, mov_rgb


def build_template(ref_pdf: str, page1: int,
                   hints: dict[str, int] | None = None,
                   ) -> tuple[np.ndarray, list[dict], dict[str, int]]:
    ref_rgb = whiten_red(render_page(ref_pdf, page1))
    h, w = ref_rgb.shape[:2]
    ref_inv = inv_gray(ref_rgb)
    kept, report, found = [], [], {}
    for donor in DONORS:
        if len(kept) >= N_KEEP:
            break
        entry = {"donor": donor, "page": None, "cc": None, "kept": False}
        hit = find_sheet_page(ref_inv, donor, (hints or {}).get(donor))
        if hit is not None:
            page, cc, warp, mov_rgb = hit
            entry.update(page=page, cc=round(cc, 4),
                         warp=np.round(warp, 6).tolist(), kept=True)
            aligned = cv2.warpAffine(
                mov_rgb, cv2.invertAffineTransform(warp), (w, h),
                flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
            kept.append(aligned)
            found[donor] = page
        report.append(entry)
        print(f"  donor {donor}: page={entry['page']} cc={entry['cc']} kept={entry['kept']}")
    if len(kept) < 5:
        raise RuntimeError(f"only {len(kept)} donors registered for {ref_pdf} "
                           f"p{page1}; template would not be robust")
    template = np.median(np.stack(kept, axis=0), axis=0).astype(np.uint8)
    return template, report, found


def cell_rects(pdf: str, page1: int, x: tuple, rows: dict) -> dict[int, tuple]:
    """Pixel rects computed exactly the way gen_hebrew_bench.py's fitz clip
    rendering produced them (Rect * Matrix -> irect)."""
    doc = fitz.open(pdf)
    r = doc[page1 - 1].rect
    zoom = ZOOM_WIDTH / r.width
    out = {}
    for row, (y0, y1) in rows.items():
        clip = fitz.Rect(r.width * x[0], r.height * y0, r.width * x[1], r.height * y1)
        ir = (clip * fitz.Matrix(zoom, zoom)).irect
        out[row] = (ir.x0, ir.y0, ir.x1, ir.y1)
    doc.close()
    return out


def isolate_cell(scan: np.ndarray, tmpl: np.ndarray, t_dark_scan: float,
                 t_dark_tmpl: float) -> dict:
    """All masks for one cell crop. scan/tmpl are red-masked RGB crops."""
    g_scan = gray_of(scan)
    g_tmpl = gray_of(tmpl)
    r = scan[:, :, 0].astype(np.int16)
    g = scan[:, :, 1].astype(np.int16)
    b = scan[:, :, 2].astype(np.int16)
    blue = (b - np.maximum(r, g)) > T_BLUE
    red2 = dilate(red_mask(scan), RED_DILATE)  # residual red edges
    dark = g_scan < t_dark_scan
    texture = cv2.blur(g_tmpl.astype(np.float32), (TEXTURE_BLUR, TEXTURE_BLUR)) < T_TEXTURE
    tdark2 = dilate(g_tmpl < t_dark_tmpl, TMPL_DILATE) | texture

    base_blue = despeckle(blue & ~red2, MIN_SPECK)
    base_full = despeckle((blue | (dark & ~tdark2)) & ~red2, MIN_SPECK)

    def finalize(base: np.ndarray) -> np.ndarray:
        m = dilate(base, INK_DILATE) & ~red2
        return m & (blue | ~tdark2)  # dilation must not re-admit print pixels

    mask_blue = finalize(base_blue)
    mask_full = finalize(base_full)

    def paint(mask: np.ndarray) -> np.ndarray:
        out = np.full_like(scan, 255)
        out[mask] = scan[mask]
        return out

    absdiff = 255 - np.abs(g_scan - g_tmpl).clip(0, 255).astype(np.uint8)
    return {
        "B": tmpl,
        "C": np.dstack([absdiff] * 3),
        "D": np.dstack([np.where(mask_full, 0, 255).astype(np.uint8)] * 3),
        "E1": paint(mask_blue),
        "E2": paint(mask_full),
        "mask_full": mask_full,
    }


def line_bands_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    profile = mask.sum(axis=1) / w
    rows = profile > LINE_PROFILE_MIN
    bands, start, gap = [], None, 0
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
    out = [(max(0, a - pad), min(h, b + pad)) for a, b in bands
           if (b - a) >= h * MIN_LINE_HEIGHT_FRAC]
    return out or [(0, h)]


def save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def label_band(text: str, width: int) -> np.ndarray:
    band = np.full((26, width, 3), 210, np.uint8)
    cv2.putText(band, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return band


def contact_sheet(cid: str, panels: list[tuple[str, np.ndarray]]) -> np.ndarray:
    width = max(p.shape[1] for _, p in panels)
    parts = []
    for name, img in panels:
        if img.shape[1] < width:
            pad = np.full((img.shape[0], width - img.shape[1], 3), 255, np.uint8)
            img = np.hstack([img, pad])
        parts.append(label_band(f"{cid}  {name}", width))
        parts.append(img)
    return np.vstack(parts)


def main() -> int:
    t0 = time.monotonic()
    ART.mkdir(parents=True, exist_ok=True)
    registration_report = {}
    manifests = {"blueonly": [], "templatesub": [], "lines": []}

    hints: dict[str, int] = {}
    for sheet, cfg in SHEETS.items():
        print(f"== {sheet}: building median template for {cfg['pdf']} p{cfg['page']}")
        template, report, found = build_template(cfg["pdf"], cfg["page"], hints)
        # The Q2 sheet follows the Q1 sheet in every booklet we have seen;
        # pass it as a hint (find_sheet_page falls back to a full search).
        hints = {donor: page + 1 for donor, page in found.items()}
        registration_report[sheet] = report
        save_png(ART / "templates" / f"{sheet}_template.png", template)

        scan = whiten_red(render_page(cfg["pdf"], cfg["page"]))
        t_dark_scan = dark_threshold(gray_of(scan))
        t_dark_tmpl = dark_threshold(gray_of(template))
        registration_report[f"{sheet}_thresholds"] = {
            "t_dark_scan": t_dark_scan, "t_dark_tmpl": t_dark_tmpl,
            "t_blue": T_BLUE,
        }
        print(f"  T_dark scan={t_dark_scan:.0f} template={t_dark_tmpl:.0f}")

        rects = cell_rects(cfg["pdf"], cfg["page"], cfg["x"], cfg["rows"])
        for row, (x0, y0, x1, y1) in rects.items():
            cid = f"{sheet}_r{row}"
            scan_c = scan[y0:y1, x0:x1]
            tmpl_c = template[y0:y1, x0:x1]

            # A = the existing bench crop (identical recipe); self-check size.
            bench_png = BENCH / "crops" / f"{cid}.png"
            a_img = cv2.cvtColor(cv2.imread(str(bench_png)), cv2.COLOR_BGR2RGB)
            if a_img.shape[:2] != scan_c.shape[:2]:
                print(f"  WARNING {cid}: bench crop {a_img.shape[:2]} != "
                      f"render slice {scan_c.shape[:2]}")

            res = isolate_cell(scan_c, tmpl_c, t_dark_scan, t_dark_tmpl)
            cell_dir = ART / "cells" / cid
            save_png(cell_dir / "A_original.png", a_img)
            save_png(cell_dir / "B_template.png", res["B"])
            save_png(cell_dir / "C_absdiff.png", res["C"])
            save_png(cell_dir / "D_mask.png", res["D"])
            save_png(cell_dir / "E1_blueonly.png", res["E1"])
            save_png(cell_dir / "E2_templatesub.png", res["E2"])

            bands = line_bands_from_mask(res["mask_full"])
            line_files = []
            for i, (a, b) in enumerate(bands, 1):
                lf = ART / "lines" / cid / f"line{i}.png"
                save_png(lf, res["E2"][a:b, :])
                line_files.append(str(lf))

            f_panels = [res["E2"][a:b, :] for a, b in bands]
            gap_bar = np.full((6, res["E2"].shape[1], 3), 128, np.uint8)
            f_img = np.vstack(sum(([p, gap_bar] for p in f_panels), [])[:-1])
            sheet_img = contact_sheet(cid, [
                ("A original", a_img), ("B template", res["B"]),
                ("C absdiff", res["C"]), ("D mask", res["D"]),
                ("E1 blue-only", res["E1"]), ("E2 template-sub", res["E2"]),
                (f"F lines (n={len(bands)})", f_img),
            ])
            save_png(ART / "contact" / f"{cid}.png", sheet_img)

            h, w = res["E2"].shape[:2]
            common = {"source": cfg["pdf"], "page": cfg["page"], "row": row,
                      "width": w, "height": h}
            manifests["blueonly"].append(
                {"id": cid, "file": str(cell_dir / "E1_blueonly.png"), **common})
            manifests["templatesub"].append(
                {"id": cid, "file": str(cell_dir / "E2_templatesub.png"), **common})
            manifests["lines"].append({"id": cid, "lines": line_files, **common})
            print(f"  {cid}: mask px={int(res['mask_full'].sum())} lines={len(bands)}")

    mdir = ART / "manifests"
    mdir.mkdir(exist_ok=True)
    for name, entries in manifests.items():
        (mdir / f"manifest_{name}.json").write_text(
            json.dumps(entries, indent=1), encoding="utf-8")
    (ART / "registration_report.json").write_text(
        json.dumps(registration_report, indent=1), encoding="utf-8")
    print(f"done in {time.monotonic() - t0:.0f}s -> {ART}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
