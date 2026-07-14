"""Build the writer-separated HTR-pilot annotation package.

Pilot exams: e003–e018 (16 dev exams; one writer per exam). Deterministic
split by numeric order — train e003–e012 (10), val e013–e015 (3),
internal_test e016–e018 (3). Exam 002 is excluded from EVERY split (it is
the hidden transcription benchmark writer); the representative exam is
excluded (different sheet layout, already audited); e019–e042 are reserved
for scale-up; the 48 held-out exams are untouched (not even referenced).

Per exam and per answer sheet (Q1, Q2):

1. Sheet identification. Q1/Q2 printed forms differ only in the title
   digit, and booklet page order varies per scan, so every page is scored
   by quarter-scale ECC against the exam-002 Q1 reference; candidate pages
   are registered at half scale and classified Q1-vs-Q2 by SSD inside the
   digit region (located automatically by diffing the two exam-002
   templates). Ambiguous or missing sheets are skipped and reported.
2. Blank template synthesis: per-pixel median of up to 12 other pilot
   exams' same-question sheets, ECC-registered onto THIS exam's page frame
   (same method and thresholds as scripts/student_ink_isolation.py).
3. Cell geometry from the template itself: the 10 horizontal table rules
   are detected on the template print mask (row dark-fraction > 0.5 across
   the explanation-column span), giving 8 rows between rules 2–10; each
   cell band is padded so ascenders/descenders and the printed borders are
   included, like the bench crops.
4. Student-ink isolation (E2 mask), line segmentation on the mask, and
   asset writing: original cell (red-masked JPEG), cleaned cell (PNG),
   line crops (PNG), per-sheet contact sheet (thumbnails + title strip).

Outputs under evaluation/htr_pilot/ use writer ids e003…e018 and contain
NO grade-bearing filenames; the source-PDF mapping (which necessarily
contains grades, since the scan files are named that way) is written to
evaluation/htr_pilot_sources.json OUTSIDE the package and must never be
read by training code.

Ground truth is not touched; nothing here transcribes anything.

    .venv/Scripts/python.exe scripts/htr_pilot_build.py [--exams e003,e004]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.student_ink_isolation import (
    CC_MIN, T_BLUE, dark_threshold, dilate, despeckle, ecc_align, gray_of,
    inv_gray, isolate_cell, line_bands_from_mask, page_count, register,
    render_page, whiten_red,
)

ROOT = Path("evaluation/htr_pilot")
ART = Path("evaluation/student_ink_isolation_artifacts")

# Exam number -> grade suffix of the scan filename. Grades appear ONLY here
# and in evaluation/htr_pilot_sources.json — never inside the package.
EXAM_FILES = {
    3: 70, 4: 58, 5: 48, 6: 86, 7: 48, 8: 52, 9: 94, 10: 52, 11: 98,
    12: 70, 13: 80, 14: 58, 15: 59, 16: 65, 17: 30, 18: 92,
}
SPLITS = {
    "train": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "val": [13, 14, 15],
    "internal_test": [16, 17, 18],
}
XSPAN = (0.06, 0.645)      # explanation column, same as the bench crops
CELL_PAD_FRAC = 0.009      # vertical padding around each row band
RULE_ROWFRAC = 0.5         # a printed rule covers > 50 % of the column span
RULE_MAX_GAP = 8           # px between dark rows of the same rule
N_KEEP_DONORS = 12
CAND_CCQ_MIN = 0.45        # quarter-scale gate for "looks like an answer sheet"
DIGIT_MARGIN_MIN = 0.05    # relative SSD margin required to accept Q1/Q2 label
BLANK_MASK_PX = 300        # below this many ink pixels a cell is expected blank
JPEG_Q = 88


def pdf_of(n: int) -> str:
    return f"test/{n:03d}_{EXAM_FILES[n]}.pdf"


def writer_of(n: int) -> str:
    return f"e{n:03d}"


def split_of(n: int) -> str:
    return next(s for s, xs in SPLITS.items() if n in xs)


def locate_digit(tmpl_gray: np.ndarray) -> tuple:
    """Digit region of the printed title, located in the template's OWN
    frame: the title is the underlined bold line in the top band; RTL puts
    the question digit at its LEFT end (…שאלה מספר N). Returns the leftmost
    glyph cluster of the title's text rows (underline rows excluded)."""
    h, w = tmpl_gray.shape
    dark = tmpl_gray < dark_threshold(tmpl_gray)
    band = dark[int(0.08 * h):int(0.15 * h), :]
    frac = band[:, int(0.15 * w):int(0.85 * w)].mean(axis=1)
    rows = np.where(frac > 0.05)[0]
    if rows.size == 0:
        raise RuntimeError("title band not found in template")
    ty0, ty1 = rows.min(), rows.max()
    # Underline rows are much wider runs than glyph rows: drop them.
    glyph_rows = [y for y in range(ty0, ty1 + 1) if frac[y] <= 0.30]
    if not glyph_rows:
        raise RuntimeError("title glyph rows not separable from underline")
    gy0, gy1 = min(glyph_rows), max(glyph_rows)
    strip = band[gy0:gy1 + 1, :]
    col = strip.mean(axis=0)
    on = np.where(col > 0.08)[0]
    if on.size == 0:
        raise RuntimeError("no title glyphs found")
    # Leftmost glyph cluster = the digit.
    clusters, cur = [], [on[0]]
    for x in on[1:]:
        if x - cur[-1] > 12:
            clusters.append(cur)
            cur = []
        cur.append(x)
    clusters.append(cur)
    c = clusters[0]
    pad = 10
    y_abs = int(0.08 * h)
    return (max(0, c[0] - pad), max(0, y_abs + gy0 - pad),
            min(w, c[-1] + pad), min(h, y_abs + gy1 + pad))


def identify_sheets(ref_inv: np.ndarray, patch1: np.ndarray,
                    patch2: np.ndarray, box: tuple, pdf: str) -> dict:
    """Find and label this exam's Q1/Q2 answer-sheet pages.

    box is the digit region in the REFERENCE frame; patch1/patch2 are the
    digit glyph patches cut from each template in its own frame (match-
    Template is translation-invariant, so frames need not coincide).
    Returns {"1": {...}, "2": {...}}; a question key is absent when no page
    could be confidently labelled.
    """
    x0, y0, x1, y1 = box
    ref_q = cv2.resize(ref_inv, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    cands = []
    for p in range(1, page_count(pdf) + 1):
        try:
            mov_q = inv_gray(whiten_red(render_page(pdf, p, width=550)))
            cc_q, _ = ecc_align(ref_q, mov_q, iters=80)
        except cv2.error:
            continue
        if cc_q >= CAND_CCQ_MIN:
            cands.append((p, cc_q))
    labelled = {}
    h, w = ref_inv.shape
    m = 60  # local search margin: digit position vs registration slack
    for p, cc_q in cands:
        try:
            rgb = whiten_red(render_page(pdf, p))
            cc, warp = register(ref_inv, inv_gray(rgb))
            aligned = cv2.warpAffine(
                gray_of(rgb).astype(np.uint8), cv2.invertAffineTransform(warp),
                (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
        except cv2.error:
            continue
        win = aligned[max(0, y0 - m):min(h, y1 + m),
                      max(0, x0 - m):min(w, x1 + m)].astype(np.float32)
        # Translation-invariant: best normalized correlation of each digit
        # patch anywhere in the window decides Q1 vs Q2.
        c1 = float(cv2.matchTemplate(win, patch1.astype(np.float32),
                                     cv2.TM_CCOEFF_NORMED).max())
        c2 = float(cv2.matchTemplate(win, patch2.astype(np.float32),
                                     cv2.TM_CCOEFF_NORMED).max())
        if max(c1, c2) < 0.5 or abs(c1 - c2) < DIGIT_MARGIN_MIN:
            continue
        q = "1" if c1 > c2 else "2"
        best = labelled.get(q)
        if best is None or cc > best["cc"]:
            labelled[q] = {"page": p, "cc": round(cc, 4), "cc_q": round(cc_q, 4),
                           "digit_corr": [round(c1, 3), round(c2, 3)]}
    return labelled


def detect_rules(template: np.ndarray) -> list[int]:
    """Y-centres of the 10 horizontal table rules inside the column span."""
    g = gray_of(template)
    h, w = g.shape
    tdark = g < dark_threshold(g)
    x0, x1 = int(XSPAN[0] * w), int(XSPAN[1] * w)
    frac = tdark[:, x0:x1].mean(axis=1)
    rows = np.where(frac > RULE_ROWFRAC)[0]
    rows = rows[(rows > 0.20 * h) & (rows < 0.95 * h)]
    clusters, cur = [], []
    for y in rows:
        if cur and y - cur[-1] > RULE_MAX_GAP:
            clusters.append(int(np.mean(cur)))
            cur = []
        cur.append(y)
    if cur:
        clusters.append(int(np.mean(cur)))
    return clusters


def fallback_rules(ref_inv: np.ndarray, ref_rules: list[int],
                   xspan_px: tuple[int, int], scan: np.ndarray) -> list[int]:
    """Map the reference sheet's rule positions into the target frame via
    page registration — used when direct detection on the synthesized
    template does not find exactly 10 rules."""
    cc, warp = register(ref_inv, inv_gray(scan))
    if cc < 0.5:
        raise RuntimeError(f"fallback registration too weak (cc={cc:.2f})")
    x0, x1 = xspan_px
    out = []
    for y in ref_rules:
        pts = np.array([[x0, y, 1.0], [x1, y, 1.0]])
        ys = pts @ warp.T
        out.append(int(round(float(ys[:, 1].mean()))))
    return out


def thumb(img: np.ndarray, height: int) -> np.ndarray:
    s = height / img.shape[0]
    return cv2.resize(img, (max(1, int(img.shape[1] * s)), height))


def save_jpg(path: Path, rgb: np.ndarray, q: int = JPEG_Q) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, q])


def save_png(path: Path, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_PNG_COMPRESSION, 6])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exams", default="", help="comma list like e003,e004 (debug)")
    args = ap.parse_args()
    only = {s.strip() for s in args.exams.split(",") if s.strip()}

    t0 = time.monotonic()
    ROOT.mkdir(parents=True, exist_ok=True)

    # Reference frame + digit region from the exam-002 isolation templates.
    ref_rgb = whiten_red(render_page("test/002_76.pdf", 11))
    ref_inv = inv_gray(ref_rgb)
    tq1 = cv2.cvtColor(cv2.imread(str(ART / "templates/e002_q1_template.png")),
                       cv2.COLOR_BGR2RGB)
    tq2 = cv2.cvtColor(cv2.imread(str(ART / "templates/e002_q2_template.png")),
                       cv2.COLOR_BGR2RGB)
    tq1_g = gray_of(tq1)
    tq2_g = gray_of(tq2)
    box1 = locate_digit(tq1_g)
    box2 = locate_digit(tq2_g)
    patch1 = tq1_g[box1[1]:box1[3], box1[0]:box1[2]]
    patch2 = tq2_g[box2[1]:box2[3], box2[0]:box2[2]]
    print(f"digit boxes: Q1 {box1}  Q2 {box2}")
    # Reference rule geometry per question, for the registration fallback.
    refs = {
        "1": (ref_inv, detect_rules(tq1)),
        "2": (inv_gray(whiten_red(render_page("test/002_76.pdf", 12))),
              detect_rules(tq2)),
    }
    for q, (_ri, rr) in refs.items():
        if len(rr) != 10:
            raise RuntimeError(f"reference Q{q} template yields {len(rr)} rules")

    # Pass 1 — identify every exam's Q1/Q2 sheet pages. Always runs over the
    # full pilot (donor templates need the other exams' pages even when
    # --exams restricts generation).
    sheets: dict[int, dict] = {}
    for n in EXAM_FILES:
        sheets[n] = identify_sheets(ref_inv, patch1, patch2, box1, pdf_of(n))
        print(f"{writer_of(n)}: sheets {json.dumps(sheets[n])}")

    # Pass 2 — per exam, per question: template, geometry, isolation, assets.
    samples, failures, source_map = [], [], {}
    con_dir = ROOT / "contact"
    for n in sorted(sheets):
        if only and writer_of(n) not in only:
            continue
        writer, split, pdf = writer_of(n), split_of(n), pdf_of(n)
        source_map[writer] = {"pdf": pdf, "sheets": sheets[n]}
        for q in ("1", "2"):
            if q not in sheets[n]:
                failures.append({"writer": writer, "question": q,
                                 "reason": "sheet page not identified"})
                continue
            sidecar = ROOT / "images" / writer / f"q{q}_meta.json"
            if sidecar.exists():
                prior = json.loads(sidecar.read_text(encoding="utf-8"))
                samples.extend(prior["samples"])
                print(f"  {writer} q{q}: reused existing sheet "
                      f"({len(prior['samples'])} samples)")
                continue
            page = sheets[n][q]["page"]
            donors = [pdf_of(m) for m in sorted(sheets) if m != n
                      and q in sheets[m]]
            hints = {pdf_of(m): sheets[m][q]["page"] for m in sorted(sheets)
                     if m != n and q in sheets[m]}
            try:
                from scripts.student_ink_isolation import build_template
                template, _rep, _found = build_template(
                    pdf, page, hints=hints, donors=donors, n_keep=N_KEEP_DONORS)
            except (RuntimeError, cv2.error) as e:
                failures.append({"writer": writer, "question": q,
                                 "reason": f"template: {e}"})
                continue
            scan = whiten_red(render_page(pdf, page))
            geometry = "template"
            rules = detect_rules(template)
            if len(rules) != 10:
                ref_i, ref_r = refs[q]
                try:
                    xs = (int(XSPAN[0] * scan.shape[1]),
                          int(XSPAN[1] * scan.shape[1]))
                    rules = fallback_rules(ref_i, ref_r, xs, scan)
                    geometry = "fallback"
                    print(f"  {writer} q{q}: rule detection gave "
                          f"{len(detect_rules(template))}; using registration fallback")
                except (RuntimeError, cv2.error) as e:
                    failures.append({"writer": writer, "question": q,
                                     "reason": f"geometry: {e}"})
                    continue
            t_dark_scan = dark_threshold(gray_of(scan))
            t_dark_tmpl = dark_threshold(gray_of(template))
            h, w = scan.shape[:2]
            x0, x1 = int(XSPAN[0] * w), int(XSPAN[1] * w)
            pad = int(CELL_PAD_FRAC * h)
            thumbs = []
            sheet_start = len(samples)
            for r in range(1, 9):
                y0 = max(0, rules[r] - pad)
                y1 = min(h, rules[r + 1] + pad)
                scan_c, tmpl_c = scan[y0:y1, x0:x1], template[y0:y1, x0:x1]
                res = isolate_cell(scan_c, tmpl_c, t_dark_scan, t_dark_tmpl)
                mask_px = int(res["mask_full"].sum())
                cell = f"q{q}_r{r}"
                img_dir = ROOT / "images" / writer
                save_jpg(img_dir / f"{cell}_cell_orig.jpg", scan_c)
                save_png(img_dir / f"{cell}_cell_clean.png", res["E2"])
                expected_blank = mask_px < BLANK_MASK_PX
                bands = [(0, scan_c.shape[0])] if expected_blank else \
                    line_bands_from_mask(res["mask_full"])
                for li, (a, b) in enumerate(bands, 1):
                    lf = img_dir / f"{cell}_l{li}.png"
                    save_png(lf, res["E2"][a:b, :])
                    samples.append({
                        "sample_id": f"{writer}_{cell}__l{li}",
                        "writer": writer, "split": split,
                        "question": int(q), "row": r, "line_index": li,
                        "n_lines": len(bands),
                        "expected_blank": expected_blank,
                        "images": {
                            "line": f"images/{writer}/{cell}_l{li}.png",
                            "cell_clean": f"images/{writer}/{cell}_cell_clean.png",
                            "cell_orig": f"images/{writer}/{cell}_cell_orig.jpg",
                        },
                        "line_size": [int(b - a), int(res['E2'].shape[1])],
                    })
                thumbs.append(np.hstack([thumb(scan_c, 110), thumb(res["E2"], 110)]))
                print(f"  {writer} {cell}: ink={mask_px} lines={len(bands)}"
                      f"{' BLANK?' if expected_blank else ''}")
            title = scan[int(0.08 * h):int(0.145 * h), int(0.25 * w):int(0.75 * w)]
            width = max(t.shape[1] for t in thumbs + [thumb(title, 60)])
            rowsimg = []
            for t in [thumb(title, 60)] + thumbs:
                if t.shape[1] < width:
                    t = np.hstack([t, np.full((t.shape[0], width - t.shape[1], 3),
                                              255, np.uint8)])
                rowsimg += [t, np.full((4, width, 3), 90, np.uint8)]
            save_jpg(con_dir / f"{writer}_q{q}.jpg", np.vstack(rowsimg[:-1]), 82)
            sidecar.write_text(json.dumps(
                {"geometry": geometry, "rules": rules,
                 "samples": samples[sheet_start:]},
                indent=1, ensure_ascii=False), encoding="utf-8")

    # Split metadata — one file per split so loaders can only see their own.
    sdir = ROOT / "splits"
    sdir.mkdir(exist_ok=True)
    for split in SPLITS:
        recs = [s for s in samples if s["split"] == split]
        (sdir / f"{split}.json").write_text(
            json.dumps(recs, indent=1, ensure_ascii=False), encoding="utf-8")
    for split in SPLITS:
        d = ROOT / "annotations" / split
        d.mkdir(parents=True, exist_ok=True)
        (d / ".gitkeep").touch()

    per_writer = {}
    for s in samples:
        per_writer.setdefault(s["writer"], {"split": s["split"], "lines": 0,
                                            "cells": set()})
        per_writer[s["writer"]]["lines"] += 1
        per_writer[s["writer"]]["cells"].add((s["question"], s["row"]))
    summary = {
        "built": time.strftime("%Y-%m-%d %H:%M:%S"),
        "splits": {s: [writer_of(n) for n in xs] for s, xs in SPLITS.items()},
        "samples_total": len(samples),
        "per_writer": {wr: {"split": v["split"], "lines": v["lines"],
                            "cells": len(v["cells"])}
                       for wr, v in sorted(per_writer.items())},
        "failures": failures,
        "constants": {"XSPAN": XSPAN, "CELL_PAD_FRAC": CELL_PAD_FRAC,
                      "T_BLUE": T_BLUE, "CC_MIN": CC_MIN,
                      "N_KEEP_DONORS": N_KEEP_DONORS,
                      "DIGIT_MARGIN_MIN": DIGIT_MARGIN_MIN,
                      "BLANK_MASK_PX": BLANK_MASK_PX},
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=1),
                                       encoding="utf-8")
    Path("evaluation/htr_pilot_sources.json").write_text(
        json.dumps(source_map, indent=1), encoding="utf-8")
    print(f"\n{len(samples)} line samples, {len(failures)} sheet failures, "
          f"{time.monotonic() - t0:.0f}s")
    print(json.dumps(summary["per_writer"], indent=0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
