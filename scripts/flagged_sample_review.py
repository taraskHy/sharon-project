"""QA review package for FLAGGED (unverified) train annotations.

Groups flagged samples (bad_segmentation / needs_recrop / skipped /
blank-unreadable) into per-group contact sheets, and renders a
deterministic re-segmentation PROPOSAL for each cell (current bands in
red vs proposed bands in green). READ-ONLY with respect to annotations:
no status is changed, no record is written, originals are untouched.
Proposed band geometry is saved to proposals.json for a later
owner-approved rebuild only.

Current bands are reconstructed with the build pipeline's parameters
(scripts/student_ink_isolation.py) on an ink mask re-derived from the
cleaned cell; the proposal uses one fixed alternative parameter set
(halved merge gap, halved profile threshold, 0.6x min height) that
splits merged lines more aggressively.

    .venv/Scripts/python.exe scripts/flagged_sample_review.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.htr_annotation_lib import load_all_annotations, load_samples  # noqa: E402
from scripts.student_ink_isolation import (  # noqa: E402
    LINE_PROFILE_MIN, MERGE_GAP_FRAC, MIN_LINE_HEIGHT_FRAC, PAD_FRAC,
)

ROOT = Path("evaluation/htr_pilot")
OUT = Path("evaluation/flagged_sample_review")
FLAGGED = ("bad_segmentation", "needs_recrop", "skipped")
GROUPS = FLAGGED + ("blank_unreadable",)  # last group: flagged-in-notes only
THUMB_W = 900


def bands_param(mask: np.ndarray, profile_min: float, merge_gap_frac: float,
                min_height_frac: float) -> list[tuple[int, int]]:
    """line_bands_from_mask with explicit parameters (same algorithm)."""
    h, w = mask.shape
    profile = mask.sum(axis=1) / w
    rows = profile > profile_min
    bands, start, gap = [], None, 0
    max_gap = int(h * merge_gap_frac)
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
           if (b - a) >= h * min_height_frac]
    return out or [(0, h)]


def ink_mask(cell_clean_gray: np.ndarray) -> np.ndarray:
    """Approximate student-ink mask from the cleaned (ink-on-white) cell."""
    m = (cell_clean_gray < 200).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    keep = np.zeros_like(m)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 12:
            keep[lab == i] = 1
    return keep


def draw_bands(rgb: np.ndarray, bands: list[tuple[int, int]],
               color: tuple[int, int, int]) -> np.ndarray:
    out = rgb.copy()
    for a, b in bands:
        cv2.rectangle(out, (2, a), (out.shape[1] - 3, b), color, 3)
    return out


def fit_w(img: np.ndarray, width: int = THUMB_W) -> np.ndarray:
    s = width / img.shape[1]
    return cv2.resize(img, (width, max(1, int(img.shape[0] * s))))


def label(text: str, width: int) -> np.ndarray:
    band = np.full((26, width, 3), 200, np.uint8)
    cv2.putText(band, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return band


def main() -> int:
    samples = {s["sample_id"]: s for s in load_samples(ROOT, "train")}
    ann = load_all_annotations(ROOT, "train")
    flagged = {g: [] for g in GROUPS}
    for sid, rec in sorted(ann.items()):
        if rec["status"] in FLAGGED:
            note = (rec.get("notes") or "").lower()
            if rec["status"] == "skipped" and ("blank" in note or "קריא" in note):
                flagged["blank_unreadable"].append(rec)
            else:
                flagged[rec["status"]].append(rec)

    OUT.mkdir(parents=True, exist_ok=True)
    proposals, index_lines = [], [
        "# Flagged-sample review", "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by "
        "`scripts/flagged_sample_review.py`. Read-only QA: no annotation "
        "status was changed; proposals require an owner-approved rebuild.",
        "",
    ]
    seen_cells = set()
    for group in GROUPS:
        recs = flagged[group]
        index_lines += [f"## {group} ({len(recs)})", ""]
        if not recs:
            index_lines += ["(none)", ""]
            continue
        panels = []
        for rec in recs:
            sid = rec["sample_id"]
            s = samples[sid]
            cell_key = (s["writer"], s["question"], s["row"])
            clean = cv2.imread(str(ROOT / s["images"]["cell_clean"]),
                               cv2.IMREAD_GRAYSCALE)
            orig = cv2.imread(str(ROOT / s["images"]["cell_orig"]))
            line = cv2.imread(str(ROOT / s["images"]["line"]))
            mask = ink_mask(clean)
            cur = bands_param(mask, LINE_PROFILE_MIN, MERGE_GAP_FRAC,
                              MIN_LINE_HEIGHT_FRAC)
            prop = bands_param(mask, LINE_PROFILE_MIN * 0.5,
                               MERGE_GAP_FRAC * 0.5,
                               MIN_LINE_HEIGHT_FRAC * 0.6)
            differs = prop != cur
            clean_rgb = cv2.cvtColor(clean, cv2.COLOR_GRAY2BGR)
            if cell_key not in seen_cells:
                seen_cells.add(cell_key)
                proposals.append({
                    "cell": f"{s['writer']}_q{s['question']}_r{s['row']}",
                    "flagged_line": sid, "status": rec["status"],
                    "current_bands": [list(b) for b in cur],
                    "proposed_bands": [list(b) for b in prop],
                    "proposal_differs": differs,
                })
            panels.append(label(
                f"{sid}  [{rec['status']}]  notes: {rec.get('notes', '')[:60]}",
                THUMB_W))
            panels.append(fit_w(orig))
            panels.append(label(
                f"current bands ({len(cur)}) RED | proposed ({len(prop)}) "
                f"GREEN {'*DIFFERS*' if differs else '(same)'}", THUMB_W))
            both = draw_bands(clean_rgb, cur, (0, 0, 255))
            both = draw_bands(both, prop, (0, 180, 0))
            panels.append(fit_w(both))
            panels.append(label("flagged line crop:", THUMB_W))
            panels.append(fit_w(line))
            panels.append(np.full((12, THUMB_W, 3), 90, np.uint8))
            index_lines.append(
                f"- `{sid}` notes: {rec.get('notes', '') or '(none)'} — "
                f"current {len(cur)} band(s), proposed {len(prop)}"
                f"{' **proposal differs**' if differs else ''}")
        sheet = np.vstack(panels)
        cv2.imwrite(str(OUT / f"{group}.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 82])
        index_lines += [f"", f"Contact sheet: `{group}.jpg`", ""]

    (OUT / "proposals.json").write_text(
        json.dumps(proposals, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUT / "README.md").write_text("\n".join(index_lines) + "\n",
                                   encoding="utf-8")
    print(json.dumps({g: len(v) for g, v in flagged.items()}))
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
