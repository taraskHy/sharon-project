"""Instructor-annotation masking.

The training/validation scans contain the instructor's red-ink grading:
per-question scores, ticks/crosses, deductions, and the final grade. If those
pixels reach the model, "grading" degenerates into copying the instructor —
label leakage. This module removes red-hued ink from rendered page images
BEFORE they are sent to any backend.

Approach (deliberately simple and auditable):

- work on the rendered PNG of each page (the original file is never touched);
- classify pixels as "red ink" by colour dominance (red channel well above
  green/blue) over several thresholds that cover pure red through dark red;
- replace red-ink pixels with white;
- record the masked regions (coarse-grid bounding boxes) and the red-pixel
  fraction per page in a ``MaskReport`` so every masking decision is auditable.

Honest limitations (also documented in docs/privacy-and-leakage.md):

- The heuristic removes red ink regardless of author. If a student wrote in
  red, their marks would be removed too — pages with unusually high red
  fractions are flagged for review rather than trusted silently.
- Non-red instructor marks (pencil ticks, blue corrections) are NOT removed;
  the survey pass's ink-separation instructions remain the second defence.
- This is pixel masking, not understanding: it cannot remove a grade written
  in blue. The leakage audit (``autograder audit-leakage``) exists to measure
  what actually leaks through.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import fitz
import numpy as np

from .ingest import PageImage

# A pixel is "red ink" when clearly red-dominant. Two bands: saturated red
# (marker/pen) and darker red ink.
RED_DOMINANCE = 50  # r - max(g, b) threshold
MIN_RED_VALUE = 70
# Pages whose red fraction exceeds this are suspicious (a red-pen student, a
# stamp, or a very heavily annotated page) and get a warning.
HIGH_RED_FRACTION = 0.02
# Coarse grid cell size (pixels) for region reporting.
GRID = 32


@dataclass
class MaskRegion:
    x0: int
    y0: int
    x1: int
    y1: int


@dataclass
class PageMaskReport:
    page_number: int
    red_pixel_fraction: float
    masked_pixels: int
    regions: list[MaskRegion] = field(default_factory=list)
    warning: str | None = None


@dataclass
class MaskReport:
    pages: list[PageMaskReport]
    params: dict = field(
        default_factory=lambda: {
            "red_dominance": RED_DOMINANCE,
            "min_red_value": MIN_RED_VALUE,
            "grid": GRID,
        }
    )

    def to_dict(self) -> dict:
        return {
            "params": self.params,
            "pages": [
                {
                    "page_number": p.page_number,
                    "red_pixel_fraction": round(p.red_pixel_fraction, 6),
                    "masked_pixels": p.masked_pixels,
                    "regions": [[r.x0, r.y0, r.x1, r.y1] for r in p.regions],
                    "warning": p.warning,
                }
                for p in self.pages
            ],
        }


def _png_to_array(png_bytes: bytes) -> np.ndarray:
    pix = fitz.Pixmap(png_bytes)
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)  # drop alpha
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return arr[:, :, :3].copy()


def _array_to_png(arr: np.ndarray) -> bytes:
    h, w, _ = arr.shape
    pix = fitz.Pixmap(fitz.csRGB, w, h, arr.tobytes(), False)
    return pix.tobytes("png")


def _red_mask(arr: np.ndarray) -> np.ndarray:
    r = arr[:, :, 0].astype(np.int16)
    g = arr[:, :, 1].astype(np.int16)
    b = arr[:, :, 2].astype(np.int16)
    dominance = r - np.maximum(g, b)
    return (dominance > RED_DOMINANCE) & (r > MIN_RED_VALUE)


def _regions_from_mask(mask: np.ndarray) -> list[MaskRegion]:
    """Coarse connected components over a GRID-sized downsampling of the mask."""
    h, w = mask.shape
    gh, gw = (h + GRID - 1) // GRID, (w + GRID - 1) // GRID
    coarse = np.zeros((gh, gw), dtype=bool)
    for gy in range(gh):
        for gx in range(gw):
            cell = mask[gy * GRID : (gy + 1) * GRID, gx * GRID : (gx + 1) * GRID]
            coarse[gy, gx] = bool(cell.any())

    seen = np.zeros_like(coarse)
    regions: list[MaskRegion] = []
    for gy in range(gh):
        for gx in range(gw):
            if not coarse[gy, gx] or seen[gy, gx]:
                continue
            stack = [(gy, gx)]
            seen[gy, gx] = True
            min_y, min_x, max_y, max_x = gy, gx, gy, gx
            while stack:
                cy, cx = stack.pop()
                min_y, max_y = min(min_y, cy), max(max_y, cy)
                min_x, max_x = min(min_x, cx), max(max_x, cx)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < gh and 0 <= nx < gw and coarse[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            regions.append(
                MaskRegion(
                    x0=min_x * GRID,
                    y0=min_y * GRID,
                    x1=min((max_x + 1) * GRID, w),
                    y1=min((max_y + 1) * GRID, h),
                )
            )
    return regions


def mask_page(page: PageImage) -> tuple[PageImage, PageMaskReport]:
    arr = _png_to_array(page.png_bytes)
    mask = _red_mask(arr)
    n_masked = int(mask.sum())
    fraction = n_masked / mask.size if mask.size else 0.0
    regions = _regions_from_mask(mask) if n_masked else []

    warning = None
    if fraction > HIGH_RED_FRACTION:
        warning = (
            f"unusually high red-ink fraction ({fraction:.1%}) — verify the student "
            "did not write in red before trusting this page's masking"
        )

    if n_masked:
        arr[mask] = 255  # replace red ink with white
    masked_page = PageImage(
        page_number=page.page_number,
        png_bytes=_array_to_png(arr) if n_masked else page.png_bytes,
        width=page.width,
        height=page.height,
        text=page.text,
    )
    return masked_page, PageMaskReport(
        page_number=page.page_number,
        red_pixel_fraction=fraction,
        masked_pixels=n_masked,
        regions=regions,
        warning=warning,
    )


def mask_pages(pages: list[PageImage]) -> tuple[list[PageImage], MaskReport]:
    masked, reports = [], []
    for p in pages:
        mp, rep = mask_page(p)
        masked.append(mp)
        reports.append(rep)
    return masked, MaskReport(pages=reports)
