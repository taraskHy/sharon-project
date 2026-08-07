"""Deterministic row-band cropping for fixed answer-sheet tables.

Motivation (measured, 2026-08-07): asked to read a full psychometric-style
answer table from a whole-page scan, the live open model returned wrong
letters for 9/10 rows at high confidence — RTL column confusion plus small
checkboxes. Cropping each table row into its own image, with the printed
header row stitched on top so the column letters stay visible, turns the task
into "which labeled column holds the mark in this ONE row" — trivially
readable at the same source resolution.

Everything here is pure geometry on the already-rendered (and, when masking
is enabled, already-masked) page image: no model calls, no randomness. The
table grid is found by run-length line detection, snapped to an arithmetic
lattice by least squares (grid spacing is uniform by construction of such
sheets), and missing/faded lines are filled by the lattice — validated
against the expected row count before any crop is produced. Any doubt raises
``TableCropError`` and the caller falls back to whole-page extraction.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import fitz  # PyMuPDF
import numpy as np

from .ingest import PageImage

# Pixels darker than this (0-255 gray) count as ink/print.
DARK_THRESHOLD = 160
# A grid line must contain dark runs at least this fraction of page width.
MIN_RUN_FRACTION = 1 / 24
# Candidate line rows must reach this fraction of the strongest line's count.
LINE_STRENGTH = 0.35
# Detected lines may deviate from their lattice slot by this fraction of the
# grid spacing before the fit is rejected.
MAX_RESIDUAL = 0.35
# Upscale factor for the final band composite (nearest-neighbour: lossless).
UPSCALE = 2
# Padding around the table bounding box, in pixels of the source render.
PAD = 14


class TableCropError(ValueError):
    """The page does not contain a readable grid matching the expected shape."""


@dataclass
class RowBand:
    row_index: int  # 0-based data-row index, top to bottom
    png_bytes: bytes
    width: int
    height: int


def _gray(png: bytes) -> np.ndarray:
    pix = fitz.Pixmap(png)
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n >= 3:
        return a[:, :, :3].mean(axis=2).astype(np.uint8)
    return a[:, :, 0].copy()


def _encode_png_gray(a: np.ndarray) -> bytes:
    """Minimal 8-bit grayscale PNG writer (keeps the package cv2/PIL-free)."""
    h, w = a.shape
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + tag + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def _line_positions(dark: np.ndarray, axis: int) -> list[int]:
    """Centres of long straight dark runs perpendicular to ``axis``:
    axis=1 finds horizontal lines (per-pixel-row counts), axis=0 vertical."""
    from numpy.lib.stride_tricks import sliding_window_view

    k = max(20, round(dark.shape[1 if axis == 1 else 0] * MIN_RUN_FRACTION))
    win = sliding_window_view(dark, k, axis=axis)
    counts = win.all(axis=-1).sum(axis=1 if axis == 1 else 0)
    peak = counts.max()
    if peak < k:  # even the best row has fewer run-pixels than one run
        return []
    idx = np.where(counts > LINE_STRENGTH * peak)[0]
    groups: list[list[int]] = []
    for y in idx:
        if groups and y - groups[-1][-1] <= 4:
            groups[-1].append(int(y))
        else:
            groups.append([int(y)])
    return [int(np.mean(g)) for g in groups]


def _lattice_fit(ys: list[int], n_lines: int) -> list[int]:
    """Snap detected line centres to an arithmetic lattice of ``n_lines``.

    Handles faded/missing lines (grid spacing bridges them) and stray long
    rules elsewhere on the page (kept only if they fall on the lattice;
    otherwise the longest consistent segment wins).
    """
    ys = sorted(ys)
    if len(ys) < max(4, n_lines // 2):
        raise TableCropError(f"only {len(ys)} grid line(s) detected")
    gaps = [b - a for a, b in zip(ys, ys[1:])]
    base = min(gaps)
    if base < 6:
        raise TableCropError("grid lines too close together to be table rows")
    unit = float(np.median([g for g in gaps if g <= 1.5 * base]))

    # Split into segments wherever the gap is not a small multiple of the
    # spacing; a stray name-field rule far above the table separates cleanly.
    segments: list[list[int]] = [[ys[0]]]
    for prev, y in zip(ys, ys[1:]):
        if (y - prev) <= 3.4 * unit:
            segments[-1].append(y)
        else:
            segments.append([y])
    seg = max(segments, key=len)
    if len(seg) < max(4, n_lines // 2):
        raise TableCropError("no consistent grid segment of sufficient size")

    ks = [0]
    for prev, y in zip(seg, seg[1:]):
        ks.append(ks[-1] + max(1, round((y - prev) / unit)))
    # Bottom lines may fade below detection (observed on scan 13): the lattice
    # may extrapolate a SHORT tail. The first detected line is anchored as the
    # table top; if the top line itself were the faded one, every band would
    # shift a row — which the printed-row-number check on each band converts
    # to review flags rather than silent misreads.
    missing_tail = (n_lines - 1) - ks[-1]
    if missing_tail < 0 or missing_tail > 3:
        raise TableCropError(
            f"grid spans {ks[-1] + 1} lattice slots, expected {n_lines}"
        )
    A = np.vstack([np.ones(len(seg)), np.array(ks, float)]).T
    (a, b), *_ = np.linalg.lstsq(A, np.array(seg, float), rcond=None)
    residuals = [abs(y - (a + b * k)) for y, k in zip(seg, ks)]
    if max(residuals) > MAX_RESIDUAL * unit:
        raise TableCropError("detected lines do not fit a uniform grid")
    return [int(round(a + b * k)) for k in range(n_lines)]


def answer_table_row_bands(page: PageImage, n_rows: int) -> list[RowBand]:
    """Crop the page's answer table into per-row composites.

    Expects a table of ``n_rows`` data rows under one header row
    (``n_rows + 2`` horizontal grid lines). Each returned band stacks the
    header band above ONE data row so the printed column letters remain
    visible directly above the row's cells. Raises ``TableCropError`` when
    the page has no such grid.
    """
    img = _gray(page.png_bytes)
    dark = img < DARK_THRESHOLD
    h_lines = _lattice_fit(_line_positions(dark, axis=1), n_rows + 2)
    v_lines = _line_positions(dark, axis=0)
    if len(v_lines) < 2:
        raise TableCropError("no vertical table borders found")
    if h_lines[0] < 0 or h_lines[-1] > img.shape[0]:
        raise TableCropError("fitted grid extends beyond the page")
    x0 = max(0, min(v_lines) - PAD)
    x1 = min(img.shape[1], max(v_lines) + PAD)
    if x1 - x0 < img.shape[1] // 4:
        raise TableCropError("table bounding box implausibly narrow")

    def slice_band(y_top: int, y_bottom: int) -> np.ndarray:
        return img[max(0, y_top - 3):min(img.shape[0], y_bottom + 3), x0:x1]

    header = slice_band(h_lines[0], h_lines[1])
    separator = np.full((3, header.shape[1]), 255, dtype=np.uint8)
    bands: list[RowBand] = []
    for i in range(n_rows):
        row = slice_band(h_lines[i + 1], h_lines[i + 2])
        comp = np.vstack([header, separator, row])
        comp = np.repeat(np.repeat(comp, UPSCALE, axis=0), UPSCALE, axis=1)
        bands.append(
            RowBand(
                row_index=i,
                png_bytes=_encode_png_gray(comp),
                width=comp.shape[1],
                height=comp.shape[0],
            )
        )
    return bands
