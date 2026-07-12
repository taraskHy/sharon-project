"""Turn input documents (PDF or image files) into page images for the model.

The student exam is typically a pure image scan with no text layer, so vision
is the only reliable channel. The answer key is often born-digital; for it we
also extract the embedded text layer, which carries information that is hard
to read visually (e.g. answers encoded by font colour).
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def _natural_key(path: Path) -> tuple:
    """Sort 'page_2.png' before 'page_10.png'."""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


@dataclass
class PageImage:
    page_number: int  # 1-based
    png_bytes: bytes
    width: int
    height: int
    text: str  # embedded text layer, "" for pure scans


def _render_page(page: fitz.Page, max_long_edge: int) -> fitz.Pixmap:
    rect = page.rect
    long_edge_pts = max(rect.width, rect.height)
    zoom = max_long_edge / long_edge_pts if long_edge_pts > 0 else 1.0
    return page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))


def load_pages(path: str | Path, max_long_edge: int = 2300) -> list[PageImage]:
    """Load a PDF, a single image, or a directory of images as page images."""
    path = Path(path)
    if path.is_dir():
        files = sorted(
            (p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES),
            key=_natural_key,
        )
        if not files:
            raise ValueError(f"no image files found in directory {path}")
        pages: list[PageImage] = []
        for i, f in enumerate(files, start=1):
            pages.extend(_load_single(f, i, max_long_edge))
        return pages
    return _load_single(path, 1, max_long_edge)


def _load_single(path: Path, start_number: int, max_long_edge: int) -> list[PageImage]:
    doc = fitz.open(path)
    pages = []
    try:
        for i, page in enumerate(doc):
            pix = _render_page(page, max_long_edge)
            pages.append(
                PageImage(
                    page_number=start_number + i,
                    png_bytes=pix.tobytes("png"),
                    width=pix.width,
                    height=pix.height,
                    text=page.get_text().strip(),
                )
            )
    finally:
        doc.close()
    return pages


def image_block(page: PageImage) -> dict:
    """Content block for the Messages API."""
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.standard_b64encode(page.png_bytes).decode("ascii"),
        },
    }


def labeled_page_blocks(pages: list[PageImage]) -> list[dict]:
    """Interleave a text label before every page image so the model can cite pages."""
    blocks: list[dict] = []
    for p in pages:
        blocks.append({"type": "text", "text": f"--- Page {p.page_number} ---"})
        blocks.append(image_block(p))
    return blocks
