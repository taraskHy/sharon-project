"""Generate the hidden-ground-truth Hebrew handwriting benchmark crops.

20 explanation-cell crops from the three sanctioned exams (representative,
002, 003 — never the held-out set). Crops are saved as raw colour PNGs at
high source resolution; preprocessing is an ITERATION variable applied at
runtime, never baked into the benchmark. Ground truth lives in a separate
JSON the runner never opens.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from autograder.ingest import PageImage
from autograder.masking import mask_pages

OUT = Path("evaluation/hebrew_bench/crops")
ZOOM_WIDTH = 2200  # source render width in px before cropping

# Per-sheet geometry: explanation-cell x-span and per-row y-spans
# (fractions of the page), tuned on 1000px renders.
SHEETS = {
    "e003_q1": {
        "pdf": "test/003_70.pdf", "page": 11,
        "x": (0.09, 0.66),
        "rows": {
            1: (0.352, 0.430), 2: (0.423, 0.494), 3: (0.486, 0.559),
            4: (0.551, 0.624), 5: (0.616, 0.684), 6: (0.676, 0.744),
            7: (0.736, 0.804), 8: (0.796, 0.874),
        },
    },
    "e002_q1": {
        "pdf": "test/002_76.pdf", "page": 11,
        "x": (0.06, 0.645),
        "rows": {
            1: (0.355, 0.434), 2: (0.426, 0.497), 3: (0.488, 0.562),
            4: (0.553, 0.625), 5: (0.616, 0.687), 6: (0.678, 0.747),
            7: (0.738, 0.810), 8: (0.801, 0.880),
        },
    },
    "rep_q1sheet": {  # sample_data/student_exam.pdf page 12 (holds Q1's answers)
        "pdf": "sample_data/student_exam.pdf", "page": 12,
        "x": (0.07, 0.64),
        "rows": {
            1: (0.330, 0.420), 2: (0.410, 0.495), 3: (0.485, 0.570),
            4: (0.555, 0.645),
        },
    },
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for sheet, cfg in SHEETS.items():
        doc = fitz.open(cfg["pdf"])
        page = doc[cfg["page"] - 1]
        r = page.rect
        zoom = ZOOM_WIDTH / r.width
        for row, (y0, y1) in cfg["rows"].items():
            clip = fitz.Rect(
                r.width * cfg["x"][0], r.height * y0,
                r.width * cfg["x"][1], r.height * y1,
            )
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
            img = PageImage(page_number=cfg["page"], png_bytes=pix.tobytes("png"),
                            width=pix.width, height=pix.height, text="")
            img = mask_pages([img])[0][0]  # red instructor ink removed, blue kept
            name = f"{sheet}_r{row}.png"
            (OUT / name).write_bytes(img.png_bytes)
            manifest.append({"id": f"{sheet}_r{row}", "file": str(OUT / name),
                             "source": cfg["pdf"], "page": cfg["page"], "row": row,
                             "width": img.width, "height": img.height})
        doc.close()
    (OUT.parent / "crops_manifest.json").write_text(
        json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{len(manifest)} crops -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
