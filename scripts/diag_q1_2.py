"""Controlled diagnostic for exam-002 Q1.2 explanation-transcription failure.

Fixed model + prompt (the pipeline's EXTRACTION_SYSTEM, single sub-item,
json_schema, temperature 0). Three inputs of the SAME content, masked
exactly like the batch:

  A. the full answer-sheet page at 1000 px (batch-identical input);
  B. the row-2 band crop (row number + letter + explanation cell);
  C. the explanation-cell-only crop at high resolution.

The raw HTTP response is captured BEFORE parsing, so "model returned
nothing" vs "parser discarded text" is decided by evidence. Ground truth
stays out of every prompt.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import fitz
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from autograder.ingest import PageImage, load_pages
from autograder.masking import mask_pages
from autograder.prompts import EXTRACTION_SYSTEM
from autograder.schema import QuestionExtraction

BASE = "http://localhost:11434/v1"
MODEL = "qwen3-vl:8b-instruct"
SOURCE = "test/002_76.pdf"
PAGE = 11

Q1_STRUCTURE = (
    "Question 1: שאלה מספר 1\n"
    "Type: matching_with_explanation\n"
    "Sub-items (1):\n"
    "  - sub-item 2: פעולה 2\n"
    "Each sub-item requires a short written justification by the student.\n"
    "The answer sheet row shows: the printed row number, the student's "
    "chosen letter, and an explanation cell with the student's handwritten "
    "justification."
)


def crop_page(zoom_width: int, x0f: float, y0f: float, x1f: float, y1f: float) -> PageImage:
    doc = fitz.open(SOURCE)
    page = doc[PAGE - 1]
    r = page.rect
    zoom = zoom_width / r.width
    clip = fitz.Rect(r.width * x0f, r.height * y0f, r.width * x1f, r.height * y1f)
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
    img = PageImage(page_number=PAGE, png_bytes=pix.tobytes("png"), width=pix.width,
                    height=pix.height, text="")
    doc.close()
    return mask_pages([img])[0][0]


def build_inputs() -> dict[str, PageImage]:
    full = mask_pages([p for p in load_pages(SOURCE, 1000) if p.page_number == PAGE])[0][0]
    return {
        "A_full_page_1000px": full,
        "B_row2_band": crop_page(2000, 0.04, 0.415, 0.90, 0.505),
        "C_explanation_cell_hires": crop_page(2600, 0.05, 0.415, 0.65, 0.505),
    }


def ask(img: PageImage, label: str) -> None:
    schema = QuestionExtraction.model_json_schema()
    payload = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 1600,
        "messages": [
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": Q1_STRUCTURE},
                    {"type": "text", "text": f"Answer-sheet image follows ({img.width}x{img.height})."},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,"
                                      + base64.standard_b64encode(img.png_bytes).decode()},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Extract the student's final answer for question 1 "
                            "now. Report every sub-item (2) exactly once. "
                            "Transcribe the handwritten explanation faithfully. "
                            "Respond with ONLY a single JSON object (no prose, "
                            "no markdown fences) that conforms exactly to this "
                            "JSON Schema:\n" + json.dumps(schema, ensure_ascii=False)
                        ),
                    },
                ],
            },
        ],
        "response_format": {"type": "json_schema",
                            "json_schema": {"name": "QuestionExtraction", "schema": schema}},
    }
    with httpx.Client(timeout=900.0) as client:
        resp = client.post(f"{BASE}/chat/completions", json=payload)
    data = resp.json()
    choice = data["choices"][0]
    raw = choice["message"].get("content") or ""
    print(f"\n===== {label} =====")
    print(f"finish={choice.get('finish_reason')} raw_len={len(raw)}")
    print("RAW (verbatim):")
    print(raw)
    try:
        parsed = QuestionExtraction.model_validate_json(raw)
        for s in parsed.sub_items:
            print(
                f"PARSED sub-item {s.sub_item_id}: answer={s.final_answer!r} "
                f"legibility={s.explanation_legibility!r}\n"
                f"  transcription={s.explanation_transcription!r}"
            )
    except Exception as e:  # noqa: BLE001
        print(f"PARSE FAILED: {type(e).__name__}: {e}")


def main() -> int:
    sp = Path(r"C:\Users\ethan\AppData\Local\Temp\claude\C--Users-ethan-OneDrive-Desktop-etgar-cp-course-sharon-project\26474a20-ce5a-4aaf-96b8-b4a3920db371\scratchpad")
    for label, img in build_inputs().items():
        (sp / f"diag_q12_{label}.png").write_bytes(img.png_bytes)
        print(f"[input saved] {label}: {img.width}x{img.height}")
    for label, img in build_inputs().items():
        ask(img, label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
