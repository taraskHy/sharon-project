"""Build the Mission-2 fixed Hebrew reading benchmark (hebrew_bench_v2).

Deterministic, no model calls. Reference tiers:

- ``owner``      — owner-verified transcriptions (gold): the 16 exam-002
                   cells from the July campaign + the 86 ok status line
                   annotations from the HTR pilot package.
- ``text-layer`` — born-digital PDFs' embedded text (gold-objective) for
                   printed RTL / mixed Hebrew-English / formula / option-row
                   crops. No AI transcription involved.

No AI-generated reference enters this benchmark as ground truth.

References are written to a SEPARATE file (references.json) that only the
post-inference evaluator may read — the manifest (items.json) carries no
reference text. Crops are copied in so the set is frozen and self-contained.
Privacy: no cover pages, no student names, no grade-bearing source names in
the manifest (handwritten items carry only anonymized writer ids; printed
sources are the two born-digital solution booklets, mapped in
sources_map.json — no student data involved).
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

import fitz
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "evaluation" / "hebrew_bench_v2"
CROPS = OUT / "crops"

HE = re.compile(r"[א-ת]")
EN = re.compile(r"[A-Za-z]{2,}")
MATHY = re.compile(r"[0-9=+\-*/^()\[\]{}<>≤≥∑∏√±·×]")


def block_category(text: str) -> str | None:
    he = len(HE.findall(text))
    if he < 8:
        return None
    if EN.search(text):
        return "mixed_he_en"
    mathy = len(MATHY.findall(text))
    if mathy / max(len(text), 1) > 0.18:
        return "formula_printed"
    return "printed_rtl"


def has_red_ink(png_path: Path) -> bool:
    pix = fitz.Pixmap(str(png_path))
    if pix.n < 3:
        return False
    a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    r, g, b = a[:, :, 0].astype(int), a[:, :, 1].astype(int), a[:, :, 2].astype(int)
    red = (r > 140) & (r - g > 50) & (r - b > 50)
    return int(red.sum()) > 60


def main() -> int:
    CROPS.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    refs: dict[str, dict] = {
        "_policy": (
            "References for hebrew_bench_v2. Read ONLY by the post-inference "
            "evaluator (scripts/m2_bench_eval.py), never by any inference "
            "code or prompt. Tiers: owner=human-verified; text-layer="
            "embedded born-digital PDF text (objective)."
        )
    }
    sources: dict[str, str] = {}

    # ---- 1. handwritten lines (owner-verified, HTR pilot package) --------
    ann_dir = REPO / "evaluation" / "htr_pilot" / "annotations"
    img_root = REPO / "evaluation" / "htr_pilot" / "images"
    n_lines = 0
    for f in sorted(ann_dir.rglob("*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict) or rec.get("status") != "ok":
            continue
        if not rec.get("human_verified") or not (rec.get("transcription") or "").strip():
            continue
        sid = rec["sample_id"]  # e003_q1_r1__l1
        writer, rest = sid.split("_", 1)
        img = img_root / writer / (rest.replace("__", "_") + ".png")
        if not img.exists():
            print(f"  missing image for {sid}: {img}")
            continue
        iid = f"hl_{sid}"
        dest = CROPS / f"{iid}.png"
        if not dest.exists():
            shutil.copy2(img, dest)
        hard = "לא קריא" in rec["transcription"]
        items.append(
            {
                "id": iid,
                "category": "handwritten_line",
                "tier": "owner",
                "hard": hard,
                "image": f"crops/{iid}.png",
                "writer": writer,
                "task": "transcribe the handwritten Hebrew line exactly as written",
            }
        )
        refs[iid] = {
            "text": rec["transcription"],
            "provenance": f"owner annotation {rec.get('saved_at', '')} (htr_pilot)",
        }
        n_lines += 1

    # ---- 2. handwritten cells (16 owner-verified exam-002 cells) ---------
    gt1 = json.loads(
        (REPO / "evaluation" / "hebrew_bench" / "verified_ground_truth.json").read_text(
            encoding="utf-8"
        )
    )
    n_cells = 0
    for cid, cell in sorted(gt1["cells"].items()):
        src = REPO / "evaluation" / "hebrew_bench" / "crops" / f"{cid}.png"
        if not src.exists():
            print(f"  missing crop {src}")
            continue
        iid = f"hc_{cid}"
        dest = CROPS / f"{iid}.png"
        if not dest.exists():
            shutil.copy2(src, dest)
        cats = ["handwritten_cell"]
        if has_red_ink(dest):
            cats.append("teacher_vs_student")
        items.append(
            {
                "id": iid,
                "category": cats[0],
                "extra_categories": cats[1:],
                "tier": "owner",
                "hard": cell.get("type") == "hard",
                "image": f"crops/{iid}.png",
                "writer": "e002",
                "task": (
                    "transcribe the student's handwritten Hebrew exactly; ignore "
                    "any red instructor ink; struck-through text is cancelled"
                ),
            }
        )
        refs[iid] = {
            "text": cell["text"],
            "unreadable_spans": cell.get("unreadable_spans", ""),
            "provenance": "owner-verified 16-cell benchmark (July campaign)",
        }
        n_cells += 1

    # ---- 3. printed blocks from born-digital PDFs (text-layer GT) --------
    printed_counts: dict[str, int] = {}
    page_counts: dict[tuple, int] = {}
    LIMITS = {"printed_rtl": 8, "mixed_he_en": 7, "formula_printed": 7}
    PER_PAGE = 2  # spread selection across the documents
    for doc_id, pdf in [("docA", REPO / "sample_data" / "Exam_solution.pdf"),
                        ("docB", REPO / "prob_data" / "sol.pdf")]:
        doc = fitz.open(pdf)
        sources[doc_id] = str(pdf.relative_to(REPO))
        for pno in range(len(doc)):
            page = doc[pno]
            for bi, block in enumerate(page.get_text("blocks")):
                x0, y0, x1, y1, text, *_ = block
                text = text.strip()
                if len(text) < 15:
                    continue
                cat = block_category(text)
                if cat is None or printed_counts.get(cat, 0) >= LIMITS[cat]:
                    continue
                if page_counts.get((doc_id, pno, cat), 0) >= PER_PAGE:
                    continue
                page_counts[(doc_id, pno, cat)] = page_counts.get((doc_id, pno, cat), 0) + 1
                rect = fitz.Rect(x0 - 4, y0 - 4, x1 + 4, y1 + 4) & page.rect
                if rect.is_empty or rect.width < 40 or rect.height < 12:
                    continue
                iid = f"pr_{doc_id}_p{pno + 1}_b{bi}"
                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=rect)
                pix.save(CROPS / f"{iid}.png")
                items.append(
                    {
                        "id": iid,
                        "category": cat,
                        "tier": "text-layer",
                        "image": f"crops/{iid}.png",
                        "source": f"{doc_id} page {pno + 1}",
                        "task": "transcribe the printed text exactly (Hebrew is RTL)",
                    }
                )
                refs[iid] = {
                    "text": text,
                    "provenance": f"embedded text layer of {doc_id} (born-digital)",
                }
                printed_counts[cat] = printed_counts.get(cat, 0) + 1
        doc.close()

    # ---- 4. option rows (row-association task, text-layer GT) ------------
    # MC option lines like  (א) 0.3   (ב) 0.47  ... — the pairing of option
    # letter to value is the association that matters.
    doc = fitz.open(REPO / "prob_data" / "sol.pdf")
    n_assoc = 0
    for pno in range(1, len(doc)):
        page = doc[pno]
        text = page.get_text()
        # question option lines carry (א) .. (ד)
        blocks = page.get_text("blocks")
        for bi, block in enumerate(blocks):
            x0, y0, x1, y1, btext, *_ = block
            # bidi extraction mangles the parens ("()ד0.51"); just require
            # all four option letters plus digits in a short block
            letters = {c for c in btext if c in "אבגד"}
            if len(letters) < 4 or not re.search(r"\d", btext) or len(btext) > 260:
                continue
            if n_assoc >= 5:
                break
            rect = fitz.Rect(x0 - 6, y0 - 6, x1 + 6, y1 + 6) & page.rect
            iid = f"assoc_docB_p{pno + 1}_b{bi}"
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), clip=rect)
            pix.save(CROPS / f"{iid}.png")
            items.append(
                {
                    "id": iid,
                    "category": "option_row_association",
                    "tier": "text-layer",
                    "image": f"crops/{iid}.png",
                    "source": f"docB page {pno + 1}",
                    "task": (
                        "this crop shows the four answer options of one "
                        "multiple-choice question; output each option letter "
                        "(א,ב,ג,ד) with its exact printed value"
                    ),
                }
            )
            refs[iid] = {
                "text": btext.strip(),
                "provenance": "embedded text layer of docB (born-digital)",
            }
            n_assoc += 1
    doc.close()

    manifest = {
        "version": 2,
        "built": "deterministic build by scripts/m2_bench_build.py",
        "reference_policy": (
            "references.json is read ONLY post-inference by the evaluator; "
            "no AI-generated text serves as ground truth (tiers: owner, "
            "text-layer)"
        ),
        "known_gaps": {
            "teacher_vs_student": (
                "EMPTY by honesty: no owner-verified crop containing red "
                "instructor ink exists (checked all 16 exam-002 cells and "
                "all htr_pilot cell originals programmatically — zero red "
                "pixels above threshold). Instructor-ink separation is "
                "covered in production by the deterministic masking module "
                "(leakage audit: 0/10 probes extracted grades) and cannot "
                "be honestly benchmarked from this material."
            ),
            "handwriting_writers": (
                "handwritten items cover writers e002-e007 only (owner-"
                "verified material available today); generalization to "
                "other hands is measured, not guaranteed"
            ),
        },
        "items": items,
    }
    (OUT / "items.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (OUT / "references.json").write_text(
        json.dumps(refs, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (OUT / "sources_map.json").write_text(
        json.dumps(sources, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    from collections import Counter
    cats = Counter(i["category"] for i in items)
    tiers = Counter(i["tier"] for i in items)
    red = sum(1 for i in items if "teacher_vs_student" in i.get("extra_categories", []))
    print(f"items: {len(items)} | categories: {dict(cats)} | tiers: {dict(tiers)}")
    print(f"handwritten lines: {n_lines} | cells: {n_cells} | red-ink tagged: {red}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
