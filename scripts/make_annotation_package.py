"""Assemble the human-annotation package for the Hebrew handwriting bench.

The AI's own readings are UNVERIFIED CANDIDATES ONLY (the system under
evaluation must not be its own annotator). human_verified=false everywhere
until the exam owner fills/confirms human_transcription.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BENCH = Path("evaluation/hebrew_bench")
PKG = BENCH / "human_annotation"
CROPS_OUT = PKG / "crops"

SOURCES = {
    "e003_q1": ("test/003_70.pdf", "1"),
    "e002_q1": ("test/002_76.pdf", "1"),
    "rep_q1sheet": ("sample_data/student_exam.pdf", "1 (content; printed sheet title says 2 — student swap)"),
}


def main() -> int:
    CROPS_OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((BENCH / "crops_manifest.json").read_text(encoding="utf-8"))
    gt = json.loads((BENCH / "ground_truth.json").read_text(encoding="utf-8"))["cells"]

    rows = []
    sheet_lines = [
        "# Hebrew handwriting bench — human annotation contact sheet",
        "",
        "Every `candidate_transcription` below was produced by the AI "
        "assistant reading the crops and is **UNVERIFIED** (`human_verified="
        "false`). Please write the authoritative text into "
        "`human_transcription` in `annotation_template.csv` (RTL Hebrew as "
        "written by the student; keep English terms as-is; mark unreadable "
        "spans like `[?]` and note struck-through text). The verified labels "
        "will NEVER be shown to any inference prompt — only to the "
        "post-inference evaluator.",
        "",
    ]
    for m in manifest:
        cid = m["id"]
        sheet_key = cid.rsplit("_r", 1)[0]
        row_no = cid.rsplit("_r", 1)[1]
        src, qid = SOURCES[sheet_key]
        dst = CROPS_OUT / f"{cid}.png"
        shutil.copyfile(m["file"], dst)
        cand = gt.get(cid, {})
        notes = []
        if cand.get("type") == "hard":
            notes.append(cand.get("note", "hard cell"))
        if cand.get("low_confidence"):
            notes.append("AI-uncertain tokens: " + ", ".join(cand["low_confidence"]))
        if cid == "rep_q1sheet_r4":
            notes.append("crop bleeds into the next row (row geometry misestimate); "
                         "transcribe ONLY the row-4 content or mark for recrop")
        rows.append({
            "crop_id": cid,
            "exam_id": src,
            "question_id": qid,
            "sub_item_id": row_no,
            "image_path": str(dst).replace("\\", "/"),
            "candidate_transcription": cand.get("text", ""),
            "human_transcription": "",
            "unreadable_spans": "",
            "explanation_present": "",
            "human_verified": "false",
            "notes": "; ".join(notes),
        })
        sheet_lines += [
            f"## {cid}",
            f"- exam: `{src}` | question {qid} | sub-item/row {row_no}",
            f"- candidate (UNVERIFIED): {cand.get('text', '(none)')}",
            f"![{cid}](crops/{cid}.png)",
            "",
        ]

    csv_path = PKG / "annotation_template.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:  # BOM: Excel-friendly Hebrew
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (PKG / "contact_sheet.md").write_text("\n".join(sheet_lines), encoding="utf-8")
    print(f"crops:  {CROPS_OUT}  ({len(rows)} files)")
    print(f"csv:    {csv_path}")
    print(f"sheet:  {PKG / 'contact_sheet.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
