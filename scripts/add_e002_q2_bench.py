"""Add exam-002 Q2 explanation cells (page 12, rows 1-8) to the bench and
attach the owner's human-verified transcriptions to the annotation CSV.

Mapping was verified visually by row position + content anchors before this
script encodes it (Q2.1 tail '255', Q2.2 'בדומה לו', Q2.3 'פריסה אחידה',
Q2.7 'צורת ההיסטוגרמה זהה', Q2.8 'x -> 2x'). Owner text is copied EXACTLY
(no normalization; [לא קריא] preserved).
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from autograder.ingest import PageImage
from autograder.masking import mask_pages

BENCH = Path("evaluation/hebrew_bench")
PKG = BENCH / "human_annotation"

ROWS = {
    1: (0.353, 0.432), 2: (0.424, 0.495), 3: (0.487, 0.556), 4: (0.548, 0.620),
    5: (0.612, 0.681), 6: (0.673, 0.750), 7: (0.742, 0.815), 8: (0.806, 0.882),
}
XSPAN = (0.06, 0.645)

OWNER_Q1 = {
    1: 'יש טשטוש בכל התדרים',
    2: 'נשאר בתמונה רק התדרים הגבוהים',
    3: 'העוצמה של הדרגות הגבוהות בפרמידה חזקה ביחס למקורי',
    4: 'נשאר בתמונה רק התדרים הנמוכים',
    5: 'מריחה בציר x עקב הקונבולוציה',
    6: 'מריחה בציר y עקב הקונבולוציה',
    7: 'סה"כ הפירמידה נראית תקינה. הדרגה 0 שלה בהירה משמעותית מהתמונה המקורית',
    8: 'בדומה לפעולה 7, מבהיר את התמונה אבל פה האיזורים הבהירים יותר נעשים יותר שינוי מהאזורים הכהים',
}
OWNER_Q2 = {
    1: 'עבור גילוי שפות יהיה רוב התמונה ב-[לא קריא] ורק עבור שפות 255',
    2: 'בדומה לו, רק שהטשטוש לפני מוביל לפחות שפות שמתגלות',
    3: 'פריסה אחידה של ההיסטוגרמה המקורית לאורך כל הטווח',
    4: '[לא קריא]',
    5: '[לא קריא] אחיד מוביל לריווח אחיד בין הערכים שקיימים בכימוי',
    6: 'המעכה זהה לסכימת תמונה זהה עם תנודה קלה. מין תמונת echo כזאת תגרום להיסטוגרמה החדשה להיות קרובה להכפלה ב-2 [לא קריא]',
    7: 'צורדת ההיסטוגרמה זהה, אך הגדלים והכמויות הם פי 2, כי גודל התמונה הינו [לא קריא]',
    8: 'כל ערך x בהיסטוגרמה ממופה ל2x בהיסטוגרמה החדשה',
}


def main() -> int:
    # 1. Crop the 8 Q2 cells (masked, 2200px source — same recipe as the rest).
    manifest = json.loads((BENCH / "crops_manifest.json").read_text(encoding="utf-8"))
    have = {m["id"] for m in manifest}
    doc = fitz.open("test/002_76.pdf")
    page = doc[11]
    r = page.rect
    zoom = 2200 / r.width
    for row, (y0, y1) in ROWS.items():
        cid = f"e002_q2_r{row}"
        clip = fitz.Rect(r.width * XSPAN[0], r.height * y0, r.width * XSPAN[1], r.height * y1)
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
        img = PageImage(page_number=12, png_bytes=pix.tobytes("png"),
                        width=pix.width, height=pix.height, text="")
        img = mask_pages([img])[0][0]
        out = BENCH / "crops" / f"{cid}.png"
        out.write_bytes(img.png_bytes)
        shutil.copyfile(out, PKG / "crops" / f"{cid}.png")
        if cid not in have:
            manifest.append({"id": cid, "file": str(out), "source": "test/002_76.pdf",
                             "page": 12, "row": row, "width": img.width, "height": img.height})
    doc.close()
    (BENCH / "crops_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    # 2. Rewrite the annotation CSV: owner labels for e002 Q1+Q2, verified.
    csv_path = PKG / "annotation_template.csv"
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        fields = rows[0].keys() if rows else None
    by_id = {row["crop_id"]: row for row in rows}
    for n, text in OWNER_Q1.items():
        row = by_id[f"e002_q1_r{n}"]
        row["human_transcription"] = text
        row["human_verified"] = "true"
        row["explanation_present"] = "true"
        row["unreadable_spans"] = "[לא קריא]" if "[לא קריא]" in text else ""
        row["notes"] = (row["notes"] + "; " if row["notes"] else "") + \
            "owner-verified 2026-07-13; mapping checked by printed row number"
    for n, text in OWNER_Q2.items():
        cid = f"e002_q2_r{n}"
        if cid not in by_id:
            new = {k: "" for k in fields}
            new.update({"crop_id": cid, "exam_id": "test/002_76.pdf", "question_id": "2",
                        "sub_item_id": str(n),
                        "image_path": f"evaluation/hebrew_bench/human_annotation/crops/{cid}.png"})
            rows.append(new)
            by_id[cid] = new
        row = by_id[cid]
        row["candidate_transcription"] = ""
        row["human_transcription"] = text
        row["human_verified"] = "true"
        row["explanation_present"] = "true"
        row["unreadable_spans"] = "[לא קריא]" if "[לא קריא]" in text else ""
        row["notes"] = "owner-verified 2026-07-13; mapping checked by row position + content anchors"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(fields))
        w.writeheader()
        w.writerows(rows)
    print(f"CSV updated: {sum(1 for r in rows if r['human_verified']=='true')} verified rows of {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
