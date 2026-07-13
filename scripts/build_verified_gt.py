"""Build verified_ground_truth.json from the owner-filled annotation CSV.

Only rows with human_verified=true and a non-empty human_transcription are
accepted. The result is read exclusively by scripts/hebrew_bench_eval.py,
strictly after inference outputs are saved — never by any inference code.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BENCH = Path("evaluation/hebrew_bench")
CSV_PATH = BENCH / "human_annotation" / "annotation_template.csv"


def main() -> int:
    cells = {}
    skipped = []
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cid = row["crop_id"].strip()
            verified = row["human_verified"].strip().lower() in ("true", "1", "yes", "y")
            text = (row["human_transcription"] or "").strip()
            if not verified or not text:
                skipped.append(cid)
                continue
            hard = "hard" in (row["notes"] or "").lower() or bool(
                (row["unreadable_spans"] or "").strip()
            )
            cells[cid] = {
                "type": "hard" if hard else "strict",
                "text": text,
                "unreadable_spans": (row["unreadable_spans"] or "").strip(),
                "explanation_present": (row["explanation_present"] or "").strip(),
                "human_verified": True,
                "notes": (row["notes"] or "").strip(),
            }
    out = BENCH / "verified_ground_truth.json"
    out.write_text(
        json.dumps(
            {
                "_policy": "HUMAN-VERIFIED ground truth. Read only by the "
                "post-inference evaluator; never by inference code.",
                "cells": cells,
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"verified cells: {len(cells)} -> {out}")
    if skipped:
        print(f"skipped (unverified/empty): {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
