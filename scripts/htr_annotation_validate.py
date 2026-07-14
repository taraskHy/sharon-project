"""Validate the HTR-pilot annotation package. Exit 0 = all checks pass.

Checks (owner criteria):
1. duplicate sample ids (within and across splits);
2. every referenced image resolves;
3. split leakage — no writer appears in more than one split, split fields
   consistent, splits match the documented pilot assignment;
4. exam 002 (or the rep exam) in any split — forbidden;
5. held-out-exam references — any writer outside e003…e018, any image path
   escaping the package images/ tree;
6. filename-grade leakage — no NNN_GG scan-filename pattern anywhere in
   package text files (splits, summary, annotations);
7. annotation records — schema/status validity, bad-segmentation (or any
   non-verified status) marked human_verified, status ok with empty
   transcription, non-NFC text, replacement characters, stray bidi
   controls, annotations for unknown sample ids or in the wrong split dir.

    .venv/Scripts/python.exe scripts/htr_annotation_validate.py [--root PATH]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.htr_annotation_lib import SPLITS, validate_record

EXPECTED_SPLITS = {
    "train": [f"e{n:03d}" for n in range(3, 13)],
    "val": [f"e{n:03d}" for n in range(13, 16)],
    "internal_test": [f"e{n:03d}" for n in range(16, 19)],
}
ALLOWED_WRITERS = {w for ws in EXPECTED_SPLITS.values() for w in ws}
FORBIDDEN_WRITERS = {"e002", "rep", "e001"}
GRADE_PATTERN = re.compile(r"\b\d{3}_\d{1,3}\b")
BIDI_CONTROLS = set("‎‏‪‫‬‭‮")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="evaluation/htr_pilot")
    args = ap.parse_args()
    root = Path(args.root)
    errors: list[str] = []
    warnings: list[str] = []

    # --- split metadata ---------------------------------------------------
    all_ids: dict[str, str] = {}
    writer_splits: dict[str, set] = {}
    samples_by_split: dict[str, dict] = {}
    for split in SPLITS:
        path = root / "splits" / f"{split}.json"
        if not path.exists():
            errors.append(f"missing splits file: {path}")
            continue
        raw = path.read_bytes()
        try:
            samples = json.loads(raw.decode("utf-8", errors="strict"))
        except UnicodeDecodeError as e:
            errors.append(f"{path}: not valid UTF-8 ({e})")
            continue
        samples_by_split[split] = {s["sample_id"]: s for s in samples}
        for s in samples:
            sid, wr = s["sample_id"], s["writer"]
            if sid in all_ids:
                errors.append(f"duplicate sample_id {sid} "
                              f"({all_ids[sid]} and {split})")
            all_ids[sid] = split
            writer_splits.setdefault(wr, set()).add(split)
            if s["split"] != split:
                errors.append(f"{sid}: split field {s['split']!r} in {split}.json")
            if wr in FORBIDDEN_WRITERS:
                errors.append(f"{sid}: forbidden writer {wr} (benchmark/rep exam)")
            if wr not in ALLOWED_WRITERS:
                errors.append(f"{sid}: writer {wr} outside the pilot set "
                              "(held-out exams must never be referenced)")
            for kind, rel in s["images"].items():
                p = root / rel
                if ".." in Path(rel).parts or not rel.startswith("images/"):
                    errors.append(f"{sid}: image path escapes package: {rel}")
                elif not p.exists():
                    errors.append(f"{sid}: missing image {rel}")

    for wr, splits_seen in writer_splits.items():
        if len(splits_seen) > 1:
            errors.append(f"SPLIT LEAKAGE: writer {wr} in {sorted(splits_seen)}")
    for split, expected in EXPECTED_SPLITS.items():
        seen = sorted(w for w, ss in writer_splits.items() if split in ss)
        extra = sorted(set(seen) - set(expected))
        if extra:
            errors.append(f"{split}: unexpected writers {extra}")
        missing = sorted(set(expected) - set(seen))
        if missing:
            warnings.append(f"{split}: writers with no samples: {missing}")
    if "e002" in writer_splits:
        errors.append("exam 002 entered a split")

    # --- grade leakage over every package text file -----------------------
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in {".json", ".md", ".txt", ".csv"}:
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        if "�" in text:
            errors.append(f"{p}: replacement characters (encoding damage)")
        hits = sorted(set(GRADE_PATTERN.findall(text)))
        if hits:
            errors.append(f"{p}: grade-bearing filename pattern(s) {hits[:4]}")

    # --- annotation records ------------------------------------------------
    n_ann = 0
    for split in SPLITS:
        d = root / "annotations" / split
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            n_ann += 1
            try:
                rec = json.loads(f.read_bytes().decode("utf-8", errors="strict"))
            except (UnicodeDecodeError, json.JSONDecodeError) as e:
                errors.append(f"{f}: unreadable ({e})")
                continue
            sid = rec.get("sample_id", f.stem)
            if sid != f.stem:
                errors.append(f"{f}: sample_id {sid!r} != filename")
            known = samples_by_split.get(split, {})
            if sid not in known:
                errors.append(f"{f}: annotation for unknown sample in {split}")
            if rec.get("split") != split:
                errors.append(f"{f}: record split {rec.get('split')!r} stored "
                              f"under {split}/ (leakage)")
            for problem in validate_record(rec):
                errors.append(f"{f}: {problem}")
            text = rec.get("transcription", "")
            if any(c in BIDI_CONTROLS for c in text):
                errors.append(f"{f}: bidi control characters in transcription")
            if text != unicodedata.normalize("NFC", text):
                errors.append(f"{f}: transcription not NFC")

    print(f"samples: {len(all_ids)}  annotations: {n_ann}")
    for w in warnings:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    print("RESULT:", "FAIL" if errors else "PASS")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
