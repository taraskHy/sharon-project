"""Storage and rules for the HTR-pilot annotation workflow (no UI here).

One annotation record per sample, stored as
<root>/annotations/<split>/<sample_id>.json — splits live in separate
directories so a training loader can only ever be pointed at its own
split. Writes are atomic (tmp file + replace) so a crash never corrupts a
record; every save IS the autosave.

Status vocabulary and verification rules:

- ok               transcription must be non-empty        -> verified
- unreadable_full  whole line unreadable; text forced to
                   the [לא קריא] token                     -> verified
- blank            no student writing in the image;
                   transcription forced empty              -> verified
- bad_segmentation line crop wrong (merged/cut)           -> NOT verified
- needs_recrop     cell geometry wrong                    -> NOT verified
- skipped          decide later                           -> NOT verified
- draft            autosaved navigation state             -> NOT verified

Partial unreadable words are written inline inside the transcription with
the same token, e.g. "מריחה [לא קריא] הקונבולוציה".
"""

from __future__ import annotations

import json
import os
import time
import unicodedata
from pathlib import Path

UNREADABLE_TOKEN = "[לא קריא]"
SCHEMA_VERSION = 1

VERIFIED_STATUSES = {"ok", "unreadable_full", "blank"}
UNVERIFIED_STATUSES = {"bad_segmentation", "needs_recrop", "skipped", "draft"}
STATUSES = VERIFIED_STATUSES | UNVERIFIED_STATUSES

DEFAULT_ROOT = Path("evaluation/htr_pilot")
SPLITS = ("train", "val", "internal_test")


def package_root() -> Path:
    return Path(os.environ.get("HTR_PILOT_ROOT", str(DEFAULT_ROOT)))


def load_samples(root: Path, split: str) -> list[dict]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    path = Path(root) / "splits" / f"{split}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def annotation_path(root: Path, split: str, sample_id: str) -> Path:
    return Path(root) / "annotations" / split / f"{sample_id}.json"


def normalize_text(text: str) -> str:
    """NFC-normalize, strip bidi control characters and outer whitespace.
    Content is never altered beyond that — no spellfix, no reordering."""
    text = unicodedata.normalize("NFC", text or "")
    text = "".join(c for c in text if c not in "‎‏‪‫‬‭‮")
    return text.strip()


def make_record(sample: dict, transcription: str, status: str,
                notes: str = "", annotator: str = "owner") -> dict:
    if status not in STATUSES:
        raise ValueError(f"invalid status {status!r}")
    text = normalize_text(transcription)
    if status == "unreadable_full":
        text = UNREADABLE_TOKEN
    if status == "blank":
        text = ""
    record = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample["sample_id"],
        "split": sample["split"],
        "writer": sample["writer"],
        "transcription": text,
        "status": status,
        "unreadable": UNREADABLE_TOKEN in text,
        "blank": status == "blank",
        "bad_segmentation": status == "bad_segmentation",
        "needs_recrop": status == "needs_recrop",
        "skipped": status == "skipped",
        "notes": normalize_text(notes),
        "annotator": annotator,
        "human_verified": status in VERIFIED_STATUSES,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    problems = validate_record(record)
    if problems:
        raise ValueError("; ".join(problems))
    return record


def validate_record(record: dict) -> list[str]:
    """Rule violations for one annotation record (empty list = valid)."""
    problems = []
    status = record.get("status")
    text = record.get("transcription", "")
    if status not in STATUSES:
        problems.append(f"invalid status {status!r}")
        return problems
    if record.get("human_verified") and status not in VERIFIED_STATUSES:
        problems.append(f"human_verified=true with status {status!r}")
    if status == "ok" and not text:
        problems.append("status ok with empty transcription")
    if status == "blank" and text:
        problems.append("status blank with non-empty transcription")
    if status == "unreadable_full" and text != UNREADABLE_TOKEN:
        problems.append("status unreadable_full must be exactly the token")
    if text != unicodedata.normalize("NFC", text):
        problems.append("transcription is not NFC-normalized")
    if "�" in text or "�" in record.get("notes", ""):
        problems.append("replacement character (encoding damage)")
    return problems


def locked_against_overwrite(existing: dict | None, unlocked: bool = False) -> bool:
    """Owner-verified records are immutable in the UI unless deliberately
    unlocked for this sample (assisted-annotation campaign, 2026-07-17:
    protects ground truth from accidental one-click replacement)."""
    return bool(existing and existing.get("human_verified")) and not unlocked


def save_annotation(root: Path, record: dict) -> Path:
    path = annotation_path(root, record["split"], record["sample_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, ensure_ascii=False, indent=1),
                   encoding="utf-8")
    os.replace(tmp, path)
    return path


def load_annotation(root: Path, split: str, sample_id: str) -> dict | None:
    path = annotation_path(root, split, sample_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_all_annotations(root: Path, split: str) -> dict[str, dict]:
    d = Path(root) / "annotations" / split
    out = {}
    if d.exists():
        for f in sorted(d.glob("*.json")):
            out[f.stem] = json.loads(f.read_text(encoding="utf-8"))
    return out


def resume_index(samples: list[dict], annotations: dict[str, dict]) -> int:
    """First sample lacking a decisive annotation (draft/skip don't count)."""
    for i, s in enumerate(samples):
        rec = annotations.get(s["sample_id"])
        if rec is None or rec["status"] in ("draft", "skipped"):
            return i
    return max(0, len(samples) - 1)


def progress(samples: list[dict], annotations: dict[str, dict]) -> dict:
    done = flagged = 0
    for s in samples:
        rec = annotations.get(s["sample_id"])
        if rec is None:
            continue
        if rec["status"] in VERIFIED_STATUSES:
            done += 1
        elif rec["status"] in ("bad_segmentation", "needs_recrop", "skipped"):
            flagged += 1
    return {"total": len(samples), "verified": done, "flagged": flagged,
            "remaining": len(samples) - done - flagged}
