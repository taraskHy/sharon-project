"""Tests for the HTR-pilot annotation workflow (lib, validator, split)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.htr_annotation_lib import (
    UNREADABLE_TOKEN, load_all_annotations, make_record, resume_index,
    save_annotation, validate_record,
)

import base64

PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def mini_package(root: Path, writers=None) -> dict:
    """Synthetic package: 2 samples per writer, valid by construction."""
    writers = writers or {"e003": "train", "e013": "val", "e016": "internal_test"}
    samples_by_split: dict[str, list] = {"train": [], "val": [], "internal_test": []}
    for wr, split in writers.items():
        img_dir = root / "images" / wr
        img_dir.mkdir(parents=True, exist_ok=True)
        for r in (1, 2):
            for name in (f"q1_r{r}_l1.png", f"q1_r{r}_cell_clean.png"):
                (img_dir / name).write_bytes(PNG_1PX)
            (img_dir / f"q1_r{r}_cell_orig.jpg").write_bytes(PNG_1PX)
            samples_by_split[split].append({
                "sample_id": f"{wr}_q1_r{r}__l1", "writer": wr, "split": split,
                "question": 1, "row": r, "line_index": 1, "n_lines": 1,
                "expected_blank": False,
                "images": {
                    "line": f"images/{wr}/q1_r{r}_l1.png",
                    "cell_clean": f"images/{wr}/q1_r{r}_cell_clean.png",
                    "cell_orig": f"images/{wr}/q1_r{r}_cell_orig.jpg",
                },
                "line_size": [1, 1],
            })
    (root / "splits").mkdir(parents=True, exist_ok=True)
    for split, recs in samples_by_split.items():
        (root / "splits" / f"{split}.json").write_text(
            json.dumps(recs, ensure_ascii=False), encoding="utf-8")
    for split in samples_by_split:
        (root / "annotations" / split).mkdir(parents=True, exist_ok=True)
    return samples_by_split


def run_validator(root: Path):
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "scripts/htr_annotation_validate.py",
         "--root", str(root)],
        capture_output=True, text=True, encoding="utf-8",
        cwd=Path(__file__).resolve().parents[1])
    return proc.returncode, proc.stdout


def test_split_assignment_is_writer_separated_and_excludes_e002():
    from scripts.htr_pilot_build import EXAM_FILES, SPLITS
    seen = [n for xs in SPLITS.values() for n in xs]
    assert len(seen) == len(set(seen)) == 16
    assert 2 not in seen and 2 not in EXAM_FILES
    assert SPLITS["train"] == list(range(3, 13))
    assert SPLITS["val"] == [13, 14, 15]
    assert SPLITS["internal_test"] == [16, 17, 18]


def test_record_rules_and_roundtrip(tmp_path):
    pkg = mini_package(tmp_path)
    sample = pkg["train"][0]
    rec = make_record(sample, "שלום עולם", "ok", notes="בסדר")
    assert rec["human_verified"] and not rec["unreadable"]
    save_annotation(tmp_path, rec)
    loaded = load_all_annotations(tmp_path, "train")
    assert loaded[sample["sample_id"]]["transcription"] == "שלום עולם"

    rec2 = make_record(sample, "whatever", "unreadable_full")
    assert rec2["transcription"] == UNREADABLE_TOKEN and rec2["human_verified"]
    rec3 = make_record(sample, "text stays", "bad_segmentation")
    assert not rec3["human_verified"]
    rec4 = make_record(sample, "", "blank")
    assert rec4["blank"] and rec4["human_verified"]
    with pytest.raises(ValueError):
        make_record(sample, "", "ok")  # empty ok transcription is invalid
    with pytest.raises(ValueError):
        make_record(sample, "x", "nonsense-status")


def test_verified_flag_cannot_accompany_bad_segmentation():
    rec = {"sample_id": "s", "split": "train", "status": "bad_segmentation",
           "transcription": "אבג", "human_verified": True, "notes": ""}
    problems = validate_record(rec)
    assert any("human_verified" in p for p in problems)


def test_resume_skips_decided_but_stops_at_draft_and_skip(tmp_path):
    pkg = mini_package(tmp_path)
    samples = pkg["train"]
    assert resume_index(samples, {}) == 0
    save_annotation(tmp_path, make_record(samples[0], "אבג", "ok"))
    ann = load_all_annotations(tmp_path, "train")
    assert resume_index(samples, ann) == 1
    save_annotation(tmp_path, make_record(samples[1], "טיוטה", "draft"))
    ann = load_all_annotations(tmp_path, "train")
    assert resume_index(samples, ann) == 1  # drafts still need a decision


def test_validator_passes_clean_package(tmp_path):
    mini_package(tmp_path)
    code, out = run_validator(tmp_path)
    assert code == 0 and "RESULT: PASS" in out


@pytest.mark.parametrize("corruption", [
    "duplicate_id", "missing_image", "leak_writer", "e002", "grade",
    "verified_badseg", "empty_ok", "heldout_writer",
])
def test_validator_catches_each_violation(tmp_path, corruption):
    pkg = mini_package(tmp_path)
    splits_dir = tmp_path / "splits"
    train = json.loads((splits_dir / "train.json").read_text(encoding="utf-8"))
    if corruption == "duplicate_id":
        train.append(dict(train[0]))
    elif corruption == "missing_image":
        train[0]["images"]["line"] = "images/e003/nope.png"
    elif corruption == "leak_writer":
        leak = dict(pkg["val"][0])
        leak["split"] = "train"
        train.append(leak)
    elif corruption == "e002":
        bad = json.loads(json.dumps(train[0]).replace("e003", "e002"))
        (tmp_path / "images" / "e002").mkdir(parents=True, exist_ok=True)
        for rel in bad["images"].values():
            (tmp_path / rel).write_bytes(PNG_1PX)
        train.append(bad)
    elif corruption == "grade":
        train[0]["notes"] = "from test/003_70.pdf"
    elif corruption == "heldout_writer":
        bad = json.loads(json.dumps(train[0]).replace("e003", "e050"))
        (tmp_path / "images" / "e050").mkdir(parents=True, exist_ok=True)
        for rel in bad["images"].values():
            (tmp_path / rel).write_bytes(PNG_1PX)
        train.append(bad)
    (splits_dir / "train.json").write_text(json.dumps(train, ensure_ascii=False),
                                           encoding="utf-8")
    if corruption == "verified_badseg":
        rec = make_record(pkg["train"][0], "אבג", "bad_segmentation")
        rec["human_verified"] = True  # simulate hand-tampered record
        save_annotation(tmp_path, rec)
    if corruption == "empty_ok":
        rec = make_record(pkg["train"][0], "אבג", "ok")
        rec["transcription"] = ""
        save_annotation(tmp_path, rec)
    code, out = run_validator(tmp_path)
    assert code == 1 and "RESULT: FAIL" in out


def test_real_package_validates_if_built():
    root = Path(__file__).resolve().parents[1] / "evaluation/htr_pilot"
    if not (root / "splits" / "train.json").exists():
        pytest.skip("pilot package not built yet")
    code, out = run_validator(root)
    assert code == 0, out
