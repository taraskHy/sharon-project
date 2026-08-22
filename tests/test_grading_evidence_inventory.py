"""grade_primary evidence inventory (autograder/benchmark/datasets.py).

The answer crops a human grades come from the AUTHORITATIVE upstream line
inventory (evaluation/htr_pilot/splits/*.json, ``n_lines`` per cell) — one
image per recorded line, in line order — never from the subset of lines that
happen to carry an audited OCR transcription. Covers: the inventory loader,
cell evidence derivation (bench crops verified byte-identical to the upstream
line images), the frozen dataset being exactly what the builder derives now,
the deterministic in-place re-freeze (inputs untouched, refuses drift), and
the accuracy-metric exclusion of transcription-incomplete cases. Offline.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from autograder.benchmark.datasets import (EVIDENCE_LABEL_FIELDS, DatasetBuildError, audited_cells,
                                           evidence_label_fields, load_line_inventory, repair_grading_evidence)

REPO = Path(__file__).resolve().parents[1]
DATASET = REPO / "evaluation" / "model_selection" / "datasets" / "grade_primary"
HTR = REPO / "evaluation" / "htr_pilot"
pytestmark = pytest.mark.skipif(not (DATASET / "manifest.json").exists() or not (HTR / "splits").exists(),
                                reason="grade_primary dataset / HTR pilot package not present")

INCOMPLETE = ["e003_q1_r5", "e003_q2_r2", "e003_q2_r3", "e003_q2_r4", "e003_q2_r7",
              "e004_q2_r3", "e004_q2_r5", "e006_q2_r6", "e007_q1_r1"]


def _rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_line_inventory_records_every_upstream_line():
    inv = load_line_inventory()
    c = inv["e004_q2_r3"]
    assert c["n_lines"] == 2 and c["source"] == "evaluation/htr_pilot/splits/train.json"
    assert [l["sample_id"] for l in c["lines"]] == ["e004_q2_r3__l1", "e004_q2_r3__l2"]
    assert [l["annotation_status"] for l in c["lines"]] == ["ok", "bad_segmentation"]
    assert [l["human_verified"] for l in c["lines"]] == [True, False]
    assert all((REPO / "evaluation" / l["image"]).exists() for l in c["lines"])
    assert c["lines"][1]["image"] == "htr_pilot/images/e004/q2_r3_l2.png"
    # a known-good two-line cell: both lines audited upstream
    good = inv["e004_q2_r1"]
    assert good["n_lines"] == 2 and [l["annotation_status"] for l in good["lines"]] == ["ok", "ok"]
    # every cell's lines are exactly 1..n_lines (the loader refuses otherwise)
    for cell in inv.values():
        assert [l["line_index"] for l in cell["lines"]] == list(range(1, cell["n_lines"] + 1))


def test_audited_cells_take_evidence_from_the_inventory():
    cells = audited_cells()
    c = cells["e004_q2_r3"]
    assert c["complete"] and c["provenance_valid"] and c["items"] == ["hl_e004_q2_r3__l1"]
    assert c["evidence_kind"] == "line_crops" and c["line_count"] == 2
    assert c["evidence_images"] == ["hebrew_bench_v2/crops/hl_e004_q2_r3__l1.png", "htr_pilot/images/e004/q2_r3_l2.png"]
    assert c["transcription_complete"] is False and c["lines_without_audited_transcription"] == ["e004_q2_r3__l2"]
    assert c["evidence"][1]["transcription_status"] == "no_audited_transcription:bad_segmentation"
    good = cells["e004_q2_r1"]
    assert good["evidence_images"] == ["hebrew_bench_v2/crops/hl_e004_q2_r1__l1.png", "hebrew_bench_v2/crops/hl_e004_q2_r1__l2.png"]
    assert good["transcription_complete"] is True and good["line_count"] == 2
    cell = cells["e002_q1_r1"]
    assert cell["evidence_kind"] == "cell_crop" and cell["line_count"] == 1 and len(cell["evidence_images"]) == 1
    usable = {k: v for k, v in cells.items() if v["complete"] and v["provenance_valid"]}
    assert len(usable) == 67
    for cid, v in usable.items():
        assert len(v["evidence_images"]) == v["line_count"] == len(v["evidence"]), cid
        assert [e["index"] for e in v["evidence"]] == list(range(1, v["line_count"] + 1)), cid
        assert all((REPO / "evaluation" / img).exists() for img in v["evidence_images"]), cid
    assert sorted(cid for cid, v in usable.items() if not v["transcription_complete"]) == INCOMPLETE
    # membership rule unchanged: leading-gap cells stay excluded (13 of them today)
    assert sum(1 for v in cells.values() if not v["complete"]) == 13


def test_frozen_dataset_is_exactly_what_the_builder_derives():
    cells = audited_cells()
    labels = _rows(DATASET / "cases_labels.jsonl")
    assert len(labels) == 67
    for row in labels:
        block = {k: row[k] for k in EVIDENCE_LABEL_FIELDS}
        assert block == evidence_label_fields(cells[row["case_id"]]), row["case_id"]
        # the evidence block sits where the builder writes it (before max_score)
        keys = list(row)
        assert keys.index("lines_without_audited_transcription") + 1 == keys.index("max_score")
    man = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    assert man["extra"]["evidence_inventory"]["evidence_images"] == 91
    assert man["extra"]["evidence_inventory"]["transcription_incomplete_cases"] == INCOMPLETE
    assert man["revisions"][-1]["kind"] == "evidence_inventory_repair"
    assert man["revisions"][-1]["inputs_changed"] is False and man["revisions"][-1]["cases_evidence_changed"] == INCOMPLETE


def _old_shape(row: dict) -> dict:
    """The pre-repair label row: only the audited lines' bench crops as
    ``evidence_images``, at the same position, no other evidence fields."""
    out: dict = {}
    for k, v in row.items():
        if k == "evidence_images":
            out[k] = [e["image"] for e in row["evidence_lines"] if e["bench_item"]]
        elif k in EVIDENCE_LABEL_FIELDS:
            continue
        else:
            out[k] = v
    return out


def test_repair_is_deterministic_and_reproduces_the_frozen_file(tmp_path):
    man = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    previous_sha = man["revisions"][0]["previous_labels_sha256"]
    # 1. reconstruct the dataset as it was BEFORE the repair
    old = tmp_path / "grade_primary"
    old.mkdir()
    shutil.copy(DATASET / "cases_inputs.jsonl", old / "cases_inputs.jsonl")
    body = "".join(json.dumps(_old_shape(r), ensure_ascii=False) + "\n" for r in _rows(DATASET / "cases_labels.jsonl"))
    (old / "cases_labels.jsonl").write_bytes(body.encode("utf-8"))
    assert _sha(old / "cases_labels.jsonl") == previous_sha              # the reconstruction is byte-exact
    old_man = {k: v for k, v in man.items() if k != "revisions"}
    old_man["labels_sha256"] = previous_sha
    old_man["extra"] = {k: v for k, v in man["extra"].items() if k != "evidence_inventory"}
    (old / "manifest.json").write_text(json.dumps(old_man, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    # 2. dry run writes nothing
    dry = repair_grading_evidence(old, dry_run=True)
    assert dry["written"] is False and _sha(old / "cases_labels.jsonl") == previous_sha
    assert [c["case_id"] for c in dry["cases_evidence_changed"]] == INCOMPLETE and len(dry["rows_updated"]) == 67
    # 3. the repair reproduces today's frozen file byte for byte; inputs untouched
    res = repair_grading_evidence(old, now="2026-08-22 20:15:00")
    assert res["written"] is True
    assert (old / "cases_labels.jsonl").read_bytes() == (DATASET / "cases_labels.jsonl").read_bytes()
    assert _sha(old / "cases_inputs.jsonl") == man["inputs_sha256"]
    new_man = json.loads((old / "manifest.json").read_text(encoding="utf-8"))
    assert new_man["labels_sha256"] == man["labels_sha256"] and new_man["inputs_sha256"] == man["inputs_sha256"]
    assert new_man["revisions"][-1]["cases_evidence_changed"] == INCOMPLETE
    assert (old / "CHECKSUMS.sha256").read_text(encoding="utf-8") == (DATASET / "CHECKSUMS.sha256").read_text(encoding="utf-8")
    # 4. idempotent: a second repair changes nothing
    again = repair_grading_evidence(old)
    assert again["written"] is False and again["rows_updated"] == []
    # 5. a drifted dataset is refused, never "repaired"
    bad = tmp_path / "bad"
    shutil.copytree(old, bad)
    (bad / "cases_inputs.jsonl").write_bytes((bad / "cases_inputs.jsonl").read_bytes() + b"\n")
    with pytest.raises(DatasetBuildError, match="does not match the manifest hash"):
        repair_grading_evidence(bad)


def test_grade_adapter_excludes_transcription_incomplete_cases_from_accuracy():
    from autograder.benchmark.roles import GradeAdapter
    ad = GradeAdapter()
    base = {"split": "DEV", "component": "ALL", "schema_failure": False, "decision": "AUTO", "score": 3.0,
            "validation_ok": True, "evidence_failure": False}
    scored = [
        {**base, "case_id": "a", "transcription_complete": True, "label_score": 3.0, "exact": True, "abs_error": 0.0,
         "harmful_upgrade": False, "harmful_downgrade": False},
        {**base, "case_id": "b", "transcription_complete": False, "label_score": 1.0, "exact": False, "abs_error": 2.0,
         "harmful_upgrade": True, "harmful_downgrade": False},
        {**base, "case_id": "c", "transcription_complete": True},
    ]
    agg = ad.aggregate(scored, [])
    assert agg["cases"] == 3 and agg["labeled_excluded_transcription_incomplete"] == 1
    assert agg["exact_score_pct"] == 100.0 and agg["harmful_upgrades"] == 0      # only the complete labeled case counts
