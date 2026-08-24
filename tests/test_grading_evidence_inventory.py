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
from autograder.benchmark.evidence_repairs import REPAIR_SOURCE
from tests.prerepair import DATASET, REPO, pre_repair_rows, repair_store, repaired_cases, repaired_line_ids

HTR = REPO / "evaluation" / "htr_pilot"
pytestmark = pytest.mark.skipif(not (DATASET / "manifest.json").exists() or not (HTR / "splits").exists(),
                                reason="grade_primary dataset / HTR pilot package not present")

#: the cells the OCR benchmark never audited every line of. The builder still
#: reports them incomplete — it derives from the frozen benchmark, which the
#: manual repair deliberately did not touch. The current dataset resolves them
#: with a supplemental HUMAN repair layer that lives outside that benchmark.
INCOMPLETE = ["e003_q1_r5", "e003_q2_r2", "e003_q2_r3", "e003_q2_r4", "e003_q2_r7",
              "e004_q2_r3", "e004_q2_r5", "e006_q2_r6", "e007_q1_r1"]


def _rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _body(records: list[dict]) -> bytes:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records).encode("utf-8")


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


def test_the_builder_alone_still_derives_the_pre_repair_evidence():
    """The builder reads the FROZEN OCR benchmark, which the manual repair never
    touched — so on its own it must still report the nine cells as incomplete.
    That is the base layer, and it stays reproducible."""
    cells = audited_cells()
    _, pre_labels = pre_repair_rows()
    assert len(pre_labels) == 67
    for row in pre_labels:
        block = {k: row[k] for k in EVIDENCE_LABEL_FIELDS}
        assert block == evidence_label_fields(cells[row["case_id"]]), row["case_id"]
        keys = list(row)
        assert keys.index("lines_without_audited_transcription") + 1 == keys.index("max_score")
    assert sorted(r["case_id"] for r in pre_labels if r["transcription_complete"] is False) == INCOMPLETE


def test_frozen_dataset_is_the_builder_output_plus_the_manual_repair_layer():
    """The current dataset = what the builder derives + the owner's supplemental
    human repairs. The two layers are composed here, never conflated: the base
    block is compared field by field, and only a repaired line is allowed to
    differ — in exactly the ways apply_repairs records."""
    cells = audited_cells()
    store = repair_store()
    labels = _rows(DATASET / "cases_labels.jsonl")
    assert len(labels) == 67
    for row in labels:
        base = evidence_label_fields(cells[row["case_id"]])
        repaired = list(row.get("evidence_repairs") or [])
        keys = list(row)
        assert keys.index("lines_without_audited_transcription") + 1 == keys.index("max_score")
        if not repaired:
            assert {k: row[k] for k in EVIDENCE_LABEL_FIELDS} == base, row["case_id"]
            continue
        # the base layer said "incomplete"; the repair layer resolves it
        assert base["transcription_complete"] is False
        assert sorted(base["lines_without_audited_transcription"]) == sorted(repaired)
        assert row["transcription_complete"] is True and row["lines_without_audited_transcription"] == []
        assert row["evidence_kind"] == base["evidence_kind"] and row["line_count"] == base["line_count"]
        assert row["line_inventory_source"] == base["line_inventory_source"]
        for cur, was in zip(row["evidence_lines"], base["evidence_lines"]):
            if cur["sample_id"] not in repaired:
                assert cur == was, row["case_id"]
                continue
            rec = store[cur["sample_id"]]
            assert {k: v for k, v in cur.items() if k in was and k not in ("image", "transcription_status")} \
                == {k: v for k, v in was.items() if k not in ("image", "transcription_status")}
            assert cur["original_image"] == was["image"], "the builder's crop path is kept, not lost"
            assert cur["original_transcription_status"] == was["transcription_status"]
            assert cur["transcription_status"] == f"{REPAIR_SOURCE}:{rec['disposition']}"
            assert cur["image"].endswith(rec["crop_path"]) and (REPO / "evaluation" / cur["image"]).exists()
    assert sorted(r["case_id"] for r in labels if r.get("evidence_repairs")) == INCOMPLETE == repaired_cases()
    man = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    # 91 recorded lines; 8 of them ruled no-text artifacts, so 83 EFFECTIVE images
    assert sum(r["line_count"] for r in labels) == 91
    assert man["extra"]["evidence_inventory"]["evidence_images"] == 83
    assert man["extra"]["evidence_inventory"]["transcription_incomplete_cases"] == []
    assert man["revisions"][0]["kind"] == "evidence_inventory_repair"
    assert man["revisions"][0]["inputs_changed"] is False
    assert man["revisions"][0]["cases_evidence_changed"] == INCOMPLETE
    rev = next(r for r in man["revisions"] if r["kind"] == REPAIR_SOURCE)
    assert sorted(rev["lines_repaired"]) == repaired_line_ids()


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
    """Historical reproducibility of the EVIDENCE-INVENTORY repair (revision 0),
    which predates the manual human repair and must stay exactly reproducible.

    Both ends of that revision are taken from the manifest itself, so the later
    manual repair cannot drag this test along with it: the input is
    `revisions[0].previous_labels_sha256` and the output is
    `revisions[0].labels_sha256` — NOT whatever the file happens to hold today.
    """
    man = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    rev0 = man["revisions"][0]
    previous_sha, produced_sha = rev0["previous_labels_sha256"], rev0["labels_sha256"]
    pre_inputs, pre_labels = pre_repair_rows()          # strip the manual layer first
    assert _sha_bytes(_body(pre_labels)) == produced_sha, "the pre-manual-repair state is revision 0's output"
    # 1. reconstruct the dataset as it was BEFORE the evidence-inventory repair
    old = tmp_path / "grade_primary"
    old.mkdir()
    (old / "cases_inputs.jsonl").write_bytes(_body(pre_inputs))
    (old / "cases_labels.jsonl").write_bytes(_body([_old_shape(r) for r in pre_labels]))
    assert _sha(old / "cases_labels.jsonl") == previous_sha              # the reconstruction is byte-exact
    old_man = {k: v for k, v in man.items() if k != "revisions"}
    old_man["labels_sha256"] = previous_sha
    old_man["inputs_sha256"] = rev0["inputs_sha256"]
    old_man["extra"] = {k: v for k, v in man["extra"].items() if k != "evidence_inventory"}
    (old / "manifest.json").write_text(json.dumps(old_man, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
    # 2. dry run writes nothing
    dry = repair_grading_evidence(old, dry_run=True)
    assert dry["written"] is False and _sha(old / "cases_labels.jsonl") == previous_sha
    assert [c["case_id"] for c in dry["cases_evidence_changed"]] == INCOMPLETE and len(dry["rows_updated"]) == 67
    # 3. the repair reproduces revision 0's output byte for byte; inputs untouched
    res = repair_grading_evidence(old, now="2026-08-22 20:15:00")
    assert res["written"] is True
    assert _sha(old / "cases_labels.jsonl") == produced_sha
    assert (old / "cases_labels.jsonl").read_bytes() == _body(pre_labels)
    assert _sha(old / "cases_inputs.jsonl") == rev0["inputs_sha256"]
    new_man = json.loads((old / "manifest.json").read_text(encoding="utf-8"))
    assert new_man["labels_sha256"] == produced_sha and new_man["inputs_sha256"] == rev0["inputs_sha256"]
    assert new_man["revisions"][-1]["cases_evidence_changed"] == INCOMPLETE
    assert new_man["revisions"][-1] == rev0, "the recorded revision is reproduced exactly"
    # 4. idempotent: a second repair changes nothing
    again = repair_grading_evidence(old)
    assert again["written"] is False and again["rows_updated"] == []
    # 5. a drifted dataset is refused, never "repaired"
    bad = tmp_path / "bad"
    shutil.copytree(old, bad)
    (bad / "cases_inputs.jsonl").write_bytes((bad / "cases_inputs.jsonl").read_bytes() + b"\n")
    with pytest.raises(DatasetBuildError, match="does not match the manifest hash"):
        repair_grading_evidence(bad)


def test_the_evidence_inventory_repair_never_saw_the_manual_layer():
    """Revision 0 predates the manual repair: it must not mention it, and its
    output hash must be the manual repair's INPUT hash. The chain is unbroken."""
    man = json.loads((DATASET / "manifest.json").read_text(encoding="utf-8"))
    rev0 = man["revisions"][0]
    rev1 = next(r for r in man["revisions"] if r["kind"] == REPAIR_SOURCE)
    assert rev0["kind"] == "evidence_inventory_repair"
    assert rev0["labels_sha256"] == rev1["previous_labels_sha256"]
    assert rev0["inputs_sha256"] == rev1["previous_inputs_sha256"]
    assert REPAIR_SOURCE not in json.dumps(rev0, ensure_ascii=False)
    _, pre_labels = pre_repair_rows()
    assert not any("evidence_repairs" in r for r in pre_labels)


def test_grade_adapter_excludes_transcription_incomplete_cases_from_accuracy():
    from autograder.benchmark.roles import GradeAdapter
    ad = GradeAdapter()
    base = {"split": "DEV", "component": "ALL", "schema_failure": False, "decision": "AUTO", "score": 3.0,
            "validation_ok": True, "evidence_failure": False}
    scored = [
        {**base, "case_id": "a", "transcription_complete": True, "label_score": 3.0,
         "label_verdict": "partially_valid", "predicted_verdict": "partially_valid",
         "verdict_exact": True, "final_exact": True, "final_abs_error": 0.0,
         "harmful_upgrade": False, "harmful_downgrade": False},
        {**base, "case_id": "b", "transcription_complete": False, "label_score": 1.0,
         "label_verdict": "partially_valid", "predicted_verdict": "valid",
         "verdict_exact": False, "final_exact": False, "final_abs_error": 2.0,
         "harmful_upgrade": True, "harmful_downgrade": False},
        {**base, "case_id": "c", "transcription_complete": True},
    ]
    agg = ad.aggregate(scored, [])
    assert agg["cases"] == 3 and agg["labeled_excluded_transcription_incomplete"] == 1
    # only the complete labeled case counts, on the verdict target and on the
    # derived final-score metric alike
    assert agg["verdict_cases"] == 1 and agg["verdict_exact_pct"] == 100.0
    assert agg["final_score_exact_pct"] == 100.0 and agg["harmful_upgrades"] == 0


def test_evidence_images_are_the_effective_evidence_not_the_full_line_list():
    """`evidence_lines` is the historical record of every recorded line;
    `evidence_images` is what a grader is actually shown. They agree on ordinary
    rows and differ ONLY where a human ruled a line to be a segmentation
    artifact — that sliver is not an answer image, though its crop, hash and
    status remain recoverable."""
    labels = _rows(DATASET / "cases_labels.jsonl")
    store = repair_store()
    narrowed = {}
    for row in labels:
        lines = [e["image"] for e in row["evidence_lines"]]
        artifacts = [e for e in row["evidence_lines"]
                     if (e.get("repair") or {}).get("disposition") == "no_text_segmentation_artifact"]
        assert row["evidence_images"] == [e["image"] for e in row["evidence_lines"] if e not in artifacts],             row["case_id"]
        assert all((REPO / "evaluation" / img).exists() for img in row["evidence_images"]), row["case_id"]
        if artifacts:
            narrowed[row["case_id"]] = artifacts
            assert row["evidence_images"] != lines
            for e in artifacts:                      # excluded, but never lost
                assert e["image"] not in row["evidence_images"]
                assert e["original_image"] not in row["evidence_images"]
                assert (REPO / "evaluation" / e["original_image"]).exists()
                assert e["repair"]["crop_sha256"] == store[e["sample_id"]]["crop_sha256"]
        else:
            assert row["evidence_images"] == lines, row["case_id"]
    assert len(narrowed) == 8, "the eight artifact rulings, and nothing else, narrow the effective evidence"
    assert sum(len(v) for v in narrowed.values()) == 8
    assert sum(len(r["evidence_images"]) for r in labels) == 83
