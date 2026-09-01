"""Manual GRADE_PRIMARY evidence repairs — the integrity rules, as tests.

The checked-in dataset is now POST-repair (the owner made nine decisions and
applied them), so the repair PROCESS is exercised against the reconstructed
PRE-repair dataset from tests/prerepair.py — hash-verified against the sha256
pair the manifest revision recorded, so it is the real earlier state and not a
fiction. The live dataset is only ever read; a separate section at the bottom
asserts its post-repair invariants.

The geometry, the crops and the frozen OCR benchmark are the real ones (read
only). No OCR, no model, no network: the only image work is exact sub-image
location and rectangle cropping.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from autograder.benchmark.evidence_repairs import (
    DISPOSITIONS, REPAIR_SOURCE, RepairError, RepairStore, apply_repairs, assert_frozen_bench_unchanged,
    case_geometry, expected_repairs, frozen_bench_hashes, locate_exact, render_band, repair_status,
    suggested_band, verify_repairs)
from tests.prerepair import DATASET as REAL_DATASET


def _without_verdict(row: dict) -> dict:
    """A label row as the EVIDENCE-REPAIR layer produced it.

    The live dataset carries a later revision (verdict_target, 2026-08-24)
    that adds the derived explanation-verdict block. These tests rebuild the
    dataset only up to the evidence-repair layer, so the comparison has to
    drop the fields that layer never wrote — otherwise a legitimate later
    revision looks like a repair-layer regression.
    """
    from autograder.benchmark.datasets import VERDICT_LABEL_FIELDS

    return {k: v for k, v in row.items() if k not in VERDICT_LABEL_FIELDS}
from tests.prerepair import (DATASET_FILES, REPO, build_pre_repair_dataset, copy_repair_store, repair_store,
                             repaired_cases, repaired_line_ids)

EVAL_ROOT = REPO / "evaluation"

pytestmark = pytest.mark.skipif(not (REAL_DATASET / "manifest.json").exists(),
                                reason="grade_primary dataset is not built here")


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    """The dataset in its PRE-repair state — the repair process needs work to do."""
    return build_pre_repair_dataset(tmp_path / "grade_primary")


def _copy_dataset(dest: Path) -> Path:
    """A second, independent pre-repair dataset (for determinism comparisons)."""
    return build_pre_repair_dataset(dest)


def _rows(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def _label_map(d: Path) -> dict[str, dict]:
    return {r["case_id"]: r for r in _rows(d / "cases_labels.jsonl")}


def _reseal(d: Path) -> None:
    """Re-hash the label file into the manifest after a test edits ground truth."""
    body = (d / "cases_labels.jsonl").read_bytes()
    man = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    man["labels_sha256"] = hashlib.sha256(body).hexdigest()
    (d / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")


def _write_labels(d: Path, rows: list[dict]) -> None:
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    (d / "cases_labels.jsonl").write_text(body, encoding="utf-8", newline="\n")
    _reseal(d)


def _repair(store: RepairStore, item: dict, *, text: str | None = None, y: tuple[int, int] | None = None,
            disposition: str = "transcribed", verified_by: str = "tester") -> dict:
    """Record one repair exactly the way the UI does: a band of the proven cell."""
    geo = case_geometry(item["case_id"], evaluation_root=EVAL_ROOT)
    band = {"y0": y[0], "y1": y[1]} if y else suggested_band(geo, item["line_id"])
    png = render_band(geo, band["y0"], band["y1"])
    line = next(l for l in geo["lines"] if l["sample_id"] == item["line_id"])
    body = "" if disposition != "transcribed" else (text if text is not None else "טקסט " + item["line_id"])
    return store.save(
        case_id=item["case_id"], line_id=item["line_id"], disposition=disposition,
        transcription=body, verified_by=verified_by, crop_png=png,
        crop_geometry={"cell_image": geo["cell_image"], "cell_sha256": geo["cell_sha256"],
                       "x0": 0, "x1": geo["cell_width"], **band},
        original_crop={"image": line["image"], "sha256": line["image_sha256"],
                       "y0": line["y0"], "y1": line["y1"], "status": line["annotation_status"]},
        line_index=line["line_index"], line_count=geo["n_lines"], now="2026-08-23 00:00:00")


def _repair_all(d: Path, **kw) -> RepairStore:
    store = RepairStore(d)
    for item in expected_repairs(d):
        _repair(store, item, **kw)
    return store


# --------------------------------------------------------------- expectation --

def test_expected_repairs_are_derived_from_the_dataset_not_hardcoded(dataset: Path):
    exp = expected_repairs(dataset)
    labels = _label_map(dataset)
    declared = sorted(sid for r in labels.values() for sid in (r.get("lines_without_audited_transcription") or []))
    assert [e["line_id"] for e in exp] == declared
    assert len(exp) == 9, "exactly nine supplemental lines are expected before completion"
    assert {e["case_id"] for e in exp} == {r["case_id"] for r in labels.values()
                                           if r.get("transcription_complete") is False}


def test_every_expected_line_belongs_to_a_case_that_is_incomplete(dataset: Path):
    labels = _label_map(dataset)
    for e in expected_repairs(dataset):
        row = labels[e["case_id"]]
        assert row["transcription_complete"] is False
        assert e["line_id"].startswith(e["case_id"] + "__")
        assert e["line_id"] in row["lines_without_audited_transcription"]
        assert e["applied"] is False


# ------------------------------------------------------------------ geometry --

def test_geometry_is_proven_pixel_exact_for_every_expected_case(dataset: Path):
    for e in expected_repairs(dataset):
        geo = case_geometry(e["case_id"], evaluation_root=EVAL_ROOT)
        assert geo["n_lines"] == len(geo["lines"]) >= 2
        for line in geo["lines"]:
            assert 0 <= line["y0"] < line["y1"] <= geo["cell_height"]
            assert line["height"] == line["y1"] - line["y0"]
        ys = [l["y0"] for l in geo["lines"]]
        assert ys == sorted(ys), "recorded lines run top-to-bottom inside the cell"


def test_geometry_refuses_a_case_it_cannot_prove(dataset: Path):
    with pytest.raises(RepairError):
        case_geometry("e999_q9_r9", evaluation_root=EVAL_ROOT)


def test_locate_exact_refuses_a_crop_that_is_not_in_the_cell(dataset: Path):
    from autograder.benchmark.evidence_repairs import _png_array
    e = expected_repairs(dataset)[0]
    geo = case_geometry(e["case_id"], evaluation_root=EVAL_ROOT)
    cell = _png_array(Path(geo["cell_image_abs"]))
    band = cell[10:40].copy()
    assert locate_exact(cell, band) == 10
    band[0, 0, 0] = (int(band[0, 0, 0]) + 40) % 256           # one pixel off is not a match
    assert locate_exact(cell, band) is None


def test_render_band_is_deterministic_and_band_specific(dataset: Path):
    e = expected_repairs(dataset)[0]
    geo = case_geometry(e["case_id"], evaluation_root=EVAL_ROOT)
    b = suggested_band(geo, e["line_id"])
    assert render_band(geo, b["y0"], b["y1"]) == render_band(geo, b["y0"], b["y1"])
    assert render_band(geo, b["y0"], b["y1"], 0, geo["cell_width"]) == render_band(geo, b["y0"], b["y1"])
    assert render_band(geo, b["y0"], b["y1"]) != render_band(geo, b["y0"] + 3, b["y1"])
    with pytest.raises(RepairError):
        render_band(geo, 50, 50)


def test_suggested_band_never_overlaps_an_audited_line(dataset: Path):
    for e in expected_repairs(dataset):
        geo = case_geometry(e["case_id"], evaluation_root=EVAL_ROOT)
        b = suggested_band(geo, e["line_id"])
        assert b["y1"] > b["y0"]
        if not geo["uncovered_bands"]:
            continue
        for line in geo["lines"]:
            if line["audited"]:
                assert not (b["y0"] < line["y1"] and line["y0"] < b["y1"]), \
                    e["line_id"] + ": the suggested band overlaps audited line " + line["sample_id"]


# --------------------------------------------------------------------- store --

def test_store_rejects_a_line_from_another_case(dataset: Path):
    exp = expected_repairs(dataset)
    store = RepairStore(dataset)
    geo = case_geometry(exp[0]["case_id"], evaluation_root=EVAL_ROOT)
    png = render_band(geo, 0, 20)
    with pytest.raises(RepairError, match="does not belong"):
        store.save(case_id=exp[1]["case_id"], line_id=exp[0]["line_id"], transcription="x",
                   verified_by="t", crop_png=png, original_crop={"image": "a.png", "sha256": "0" * 64})
    with pytest.raises(RepairError, match="not a recorded line"):     # geometry refuses even earlier
        suggested_band(geo, exp[1]["line_id"])
    assert not store.records()


def test_store_rejects_inadmissible_repairs(dataset: Path):
    item = expected_repairs(dataset)[0]
    store = RepairStore(dataset)
    geo = case_geometry(item["case_id"], evaluation_root=EVAL_ROOT)
    png = render_band(geo, 0, 20)
    base = dict(case_id=item["case_id"], line_id=item["line_id"], verified_by="t", crop_png=png,
                original_crop={"image": "a.png", "sha256": "0" * 64})
    with pytest.raises(RepairError, match="handwritten text"):
        store.save(**base, transcription="   ")
    with pytest.raises(RepairError, match="carries no transcription"):
        store.save(**base, transcription="x", disposition="no_text_segmentation_artifact")
    with pytest.raises(RepairError, match="who verified"):
        _repair(store, item, verified_by="  ")
    with pytest.raises(RepairError, match="unknown disposition"):
        _repair(store, item, disposition="guessed")
    line = next(l for l in geo["lines"] if l["sample_id"] == item["line_id"])
    oc = {"image": line["image"], "sha256": line["image_sha256"], "status": line["annotation_status"]}
    with pytest.raises(RepairError, match="persist the crop"):
        store.save(case_id=item["case_id"], line_id=item["line_id"], transcription="x", verified_by="t",
                   original_crop=oc)
    with pytest.raises(RepairError, match="crop it replaces"):
        store.save(case_id=item["case_id"], line_id=item["line_id"], transcription="x", verified_by="t",
                   crop_png=png)
    assert not store.records(), "no half-written record survives a refusal"


def test_a_line_can_only_have_one_record(dataset: Path):
    item = expected_repairs(dataset)[0]
    store = RepairStore(dataset)
    first = _repair(store, item, text="ראשון")
    _repair(store, item, text="שני")
    assert len(store.path.read_text(encoding="utf-8").strip().splitlines()) == 1
    assert store.get(item["line_id"])["transcription"] == "שני"
    assert store.get(item["line_id"])["created_at"] == first["created_at"], "the first entry keeps its timestamp"
    assert not verify_repairs(dataset, evaluation_root=EVAL_ROOT)["problems"]


def test_store_is_deterministic_regardless_of_entry_order(dataset: Path, tmp_path: Path):
    a = _repair_all(dataset).sha256()
    other = _copy_dataset(tmp_path / "again")
    for item in reversed(expected_repairs(other)):                 # entered in a different order
        _repair(RepairStore(other), item)
    assert RepairStore(other).sha256() == a


def test_a_repair_persists_its_crop_and_geometry(dataset: Path):
    item = expected_repairs(dataset)[0]
    rec = _repair(RepairStore(dataset), item)
    crop = dataset / rec["crop_path"]
    assert crop.exists() and _sha(crop) == rec["crop_sha256"]
    geo = case_geometry(item["case_id"], evaluation_root=EVAL_ROOT)
    g = rec["crop_geometry"]
    assert render_band(geo, g["y0"], g["y1"], g["x0"], g["x1"]) == crop.read_bytes(), "the crop re-derives exactly"
    assert rec["human_verified"] is True and rec["source"] == REPAIR_SOURCE
    assert rec["disposition"] in DISPOSITIONS
    assert rec["original_crop"] is None or "sha256" in rec["original_crop"]


def test_removing_a_repair_puts_the_line_back_on_the_worklist(dataset: Path):
    item = expected_repairs(dataset)[0]
    store = RepairStore(dataset)
    _repair(store, item)
    assert repair_status(dataset)["repaired"] == 1
    assert store.delete(item["line_id"]) is True
    assert store.delete(item["line_id"]) is False
    st = repair_status(dataset)
    assert st["repaired"] == 0 and item["line_id"] in st["remaining"]


# ----------------------------------------------------------------- integrity --

def test_verify_catches_a_tampered_crop(dataset: Path):
    item = expected_repairs(dataset)[0]
    rec = _repair(RepairStore(dataset), item)
    (dataset / rec["crop_path"]).write_bytes(b"not the crop")
    problems = verify_repairs(dataset, evaluation_root=EVAL_ROOT)["problems"]
    assert any("hash does not match" in p["problem"] for p in problems)


def test_verify_catches_a_missing_crop(dataset: Path):
    item = expected_repairs(dataset)[0]
    rec = _repair(RepairStore(dataset), item)
    (dataset / rec["crop_path"]).unlink()
    problems = verify_repairs(dataset, evaluation_root=EVAL_ROOT)["problems"]
    assert any("crop file missing" in p["problem"] for p in problems)


def test_verify_catches_geometry_that_does_not_re_derive_the_crop(dataset: Path):
    item = expected_repairs(dataset)[0]
    store = RepairStore(dataset)
    rec = dict(_repair(store, item))
    rec["crop_geometry"] = {**rec["crop_geometry"], "y0": rec["crop_geometry"]["y0"] + 5}
    store._rewrite({rec["line_id"]: rec})
    problems = verify_repairs(dataset, evaluation_root=EVAL_ROOT)["problems"]
    assert any("does not re-derive" in p["problem"] for p in problems)


def test_verify_catches_duplicate_line_ids_and_stray_records(dataset: Path):
    item = expected_repairs(dataset)[0]
    store = RepairStore(dataset)
    rec = _repair(store, item)
    with store.path.open("a", encoding="utf-8", newline="\n") as f:      # hand-appended duplicate + stray
        f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        f.write(json.dumps({**rec, "line_id": "e999_q9_r9__l2", "case_id": "e999_q9_r9"},
                           ensure_ascii=False, sort_keys=True) + "\n")
    problems = verify_repairs(dataset, evaluation_root=EVAL_ROOT)["problems"]
    assert any(p["problem"].startswith("duplicate line id") for p in problems)
    assert any("not an expected repair" in p["problem"] for p in problems)


def test_verify_catches_an_unverified_record(dataset: Path):
    item = expected_repairs(dataset)[0]
    store = RepairStore(dataset)
    rec = _repair(store, item)
    store._rewrite({rec["line_id"]: {**rec, "human_verified": False}})
    problems = verify_repairs(dataset, evaluation_root=EVAL_ROOT)["problems"]
    assert any(p["problem"] == "not human_verified" for p in problems)


def test_verify_catches_a_line_id_that_does_not_match_its_case(dataset: Path):
    item = expected_repairs(dataset)[0]
    store = RepairStore(dataset)
    rec = _repair(store, item)
    store._rewrite({rec["line_id"]: {**rec, "case_id": "e005_q1_r1"}})
    problems = verify_repairs(dataset, evaluation_root=EVAL_ROOT)["problems"]
    assert any("case mismatch" in p["problem"] for p in problems)
    assert any("does not belong to its case" in p["problem"] for p in problems)


def test_frozen_bench_hash_guard_actually_fires(tmp_path: Path):
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "items.json").write_text('{"items": []}', encoding="utf-8")
    before = frozen_bench_hashes(bench)
    assert assert_frozen_bench_unchanged(before, bench) == before
    (bench / "items.json").write_text('{"items": [1]}', encoding="utf-8")
    with pytest.raises(RepairError, match="frozen OCR benchmark changed"):
        assert_frozen_bench_unchanged(before, bench)


# ------------------------------------------------------------------ applying --

def test_apply_refuses_while_repairs_are_missing(dataset: Path):
    before = (_sha(dataset / "cases_inputs.jsonl"), _sha(dataset / "cases_labels.jsonl"))
    _repair(RepairStore(dataset), expected_repairs(dataset)[0])
    with pytest.raises(RepairError, match="still missing"):
        apply_repairs(dataset, evaluation_root=EVAL_ROOT)
    assert (_sha(dataset / "cases_inputs.jsonl"), _sha(dataset / "cases_labels.jsonl")) == before


def test_apply_refuses_an_inadmissible_store(dataset: Path):
    store = _repair_all(dataset)
    rec = store.get(expected_repairs(dataset)[0]["line_id"])
    (dataset / rec["crop_path"]).unlink()
    with pytest.raises(RepairError, match="not admissible"):
        apply_repairs(dataset, evaluation_root=EVAL_ROOT)


def test_apply_refuses_a_drifted_dataset(dataset: Path):
    from autograder.benchmark.datasets import DatasetBuildError
    _repair_all(dataset)
    with (dataset / "cases_inputs.jsonl").open("a", encoding="utf-8") as f:
        f.write("\n")
    with pytest.raises(DatasetBuildError, match="do not match the manifest"):
        apply_repairs(dataset, evaluation_root=EVAL_ROOT)


def test_dry_run_changes_nothing(dataset: Path):
    _repair_all(dataset)
    before = {p.name: _sha(p) for p in (dataset / n for n in DATASET_FILES)}
    exp = expected_repairs(dataset)
    out = apply_repairs(dataset, evaluation_root=EVAL_ROOT, dry_run=True)
    assert out["written"] is False and out["inputs_changed"] is True
    assert out["lines_repaired"] == sorted(e["line_id"] for e in exp)          # derived, not a literal
    assert {c["case_id"] for c in out["cases_changed"]} == {e["case_id"] for e in exp}
    assert len(out["lines_repaired"]) == 9 and len(out["cases_changed"]) == 9
    assert {p.name: _sha(p) for p in (dataset / n for n in DATASET_FILES)} == before


def test_incomplete_repairs_cannot_make_a_case_scorable(dataset: Path):
    """A partial apply completes ONLY the cases whose lines are all repaired."""
    exp = expected_repairs(dataset)
    store = RepairStore(dataset)
    done = exp[:3]
    for item in done:
        _repair(store, item)
    out = apply_repairs(dataset, evaluation_root=EVAL_ROOT, allow_partial=True)
    labels = _label_map(dataset)
    assert out["written"] is True
    for item in done:
        assert labels[item["case_id"]]["transcription_complete"] is True
    for item in exp[3:]:
        row = labels[item["case_id"]]
        assert row["transcription_complete"] is False
        assert item["line_id"] in row["lines_without_audited_transcription"]
    assert sum(1 for r in labels.values() if r["transcription_complete"] is False) == 6
    # the MIXED state — three applied, six still pending — is accounted for exactly
    mixed = {e["line_id"]: e["applied"] for e in expected_repairs(dataset)}
    assert len(mixed) == 9
    assert [lid for lid, ap in mixed.items() if ap] == sorted(i["line_id"] for i in done)
    assert [lid for lid, ap in mixed.items() if not ap] == sorted(i["line_id"] for i in exp[3:])
    st = repair_status(dataset)
    assert st["repaired"] == 3 and st["remaining"] == sorted(i["line_id"] for i in exp[3:])
    assert st["complete"] is False and st["unexpected_records"] == []
    assert verify_repairs(dataset, evaluation_root=EVAL_ROOT)["problems"] == []


def test_complete_repairs_make_every_case_scorable(dataset: Path):
    _repair_all(dataset)
    out = apply_repairs(dataset, evaluation_root=EVAL_ROOT)
    labels = _label_map(dataset)
    assert out["written"] is True and len(labels) == 67
    assert all(r["transcription_complete"] is True for r in labels.values())
    assert all(not r["lines_without_audited_transcription"] for r in labels.values())
    exp = expected_repairs(dataset)
    assert len(exp) == 9 and all(e["applied"] for e in exp), "applied repairs stay accounted for"
    st = repair_status(dataset)
    assert st["complete"] is True and st["remaining"] == [] and st["unexpected_records"] == []
    assert verify_repairs(dataset, evaluation_root=EVAL_ROOT)["ok"] is True


def test_repaired_text_lands_in_authoritative_line_order(dataset: Path):
    exp = expected_repairs(dataset)
    store = RepairStore(dataset)
    marks = {e["line_id"]: "MARK-" + str(i) for i, e in enumerate(exp)}
    for item in exp:
        _repair(store, item, text=marks[item["line_id"]])
    before_inputs = {r["case_id"]: r["transcription"] for r in _rows(dataset / "cases_inputs.jsonl")}
    apply_repairs(dataset, evaluation_root=EVAL_ROOT)
    inputs = {r["case_id"]: r["transcription"] for r in _rows(dataset / "cases_inputs.jsonl")}
    labels = _label_map(dataset)
    for item in exp:
        cid, lid = item["case_id"], item["line_id"]
        row = labels[cid]
        text_lines = inputs[cid].split("\n")
        assert len(text_lines) == row["line_count"], cid + ": one text line per recorded line"
        idx = next(e["index"] for e in row["evidence_lines"] if e["sample_id"] == lid)
        assert text_lines[idx - 1] == marks[lid], cid + ": the repaired line sits at its own position"
        assert [t for t in text_lines if t != marks[lid]] == before_inputs[cid].split("\n"), \
            "audited lines keep their text and their order"
        assert row["evidence_repairs"] == [lid]
        assert REPAIR_SOURCE in row["transcription_source"]
        line = next(e for e in row["evidence_lines"] if e["sample_id"] == lid)
        assert line["repair"]["source"] == REPAIR_SOURCE and line["repair"]["verified_by"] == "tester"
        assert line["original_image"] == item["image"], "the historic crop path is preserved, not overwritten"
        assert line["original_transcription_status"] == item["transcription_status"]
        assert (dataset.parent.parent / line["image"]).exists() or line["image"] == line["original_image"]


def test_a_segmentation_artifact_adds_no_text(dataset: Path):
    exp = expected_repairs(dataset)
    store = RepairStore(dataset)
    for i, item in enumerate(exp):
        _repair(store, item, disposition="no_text_segmentation_artifact" if i == 0 else "transcribed")
    before = {r["case_id"]: r["transcription"] for r in _rows(dataset / "cases_inputs.jsonl")}
    apply_repairs(dataset, evaluation_root=EVAL_ROOT)
    inputs = {r["case_id"]: r["transcription"] for r in _rows(dataset / "cases_inputs.jsonl")}
    cid = exp[0]["case_id"]
    assert inputs[cid] == before[cid], "an artifact contributes no invented text"
    labels = _label_map(dataset)
    assert labels[cid]["transcription_complete"] is True
    line = next(e for e in labels[cid]["evidence_lines"] if e["sample_id"] == exp[0]["line_id"])
    assert line["transcription_status"] == REPAIR_SOURCE + ":no_text_segmentation_artifact"


def test_apply_is_deterministic_and_idempotent(dataset: Path, tmp_path: Path):
    _repair_all(dataset)
    apply_repairs(dataset, evaluation_root=EVAL_ROOT, now="2026-08-23 00:00:00")
    first = {n: _sha(dataset / n) for n in ("cases_inputs.jsonl", "cases_labels.jsonl")}
    n_rev = len(json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))["revisions"])
    again = apply_repairs(dataset, evaluation_root=EVAL_ROOT, now="2026-08-23 00:00:00")
    assert again["written"] is False and again["lines_repaired"] == []
    assert {n: _sha(dataset / n) for n in ("cases_inputs.jsonl", "cases_labels.jsonl")} == first
    assert len(json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))["revisions"]) == n_rev

    other = _copy_dataset(tmp_path / "twin")           # an independent run matches byte for byte
    _repair_all(other)
    apply_repairs(other, evaluation_root=EVAL_ROOT, now="2026-08-23 00:00:00")
    assert {n: _sha(other / n) for n in ("cases_inputs.jsonl", "cases_labels.jsonl")} == first


def test_case_ids_scores_and_labels_survive_the_repair(dataset: Path):
    """Erik's ground truth is never touched: ids, order, scores, rubric, status."""
    rows = _rows(dataset / "cases_labels.jsonl")
    for i, r in enumerate(rows):                       # stand in for the imported instructor grades
        r["score"] = float(i % 5)
        r["label_status"] = "OWNER_LABELED"
        r["label_source"] = "original_instructor_grade"
    _write_labels(dataset, rows)

    _repair_all(dataset)
    apply_repairs(dataset, evaluation_root=EVAL_ROOT)

    after = _rows(dataset / "cases_labels.jsonl")
    assert [r["case_id"] for r in after] == [r["case_id"] for r in rows], "case ids and their order are stable"
    for b, a in zip(rows, after):
        assert (a["score"], a["max_score"], a["rubric_met"]) == (b["score"], b["max_score"], b["rubric_met"])
        assert (a["label_status"], a["label_source"]) == (b["label_status"], b["label_source"])
        assert (a["writer"], a["question_id"], a["sub_item_id"], a["split"]) == \
               (b["writer"], b["question_id"], b["sub_item_id"], b["split"])
    assert sum(1 for r in after if r["score"] is not None) == 67, "all 67 ground-truth labels remain valid"
    assert sum(1 for r in after if r["transcription_complete"] is True) == 67


def test_apply_leaves_the_frozen_ocr_benchmark_untouched(dataset: Path):
    before = frozen_bench_hashes()
    assert before, "the frozen OCR benchmark is present"
    _repair_all(dataset)
    out = apply_repairs(dataset, evaluation_root=EVAL_ROOT)
    assert out["frozen_bench_sha256"] == before
    assert frozen_bench_hashes() == before, "evaluation/hebrew_bench_v2 is unchanged by a repair"
    assert not list(dataset.rglob("hebrew_bench_v2")), "repairs never write into the OCR benchmark tree"


def test_apply_records_the_revision_and_the_new_checksums(dataset: Path):
    man_before = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    old_inputs = man_before["inputs_sha256"]
    store = _repair_all(dataset)
    out = apply_repairs(dataset, evaluation_root=EVAL_ROOT, now="2026-08-23 12:00:00")
    man = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    rev = man["revisions"][-1]
    assert len(man["revisions"]) == len(man_before["revisions"]) + 1
    assert rev["kind"] == REPAIR_SOURCE and rev["at"] == "2026-08-23 12:00:00"
    assert rev["previous_inputs_sha256"] == old_inputs != rev["inputs_sha256"]
    assert rev["inputs_changed"] is True, "the model input changed and the manifest says so"
    assert sorted(rev["lines_repaired"]) == [e["line_id"] for e in expected_repairs(dataset)]
    assert rev["repair_store_sha256"] == store.sha256()
    assert rev["frozen_bench_sha256"] == frozen_bench_hashes()
    assert man["inputs_sha256"] == _sha(dataset / "cases_inputs.jsonl") == out["inputs_sha256"]
    assert man["labels_sha256"] == _sha(dataset / "cases_labels.jsonl") == out["labels_sha256"]
    checks = (dataset / "CHECKSUMS.sha256").read_text(encoding="utf-8")
    assert man["inputs_sha256"] in checks and man["labels_sha256"] in checks
    assert man["extra"]["evidence_inventory"]["transcription_incomplete_cases"] == []


def test_status_reports_ready_only_after_the_repair(dataset: Path):
    """The transcription dimension gates READY independently of ground truth."""
    from autograder.benchmark import status as status_mod
    rows = _rows(dataset / "cases_labels.jsonl")
    for r in rows:
        r["score"] = 1.0
    _write_labels(dataset, rows)

    st = status_mod.role_dataset_status("grade_primary", datasets_root=dataset.parent)
    assert st["status"] == "PARTIALLY_READY" and st["labeled"] == 67
    assert st["scorable_for_accuracy"] == 58 and st["transcription_incomplete"] == 9

    _repair_all(dataset)
    apply_repairs(dataset, evaluation_root=EVAL_ROOT)
    st = status_mod.role_dataset_status("grade_primary", datasets_root=dataset.parent)
    assert st["status"] == "READY"
    assert st["scorable_for_accuracy"] == 67 and st["transcription_incomplete"] == 0


# ------------------------------------------- the LIVE, already-repaired dataset --
# The checked-in dataset is post-repair. These read it and never write to it.

def test_live_dataset_is_fully_repaired():
    labels = _label_map(REAL_DATASET)
    assert len(labels) == 67
    assert all(r["transcription_complete"] is True for r in labels.values())
    assert [c for c, r in labels.items() if r["lines_without_audited_transcription"]] == []
    assert repaired_cases() == sorted({c for c, r in labels.items() if r.get("evidence_repairs")})
    assert len(repaired_cases()) == 9 and len(repaired_line_ids()) == 9


def test_live_repair_store_holds_exactly_the_owners_nine_decisions():
    store = repair_store()
    assert len(store) == 9
    assert sorted(store) == repaired_line_ids(), "one record per repaired line, no strays, no duplicates"
    by_disposition = {d: sorted(k for k, r in store.items() if r["disposition"] == d) for d in DISPOSITIONS}
    assert by_disposition["transcribed"] == ["e004_q2_r3__l2"], "exactly one line was transcribed by hand"
    assert len(by_disposition["no_text_segmentation_artifact"]) == 8
    for line_id, r in store.items():
        assert r["human_verified"] is True and r["verified_by"], line_id
        assert r["source"] == REPAIR_SOURCE and r["line_id"] == line_id
        assert line_id.startswith(r["case_id"] + "__"), "the record belongs to its own case"
        crop = REAL_DATASET / r["crop_path"]
        assert crop.exists() and _sha(crop) == r["crop_sha256"], line_id
        assert (r["transcription"].strip() != "") is (r["disposition"] == "transcribed"), line_id


def test_live_repair_store_passes_every_integrity_rule():
    rep = verify_repairs(REAL_DATASET, evaluation_root=EVAL_ROOT)
    assert rep["problems"] == []
    assert rep["ok"] is True and rep["complete"] is True
    assert rep["expected"] == 9 and rep["repaired"] == 9 and rep["remaining"] == []
    assert rep["unexpected_records"] == [] and sorted(rep["applied"]) == repaired_line_ids()
    assert rep["by_disposition"] == {"transcribed": 1, "no_text_segmentation_artifact": 8}
    assert rep["frozen_bench_sha256"] == frozen_bench_hashes()


def test_live_repaired_lines_carry_manual_repair_provenance():
    store = repair_store()
    for row in _rows(REAL_DATASET / "cases_labels.jsonl"):
        for sid in (row.get("evidence_repairs") or []):
            line = next(e for e in row["evidence_lines"] if e["sample_id"] == sid)
            rec = store[sid]
            assert line["transcription_status"] == REPAIR_SOURCE + ":" + rec["disposition"]
            assert line["repair"]["crop_sha256"] == rec["crop_sha256"]
            assert line["repair"]["verified_by"] == rec["verified_by"]
            assert line["original_image"], sid
            assert line["original_transcription_status"].startswith("no_audited_transcription:"), \
                "the pre-repair record is preserved, not overwritten"
            assert REPAIR_SOURCE in row["transcription_source"]


def test_live_transcriptions_gained_text_only_where_a_human_typed_it():
    """The eight artifact rulings added nothing; only e004_q2_r3 gained a line."""
    store = repair_store()
    inputs = {r["case_id"]: r for r in _rows(REAL_DATASET / "cases_inputs.jsonl")}
    for row in _rows(REAL_DATASET / "cases_labels.jsonl"):
        text = inputs[row["case_id"]]["transcription"]
        lines = text.split("\n") if text else []
        reps = [store[s] for s in (row.get("evidence_repairs") or [])]
        typed = [r for r in reps if r["disposition"] == "transcribed"]
        artifacts = [r for r in reps if r["disposition"] == "no_text_segmentation_artifact"]
        assert len(lines) == row["line_count"] - len(artifacts), row["case_id"]
        for rec in typed:
            assert rec["transcription"] in lines, "the human line is present, verbatim"
        for rec in artifacts:
            assert rec["transcription"] == "", "an artifact ruling contributes no text at all"


def test_live_manifest_records_the_evidence_version_transition():
    man = json.loads((REAL_DATASET / "manifest.json").read_text(encoding="utf-8"))
    revs = man["revisions"]
    manual = [r for r in revs if r["kind"] == REPAIR_SOURCE]
    assert len(manual) == 1, "the repair was applied exactly once"
    rev = manual[0]
    assert rev["inputs_changed"] is True
    assert rev["previous_inputs_sha256"] != rev["inputs_sha256"]
    # inputs moved again later (the 2026-09-01 owner-confirmed transposition);
    # the recorded chain from this revision must reach today's file
    from prerepair import chain_end
    assert chain_end(revs, rev["inputs_sha256"], "inputs") == man["inputs_sha256"]
    assert rev["previous_labels_sha256"] != rev["labels_sha256"]
    # the labels file has moved on since (the effective-evidence layer), so this
    # revision's output is the NEXT revision's input, not today's file
    following = revs[revs.index(rev) + 1:]
    assert rev["labels_sha256"] == (following[0]["previous_labels_sha256"] if following else man["labels_sha256"])
    assert sorted(rev["lines_repaired"]) == repaired_line_ids()
    assert sorted(rev["cases_changed"]) == repaired_cases()
    assert rev["repair_store_sha256"] == _sha(REAL_DATASET / "manual_evidence_repairs.jsonl")
    assert rev["frozen_bench_sha256"] == frozen_bench_hashes()
    assert man["extra"]["evidence_inventory"]["transcription_incomplete_cases"] == []
    assert man["inputs_sha256"] == _sha(REAL_DATASET / "cases_inputs.jsonl")
    assert man["labels_sha256"] == _sha(REAL_DATASET / "cases_labels.jsonl")
    checks = (REAL_DATASET / "CHECKSUMS.sha256").read_text(encoding="utf-8")
    assert man["inputs_sha256"] in checks and man["labels_sha256"] in checks
    # the revision chain is unbroken: build -> evidence inventory -> manual repair
    assert revs[0]["kind"] == "evidence_inventory_repair"
    assert revs[0]["labels_sha256"] == rev["previous_labels_sha256"]
    assert revs[0]["inputs_sha256"] == rev["previous_inputs_sha256"]


def test_live_dataset_is_scorable_for_every_case():
    from autograder.benchmark import status as status_mod
    st = status_mod.role_dataset_status("grade_primary")
    assert st["cases"] == 67
    assert st["transcription_incomplete"] == 0 and st["transcription_incomplete_cases"] == []
    assert st["labeled_not_scorable"] == 0
    assert st["scorable_for_accuracy"] == st["labeled"], "nothing is held back by the transcription dimension"


def test_applying_again_is_a_no_op_that_never_duplicates_a_revision(live_dataset_copy: Path):
    """The dataset is already repaired: a re-run must not double-apply."""
    d = live_dataset_copy
    before = {n: _sha(d / n) for n in DATASET_FILES}
    n_rev = len(json.loads((d / "manifest.json").read_text(encoding="utf-8"))["revisions"])

    dry = apply_repairs(d, evaluation_root=EVAL_ROOT, dry_run=True)
    assert dry["written"] is False and dry["inputs_changed"] is False
    assert dry["lines_repaired"] == [] and dry["cases_changed"] == []
    assert {n: _sha(d / n) for n in DATASET_FILES} == before

    for _ in range(2):
        out = apply_repairs(d, evaluation_root=EVAL_ROOT)
        assert out["written"] is False and out["lines_repaired"] == []
        assert out["inputs_sha256"] == out["previous_inputs_sha256"]
    assert {n: _sha(d / n) for n in DATASET_FILES} == before
    assert len(json.loads((d / "manifest.json").read_text(encoding="utf-8"))["revisions"]) == n_rev


def test_the_live_dataset_is_the_pre_repair_dataset_plus_the_owners_decisions(tmp_path: Path):
    """The strongest statement available: rebuild the current dataset from the
    reconstructed earlier state plus the owner's real repair records, and require
    the result to match. Nothing about her transcription is re-derived — her own
    records supply it.

    One field cannot match from a scratch directory: a repaired line's `image`
    is the repair crop resolved against the evidence root, and a rebuild in
    tmp_path sits outside that root, so `apply_repairs` keeps the historic path
    instead. The test pins that down rather than waving it through — it is the
    ONLY difference allowed, and only on the nine repaired lines."""
    live_man = json.loads((REAL_DATASET / "manifest.json").read_text(encoding="utf-8"))
    when = next(r["at"] for r in live_man["revisions"] if r["kind"] == REPAIR_SOURCE)
    d = build_pre_repair_dataset(tmp_path / "grade_primary")
    copy_repair_store(d)
    out = apply_repairs(d, evaluation_root=EVAL_ROOT, now=when)
    assert out["written"] is True and len(out["lines_repaired"]) == 9

    # the rebuild lands on the pre-transposition state (the 2026-09-01 swap is
    # a LATER owner-confirmed revision); compare against today's dataset with
    # the self-inverse swap un-applied
    from prerepair import body as _body, pre_transposition_live
    pt_inputs, pt_labels = pre_transposition_live(REAL_DATASET)
    assert (d / "cases_inputs.jsonl").read_bytes() == _body(pt_inputs), \
        "the model input rebuilds byte for byte from the owner's own records"

    store, live = repair_store(), {r["case_id"]: r for r in pt_labels}
    substituted = []
    for row in _rows(d / "cases_labels.jsonl"):
        for e in row["evidence_lines"]:
            sid = e.get("sample_id")
            if sid in (row.get("evidence_repairs") or []):
                live_line = next(x for x in live[row["case_id"]]["evidence_lines"] if x["sample_id"] == sid)
                assert live_line["image"].endswith(store[sid]["crop_path"]), sid
                assert (EVAL_ROOT / live_line["image"]).read_bytes() == \
                       (REAL_DATASET / store[sid]["crop_path"]).read_bytes(), sid
                assert e["image"] == e["original_image"], "outside the evidence root the historic path is kept"
                substituted.append(sid)
                if e["image"] in row["evidence_images"]:      # transcribed lines are effective evidence
                    row["evidence_images"][row["evidence_images"].index(e["image"])] = live_line["image"]
                e["image"] = live_line["image"]
        assert row == _without_verdict(live[row["case_id"]]), row["case_id"]
    assert sorted(substituted) == repaired_line_ids(), "nothing else needed adjusting"

    rebuilt = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    from prerepair import chain_end
    assert chain_end(live_man["revisions"], rebuilt["inputs_sha256"], "inputs") \
        == live_man["inputs_sha256"], "the recorded chain reaches today's inputs"
    assert rebuilt["revisions"][0] == live_man["revisions"][0], "the historical revision is untouched"
    # The live dataset reached this state in TWO recorded steps (the repair, then
    # the effective-evidence layer); a one-pass rebuild records them as one. The
    # bookkeeping differs by construction — the FACTS about the repair must not.
    a = rebuilt["revisions"][-1]
    b = next(r for r in live_man["revisions"] if r["kind"] == REPAIR_SOURCE)
    assert a["kind"] == b["kind"] == REPAIR_SOURCE
    for key in ("previous_inputs_sha256", "inputs_sha256", "inputs_changed", "previous_labels_sha256",
                "lines_repaired", "cases_changed", "repair_store", "repair_store_sha256", "frozen_bench_sha256"):
        assert a[key] == b[key], key
    # Compare only the EVIDENCE chain: the live dataset may carry later,
    # unrelated revisions (verdict_target) that a repair rebuild never writes.
    from autograder.benchmark.evidence_repairs import EFFECTIVE_EVIDENCE_SOURCE

    live_kinds = [r["kind"] for r in live_man["revisions"]]
    evidence_chain = live_kinds[:live_kinds.index(EFFECTIVE_EVIDENCE_SOURCE) + 1]
    assert len(rebuilt["revisions"]) == len(evidence_chain) - 1


def test_the_repair_left_the_frozen_ocr_benchmark_alone():
    """hebrew_bench_v2 is not part of this workflow's write surface, at all."""
    assert frozen_bench_hashes() == {
        "items.json": "25463e91b95db8eecfc306644dd0839f65e42910d4161f6991383b6f4a524a8b",
        "references.json": "4a93e826e6e94777d445e64ae2c3f5ed10def46aa021ff763a45a9807fb913b5",
        "reference_audit.json": "60349119f4d07dcbb5f2bd50b671d10b7db2ceb0385f949fee455e3a9640f1da",
        "reference_audit_manifest.json": "65ac16385979480945938f45f0d3e5e3255afa7bcb41c4ea7afe2ff2ebcc405f",
    }
    assert not list((REAL_DATASET / "manual_evidence_repairs").rglob("*hebrew_bench*"))


# ------------------------------------- EFFECTIVE vs HISTORICAL evidence --
# `evidence_lines` is the immutable historical record — every recorded line,
# the crop it came from, and what was decided about it. `evidence_images` is
# the EFFECTIVE evidence: what a grader should be shown now. A transcribed
# line contributes its repaired crop; a line ruled a segmentation artifact
# contributes nothing, because a bogus sliver is not a separate answer image.

def test_effective_evidence_is_derived_from_the_repair_resolution():
    """Not stored-and-hoped: re-derive it from the row and require a match."""
    from autograder.benchmark.evidence_repairs import resolution_summary
    for row in _rows(REAL_DATASET / "cases_labels.jsonl"):
        derived = resolution_summary(row)
        for key, value in derived.items():
            assert row[key] == value, f"{row['case_id']}.{key}"
        assert row["evidence_images"] == [e["image"] for e in row["evidence_lines"]
                                          if (e.get("repair") or {}).get("disposition")
                                          != "no_text_segmentation_artifact"]


def test_a_transcribed_repair_shows_the_repaired_crop_and_keeps_the_old_one():
    store = repair_store()
    typed = [lid for lid, r in store.items() if r["disposition"] == "transcribed"]
    assert typed, "at least one line was transcribed by hand"
    labels = _label_map(REAL_DATASET)
    for line_id in typed:
        row = labels[store[line_id]["case_id"]]
        line = next(e for e in row["evidence_lines"] if e["sample_id"] == line_id)
        # effective: the repaired crop IS the grader-visible image
        assert line["image"].endswith(store[line_id]["crop_path"])
        assert line["image"] in row["evidence_images"]
        assert (EVAL_ROOT / line["image"]).read_bytes() == (REAL_DATASET / store[line_id]["crop_path"]).read_bytes()
        # historical: the crop it replaced is still named, and still on disk
        assert line["original_image"] != line["image"]
        assert (EVAL_ROOT / line["original_image"]).exists()
        assert line["original_image"] not in row["evidence_images"], "the known-bad crop is not active evidence"


def test_an_artifact_ruling_is_excluded_from_effective_evidence_but_kept_historically():
    store = repair_store()
    artifacts = [lid for lid, r in store.items() if r["disposition"] == "no_text_segmentation_artifact"]
    assert len(artifacts) == 8
    labels = _label_map(REAL_DATASET)
    for line_id in artifacts:
        row = labels[store[line_id]["case_id"]]
        line = next(e for e in row["evidence_lines"] if e["sample_id"] == line_id)
        # excluded from what a grader sees — neither the sliver nor its repair crop
        assert line["image"] not in row["evidence_images"], line_id
        assert line["original_image"] not in row["evidence_images"], line_id
        assert not any(line_id in img for img in row["evidence_images"]), line_id
        # ...but the line itself is still a recorded line, with its full history
        assert line["sample_id"] in {e["sample_id"] for e in row["evidence_lines"]}
        assert line["transcription_status"] == REPAIR_SOURCE + ":no_text_segmentation_artifact"
        assert (EVAL_ROOT / line["original_image"]).exists()


def test_e004_q2_r3_has_exactly_two_useful_effective_images():
    row = _label_map(REAL_DATASET)["e004_q2_r3"]
    store = repair_store()
    assert row["line_count"] == 2 and len(row["evidence_lines"]) == 2
    assert len(row["evidence_images"]) == 2, "one audited line + one hand-transcribed line"
    assert row["evidence_images"][0] == row["evidence_lines"][0]["image"]
    assert row["evidence_images"][0].startswith("hebrew_bench_v2/crops/")
    assert row["evidence_images"][1].endswith(store["e004_q2_r3__l2"]["crop_path"])
    assert all((EVAL_ROOT / img).exists() for img in row["evidence_images"])
    assert (row["lines_transcribed"], row["lines_no_text_artifact"], row["lines_resolved"]) == (2, 0, 2)
    assert row["transcription_complete"] is True


def test_an_artifact_case_does_not_expose_its_bogus_sliver():
    store = repair_store()
    cid = next(r["case_id"] for r in store.values() if r["disposition"] == "no_text_segmentation_artifact")
    row = _label_map(REAL_DATASET)[cid]
    line_id = row["evidence_repairs"][0]
    assert row["line_count"] == len(row["evidence_lines"])
    assert len(row["evidence_images"]) == row["line_count"] - 1, "the sliver is not an extra answer image"
    assert (row["lines_transcribed"], row["lines_no_text_artifact"]) == (row["line_count"] - 1, 1)
    assert row["lines_resolved"] == row["line_count"] and row["transcription_complete"] is True
    bogus = next(e for e in row["evidence_lines"] if e["sample_id"] == line_id)
    assert bogus["original_image"] not in row["evidence_images"]
    assert bogus["image"] not in row["evidence_images"]


def test_the_historical_crop_and_hash_stay_recoverable_for_all_nine():
    store = repair_store()
    labels = _label_map(REAL_DATASET)
    for line_id, rec in store.items():
        row = labels[rec["case_id"]]
        line = next(e for e in row["evidence_lines"] if e["sample_id"] == line_id)
        # the original crop: path, bytes and sha256, all still reachable
        assert (EVAL_ROOT / line["original_image"]).exists(), line_id
        original = rec["original_crop"]
        assert original["image"] == line["original_image"]
        assert _sha(EVAL_ROOT / original["image"]) == original["sha256"], line_id
        assert original["status"] == "bad_segmentation"
        assert line["original_transcription_status"].startswith("no_audited_transcription:")
        # ...and the repair that superseded it
        assert line["repair"]["crop_sha256"] == rec["crop_sha256"]
        assert _sha(REAL_DATASET / rec["crop_path"]) == rec["crop_sha256"], line_id
    # the manifest revision names every one of them too
    man = json.loads((REAL_DATASET / "manifest.json").read_text(encoding="utf-8"))
    rev = next(r for r in man["revisions"] if r["kind"] == REPAIR_SOURCE)
    assert sorted(rev["lines_repaired"]) == sorted(store) == repaired_line_ids()


# ------------------------------------------------ the line dimensions --

def test_lines_transcribed_counts_only_text_bearing_lines():
    store = repair_store()
    for row in _rows(REAL_DATASET / "cases_labels.jsonl"):
        artifacts = [s for s in (row.get("evidence_repairs") or [])
                     if store[s]["disposition"] == "no_text_segmentation_artifact"]
        assert row["lines_transcribed"] == row["line_count"] - len(artifacts), row["case_id"]
        assert row["lines_transcribed"] == len(row["evidence_images"]), \
            "one effective image per text-bearing line"


def test_lines_no_text_artifact_counts_artifact_resolutions():
    store = repair_store()
    for row in _rows(REAL_DATASET / "cases_labels.jsonl"):
        artifacts = [s for s in (row.get("evidence_repairs") or [])
                     if store[s]["disposition"] == "no_text_segmentation_artifact"]
        assert row["lines_no_text_artifact"] == len(artifacts), row["case_id"]
    total = sum(r["lines_no_text_artifact"] for r in _rows(REAL_DATASET / "cases_labels.jsonl"))
    assert total == sum(1 for r in store.values() if r["disposition"] == "no_text_segmentation_artifact") == 8


def test_lines_resolved_equals_line_count_for_every_complete_case():
    rows = _rows(REAL_DATASET / "cases_labels.jsonl")
    for row in rows:
        assert row["lines_resolved"] == row["lines_transcribed"] + row["lines_no_text_artifact"], row["case_id"]
        assert (row["lines_resolved"] == row["line_count"]) is row["transcription_complete"], row["case_id"]
        assert row["lines_resolved"] == row["line_count"], row["case_id"]
    assert sum(r["line_count"] for r in rows) == sum(r["lines_resolved"] for r in rows)
    assert sum(r["lines_transcribed"] for r in rows) + sum(r["lines_no_text_artifact"] for r in rows) \
        == sum(r["line_count"] for r in rows)


def test_the_two_worked_examples_have_the_dimensions_the_owner_specified():
    labels = _label_map(REAL_DATASET)
    store = repair_store()
    e = labels["e004_q2_r3"]                                     # the hand-transcribed case
    assert (e["line_count"], e["lines_transcribed"], e["lines_no_text_artifact"], e["lines_resolved"]) == (2, 2, 0, 2)
    assert e["transcription_complete"] is True
    art_cid = next(r["case_id"] for lid, r in sorted(store.items())
                   if r["disposition"] == "no_text_segmentation_artifact"
                   and labels[r["case_id"]]["line_count"] == 2)
    a = labels[art_cid]                                          # a representative artifact case
    assert (a["line_count"], a["lines_transcribed"], a["lines_no_text_artifact"], a["lines_resolved"]) == (2, 1, 1, 2)
    assert a["transcription_complete"] is True


def test_the_effective_evidence_revision_is_recorded_and_the_chain_is_unbroken():
    from autograder.benchmark.evidence_repairs import EFFECTIVE_EVIDENCE_SOURCE
    man = json.loads((REAL_DATASET / "manifest.json").read_text(encoding="utf-8"))
    revs = man["revisions"]
    kinds = [r["kind"] for r in revs]
    assert kinds[:3] == ["evidence_inventory_repair", REPAIR_SOURCE, EFFECTIVE_EVIDENCE_SOURCE]
    # later, unrelated revisions may follow (verdict_target); the evidence chain
    # is the prefix and must stay unbroken regardless of what came after
    eff = revs[2]
    assert eff["inputs_changed"] is False, "the grading model's text input did not change"
    assert eff["previous_inputs_sha256"] == eff["inputs_sha256"]
    from prerepair import chain_end
    assert chain_end(revs, eff["inputs_sha256"], "inputs") == man["inputs_sha256"]
    assert eff["previous_labels_sha256"] != eff["labels_sha256"]
    if len(revs) == 3:
        assert eff["labels_sha256"] == man["labels_sha256"]
    assert eff["lines_repaired"] == [], "no human decision was re-made"
    assert len(eff["rows_changed"]) == 67, "every row now states the same dimensions"
    assert eff["frozen_bench_sha256"] == frozen_bench_hashes()
    for a, b in zip(revs, revs[1:]):                       # each revision starts where the last ended
        assert a["labels_sha256"] == b["previous_labels_sha256"]
        assert a["inputs_sha256"] == b["previous_inputs_sha256"]
    assert man["labels_sha256"] == _sha(REAL_DATASET / "cases_labels.jsonl")
    assert man["inputs_sha256"] == _sha(REAL_DATASET / "cases_inputs.jsonl")


def test_the_effective_evidence_layer_is_idempotent(live_dataset_copy: Path):
    """Re-running must not duplicate transcription, duplicate a revision, or
    move a hash a second time."""
    d = live_dataset_copy
    before = {n: _sha(d / n) for n in DATASET_FILES}
    n_rev = len(json.loads((d / "manifest.json").read_text(encoding="utf-8"))["revisions"])
    for _ in range(3):
        out = apply_repairs(d, evaluation_root=EVAL_ROOT)
        assert out["written"] is False and out["lines_repaired"] == []
        assert out["labels_sha256"] == out["previous_labels_sha256"]
        assert out["inputs_sha256"] == out["previous_inputs_sha256"]
    assert {n: _sha(d / n) for n in DATASET_FILES} == before
    assert len(json.loads((d / "manifest.json").read_text(encoding="utf-8"))["revisions"]) == n_rev
    inputs = {r["case_id"]: r["transcription"] for r in _rows(d / "cases_inputs.jsonl")}
    live = {r["case_id"]: r["transcription"] for r in _rows(REAL_DATASET / "cases_inputs.jsonl")}
    assert inputs == live, "no transcription was duplicated or re-appended"


def test_applying_the_effective_layer_to_the_pre_repair_dataset_reaches_the_live_state(tmp_path: Path):
    """One apply, from the reconstructed earlier state plus the owner's real
    decisions, lands on today's dataset — repair layer and effective-evidence
    layer together, with no second pass needed."""
    d = build_pre_repair_dataset(tmp_path / "grade_primary")
    copy_repair_store(d)
    out = apply_repairs(d, evaluation_root=EVAL_ROOT)
    assert out["written"] is True and len(out["lines_repaired"]) == 9
    # the forward walk lands on the state BEFORE the 2026-09-01 owner-confirmed
    # row transposition; compare against today's dataset with that (self-
    # inverse) swap un-applied — provably the genuine historical state
    from prerepair import body as _body, pre_transposition_live, transposition_revisions
    pt_inputs, pt_labels = pre_transposition_live(REAL_DATASET)
    assert (d / "cases_inputs.jsonl").read_bytes() == _body(pt_inputs)
    for rev in transposition_revisions(REAL_DATASET):
        assert out["inputs_sha256"] == rev["previous_inputs_sha256"]
    live = {r["case_id"]: r for r in pt_labels}
    for row in _rows(d / "cases_labels.jsonl"):
        want = live[row["case_id"]]
        assert row["lines_transcribed"] == want["lines_transcribed"]
        assert row["lines_no_text_artifact"] == want["lines_no_text_artifact"]
        assert row["lines_resolved"] == want["lines_resolved"]
        assert row["transcription_complete"] == want["transcription_complete"]
        assert len(row["evidence_images"]) == len(want["evidence_images"])


# --------------------------------- guards found by the post-change audit --

def test_the_builder_refuses_to_re_derive_evidence_over_an_applied_repair(live_dataset_copy: Path):
    """`repair_grading_evidence` re-derives the evidence block from the FROZEN OCR
    benchmark, which knows nothing about the human repair layer. On a repaired
    dataset that would put the mis-segmented crops back as ACTIVE evidence and
    strip the repair provenance — it must refuse, loudly, not do it."""
    from autograder.benchmark.datasets import DatasetBuildError, repair_grading_evidence
    d = live_dataset_copy
    before = {n: _sha(d / n) for n in DATASET_FILES}
    with pytest.raises(DatasetBuildError, match="manual evidence repairs"):
        repair_grading_evidence(d, dry_run=True)
    with pytest.raises(DatasetBuildError, match="apply-evidence-repairs"):
        repair_grading_evidence(d)
    assert {n: _sha(d / n) for n in DATASET_FILES} == before
    # ...and it still works on a dataset that carries no repairs
    pre = build_pre_repair_dataset(d.parent / "pre")
    out = repair_grading_evidence(pre, dry_run=True)
    assert out["written"] is False


def test_an_unknown_disposition_is_refused_rather_than_treated_as_evidence():
    """Failing open would silently serve an unclassified crop as answer evidence."""
    from autograder.benchmark.evidence_repairs import line_resolution, resolution_summary
    row = dict(_label_map(REAL_DATASET)[repaired_cases()[0]])
    sid = row["evidence_repairs"][0]
    assert line_resolution(next(e for e in row["evidence_lines"] if e["sample_id"] == sid),
                           unresolved=set()) == "artifact"
    row["evidence_lines"] = [{**e, "repair": {**e["repair"], "disposition": "something_new"}}
                             if e["sample_id"] == sid else e for e in row["evidence_lines"]]
    with pytest.raises(RepairError, match="not one of"):
        resolution_summary(row)


def test_the_effective_evidence_is_ordered_by_line_index_not_by_storage_order():
    from autograder.benchmark.evidence_repairs import resolution_summary
    row = dict(_label_map(REAL_DATASET)["e004_q2_r5"])          # a three-line cell
    want = resolution_summary(row)["evidence_images"]
    shuffled = {**row, "evidence_lines": list(reversed(row["evidence_lines"]))}
    assert resolution_summary(shuffled)["evidence_images"] == want, "line order is enforced, not inherited"
    assert want == [e["image"] for e in sorted(row["evidence_lines"], key=lambda l: l["index"])
                    if (e.get("repair") or {}).get("disposition") != "no_text_segmentation_artifact"]


def test_editing_the_store_after_the_repair_was_applied_is_detected(live_dataset_copy: Path):
    """The decisions are embedded in the dataset once applied; a store edited
    afterwards no longer agrees with what the dataset says was decided."""
    d = live_dataset_copy
    assert verify_repairs(d, evaluation_root=EVAL_ROOT)["problems"] == []
    store = RepairStore(d)
    recs = store.records()
    typed = next(k for k, r in recs.items() if r["disposition"] == "transcribed")
    artifact = next(k for k, r in recs.items() if r["disposition"] == "no_text_segmentation_artifact")

    store._rewrite({**recs, typed: {**recs[typed], "transcription": "FABRICATED"}})
    problems = verify_repairs(d, evaluation_root=EVAL_ROOT)["problems"]
    assert any("not the text applied" in p["problem"] for p in problems)

    store._rewrite({**recs, artifact: {**recs[artifact], "disposition": "transcribed",
                                       "transcription": "INVENTED"}})
    problems = verify_repairs(d, evaluation_root=EVAL_ROOT)["problems"]
    assert any("disposition no longer matches" in p["problem"] for p in problems)

    store._rewrite({**recs, typed: {**recs[typed], "verified_by": "someone else"}})
    assert any("verified_by no longer matches" in p["problem"]
               for p in verify_repairs(d, evaluation_root=EVAL_ROOT)["problems"])

    store._rewrite({k: v for k, v in recs.items() if k != typed})
    assert any("missing from the repair store" in p["problem"]
               for p in verify_repairs(d, evaluation_root=EVAL_ROOT)["problems"])

    store._rewrite(recs)                                        # restored: clean again
    assert verify_repairs(d, evaluation_root=EVAL_ROOT)["problems"] == []


def test_the_result_is_the_same_however_the_decisions_arrive(tmp_path: Path):
    """The dataset's identity hash anchors the revision chain, CHECKSUMS and the
    labeling bundle, so it must be a function of the DECISIONS — not of how many
    passes folded them in. Applying eight then one must equal applying nine."""
    one_shot = build_pre_repair_dataset(tmp_path / "one_shot")
    copy_repair_store(one_shot)
    apply_repairs(one_shot, evaluation_root=EVAL_ROOT, now="2026-08-23 00:00:00")

    staged = build_pre_repair_dataset(tmp_path / "staged")
    copy_repair_store(staged)
    store = RepairStore(staged)
    everything = store.records()
    held_back = "e004_q2_r3__l2"                      # the one transcribed decision, applied last
    store._rewrite({k: v for k, v in everything.items() if k != held_back})
    partial = apply_repairs(staged, evaluation_root=EVAL_ROOT, allow_partial=True, now="2026-08-23 00:00:00")
    assert partial["written"] is True and len(partial["lines_repaired"]) == 8
    store._rewrite(everything)
    rest = apply_repairs(staged, evaluation_root=EVAL_ROOT, now="2026-08-23 00:00:00")
    assert rest["written"] is True and rest["lines_repaired"] == [held_back]

    assert (staged / "cases_labels.jsonl").read_bytes() == (one_shot / "cases_labels.jsonl").read_bytes()
    assert (staged / "cases_inputs.jsonl").read_bytes() == (one_shot / "cases_inputs.jsonl").read_bytes()
    assert (staged / "CHECKSUMS.sha256").read_bytes() == (one_shot / "CHECKSUMS.sha256").read_bytes()
    # ...including the field ORDER inside each row, which is what made this
    # order-dependent before: the repair layer's fields sit in a fixed position
    for a, b in zip(_rows(staged / "cases_labels.jsonl"), _rows(one_shot / "cases_labels.jsonl")):
        assert list(a) == list(b) == list(
            _without_verdict(_label_map(REAL_DATASET)[a["case_id"]]))


def test_changing_a_decision_after_it_was_applied_is_refused_not_ignored(live_dataset_copy: Path):
    """Idempotency must not become "silently stuck": a decision the owner edits
    after it was folded in has to be reported, never quietly ignored."""
    d = live_dataset_copy
    store = RepairStore(d)
    recs = store.records()
    flipped = next(k for k, r in recs.items() if r["disposition"] == "transcribed")
    store._rewrite({**recs, flipped: {**recs[flipped], "disposition": "no_text_segmentation_artifact",
                                      "transcription": ""}})
    before = {n: _sha(d / n) for n in DATASET_FILES}
    with pytest.raises(RepairError, match="not admissible"):
        apply_repairs(d, evaluation_root=EVAL_ROOT)
    with pytest.raises(RepairError, match="not admissible"):
        apply_repairs(d, evaluation_root=EVAL_ROOT, dry_run=True)
    assert {n: _sha(d / n) for n in DATASET_FILES} == before, "a divergent store never rewrites the dataset"
    problems = verify_repairs(d, evaluation_root=EVAL_ROOT)["problems"]
    assert any(p["line_id"] == flipped and "disposition no longer matches" in p["problem"] for p in problems)


def test_a_role_without_an_evidence_block_is_left_alone(tmp_path: Path):
    """apply-evidence-repairs is a grade_primary concern; an MC/variant dataset
    has no evidence lines to resolve and must not be rewritten by it."""
    other = REAL_DATASET.parent / "mc_resolve_cloud"
    if not (other / "manifest.json").exists():
        pytest.skip("mc_resolve_cloud is not built here")
    d = tmp_path / "mc_resolve_cloud"
    d.mkdir()
    for name in DATASET_FILES:
        shutil.copy2(other / name, d / name)
    before = {n: _sha(d / n) for n in DATASET_FILES}
    out = apply_repairs(d, evaluation_root=EVAL_ROOT, allow_partial=True)
    assert out["written"] is False and out["lines_repaired"] == []
    assert {n: _sha(d / n) for n in DATASET_FILES} == before


def test_text_attached_to_a_no_text_ruling_is_detected(live_dataset_copy: Path):
    """The writer refuses artifact-with-text; the verifier must too, or eight of
    the nine decisions could be annotated with words the owner never wrote."""
    d = live_dataset_copy
    store = RepairStore(d)
    recs = store.records()
    artifact = next(k for k, r in recs.items() if r["disposition"] == "no_text_segmentation_artifact")
    with pytest.raises(RepairError, match="carries no transcription"):     # the writer's rule
        store.save(case_id=recs[artifact]["case_id"], line_id=artifact, transcription="INVENTED",
                   disposition="no_text_segmentation_artifact", verified_by="t",
                   crop_png=(d / recs[artifact]["crop_path"]).read_bytes(),
                   original_crop=recs[artifact]["original_crop"])
    store._rewrite({**recs, artifact: {**recs[artifact], "transcription": "WORDS SHE NEVER WROTE"}})
    problems = verify_repairs(d, evaluation_root=EVAL_ROOT)["problems"]
    assert any("carries a transcription" in p["problem"] for p in problems)
    assert any("never contained it" in p["problem"] for p in problems)


def test_the_recovery_chain_to_the_replaced_crop_is_verified_not_just_claimed(live_dataset_copy: Path):
    """Decision 1 requires the old crop to stay recoverable. That has to be
    checked, not asserted: point a record at a crop that is gone, or at one
    whose bytes moved, and the verifier must say so."""
    d = live_dataset_copy
    store = RepairStore(d)
    recs = store.records()
    assert verify_repairs(d, evaluation_root=EVAL_ROOT)["problems"] == []
    lid = sorted(recs)[0]

    store._rewrite({**recs, lid: {**recs[lid], "original_crop": {**recs[lid]["original_crop"],
                                                                 "sha256": "0" * 64}}})
    assert any("no longer hashes to the sha256" in p["problem"]
               for p in verify_repairs(d, evaluation_root=EVAL_ROOT)["problems"])

    store._rewrite({**recs, lid: {**recs[lid], "original_crop": {**recs[lid]["original_crop"],
                                                                 "image": "htr_pilot/images/e003/GONE.png"}}})
    assert any("is gone" in p["problem"] for p in verify_repairs(d, evaluation_root=EVAL_ROOT)["problems"])

    store._rewrite({**recs, lid: {**recs[lid], "original_crop": None}})
    assert any("no record of the crop this repair replaced" in p["problem"]
               for p in verify_repairs(d, evaluation_root=EVAL_ROOT)["problems"])

    store._rewrite({**recs, lid: {**recs[lid], "created_at": "1999-01-01 00:00:00"}})
    assert any("created_at no longer matches" in p["problem"]
               for p in verify_repairs(d, evaluation_root=EVAL_ROOT)["problems"])

    store._rewrite(recs)
    assert verify_repairs(d, evaluation_root=EVAL_ROOT)["problems"] == []


def test_the_live_recovery_chain_is_actually_intact():
    """All nine replaced crops are still on disk and still hash to what the
    decision recorded — the history Decision 1 promised is really there."""
    rep = verify_repairs(REAL_DATASET, evaluation_root=EVAL_ROOT)
    assert rep["problems"] == [] and rep["ok"] is True
    for line_id, rec in repair_store().items():
        oc = rec["original_crop"]
        assert (EVAL_ROOT / oc["image"]).exists(), line_id
        assert _sha(EVAL_ROOT / oc["image"]) == oc["sha256"], line_id
