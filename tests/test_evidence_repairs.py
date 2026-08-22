"""Manual GRADE_PRIMARY evidence repairs — the integrity rules, as tests.

Every test works on a COPY of the real frozen grading dataset in tmp_path, so
the geometry, the crops and the frozen OCR benchmark are the real ones (read
only) while nothing under version control is written. No OCR, no model, no
network: the only image work is exact sub-image location and rectangle cropping.
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
from autograder.benchmark.manifests import DEFAULT_DATASETS_ROOT, REPO_ROOT

EVAL_ROOT = REPO_ROOT / "evaluation"
REAL_DATASET = DEFAULT_DATASETS_ROOT / "grade_primary"
DATASET_FILES = ("cases_inputs.jsonl", "cases_labels.jsonl", "manifest.json", "CHECKSUMS.sha256")

pytestmark = pytest.mark.skipif(not (REAL_DATASET / "manifest.json").exists(),
                                reason="grade_primary dataset is not built here")


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _copy_dataset(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    for name in DATASET_FILES:
        shutil.copy2(REAL_DATASET / name, dest / name)
    return dest


@pytest.fixture()
def dataset(tmp_path: Path) -> Path:
    """A writable copy of the frozen grading dataset."""
    return _copy_dataset(tmp_path / "grade_primary")


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
                   verified_by="t", crop_png=png)
    with pytest.raises(RepairError, match="not a recorded line"):     # geometry refuses even earlier
        suggested_band(geo, exp[1]["line_id"])
    assert not store.records()


def test_store_rejects_inadmissible_repairs(dataset: Path):
    item = expected_repairs(dataset)[0]
    store = RepairStore(dataset)
    geo = case_geometry(item["case_id"], evaluation_root=EVAL_ROOT)
    png = render_band(geo, 0, 20)
    base = dict(case_id=item["case_id"], line_id=item["line_id"], verified_by="t", crop_png=png)
    with pytest.raises(RepairError, match="handwritten text"):
        store.save(**base, transcription="   ")
    with pytest.raises(RepairError, match="carries no transcription"):
        store.save(**base, transcription="x", disposition="no_text_segmentation_artifact")
    with pytest.raises(RepairError, match="who verified"):
        _repair(store, item, verified_by="  ")
    with pytest.raises(RepairError, match="unknown disposition"):
        _repair(store, item, disposition="guessed")
    with pytest.raises(RepairError, match="persist the crop"):
        store.save(case_id=item["case_id"], line_id=item["line_id"], transcription="x", verified_by="t")
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
    out = apply_repairs(dataset, evaluation_root=EVAL_ROOT, dry_run=True)
    assert out["written"] is False and out["inputs_changed"] is True
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
