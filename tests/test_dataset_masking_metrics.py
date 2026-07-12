"""Dataset discovery/split, annotation masking, and metrics tests."""

import numpy as np

from autograder.dataset import (
    ExamRecord,
    anon_id_for,
    assign_split,
    discover_exams,
    load_manifest,
    parse_exam_filename,
    write_manifests,
)
from autograder.ingest import PageImage
from autograder.masking import _array_to_png, _png_to_array, mask_page
from autograder.metrics import ExamOutcome, compute_metrics
from tests.conftest import make_pdf


# --------------------------------------------------------------------------
# dataset
# --------------------------------------------------------------------------


def test_parse_exam_filename():
    assert parse_exam_filename("02_78.pdf") == ("02", 78)
    assert parse_exam_filename("042_100.PDF") == ("042", 100)
    assert parse_exam_filename("042_101.pdf") is None  # grade > 100
    assert parse_exam_filename("exam.pdf") is None
    assert parse_exam_filename("02-78.pdf") is None
    assert parse_exam_filename("02_78.docx") is None


def test_anon_id_contains_no_grade():
    assert anon_id_for("02") == "exam-002"
    assert "78" not in anon_id_for("02")


def test_discover_reports_malformed_and_duplicates(tmp_path):
    root = tmp_path / "exams"
    make_pdf(root / "01_50.pdf")
    make_pdf(root / "001_60.pdf")  # duplicate index 1 (canonicalised)
    make_pdf(root / "notes.pdf")  # malformed
    make_pdf(root / "02_80.pdf")
    report = discover_exams(root, repo_root=tmp_path)
    assert [r.anon_id for r in report.records] == ["exam-001", "exam-002"]
    assert report.malformed == ["notes.pdf"]
    assert len(report.duplicate_indices) == 1
    # page-count warning: synthetic PDFs are 2 pages, expected form is 13
    assert all(r.warnings for r in report.records)


def test_split_is_deterministic_and_disjoint(tmp_path):
    root = tmp_path / "exams"
    for i in range(1, 11):
        make_pdf(root / f"{i:02d}_{50 + i}.pdf", pages=1)
    r1 = discover_exams(root, repo_root=tmp_path)
    r2 = discover_exams(root, repo_root=tmp_path)
    assign_split(r1.records, seed=42)
    assign_split(r2.records, seed=42)
    assert [(r.anon_id, r.split) for r in r1.records] == [
        (r.anon_id, r.split) for r in r2.records
    ]
    train = {r.anon_id for r in r1.records if r.split == "train"}
    val = {r.anon_id for r in r1.records if r.split == "validation"}
    assert train and val
    assert not train & val


def test_manifests_roundtrip_and_final_placeholder(tmp_path):
    root = tmp_path / "exams"
    for i in range(1, 6):
        make_pdf(root / f"{i:02d}_{60 + i}.pdf", pages=1)
    report = discover_exams(root, repo_root=tmp_path)
    assign_split(report.records)
    paths = write_manifests(report, tmp_path / "datasets")
    train = load_manifest(paths["train"])
    val = load_manifest(paths["validation"])
    assert len(train) + len(val) == 5
    assert all(isinstance(r, ExamRecord) for r in train + val)
    import json

    final = json.loads(paths["final_test"].read_text(encoding="utf-8"))
    assert final["entries"] == []
    assert "unseen" in final["note"]


# --------------------------------------------------------------------------
# masking
# --------------------------------------------------------------------------


def _page_from_array(arr: np.ndarray) -> PageImage:
    return PageImage(
        page_number=1,
        png_bytes=_array_to_png(arr),
        width=arr.shape[1],
        height=arr.shape[0],
        text="",
    )


def test_mask_removes_red_keeps_blue_and_black():
    arr = np.full((100, 100, 3), 255, dtype=np.uint8)
    arr[10:20, 10:30] = [200, 30, 30]  # red ink (instructor)
    arr[50:60, 10:30] = [30, 30, 200]  # blue ink (student)
    arr[80:85, 10:30] = [10, 10, 10]  # black print
    masked, report = mask_page(_page_from_array(arr))
    out = _png_to_array(masked.png_bytes)
    assert (out[15, 20] == [255, 255, 255]).all(), "red must be masked to white"
    assert (out[55, 20] == [30, 30, 200]).all(), "blue must be preserved"
    assert (out[82, 20] == [10, 10, 10]).all(), "black must be preserved"
    assert report.masked_pixels == 10 * 20
    assert report.regions, "masked regions must be recorded"
    r = report.regions[0]
    assert r.x0 <= 10 and r.y0 <= 10 and r.x1 >= 30 and r.y1 >= 20


def test_mask_flags_high_red_fraction():
    arr = np.full((100, 100, 3), 255, dtype=np.uint8)
    arr[:50, :] = [220, 40, 40]  # half the page is red
    _, report = mask_page(_page_from_array(arr))
    assert report.warning is not None


def test_mask_noop_on_clean_page():
    arr = np.full((60, 60, 3), 255, dtype=np.uint8)
    page = _page_from_array(arr)
    masked, report = mask_page(page)
    assert report.masked_pixels == 0
    assert masked.png_bytes == page.png_bytes


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_metrics_computation():
    outcomes = [
        ExamOutcome(anon_id="a", expected=80, predicted=80, review_items=0, runtime_s=10),
        ExamOutcome(anon_id="b", expected=70, predicted=75, review_items=2, runtime_s=20),
        ExamOutcome(anon_id="c", expected=60, predicted=50, review_items=0, runtime_s=30),
        ExamOutcome(anon_id="d", failed=True, failure_reason="boom", expected=90),
    ]
    m = compute_metrics(outcomes)
    assert m.processed == 4
    assert m.failures == 1
    assert m.scored == 3
    assert m.exact == 1 / 3
    assert m.within_5 == 2 / 3
    assert m.within_10 == 1.0
    assert m.mae == 5.0
    assert m.median_ae == 5.0
    assert m.mean_signed_error == (0 + 5 - 10) / 3
    assert m.max_abs_error == 10
    assert m.review_rate == 1 / 3
    assert m.mean_runtime_s == 20


def test_metrics_empty():
    m = compute_metrics([])
    assert m.processed == 0
    assert m.to_dict()["mae"] == 0
