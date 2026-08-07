"""Row-band table cropping: pure-geometry tests plus the banded extraction
path end-to-end on a mock backend.

Motivating failure (live, 2026-08-07): whole-page extraction of the prob
answer table returned 9/10 wrong letters at high confidence with zero review
flags. Banding must therefore be deterministic, validated, and fall back
loudly — these tests pin all three properties.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import fitz
import numpy as np
import pytest

from autograder.backends import BackendConfig
from autograder.backends.mock import MockBackend
from autograder.cli import run_grade_pipeline
from autograder.ingest import PageImage, load_pages
from autograder.schema import BandRowExtraction, SubItemExtraction, VariantDetection
from autograder.tablecrop import (
    TableCropError,
    _encode_png_gray,
    answer_table_row_bands,
)
from tests.test_mission_ui_modes import FIXTURES, PROB_KEY, _fixture

REPO = Path(__file__).resolve().parents[1]

N_ROWS = 10


def _grid_image(
    height: int = 1400,
    width: int = 960,
    y0: int = 500,
    spacing: int = 52,
    x_range: tuple[int, int] = (230, 760),
    skip_lines: tuple[int, ...] = (),
    extra_line_y: int | None = None,
) -> np.ndarray:
    img = np.full((height, width), 255, dtype=np.uint8)
    for k in range(N_ROWS + 2):
        if k in skip_lines:
            continue
        y = y0 + k * spacing
        img[y : y + 2, x_range[0] : x_range[1]] = 0
    for x in (230, 340, 450, 560, 670, 760):
        img[y0 : y0 + (N_ROWS + 1) * spacing + 2, x : x + 2] = 0
    if extra_line_y is not None:
        img[extra_line_y : extra_line_y + 2, 100:900] = 0
    return img


def _page(img: np.ndarray) -> PageImage:
    return PageImage(
        page_number=1,
        png_bytes=_encode_png_gray(img),
        width=img.shape[1],
        height=img.shape[0],
        text="",
    )


def test_synthetic_grid_yields_ordered_bands():
    bands = answer_table_row_bands(_page(_grid_image()), n_rows=N_ROWS)
    assert [b.row_index for b in bands] == list(range(N_ROWS))
    assert len({b.width for b in bands}) == 1
    # header + separator + row, upscaled 2x: roughly 2*(52+3+52) plus margins
    for b in bands:
        assert 180 <= b.height <= 260
        assert b.png_bytes.startswith(b"\x89PNG")


def test_faded_interior_and_edge_lines_are_lattice_filled():
    img = _grid_image(skip_lines=(4, 9, 11))
    bands = answer_table_row_bands(_page(img), n_rows=N_ROWS)
    assert len(bands) == N_ROWS


def test_stray_long_rule_far_above_table_is_ignored():
    img = _grid_image(extra_line_y=120)  # e.g. a name-field underline
    bands = answer_table_row_bands(_page(img), n_rows=N_ROWS)
    assert len(bands) == N_ROWS


def test_blank_page_raises():
    img = np.full((1400, 960), 255, dtype=np.uint8)
    with pytest.raises(TableCropError):
        answer_table_row_bands(_page(img), n_rows=N_ROWS)


def test_wrong_row_count_raises():
    with pytest.raises(TableCropError):
        answer_table_row_bands(_page(_grid_image()), n_rows=7)


@pytest.mark.skipif(
    not (REPO / "prob_data" / "02.pdf").exists(), reason="prob dataset not present"
)
@pytest.mark.parametrize("scan", ["02", "13"])
def test_real_prob_scans_produce_ten_bands(scan):
    """13.pdf has faded grid lines — the lattice fill must bridge them."""
    pages = load_pages(REPO / "prob_data" / f"{scan}.pdf", max_long_edge=1400)
    bands = answer_table_row_bands(pages[0], n_rows=10)
    assert len(bands) == 10


# --------------------------------------------------------------------------
# banded extraction end-to-end (deterministic CV first; mock model)
# --------------------------------------------------------------------------

CLUB_ANSWERS = ["C", "A", "B", "D", "A", "A", "D", "D", "B", "A"]

# Grid geometry (PDF points). Option column x-spans, RIGHT to LEFT = A,B,C,D.
_Y0, _SPACING, _X0, _X1 = 250.0, 32.0, 60.0, 540.0
_VLINES = (60.0, 156.0, 252.0, 348.0, 444.0, 540.0)
_COL_SPAN = {"A": (348.0, 444.0), "B": (252.0, 348.0), "C": (156.0, 252.0), "D": (60.0, 156.0)}


def make_grid_exam_pdf(
    path: Path,
    answers: list[str | None] = None,
    scribble: dict[int, str] | None = None,
) -> Path:
    """Three-page exam PDF; page 1 holds a 10-row grid with drawn marks.

    ``answers[i]`` draws a clean X in that letter's cell of row i+1 (None =
    leave the row empty). ``scribble`` additionally BLACKS OUT a cell per
    row number (1-based) — a cancelled choice."""
    answers = CLUB_ANSWERS if answers is None else answers
    scribble = scribble or {}
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    for k in range(12):
        y = _Y0 + k * _SPACING
        page.draw_line(fitz.Point(_X0, y), fitz.Point(_X1, y), width=1.5)
    for x in _VLINES:
        page.draw_line(fitz.Point(x, _Y0), fitz.Point(x, _Y0 + 11 * _SPACING), width=1.5)

    def cell_center(row_1based: int, letter: str):
        x0, x1 = _COL_SPAN[letter]
        yc = _Y0 + (row_1based + 0.5) * _SPACING  # +1 row for the header band
        return (x0 + x1) / 2, yc

    for i, letter in enumerate(answers, start=1):
        if letter is None:
            continue
        cx, cy = cell_center(i, letter)
        r = 6.0
        page.draw_line(fitz.Point(cx - r, cy - r), fitz.Point(cx + r, cy + r), width=2.2)
        page.draw_line(fitz.Point(cx - r, cy + r), fitz.Point(cx + r, cy - r), width=2.2)
    for row, letter in scribble.items():
        cx, cy = cell_center(row, letter)
        page.draw_rect(fitz.Rect(cx - 8, cy - 8, cx + 8, cy + 8), color=0, fill=0)
    doc.new_page(width=595, height=842)
    doc.new_page(width=595, height=842)
    doc.save(path)
    doc.close()
    return path


def _cv_backend() -> MockBackend:
    """Variant detection + advisory disambiguation only — extraction itself
    must be deterministic (any BandRowExtraction/QuestionExtraction call fails)."""
    from autograder.schema import MarkDisambiguation

    detection = VariantDetection.model_validate(_fixture("VariantDetection"))
    detection.matched_marker = "club"
    detection.marker_seen = "a club symbol next to בהצלחה!"

    def responder(model, system, blocks):
        if model is VariantDetection:
            return detection.model_copy(deep=True)
        if model is MarkDisambiguation:
            return MarkDisambiguation(
                cancelled_columns=["D"], final_column="B",
                reasoning="the D cell is blacked out; B holds a clean X",
            )
        raise AssertionError(f"unexpected model call: {model.__name__}")

    return MockBackend(
        config=BackendConfig(backend="mock", model="prob-cv"),
        responder=responder,
    )


def _run(tmp_path: Path, backend: MockBackend, exam: Path):
    ns = argparse.Namespace(
        key=str(PROB_KEY), rubric=None, resume=False, version="auto",
        exam=str(exam), variant_map=None, alignment_map=None, template=None,
        no_key_cache=True, key_cache_dir=str(tmp_path / "kc"), mask=False,
    )
    return run_grade_pipeline(
        ns, backend, tmp_path / "out", 800,
        exam_path=exam, exam_label="exam-001", survey_image_edge=400,
    )


def test_clean_marks_read_deterministically_no_extraction_model_calls(tmp_path):
    backend = _cv_backend()
    exam = make_grid_exam_pdf(tmp_path / "student.pdf")
    result = _run(tmp_path, backend, exam)

    called = [c.output_model for c in backend.calls]
    assert "VariantDetection" in called
    assert "BandRowExtraction" not in called, "clean rows need no model reads"
    assert "QuestionExtraction" not in called
    assert "MarkDisambiguation" not in called, "no multi-mark rows here"
    assert result.detected_version == "club"
    assert result.total_awarded == 100.0
    assert not result.needs_human_review


def test_multi_mark_row_flagged_ambiguous_with_advisory_proposal(tmp_path):
    """Row 4: clean X in B plus a blacked-out D cell (cancelled). The row must
    be ambiguous + review with both candidates; the model proposal is recorded
    but never decides; the answer earns 0 pending review."""
    backend = _cv_backend()
    answers = list(CLUB_ANSWERS)
    answers[3] = "B"  # the clean X
    exam = make_grid_exam_pdf(tmp_path / "student.pdf", answers=answers,
                              scribble={4: "D"})
    result = _run(tmp_path, backend, exam)

    assert "MarkDisambiguation" in [c.output_model for c in backend.calls]
    q = result.questions[0]
    row4 = next(s for s in q.sub_results if s.sub_item_id == "4")
    assert row4.status == "ambiguous"
    assert row4.student_answer is None
    assert set("BD") <= set("".join(row4.reason.split())) or "B" in row4.reason
    assert any(
        item.sub_item_id == "4" for item in result.needs_human_review
    ), "multi-mark rows must reach human review"
    # advisory proposal recorded in the audit trail, clearly labeled
    assert "ADVISORY ONLY" in row4.reason
    # other 9 rows still read deterministically and correctly (club column)
    assert result.total_awarded == 90.0


def test_unanswered_row_detected_and_noted(tmp_path):
    backend = _cv_backend()
    answers = list(CLUB_ANSWERS)
    answers[6] = None  # row 7 left empty
    exam = make_grid_exam_pdf(tmp_path / "student.pdf", answers=answers)
    result = _run(tmp_path, backend, exam)
    q = result.questions[0]
    row7 = next(s for s in q.sub_results if s.sub_item_id == "7")
    assert row7.status == "unanswered"
    assert result.total_awarded == 90.0


def test_grid_analysis_failure_falls_back_to_vlm_band_reads(tmp_path, monkeypatch):
    """When cell analysis raises, the per-row VLM path takes over."""
    import autograder.extract as extract_mod
    from autograder.tablecrop import TableCropError

    def boom(page, n_rows, n_options=4):
        raise TableCropError("synthetic failure")

    import autograder.tablecrop as tc
    monkeypatch.setattr(tc, "analyze_answer_table", boom)

    detection = VariantDetection.model_validate(_fixture("VariantDetection"))
    detection.matched_marker = "club"

    def responder(model, system, blocks):
        if model is VariantDetection:
            return detection.model_copy(deep=True)
        if model is BandRowExtraction:
            text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            m = re.search(r"question number (\S+) ", text)
            rid = m.group(1)
            return BandRowExtraction(
                printed_row_number=rid,
                row=SubItemExtraction(
                    sub_item_id=rid, status="answered", answer_origin="answer_sheet",
                    final_answer=CLUB_ANSWERS[int(rid) - 1],
                    interpretation_rationale="X", confidence=0.9,
                ),
            )
        raise AssertionError(f"unexpected model call: {model.__name__}")

    backend = MockBackend(
        config=BackendConfig(backend="mock", model="prob-fallback"),
        responder=responder,
    )
    exam = make_grid_exam_pdf(tmp_path / "student.pdf")
    result = _run(tmp_path, backend, exam)
    assert [c for c in backend.calls if c.output_model == "BandRowExtraction"], (
        "VLM band fallback must engage when analysis fails"
    )
    assert result.total_awarded == 100.0


def test_band_printed_number_mismatch_is_flagged_ambiguous(tmp_path, monkeypatch):
    """VLM fallback path: a mismatched printed row number → review."""
    import autograder.tablecrop as tc
    from autograder.tablecrop import TableCropError

    monkeypatch.setattr(
        tc, "analyze_answer_table",
        lambda page, n_rows, n_options=4: (_ for _ in ()).throw(TableCropError("x")),
    )
    detection = VariantDetection.model_validate(_fixture("VariantDetection"))
    detection.matched_marker = "club"

    def responder(model, system, blocks):
        if model is VariantDetection:
            return detection.model_copy(deep=True)
        if model is BandRowExtraction:
            text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            rid = re.search(r"question number (\S+) ", text).group(1)
            return BandRowExtraction(
                printed_row_number="9" if rid == "3" else rid,
                row=SubItemExtraction(
                    sub_item_id=rid, status="answered", answer_origin="answer_sheet",
                    final_answer=CLUB_ANSWERS[int(rid) - 1],
                    interpretation_rationale="X", confidence=0.9,
                ),
            )
        raise AssertionError(f"unexpected: {model.__name__}")

    backend = MockBackend(
        config=BackendConfig(backend="mock", model="prob-mismatch"),
        responder=responder,
    )
    exam = make_grid_exam_pdf(tmp_path / "student.pdf")
    result = _run(tmp_path, backend, exam)
    row3 = next(s for s in result.questions[0].sub_results if s.sub_item_id == "3")
    assert row3.student_answer is None
    assert any(
        "registration mismatch" in (item.reason or "")
        for item in result.needs_human_review
    )


def test_blank_first_page_falls_back_to_whole_page_extraction(tmp_path):
    """No grid on page 1 → banding declines entirely, generic extraction runs."""
    from tests.test_mission_ui_modes import _grade_prob, _prob_backend

    backend = _prob_backend("club")
    result = _grade_prob(tmp_path, backend)
    assert [c for c in backend.calls if c.output_model == "QuestionExtraction"]
    assert not [c for c in backend.calls if c.output_model == "BandRowExtraction"]
    assert result.total_awarded == 100.0
