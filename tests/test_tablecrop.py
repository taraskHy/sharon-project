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
# banded extraction end-to-end (mock model)
# --------------------------------------------------------------------------

CLUB_ANSWERS = ["C", "A", "B", "D", "A", "A", "D", "D", "B", "A"]


def make_grid_exam_pdf(path: Path) -> Path:
    """Three-page exam PDF whose first page holds a real 10-row grid table."""
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)
    y0, spacing, x0, x1 = 250.0, 32.0, 60.0, 540.0
    for k in range(12):
        y = y0 + k * spacing
        page.draw_line(fitz.Point(x0, y), fitz.Point(x1, y), width=1.5)
    for x in (60.0, 156.0, 252.0, 348.0, 444.0, 540.0):
        page.draw_line(
            fitz.Point(x, y0), fitz.Point(x, y0 + 11 * spacing), width=1.5
        )
    doc.new_page(width=595, height=842)
    doc.new_page(width=595, height=842)
    doc.save(path)
    doc.close()
    return path


def _banded_backend(answers: dict[str, str] | None = None,
                    printed_override: dict[str, str] | None = None) -> MockBackend:
    """Serves VariantDetection (club) and per-row BandRowExtraction calls."""
    detection = VariantDetection.model_validate(_fixture("VariantDetection"))
    detection.matched_marker = "club"
    detection.marker_seen = "a club symbol next to בהצלחה!"
    answers = answers or {str(i): a for i, a in enumerate(CLUB_ANSWERS, start=1)}
    printed_override = printed_override or {}

    def responder(model, system, blocks):
        if model is VariantDetection:
            return detection.model_copy(deep=True)
        if model is BandRowExtraction:
            text = " ".join(b.get("text", "") for b in blocks if b.get("type") == "text")
            m = re.search(r"question number (\S+) ", text)
            assert m, f"band call without a requested row: {text!r}"
            rid = m.group(1)
            return BandRowExtraction(
                printed_row_number=printed_override.get(rid, rid),
                row=SubItemExtraction(
                    sub_item_id=rid,
                    status="answered",
                    answer_origin="answer_sheet",
                    final_answer=answers[rid],
                    interpretation_rationale=f"single X mark under column {answers[rid]}",
                    confidence=0.97,
                ),
            )
        raise AssertionError(f"unexpected model call: {model.__name__}")

    return MockBackend(
        config=BackendConfig(backend="mock", model="prob-banded"),
        responder=responder,
    )


def _run(tmp_path: Path, backend: MockBackend):
    exam = make_grid_exam_pdf(tmp_path / "student.pdf")
    ns = argparse.Namespace(
        key=str(PROB_KEY), rubric=None, resume=False, version="auto",
        exam=str(exam), variant_map=None, alignment_map=None, template=None,
        no_key_cache=True, key_cache_dir=str(tmp_path / "kc"), mask=False,
    )
    return run_grade_pipeline(
        ns, backend, tmp_path / "out", 800,
        exam_path=exam, exam_label="exam-001", survey_image_edge=400,
    )


def test_banded_extraction_one_small_call_per_row(tmp_path):
    backend = _banded_backend()
    result = _run(tmp_path, backend)

    band_calls = [c for c in backend.calls if c.output_model == "BandRowExtraction"]
    assert len(band_calls) == 10, "exactly one call per table row"
    assert not [c for c in backend.calls if c.output_model == "QuestionExtraction"], (
        "banded path must replace whole-page extraction"
    )
    full_page = load_pages(tmp_path / "student.pdf", max_long_edge=800)[0]
    for c in band_calls:
        images = [b for b in c.content_blocks if b.get("type") == "image"]
        assert len(images) == 1, "each band call carries exactly one crop"
        import base64 as b64
        crop = len(b64.b64decode(images[0]["source"]["data"]))
        assert crop < len(full_page.png_bytes), "crop must be smaller than the page"

    assert result.detected_version == "club"
    assert result.total_awarded == 100.0
    assert not result.needs_human_review


def test_band_printed_number_mismatch_is_flagged_ambiguous(tmp_path):
    backend = _banded_backend(printed_override={"3": "9"})
    result = _run(tmp_path, backend)
    q = result.questions[0]
    row3 = next(s for s in q.sub_results if s.sub_item_id == "3")
    assert row3.student_answer is None
    assert any(
        "registration mismatch" in (item.reason or "")
        for item in result.needs_human_review
    ), "a band registration mismatch must reach human review"
    assert result.total_awarded < 100.0


def test_blank_first_page_falls_back_to_whole_page_extraction(tmp_path):
    """No grid on page 1 → banding declines, generic extraction runs."""
    from tests.test_mission_ui_modes import _grade_prob, _prob_backend

    backend = _prob_backend("club")
    result = _grade_prob(tmp_path, backend)
    assert [c for c in backend.calls if c.output_model == "QuestionExtraction"]
    assert not [c for c in backend.calls if c.output_model == "BandRowExtraction"]
    assert result.total_awarded == 100.0
