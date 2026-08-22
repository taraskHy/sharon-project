"""Streamlit smoke test for the manual evidence-repair UI (offline, no model).

The tool exists so a human can transcribe the nine student lines the OCR
benchmark never audited. These tests prove it renders the real evidence, that
saving through the real button produces an admissible record, and that the
instructor's grade is never on screen while the transcription is being typed.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from autograder.benchmark.evidence_repairs import RepairStore, expected_repairs, repair_status
from autograder.benchmark.manifests import DEFAULT_DATASETS_ROOT

REPO_ROOT = Path(__file__).resolve().parents[1]
UI = REPO_ROOT / "scripts" / "evidence_repair_ui.py"
REAL_DATASET = DEFAULT_DATASETS_ROOT / "grade_primary"
DATASET_FILES = ("cases_inputs.jsonl", "cases_labels.jsonl", "manifest.json", "CHECKSUMS.sha256")

pytestmark = pytest.mark.skipif(not (REAL_DATASET / "manifest.json").exists(),
                                reason="grade_primary dataset is not built here")


@pytest.fixture()
def datasets_root(tmp_path: Path) -> Path:
    """A writable copy of the dataset, laid out the way the UI expects."""
    d = tmp_path / "datasets" / "grade_primary"
    d.mkdir(parents=True)
    for name in DATASET_FILES:
        shutil.copy2(REAL_DATASET / name, d / name)
    return d.parent


def _open(datasets_root: Path, timeout: int = 180) -> AppTest:
    at = AppTest.from_file(str(UI), default_timeout=timeout).run()
    assert not at.exception, at.exception
    at.sidebar.text_input[0].set_value(str(datasets_root)).run()
    assert not at.exception, at.exception
    return at


def _screen_text(at: AppTest) -> str:
    parts: list[str] = []
    for kind in ("title", "subheader", "header", "markdown", "caption", "warning", "info", "error", "success", "code"):
        parts += [str(getattr(e, "value", "")) for e in getattr(at, kind)]
    parts += [str(e.value) for e in at.text_area]
    parts += [str(e.value) for e in at.text_input]
    parts += [json.dumps(e.value, ensure_ascii=False, default=str) for e in at.json]
    parts += [str(getattr(e, "label", "")) for e in at.button]
    return "\n".join(parts)


def test_ui_shows_the_first_unrepaired_line_with_its_real_context(datasets_root: Path, no_network):
    at = _open(datasets_root)
    first = expected_repairs(datasets_root / "grade_primary")[0]
    text = _screen_text(at)
    assert first["case_id"] in text
    assert first["line_id"] in text
    assert "line 2 of 2" in text, "the human is told which line of the answer this is"
    assert "bad_segmentation" in text, "the mis-segmentation is called out before transcribing"
    assert any("Transcription of THIS line" in (t.label or "") for t in at.text_area)
    assert any(n.label == "top (y0)" for n in at.number_input), "the crop rectangle is adjustable"
    assert "1 / 9" in text, "the worklist is the nine expected lines"


def test_ui_never_shows_the_instructor_grade(datasets_root: Path, no_network):
    """Seeing the grade first would bias the transcription."""
    d = datasets_root / "grade_primary"
    rows = [json.loads(l) for l in (d / "cases_labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    for r in rows:
        r["score"] = 7.0
        r["rubric_met"] = ["SENTINEL_RUBRIC_LEAK"]
        r["owner_note"] = "SENTINEL_NOTE_LEAK"
    (d / "cases_labels.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                          encoding="utf-8", newline="\n")
    text = _screen_text(_open(datasets_root))
    assert "SENTINEL_RUBRIC_LEAK" not in text
    assert "SENTINEL_NOTE_LEAK" not in text
    assert "score" not in text.lower()


def test_saving_through_the_ui_records_an_admissible_repair(datasets_root: Path, no_network):
    d = datasets_root / "grade_primary"
    item = expected_repairs(d)[0]
    at = _open(datasets_root)
    at.text_area(key="tx_" + item["line_id"]).set_value("שורה שנייה שנכתבה ביד").run()
    at.text_input(key="by_" + item["line_id"]).set_value("owner").run()
    at.button(key="save_" + item["line_id"]).click().run()
    assert not at.exception, at.exception

    rec = RepairStore(d).get(item["line_id"])
    assert rec is not None
    assert rec["transcription"] == "שורה שנייה שנכתבה ביד"
    assert rec["human_verified"] is True and rec["verified_by"] == "owner"
    assert rec["disposition"] == "transcribed"
    assert (d / rec["crop_path"]).exists()
    assert rec["crop_geometry"]["y1"] > rec["crop_geometry"]["y0"]
    assert rec["source_pdf"] and rec["source_page"], "the source page is recorded with the repair"
    assert rec["original_crop"]["status"] == "bad_segmentation", "the crop it replaces is recorded too"
    assert repair_status(d)["repaired"] == 1
    after = _screen_text(at)                      # the repaired line leaves the worklist; the next one is shown
    assert expected_repairs(d)[1]["line_id"] in after
    assert item["line_id"] not in after
    assert "1 / 8" in after and "8 of 9" not in after
    assert "1 of 9 lines repaired" in after


def test_the_artifact_path_saves_without_inventing_text(datasets_root: Path, no_network):
    d = datasets_root / "grade_primary"
    item = expected_repairs(d)[0]
    at = _open(datasets_root)
    at.checkbox(key="art_" + item["line_id"]).set_value(True).run()
    assert not at.text_area, "no transcription box is offered once the region is declared an artifact"
    at.button(key="save_" + item["line_id"]).click().run()
    assert not at.exception, at.exception
    rec = RepairStore(d).get(item["line_id"])
    assert rec["disposition"] == "no_text_segmentation_artifact" and rec["transcription"] == ""


def test_the_crop_buttons_move_the_rectangle(datasets_root: Path, no_network):
    d = datasets_root / "grade_primary"
    item = expected_repairs(d)[0]
    at = _open(datasets_root)
    suggested = (at.number_input(key="y0_" + item["line_id"]).value, at.number_input(key="y1_" + item["line_id"]).value)
    at.button(key="orig_" + item["line_id"]).click().run()
    original = (at.number_input(key="y0_" + item["line_id"]).value, at.number_input(key="y1_" + item["line_id"]).value)
    assert original != suggested, "the mis-segmented band and the proposed repair differ"
    at.button(key="whole_" + item["line_id"]).click().run()
    assert at.number_input(key="y0_" + item["line_id"]).value == 0
    at.button(key="sug_" + item["line_id"]).click().run()
    assert (at.number_input(key="y0_" + item["line_id"]).value,
            at.number_input(key="y1_" + item["line_id"]).value) == suggested


def test_ui_reports_completion_and_the_apply_command(datasets_root: Path, no_network):
    from tests.test_evidence_repairs import _repair_all
    d = datasets_root / "grade_primary"
    _repair_all(d)
    text = _screen_text(_open(datasets_root))
    assert "All expected lines are repaired." in text
    assert "apply-evidence-repairs" in text, "the tool hands over the exact next command"
