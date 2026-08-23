"""Shared test helpers."""

from pathlib import Path

import fitz
import pytest


def make_pdf(path: Path, pages: int = 2, with_red: bool = True) -> Path:
    """Create a small synthetic 'scan' PDF (text + optional red rectangle)."""
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=200, height=280)
        page.insert_text((20, 40), f"synthetic page {i + 1}")
        if with_red:
            page.draw_rect(
                fitz.Rect(20, 60, 80, 100), color=(1, 0, 0), fill=(1, 0, 0)
            )
        page.draw_rect(fitz.Rect(20, 120, 80, 160), color=(0, 0, 1), fill=(0, 0, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def no_network(monkeypatch):
    """Fail the test if anything attempts a network connection."""
    import socket

    def _blocked(*args, **kwargs):
        raise AssertionError("network access attempted during an offline test")

    monkeypatch.setattr(socket, "create_connection", _blocked)
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    return None


@pytest.fixture(autouse=True, scope="session")
def _labeling_data_dir_is_never_the_live_one(tmp_path_factory):
    """No test may resolve to the real deployment's labeling data directory.

    Belt: every test session runs with LABELING_DATA_DIR pointed at a throwaway
    directory, so `default_data_dir()` can never return the live one.
    Braces: `LabelDB.__init__` refuses the live labels.db under pytest anyway
    (labeling_app.db.assert_not_live_database) — a test that hardcodes the path
    still fails before SQLite is opened.

    This exists because a test once opened the live database read-write and
    physically corrupted its `items` table."""
    import os
    sandbox = tmp_path_factory.mktemp("labeling_data_dir")
    previous = os.environ.get("LABELING_DATA_DIR")
    os.environ["LABELING_DATA_DIR"] = str(sandbox)
    try:
        yield sandbox
    finally:
        if previous is None:
            os.environ.pop("LABELING_DATA_DIR", None)
        else:
            os.environ["LABELING_DATA_DIR"] = previous


@pytest.fixture
def pre_repair_dataset(tmp_path):
    """The grade_primary dataset as it was BEFORE the owner's manual evidence
    repair, reconstructed into tmp_path and hash-verified against the sha256
    pair the manifest revision recorded. The checked-in dataset is post-repair
    and is never written by a test."""
    from tests.prerepair import DATASET, build_pre_repair_dataset
    if not (DATASET / "manifest.json").exists():
        pytest.skip("grade_primary dataset is not built here")
    return build_pre_repair_dataset(tmp_path / "datasets" / "grade_primary")


@pytest.fixture
def live_dataset_copy(tmp_path):
    """A writable copy of the CURRENT (repaired) dataset, with the owner's real
    repair store beside it — for exercising apply/verify against the real
    post-repair state without touching the original."""
    from tests.prerepair import DATASET, copy_live_dataset
    if not (DATASET / "manifest.json").exists():
        pytest.skip("grade_primary dataset is not built here")
    return copy_live_dataset(tmp_path / "datasets" / "grade_primary", with_repairs=True)


@pytest.fixture(autouse=True)
def _reset_pipeline_hooks():
    """``orchestrator.install_hooks`` mutates PROCESS-GLOBAL state (the MC
    resolution chain and the per-question grading policies). A test that
    enables the gateway runtime must not change how the next test's pipeline
    behaves, so the hooks are removed after every test."""
    yield
    from autograder import extract, grade

    grade.set_grading_policies(None)
    extract.set_mc_resolver(None)
