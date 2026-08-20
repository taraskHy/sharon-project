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
