"""Streamlit smoke test for the manual reference-audit UI (offline)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

REPO_ROOT = Path(__file__).resolve().parents[1]
UI = REPO_ROOT / "scripts" / "reference_audit_ui.py"


def _bench(tmp_path: Path) -> Path:
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNiAAAABgADNjd8"
        "qAAAAABJRU5ErkJggg==")
    d = tmp_path / "bench"
    (d / "crops").mkdir(parents=True)
    items = []
    for item_id, text in (("a1", "שורה ראשונה"), ("a2", "שורה שנייה")):
        (d / "crops" / f"{item_id}.png").write_bytes(png)
        items.append({"id": item_id, "category": "handwritten_line",
                      "tier": "owner", "hard": False,
                      "image": f"crops/{item_id}.png", "writer": "e003",
                      "task": "transcribe"})
    (d / "items.json").write_text(json.dumps({"version": 2, "items": items},
                                             ensure_ascii=False), encoding="utf-8")
    (d / "references.json").write_text(json.dumps(
        {"a1": {"text": "שורה ראשונה", "provenance": "owner"},
         "a2": {"text": "שורה שנייה", "provenance": "owner"}},
        ensure_ascii=False), encoding="utf-8")
    return d


def test_ui_confirm_persists_and_resumes(tmp_path, monkeypatch, no_network):
    bench = _bench(tmp_path)
    monkeypatch.setenv("REFAUDIT_BENCH_DIR", str(bench))

    at = AppTest.from_file(str(UI), default_timeout=30)
    at.run()
    assert not at.exception
    # confirm the first item through the real button
    at.button(key="confirm").click().run()
    assert not at.exception

    audit = json.loads((bench / "reference_audit.json").read_text(encoding="utf-8"))
    assert audit["entries"]["a1"]["status"] == "confirmed"
    assert audit["entries"]["a1"]["audited_reference"] == "שורה ראשונה"

    # a fresh session resumes: item a1 is decided, position moves to a2
    at2 = AppTest.from_file(str(UI), default_timeout=30)
    at2.run()
    assert not at2.exception
    assert at2.session_state["pos"] == 1
