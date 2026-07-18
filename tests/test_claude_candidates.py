"""Safety tests for the Claude Vision assisted-annotation experiment.

Prove: candidates can never become verified automatically; verified
labels stay overwrite-locked; hidden ground truth cannot enter the
inference payload; candidate generation is absent from the app unless
the pre-registered gate passes (disabled by default / manual workflow
unchanged on failure); the acceptance gate cannot be silently weakened.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.claude_candidates_run import (
    B_CONTEXT_TEXT, INSTRUCTION, MODEL, build_request,
)
from scripts.htr_annotation_lib import locked_against_overwrite, make_record
from tests.test_htr_annotation import mini_package

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "evaluation/claude_candidates"


# ---- ground truth cannot enter the inference payload -------------------

def test_request_builder_takes_only_pixels():
    """build_request accepts image bytes only — there is no parameter
    through which a transcription, id, or filename could enter."""
    params = list(inspect.signature(build_request).parameters)
    assert params == ["config", "line_png", "cell_png"]


@pytest.mark.parametrize("config", ["claude_line", "claude_line_cell"])
def test_payload_contains_only_fixed_text_and_images(config):
    req = build_request(config, b"PNG1",
                        b"PNG2" if config == "claude_line_cell" else None)
    assert req["model"] == MODEL
    assert req["system"] == INSTRUCTION
    texts = [b["text"] for b in req["messages"][0]["content"]
             if b.get("type") == "text"]
    assert texts == ([] if config == "claude_line" else [B_CONTEXT_TEXT])
    # fixed text carries no Hebrew beyond the owner's unreadable token
    for t in [INSTRUCTION, *texts]:
        rest = t.replace("[לא קריא]", "")
        assert not any("֐" <= c <= "ת" for c in rest)
    # every non-text block is a base64 image; nothing else rides along
    for b in req["messages"][0]["content"]:
        assert b["type"] in ("image", "text")
        if b["type"] == "image":
            assert set(b["source"]) == {"type", "media_type", "data"}


def test_generator_never_writes_annotations_and_never_sends_cell_orig():
    src = (REPO / "scripts/claude_candidates_run.py").read_text(encoding="utf-8")
    for symbol in ("make_record", "save_annotation"):
        assert symbol not in src
    # cell_orig may carry instructor ink — the code must never access it
    # (the docstring may mention it as the documented exclusion)
    assert '["cell_orig"]' not in src and "'cell_orig'" not in src


def test_selection_is_ids_only_and_excludes_overfit():
    sel_path = OUT / "claude_bench_ids.json"
    if not sel_path.exists():
        pytest.skip("selection not recorded yet")
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    assert set(sel) <= {"rule", "n", "per_writer", "ids", "at"}
    overfit = set(json.loads(
        (REPO / "evaluation/htr_overfit_test/selected_ids.json")
        .read_text(encoding="utf-8"))["picked"])
    assert not set(sel["ids"]) & overfit
    assert sel["per_writer"] == {"e004": 10, "e005": 10, "e006": 10}
    assert not any(s.startswith(("e003", "e007")) for s in sel["ids"])


# ---- candidates can never become verified -------------------------------

def test_claude_outputs_are_never_verified_and_live_outside_annotations():
    outputs = list((OUT / "outputs").rglob("e0*.json"))
    if not outputs:
        pytest.skip("no Claude outputs generated yet")
    for p in outputs:
        rec = json.loads(p.read_text(encoding="utf-8"))
        assert rec["verified"] is False, p
        assert "human_verified" not in rec, p
        assert "htr_pilot" not in str(p)


def test_verified_labels_remain_locked(tmp_path):
    pkg = mini_package(tmp_path)
    verified = make_record(pkg["train"][0], "שלום", "ok")
    assert locked_against_overwrite(verified) is True
    assert locked_against_overwrite(verified, unlocked=True) is False


# ---- disabled by default / manual workflow unchanged on failure ---------

def test_app_has_no_claude_candidate_ui_unless_gate_accepted():
    summary_path = OUT / "eval_summary.json"
    accepted = (summary_path.exists() and
                json.loads(summary_path.read_text(encoding="utf-8"))
                .get("verdict") == "ACCEPT")
    if accepted:
        pytest.skip("gate ACCEPTED — candidate UI is permitted")
    app_src = (REPO / "scripts/htr_annotation_app.py").read_text(encoding="utf-8")
    lib_src = (REPO / "scripts/htr_annotation_lib.py").read_text(encoding="utf-8")
    assert "claude" not in app_src.lower(), \
        "candidate UI present without an ACCEPT verdict"
    assert "claude" not in lib_src.lower()


# ---- the acceptance gate cannot be silently weakened ---------------------

def test_gate_constants_match_owner_protocol():
    from scripts.claude_candidates_eval import GATE
    assert GATE["min_exact_rate"] == 0.40
    assert GATE["max_median_cer"] == 0.10
    assert GATE["max_major_halluc_line_rate"] == 0.05
    assert GATE["agreement_subset_max_cer"] == 0.10
    protocol = (OUT / "PROTOCOL.md").read_text(encoding="utf-8")
    assert "at least 40 % exact-match lines" in protocol
    assert "median CER at most 0.10" in protocol
    assert "more than 5 % of lines" in protocol


def test_eval_refuses_incomplete_outputs(tmp_path, monkeypatch):
    """Raw-before-eval: the evaluator must refuse to score a partial run."""
    import subprocess
    if not (OUT / "outputs" / "claude_line").exists():
        proc = subprocess.run(
            [sys.executable, "-X", "utf8", "scripts/claude_candidates_eval.py"],
            capture_output=True, text=True, encoding="utf-8", cwd=REPO)
        assert proc.returncode == 2 and "REFUSING" in proc.stdout
    else:
        pytest.skip("outputs exist — refusal path not applicable")
