"""Safety tests for the assisted-annotation campaign (2026-07-17).

Covers the campaign's non-negotiables: candidate text can never become a
verified label automatically, verified labels are overwrite-locked,
ground truth cannot enter candidate prompts, candidate metadata carries
image hashes, writer-grouped folds are leak-free, contaminated CRNN
predictions are excluded, held-out protections hold, and the annotation
backup is restorable.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.annotation_candidates_eval import CONFIGS as EVAL_CONFIGS
from scripts.annotation_candidates_run import STRICT_PROMPT, build_messages
from scripts.htr_annotation_lib import (
    locked_against_overwrite, make_record, save_annotation,
)
from tests.test_htr_annotation import mini_package

REPO = Path(__file__).resolve().parents[1]
CAND = REPO / "evaluation/htr_candidates"
DIAG = REPO / "evaluation/htr_gen_diag"


# ---- candidate text can never become verified -------------------------

def test_candidate_records_are_never_verified():
    """Every candidate artifact on disk must be marked unverified."""
    outputs = list((CAND / "outputs").rglob("e0*.json"))
    if not outputs:
        pytest.skip("no candidate outputs generated")
    for p in outputs:
        rec = json.loads(p.read_text(encoding="utf-8"))
        assert rec["verified"] is False, p
        assert "human_verified" not in rec, p


def test_candidate_module_cannot_write_annotations():
    """The generator must not even import the annotation-write API."""
    src = (REPO / "scripts/annotation_candidates_run.py").read_text(
        encoding="utf-8")
    for symbol in ("make_record", "save_annotation", "load_all_annotations",
                   "load_annotation"):
        assert symbol not in src, symbol
    # candidates live outside the pilot package's annotations tree
    for p in (CAND / "outputs").rglob("*.json"):
        assert "htr_pilot" not in str(p)


# ---- verified labels are overwrite-locked -----------------------------

def test_locked_against_overwrite_semantics(tmp_path):
    pkg = mini_package(tmp_path)
    sample = pkg["train"][0]
    verified = make_record(sample, "שלום", "ok")
    flagged = make_record(sample, "x", "bad_segmentation")
    assert locked_against_overwrite(verified) is True
    assert locked_against_overwrite(verified, unlocked=True) is False
    assert locked_against_overwrite(flagged) is False   # unverified: editable
    assert locked_against_overwrite(None) is False       # new sample


def test_app_commit_is_guarded():
    src = (REPO / "scripts/htr_annotation_app.py").read_text(encoding="utf-8")
    commit_body = src.split("def commit(")[1]
    assert "locked_against_overwrite" in commit_body.split("def ")[0], \
        "app commit() lost its verified-record guard"


# ---- ground truth cannot enter prompts --------------------------------

@pytest.mark.parametrize("config", ["qwen_line", "qwen_line_cell"])
def test_prompts_contain_only_fixed_text_and_images(config):
    line_png, cell_png = b"PNG1", b"PNG2"
    msgs = build_messages(config, line_png,
                          cell_png if config == "qwen_line_cell" else None)
    assert msgs[0] == {"role": "system", "content": STRICT_PROMPT}
    texts = [part["text"] for part in msgs[1]["content"]
             if part.get("type") == "text"]
    allowed = {
        "Transcribe now.",
        "The first image is the full answer cell (context only). "
        "The second image is a single line cropped from that cell. "
        "Transcribe ONLY the line shown in the second image.",
    }
    assert set(texts) <= allowed
    # no Hebrew (i.e. no transcription content) anywhere in the fixed text
    for t in [STRICT_PROMPT, *texts]:
        assert not any("֐" <= c <= "ת" for c in t)


def test_bench_selection_stores_ids_only():
    sel_path = CAND / "bench_ids.json"
    if not sel_path.exists():
        pytest.skip("benchmark selection not built")
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    assert set(sel) <= {"rule", "excluded_overfit_ids", "n", "ids", "at"}
    assert all(isinstance(i, str) and i.startswith("e0") for i in sel["ids"])


# ---- candidate metadata + image hashes --------------------------------

def test_candidate_metadata_and_image_hashes():
    outputs = sorted((CAND / "outputs").rglob("e0*.json"))
    if not outputs:
        pytest.skip("no candidate outputs generated")
    root = REPO / "evaluation/htr_pilot"
    for p in outputs[::7]:  # sample every 7th: full pass is slow on images
        rec = json.loads(p.read_text(encoding="utf-8"))
        for key in ("sample_id", "config", "model", "model_digest", "raw",
                    "candidate", "confidence", "latency_s", "at"):
            assert key in rec, (p, key)
        for img in rec["images"].values():
            digest = hashlib.sha256(
                (root / img["path"]).read_bytes()).hexdigest()
            assert digest == img["sha256"], p


# ---- writer-grouped fold integrity ------------------------------------

def test_fold_workspaces_are_writer_separated():
    folds = sorted(DIAG.glob("fold_*"))
    if not folds:
        pytest.skip("diagnostic folds not built")
    for ws in folds:
        held = ws.name.split("_")[1]
        train_ids = (ws / "lists/train.txt").read_text(encoding="utf-8").split()
        held_ids = (ws / "lists/heldout.txt").read_text(encoding="utf-8").split()
        assert train_ids and held_ids
        assert all(i.startswith(held) for i in held_ids)
        assert not any(i.startswith(held) for i in train_ids), \
            f"{ws.name}: held-out writer leaked into training"
        # empty val: nothing from the held-out writer can steer selection
        assert (ws / "lists/val.txt").read_text(encoding="utf-8").strip() == ""
        # symbol table exists and starts with the CTC blank
        syms = (ws / "syms.txt").read_text(encoding="utf-8").splitlines()
        assert syms[0].startswith("<ctc>")


def test_diagnostic_eligibility_excludes_flagged_and_span(tmp_path, monkeypatch):
    import scripts.writer_gen_diagnostic as wgd
    pkg = mini_package(tmp_path)
    s1, s2 = pkg["train"]
    save_annotation(tmp_path, make_record(s1, "שלום עולם", "ok"))
    save_annotation(tmp_path, make_record(s2, "יש [לא קריא] כאן", "ok"))
    monkeypatch.setattr(wgd, "ROOT", tmp_path)
    kept = wgd.eligible_lines()
    assert [s["sample_id"] for s, _t in kept] == [s1["sample_id"]]


# ---- contaminated CRNN predictions are excluded ------------------------

def test_bench_excludes_overfit_ids_and_eval_tracks_contamination():
    sel_path = CAND / "bench_ids.json"
    if not sel_path.exists():
        pytest.skip("benchmark selection not built")
    sel = json.loads(sel_path.read_text(encoding="utf-8"))
    overfit = set(json.loads(
        (REPO / "evaluation/htr_overfit_test/selected_ids.json")
        .read_text(encoding="utf-8"))["picked"])
    assert not set(sel["ids"]) & overfit
    assert set(sel["excluded_overfit_ids"]) == overfit
    summary_path = CAND / "eval_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        assert set(summary["contaminated_excluded"]) == set(EVAL_CONFIGS)
        assert not any(summary["contaminated_excluded"].values()), \
            "contaminated ids reached the benchmark"


# ---- held-out protections stay wired -----------------------------------

def test_crnn_decode_refuses_internal_test_without_flag(tmp_path):
    train_py = REPO / ".venv-train/Scripts/python.exe"
    if not train_py.exists():
        pytest.skip("training venv not installed")
    ws = tmp_path / "ws"
    (ws / "lists").mkdir(parents=True)
    (ws / "syms.txt").write_text("<ctc> 0\n<space> 1\n", encoding="utf-8")
    proc = subprocess.run(
        [str(train_py), "-X", "utf8", "scripts/htr_pilot_train.py",
         "--workspace", str(ws), "decode", "--split", "internal_test",
         "--out", "d.txt", "--device", "cpu"],
        capture_output=True, text=True, encoding="utf-8", cwd=REPO)
    assert proc.returncode == 2 and "REFUSING" in proc.stdout


# ---- backup restorability ----------------------------------------------

def test_latest_annotation_backup_is_restorable():
    backups = sorted((REPO / "evaluation/annotation_backups").glob("*/"))
    if not backups:
        pytest.skip("no annotation backup yet")
    bk = backups[-1]
    manifest = (bk / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines()
    assert manifest
    for line in manifest:
        digest, rel = line.split(maxsplit=1)
        rel = rel.lstrip("*")
        data = (bk / rel).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest, rel
