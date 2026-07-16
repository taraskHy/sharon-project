"""Tests for the HTR training-data preparation (label filtering, split
protection, deterministic augmentation)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.htr_annotation_lib import make_record, save_annotation
from tests.test_htr_annotation import mini_package

REPO = Path(__file__).resolve().parents[1]


def run_prepare(root: Path, out: Path, *extra: str):
    proc = subprocess.run(
        [sys.executable, "-X", "utf8", "scripts/htr_train_prepare.py",
         "--root", str(root), "--out", str(out), "--aug", "2", *extra],
        capture_output=True, text=True, encoding="utf-8", cwd=REPO)
    return proc.returncode, proc.stdout + proc.stderr


def annotate_mix(tmp_path, pkg):
    """train sample 1: ok; train sample 2: unreadable; val 1: ok."""
    s1, s2 = pkg["train"]
    save_annotation(tmp_path, make_record(s1, "שלום עולם", "ok"))
    save_annotation(tmp_path, make_record(s2, "", "unreadable_full"))
    v1 = pkg["val"][0]
    save_annotation(tmp_path, make_record(v1, "בדיקה", "ok"))


def test_prepare_filters_and_outputs(tmp_path):
    pkg = mini_package(tmp_path)
    annotate_mix(tmp_path, pkg)
    out = tmp_path / "ws"
    code, log = run_prepare(tmp_path, out)
    assert code == 0, log
    summary = json.loads((out / "prepare_summary.json").read_text(encoding="utf-8"))
    tr = summary["splits"]["train"]
    assert tr["kept_lines"] == 1
    assert tr["excluded"]["unreadable_full"] == 1
    assert tr["written_images"] == 3  # base + 2 augs
    va = summary["splits"]["val"]
    assert va["kept_lines"] == 1 and va["written_images"] == 1  # no val aug
    syms = (out / "syms.txt").read_text(encoding="utf-8").splitlines()
    assert syms[0].startswith("<ctc>") and syms[1].startswith("<space>")
    assert any(line.split()[0] == "ש" for line in syms)
    text = (out / "text" / "train.txt").read_text(encoding="utf-8")
    assert "<space>" in text and "__aug2" in text


def test_prepare_refuses_internal_test_without_flag(tmp_path):
    pkg = mini_package(tmp_path)
    annotate_mix(tmp_path, pkg)
    code, log = run_prepare(tmp_path, tmp_path / "ws",
                            "--splits", "train,val,internal_test")
    assert code == 2 and "REFUSING" in log


def test_prepare_requires_train_labels(tmp_path):
    mini_package(tmp_path)  # no annotations at all
    code, log = run_prepare(tmp_path, tmp_path / "ws")
    assert code == 3 and "annotate" in log


def test_display_order_involution_and_ltr_runs():
    from scripts.htr_train_prepare import to_display_order as disp
    cases = [
        "שלום עולם",
        "העברנו High pass 123 מסנן",
        "כל ערך x בהיסטוגרמה ממופה ל2x",
        "פי 2",
        "abc",
        "",
    ]
    for s in cases:
        assert disp(disp(s)) == s, f"not an involution for {s!r}"
    # pure Hebrew: plain reversal
    assert disp("אבג") == "גבא"
    # embedded LTR run keeps internal order, Hebrew reverses around it
    d = disp("אבג High pass דהו")
    assert "High pass" in d and d.startswith("והד") and d.endswith("גבא")


def test_augmentation_is_deterministic(tmp_path):
    pkg = mini_package(tmp_path)
    annotate_mix(tmp_path, pkg)
    out1, out2 = tmp_path / "ws1", tmp_path / "ws2"
    assert run_prepare(tmp_path, out1)[0] == 0
    assert run_prepare(tmp_path, out2)[0] == 0
    sid = pkg["train"][0]["sample_id"]
    a = (out1 / "imgs/train" / f"{sid}__aug1.png").read_bytes()
    b = (out2 / "imgs/train" / f"{sid}__aug1.png").read_bytes()
    assert a == b
