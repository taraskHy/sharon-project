"""End-to-end smoke of the training pipeline on SYNTHETIC data only.

Proves prepare -> train -> decode -> eval runs on this machine (GPU if
available) without any real annotations: builds a throwaway package with
machine-drawn scribble images and dummy Hebrew labels, trains the CRNN a
few dozen epochs (it should start memorizing the tiny set — loss must
drop), decodes val, and scores it with the real eval harness. Nothing
touches evaluation/htr_pilot or any real label.

    .venv-train/Scripts/python.exe scripts/htr_train_smoke.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.htr_annotation_lib import make_record, save_annotation

REPO = Path(__file__).resolve().parents[1]
WORDS = ["שלום", "עולם", "תמונה", "היסטוגרמה", "תדרים", "פעולה"]


def draw_line_image(rng: np.random.Generator, w: int = 420) -> np.ndarray:
    img = np.full((128, w), 255, np.uint8)
    for _ in range(rng.integers(6, 14)):
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(20, 108))
        x1, y1 = int(rng.integers(0, w)), int(rng.integers(20, 108))
        cv2.line(img, (x0, y0), (x1, y1), 0, int(rng.integers(2, 5)))
    return img


def build_dummy_package(root: Path) -> None:
    rng = np.random.default_rng(7)
    samples_by_split = {"train": [], "val": [], "internal_test": []}
    writers = {"e003": "train", "e004": "train", "e013": "val"}
    for wr, split in writers.items():
        img_dir = root / "images" / wr
        img_dir.mkdir(parents=True, exist_ok=True)
        for r in range(1, 7):
            cell = f"q1_r{r}"
            line = draw_line_image(rng)
            cv2.imwrite(str(img_dir / f"{cell}_l1.png"), line)
            cv2.imwrite(str(img_dir / f"{cell}_cell_clean.png"), line)
            cv2.imwrite(str(img_dir / f"{cell}_cell_orig.jpg"), line)
            samples_by_split[split].append({
                "sample_id": f"{wr}_{cell}__l1", "writer": wr, "split": split,
                "question": 1, "row": r, "line_index": 1, "n_lines": 1,
                "expected_blank": False,
                "images": {"line": f"images/{wr}/{cell}_l1.png",
                           "cell_clean": f"images/{wr}/{cell}_cell_clean.png",
                           "cell_orig": f"images/{wr}/{cell}_cell_orig.jpg"},
                "line_size": [128, 420],
            })
    (root / "splits").mkdir(parents=True)
    for split, recs in samples_by_split.items():
        (root / "splits" / f"{split}.json").write_text(
            json.dumps(recs, ensure_ascii=False), encoding="utf-8")
    for split in samples_by_split:
        (root / "annotations" / split).mkdir(parents=True)
    rng2 = np.random.default_rng(11)
    for split in ("train", "val"):
        for s in samples_by_split[split]:
            text = " ".join(rng2.choice(WORDS, size=2))
            save_annotation(root, make_record(s, text, "ok",
                                              annotator="smoke-dummy"))


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], cwd=REPO, capture_output=True,
                          text=True, encoding="utf-8", **kw)


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="htr_train_smoke_"))
    ws = tmp / "ws"
    try:
        build_dummy_package(tmp / "pkg")
        py = sys.executable

        p = run([py, "-X", "utf8", "scripts/htr_train_prepare.py",
                 "--root", tmp / "pkg", "--out", ws, "--aug", "3"])
        assert p.returncode == 0, p.stdout + p.stderr
        summary = json.loads((ws / "prepare_summary.json").read_text(encoding="utf-8"))
        assert summary["splits"]["train"]["written_images"] == 12 * 4
        print("1. prepare OK:", summary["splits"]["train"])

        p = run([py, "-X", "utf8", "scripts/htr_pilot_train.py",
                 "--workspace", ws, "train", "--epochs", "60",
                 "--batch-size", "8", "--patience", "60"])
        assert p.returncode == 0, p.stdout + p.stderr
        lines = [ln for ln in p.stdout.splitlines() if ln.startswith("epoch")]
        first = float(lines[0].split()[3])
        best = min(float(ln.split()[3]) for ln in lines)
        assert best < first, f"loss never dropped: first={first} best={best}"
        assert (ws / "model" / "crnn_best.pt").exists()
        print(f"2. train OK: {len(lines)} epochs, loss {first:.2f} -> {best:.2f}")

        p = run([py, "-X", "utf8", "scripts/htr_pilot_train.py",
                 "--workspace", ws, "decode", "--split", "val",
                 "--out", "decodes/val_smoke.txt"])
        assert p.returncode == 0, p.stdout + p.stderr
        dec = (ws / "decodes/val_smoke.txt").read_text(encoding="utf-8")
        assert len(dec.splitlines()) == 6
        print("3. decode OK: 6 val lines with confidences")

        p = run([py, "-X", "utf8", "scripts/htr_pilot_train.py",
                 "--workspace", ws, "decode", "--split", "internal_test",
                 "--out", "decodes/never.txt"])
        assert p.returncode == 2 and "REFUSING" in p.stdout
        print("4. internal_test decode refused without flag")

        p = run([py, "-X", "utf8", "scripts/htr_pilot_eval.py",
                 str(ws / "decodes/val_smoke.txt"),
                 "--root", tmp / "pkg", "--split", "val"])
        assert p.returncode == 0, p.stdout + p.stderr
        rep = json.loads(p.stdout[p.stdout.index("{"):p.stdout.rindex("}") + 1])
        assert rep["scoreable_lines"] == 6 and "confidence_abstention_curve" in rep
        print(f"5. eval OK: cell CER {rep['mean_cell_cer']} on dummy data "
              "(number meaningless by design)")
        print("SMOKE: PASS")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
