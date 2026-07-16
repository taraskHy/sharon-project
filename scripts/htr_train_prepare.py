"""Prepare PyLaia training data from the HTR-pilot annotations.

Reads ONLY the requested splits' metadata + annotation directories
(default: train and val — internal_test is refused without the explicit
--allow-internal-test flag, which exists solely for the final one-shot
report). Training lines are the owner's verified `ok` records; lines that
are blank, whole-line unreadable, contain a partial [לא קריא] span, are
flagged (bad_segmentation / needs_recrop), skipped, drafts, or simply not
yet annotated are excluded and counted per reason.

Outputs under --out (default evaluation/htr_train_workspace/, which is
git-ignored except for its README):

    imgs/<split>/<id>.png        fixed-height grayscale line images
    imgs/train/<id>__augK.png    K deterministic augmentations (train only)
    lists/<split>.txt            image ids, one per line
    text/<split>.txt             "<id> <sym> <sym> ...": char symbols,
                                 spaces as <space>
    syms.txt                     CTC symbol table (fixed base alphabet +
                                 any train chars), <ctc> first
    prepare_summary.json         counts + exclusions per split

Augmentation is deterministic (crc32-seeded per sample copy): small
affine (rotation/shear/scale), elastic displacement, brightness/contrast
jitter, 1 px morphological thickness jitter. Nothing is ever invented —
these are label-preserving distortions of real ink.

    .venv/Scripts/python.exe scripts/htr_train_prepare.py [--aug 5]
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from scripts.htr_annotation_lib import (  # noqa: E402
    UNREADABLE_TOKEN, load_all_annotations, load_samples,
)

LINE_HEIGHT = 128
SEED_TAG = 20260714

# Fixed base alphabet so a val/test char outside the train text never
# forces exclusion: Hebrew (incl. finals), digits, Latin, punctuation.
BASE_CHARS = (
    "אבגדהוזחטיכךלמםנןסעפףצץקרשת"
    "0123456789"
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ".,;:!?()[]{}\"'`%+-=*/\\<>_|#@&^~$"
)


def rng_for(sample_id: str, k: int) -> np.random.Generator:
    seed = zlib.crc32(f"{SEED_TAG}:{sample_id}:{k}".encode()) & 0xFFFFFFFF
    return np.random.default_rng(seed)


def load_line_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    s = LINE_HEIGHT / img.shape[0]
    w = max(8, int(round(img.shape[1] * s)))
    return cv2.resize(img, (w, LINE_HEIGHT), interpolation=cv2.INTER_AREA)


def augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape
    # small affine: rotation, shear, scale
    ang = rng.uniform(-2.0, 2.0)
    shear = rng.uniform(-0.05, 0.05)
    scale = rng.uniform(0.95, 1.05)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, scale)
    M[0, 1] += shear
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderValue=255)
    # elastic displacement
    alpha, sigma = rng.uniform(8, 18), rng.uniform(8, 12)
    dx = cv2.GaussianBlur((rng.random((h, w)).astype(np.float32) * 2 - 1),
                          (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((rng.random((h, w)).astype(np.float32) * 2 - 1),
                          (0, 0), sigma) * alpha
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    out = cv2.remap(out, xx + dx, yy + dy, cv2.INTER_LINEAR,
                    borderValue=255, borderMode=cv2.BORDER_CONSTANT)
    # thickness jitter
    r = rng.random()
    if r < 0.33:
        out = cv2.erode(out, np.ones((2, 2), np.uint8))
    elif r < 0.66:
        out = cv2.dilate(out, np.ones((2, 2), np.uint8))
    # brightness/contrast jitter
    a = rng.uniform(0.85, 1.1)
    b = rng.uniform(-12, 12)
    return np.clip(out.astype(np.float32) * a + b, 0, 255).astype(np.uint8)


LTR_RUN = __import__("re").compile(r"[A-Za-z0-9]+(?: [A-Za-z0-9]+)*")


def to_display_order(text: str) -> str:
    """Logical (typed) Hebrew -> left-to-right display order.

    CTC alignment is monotonic in TIME (image frames left to right), while
    Hebrew is written right to left: training on logical-order labels is
    unlearnable beyond trivial lengths. Weak-bidi approximation: reverse
    the whole string, then restore each embedded LTR run (Latin/digit
    words incl. their inner spaces) to forward order. The transform is an
    involution, so the same function converts model output back to
    logical order. Neutral punctuation inside math (e.g. '->') is not
    special-cased — documented pilot limitation."""
    rev = text[::-1]
    return LTR_RUN.sub(lambda m: m.group(0)[::-1], rev)


def to_syms(text: str) -> list[str]:
    return ["<space>" if c == " " else c for c in to_display_order(text)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="evaluation/htr_pilot")
    ap.add_argument("--out", default="evaluation/htr_train_workspace")
    ap.add_argument("--splits", default="train,val")
    ap.add_argument("--aug", type=int, default=5,
                    help="augmented copies per TRAIN line (0 disables)")
    ap.add_argument("--allow-internal-test", action="store_true",
                    help="required to touch internal_test (final report only)")
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if "internal_test" in splits and not args.allow_internal_test:
        print("REFUSING: internal_test requested without --allow-internal-test "
              "(reserved for the single final report)")
        return 2

    summary: dict = {"splits": {}, "aug_per_train_line": args.aug,
                     "line_height": LINE_HEIGHT, "seed_tag": SEED_TAG}
    train_chars: set[str] = set()
    kept_by_split: dict[str, list] = {}

    for split in splits:
        samples = load_samples(root, split)
        ann = load_all_annotations(root, split)
        kept, excluded = [], {}
        for s in samples:
            rec = ann.get(s["sample_id"])
            if rec is None:
                reason = "unannotated"
            elif rec["status"] != "ok":
                reason = rec["status"]
            elif UNREADABLE_TOKEN in rec["transcription"]:
                reason = "partial_unreadable_span"
            elif not rec["human_verified"]:
                reason = "not_verified"
            else:
                kept.append((s, rec["transcription"]))
                continue
            excluded[reason] = excluded.get(reason, 0) + 1
        kept_by_split[split] = kept
        summary["splits"][split] = {
            "total_samples": len(samples), "kept_lines": len(kept),
            "excluded": excluded,
        }
        if split == "train":
            for _s, t in kept:
                train_chars.update(t)

    if not kept_by_split.get("train"):
        print("no usable train lines yet — annotate the train split first "
              f"({json.dumps(summary['splits'].get('train', {}))})")
        return 3

    # symbol table: fixed base + observed train chars (space handled as token)
    charset = sorted(set(BASE_CHARS) | (train_chars - {" "}))
    syms = ["<ctc>", "<space>"] + charset
    out.mkdir(parents=True, exist_ok=True)
    (out / "syms.txt").write_text(
        "\n".join(f"{s} {i}" for i, s in enumerate(syms)) + "\n",
        encoding="utf-8")

    (out / "lists").mkdir(exist_ok=True)
    (out / "text").mkdir(exist_ok=True)
    known = set(syms)
    for split in splits:
        img_dir = out / "imgs" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        ids, text_rows, skipped_chars = [], [], 0
        for s, text in kept_by_split[split]:
            symbols = to_syms(text)
            if any(sym not in known for sym in symbols):
                skipped_chars += 1
                continue
            sid = s["sample_id"]
            base = load_line_gray(root / s["images"]["line"])
            cv2.imwrite(str(img_dir / f"{sid}.png"), base)
            ids.append(sid)
            text_rows.append(f"{sid} {' '.join(symbols)}")
            if split == "train":
                for k in range(1, args.aug + 1):
                    aug_id = f"{sid}__aug{k}"
                    cv2.imwrite(str(img_dir / f"{aug_id}.png"),
                                augment(base, rng_for(sid, k)))
                    ids.append(aug_id)
                    text_rows.append(f"{aug_id} {' '.join(symbols)}")
        (out / "lists" / f"{split}.txt").write_text(
            "\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")
        (out / "text" / f"{split}.txt").write_text(
            "\n".join(text_rows) + ("\n" if text_rows else ""), encoding="utf-8")
        summary["splits"][split]["written_images"] = len(ids)
        summary["splits"][split]["skipped_unknown_char_lines"] = skipped_chars

    summary["n_symbols"] = len(syms)
    (out / "prepare_summary.json").write_text(json.dumps(summary, indent=1),
                                              encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
