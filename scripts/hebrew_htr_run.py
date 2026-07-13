"""Iteration-6 HTR pipeline: deterministic segmentation + sivan22/hdd-words-ocr.

Runs INSIDE .venv-htr (CPU torch + transformers). Dedicated Hebrew
handwriting recognizer, word-level VisionEncoderDecoder loaded from
SAFETENSORS at a pinned revision with trust_remote_code=False. The complete
segmentation-plus-recognition pipeline is what gets evaluated:

  cell crop -> line bands (projection profile, scripts/segment_lines.py
  logic) -> word bands per line (vertical projection, RTL order) -> one
  recognition call per word -> words joined right-to-left per line ->
  lines joined top-to-bottom.

Raw output only: no spell correction, no LM repair, no semantic completion,
no exam context. Word crops are saved for inspection. Ground truth is never
read here.

    .venv-htr/Scripts/python.exe scripts/hebrew_htr_run.py --config-id it6_hdd_words --runs 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BENCH = Path("evaluation/hebrew_bench")
MODEL_ID = "sivan22/hdd-words-ocr"
REVISION = "e089ce717594610492d8c53d9e35ec5b80b402bb"  # pinned 2023-06-05

MIN_WORD_WIDTH_FRAC = 0.015
WORD_GAP_FRAC = 0.012


def to_gray(png: bytes) -> np.ndarray:
    from PIL import Image
    import io

    img = Image.open(io.BytesIO(png)).convert("L")
    return np.asarray(img, dtype=np.float64)


def line_bands(gray: np.ndarray) -> list[tuple[int, int]]:
    h, w = gray.shape
    lo, hi = np.percentile(gray, 5), np.percentile(gray, 95)
    ink = gray < (lo + 0.45 * (hi - lo))
    fr_h, fr_w = int(h * 0.02) + 1, int(w * 0.02) + 1
    ink[:fr_h, :] = ink[-fr_h:, :] = False
    ink[:, :fr_w] = ink[:, -fr_w:] = False
    profile = ink.sum(axis=1) / w
    rows = profile > 0.01
    bands, start, gap = [], None, 0
    max_gap = int(h * 0.035)
    for y in range(h):
        if rows[y]:
            if start is None:
                start = y
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                bands.append((start, y - gap))
                start = None
    if start is not None:
        bands.append((start, h - 1))
    pad = int(h * 0.02) + 1
    out = [(max(0, a - pad), min(h, b + pad)) for a, b in bands if (b - a) >= h * 0.08]
    return out or [(0, h)]


def word_bands(line: np.ndarray) -> list[tuple[int, int]]:
    h, w = line.shape
    lo, hi = np.percentile(line, 5), np.percentile(line, 95)
    ink = line < (lo + 0.45 * (hi - lo))
    cols = ink.sum(axis=0) / max(h, 1)
    on = cols > 0.02
    bands, start, gap = [], None, 0
    max_gap = max(int(w * WORD_GAP_FRAC), 3)
    for x in range(w):
        if on[x]:
            if start is None:
                start = x
            gap = 0
        elif start is not None:
            gap += 1
            if gap > max_gap:
                bands.append((start, x - gap))
                start = None
    if start is not None:
        bands.append((start, w - 1))
    pad = max(int(w * 0.004), 2)
    out = [(max(0, a - pad), min(w, b + pad)) for a, b in bands if (b - a) >= w * MIN_WORD_WIDTH_FRAC]
    return out or [(0, w)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-id", default="it6_hdd_words")
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--manifest", default=str(BENCH / "crops_manifest.json"),
                    help="crop manifest; entries with a 'lines' list are "
                         "treated as pre-segmented line images (the internal "
                         "line segmentation is skipped, word segmentation kept)")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoImageProcessor, AutoTokenizer

    t_load = time.monotonic()
    try:
        processor = TrOCRProcessor.from_pretrained(MODEL_ID, revision=REVISION)
        image_processor, tokenizer = processor.image_processor, processor.tokenizer
    except Exception:
        image_processor = AutoImageProcessor.from_pretrained(MODEL_ID, revision=REVISION)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    model = VisionEncoderDecoderModel.from_pretrained(
        MODEL_ID, revision=REVISION, use_safetensors=True, trust_remote_code=False
    )
    model.eval()
    torch.set_num_threads(6)
    print(f"model loaded in {time.monotonic() - t_load:.1f}s; params={sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    outdir = BENCH / "outputs" / args.config_id
    segroot = BENCH / "segments_words" / args.config_id
    meta = {
        "config_id": args.config_id, "model": f"{MODEL_ID}@{REVISION[:12]}",
        "prompt": "n/a (dedicated HTR; greedy decode)", "preproc": "gray+projection-line+word-segmentation",
        "max_tokens": 24, "runs": args.runs, "manifest": args.manifest,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    t0 = time.monotonic()
    import io as _io

    def png_of(arr: np.ndarray) -> Image.Image:
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")

    for run in range(1, args.runs + 1):
        rundir = outdir / f"run{run}"
        rundir.mkdir(exist_ok=True)
        for cell in manifest:
            target = rundir / f"{cell['id']}.json"
            if target.exists():
                continue
            if "lines" in cell:
                line_arrays = [to_gray(Path(p).read_bytes()) for p in cell["lines"]]
            else:
                gray = to_gray(Path(cell["file"]).read_bytes())
                line_arrays = [gray[y0:y1, :] for y0, y1 in line_bands(gray)]
            t1 = time.monotonic()
            lines_text = []
            n_words = 0
            for li, line in enumerate(line_arrays, 1):
                words = word_bands(line)
                # RTL reading order: rightmost word first.
                words = sorted(words, key=lambda ab: -ab[0])
                tokens = []
                for wi, (x0, x1) in enumerate(words, 1):
                    crop = line[:, x0:x1]
                    if run == 1:
                        segdir = segroot / cell["id"]
                        segdir.mkdir(parents=True, exist_ok=True)
                        png_of(crop).save(segdir / f"line{li}_word{wi}.png")
                    pixel_values = image_processor(png_of(crop), return_tensors="pt").pixel_values
                    with torch.no_grad():
                        ids = model.generate(pixel_values, max_new_tokens=24, num_beams=1, do_sample=False)
                    tokens.append(tokenizer.decode(ids[0], skip_special_tokens=True).strip())
                    n_words += 1
                lines_text.append(" ".join(t for t in tokens if t))
            text = " ".join(t for t in lines_text if t)
            dt = time.monotonic() - t1
            target.write_text(json.dumps({
                "cell": cell["id"], "run": run, "raw": text, "transcription": text,
                "error": None, "latency_s": round(dt, 2), "n_words": n_words,
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"[{args.config_id} run{run}] {cell['id']}: {dt:.1f}s ({n_words} words)")
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["total_wall_s"] = round(time.monotonic() - t0, 1)
    (outdir / "config.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"done in {meta['total_wall_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
