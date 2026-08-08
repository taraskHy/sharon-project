"""Unlimited-OCR local inference runner — reference-free, resumable.

Loads the pinned local snapshot of baidu/Unlimited-OCR (trust_remote_code,
BF16, eager attention, single GPU, offline mode forced) and transcribes
explicitly listed hebrew_bench_v2 crops. Neither references.json nor
items.json is read: the entire input surface is item id ->
crops/<id>.png plus ONE uniform literal-transcription prompt.

One eval-compatible JSON per item is written to outputs/<config_id>/run1/
(existing files are skipped, so interrupted runs resume) and a full
record including generation settings goes to .../raw_predictions/.
Records are persisted immediately after each item.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "evaluation" / "hebrew_bench_v2"

REVISION = "07dea832e22aefee32ad281d4b80551282e1c168"

# Phase-5 diagnosis (evaluation/unlimited_ocr/diag_prompts.json): the model
# responds ONLY to its README-documented task prompt. Instruction-style
# prompts (including the repo's own commented 'Free OCR.' / 'Extract the
# text...') return empty strings on this revision. The uniform prompt is
# therefore the documented task syntax; literal-transcription discipline is
# enforced by the deterministic marker-stripping parser + post-hoc eval.
PROMPT = "<image>document parsing."

REF_DET = re.compile(r"<\|ref\|>.*?<\|/ref\|>\s*<\|det\|>.*?<\|/det\|>", re.DOTALL)
DET_LABEL = re.compile(r"<\|det\|>\s*[A-Za-z_][\w-]*\s*\[[^\]]*\]\s*<\|/det\|>")
DET_ANY = re.compile(r"<\|det\|>.*?<\|/det\|>", re.DOTALL)
MARKER = re.compile(r"<\|[^|>]{0,60}\|>")
COORD_BLOCK = re.compile(r"\[\[[0-9,\s\[\]]+\]\]")


def parse_transcription(raw: str | None) -> str | None:
    """Deterministic cleanup of model layout markup; frozen before any
    reference was read (observed constructs: evaluation/unlimited_ocr/
    diag_prompts.json). Never touches references."""
    if raw is None:
        return None
    t = REF_DET.sub("", raw)
    t = DET_LABEL.sub("", t)
    t = DET_ANY.sub("", t)
    t = MARKER.sub("", t)
    t = COORD_BLOCK.sub("", t)
    t = re.sub(r"^```[a-z]*\s*|\s*```$", "", t.strip())
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", t)  # markdown image refs
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{2,}", "\n", t)
    return t.strip() or None


def vram_used_mib() -> int | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, check=False,
        ).stdout.strip().splitlines()
        return int(out[0]) if out else None
    except Exception:  # noqa: BLE001
        return None


def category_of(iid: str) -> str:
    return "handwritten_line" if iid.startswith("hl_") else \
        "handwritten_cell" if iid.startswith("hc_") else "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-id", default="unlimited_ocr_gundam_eager")
    ap.add_argument("--items", required=True, help="comma list of item ids")
    ap.add_argument("--snapshot", required=True, help="pinned local snapshot dir")
    ap.add_argument("--base-size", type=int, default=1024)
    ap.add_argument("--image-size", type=int, default=640)
    ap.add_argument("--crop-mode", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=32768)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=35)
    ap.add_argument("--ngram-window", type=int, default=128)
    args = ap.parse_args()

    ids = [i for i in args.items.split(",") if i]
    missing = [i for i in ids if not (BENCH / "crops" / f"{i}.png").exists()]
    if missing:
        sys.exit(f"missing crops: {missing}")

    outdir = BENCH / "outputs" / args.config_id
    rundir = outdir / "run1"
    rawdir = outdir / "raw_predictions"
    rundir.mkdir(parents=True, exist_ok=True)
    rawdir.mkdir(parents=True, exist_ok=True)

    gen_settings = {
        "base_size": args.base_size, "image_size": args.image_size,
        "crop_mode": bool(args.crop_mode), "max_length": args.max_length,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "ngram_window": args.ngram_window, "temperature": 0.0,
        "eval_mode": True, "save_results": False,
        "dtype": "bfloat16", "attn_implementation": "eager",
        "prompt": PROMPT,
    }

    vram_before = vram_used_mib()
    t0 = time.monotonic()
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.snapshot, trust_remote_code=True, use_safetensors=True,
        torch_dtype=torch.bfloat16, attn_implementation="eager",
    )
    model = model.eval().cuda()
    load_s = round(time.monotonic() - t0, 1)
    vram_after_load = vram_used_mib()
    print(f"model loaded in {load_s}s | VRAM {vram_before} -> {vram_after_load} MiB")

    cfg = {
        "config_id": args.config_id, "backend": "unlimited_ocr_local_transformers",
        "model": "baidu/Unlimited-OCR", "revision": REVISION,
        "snapshot": args.snapshot, "preproc": "none",
        **gen_settings,
        "load_s": load_s, "vram_before_mib": vram_before,
        "vram_after_load_mib": vram_after_load,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (outdir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1),
                                        encoding="utf-8")

    scratch = outdir / "infer_scratch"
    scratch.mkdir(exist_ok=True)
    done = 0
    for iid in ids:
        target = rundir / f"{iid}.json"
        if target.exists():
            print(f"skip (exists): {iid}")
            continue
        img = BENCH / "crops" / f"{iid}.png"
        torch.cuda.reset_peak_memory_stats()
        t1 = time.monotonic()
        raw, err = None, None
        try:
            with torch.no_grad():
                res = model.infer(
                    tokenizer, prompt=PROMPT, image_file=str(img),
                    output_path=str(scratch), base_size=args.base_size,
                    image_size=args.image_size, crop_mode=bool(args.crop_mode),
                    save_results=False, eval_mode=True,
                    max_length=args.max_length,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    ngram_window=args.ngram_window,
                )
            raw = res if isinstance(res, str) else (None if res is None else str(res))
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        dt = round(time.monotonic() - t1, 2)
        peak_alloc_mib = int(torch.cuda.max_memory_allocated() / (1024 * 1024))
        rec = {
            "raw": raw,
            "transcription": parse_transcription(raw),
            "error": err,
            "status": 200 if err is None else -1,
            "item": iid,
            "category": category_of(iid),
            "run": 1,
            "latency_s": dt,
        }
        target.write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                          encoding="utf-8")
        full = {**rec, "model": "baidu/Unlimited-OCR", "revision": REVISION,
                "gen_settings": gen_settings,
                "vram_used_mib": vram_used_mib(),
                "torch_peak_alloc_mib": peak_alloc_mib}
        (rawdir / f"{iid}.json").write_text(json.dumps(full, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
        done += 1
        preview = (rec["transcription"] or "")[:60].replace("\n", " ")
        print(f"[{iid}] {dt}s peak={peak_alloc_mib}MiB"
              + (f" ERR {err[:80]}" if err else f" -> {preview!r}"))

    cfg["calls_made"] = done
    cfg["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (outdir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(f"done: {done} new items")
    return 0


if __name__ == "__main__":
    sys.exit(main())
