"""Run one transcription configuration over the Hebrew handwriting bench.

NEVER reads ground_truth.json. Saves raw responses + parsed transcriptions
per cell per repeat under evaluation/hebrew_bench/outputs/<config_id>/.

    python scripts/hebrew_bench_run.py --config-id it1_baseline_8b \
        --model qwen3-vl:8b-instruct --prompt baseline --runs 3
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BENCH = Path("evaluation/hebrew_bench")

PROMPTS = {
    "baseline": (
        "You transcribe handwritten Hebrew academic text from an exam answer "
        "cell. Transcribe the handwriting in this image exactly as written. "
        'Reply with ONLY a JSON object: {"transcription": "<the text>"}'
    ),
    "strict_fidelity": (
        "You transcribe handwritten Hebrew (RTL, may mix English terms) from "
        "an exam answer cell. Rules:\n"
        "1. Copy EXACTLY the visible handwriting, word by word. Never "
        "complete, paraphrase, correct, or invent words. Fluency is NOT the "
        "goal; fidelity is.\n"
        "2. If a single word is unreadable, write [?] in its place.\n"
        "3. If the whole cell is unreadable or fully crossed out, output "
        "[unreadable].\n"
        "4. Text struck through by the writer is cancelled — skip it.\n"
        "5. It is ALWAYS better to output [?] than to guess a word.\n"
        'Reply with ONLY a JSON object: {"transcription": "<the text>"}'
    ),
}

SCHEMA = {
    "type": "object",
    "properties": {"transcription": {"type": "string"}},
    "required": ["transcription"],
}


def _encode_png_gray(gray) -> bytes:
    """Minimal 8-bit grayscale PNG encoder (numpy + zlib; no new deps)."""
    import struct
    import zlib

    import numpy as np

    h, w = gray.shape
    raw = b"".join(b"\x00" + gray[y].astype(np.uint8).tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)  # 8-bit grayscale
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def preprocess(png: bytes, mode: str) -> bytes:
    if mode == "none":
        return png
    import fitz
    import numpy as np

    pix = fitz.Pixmap(png)
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = arr.mean(axis=2)
    if mode == "contrast":
        # Percentile stretch: ink toward black, paper toward white.
        lo, hi = np.percentile(gray, 5), np.percentile(gray, 95)
        stretched = np.clip((gray - lo) / max(hi - lo, 1.0) * 255.0, 0, 255)
        return _encode_png_gray(stretched)
    raise ValueError(f"unknown preproc mode {mode!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-id", required=True)
    ap.add_argument("--model", default="qwen3-vl:8b-instruct")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--prompt", choices=sorted(PROMPTS), default="baseline")
    ap.add_argument("--preproc", default="none")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--cells", default="", help="comma list to restrict (debug)")
    ap.add_argument("--manifest", default=str(BENCH / "crops_manifest.json"),
                    help="crop manifest (id+file entries); alternate manifests "
                         "point at preprocessed crops, e.g. student-ink isolation")
    args = ap.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.cells:
        keep = set(args.cells.split(","))
        manifest = [m for m in manifest if m["id"] in keep]
    outdir = BENCH / "outputs" / args.config_id
    meta = {
        "config_id": args.config_id, "model": args.model, "prompt": args.prompt,
        "prompt_text": PROMPTS[args.prompt], "preproc": args.preproc,
        "max_tokens": args.max_tokens, "runs": args.runs,
        "manifest": args.manifest,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")

    t0 = time.monotonic()
    n_calls = 0
    with httpx.Client(timeout=600.0) as client:
        for run in range(1, args.runs + 1):
            rundir = outdir / f"run{run}"
            rundir.mkdir(exist_ok=True)
            for cell in manifest:
                target = rundir / f"{cell['id']}.json"
                if target.exists():
                    continue
                png = preprocess(Path(cell["file"]).read_bytes(), args.preproc)
                payload = {
                    "model": args.model,
                    "temperature": 0,
                    "max_tokens": args.max_tokens,
                    "messages": [
                        {"role": "system", "content": PROMPTS[args.prompt]},
                        {"role": "user", "content": [
                            {"type": "image_url", "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.standard_b64encode(png).decode()}},
                            {"type": "text", "text": "Transcribe now."},
                        ]},
                    ],
                    "response_format": {"type": "json_schema", "json_schema": {
                        "name": "T", "schema": SCHEMA}},
                }
                t1 = time.monotonic()
                resp = client.post(f"{args.base_url}/chat/completions", json=payload)
                dt = time.monotonic() - t1
                n_calls += 1
                data = resp.json()
                raw = ""
                err = None
                try:
                    choice = data["choices"][0]
                    raw = choice["message"].get("content") or ""
                    parsed = json.loads(raw).get("transcription")
                except Exception as e:  # noqa: BLE001
                    parsed, err = None, f"{type(e).__name__}: {e} | body={str(data)[:200]}"
                target.write_text(json.dumps({
                    "cell": cell["id"], "run": run, "raw": raw,
                    "transcription": parsed, "error": err, "latency_s": round(dt, 2),
                }, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"[{args.config_id} run{run}] {cell['id']}: {dt:.1f}s")
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["total_wall_s"] = round(time.monotonic() - t0, 1)
    meta["calls"] = n_calls
    (outdir / "config.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"done: {n_calls} calls in {meta['total_wall_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
