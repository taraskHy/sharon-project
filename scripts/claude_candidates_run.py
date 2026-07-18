"""Claude Vision candidate transcriptions — generation side.

Owner-authorized experiment (2026-07-17): sends ANONYMIZED, CLEANED
handwriting crops to the Anthropic API. Protocol + privacy pre-flight:
evaluation/claude_candidates/PROTOCOL.md.

GT-FREE BY CONSTRUCTION: this module never opens anything under
<root>/annotations/ — it reads split metadata (image paths only), the
ids file, and image bytes. Payloads contain base64 image bytes plus the
fixed instruction text only: no sample ids, no file names, no labels,
no question text, no keys/rubrics/vocabulary, no other student's work.
Config B context uses *_cell_clean.png (instructor ink removed) — never
cell_orig. Claude output is CANDIDATE TEXT ONLY, saved with
verified=false; nothing here writes annotation records.

    # 1. record the selection (ids only)
    .venv/Scripts/python.exe scripts/claude_candidates_run.py select

    # 2. generate (needs ANTHROPIC_API_KEY or an `ant auth login` profile)
    .venv/Scripts/python.exe scripts/claude_candidates_run.py generate --config claude_line
    .venv/Scripts/python.exe scripts/claude_candidates_run.py generate --config claude_line_cell
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path("evaluation/htr_pilot")
OUT = Path("evaluation/claude_candidates")

MODEL = "claude-opus-4-8"
PROMPT_VERSION = "claude_htr_v1"
MAX_TOKENS = 16000

# The owner's transcription-only instruction, verbatim (PROTOCOL.md).
INSTRUCTION = (
    "Transcribe exactly the handwritten text visible in the image.\n"
    "Preserve spelling mistakes, punctuation, English words, numbers, "
    "and formulas.\n"
    "Do not explain, correct, complete, or infer likely words.\n"
    "Use [לא קריא] for any unreadable span.\n"
    "Return only the transcription."
)

B_CONTEXT_TEXT = (
    "The first image is the student's full answer cell (context only). "
    "The second image is a single line cropped from that cell. "
    "Transcribe ONLY the line shown in the second image."
)

CONFIGS = ("claude_line", "claude_line_cell")
WRITERS = ("e004", "e005", "e006")
PER_WRITER = 10


def _image_block(png: bytes) -> dict:
    import base64

    return {"type": "image", "source": {
        "type": "base64", "media_type": "image/png",
        "data": base64.standard_b64encode(png).decode()}}


def build_request(config: str, line_png: bytes, cell_png: bytes | None) -> dict:
    """Full request kwargs for one sample. Pure function of fixed text and
    image bytes — nothing sample-specific beyond the pixels can enter."""
    if config == "claude_line":
        content = [_image_block(line_png)]
    elif config == "claude_line_cell":
        assert cell_png is not None
        content = [_image_block(cell_png), _image_block(line_png),
                   {"type": "text", "text": B_CONTEXT_TEXT}]
    else:
        raise ValueError(f"unknown config {config!r}")
    return {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": INSTRUCTION,
        "thinking": {"type": "adaptive"},
        "messages": [{"role": "user", "content": content}],
    }


def load_split_index(root: Path) -> dict[str, dict]:
    samples = json.loads((root / "splits" / "train.json").read_text(encoding="utf-8"))
    return {s["sample_id"]: s for s in samples}


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def cmd_select(args) -> int:
    """Deterministic 30-line selection (PROTOCOL.md). Reads annotation
    files for STATUS/span flags only — no transcription text is copied;
    output contains ids only."""
    overfit = set(json.loads(
        Path("evaluation/htr_overfit_test/selected_ids.json")
        .read_text(encoding="utf-8"))["picked"])
    ann_dir = ROOT / "annotations" / "train"
    per_writer: dict[str, list[str]] = {w: [] for w in WRITERS}
    for f in sorted(ann_dir.glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        w = rec.get("writer")
        if (w in per_writer and rec.get("status") == "ok"
                and rec.get("human_verified")
                and "[לא קריא]" not in rec.get("transcription", "")
                and rec["sample_id"] not in overfit
                and len(per_writer[w]) < PER_WRITER):
            per_writer[w].append(rec["sample_id"])
    ids = [sid for w in WRITERS for sid in per_writer[w]]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "claude_bench_ids.json").write_text(json.dumps({
        "rule": (f"first {PER_WRITER} clean-ok non-overfit lines per writer "
                 f"{WRITERS} in sorted sample_id order"),
        "n": len(ids), "per_writer": {w: len(v) for w, v in per_writer.items()},
        "ids": ids, "at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=1), encoding="utf-8")
    print(f"{len(ids)} bench ids -> {OUT / 'claude_bench_ids.json'}")
    return 0 if len(ids) == PER_WRITER * len(WRITERS) else 3


def cmd_generate(args) -> int:
    import anthropic

    client = anthropic.Anthropic()  # env key or `ant auth login` profile
    ids = json.loads((OUT / "claude_bench_ids.json")
                     .read_text(encoding="utf-8"))["ids"]
    index = load_split_index(ROOT)
    outdir = OUT / "outputs" / args.config
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": args.config, "model": MODEL,
        "prompt_version": PROMPT_VERSION, "instruction": INSTRUCTION,
        "max_tokens": MAX_TOKENS, "thinking": "adaptive",
        "n_ids": len(ids), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (outdir / "config.json").write_text(json.dumps(meta, ensure_ascii=False,
                                                   indent=1), encoding="utf-8")
    t0 = time.monotonic()
    done = 0
    for sid in ids:
        target = outdir / f"{sid}.json"
        if target.exists():
            continue
        s = index[sid]
        line_png = (ROOT / s["images"]["line"]).read_bytes()
        cell_png = None
        images_meta = {"line": {"path": s["images"]["line"],
                                "sha256": sha256(line_png)}}
        if args.config == "claude_line_cell":
            cell_png = (ROOT / s["images"]["cell_clean"]).read_bytes()
            images_meta["cell_clean"] = {"path": s["images"]["cell_clean"],
                                         "sha256": sha256(cell_png)}
        req = build_request(args.config, line_png, cell_png)
        t1 = time.monotonic()
        raw_content, candidate, err, stop, usage, resp_model = (
            None, None, None, None, None, None)
        try:
            resp = client.messages.create(**req)
            stop = resp.stop_reason
            resp_model = resp.model
            usage = {"input_tokens": resp.usage.input_tokens,
                     "output_tokens": resp.usage.output_tokens}
            raw_content = [b.to_dict() for b in resp.content]
            if stop == "refusal":
                err = "refusal"
            else:
                texts = [b.text for b in resp.content if b.type == "text"]
                candidate = "\n".join(texts).strip() or None
                if stop == "max_tokens":
                    err = "max_tokens (truncated)"
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        dt = time.monotonic() - t1
        target.write_text(json.dumps({
            "sample_id": sid, "config": args.config,
            "input_type": ("line_crop" if args.config == "claude_line"
                           else "line_crop+cell_context"),
            "model_requested": MODEL, "model": resp_model,
            "prompt_version": PROMPT_VERSION,
            "images": images_meta, "raw_content": raw_content,
            "stop_reason": stop, "usage": usage,
            "candidate": candidate, "confidence": None,
            "error": err, "latency_s": round(dt, 2),
            "verified": False,
            "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        done += 1
        print(f"[{args.config}] {sid}: {dt:.1f}s"
              + (f" ERROR {err}" if err else ""), flush=True)
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    meta["total_wall_s"] = round(time.monotonic() - t0, 1)
    meta["calls"] = done
    (outdir / "config.json").write_text(json.dumps(meta, ensure_ascii=False,
                                                   indent=1), encoding="utf-8")
    print(f"done: {done} calls in {meta['total_wall_s']}s")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("select")
    ge = sub.add_parser("generate")
    ge.add_argument("--config", choices=CONFIGS, required=True)
    args = ap.parse_args()
    return {"select": cmd_select, "generate": cmd_generate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
