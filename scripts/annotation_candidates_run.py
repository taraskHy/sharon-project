"""Generate UNVERIFIED candidate transcriptions for annotation assistance.

GT-FREE BY CONSTRUCTION: this module never opens anything under
<root>/annotations/ — it reads only split metadata (paths/geometry),
an ids file, and image bytes. Prompts are fixed strings; no answer key,
rubric, course vocabulary, or previous label can enter them. Raw model
responses are written to disk before any evaluation script runs.
Candidates are CANDIDATE TEXT ONLY — nothing here writes annotation
records or marks anything verified.

    # 1. record the benchmark selection (ids only, no text)
    .venv/Scripts/python.exe scripts/annotation_candidates_run.py select \
        --exclude-ids evaluation/htr_overfit_test/selected_ids.json

    # 2. VLM candidates (Ollama must be serving)
    ... generate --config qwen_line
    ... generate --config qwen_line_cell

    # 3. CRNN decode workspace (decode itself runs via htr_pilot_train.py)
    ... crnn-prep
    ... crnn-collect
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path("evaluation/htr_pilot")
OUT = Path("evaluation/htr_candidates")

# The campaign's strict-fidelity prompt (verbatim from
# scripts/hebrew_bench_run.py — the best-performing prompt of the closed
# transcription campaign). Fixed text; nothing sample-specific.
STRICT_PROMPT = (
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
)

SCHEMA = {
    "type": "object",
    "properties": {"transcription": {"type": "string"}},
    "required": ["transcription"],
}

CONFIGS = ("qwen_line", "qwen_line_cell")


def _b64_image(png: bytes) -> dict:
    return {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.standard_b64encode(png).decode()}}


def build_messages(config: str, line_png: bytes, cell_png: bytes | None) -> list:
    """Chat messages for one sample. Pure function of fixed prompt text and
    image bytes — provably no ground truth can enter the prompt."""
    if config == "qwen_line":
        user = [_b64_image(line_png), {"type": "text", "text": "Transcribe now."}]
    elif config == "qwen_line_cell":
        assert cell_png is not None
        user = [
            _b64_image(cell_png),
            _b64_image(line_png),
            {"type": "text", "text":
                "The first image is the full answer cell (context only). "
                "The second image is a single line cropped from that cell. "
                "Transcribe ONLY the line shown in the second image."},
        ]
    else:
        raise ValueError(f"unknown config {config!r}")
    return [{"role": "system", "content": STRICT_PROMPT},
            {"role": "user", "content": user}]


def load_split_index(root: Path) -> dict[str, dict]:
    samples = json.loads((root / "splits" / "train.json").read_text(encoding="utf-8"))
    return {s["sample_id"]: s for s in samples}


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def cmd_select(args) -> int:
    """Record which train ids form the benchmark. Reads annotation files for
    their STATUS ONLY (which lines are owner-verified `ok`); transcription
    text is never copied anywhere. Output contains ids only."""
    excl = set()
    if args.exclude_ids:
        excl = set(json.loads(Path(args.exclude_ids).read_text(encoding="utf-8"))["picked"])
    ann_dir = ROOT / "annotations" / "train"
    ids = []
    for f in sorted(ann_dir.glob("*.json")):
        rec = json.loads(f.read_text(encoding="utf-8"))
        if rec.get("status") == "ok" and rec["sample_id"] not in excl:
            ids.append(rec["sample_id"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bench_ids.json").write_text(json.dumps({
        "rule": "train annotations with status==ok, minus overfit-test ids",
        "excluded_overfit_ids": sorted(excl),
        "n": len(ids), "ids": ids,
        "at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=1), encoding="utf-8")
    print(f"{len(ids)} bench ids -> {OUT / 'bench_ids.json'}")
    return 0


def _ollama_digest(client, base_url: str, model: str) -> str:
    try:
        tags = client.get(base_url.replace("/v1", "") + "/api/tags").json()
        for m in tags.get("models", []):
            if m["name"] == model:
                return m["digest"]
    except Exception:  # noqa: BLE001
        pass
    return "unknown"


def cmd_generate(args) -> int:
    import httpx

    ids = json.loads((OUT / (args.ids_file or "bench_ids.json"))
                     .read_text(encoding="utf-8"))["ids"]
    index = load_split_index(ROOT)
    outdir = OUT / "outputs" / args.config
    outdir.mkdir(parents=True, exist_ok=True)
    meta = {
        "config": args.config, "model": args.model, "prompt_id": "strict_fidelity",
        "prompt_text": STRICT_PROMPT, "temperature": 0,
        "max_tokens": args.max_tokens, "ids_file": args.ids_file or "bench_ids.json",
        "n_ids": len(ids), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    t0 = time.monotonic()
    done = 0
    with httpx.Client(timeout=600.0) as client:
        meta["model_digest"] = _ollama_digest(client, args.base_url, args.model)
        (outdir / "config.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        for sid in ids:
            target = outdir / f"{sid}.json"
            if target.exists():
                continue
            s = index[sid]
            line_png = (ROOT / s["images"]["line"]).read_bytes()
            cell_png = None
            images_meta = {"line": {"path": s["images"]["line"],
                                    "sha256": sha256(line_png)}}
            if args.config == "qwen_line_cell":
                cell_png = (ROOT / s["images"]["cell_clean"]).read_bytes()
                images_meta["cell_clean"] = {"path": s["images"]["cell_clean"],
                                             "sha256": sha256(cell_png)}
            payload = {
                "model": args.model, "temperature": 0,
                "max_tokens": args.max_tokens,
                "messages": build_messages(args.config, line_png, cell_png),
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": "T", "schema": SCHEMA}},
            }
            t1 = time.monotonic()
            raw, parsed, err = "", None, None
            try:
                resp = client.post(f"{args.base_url}/chat/completions", json=payload)
                data = resp.json()
                raw = data["choices"][0]["message"].get("content") or ""
                parsed = json.loads(raw).get("transcription")
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
            dt = time.monotonic() - t1
            target.write_text(json.dumps({
                "sample_id": sid, "config": args.config, "model": args.model,
                "model_digest": meta["model_digest"], "prompt_id": "strict_fidelity",
                "images": images_meta, "raw": raw, "candidate": parsed,
                "confidence": None, "error": err, "latency_s": round(dt, 2),
                "verified": False,
                "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, ensure_ascii=False, indent=1), encoding="utf-8")
            done += 1
            print(f"[{args.config}] {sid}: {dt:.1f}s", flush=True)
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["total_wall_s"] = round(time.monotonic() - t0, 1)
    meta["calls"] = done
    (outdir / "config.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"done: {done} calls in {meta['total_wall_s']}s")
    return 0


def cmd_crnn_prep(args) -> int:
    """Build a decode-only workspace for the overfit CRNN checkpoint."""
    import shutil

    from scripts.htr_train_prepare import load_line_gray
    import cv2

    src_ws = Path("evaluation/htr_overfit_test/ws")
    ws = OUT / "crnn_ws"
    (ws / "model").mkdir(parents=True, exist_ok=True)
    (ws / "lists").mkdir(exist_ok=True)
    (ws / "imgs" / "bench").mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_ws / "syms.txt", ws / "syms.txt")
    shutil.copy2(src_ws / "model" / "crnn_best.pt", ws / "model" / "crnn_best.pt")
    ids = json.loads((OUT / "bench_ids.json").read_text(encoding="utf-8"))["ids"]
    index = load_split_index(ROOT)
    for sid in ids:
        img = load_line_gray(ROOT / index[sid]["images"]["line"])
        cv2.imwrite(str(ws / "imgs" / "bench" / f"{sid}.png"), img)
    (ws / "lists" / "bench.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    ckpt = (ws / "model" / "crnn_best.pt").read_bytes()
    (ws / "checkpoint_meta.json").write_text(json.dumps({
        "source": str(src_ws / "model" / "crnn_best.pt"),
        "sha256": sha256(ckpt),
        "trained_on": "the 20 overfit-test ids only (selected_ids.json)",
    }, indent=1), encoding="utf-8")
    print(f"{len(ids)} bench images -> {ws}")
    return 0


def cmd_crnn_collect(args) -> int:
    """Convert the decode file into per-sample candidate records."""
    ws = OUT / "crnn_ws"
    dec = ws / "decodes" / "bench.txt"
    ckpt_meta = json.loads((ws / "checkpoint_meta.json").read_text(encoding="utf-8"))
    index = load_split_index(ROOT)
    outdir = OUT / "outputs" / "crnn_overfit"
    outdir.mkdir(parents=True, exist_ok=True)
    n = 0
    for line in dec.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        sid, conf, text = (line.split(" ", 2) + ["", ""])[:3]
        img = (ROOT / index[sid]["images"]["line"]).read_bytes()
        (outdir / f"{sid}.json").write_text(json.dumps({
            "sample_id": sid, "config": "crnn_overfit",
            "model": "in-repo CRNN+CTC (overfit-test checkpoint)",
            "model_digest": ckpt_meta["sha256"],
            "images": {"line": {"path": index[sid]["images"]["line"],
                                "sha256": sha256(img)}},
            "raw": line, "candidate": text.strip() or None,
            "confidence": float(conf), "error": None, "latency_s": None,
            "verified": False,
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        n += 1
    print(f"{n} CRNN candidates -> {outdir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    se = sub.add_parser("select")
    se.add_argument("--exclude-ids", default="")
    ge = sub.add_parser("generate")
    ge.add_argument("--config", choices=CONFIGS, required=True)
    ge.add_argument("--model", default="qwen3-vl:8b-instruct")
    ge.add_argument("--base-url", default="http://localhost:11434/v1")
    ge.add_argument("--max-tokens", type=int, default=400)
    ge.add_argument("--ids-file", default="")
    sub.add_parser("crnn-prep")
    sub.add_parser("crnn-collect")
    args = ap.parse_args()
    return {"select": cmd_select, "generate": cmd_generate,
            "crnn-prep": cmd_crnn_prep, "crnn-collect": cmd_crnn_collect}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
