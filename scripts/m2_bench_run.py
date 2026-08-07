"""Mission-2 benchmark runner — provider-agnostic, resumable.

Runs one backend configuration over evaluation/hebrew_bench_v2 items and
writes one JSON per item under outputs/<config_id>/run<N>/ (existing files
are skipped, so interrupted runs resume). References are NEVER read here.

Adapters (--backend):
- qwen_local   : Ollama/OpenAI-compatible VLM (strict-fidelity prompt,
                 optional contrast preproc) — the Mission-1 local model.
- gemini       : Google Generative Language API (GEMINI_API_KEY env);
                 anonymized crops only; 429-aware retry with backoff.
- openrouter   : OpenRouter chat completions (OPENROUTER_API_KEY env),
                 --model selects the vision model; same quota handling.
- easyocr      : local EasyOCR (Hebrew+English) — printed-text OCR arm.
- tesseract    : local pytesseract (heb+eng) if a tesseract binary is
                 available (--tesseract-cmd).
- deepseek_ocr : reserved; runs via scripts/m2_deepseek_ocr_run.py in its
                 own isolated venv (documented separately).

Per-category task prompts keep the same fidelity rules as the July
campaign (never guess; [?] for unreadable words; struck-through text is
cancelled). OCR engines ignore prompts and return raw text.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "evaluation" / "hebrew_bench_v2"

STRICT_RULES = (
    "Rules:\n"
    "1. Copy EXACTLY the visible text, word by word. Never complete, "
    "paraphrase, correct, or invent words. Fidelity over fluency.\n"
    "2. If a single word is unreadable, write [?] in its place.\n"
    "3. If everything is unreadable, output [unreadable].\n"
    "4. Text struck through by the writer is cancelled - skip it.\n"
    "5. Hebrew is written right-to-left; output the text in normal logical "
    "order (first word first).\n"
    'Reply with ONLY a JSON object: {"transcription": "<the text>"}'
)

PROMPTS = {
    "handwritten_line": (
        "You transcribe ONE LINE of handwritten Hebrew (may mix English "
        "terms) cropped from a university exam answer sheet.\n" + STRICT_RULES
    ),
    "handwritten_cell": (
        "You transcribe the handwritten Hebrew answer text (may mix English "
        "terms) in this exam answer cell. Ignore any red instructor ink.\n"
        + STRICT_RULES
    ),
    "printed_rtl": (
        "You transcribe the PRINTED Hebrew text in this crop from an exam "
        "paper.\n" + STRICT_RULES
    ),
    "mixed_he_en": (
        "You transcribe the PRINTED text in this crop (Hebrew mixed with "
        "English technical terms).\n" + STRICT_RULES
    ),
    "formula_printed": (
        "You transcribe the PRINTED text in this crop (Hebrew with "
        "mathematical notation, numbers and formulas).\n" + STRICT_RULES
    ),
    "option_row_association": (
        "This crop shows the four printed answer options of one multiple-"
        "choice question (Hebrew, RTL: option letters are א ב ג ד). Report "
        "each option letter with its EXACT printed value. Reply with ONLY a "
        'JSON object: {"transcription": "<letter>: <value>; <letter>: '
        '<value>; ..."}'
    ),
}

SCHEMA = {
    "type": "object",
    "properties": {"transcription": {"type": "string"}},
    "required": ["transcription"],
}


def _encode_png_gray(gray) -> bytes:
    import struct
    import zlib

    import numpy as np

    a = np.clip(gray, 0, 255).astype(np.uint8)
    h, w = a.shape
    raw = b"".join(b"\x00" + a[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def preprocess(png: bytes, mode: str, max_edge: int = 0) -> bytes:
    if mode == "none" and not max_edge:
        return png
    import fitz
    import numpy as np

    pix = fitz.Pixmap(png)
    if pix.alpha:
        pix = fitz.Pixmap(pix, 0)
    if max_edge and max(pix.width, pix.height) > max_edge:
        z = max_edge / max(pix.width, pix.height)
        with fitz.open(stream=png, filetype="png") as d:
            pix = d[0].get_pixmap(matrix=fitz.Matrix(z, z))
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = arr.mean(axis=2) if pix.n >= 3 else arr[:, :, 0].astype(float)
    if mode == "contrast":
        lo, hi = np.percentile(gray, 5), np.percentile(gray, 95)
        return _encode_png_gray((gray - lo) / max(hi - lo, 1.0) * 255.0)
    if mode == "none":
        return _encode_png_gray(gray)
    raise ValueError(f"unknown preproc {mode!r}")


# --------------------------------------------------------------------------
# adapters
# --------------------------------------------------------------------------


class ChatVLM:
    """OpenAI-compatible chat endpoint (Ollama local or OpenRouter hosted)."""

    def __init__(self, base_url, model, api_key=None, structured=True,
                 timeout=600.0, max_tokens=500):
        import httpx

        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = httpx.Client(timeout=timeout, headers=headers)
        self.base_url, self.model = base_url, model
        self.structured, self.max_tokens = structured, max_tokens

    def transcribe(self, png: bytes, prompt: str, structured: bool | None = None) -> dict:
        structured = self.structured if structured is None else structured
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {
                        "url": "data:image/png;base64,"
                        + base64.standard_b64encode(png).decode()}},
                    {"type": "text", "text": "Transcribe now."},
                ]},
            ],
        }
        if structured:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "T", "schema": SCHEMA},
            }
        for attempt in range(5):
            resp = self.client.post(f"{self.base_url}/chat/completions", json=payload)
            if resp.status_code == 429:
                wait = min(60.0 * (attempt + 1), 240.0)
                retry_after = resp.headers.get("retry-after")
                if retry_after and retry_after.replace(".", "").isdigit():
                    wait = min(float(retry_after) + 1.0, 300.0)
                print(f"  429 rate-limited; sleeping {wait:.0f}s")
                time.sleep(wait)
                continue
            data = resp.json()
            raw = ""
            try:
                raw = data["choices"][0]["message"].get("content") or ""
                try:
                    parsed = json.loads(raw).get("transcription")
                except json.JSONDecodeError:
                    parsed = raw.strip() or None  # non-structured providers
                return {"raw": raw, "transcription": parsed, "error": None,
                        "status": resp.status_code}
            except Exception as e:  # noqa: BLE001
                return {"raw": raw, "transcription": None,
                        "error": f"{type(e).__name__}: {e} | body={str(data)[:300]}",
                        "status": resp.status_code}
        return {"raw": "", "transcription": None,
                "error": "rate-limited after 5 attempts", "status": 429}


class Gemini:
    """generativelanguage.googleapis.com generateContent (REST, no SDK)."""

    def __init__(self, model, api_key, timeout=180.0):
        import httpx

        self.client = httpx.Client(timeout=timeout)
        self.model, self.key = model, api_key

    def transcribe(self, png: bytes, prompt: str) -> dict:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.key}"
        )
        body = {
            "contents": [{"parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": "image/png",
                                 "data": base64.standard_b64encode(png).decode()}},
            ]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 800},
        }
        rl_waits, rl_wait_s = 0, 0.0
        for attempt in range(6):
            resp = self.client.post(url, json=body)
            if resp.status_code in (429, 503):
                wait = min(30.0 * (attempt + 1), 180.0)
                print(f"  {resp.status_code} from Gemini; sleeping {wait:.0f}s")
                rl_waits += 1
                rl_wait_s += wait
                time.sleep(wait)
                continue
            data = resp.json()
            try:
                raw = data["candidates"][0]["content"]["parts"][0]["text"]
                parsed = None
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                if m:
                    try:
                        parsed = json.loads(m.group(0)).get("transcription")
                    except json.JSONDecodeError:
                        parsed = None
                if parsed is None:
                    cleaned = re.sub(r"^```[a-z]*\s*|\s*```$", "", raw.strip())
                    parsed = cleaned.strip() or None
                return {"raw": raw, "transcription": parsed, "error": None,
                        "status": resp.status_code,
                        "rate_limit_waits": rl_waits,
                        "rate_limit_wait_s": round(rl_wait_s, 1)}
            except Exception as e:  # noqa: BLE001
                return {"raw": "", "transcription": None,
                        "error": f"{type(e).__name__}: {e} | body={str(data)[:300]}",
                        "status": resp.status_code,
                        "rate_limit_waits": rl_waits,
                        "rate_limit_wait_s": round(rl_wait_s, 1)}
        return {"raw": "", "transcription": None,
                "error": "rate-limited after 6 attempts", "status": 429,
                "rate_limited_out": True,
                "rate_limit_waits": rl_waits, "rate_limit_wait_s": round(rl_wait_s, 1)}


class EasyOCRAdapter:
    def __init__(self):
        import easyocr  # heavy import

        self.reader = easyocr.Reader(["he", "en"], gpu=False, verbose=False)

    def transcribe(self, png: bytes, prompt: str) -> dict:
        import numpy as np
        import fitz

        pix = fitz.Pixmap(png)
        if pix.alpha:
            pix = fitz.Pixmap(pix, 0)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, pix.n
        )
        result = self.reader.readtext(arr[:, :, :3] if pix.n >= 3 else arr[:, :, 0])
        # sort boxes top-to-bottom then RIGHT-to-left (RTL reading order)
        def key(entry):
            box = entry[0]
            ys = sum(p[1] for p in box) / 4
            xs = sum(p[0] for p in box) / 4
            return (round(ys / 20), -xs)
        result = sorted(result, key=key)
        text = " ".join(entry[1] for entry in result).strip()
        return {"raw": json.dumps([[e[1], round(float(e[2]), 3)] for e in result],
                                  ensure_ascii=False),
                "transcription": text or None, "error": None, "status": 200}


class TesseractAdapter:
    def __init__(self, cmd):
        import pytesseract

        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        self.pt = pytesseract

    def transcribe(self, png: bytes, prompt: str) -> dict:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(png))
        text = self.pt.image_to_string(img, lang="heb+eng")
        return {"raw": text, "transcription": text.strip() or None,
                "error": None, "status": 200}


def make_adapter(args):
    if args.backend == "qwen_local":
        return ChatVLM(args.base_url, args.model, structured=True)
    if args.backend == "openrouter":
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            sys.exit("OPENROUTER_API_KEY not set")
        return ChatVLM("https://openrouter.ai/api/v1", args.model, api_key=key,
                       structured=False, timeout=300.0)
    if args.backend == "gemini":
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            sys.exit("GEMINI_API_KEY not set")
        return Gemini(args.model, key)
    if args.backend == "easyocr":
        return EasyOCRAdapter()
    if args.backend == "tesseract":
        return TesseractAdapter(args.tesseract_cmd)
    sys.exit(f"unknown backend {args.backend}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-id", required=True)
    ap.add_argument("--backend", required=True)
    ap.add_argument("--model", default="qwen3-vl:8b-instruct")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--preproc", default="none")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--categories", default="", help="comma list to restrict")
    ap.add_argument("--items", default="", help="comma list to restrict (debug)")
    ap.add_argument("--tesseract-cmd", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-edge", type=int, default=0,
                    help="downscale crops so the long edge <= this before sending")
    ap.add_argument("--min-interval", type=float, default=0.0,
                    help="minimum seconds between request STARTS (proactive "
                         "RPM pacing for quota-limited hosted APIs)")
    args = ap.parse_args()

    manifest = json.loads((BENCH / "items.json").read_text(encoding="utf-8"))["items"]
    if args.categories:
        keep = set(args.categories.split(","))
        manifest = [m for m in manifest if m["category"] in keep]
    if args.items:
        keep = set(args.items.split(","))
        manifest = [m for m in manifest if m["id"] in keep]
    if args.limit:
        manifest = manifest[: args.limit]

    outdir = BENCH / "outputs" / args.config_id
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config.json").write_text(json.dumps({
        "config_id": args.config_id, "backend": args.backend, "model": args.model,
        "preproc": args.preproc, "max_edge": args.max_edge,
        "min_interval_s": args.min_interval, "runs": args.runs,
        "categories": args.categories or "all", "n_items": len(manifest),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=1), encoding="utf-8")

    adapter = make_adapter(args)
    t0 = time.monotonic()
    done = 0
    last_start = 0.0
    for run in range(1, args.runs + 1):
        rundir = outdir / f"run{run}"
        rundir.mkdir(exist_ok=True)
        for item in manifest:
            target = rundir / f"{item['id']}.json"
            if target.exists():
                continue
            png = preprocess((BENCH / item["image"]).read_bytes(), args.preproc,
                             max_edge=args.max_edge)
            prompt = PROMPTS[item["category"]]
            if args.min_interval:
                pace = args.min_interval - (time.monotonic() - last_start)
                if pace > 0:
                    time.sleep(pace)
            last_start = time.monotonic()
            t1 = time.monotonic()
            try:
                # association answers die inside the constrained JSON grammar
                # on the local model — use free text there and parse post-hoc
                if item["category"] == "option_row_association" and isinstance(adapter, ChatVLM):
                    res = adapter.transcribe(png, prompt, structured=False)
                else:
                    res = adapter.transcribe(png, prompt)
            except Exception as e:  # noqa: BLE001
                res = {"raw": "", "transcription": None,
                       "error": f"{type(e).__name__}: {e}", "status": -1}
            dt = time.monotonic() - t1
            res.update({"item": item["id"], "category": item["category"],
                        "run": run, "latency_s": round(dt, 2)})
            target.write_text(json.dumps(res, ensure_ascii=False, indent=1),
                              encoding="utf-8")
            done += 1
            print(f"[{args.config_id} run{run}] {item['id']}: {dt:.1f}s"
                  + (f" ERR {res['error'][:60]}" if res.get("error") else ""))
    cfg = json.loads((outdir / "config.json").read_text(encoding="utf-8"))
    cfg["total_wall_s"] = round(time.monotonic() - t0, 1)
    cfg["calls_made"] = done
    (outdir / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    print(f"done: {done} calls in {cfg['total_wall_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
