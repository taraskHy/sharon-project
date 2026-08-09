"""Signal 2: image-grounded STRICT TRANSCRIPTION-FIDELITY verification.

Protocol (frozen before any output existed):
- Verifier inputs are ONLY (1) the original crop image and (2) the frozen
  Gemini transcription of that item. NO reference text, NO answer key, NO
  rubric, NO printed question context, NO expected terminology — semantic
  context is deliberately withheld (the pc->DC failure shows context can
  encourage normalization).
- The verifier judges fidelity to the pixels, never semantic correctness.
- Per-ITEM verification; a cell's verdict aggregates as: "review" if any
  constituent item says review; confidence = min over items
  (high > medium > low).
- Fixed report thresholds, declared a priori:
    T1: review iff verdict == "review"
    T2: T1 or confidence == "low"
    T3: T1 or confidence in {low, medium}
    T4: any non-empty omissions/substitutions/additions list
- Original OCR results are never modified; verifier outputs persist to
  outputs/gemini3_flash_verify/run1/<item>.json (resumable; daily-quota
  fast-stop identical to the benchmark runner).

Verifier model: gemini-3-flash-preview (same model family as the reads,
documented here); temperature 0; maxOutputTokens 800.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "evaluation" / "hebrew_bench_v2"
OUTDIR = BENCH / "outputs" / "gemini3_flash_verify" / "run1"
MODEL = "gemini-3-flash-preview"

VERIFIER_PROMPT = (
    "You are a strict transcription-fidelity checker for handwritten Hebrew "
    "exam text (may mix English technical tokens). You receive an image of "
    "ONE handwritten line/cell and a proposed transcription of it.\n"
    "Judge ONLY whether the proposed transcription faithfully matches the "
    "visible handwriting. Rules:\n"
    "- Do NOT solve any question. Do NOT improve the student's wording. Do "
    "NOT correct terminology or spelling. Do NOT infer what the student "
    "probably intended. Student mistakes must be preserved verbatim.\n"
    "- Compare the proposed transcription against the PIXELS.\n"
    "- Report omitted visible text (missing words/clauses, especially at "
    "line ends), added text that is not visible, and substitutions.\n"
    "- Pay special attention to short technical tokens, Latin letters, "
    "numbers, operators, negations, and clause endings.\n"
    'Reply with ONLY this JSON: {"verdict": "supported" or "review", '
    '"omissions": [..], "substitutions": [..], "additions": [..], '
    '"uncertain_regions": [..], "confidence": "high"|"medium"|"low"}'
)


def eligible_items() -> list[tuple[str, str]]:
    """(item_id, frozen transcription) for every item composing an eligible
    Gemini grading cell. Derived from the frozen grading ledger."""
    cells = [json.loads(l)["cell"] for l in
             (REPO / "evaluation" / "m2_grading" / "gemini3_flash.jsonl")
             .read_text(encoding="utf-8").splitlines()]
    items = json.loads((BENCH / "items.json").read_text(encoding="utf-8"))["items"]
    cellmap = defaultdict(list)
    for it in items:
        iid = it["id"]
        if iid.startswith("hl_"):
            cellmap[iid.split("__")[0].replace("hl_", "")].append(iid)
        elif iid.startswith("hc_"):
            cellmap[iid.replace("hc_", "")].append(iid)
    out = []
    for cell in cells:
        for iid in sorted(cellmap[cell]):
            rec = json.loads((BENCH / "outputs" / "gemini3_flash" / "run1" / f"{iid}.json")
                             .read_text(encoding="utf-8"))
            if not rec.get("error") and (rec.get("transcription") or "").strip():
                out.append((iid, rec["transcription"]))
    return out


def main() -> int:
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY not set")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR.parent / "config.json").write_text(json.dumps({
        "config_id": "gemini3_flash_verify", "model": MODEL,
        "protocol": "strict fidelity; inputs = crop + frozen transcription ONLY",
        "thresholds_declared_a_priori": ["T1 verdict", "T2 +low-conf",
                                        "T3 +medium-conf", "T4 any-issue-list"],
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=1), encoding="utf-8")

    todo = [(i, t) for i, t in eligible_items() if not (OUTDIR / f"{i}.json").exists()]
    print(f"verifier items todo: {len(todo)}")
    client = httpx.Client(timeout=180.0)
    last = 0.0
    for iid, transcription in todo:
        wait = 15.0 - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)
        last = time.monotonic()
        png = (BENCH / "crops" / f"{iid}.png").read_bytes()
        body = {
            "contents": [{"parts": [
                {"text": VERIFIER_PROMPT},
                {"inline_data": {"mime_type": "image/png",
                                 "data": base64.standard_b64encode(png).decode()}},
                {"text": "Proposed transcription:\n" + transcription
                         + "\nCheck fidelity now."},
            ]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": 800},
        }
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{MODEL}:generateContent?key={key}")
        t0 = time.monotonic()
        resp = client.post(url, json=body)
        if resp.status_code in (429, 503):
            if "PerDay" in resp.text or "per day" in resp.text.lower():
                print(f"DAILY QUOTA EXHAUSTED at {iid} — stopping (resumable)")
                return 3
            print("  429; sleeping 60s")
            time.sleep(60)
            resp = client.post(url, json=body)
        data = resp.json()
        rec = {"item": iid, "transcription_checked": transcription,
               "latency_s": round(time.monotonic() - t0, 2), "error": None}
        try:
            raw = data["candidates"][0]["content"]["parts"][0]["text"]
            rec["raw"] = raw
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            rec["verdict_json"] = json.loads(m.group(0)) if m else None
            if rec["verdict_json"] is None:
                rec["error"] = "no JSON object in verifier output"
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {e} | body={str(data)[:200]}"
        (OUTDIR / f"{iid}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1),
                                            encoding="utf-8")
        v = (rec.get("verdict_json") or {}).get("verdict", "?")
        print(f"  {iid}: {v} ({rec['latency_s']}s)"
              + (f" ERR {rec['error'][:50]}" if rec["error"] else ""))
    print("verifier run complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
