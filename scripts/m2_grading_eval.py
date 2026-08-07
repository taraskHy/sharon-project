"""Downstream grading-decision metric for Mission 2.

Question answered per OCR backend: is the RECOGNIZED text sufficient to
reach the same grading decision as the owner-verified transcription —
not merely how many characters match.

Design (documented limitation included):
- A FIXED judge (local qwen3-vl text-only, temperature 0) grades each
  explanation text against the printed question context (extracted
  deterministically from the born-digital booklet's text layer). The
  image-processing key's reference reasoning is NOT available on this
  machine (key cache lives on the strong PC), so verdicts are the judge's
  own reading of the question — identical context for the gold text and
  every hypothesis, which keeps the DECISION-MATCH comparison controlled;
  absolute verdict quality is a separate question and is not claimed here.
- Metric per backend: decision_match_rate (verdict on OCR text == verdict
  on gold text), safe_rate (match OR the OCR text honestly flags
  unreadability -> human review, which is safe), verdict shift table.
- Fixed evaluation subset: cells with >= 25 chars of gold text, sorted by
  id, up to 3 per writer, 12 total — chosen deterministically before any
  backend is scored.

Judge calls are cached by (cell, sha1(text)) under grading_cache/ so gold
verdicts are computed once and every backend reuses them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "evaluation" / "hebrew_bench_v2"
CACHE = BENCH / "grading_cache"
CSV_OUT = REPO / "evaluation" / "m2_grading_results.csv"

VERDICTS = ["valid", "partially_valid", "invalid", "unintelligible"]

JUDGE_SYSTEM = (
    "You judge one short handwritten justification from a Hebrew image-"
    "processing exam. You get the printed question context and the "
    "student's justification text for one matching item. Judge ONLY "
    "whether the justification expresses correct, relevant reasoning for "
    "that item:\n"
    "- valid: correct core reasoning, any wording/language mix.\n"
    "- partially_valid: a correct central idea with a material part wrong "
    "or missing.\n"
    "- invalid: wrong, irrelevant, or empty reasoning (e.g. restating the "
    "choice).\n"
    "- unintelligible: the text is too garbled or fragmentary to judge at "
    "all.\n"
    'Reply ONLY with JSON: {"verdict": "<one of valid|partially_valid|'
    'invalid|unintelligible>", "reason": "<one short sentence>"}'
)

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": VERDICTS},
        "reason": {"type": "string"},
    },
    "required": ["verdict"],
}

UNREADABLE_MARKERS = ["[unreadable]", "[?]", "לא קריא", "unreadable"]


def question_context() -> dict[str, str]:
    """Deterministic printed context for Q1/Q2 from the booklet text layer."""
    import fitz

    doc = fitz.open(REPO / "sample_data" / "Exam_solution.pdf")
    q1 = doc[1].get_text()[:1200]
    q2_pages = [p for p in range(2, 5)]
    q2 = "\n".join(doc[p].get_text() for p in q2_pages)[:1200]
    doc.close()
    return {"1": q1, "2": q2}


def gold_cells() -> dict[str, str]:
    """cell id -> owner-verified text, joining multi-line cells."""
    refs = json.loads((BENCH / "references.json").read_text(encoding="utf-8"))
    items = json.loads((BENCH / "items.json").read_text(encoding="utf-8"))["items"]
    by_cell: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for it in items:
        iid = it["id"]
        if iid.startswith("hl_"):
            cell = iid.split("__")[0].replace("hl_", "")
            by_cell[cell].append((iid, refs[iid]["text"]))
        elif iid.startswith("hc_"):
            cell = iid.replace("hc_", "")
            by_cell[cell].append((iid, refs[iid]["text"]))
    return {
        cell: " ".join(t for _, t in sorted(parts))
        for cell, parts in by_cell.items()
    }


def eval_subset() -> list[str]:
    cells = gold_cells()
    picked: list[str] = []
    per_writer: Counter = Counter()
    for cell in sorted(cells):
        writer = cell.split("_")[0]
        if len(cells[cell]) < 25 or per_writer[writer] >= 3:
            continue
        picked.append(cell)
        per_writer[writer] += 1
        if len(picked) >= 12:
            break
    return picked


def backend_cell_texts(config_id: str) -> dict[str, str | None]:
    """cell id -> joined hypothesis text from a benchmark run's outputs."""
    outdir = BENCH / "outputs" / config_id / "run1"
    by_cell: dict[str, list[tuple[str, str | None]]] = defaultdict(list)
    for f in outdir.glob("*.json"):
        rec = json.loads(f.read_text(encoding="utf-8"))
        iid = rec["item"]
        if iid.startswith("hl_"):
            cell = iid.split("__")[0].replace("hl_", "")
        elif iid.startswith("hc_"):
            cell = iid.replace("hc_", "")
        else:
            continue
        by_cell[cell].append((iid, rec.get("transcription")))
    return {
        cell: " ".join((t or "") for _, t in sorted(parts)).strip() or None
        for cell, parts in by_cell.items()
    }


def judge(cell: str, text: str, ctx: dict[str, str], base_url: str, model: str) -> dict:
    key = hashlib.sha1(f"{cell}|{text}".encode()).hexdigest()
    CACHE.mkdir(exist_ok=True)
    cached = CACHE / f"{key}.json"
    if cached.exists():
        return json.loads(cached.read_text(encoding="utf-8"))
    qid = cell.split("_")[1].replace("q", "")
    row = cell.split("_")[2].replace("r", "")
    prompt = (
        f"Question {qid} context (printed, may be truncated):\n"
        f"{ctx.get(qid, '')}\n\n"
        f"Matching item (row) {row}. Student's justification text:\n"
        f"---\n{text}\n---\nJudge it now."
    )
    with httpx.Client(timeout=900.0) as client:
        resp = client.post(f"{base_url}/chat/completions", json={
            "model": model, "temperature": 0, "max_tokens": 300,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "V", "schema": SCHEMA}},
        })
        data = resp.json()
    try:
        out = json.loads(data["choices"][0]["message"]["content"])
    except Exception as e:  # noqa: BLE001
        out = {"verdict": "unintelligible", "reason": f"judge error: {e}"}
    out["cell"] = cell
    out["text_sha1"] = key
    cached.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-id", required=True,
                    help="benchmark run whose OCR outputs to score, or GOLD "
                         "to only compute/refresh the gold verdicts")
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--judge-model", default="qwen3-vl:8b-instruct")
    args = ap.parse_args()

    ctx = question_context()
    subset = eval_subset()
    gold = gold_cells()
    print(f"fixed subset ({len(subset)} cells): {subset}")

    t0 = time.monotonic()
    gold_verdicts = {}
    for cell in subset:
        v = judge(cell, gold[cell], ctx, args.base_url, args.judge_model)
        gold_verdicts[cell] = v["verdict"]
        print(f"  gold {cell}: {v['verdict']}")
    if args.config_id == "GOLD":
        print(f"gold pass done in {time.monotonic() - t0:.0f}s")
        return 0

    hyps = backend_cell_texts(args.config_id)
    match = safe = n = 0
    shifts: Counter = Counter()
    rows_detail = []
    for cell in subset:
        hyp = hyps.get(cell)
        n += 1
        flagged = hyp and any(m in hyp.lower() for m in UNREADABLE_MARKERS)
        if not hyp:
            hyp_verdict = "unintelligible"
            v_reason = "no OCR output"
        else:
            v = judge(cell, hyp, ctx, args.base_url, args.judge_model)
            hyp_verdict = v["verdict"]
            v_reason = v.get("reason", "")
        gv = gold_verdicts[cell]
        m = hyp_verdict == gv
        s = m or flagged or hyp_verdict == "unintelligible"
        match += m
        safe += s
        shifts[f"{gv}->{hyp_verdict}"] += 1
        rows_detail.append(f"{cell}: gold={gv} ocr={hyp_verdict} "
                           f"{'MATCH' if m else 'SAFE' if s else 'WRONG-DECISION'} | {v_reason[:60]}")
    for line in rows_detail:
        print(" ", line)

    import csv
    row = {
        "config_id": args.config_id, "judge_model": args.judge_model,
        "cells": n, "decision_match_rate": round(match / max(n, 1), 4),
        "safe_rate": round(safe / max(n, 1), 4),
        "verdict_shifts": "; ".join(f"{k}x{c}" for k, c in sorted(shifts.items())),
    }
    exists = CSV_OUT.exists()
    with CSV_OUT.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            w.writeheader()
        w.writerow(row)
    print(json.dumps(row, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
