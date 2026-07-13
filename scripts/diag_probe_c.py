"""Diagnostic for probe C truncation: send page 13 in PROMPT mode (no
response_format) and print the RAW partial/full output, finish_reason and
token counts, so we can see WHAT consumes the budget (verbose enumeration vs
repetition loop). Bypasses the backend's truncation error on purpose.

    python scripts/diag_probe_c.py --model qwen3-vl:8b-instruct --max-tokens 1200
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

from autograder.ingest import load_pages

SCHEMA_NOTE = (
    "Respond with ONLY a single JSON object with keys: "
    "handwritten_note_transcription (string|null), note_meaning (string|null), "
    "question_1_final_answer (string|null), marks_description (string). "
    "Be CONCISE: describe kinds of marks and the overall pattern, do NOT "
    "enumerate every row of the table."
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="qwen3-vl:8b-instruct")
    ap.add_argument("--exam", default="sample_data/student_exam.pdf")
    ap.add_argument("--page", type=int, default=13)
    ap.add_argument("--max-image-edge", type=int, default=1000)
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--concise", action="store_true", help="add the be-concise instruction")
    ap.add_argument("--json-schema", action="store_true", help="use constrained decoding too")
    ap.add_argument(
        "--exact-probe", action="store_true",
        help="replicate the smoke probe's EXACT request: Pydantic schema of "
        "TableReadProbe in both the schema note and response_format",
    )
    args = ap.parse_args()

    pages = {p.page_number: p for p in load_pages(args.exam, args.max_image_edge)}
    page = pages[args.page]
    b64 = base64.standard_b64encode(page.png_bytes).decode("ascii")

    system = (
        "You read a scanned answer table from a Hebrew exam. The student may "
        "have used several kinds of marks and written a note about which mark "
        "counts. Report what you see."
    )
    note = SCHEMA_NOTE if args.concise else (
        "Respond with ONLY a single JSON object with keys: "
        "handwritten_note_transcription (string|null), note_meaning (string|null), "
        "question_1_final_answer (string|null), marks_description (string)."
    )
    payload: dict = {
        "model": args.model,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": note},
                ],
            },
        ],
    }
    if args.exact_probe:
        from scripts.smoke_probe import TableReadProbe

        payload["messages"][0]["content"] = (
            "You read a scanned answer table from a Hebrew exam. The student may "
            "have used several kinds of marks and written a note about which mark "
            "counts. Report what you see. Answer in the JSON schema."
        )
        schema = TableReadProbe.model_json_schema()
        payload["messages"][1]["content"][1]["text"] = (
            "Respond with ONLY a single JSON object (no prose, no markdown fences) "
            "that conforms exactly to this JSON Schema:\n"
            + json.dumps(schema, ensure_ascii=False)
        )
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "TableReadProbe", "schema": schema},
        }
    elif args.json_schema:
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "TableReadProbe",
                "schema": {
                    "type": "object",
                    "properties": {
                        "handwritten_note_transcription": {"type": ["string", "null"]},
                        "note_meaning": {"type": ["string", "null"]},
                        "question_1_final_answer": {"type": ["string", "null"]},
                        "marks_description": {"type": "string"},
                    },
                    "required": ["marks_description"],
                },
            },
        }

    t0 = time.monotonic()
    with httpx.Client(timeout=600.0) as client:
        resp = client.post(f"{args.base_url}/chat/completions", json=payload)
    dt = time.monotonic() - t0
    data = resp.json()
    choice = data["choices"][0]
    usage = data.get("usage", {})
    print(f"elapsed: {dt:.1f}s  finish_reason={choice.get('finish_reason')}")
    print(f"usage: {json.dumps(usage)}")
    print(f"content length: {len(choice['message'].get('content') or '')} chars")
    print("--- RAW CONTENT ---")
    print(choice["message"].get("content") or "(empty)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
