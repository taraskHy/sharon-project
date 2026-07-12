"""Local-model smoke validation against the REAL sample exam.

Runs three probes against a configured OpenAI-compatible backend and prints
timings and raw structured outputs for manual inspection. This is a
capability probe, not a benchmark: it answers "does the plumbing work, can
the model read this material at all, and how slow is it on this hardware".

    python scripts/smoke_probe.py --base-url http://localhost:11434/v1 --model qwen3-vl:8b

Probes:
  A (text-only)  judge one real Hebrew student explanation against the key's
                 reference reasoning — tests Hebrew comprehension + JSON.
  B (vision)     read page 6 of the student exam (printed Hebrew MC questions
                 with handwritten circle/X marks) — tests printed-Hebrew OCR
                 and mark detection.
  C (vision)     read page 13 (the multiple-choice bubble table where the
                 student used circles AND X marks and wrote a note declaring
                 X as final) — tests handwriting + convention understanding.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Hebrew output on a cp1252 Windows console would crash print().
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pydantic import BaseModel, Field

from autograder.backends import BackendConfig, create_backend
from autograder.ingest import image_block, load_pages


class JudgeProbe(BaseModel):
    verdict: str = Field(description="valid | partially_valid | invalid")
    reasoning: str


class PageReadProbe(BaseModel):
    page_language: str
    first_question_transcription: str = Field(
        description="Transcription of the first printed question on the page (Hebrew)."
    )
    handwritten_marks_seen: list[str] = Field(
        description="Each handwritten mark visible on the page: kind + which option it touches."
    )
    marked_option_of_first_question: Optional[str] = None


class TableReadProbe(BaseModel):
    handwritten_note_transcription: Optional[str] = Field(
        default=None,
        description="Any handwritten note near the answer table, transcribed as written.",
    )
    note_meaning: Optional[str] = None
    question_1_final_answer: Optional[str] = Field(
        default=None, description="Final marked option (A-D) for row 1 under the student's convention."
    )
    marks_description: str = Field(
        description="What kinds of marks appear in the table (circles, X, cross-outs)."
    )


def run_probe(name, backend, system, blocks, model_cls, max_tokens=None):
    print(f"\n=== Probe {name} ===", flush=True)
    t0 = time.monotonic()
    try:
        result = backend.parse(
            system=system, content_blocks=blocks, output_model=model_cls, max_tokens=max_tokens
        )
        dt = time.monotonic() - t0
        print(f"OK in {dt:.1f}s")
        print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
        return dt, result
    except Exception as e:  # noqa: BLE001 - report and continue
        dt = time.monotonic() - t0
        print(f"FAILED after {dt:.1f}s: {type(e).__name__}: {e}")
        return dt, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="qwen3-vl:8b")
    ap.add_argument("--structured-mode", default="json_schema")
    ap.add_argument("--max-image-edge", type=int, default=1400)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--exam", default="sample_data/student_exam.pdf")
    ap.add_argument("--skip", default="", help="comma list of probes to skip, e.g. B,C")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument(
        "--no-think", action="store_true",
        help="send think=false (Ollama: disables reasoning tokens that eat the output budget)",
    )
    args = ap.parse_args()

    backend = create_backend(
        BackendConfig(
            backend="openai",
            model=args.model,
            base_url=args.base_url,
            structured_mode=args.structured_mode,
            timeout_s=args.timeout,
            temperature=0.0,
            max_tokens=args.max_tokens,
            extra_generation={"think": False} if args.no_think else {},
        )
    )
    print("health:", backend.health_check())
    skip = {s.strip().upper() for s in args.skip.split(",") if s.strip()}

    results = {}
    if "A" not in skip:
        results["A"] = run_probe(
            "A: text-only Hebrew judging",
            backend,
            system=(
                "You judge whether a student's short Hebrew explanation expresses "
                "the same core reasoning as the reference. Answer in the JSON schema."
            ),
            blocks=[
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "question": "מה קורה כשמוסיפים 100 לכל ערכי התמונה, בייצוג פירמידת wavelet?",
                            "reference_reasoning": (
                                "הוספת קבוע משפיעה רק על רכיב ה-DC, כלומר רק על הרמה "
                                "הגסה ביותר (האחרונה) של הפירמידה; שאר התדרים אינם משתנים."
                            ),
                            "student_explanation": "הוספה של 100 משפיע רק על הרמה האחרונה, שהיא הבהירות",
                        },
                        ensure_ascii=False,
                    ),
                }
            ],
            model_cls=JudgeProbe,
        )

    pages = load_pages(args.exam, args.max_image_edge)
    by_num = {p.page_number: p for p in pages}

    if "B" not in skip:
        results["B"] = run_probe(
            "B: printed Hebrew + marks (page 6)",
            backend,
            system=(
                "You read a scanned Hebrew exam page (RTL). Transcribe faithfully "
                "and report handwritten marks. Answer in the JSON schema."
            ),
            blocks=[image_block(by_num[6])],
            model_cls=PageReadProbe,
        )

    if "C" not in skip:
        results["C"] = run_probe(
            "C: bubble table + convention note (page 13)",
            backend,
            system=(
                "You read a scanned answer table from a Hebrew exam. The student may "
                "have used several kinds of marks and written a note about which mark "
                "counts. Report what you see. Answer in the JSON schema."
            ),
            blocks=[image_block(by_num[13])],
            model_cls=TableReadProbe,
        )

    print("\n=== Summary ===")
    for name, (dt, result) in results.items():
        print(f"Probe {name}: {'OK' if result else 'FAILED'} ({dt:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
