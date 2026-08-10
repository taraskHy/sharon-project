"""qwen_rag_ocr_v1 — EXPERIMENTAL course-RAG OCR-repair arm.

Per item:  frozen raw Qwen OCR  (+ optional printed-question text from an
           operator-supplied ANSWER-FREE file, --question-context)
           -> retrieve top_k course-summary chunks (course index)
           -> local repair model (text-only) with a fidelity-first prompt
           -> suggested transcription + structured edit evidence.

The repair stage NEVER sees: answer key, rubric, reference transcription,
grader decisions, benchmark labels. Question text FAILS CLOSED: without an
explicit --question-context file none is used — the solved booklet
(sample_data/Exam_solution.pdf) is never read by this arm, and paths under
sample_data/ are refused. raw_text is preserved byte-for-byte;
suggested_text never overwrites it. Semantic-risk changes are flagged for
review, not auto-accepted. Production grading is untouched.

Everything is cached/persisted per item (retrieval chunk ids + scores,
repair outputs) under evaluation/hebrew_bench_v2/outputs/qwen_rag_ocr_v1/
so the frozen evaluation is auditable and resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import httpx

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
BENCH = REPO / "evaluation" / "hebrew_bench_v2"

sys.path.insert(0, str(REPO))
from autograder import courses  # noqa: E402


def load_question_context(path: str | None) -> dict:
    """Operator-supplied answer-free question text: {question_id: text}.

    FAIL-CLOSED default: with no file, the repair stage gets no printed
    question text at all. The solved booklet and anything else under
    sample_data/ (solutions, keys, overrides) are refused outright — an
    answer-free source must be curated by the operator, not sliced out of
    a solved document."""
    if not path:
        return {}
    p = Path(path).resolve()
    if (REPO / "sample_data").resolve() in p.parents:
        sys.exit("--question-context must not point into sample_data/ "
                 "(solved/graded artifacts are not an answer-free source)")
    if p.suffix.lower() != ".json":
        sys.exit("--question-context must be a curated .json file mapping "
                 "question ids to plain question text")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in data.items()
    ):
        sys.exit("--question-context must be a JSON object {question_id: text}")
    return data

REPAIR_SYSTEM = (
    "You are repairing OCR from a handwritten student exam.\n"
    "The course passages below are context for vocabulary and terminology "
    "only.\n"
    "Do NOT solve the exam question.\n"
    "Do NOT make the student's answer more correct.\n"
    "Do NOT replace a concept merely because another term would be the "
    "correct course answer.\n"
    "Preserve student mistakes.\n"
    "Use course context only when it helps identify a plausible OCR "
    "corruption.\n"
    "If the raw text could genuinely represent an incorrect student "
    "statement, preserve it or mark it uncertain rather than silently "
    "correcting it.\n"
    "Short technical tokens, numbers, negations, variables, operators, and "
    "English abbreviations require especially strong caution."
)

REPAIR_SCHEMA = {
    "type": "object",
    "properties": {
        "suggested_text": {"type": "string"},
        "edits": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "raw": {"type": "string"},
                "suggested": {"type": "string"},
                "reason": {"type": "string"},
                "supporting_chunk_ids": {"type": "array", "items": {"type": "string"}},
                "risk": {"type": "string", "enum": ["low", "semantic"]},
            },
            "required": ["raw", "suggested", "reason", "risk"],
        }},
        "uncertain_regions": {"type": "array", "items": {"type": "string"}},
        "semantic_change_risk": {"type": "boolean"},
    },
    "required": ["suggested_text", "edits", "semantic_change_risk"],
}


def build_repair_prompt(question_text: str, raw_ocr: str, chunks: list[dict]) -> str:
    """The ONLY inputs are the printed question, the raw OCR, and retrieved
    course chunks — never keys/rubrics/references (tested)."""
    ctx = "\n\n".join(
        f"[chunk {c['chunk_id']} | {c['source']}"
        + (f" p.{c['page']}" if c.get("page") else "")
        + f"]\n{c['text']}"
        for c in chunks
    ) or "(no relevant course passages retrieved)"
    q = question_text.strip() or "(not provided — no answer-free question source configured)"
    return (
        f"Printed exam question (context):\n{q}\n\n"
        f"Course passages (terminology context only):\n{ctx}\n\n"
        f"RAW OCR of the student's handwritten answer:\n---\n{raw_ocr}\n---\n"
        "Repair ONLY plausible OCR corruptions. Reply with the JSON object."
    )


def repair_call(base_url: str, model: str, prompt: str) -> dict:
    with httpx.Client(timeout=900.0) as client:
        resp = client.post(f"{base_url}/chat/completions", json={
            "model": model, "temperature": 0, "max_tokens": 1200,
            "messages": [{"role": "system", "content": REPAIR_SYSTEM},
                         {"role": "user", "content": prompt}],
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "R", "schema": REPAIR_SCHEMA}},
        })
        data = resp.json()
    return json.loads(data["choices"][0]["message"]["content"])


def assemble_record(item: str, raw_text: str, rep: dict | None,
                    chunks: list[dict], err: str | None = None) -> dict:
    """Pure record assembly. Guarantees: raw_text preserved byte-for-byte;
    on repair failure/absence suggested_text degrades to raw_text; semantic
    risk or uncertain regions force needs_review."""
    suggested = (rep or {}).get("suggested_text") or raw_text
    return {
        "item": item,
        "raw_text": raw_text,  # NEVER overwritten
        "suggested_text": suggested,
        # Provider contract: every arm's run1/<item>.json exposes the text it
        # proposes as "transcription" — the field the frozen bench/grading
        # evaluators score. For this arm that is the repaired text; raw_text
        # stays available for paired raw-vs-suggested analysis.
        "transcription": suggested,
        "edits": (rep or {}).get("edits", []),
        "uncertain_regions": (rep or {}).get("uncertain_regions", []),
        "semantic_change_risk": bool((rep or {}).get("semantic_change_risk")),
        "needs_review": bool((rep or {}).get("semantic_change_risk"))
                        or any(e.get("risk") == "semantic" for e in (rep or {}).get("edits", []))
                        or bool((rep or {}).get("uncertain_regions")),
        "retrieved": [{"chunk_id": c["chunk_id"], "source": c["source"],
                       "page": c.get("page"), "similarity": c["similarity"]}
                      for c in chunks],
        "retrieval_empty": not chunks,
        "error": err,
    }


def eligible_items(source_config: str) -> list[dict]:
    """Handwritten items with a valid frozen raw prediction."""
    items = json.loads((BENCH / "items.json").read_text(encoding="utf-8"))["items"]
    out = []
    for it in items:
        if it["category"] not in ("handwritten_line", "handwritten_cell"):
            continue
        p = BENCH / "outputs" / source_config / "run1" / f"{it['id']}.json"
        if not p.exists():
            continue
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("error") or not (rec.get("transcription") or "").strip():
            continue
        out.append({"item": it["id"], "raw_text": rec["transcription"]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--course", required=True, help="course id with a built index")
    ap.add_argument("--source-config", default="qwen8b_strict_contrast",
                    help="frozen OCR arm supplying raw text")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--repair-model", default="qwen3-vl:8b-instruct")
    ap.add_argument("--embed-base-url", default="http://localhost:11434")
    ap.add_argument("--embed-model", default=courses.DEFAULT_EMBED_MODEL)
    ap.add_argument("--items", default="", help="comma list to restrict")
    ap.add_argument("--question-context", default=None,
                    help="optional operator-curated ANSWER-FREE JSON file "
                         "{question_id: printed question text}; omitted = "
                         "no question text (fail closed)")
    ap.add_argument("--config-id", default="qwen_rag_ocr_v1",
                    help="output arm id under outputs/ — lets the identical "
                         "frozen repair run against a different frozen raw "
                         "source (e.g. gemini_rag_ocr_v1)")
    args = ap.parse_args()
    outdir = BENCH / "outputs" / args.config_id

    status = courses.index_status(args.course)
    if not status["indexed"]:
        sys.exit(f"course {args.course!r} has no built index")
    if status["stale"]:
        sys.exit(f"course {args.course!r} index is STALE (sources/config changed) — rebuild first")

    run_dir = outdir / "run1"
    run_dir.mkdir(parents=True, exist_ok=True)
    ctx = load_question_context(args.question_context)
    embed_fn = courses.ollama_embed_fn(args.embed_model, args.embed_base_url)

    config = {
        "config_id": args.config_id, "course": args.course,
        "course_config_hash": status.get("config_hash"),
        "source_config": args.source_config, "top_k": args.top_k,
        "repair_model": args.repair_model,
        "embed_model": args.embed_model,
        "question_context_file": args.question_context or None,
        "question_context_sha256": (
            hashlib.sha256(Path(args.question_context).read_bytes()).hexdigest()[:16]
            if args.question_context else None
        ),
        "prompt_sha256": hashlib.sha256((REPAIR_SYSTEM + json.dumps(REPAIR_SCHEMA, sort_keys=True)).encode()).hexdigest()[:16],
        "decoding": {"temperature": 0, "max_tokens": 1200},
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (outdir / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")

    todo = eligible_items(args.source_config)
    if args.items:
        keep = set(args.items.split(","))
        todo = [t for t in todo if t["item"] in keep]
    print(f"eligible items: {len(todo)}")
    for n, t in enumerate(todo, 1):
        target = run_dir / f"{t['item']}.json"
        if target.exists():
            continue
        qid = t["item"].split("_")[2].replace("q", "") if "_q" in t["item"] else "1"
        question_text = ctx.get(qid, "")[:900]
        query = "\n".join(x for x in (question_text, t["raw_text"]) if x)
        t0 = time.monotonic()
        chunks = courses.retrieve(args.course, query, args.top_k, embed_fn)
        retrieval_s = round(time.monotonic() - t0, 2)
        prompt = build_repair_prompt(question_text, t["raw_text"], chunks)
        t1 = time.monotonic()
        try:
            rep = repair_call(args.base_url, args.repair_model, prompt)
            err = None
        except Exception as e:  # noqa: BLE001
            rep, err = None, f"{type(e).__name__}: {e}"
        rec = assemble_record(t["item"], t["raw_text"], rep, chunks, err)
        rec["retrieval_s"] = retrieval_s
        rec["repair_s"] = round(time.monotonic() - t1, 2)
        target.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{n}/{len(todo)}] {t['item']}: edits={len(rec['edits'])} "
              f"sem_risk={rec['semantic_change_risk']} "
              f"chunks={len(chunks)} ({rec['repair_s']}s)"
              + (f" ERR {err[:60]}" if err else ""))
    print(f"{args.config_id} run complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
