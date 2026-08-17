"""F: grading-RAG paired A/B on 5 pre-registered audited answers.
G: escalation-only-RAG analysis over the persisted F outputs (no new calls).

Requires GRADE_PRIMARY_MODEL explicitly (NEVER falls back to SMOKE_MODEL).
Both arms use the SAME model, prompt, decoding, scoring schema and frozen
inputs; arm B adds ONLY the frozen local bge-m3 top-2 evidence under the
existing pack character budget. Both arms are persisted before any
reference verdict is read. References (frozen reference-side fixed-judge
verdicts from evaluation/m2_grading/gemini3_flash.jsonl) are joined
strictly post-hoc — they are NOT human ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autograder import courses  # noqa: E402
from autograder.escalation import GRADE_SYSTEM, GradeResult, grade_prompt, validate_grade  # noqa: E402
from autograder.gradingpack import build_pack  # noqa: E402
from autograder.orchestrator import setup_from_config  # noqa: E402
from autograder.schema import AnswerKey  # noqa: E402

# Pre-registered BEFORE any credentials existed (2026-08-17): five cells
# spanning writers e002/e003/e004/e006 and the three reference verdict
# classes; chosen from the frozen ledger's cell ids only.
PREREGISTERED_CELLS = ["e002_q1_r2", "e003_q1_r1", "e003_q1_r5", "e004_q1_r1", "e006_q1_r2"]
COURSE = "CV"

# The pipeline's verdict->score mapping for a 4-point sub-item (fixed judge scale)
VERDICT_SCORE = {"valid": 4.0, "partially_valid": 2.0, "invalid": 0.0, "unintelligible": None}


def score_to_verdict(score: float, uncertain: bool, max_score: float) -> str:
    if uncertain:
        return "uncertain"
    if score >= 0.75 * max_score:
        return "valid"
    if score > 0:
        return "partially_valid"
    return "invalid"


def cell_to_item(cell: str) -> tuple[str, str, str]:
    """e003_q1_r1 -> (writer, question_id, row) and the bench item id prefix."""
    w, q, r = cell.split("_")
    return w, q.replace("q", ""), r.replace("r", "")


def frozen_transcription(cell: str) -> str:
    """Protocol-clean Gemini transcription for the cell (joins hl_ line
    records or the hc_ cell record). Never reads references."""
    w, q, r = cell_to_item(cell)
    run = REPO / "evaluation" / "hebrew_bench_v2" / "outputs" / "gemini_protocol_clean_v1" / "run1"
    hc = run / f"hc_{w}_q{q}_r{r}.json"
    if hc.exists():
        return json.loads(hc.read_text(encoding="utf-8"))["transcription"]
    parts = []
    for p in sorted(run.glob(f"hl_{w}_q{q}_r{r}__l*.json")):
        parts.append(json.loads(p.read_text(encoding="utf-8"))["transcription"])
    if not parts:
        raise SystemExit(f"no frozen transcription for cell {cell}")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="evaluation/grading_rag_ab_v1")
    ap.add_argument("--analyze-only", action="store_true", help="G only: no calls")
    args = ap.parse_args()
    out = REPO / args.out
    (out / "armA").mkdir(parents=True, exist_ok=True)
    (out / "armB").mkdir(parents=True, exist_ok=True)

    # Frozen parsed key of the image-processing exam (A1/A2/A3): the
    # validated production parse in the persistent key cache (2026-07-12,
    # fingerprint 0758cd7f...). Override with --key-json if needed.
    key_json = os.environ.get("AB_KEY_JSON")
    if not key_json:
        cache = Path(os.environ.get("LOCALAPPDATA", "")) / "autograder" / "key_cache"
        hits = sorted(cache.glob("0758cd7fa39b5949*.json"))
        if not hits:
            raise SystemExit("frozen Exam_solution key not found in key cache; set AB_KEY_JSON")
        key_json = str(hits[0])
    kd = json.loads(Path(key_json).read_text(encoding="utf-8"))
    key = AnswerKey.model_validate(kd.get("answer_key", kd))
    print(f"key: {key.exam_title} versions={key.versions} questions={len(key.questions)}")

    if not args.analyze_only:
        model = os.environ.get("GRADE_PRIMARY_MODEL")
        if not model:
            print("STOP: GRADE_PRIMARY_MODEL is not set — F requires an explicitly configured grading model "
                  "(no fallback to SMOKE_MODEL).")
            return 2
        if not os.environ.get("OPENROUTER_API_KEY"):
            print("STOP: OPENROUTER_API_KEY not set")
            return 2
        cfg = out / "models_ab.toml"
        cfg.write_text(
            '[defaults]\nstructured_mode = "json_schema"\ntemperature = 0.0\ntimeout_s = 180.0\n'
            '[models.grade_primary]\nbackend = "openrouter"\nmodel = "${GRADE_PRIMARY_MODEL}"\n'
            'max_tokens = 300\nreasoning = { effort = "none" }\nprompt_version = "grade-v1"\ncacheable = false\n'
            '[budget]\nenabled = true\nmax_calls_per_job = 12\nsoft_fraction = 0.8\n', encoding="utf-8")
        rt = setup_from_config(cfg, out / "state")
        embed_fn = courses.ollama_embed_fn()

        for cell in PREREGISTERED_CELLS:
            w, qid, row = cell_to_item(cell)
            q = next(x for x in key.questions if x.id == qid)
            transcription = frozen_transcription(cell)
            # student's selected option is not part of these ledger cells' fidelity task; grade the explanation
            for arm, use_rag in (("A", False), ("B", True)):
                target = out / f"arm{arm}" / f"{cell}.json"
                if target.exists():
                    continue
                pack = build_pack(key, q, grading_policy="choice_and_explanation_independent",
                                  course_id=COURSE if use_rag else None,
                                  retrieve=courses.retrieve if use_rag else None, embed_fn=embed_fn if use_rag else None,
                                  rag_top_k=2 if use_rag else 0)
                # restrict the pack context to THIS sub-item so both arms are compact & identical apart
                # from RAG, and grade on the SUB-ITEM's point range (the reference verdict scale)
                sub = next(s for s in q.sub_items if s.id == row)
                pack.correct_by_version = {row: pack.correct_by_version.get(row, {})}
                pack.official_solution = {row: pack.official_solution[row]} if row in pack.official_solution else {}
                pack.question_text = f"{q.title}\n- ({row}) {sub.prompt}"
                pack.max_score = float(sub.points)
                pack.compute_hash()
                blocks = grade_prompt(pack, selected=None, transcription=transcription, version=None)
                t0 = time.monotonic()
                try:
                    res = rt.gateway.call(task="grade_primary", system=GRADE_SYSTEM, content_blocks=blocks,
                                          output_model=GradeResult,
                                          meta={"job_id": "rag_ab", "exam_id": cell, "question_id": qid,
                                                "stage": f"arm{arm}", "pack_hash": pack.hash})
                    val = res.value
                    v = validate_grade(val, pack, selection_correct=None, selected=None)
                    rec = {"cell": cell, "arm": arm, "rag": use_rag, "model": model,
                           "score": val.score, "rubric_items_met": val.rubric_items_met, "uncertain": val.uncertain,
                           "evidence": val.evidence, "validation_ok": v.ok, "validation_problems": v.problems,
                           "usage": res.usage, "latency_s": round(time.monotonic() - t0, 2),
                           "prompt_chars": len(blocks[0]["text"]), "pack_hash": pack.hash,
                           "rag_chunks": [e.chunk_id for e in pack.rag_evidence], "max_score": pack.max_score,
                           "error": None}
                except Exception as e:  # noqa: BLE001
                    rec = {"cell": cell, "arm": arm, "rag": use_rag, "model": model, "error": f"{type(e).__name__}: {str(e)[:200]}",
                           "latency_s": round(time.monotonic() - t0, 2)}
                target.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
                if rec.get("error"):
                    print(f"arm{arm} {cell}: ERR {rec['error']}")
                else:
                    print(f"arm{arm} {cell}: score={rec['score']} unc={rec['uncertain']} "
                          f"tok={rec['usage'].get('total_tokens')}")
        print("both arms persisted; joining references now")

    # ---------------- post-hoc: references + G analysis ----------------
    ledger = {json.loads(l)["cell"]: json.loads(l) for l in
              (REPO / "evaluation" / "m2_grading" / "gemini3_flash.jsonl").read_text(encoding="utf-8").splitlines()}
    rows = []
    for cell in PREREGISTERED_CELLS:
        a = json.loads((out / "armA" / f"{cell}.json").read_text(encoding="utf-8"))
        b = json.loads((out / "armB" / f"{cell}.json").read_text(encoding="utf-8"))
        ref_v = ledger[cell]["verdict_ref"]
        mx = a.get("max_score") or b.get("max_score") or 4.0
        ref_score = VERDICT_SCORE.get(ref_v)
        ref_score = ref_score if ref_score is None else ref_score * (mx / 4.0)

        def side(r):
            if r.get("error"):
                return {"verdict": "error", "score": None, "uncertain": True, "in": None, "out": None, "cost": None,
                        "latency": r.get("latency_s")}
            return {"verdict": score_to_verdict(r["score"], r["uncertain"], mx), "score": r["score"],
                    "uncertain": r["uncertain"] or not r["validation_ok"],
                    "in": r["usage"].get("input_tokens"), "out": r["usage"].get("output_tokens"),
                    "reasoning": r["usage"].get("reasoning_tokens"), "cost": r["usage"].get("reported_cost"),
                    "latency": r.get("latency_s")}
        A, B = side(a), side(b)
        rows.append({"cell": cell, "ref_verdict": ref_v, "ref_score": ref_score, "A": A, "B": B,
                     "rag_extra_input_tokens": (B["in"] or 0) - (A["in"] or 0) if A["in"] is not None and B["in"] is not None else None})

    def summarize(side_key):
        n = len(rows)
        dec = sum(1 for r in rows if r[side_key]["verdict"] == r["ref_verdict"])
        exact = sum(1 for r in rows if r["ref_score"] is not None and r[side_key]["score"] == r["ref_score"])
        unc = sum(1 for r in rows if r[side_key]["uncertain"])
        up = sum(1 for r in rows if r["ref_score"] is not None and r[side_key]["score"] is not None and r[side_key]["score"] > r["ref_score"] and not r[side_key]["uncertain"])
        down = sum(1 for r in rows if r["ref_score"] is not None and r[side_key]["score"] is not None and r[side_key]["score"] < r["ref_score"] and not r[side_key]["uncertain"])
        return {"n": n, "decision_correct": dec, "exact_score_correct": exact, "uncertain_or_review": unc,
                "harmful_upgrade_silent": up, "harmful_downgrade_silent": down,
                "input_tokens": sum(r[side_key]["in"] or 0 for r in rows),
                "output_tokens": sum(r[side_key]["out"] or 0 for r in rows),
                "reported_cost": round(sum(r[side_key]["cost"] or 0 for r in rows), 6),
                "mean_latency_s": round(sum(r[side_key]["latency"] or 0 for r in rows) / n, 2)}
    sA, sB = summarize("A"), summarize("B")
    extra_in = sum(r["rag_extra_input_tokens"] or 0 for r in rows)
    gained = sB["decision_correct"] - sA["decision_correct"]
    # G: escalation-only replay — B is consulted only where A is uncertain/failed
    esc_dec = esc_in = esc_out = 0
    esc_cost = 0.0
    esc_calls_b = 0
    for r in rows:
        if r["A"]["uncertain"]:
            esc_calls_b += 1
            use = r["B"]
            esc_in += (r["A"]["in"] or 0) + (r["B"]["in"] or 0)
            esc_out += (r["A"]["out"] or 0) + (r["B"]["out"] or 0)
            esc_cost += (r["A"]["cost"] or 0) + (r["B"]["cost"] or 0)
        else:
            use = r["A"]
            esc_in += r["A"]["in"] or 0
            esc_out += r["A"]["out"] or 0
            esc_cost += r["A"]["cost"] or 0
        esc_dec += 1 if use["verdict"] == r["ref_verdict"] else 0
    report = {"model": (rows and (json.loads((out / "armA" / f"{rows[0]['cell']}.json").read_text(encoding="utf-8")).get("model"))),
              "cells": PREREGISTERED_CELLS, "rows": rows,
              "armA_no_rag": sA, "armB_rag": sB,
              "rag_extra_input_tokens_total": extra_in,
              "rag_extra_input_tokens_per_answer": round(extra_in / len(rows), 1) if rows else None,
              "decisions_gained_by_rag": gained,
              "extra_tokens_per_correct_decision_gained": (round(extra_in / gained, 1) if gained > 0 else None),
              "G_escalation_only": {"decision_correct": esc_dec, "rag_calls_used": esc_calls_b,
                                    "input_tokens": esc_in, "output_tokens": esc_out, "reported_cost": round(esc_cost, 6)},
              "note": "reference = frozen reference-side FIXED-JUDGE verdict, NOT human ground truth"}
    (out / "ab_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, ensure_ascii=False, indent=1))
    for r in rows:
        print(f"  {r['cell']}: ref={r['ref_verdict']} | A={r['A']['verdict']}({r['A']['score']}) B={r['B']['verdict']}({r['B']['score']}) "
              f"| in A={r['A']['in']} B={r['B']['in']} (+{r['rag_extra_input_tokens']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
