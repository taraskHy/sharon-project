"""Two pre-registered local improvement arms over the frozen SEEN-46 reference.

    python scripts/improvement_arms.py freeze     # pre-register (no calls)
    python scripts/improvement_arms.py run-a      # ARM A: q8_0 on all 46
    python scripts/improvement_arms.py run-b      # ARM B: pass-2 verifier on all 46
    python scripts/improvement_arms.py report     # evaluate all three arms

ARM A — higher-precision quantization: qwen3-vl:8b-instruct-q8_0 (same base
model family as the baseline, Q8_0 vs Q4_K_M — NOT an independent model),
with the UNCHANGED grade-v4-charitable-local prompt/adapter/validator.

ARM B — two-pass local verification: pass 1 = the EXISTING frozen baseline
outputs (never re-run); pass 2 = an independent local verification of the
proposed verdict against question + rubric + official solution + frozen
transcription + pass-1 grounded evidence. Pass 2 never sees the human
reference, instructor score, split, audit decisions, or historical outcomes
(a leakage scan runs on every request). The deterministic combination rule
below was source-derived from the e002/e003 baseline errors ONLY and frozen
BEFORE any new output; e004 serves as the held writer for the rule.

THE PRE-REGISTERED COMBINATION RULE (ordered; first match wins):
  0. pass-1 decision REVIEW (its own validation failed) -> final = pass-1
     verdict, REVIEW ("pass1_review_sticks").
  1. pass-2 unusable (schema failure, ungrounded/short quote, empty evidence
     while crediting, or uncertain=true) -> final = pass-1 verdict, REVIEW
     ("verifier_unusable").
  2. pass-2 recommendation == pass-1 verdict -> final = that verdict, AUTO
     ("agreed").
  3. pass-2 exactly ONE rank above pass-1 AND grounded AND (target valid ->
     central_idea_present; target partially_valid -> central_idea_present or
     directionally_correct_but_incomplete) -> final = pass-2 verdict, AUTO
     ("verifier_upgrade"; the student-protective path, conditional on
     grounded evidence).
  4. pass-2 above pass-1 by TWO ranks -> final = pass-1 verdict, REVIEW
     ("two_step_disagreement").
  5. pass-2 below pass-1 -> final = pass-1 verdict, REVIEW
     ("verifier_flags_generosity"; a downgrade is never automated).
  6. anything else -> final = pass-1 verdict, REVIEW ("unresolved").

Local Ollama only; cache bypassed; RAG_DISABLED; no OCR; HELD_OUT
structurally absent. Failures are preserved, never retried silently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

EXPERIMENTS = REPO / "evaluation" / "model_selection" / "experiments"
RUNS = REPO / "evaluation" / "model_selection" / "runs" / "local_grade_primary"
SPEC_PATH = EXPERIMENTS / "LOCAL_IMPROVEMENT_ARMS_2026-09-02.json"
REF_PATH = RUNS / "FINAL_HUMAN_REFERENCE_2026-09-02.json"
RERUN_JSONL = RUNS / "CORRECTED_RERUN_2026-09-02.jsonl"
ARM_A_JSONL = RUNS / "ARM_A_Q8_2026-09-02.jsonl"
ARM_B_JSONL = RUNS / "ARM_B_VERIFY_2026-09-02.jsonl"
REPORT_JSON = RUNS / "LOCAL_IMPROVEMENT_REPORT_2026-09-02.json"
REPORT_MD = RUNS / "LOCAL_IMPROVEMENT_REPORT_2026-09-02.md"
SEEN46_RUN_DIRS = ("dev__all__qwen3-vl-8b-instruct__72e19378d1",
                   "calibration__all__qwen3-vl-8b-instruct__e2a3cfc925")
REPAIRED = ("e004_q2_r6", "e004_q2_r8")

BASELINE_MODEL = "qwen3-vl:8b-instruct"
ARM_A_MODEL = "qwen3-vl:8b-instruct-q8_0"
PROMPT_VERSION = "grade-v4-charitable-local"
VERIFY_PROMPT_VERSION = "verify-v1-local"
BASE_URL = "http://localhost:11434/v1"

VERDICTS = ("invalid", "partially_valid", "valid")
RANK = {v: i for i, v in enumerate(VERDICTS)}
MIN_QUOTE_CHARS = 3

# strings that must NEVER appear in any model-visible request text
FORBIDDEN_IN_REQUEST = ("final_human", "human_reference", "instructor_derived",
                        "label_verdict", "explanation_verdict", "selection_correct",
                        "adjudicated", "HELD_OUT", "reference_source",
                        "two_reviewer_consensus")

VERIFY_SYSTEM = """You verify a proposed grade for one Hebrew exam answer.
You receive the question, the grading rubric, the official solution, the student's
verbatim transcribed text, and a PROPOSED verdict with its cited evidence.

Judge ONLY what the student actually wrote. Missing reasoning stays missing: never
imagine, complete, or invent content the student did not write.

Answer these questions about the proposal:
- Is the proposed verdict supported by the actual student text?
- Is a central correct idea (per the rubric/official solution) present in the text?
- Is the answer merely directionally correct but incomplete?
- If the proposal is partially_valid or invalid: is it excessively strict?
- If the proposal is valid: is it too generous?

Output rules (mechanical, mandatory):
- `recommended_verdict` is your own independent verdict for the explanation quality:
  one of "invalid", "partially_valid", "valid".
- `evidence` must contain EXACT substrings copied verbatim from the student text
  (each at most 200 characters). Every claim of present content must be backed by
  such a quote. If you recommend any credit (partially_valid or valid) you MUST
  provide at least one quote showing the credited idea.
- If you cannot ground your recommendation in exact quotes, set `uncertain` to true
  instead of guessing.
- No prose anywhere else; only the JSON object."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _verify_result_model():
    from typing import Literal

    from pydantic import BaseModel, ConfigDict

    class VerifyQuote(BaseModel):
        model_config = ConfigDict(extra="forbid")
        quote: str

    class VerifyResult(BaseModel):
        model_config = ConfigDict(extra="forbid")
        supported: bool
        central_idea_present: bool
        directionally_correct_but_incomplete: bool
        proposed_too_strict: bool
        proposed_too_generous: bool
        recommended_verdict: Literal["invalid", "partially_valid", "valid"]
        evidence: list[VerifyQuote]
        uncertain: bool

    return VerifyResult


def rule_source_sha() -> str:
    """The combination rule's actual code — frozen in the spec, asserted at
    run and report time so the rule can never drift after registration."""
    import inspect
    return _sha(inspect.getsource(combine) + inspect.getsource(verifier_usable))


def verify_blocks(pack, transcription: str, pass1_verdict: str,
                  pass1_rubric_items: list[dict]) -> list[dict]:
    """Pass-2 request content. ONLY: question/rubric/official solution (the
    grader's own public pack context), the frozen transcription, and pass-1's
    proposed verdict + grounded evidence. Nothing else exists here."""
    ev = json.dumps([{"id": ri.get("id"), "met": ri.get("met"),
                      "student_evidence": ri.get("student_evidence")}
                     for ri in pass1_rubric_items], ensure_ascii=False)
    return [{"type": "text", "text": (
        pack.to_grader_context(include_scoring_rules=False) + "\n\n"
        + f"Student explanation (verbatim transcription):\n---\n{transcription}\n---\n"
        + f"PROPOSED verdict: {pass1_verdict}\n"
        + f"PROPOSED evidence (pass-1 rubric items): {ev}\n")}]


def verifier_usable(v: dict, transcription: str) -> tuple[bool, str]:
    """Mechanical usability of a pass-2 output: schema-valid (already enforced
    upstream), grounded quotes, evidence present when crediting, not uncertain."""
    if v.get("uncertain"):
        return False, "uncertain"
    if v.get("recommended_verdict") not in VERDICTS:
        return False, "bad_verdict_value"
    quotes = [q.get("quote") or "" for q in v.get("evidence") or []]
    for q in quotes:
        if len(q) < MIN_QUOTE_CHARS or q not in (transcription or ""):
            return False, "ungrounded_quote"
    if v["recommended_verdict"] in ("valid", "partially_valid") and not quotes:
        return False, "credit_without_evidence"
    return True, "ok"


def combine(pass1_verdict: str, pass1_decision: str, pass2: dict | None,
            transcription: str) -> dict:
    """THE pre-registered deterministic combination (module docstring, frozen
    in the spec). Returns {final_verdict, decision, rule}."""
    if pass1_decision == "REVIEW":
        return {"final_verdict": pass1_verdict, "decision": "REVIEW",
                "rule": "pass1_review_sticks"}
    if pass2 is None:
        return {"final_verdict": pass1_verdict, "decision": "REVIEW",
                "rule": "verifier_unusable"}
    ok, _why = verifier_usable(pass2, transcription)
    if not ok:
        return {"final_verdict": pass1_verdict, "decision": "REVIEW",
                "rule": "verifier_unusable"}
    p2 = pass2["recommended_verdict"]
    if p2 == pass1_verdict:
        return {"final_verdict": pass1_verdict, "decision": "AUTO", "rule": "agreed"}
    delta = RANK[p2] - RANK[pass1_verdict]
    if delta == 1:
        gate = (pass2.get("central_idea_present") if p2 == "valid"
                else pass2.get("central_idea_present")
                or pass2.get("directionally_correct_but_incomplete"))
        if gate:
            return {"final_verdict": p2, "decision": "AUTO", "rule": "verifier_upgrade"}
        return {"final_verdict": pass1_verdict, "decision": "REVIEW", "rule": "unresolved"}
    if delta == 2:
        return {"final_verdict": pass1_verdict, "decision": "REVIEW",
                "rule": "two_step_disagreement"}
    return {"final_verdict": pass1_verdict, "decision": "REVIEW",
            "rule": "verifier_flags_generosity"}


# ---------------------------------------------------------------- loading ----

def load_reference() -> dict:
    doc = json.loads(REF_PATH.read_text(encoding="utf-8"))
    payload = json.dumps({k: v for k, v in doc.items() if k != "reference_sha256"},
                         ensure_ascii=False, sort_keys=True)
    assert doc["reference_sha256"] == _sha(payload), "reference freeze tampered"
    return doc


def load_baseline_outputs() -> dict[str, dict]:
    """Pass-1 = the frozen baseline outputs (44 SEEN-46 + 2 corrected)."""
    out: dict[str, dict] = {}
    for d in SEEN46_RUN_DIRS:
        for s in json.loads((RUNS / "grade_primary" / d / "scored.jsonl.json").read_text(encoding="utf-8")):
            if s["case_id"] not in REPAIRED:
                out[s["case_id"]] = s
    for line in RERUN_JSONL.read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        out[s["case_id"]] = s
    assert len(out) == 46
    return out


def load_baseline_rubric_items() -> dict[str, list[dict]]:
    import os
    data = Path(os.environ["LOCALAPPDATA"], "autograder", "review46", "bundle")
    id_map = json.loads((data / "private" / "id_map.json").read_text(encoding="utf-8"))
    props = json.loads((data / "private" / "model_proposals.json").read_text(encoding="utf-8"))
    out = {id_map[iid]: (p.get("rubric_items") or []) for iid, p in props.items()
           if not p.get("stale")}
    for line in RERUN_JSONL.read_text(encoding="utf-8").splitlines():
        s = json.loads(line)
        out[s["case_id"]] = s.get("rubric_items") or []
    assert len(out) == 46
    return out


def _bench():
    from autograder.benchmark.manifests import DEFAULT_BENCH_ROOT, load_manifest
    from autograder.benchmark.roles import GradeAdapter
    from autograder.benchmark.runner import files_root_for
    m = load_manifest("grade_primary")
    adapter = GradeAdapter("grade_primary", prompt_version=PROMPT_VERSION)
    return m, adapter, {c.case_id: c for c in m.cases}, files_root_for(m, DEFAULT_BENCH_ROOT)


def _gateway(model: str, prompt_version: str):
    from autograder.cloudboundary import check_cloud_call
    from autograder.gateway import ModelGateway, TaskRoute
    from autograder.usage import UsageLedger
    check_cloud_call(task="grade_primary", backend="ollama", base_url=BASE_URL,
                     execution_mode="production")
    state = REPO / "evaluation" / "model_selection" / "state"
    route = TaskRoute(task="grade_primary", backend="ollama", model=model,
                      base_url=BASE_URL, prompt_version=prompt_version,
                      cacheable=False, enabled=True, structured_mode="json_schema",
                      max_tokens=600, temperature=0.0)
    return ModelGateway({"grade_primary": route}, cache=None,
                        ledger=UsageLedger(state / "gateway_ledger" / "usage.jsonl"),
                        budget=None, execution_mode="production")


def _scan_request(text: str) -> None:
    for banned in FORBIDDEN_IN_REQUEST:
        assert banned not in text, f"target leakage into a model request: {banned}"


def _ollama_ps() -> str:
    try:
        return subprocess.run(["ollama", "ps"], capture_output=True, text=True,
                              timeout=15).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"unavailable: {e}"


# ----------------------------------------------------------------- freeze ----

def build_spec() -> dict:
    from autograder.escalation import GRADE_VALIDATION_VERSION, GradeResult, grade_system_for
    from autograder.benchmark.roles import GradeAdapter

    ref = load_reference()
    case_ids = [c["case_id"] for c in ref["cases"]]          # frozen order (sorted)
    assert case_ids == sorted(case_ids) and len(case_ids) == 46
    digests = {}
    for line in subprocess.run(["ollama", "list"], capture_output=True, text=True,
                               timeout=20).stdout.splitlines():
        parts = line.split()
        if parts and parts[0] in (BASELINE_MODEL, ARM_A_MODEL):
            digests[parts[0]] = parts[1]
    assert set(digests) == {BASELINE_MODEL, ARM_A_MODEL}, digests
    git_commit = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=15).stdout.strip()
    verify_schema = json.dumps(_verify_result_model().model_json_schema(), sort_keys=True)
    grade_schema = json.dumps(GradeResult.model_json_schema(), sort_keys=True)
    e23_errors = {"overgrades": ["e002_q1_r5", "e002_q1_r6", "e003_q2_r1", "e003_q2_r6"],
                  "undergrades": ["e002_q1_r7", "e002_q2_r4", "e002_q2_r5",
                                  "e003_q1_r6", "e003_q1_r7"]}
    doc = {
        "experiment": "local_improvement_arms_2026-09-02",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git_commit,
        "population": {"case_ids_in_order": case_ids, "count": 46,
                       "reference_sha256": ref["reference_sha256"],
                       "reference_sources": ref["by_source"],
                       "class_distribution": ref["class_distribution"]},
        "writer_folds": {"e002": 16, "e003": 15, "e004": 14,
                         "e007_reported_separately": 1,
                         "rule_derivation_writers": ["e002", "e003"],
                         "held_writer_for_rule": "e004"},
        "rule_derivation_basis": {
            "note": "combination rule derived ONLY from the e002/e003 baseline errors "
                    "below, before any new output existed; e004 errors were not consulted",
            "e002_e003_baseline_errors": e23_errors},
        "arms": {
            "baseline": {"model": BASELINE_MODEL, "digest": digests[BASELINE_MODEL],
                         "quantization": "Q4_K_M", "outputs": "frozen (44 SEEN-46 + 2 corrected)"},
            "arm_a": {"model": ARM_A_MODEL, "digest": digests[ARM_A_MODEL],
                      "quantization": "Q8_0", "size": "9.8GB",
                      "independence": "NOT independent - same qwen3vl 8.8B base, higher-"
                                      "precision quantization only",
                      "prompt_version": PROMPT_VERSION,
                      "max_local_evaluations": 46},
            "arm_b": {"pass1": "frozen baseline outputs (never re-run)",
                      "pass2_model": BASELINE_MODEL,
                      "pass2_prompt_version": VERIFY_PROMPT_VERSION,
                      "pass2_system_sha256": _sha(VERIFY_SYSTEM),
                      "pass2_schema_sha256": _sha(verify_schema),
                      "pass2_visible": ["question", "rubric", "official_solution",
                                        "frozen transcription", "pass-1 verdict",
                                        "pass-1 rubric_items evidence"],
                      "pass2_forbidden": ["human reference", "instructor score",
                                          "expected verdict", "split", "audit decision",
                                          "historical model outcome"],
                      "pass2_modality": "text-only (verification concerns rubric "
                                        "application; the frozen transcription is the "
                                        "audited reading)",
                      "combination_rule": "ordered rules 0-6, module docstring; "
                                          "combine() in this file",
                      "max_local_evaluations": 46},
        },
        "grade_prompt_sha256": _sha(grade_system_for(PROMPT_VERSION)),
        "grade_schema_sha256": _sha(grade_schema),
        "adapter_version": GradeAdapter.adapter_version,
        "validation_version": GRADE_VALIDATION_VERSION,
        "combination_rule_sha256": rule_source_sha(),
        "backend": {"backend": "ollama", "base_url": BASE_URL, "temperature": 0.0,
                    "max_tokens": 600, "cacheable": False, "rag_policy": "RAG_DISABLED"},
        "leakage_scan_tokens": list(FORBIDDEN_IN_REQUEST),
        "prohibitions": ["no cloud/OpenRouter", "no OCR", "no RAG", "no HELD_OUT",
                         "no 27B/30B", "no prompt change to grade-v4-charitable-local",
                         "no target field in any request", "failures preserved"],
    }
    payload = json.dumps({k: v for k, v in doc.items() if k != "spec_sha256"},
                         ensure_ascii=False, sort_keys=True)
    doc["spec_sha256"] = _sha(payload)
    return doc


def _spec() -> dict:
    doc = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    payload = json.dumps({k: v for k, v in doc.items() if k != "spec_sha256"},
                         ensure_ascii=False, sort_keys=True)
    assert doc["spec_sha256"] == _sha(payload), "spec tampered"
    return doc


# ------------------------------------------------------------------- runs ----

def run_a() -> int:
    from autograder.escalation import GradeResult, grade_system_for
    spec = _spec()
    assert _sha(grade_system_for(PROMPT_VERSION)) == spec["grade_prompt_sha256"]
    m, adapter, by_id, files_root = _bench()
    gw = _gateway(ARM_A_MODEL, PROMPT_VERSION)
    done = set()
    if ARM_A_JSONL.exists():
        done = {json.loads(l)["case_id"] for l in
                ARM_A_JSONL.read_text(encoding="utf-8").splitlines()}
    todo = [cid for cid in spec["population"]["case_ids_in_order"] if cid not in done]
    if len(done) + len(todo) > spec["arms"]["arm_a"]["max_local_evaluations"]:
        print("REFUSED: over budget")
        return 3
    print(f"ARM A ({ARM_A_MODEL}): done {len(done)} | to run {len(todo)}")
    with ARM_A_JSONL.open("a", encoding="utf-8", newline="\n") as out:
        for i, cid in enumerate(todo, 1):
            case = by_id[cid]
            req = adapter.build_request(case.inputs, files_root)
            _scan_request(req.text_for_inspection())
            row = {"case_id": cid, "arm": "arm_a_q8_0", "model": ARM_A_MODEL,
                   "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
            try:
                res = gw.call(task="grade_primary", system=req.system,
                              content_blocks=req.content_blocks, output_model=GradeResult,
                              max_tokens=req.max_tokens,
                              meta={"job_id": "local_improvement_arm_a", "stage": "arm_a",
                                    "exam_id": cid})
                assert res.cache_hit is False
                g = res.value
                row.update(adapter.score(case, g.model_dump(), None))
                row.update({"rubric_items": [ri.model_dump() for ri in g.rubric_items],
                            "latency_s": res.latency_s, "cache_hit": False})
            except Exception as e:  # noqa: BLE001 — preserved, never silently retried
                row.update({"ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}",
                            "decision": "REVIEW", "schema_failure": True})
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")
    (RUNS / "ARM_A_RESOURCES_2026-09-02.txt").write_text(_ollama_ps() + "\n",
                                                         encoding="utf-8")
    print("ARM A complete; ollama ps captured")
    return 0


def run_b() -> int:
    from autograder.benchmark.roles import pack_from_inputs
    from autograder.benchmark.verdicts import verdict_from_model_score
    spec = _spec()
    assert _sha(VERIFY_SYSTEM) == spec["arms"]["arm_b"]["pass2_system_sha256"], \
        "verifier prompt drifted after freeze"
    assert rule_source_sha() == spec["combination_rule_sha256"], \
        "combination rule drifted after freeze"
    VerifyResult = _verify_result_model()
    assert _sha(json.dumps(VerifyResult.model_json_schema(), sort_keys=True)) == \
        spec["arms"]["arm_b"]["pass2_schema_sha256"], "verifier schema drifted"
    m, adapter, by_id, files_root = _bench()
    baseline = load_baseline_outputs()
    rubric_items = load_baseline_rubric_items()
    gw = _gateway(BASELINE_MODEL, VERIFY_PROMPT_VERSION)
    done = set()
    if ARM_B_JSONL.exists():
        done = {json.loads(l)["case_id"] for l in
                ARM_B_JSONL.read_text(encoding="utf-8").splitlines()}
    todo = [cid for cid in spec["population"]["case_ids_in_order"] if cid not in done]
    if len(done) + len(todo) > spec["arms"]["arm_b"]["max_local_evaluations"]:
        print("REFUSED: over budget")
        return 3
    print(f"ARM B verifier ({BASELINE_MODEL}, {VERIFY_PROMPT_VERSION}): "
          f"done {len(done)} | to run {len(todo)}")
    with ARM_B_JSONL.open("a", encoding="utf-8", newline="\n") as out:
        for i, cid in enumerate(todo, 1):
            case = by_id[cid]
            b = baseline[cid]
            p1_verdict = b["predicted_verdict"]
            pack = pack_from_inputs(case.inputs["pack"])
            assert verdict_from_model_score(b["score"], pack.max_score) == p1_verdict
            blocks = verify_blocks(pack, case.inputs["transcription"] or "",
                                   p1_verdict, rubric_items[cid])
            _scan_request(VERIFY_SYSTEM + "\n" + blocks[0]["text"])
            row = {"case_id": cid, "arm": "arm_b_verify", "model": BASELINE_MODEL,
                   "prompt_version": VERIFY_PROMPT_VERSION,
                   "pass1_verdict": p1_verdict, "pass1_decision": b["decision"],
                   "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
            try:
                res = gw.call(task="grade_primary", system=VERIFY_SYSTEM,
                              content_blocks=blocks, output_model=VerifyResult,
                              max_tokens=600,
                              meta={"job_id": "local_improvement_arm_b", "stage": "arm_b",
                                    "exam_id": cid})
                assert res.cache_hit is False
                v = res.value.model_dump()
                usable, why = verifier_usable(v, case.inputs["transcription"] or "")
                row.update({"pass2": v, "pass2_usable": usable, "pass2_unusable_reason":
                            None if usable else why, "latency_s": res.latency_s,
                            "cache_hit": False})
            except Exception as e:  # noqa: BLE001 — preserved
                row.update({"pass2": None, "pass2_usable": False,
                            "pass2_unusable_reason": f"{type(e).__name__}: {str(e)[:300]}"})
            comb = combine(p1_verdict, b["decision"], row.get("pass2"),
                           case.inputs["transcription"] or "")
            row["combined"] = comb
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)}")
    (RUNS / "ARM_B_RESOURCES_2026-09-02.txt").write_text(_ollama_ps() + "\n",
                                                         encoding="utf-8")
    print("ARM B complete; ollama ps captured")
    return 0


# ----------------------------------------------------------------- report ----

def _metrics(pairs: list[tuple[str, str]]) -> dict:
    n = len(pairs)
    agree = sum(1 for a, b in pairs if a == b)
    conf: dict[str, dict[str, int]] = {}
    for a, b in pairs:
        conf.setdefault(a, {}).setdefault(b, 0)
        conf[a][b] += 1
    per_class = {}
    for cls in VERDICTS:
        support = sum(1 for a, _ in pairs if a == cls)
        predicted = sum(1 for _, b in pairs if b == cls)
        tp = sum(1 for a, b in pairs if a == b == cls)
        rec = tp / support if support else None
        prec = tp / predicted if predicted else None
        f1 = (2 * prec * rec / (prec + rec)
              if prec is not None and rec is not None and (prec + rec) else 0.0)
        per_class[cls] = {"support": support, "tp": tp,
                          "recall_exact": f"{tp}/{support}",
                          "recall": round(rec, 4) if rec is not None else None,
                          "precision": round(prec, 4) if prec is not None else None,
                          "f1": round(f1, 4)}
    recalls = [per_class[c]["tp"] / per_class[c]["support"] for c in VERDICTS
               if per_class[c]["support"]]
    f1s = [per_class[c]["f1"] for c in VERDICTS
           if per_class[c]["support"] or any(b == c for _, b in pairs)]
    return {"cases": n, "exact_agreement": agree,
            "exact_agreement_pct": round(100 * agree / n, 1) if n else None,
            "confusion_rows_reference": conf, "per_class": per_class,
            "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
            "balanced_accuracy": round(sum(recalls) / len(recalls), 4) if recalls else None,
            "harmful_overgrades": sum(1 for a, b in pairs if RANK[b] > RANK[a]),
            "harmful_undergrades": sum(1 for a, b in pairs if RANK[b] < RANK[a])}


GATES = {
    "exact_agreement_pct >= 85": lambda s: (s["metrics"]["exact_agreement_pct"] or 0) >= 85,
    "macro_f1 >= 0.80": lambda s: (s["metrics"]["macro_f1"] or 0) >= 0.80,
    "balanced_accuracy >= 0.80": lambda s: (s["metrics"]["balanced_accuracy"] or 0) >= 0.80,
    "harmful_undergrades <= 1/46": lambda s: s["metrics"]["harmful_undergrades"] <= 1,
    "harmful_overgrades <= 3/46": lambda s: s["metrics"]["harmful_overgrades"] <= 3,
    "evidence_or_schema_failure <= 2pct": lambda s: (s["evidence_failures"]
                                                     + s["schema_failures"]) <= 0.02 * 46,
    "prospective_AUTO_coverage >= 85pct": lambda s: s["auto_pct"] >= 85,
}


def report() -> int:
    spec = _spec()
    assert rule_source_sha() == spec["combination_rule_sha256"], \
        "combination rule drifted after freeze"
    ref = load_reference()
    assert ref["reference_sha256"] == spec["population"]["reference_sha256"]
    href = {c["case_id"]: c["final_verdict"] for c in ref["cases"]}
    source = {c["case_id"]: c["reference_source"] for c in ref["cases"]}
    writer = lambda c: c.split("_")[0]                                   # noqa: E731
    question = lambda c: c.split("_")[1]                                 # noqa: E731

    baseline = load_baseline_outputs()
    arm_a = {json.loads(l)["case_id"]: json.loads(l)
             for l in ARM_A_JSONL.read_text(encoding="utf-8").splitlines()}
    arm_b = {json.loads(l)["case_id"]: json.loads(l)
             for l in ARM_B_JSONL.read_text(encoding="utf-8").splitlines()}
    assert set(arm_a) == set(href) == set(arm_b)

    def arm_summary(name, verdicts, decisions, evid_fail, schema_fail, latencies,
                    calls) -> dict:
        # a failed call has NO verdict: it is excluded from pair metrics and
        # reported as a missing output (it always counts as REVIEW)
        pairs_by_case = {c: (href[c], verdicts[c]) for c in href if verdicts.get(c)}
        missing = sorted(c for c in href if not verdicts.get(c))
        s = {"arm": name,
             "metrics": _metrics(list(pairs_by_case.values())),
             "missing_outputs": missing,
             "evidence_failures": evid_fail, "schema_failures": schema_fail,
             "auto": sum(1 for d in decisions.values() if d == "AUTO"),
             "review": sum(1 for d in decisions.values() if d == "REVIEW"),
             "auto_pct": round(100 * sum(1 for d in decisions.values() if d == "AUTO") / 46, 1),
             "latency_s": {"median": (sorted(latencies)[len(latencies) // 2]
                                      if latencies else None),
                           "max": max(latencies) if latencies else None,
                           "total": round(sum(latencies), 1) if latencies else None},
             "new_local_calls": calls,
             "by_writer": {}, "by_question": {}, "by_reference_source": {}}
        for keyname, keyfn in (("by_writer", writer), ("by_question", question),
                               ("by_reference_source", lambda c: source[c])):
            groups: dict[str, list] = {}
            for c, p in pairs_by_case.items():
                groups.setdefault(keyfn(c), []).append(p)
            s[keyname] = {k: {"cases": len(v),
                              "agree": sum(1 for a, b in v if a == b),
                              "agree_pct": round(100 * sum(1 for a, b in v if a == b) / len(v), 1)}
                          for k, v in sorted(groups.items())}
        s["gates"] = {g: bool(fn(s)) for g, fn in GATES.items()}
        s["gates_passed"] = sum(s["gates"].values())
        return s

    base = arm_summary(
        "baseline_8b_one_pass",
        {c: baseline[c]["predicted_verdict"] for c in href},
        {c: baseline[c]["decision"] for c in href},
        sum(1 for b in baseline.values() if b.get("evidence_failure")),
        sum(1 for b in baseline.values() if b.get("schema_failure")),
        [], 0)
    base["latency_s"]["note"] = "frozen outputs; latency reported in their own run artifacts"

    a = arm_summary(
        "arm_a_q8_0",
        {c: arm_a[c].get("predicted_verdict") for c in href},
        {c: arm_a[c].get("decision", "REVIEW") for c in href},
        sum(1 for r in arm_a.values() if r.get("evidence_failure")),
        sum(1 for r in arm_a.values() if r.get("schema_failure")),
        [r["latency_s"] for r in arm_a.values() if r.get("latency_s") is not None],
        sum(1 for r in arm_a.values() if r.get("cache_hit") is False))

    b = arm_summary(
        "arm_b_two_pass",
        {c: arm_b[c]["combined"]["final_verdict"] for c in href},
        {c: arm_b[c]["combined"]["decision"] for c in href},
        sum(1 for c in href if baseline[c].get("evidence_failure")),
        sum(1 for c in href if baseline[c].get("schema_failure")),
        [r["latency_s"] for r in arm_b.values() if r.get("latency_s") is not None],
        sum(1 for r in arm_b.values() if r.get("cache_hit") is False))
    b["combination_rule_usage"] = {}
    for r in arm_b.values():
        rule = r["combined"]["rule"]
        b["combination_rule_usage"][rule] = b["combination_rule_usage"].get(rule, 0) + 1
    b["pass2_unusable"] = sum(1 for r in arm_b.values() if not r.get("pass2_usable"))

    # writer-held analysis: e004 was never consulted for the rule
    held = {}
    for arm_name, verdicts in (
            ("baseline", {c: baseline[c]["predicted_verdict"] for c in href}),
            ("arm_a_q8_0", {c: arm_a[c].get("predicted_verdict") for c in href}),
            ("arm_b_two_pass", {c: arm_b[c]["combined"]["final_verdict"] for c in href})):
        held[arm_name] = {w: _metrics([(href[c], verdicts[c]) for c in href
                                       if writer(c) == w and verdicts.get(c)])
                          for w in ("e002", "e003", "e004", "e007")}

    doc = {
        "artifact": "local_improvement_report",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spec_sha256": spec["spec_sha256"],
        "reference_sha256": ref["reference_sha256"],
        "arms": {"baseline": base, "arm_a": a, "arm_b": b},
        "writer_held_analysis": {
            "note": "the combination rule was derived from e002/e003 baseline errors only; "
                    "e004 is the held writer for ARM B. e007 (1 case) reported separately. "
                    "ALL-SEEN aggregates are DESCRIPTIVE, not unbiased validation - the 46 "
                    "cases are seen development data",
            "per_writer_per_arm": held},
        "confirmations": {"cloud_calls": 0, "ocr_calls": 0, "rag_calls": 0,
                          "held_out_calls_or_exposure": 0,
                          "human_references_changed": 0,
                          "new_local_calls_arm_a": a["new_local_calls"],
                          "new_local_calls_arm_b": b["new_local_calls"]},
    }
    REPORT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                           encoding="utf-8", newline="\n")

    def gate_row(s):
        return " | ".join("PASS" if s["gates"][g] else "fail" for g in GATES)

    md = [f"# Local improvement arms — report ({doc['created_at']})", "",
          "All-seen numbers are DESCRIPTIVE (seen development data), not independent "
          "validation.", "",
          "| arm | exact | macro-F1 | bal.acc | over | under | AUTO% | gates passed |",
          "|---|---|---|---|---|---|---|---|"]
    for s in (base, a, b):
        m_ = s["metrics"]
        md.append(f"| {s['arm']} | {m_['exact_agreement']}/46 = {m_['exact_agreement_pct']}% "
                  f"| {m_['macro_f1']} | {m_['balanced_accuracy']} | "
                  f"{m_['harmful_overgrades']} | {m_['harmful_undergrades']} | "
                  f"{s['auto_pct']}% | {s['gates_passed']}/7 |")
    md += ["", f"Gate columns: {' | '.join(GATES)}", "",
           "| arm | " + " | ".join(GATES) + " |",
           "|---|" + "---|" * len(GATES)]
    for s in (base, a, b):
        md.append(f"| {s['arm']} | " + gate_row(s) + " |")
    REPORT_MD.write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({arm: {"exact": s["metrics"]["exact_agreement_pct"],
                            "macro_f1": s["metrics"]["macro_f1"],
                            "balanced_accuracy": s["metrics"]["balanced_accuracy"],
                            "over": s["metrics"]["harmful_overgrades"],
                            "under": s["metrics"]["harmful_undergrades"],
                            "auto_pct": s["auto_pct"],
                            "gates_passed": s["gates_passed"]}
                      for arm, s in (("baseline", base), ("arm_a", a), ("arm_b", b))},
                     indent=1))
    print("written:", REPORT_JSON.name, REPORT_MD.name)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["freeze", "run-a", "run-b", "report"])
    args = ap.parse_args(argv)
    if args.cmd == "freeze":
        if SPEC_PATH.exists():
            print("REFUSED: spec already frozen")
            return 3
        doc = build_spec()
        SPEC_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")
        print(f"wrote {SPEC_PATH.name}, spec {doc['spec_sha256'][:12]}")
        return 0
    if args.cmd == "run-a":
        return run_a()
    if args.cmd == "run-b":
        return run_b()
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
