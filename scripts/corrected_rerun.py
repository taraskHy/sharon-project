"""Owner-authorized corrected rerun — exactly TWO local calls (2026-09-02).

    python scripts/corrected_rerun.py freeze    # pre-register the spec (no calls)
    python scripts/corrected_rerun.py run       # the two live local evaluations
    python scripts/corrected_rerun.py report    # verify + write the artifact

After the owner-confirmed e004_q2_r6 <-> e004_q2_r8 row transposition was
repaired in the frozen dataset, the two original SEEN-46 model outputs for
those cases became invalid (registered in STALE_MODEL_OUTPUTS_2026-09-01.json;
they are PRESERVED, never deleted or edited). The owner authorized re-running
the production grader on ONLY the two corrected cases:

* candidate qwen3-vl:8b-instruct, prompt grade-v4-charitable-local, adapter
  grade-bench-v3, validation grade-validation-v2 — all UNCHANGED;
* the gateway cache is BYPASSED (cacheable=False, cache=None): the swap probe
  evaluated the same (pack, transcription) combinations as diagnostic flags,
  and a cache hit would silently promote a diagnostic response into an
  official output with zero live calls;
* the other 44 frozen outputs are never re-run; HELD_OUT is structurally
  absent; the route is local Ollama — a cloud call is impossible
  (cloudboundary check + localhost base_url).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

EXPERIMENTS = REPO / "evaluation" / "model_selection" / "experiments"
RUNS = REPO / "evaluation" / "model_selection" / "runs" / "local_grade_primary"
SPEC_PATH = EXPERIMENTS / "CORRECTED_RERUN_2026-09-02.json"
OUT_JSONL = RUNS / "CORRECTED_RERUN_2026-09-02.jsonl"
REPORT_JSON = RUNS / "CORRECTED_RERUN_2026-09-02.json"
REPORT_MD = RUNS / "CORRECTED_RERUN_2026-09-02.md"
CAMPAIGN_PATH = EXPERIMENTS / "LOCAL_GRADE_PRIMARY_SEEN_46_CAMPAIGN_2026-08-28.json"
STALE_REGISTRY = RUNS / "STALE_MODEL_OUTPUTS_2026-09-01.json"

CASES = ("e004_q2_r6", "e004_q2_r8")
CANDIDATE = "qwen3-vl:8b-instruct"
PROMPT_VERSION = "grade-v4-charitable-local"
BASE_URL = "http://localhost:11434/v1"
SEEN46_RUN_DIRS = ("dev__all__qwen3-vl-8b-instruct__72e19378d1",
                   "calibration__all__qwen3-vl-8b-instruct__e2a3cfc925")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def build_spec() -> dict:
    from autograder.benchmark.manifests import load_manifest
    from autograder.benchmark.roles import GradeAdapter
    from autograder.escalation import GRADE_VALIDATION_VERSION, GradeResult, grade_system_for

    m = load_manifest("grade_primary")
    for cid in CASES:
        c = next(c for c in m.cases if c.case_id == cid)
        assert c.split == "CALIBRATION", (cid, c.split)
    camp = json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))
    stale = json.loads(STALE_REGISTRY.read_text(encoding="utf-8"))
    assert stale["reason"] == "invalid_due_to_confirmed_source_transposition"
    seen46_stale = [o for o in stale["affected_outputs"]
                    if o["case_id"] in CASES and o["run_id"] in SEEN46_RUN_DIRS]
    assert len(seen46_stale) == 2, seen46_stale
    system = grade_system_for(PROMPT_VERSION)
    schema = json.dumps(GradeResult.model_json_schema(), sort_keys=True)
    doc = {
        "experiment": "corrected_rerun_2026-09-02",
        "purpose": ("owner-authorized rerun of the production grader on ONLY the two cases "
                    "repaired by the confirmed row transposition; the historical outputs "
                    "stay preserved as invalid_due_to_confirmed_source_transposition"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cases": list(CASES),
        "max_local_evaluations": 2,
        "candidate": CANDIDATE,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": _sha(system),
        "schema_sha256": _sha(schema),
        "adapter_version": GradeAdapter.adapter_version,
        "validation_version": GRADE_VALIDATION_VERSION,
        "dataset_hashes": dict(m.hashes),
        "dataset_revision": "confirmed_row_transposition (owner-confirmed, 2026-09-01)",
        "campaign_sha256": camp["campaign_sha256"],
        "stale_registry_sha256": _sha(STALE_REGISTRY.read_text(encoding="utf-8")),
        "superseded_outputs": seen46_stale,
        "backend": {"backend": "ollama", "base_url": BASE_URL, "temperature": 0.0,
                    "max_tokens": 600, "rag_policy": "RAG_DISABLED",
                    "cacheable": False,
                    "cache_bypass_reason": ("the swap probe cached the same "
                                            "(pack, transcription) combinations as diagnostic "
                                            "flags; official outputs must come from fresh "
                                            "live calls")},
        "prohibitions": ["no other case is evaluated", "no cloud call", "no OCR call",
                         "no RAG retrieval", "no HELD_OUT access",
                         "no change to instructor grades, human reviews, historical model "
                         "outputs, or the prompt"],
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


def run() -> int:
    from autograder.benchmark.manifests import DEFAULT_BENCH_ROOT, load_manifest
    from autograder.benchmark.roles import GradeAdapter
    from autograder.benchmark.runner import files_root_for
    from autograder.cloudboundary import check_cloud_call
    from autograder.escalation import GradeResult, grade_system_for
    from autograder.gateway import ModelGateway, TaskRoute
    from autograder.usage import UsageLedger

    spec = _spec()
    check_cloud_call(task="grade_primary", backend="ollama", base_url=BASE_URL,
                     execution_mode="production")
    m = load_manifest("grade_primary")
    assert dict(m.hashes) == spec["dataset_hashes"], "dataset moved since the spec froze"
    adapter = GradeAdapter("grade_primary", prompt_version=PROMPT_VERSION)
    assert _sha(grade_system_for(PROMPT_VERSION)) == spec["prompt_sha256"], "prompt drifted"
    by_id = {c.case_id: c for c in m.cases}
    files_root = files_root_for(m, DEFAULT_BENCH_ROOT)

    done = set()
    if OUT_JSONL.exists():
        for l in OUT_JSONL.read_text(encoding="utf-8").splitlines():
            done.add(json.loads(l)["case_id"])
    todo = [cid for cid in spec["cases"] if cid not in done]
    if len(done) + len(todo) > spec["max_local_evaluations"]:
        print("REFUSED: would exceed the pre-registered evaluation budget")
        return 3
    print(f"cases {len(spec['cases'])} | done {len(done)} | to run {len(todo)}")

    state = REPO / "evaluation" / "model_selection" / "state"
    route = TaskRoute(task="grade_primary", backend="ollama", model=CANDIDATE,
                      base_url=BASE_URL, prompt_version=PROMPT_VERSION,
                      cacheable=False,                       # NEVER reuse probe responses
                      enabled=True, structured_mode="json_schema", max_tokens=600,
                      temperature=0.0)
    gw = ModelGateway({"grade_primary": route}, cache=None,   # belt and braces
                      ledger=UsageLedger(state / "gateway_ledger" / "usage.jsonl"),
                      budget=None, execution_mode="production")

    n_live = 0
    with OUT_JSONL.open("a", encoding="utf-8", newline="\n") as out:
        for cid in todo:
            case = by_id[cid]
            req = adapter.build_request(case.inputs, files_root)
            text = req.text_for_inspection()
            for k in ("explanation_verdict", "selection_correct", "label_verdict"):
                assert k not in text, f"label leakage: {k}"
            res = gw.call(task="grade_primary", system=req.system,
                          content_blocks=req.content_blocks, output_model=GradeResult,
                          max_tokens=req.max_tokens,
                          meta={"job_id": "corrected_rerun_2026-09-02", "stage": "rerun",
                                "exam_id": cid})
            assert res.cache_hit is False, "a cache hit must be impossible here"
            n_live += 1
            g = res.value
            scored = adapter.score(case, g.model_dump(), None)
            scored.update({"ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "rubric_items": [ri.model_dump() for ri in g.rubric_items],
                           "cache_hit": res.cache_hit, "latency_s": res.latency_s,
                           "model": CANDIDATE, "prompt_version": PROMPT_VERSION,
                           "run_id": "corrected_rerun_2026-09-02",
                           "replaces": {"reason": "invalid_due_to_confirmed_source_transposition",
                                        "registry": STALE_REGISTRY.name}})
            out.write(json.dumps(scored, ensure_ascii=False) + "\n")
            out.flush()
            print(f"  {cid}: score={scored['score']} verdict={scored['predicted_verdict']} "
                  f"decision={scored['decision']} live={not res.cache_hit} "
                  f"latency={res.latency_s}s")
    print(f"run complete: live calls this invocation = {n_live}")
    return 0


def report() -> int:
    spec = _spec()
    rows = [json.loads(l) for l in OUT_JSONL.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and {r["case_id"] for r in rows} == set(CASES), "expected exactly 2 rows"
    assert all(r["cache_hit"] is False for r in rows), "official outputs must be live calls"
    doc = {
        "artifact": "corrected_rerun_report",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spec_sha256": spec["spec_sha256"],
        "outputs": rows,
        "confirmations": {
            "new_local_model_calls": len(rows),
            "cloud_calls": 0, "ocr_calls": 0, "rag_calls": 0,
            "held_out_calls_or_exposure": 0,
            "other_44_cases_rerun": 0,
            "historical_outputs_modified": 0,
        },
        "notes": ["the two SEEN-46 outputs for these cases remain preserved and registered "
                  "invalid_due_to_confirmed_source_transposition; this artifact is the sole "
                  "official source for the corrected model outputs"],
    }
    REPORT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8",
                           newline="\n")
    md = [f"# Corrected rerun — {', '.join(CASES)} ({doc['created_at']})", "",
          f"Production grader ({spec['candidate']}, {spec['prompt_version']}), local Ollama, "
          "cache bypassed, $0 cloud. Exactly two live calls; the historical outputs stay "
          "preserved as invalid.", "",
          "| case | score | verdict | decision | evidence failure | uncertain | latency |",
          "|---|---|---|---|---|---|---|"]
    md += [f"| {r['case_id']} | {r['score']:g} | {r['predicted_verdict']} | {r['decision']} | "
           f"{r['evidence_failure']} | {r['uncertain']} | {r['latency_s']}s |" for r in rows]
    REPORT_MD.write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"outputs": [{k: r[k] for k in ("case_id", "score", "predicted_verdict",
                                                     "decision", "evidence_failure",
                                                     "cache_hit", "latency_s")} for r in rows],
                      "confirmations": doc["confirmations"]}, ensure_ascii=False, indent=1))
    print("written:", REPORT_JSON.name, REPORT_MD.name)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["freeze", "run", "report"])
    args = ap.parse_args(argv)
    if args.cmd == "freeze":
        if SPEC_PATH.exists():
            print("REFUSED: spec already frozen")
            return 3
        doc = build_spec()
        SPEC_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                             encoding="utf-8")
        print(f"wrote {SPEC_PATH.name}: {len(doc['cases'])} cases, "
              f"spec {doc['spec_sha256'][:12]}")
        return 0
    if args.cmd == "run":
        return run()
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
