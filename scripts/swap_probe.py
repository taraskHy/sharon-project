"""Neighbor-fit swap probe — does an answer fit a SIBLING sub-item's slot?

    python scripts/swap_probe.py freeze    # pre-register the pair plan (no calls)
    python scripts/swap_probe.py run       # execute the local cross-matrix
    python scripts/swap_probe.py report    # analyze -> SWAP_PROBE artifacts

A transposed answer (written in the wrong row) is invisible to the per-cell
grader: it grades one (pack, transcription) pair and zeroes off-topic text
without ever knowing the text belongs one row over. This probe measures the
counterfactual directly: every SEEN answer is graded by the SAME production
grader (grade-v4-charitable-local, same schema, temp 0, local Ollama) against
every sibling sub-item of the same writer+question. Outputs are DIAGNOSTIC
FLAGS only:

* a suspected swap NEVER credits the neighbor slot and NEVER changes any
  grade, label, frozen run, or review-campaign artifact;
* the classic transposition signature is RECIPROCAL fit (A fits B's slot AND
  B fits A's) combined with weak own-fit;
* the same matrix yields the detector's NOISE FLOOR (how often a non-swapped
  answer fits a sibling anyway) — the number that decides whether a
  production SWAP_SUSPECT->REVIEW hook is viable at all.

Population: the frozen SEEN-46 campaign only (DEV+CALIBRATION). HELD_OUT is
structurally absent. Own-slot verdicts are NOT re-run — they are read from
the frozen SEEN-46 model outputs. Cloud calls: impossible (local route;
production boundary).
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

SPEC_PATH = REPO / "evaluation" / "model_selection" / "experiments" / "SWAP_PROBE_2026-08-30.json"
RUNS = REPO / "evaluation" / "model_selection" / "runs" / "local_grade_primary"
OUT_JSONL = RUNS / "SWAP_PROBE_2026-08-30.jsonl"
REPORT_JSON = RUNS / "SWAP_PROBE_2026-08-30.json"
REPORT_MD = RUNS / "SWAP_PROBE_2026-08-30.md"
CAMPAIGN_PATH = REPO / "evaluation" / "model_selection" / "experiments" / \
    "LOCAL_GRADE_PRIMARY_SEEN_46_CAMPAIGN_2026-08-28.json"

CANDIDATE = "qwen3-vl:8b-instruct"
PROMPT_VERSION = "grade-v4-charitable-local"
BASE_URL = "http://localhost:11434/v1"
FORBIDDEN_WRITERS = ("e005", "e006")


def _campaign() -> dict:
    return json.loads(CAMPAIGN_PATH.read_text(encoding="utf-8"))


def _groups(case_ids: list[str]) -> dict[tuple[str, str], list[str]]:
    g: dict[tuple[str, str], list[str]] = {}
    for cid in case_ids:
        w, q, _ = cid.split("_")
        g.setdefault((w, q), []).append(cid)
    return g


def build_spec() -> dict:
    from autograder.benchmark.roles import GradeAdapter
    from autograder.escalation import GRADE_VALIDATION_VERSION, GradeResult, grade_system_for

    camp = _campaign()
    case_ids = [c["case_id"] for c in camp["cases"]]
    for cid in case_ids:
        assert cid.split("_")[0] not in FORBIDDEN_WRITERS, cid
    pairs = []
    for (_, _), members in sorted(_groups(case_ids).items()):
        for target in members:                 # the SLOT (pack) being tested
            for source in members:             # the TEXT tried in that slot
                if source != target:
                    pairs.append({"target_pack": target, "source_text": source})
    system = grade_system_for(PROMPT_VERSION)
    schema = json.dumps(GradeResult.model_json_schema(), sort_keys=True)
    doc = {
        "probe": "swap_probe_2026-08-30",
        "purpose": ("diagnostic only: measure neighbor-slot fit of every seen answer with "
                    "the production grader. Flags never credit a neighbor slot, never alter "
                    "grades/labels/frozen runs/review data; a production SWAP_SUSPECT->REVIEW "
                    "hook is a separate, owner-gated decision informed by this probe"),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "campaign_sha256": camp["campaign_sha256"],
        "candidate": CANDIDATE,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": hashlib.sha256(system.encode()).hexdigest(),
        "schema_sha256": hashlib.sha256(schema.encode()).hexdigest(),
        "adapter_version": GradeAdapter.adapter_version,
        "validation_version": GRADE_VALIDATION_VERSION,
        "backend": {"backend": "ollama", "base_url": BASE_URL, "temperature": 0.0,
                    "max_tokens": 600, "rag_policy": "RAG_DISABLED"},
        "population": {"cases": len(case_ids), "held_out": 0,
                       "groups": {f"{w}_{q}": len(m) for (w, q), m in sorted(_groups(case_ids).items())}},
        "cross_pairs": len(pairs),
        "own_slot_source": ("frozen SEEN-46 run outputs (never re-run): "
                            "dev__all__qwen3-vl-8b-instruct__72e19378d1 + "
                            "calibration__all__qwen3-vl-8b-instruct__e2a3cfc925"),
        "max_local_evaluations": len(pairs),
        "pairs": pairs,
    }
    payload = json.dumps({k: v for k, v in doc.items() if k != "spec_sha256"},
                         ensure_ascii=False, sort_keys=True)
    doc["spec_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    return doc


def _spec() -> dict:
    doc = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    payload = json.dumps({k: v for k, v in doc.items() if k != "spec_sha256"},
                         ensure_ascii=False, sort_keys=True)
    assert doc["spec_sha256"] == hashlib.sha256(payload.encode()).hexdigest(), "spec tampered"
    return doc


def _adapter_and_cases():
    from autograder.benchmark.manifests import DEFAULT_BENCH_ROOT, load_manifest
    from autograder.benchmark.roles import GradeAdapter
    from autograder.benchmark.runner import files_root_for
    m = load_manifest("grade_primary")
    adapter = GradeAdapter("grade_primary", prompt_version=PROMPT_VERSION)
    return adapter, {c.case_id: c for c in m.cases}, files_root_for(m, DEFAULT_BENCH_ROOT)


def run() -> int:
    from autograder.cloudboundary import check_cloud_call
    from autograder.escalation import GradeResult
    from autograder.gateway import ModelGateway, TaskRoute
    from autograder.requestcache import RequestCache
    from autograder.usage import UsageLedger

    spec = _spec()
    check_cloud_call(task="grade_primary", backend="ollama", base_url=BASE_URL,
                     execution_mode="production")     # local-only sanity, refuses remote
    adapter, by_id, files_root = _adapter_and_cases()
    state = REPO / "evaluation" / "model_selection" / "state"
    route = TaskRoute(task="grade_primary", backend="ollama", model=CANDIDATE,
                      base_url=BASE_URL, prompt_version=PROMPT_VERSION, cacheable=True,
                      enabled=True, structured_mode="json_schema", max_tokens=600,
                      temperature=0.0)
    gw = ModelGateway({"grade_primary": route}, cache=RequestCache(state / "gateway_cache"),
                      ledger=UsageLedger(state / "gateway_ledger" / "usage.jsonl"),
                      budget=None, execution_mode="production")

    done = set()
    if OUT_JSONL.exists():
        for l in OUT_JSONL.read_text(encoding="utf-8").splitlines():
            r = json.loads(l)
            done.add((r["target_pack"], r["source_text"]))
    todo = [p for p in spec["pairs"] if (p["target_pack"], p["source_text"]) not in done]
    print(f"pairs total {len(spec['pairs'])} | done {len(done)} | to run {len(todo)}")
    n_ok = n_fail = 0
    with OUT_JSONL.open("a", encoding="utf-8", newline="\n") as out:
        for i, p in enumerate(todo, 1):
            target, source = by_id[p["target_pack"]], by_id[p["source_text"]]
            inputs = {"case_id": f"swap::{p['target_pack']}<-{p['source_text']}",
                      "pack": target.inputs["pack"], "selected": None,
                      "transcription": source.inputs["transcription"], "version": None}
            req = adapter.build_request(inputs, files_root)
            # structural leakage guarantee: build_request never sees any label;
            # belt-and-braces scan for the label vocabulary of BOTH cases
            text = req.text_for_inspection()
            for lab in (target.label, source.label):
                for k in ("explanation_verdict", "selection_correct"):
                    assert str(k) not in text
            row = {"target_pack": p["target_pack"], "source_text": p["source_text"],
                   "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
            try:
                res = gw.call(task="grade_primary", system=req.system,
                              content_blocks=req.content_blocks, output_model=GradeResult,
                              max_tokens=req.max_tokens,
                              meta={"job_id": "swap_probe_2026-08-30", "stage": "diagnostic",
                                    "exam_id": row["target_pack"]})
                g = res.value
                row.update({"ok": True, "score": g.score,
                            "rubric_items": [ri.model_dump() for ri in g.rubric_items],
                            "uncertain": g.uncertain, "latency_s": res.latency_s,
                            "cache_hit": res.cache_hit})
                n_ok += 1
            except Exception as e:  # noqa: BLE001 — preserved, never retried silently
                row.update({"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"})
                n_fail += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} (ok {n_ok}, failed {n_fail})")
    print(f"run complete: ok {n_ok}, failed {n_fail}")
    return 0 if n_fail == 0 else 1


def report() -> int:
    from autograder.benchmark.verdicts import verdict_from_model_score

    spec = _spec()
    camp = _campaign()
    by_case = {c["case_id"]: c for c in camp["cases"]}
    rows = [json.loads(l) for l in OUT_JSONL.read_text(encoding="utf-8").splitlines()]
    cross: dict[tuple[str, str], str] = {}
    for r in rows:
        if r.get("ok"):
            cross[(r["target_pack"], r["source_text"])] = verdict_from_model_score(r["score"], 4.0)

    # own-slot verdicts from the FROZEN seen-46 runs
    own: dict[str, str] = {}
    for d in ("dev__all__qwen3-vl-8b-instruct__72e19378d1",
              "calibration__all__qwen3-vl-8b-instruct__e2a3cfc925"):
        for s in json.loads((RUNS / "grade_primary" / d / "scored.jsonl.json").read_text(encoding="utf-8")):
            own[s["case_id"]] = s["predicted_verdict"]

    fits = lambda v: v in ("valid", "partially_valid")               # noqa: E731
    case_ids = [c["case_id"] for c in camp["cases"]]
    groups = _groups(case_ids)
    per_case, suspects, reciprocal = [], [], []
    noise_pairs = fit_pairs = 0
    for (_, _), members in sorted(groups.items()):
        for t in members:
            sib = {s: cross.get((t, s)) for s in members if s != t}
            fit_sibs = sorted(s for s, v in sib.items() if v and fits(v))
            noise_pairs += len([v for v in sib.values() if v])
            fit_pairs += len(fit_sibs)
            row = {"case": t, "own_verdict": own.get(t), "siblings": sib,
                   "siblings_fitting_this_slot": fit_sibs}
            per_case.append(row)
            if own.get(t) == "invalid" and fit_sibs:
                suspects.append({"slot": t, "own_verdict": "invalid",
                                 "texts_fitting_this_slot": fit_sibs,
                                 "instructor_score_of_slot": by_case[t]["actual_instructor_score"]})
    for s in suspects:
        for other in s["texts_fitting_this_slot"]:
            if any(s2["slot"] == other and s["slot"] in s2["texts_fitting_this_slot"]
                   for s2 in suspects):
                pair = tuple(sorted((s["slot"], other)))
                if pair not in [tuple(sorted((a, b))) for a, b in reciprocal]:
                    reciprocal.append(pair)

    doc = {
        "artifact": "swap_probe_report",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "spec_sha256": spec["spec_sha256"],
        "pairs_evaluated": len(cross),
        "pairs_failed": sum(1 for r in rows if not r.get("ok")),
        "noise_floor": {
            "cross_pairs_judged": noise_pairs,
            "cross_pairs_fitting_a_foreign_slot": fit_pairs,
            "foreign_fit_rate_pct": round(100 * fit_pairs / noise_pairs, 1) if noise_pairs else None,
            "meaning": ("how often the grader accepts an answer in a slot it does NOT belong "
                        "to; the false-positive ceiling for any swap detector built on it"),
        },
        "swap_suspects": suspects,
        "reciprocal_transposition_signatures": [list(p) for p in reciprocal],
        "policy": ("flags only — nothing here credits a neighbor slot or changes any grade, "
                   "frozen run, label, or review-campaign record"),
        "per_case": per_case,
    }
    REPORT_JSON.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8",
                           newline="\n")
    md = [f"# Swap probe — neighbor-slot fit ({doc['created_at']})", "",
          f"{len(cross)} cross-pairs judged by the production grader "
          f"({spec['candidate']}, {spec['prompt_version']}), local only, $0. "
          "Diagnostic flags only; nothing was credited or changed.", "",
          f"- foreign-slot fit rate (noise floor): **{doc['noise_floor']['foreign_fit_rate_pct']}%** "
          f"({fit_pairs}/{noise_pairs})",
          f"- swap suspects (own slot invalid AND a sibling text fits): **{len(suspects)}**",
          f"- reciprocal transposition signatures: **{len(reciprocal)}**", ""]
    if suspects:
        md += ["| slot | instructor score | sibling texts fitting it |", "|---|---|---|"]
        md += [f"| {s['slot']} | {s['instructor_score_of_slot']:g} | "
               f"{', '.join(s['texts_fitting_this_slot'])} |" for s in suspects]
    REPORT_MD.write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({k: doc[k] for k in ("pairs_evaluated", "pairs_failed", "noise_floor",
                                          "swap_suspects", "reciprocal_transposition_signatures")},
                     ensure_ascii=False, indent=1))
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
        SPEC_PATH.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {SPEC_PATH.name}: {doc['cross_pairs']} cross-pairs, "
              f"spec {doc['spec_sha256'][:12]}")
        return 0
    if args.cmd == "run":
        return run()
    return report()


if __name__ == "__main__":
    raise SystemExit(main())
