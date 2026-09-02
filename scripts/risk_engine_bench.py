"""Deterministic performance benchmark of the risk engine — NO inference.

    python scripts/risk_engine_bench.py

Measures pure policy-evaluation latency/throughput, shadow-event size, log
growth and admin-aggregation time over synthetic replicated inputs at 46 /
100 / 1,000 / 10,000 cases. Synthetic inputs replicate the STRUCTURE of the
frozen SEEN-46 rows (verdict mix and structural-flag mix); no benchmark case
id, no reference verdict, and no model call is involved.

Writes RISK_ENGINE_BENCH_<date>.{json,md}. Full-exam extrapolation is
explicitly labelled as EXCLUDING OCR and model-inference latency, which this
benchmark cannot and does not measure.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autograder import riskengine  # noqa: E402

RUNS = REPO / "evaluation" / "model_selection" / "runs" / "local_grade_primary"
SIZES = (46, 100, 1_000, 10_000)
POLICY = "prospective_noninvalid_v1"

# structural mix mirroring the seen data: mostly clean, a few flagged
_VARIANTS = (
    ({"semantic_verdict": "valid"}, 24),
    ({"semantic_verdict": "partially_valid"}, 12),
    ({"semantic_verdict": "invalid"}, 6),
    ({"semantic_verdict": "valid", "evidence_ok": False,
      "validation_ok": False}, 2),
    ({"semantic_verdict": "partially_valid", "uncertain": True}, 1),
    ({"semantic_verdict": "valid", "transcription_complete": False}, 1),
)


def make_inputs(n: int) -> list[riskengine.ProspectiveDecisionInput]:
    base = {"semantic_verdict": "valid", "schema_ok": True, "evidence_ok": True,
            "validation_ok": True, "uncertain": False,
            "transcription_complete": True, "source_integrity": "current",
            "model_output_current": True, "local_grader_available": True,
            "model_digest": "bench-digest", "prompt_version":
            "grade-v4-charitable-local", "prompt_sha256": "bench-prompt",
            "schema_sha256": "bench-schema",
            "validation_version": "grade-validation-v2"}
    pool = []
    for patch, weight in _VARIANTS:
        pool.extend([patch] * weight)
    out = []
    for i in range(n):
        out.append(riskengine.ProspectiveDecisionInput.from_mapping(
            {**base, **pool[i % len(pool)]}))
    return out


def bench_size(n: int, tmp: Path) -> dict:
    inputs = make_inputs(n)
    eng = riskengine.build_engine(mode="shadow", policy_id=POLICY)
    now = "2026-09-02 00:00:00"

    tracemalloc.start()
    latencies = []
    t0 = time.perf_counter()
    decisions = []
    for d in inputs:
        s = time.perf_counter()
        decisions.append(eng.decide(d, now=now))
        latencies.append(time.perf_counter() - s)
    wall = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    log_path = tmp / f"shadow_{n}.jsonl"
    log = riskengine.ShadowLog(log_path)
    t0 = time.perf_counter()
    for i, (d, dec) in enumerate(zip(inputs, decisions)):
        log.append(riskengine.build_shadow_event(f"synthetic_{i:06d}", "bench",
                                                 d, dec, {"offline_only": True}))
    write_wall = time.perf_counter() - t0
    log_bytes = log_path.stat().st_size

    t0 = time.perf_counter()
    events = riskengine.ShadowLog(log_path).events()
    agg = {}
    for ev in events:
        key = (ev["decision"]["action"], ev["decision"]["reason"])
        agg[key] = agg.get(key, 0) + 1
    agg_wall = time.perf_counter() - t0
    assert sum(agg.values()) == n

    lat_sorted = sorted(latencies)
    return {
        "cases": n,
        "decide_mean_us": round(1e6 * statistics.mean(latencies), 2),
        "decide_p95_us": round(1e6 * lat_sorted[int(0.95 * (n - 1))], 2),
        "decide_throughput_per_s": round(n / wall, 1) if wall else None,
        "decide_wall_s": round(wall, 4),
        "peak_tracemalloc_mb": round(peak / 1e6, 3),
        "shadow_write_wall_s": round(write_wall, 4),
        "shadow_log_bytes": log_bytes,
        "mean_event_bytes": round(log_bytes / n, 1),
        "admin_aggregation_wall_s": round(agg_wall, 4),
        "aggregated_events": sum(agg.values()),
    }


def main() -> int:
    today = time.strftime("%Y-%m-%d")
    out_json = RUNS / f"RISK_ENGINE_BENCH_{today}.json"
    out_md = RUNS / f"RISK_ENGINE_BENCH_{today}.md"
    results = []
    with tempfile.TemporaryDirectory() as td:
        for n in SIZES:
            results.append(bench_size(n, Path(td)))
            print(f"  {n}: {results[-1]['decide_throughput_per_s']}/s, "
                  f"event {results[-1]['mean_event_bytes']}B")
    r46 = results[0]
    per_case_s = r46["decide_wall_s"] / 46 + r46["shadow_write_wall_s"] / 46
    doc = {
        "artifact": "risk_engine_bench",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": subprocess.run(["git", "-C", str(REPO), "rev-parse",
                                      "HEAD"], capture_output=True, text=True,
                                     timeout=15).stdout.strip(),
        "engine_version": riskengine.RISK_ENGINE_VERSION,
        "policy": POLICY,
        "results": results,
        "projection": {
            "per_100_explanation_cases_s": round(100 * per_case_s, 4),
            "per_100_full_exams_note":
                "risk-layer overhead only. A full exam contains OCR and "
                "local-model inference whose latency this benchmark does NOT "
                "measure; no full-exam automation or latency claim is made "
                "from explanation-only data.",
            "per_100_full_exams_risk_layer_only_s":
                round(100 * 2 * per_case_s, 4),
            "full_exam_assumption": "~2 explanation cases per exam in the "
                                    "seen data (2 questions/writer)",
        },
        "no_inference": {"model_calls": 0, "ocr_calls": 0, "cloud_calls": 0},
    }
    payload = json.dumps({k: v for k, v in doc.items() if k != "content_sha256"},
                         ensure_ascii=False, sort_keys=True)
    doc["content_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    out_json.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n",
                        encoding="utf-8", newline="\n")
    md = [f"# Risk-engine performance bench ({doc['created_at']})", "",
          f"Engine `{doc['engine_version']}`, policy `{POLICY}`, synthetic "
          "structurally-replicated inputs; zero inference.", "",
          "| cases | mean decide | p95 | throughput | peak mem | event size | "
          "log bytes | admin agg |", "|---|---|---|---|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['cases']} | {r['decide_mean_us']}µs | "
                  f"{r['decide_p95_us']}µs | {r['decide_throughput_per_s']}/s "
                  f"| {r['peak_tracemalloc_mb']}MB | {r['mean_event_bytes']}B "
                  f"| {r['shadow_log_bytes']} | "
                  f"{r['admin_aggregation_wall_s']}s |")
    md += ["", f"Projected risk-layer overhead per 100 explanation cases: "
           f"**{doc['projection']['per_100_explanation_cases_s']}s**. "
           "Full-exam figures cover the risk layer ONLY — OCR and model "
           "inference latency are not measured here and no full-exam "
           "automation claim is made."]
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8", newline="\n")
    print("written:", out_json.name, out_md.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
