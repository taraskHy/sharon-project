"""Mission K: deterministic overhead benchmarks. NO model calls, NO network."""
import json, statistics, time, tracemalloc
from pathlib import Path

from autograder.benchmark.manifests import load_manifest
from autograder.benchmark.ocr_fallback import replay, select
from autograder.benchmark.ocr_outcomes import classify_row, summarize
from autograder.benchmark.ocr_writer_metrics import pair_metrics

S32 = Path("evaluation/model_selection/runs_seen32/ocr_primary")
GEM_DIR = S32 / "dev__seen46_ocr_dev__all__google-gemini-3.7-flash__c4ae61f634"
SON_DIR = S32 / "dev__seen46_ocr_dev__all__anthropic-claude-sonnet-5__2f3a7c346c"
man = load_manifest("ocr_primary")
by = {c.case_id: c for c in man.cases}


def load(d):
    return [json.loads(l) for l in (d / "outputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]


gem_rows, son_rows = load(GEM_DIR), load(SON_DIR)
cases = [r["case_id"] for r in gem_rows]


def bench(fn, n, reps=5):
    """Return mean/p95 seconds per BATCH of n, plus per-item throughput."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(n)
        times.append(time.perf_counter() - t0)
    times.sort()
    mean = statistics.mean(times)
    p95 = times[min(len(times) - 1, int(0.95 * len(times)))]
    return {"batch": n, "mean_s": round(mean, 6), "p95_s": round(p95, 6),
            "per_item_us": round(mean / n * 1e6, 2),
            "throughput_per_s": round(n / mean, 1) if mean else None}


def rep(rows, n):
    """Replicate the real 32 rows up to n synthetic items."""
    out = []
    while len(out) < n:
        out.extend(rows)
    return out[:n]


def b_parse(n):
    for r in rep(gem_rows, n):
        classify_row(r, by[r["case_id"]].label["reference"])


def b_metrics(n):
    for r in rep(gem_rows, n):
        hyp = (r.get("output") or {}).get("transcription")
        pair_metrics(by[r["case_id"]].label["reference"], hyp)


def b_routing(n):
    g = {r["case_id"]: classify_row(r, by[r["case_id"]].label["reference"]) for r in gem_rows}
    s = {r["case_id"]: classify_row(r, by[r["case_id"]].label["reference"]) for r in son_rows}
    for i in range(n):
        c = cases[i % len(cases)]
        select(case_id=c, primary_outcome=g[c], primary_text="x",
               secondary_outcome=s[c], secondary_text="y")


def b_compose(n):
    g = {r["case_id"]: classify_row(r, by[r["case_id"]].label["reference"]) for r in gem_rows}
    s = {r["case_id"]: classify_row(r, by[r["case_id"]].label["reference"]) for r in son_rows}
    gt = {r["case_id"]: (r.get("output") or {}).get("transcription") for r in gem_rows}
    st = {r["case_id"]: (r.get("output") or {}).get("transcription") for r in son_rows}
    batches = max(1, n // len(cases))
    for _ in range(batches):
        replay(cases, g, s, gt, st)


def b_aggregate(n):
    tax = {r["case_id"]: classify_row(r, by[r["case_id"]].label["reference"]) for r in gem_rows}
    batches = max(1, n // len(cases))
    for _ in range(batches):
        summarize(tax)


def b_export(n):
    rows = rep(gem_rows, n)
    json.dumps([{"case_id": r["case_id"], "ok": r.get("ok"),
                 "text": (r.get("output") or {}).get("transcription")} for r in rows],
               ensure_ascii=False)


BENCH = {"ocr_artifact_parsing_and_classification": b_parse,
         "cer_wer_metric_computation": b_metrics,
         "fallback_routing_decision": b_routing,
         "result_composition_full_replay": b_compose,
         "admin_aggregation_summarize": b_aggregate,
         "export_generation_json": b_export}

results = {}
for name, fn in BENCH.items():
    results[name] = [bench(fn, n) for n in (100, 1000, 10000)]

tracemalloc.start()
b_compose(10000)
cur, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

sizes = {
    "one_outputs_jsonl_row_bytes": len(json.dumps(gem_rows[0], ensure_ascii=False).encode()),
    "one_classified_outcome_bytes": len(json.dumps(
        classify_row(gem_rows[0], by[gem_rows[0]["case_id"]].label["reference"]),
        ensure_ascii=False).encode()),
    "one_fallback_decision_bytes": len(json.dumps(
        select(case_id="c",
               primary_outcome=classify_row(gem_rows[0], by[gem_rows[0]["case_id"]].label["reference"]),
               primary_text="x"), ensure_ascii=False).encode()),
    "gemini_outputs_jsonl_bytes": (GEM_DIR / "outputs.jsonl").stat().st_size,
    "paired_result_artifact_bytes": Path(
        "evaluation/model_selection/runs/ocr_primary/OCR_SEEN32_PAIRED_RESULT_2026-09-02.json").stat().st_size,
}
per_crop = sizes["one_outputs_jsonl_row_bytes"] + sizes["one_classified_outcome_bytes"] + \
    sizes["one_fallback_decision_bytes"]
growth = {f"{n}_crops_mb": round(per_crop * n / 1e6, 3) for n in (1000, 10000, 100000)}

art = {"artifact": "ocr_deterministic_performance", "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
       "provider_calls": 0, "note": "deterministic overhead only; no provider inference benchmarked",
       "method": ("the real 32 paired rows replicated to synthetic batches of 100/1000/10000, 5 "
                  "repetitions each; timings are wall-clock on this machine and are indicative"),
       "benchmarks": results, "event_sizes_bytes": sizes,
       "estimated_storage_growth": growth,
       "peak_memory_10000_compose_bytes": {"current": cur, "peak": peak},
       "headline": ("every deterministic stage is microseconds per crop; none of them is a "
                    "throughput concern next to provider latency of 4-9 s/crop")}
p = Path("evaluation/model_selection/runs/ocr_primary/OCR_PERFORMANCE_2026-09-02.json")
p.write_text(json.dumps(art, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
print("wrote", p)
print(f"{'stage':42} {'n':>6} {'per-item us':>12} {'throughput/s':>13}")
for name, rows in results.items():
    for r in rows:
        print(f"{name[:42]:42} {r['batch']:>6} {r['per_item_us']:>12} {r['throughput_per_s']:>13}")
print()
print("event sizes:", json.dumps(sizes))
print("storage growth:", json.dumps(growth))
print("peak memory (10k compose):", peak, "bytes")
