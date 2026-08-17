"""Local Qwen MC-resolver benchmark on the historically ambiguous rows.

Only rows the deterministic extractor left AMBIGUOUS in the audited prob
batch are sent (never confident rows). Row-band crops are regenerated with
the same tablecrop path the extractor uses; the resolver receives ONLY the
band crop + deterministic candidate letters. Audited answers are joined
strictly AFTER all resolver outputs are persisted. Zero cloud calls
(cloud stage disabled). May load the 27B local model (lazily); Ollama
idle-unloads afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autograder.gateway import ModelGateway  # noqa: E402
from autograder.ingest import load_pages  # noqa: E402
from autograder.mcresolve import resolve_row  # noqa: E402
from autograder.requestcache import RequestCache  # noqa: E402
from autograder.template import load_template  # noqa: E402
from autograder.usage import UsageLedger  # noqa: E402


def ambiguous_rows(job: Path) -> list[dict]:
    rows = []
    for ep in sorted(job.glob("exams/exam-*/extraction.json")):
        exam = ep.parent.name
        e = json.loads(ep.read_text(encoding="utf-8"))
        for q in e["questions"]:
            for s in q["sub_items"]:
                if s["status"] == "ambiguous":
                    rows.append({"exam": exam, "question_id": q["question_id"], "sub_item_id": s["sub_item_id"],
                                 "candidates": list(s.get("candidate_answers") or [])})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default="jobs/job-20260809-042546")
    ap.add_argument("--models-config", required=True)
    ap.add_argument("--out", default="evaluation/mc_fallback/qwen38_local_v1")
    ap.add_argument("--max-image-edge", type=int, default=1400)
    args = ap.parse_args()
    job = REPO / args.job
    out = REPO / args.out
    (out / "rows").mkdir(parents=True, exist_ok=True)

    from autograder.tablecrop import answer_table_row_bands

    key = json.loads((job / "uploads" / "answer_key.json").read_text(encoding="utf-8"))
    template = load_template(job / "uploads" / "answer_key.json")
    letters = list(template.answer_table_columns_rtl)
    n_rows_by_q = {q["id"]: len(q["sub_items"]) for q in key["questions"]}
    page_number = template.answer_sheet_pages[0]

    gw = ModelGateway.from_file(args.models_config, cache=RequestCache(out / "cache"),
                                ledger=UsageLedger(out / "ledger.jsonl"))
    rows = ambiguous_rows(job)
    print(f"ambiguous rows (deterministic unresolved): {len(rows)}")
    (out / "rows_manifest.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")

    bands_cache: dict[str, dict] = {}
    t_all = time.monotonic()
    for i, r in enumerate(rows, 1):
        target = out / "rows" / f"{r['exam']}_q{r['question_id']}_r{r['sub_item_id']}.json"
        if target.exists():
            continue
        if r["exam"] not in bands_cache:
            pdf = job / "uploads" / "exams" / f"{r['exam']}.pdf"
            pages = load_pages(pdf, args.max_image_edge)
            sheet = next(p for p in pages if p.page_number == page_number)
            bands_cache[r["exam"]] = {qid: answer_table_row_bands(sheet, n_rows=n) for qid, n in n_rows_by_q.items()}
        band = bands_cache[r["exam"]][r["question_id"]][int(r["sub_item_id"]) - 1]
        t0 = time.monotonic()
        res, trace = resolve_row(band_png=band.png_bytes, letters=letters, candidates=r["candidates"],
                                 gateway=gw, allow_cloud=False,
                                 meta={"job_id": job.name, "exam_id": r["exam"], "question_id": r["question_id"],
                                       "stage": "mc_bench"})
        rec = {**r, "resolution": {"selected": res.selected, "state": res.state, "confidence": res.confidence,
                                   "source": res.source}, "trace": trace.stages,
               "latency_s": round(time.monotonic() - t0, 2)}
        target.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[{i}/{len(rows)}] {r['exam']} q{r['question_id']} r{r['sub_item_id']} cands={r['candidates']} "
              f"-> {res.selected} ({res.state}, {res.source}) {rec['latency_s']}s")
    print(f"resolver pass complete in {time.monotonic()-t_all:.0f}s — joining audited answers now")

    # ---- post-hoc join (only after all rows persisted) ----
    audit = json.loads((REPO / "evaluation" / "prob" / "manual_audit.json").read_text(encoding="utf-8"))
    name_map = json.loads((job / "uploads" / "name_map.json").read_text(encoding="utf-8"))
    # audit is a list keyed by source_index (original file stem, e.g. "02")
    by_src = {e["source_index"]: e for e in audit["exams"]}
    stats = {"n_ambiguous": len(rows), "resolved": 0, "correct": 0, "incorrect": 0, "agreement": 0,
             "disagreement": 0, "unclear": 0, "no_reference": 0}
    details = []
    for r in rows:
        rec = json.loads((out / "rows" / f"{r['exam']}_q{r['question_id']}_r{r['sub_item_id']}.json").read_text(encoding="utf-8"))
        sel = rec["resolution"]["selected"]
        orig = Path(name_map.get(r["exam"], "")).stem
        ex_audit = by_src.get(orig)
        ref = None
        if ex_audit:
            ans = ex_audit.get("answers") or {}
            ref = ans.get(r["sub_item_id"]) if isinstance(ans, dict) else None
            if ref in ("", "—", None):
                ref = None
        if ref is None:
            stats["no_reference"] += 1
        if sel is None:
            stats["unclear"] += 1
        else:
            stats["resolved"] += 1
            if ref is not None:
                if str(sel).upper() == str(ref).upper():
                    stats["correct"] += 1
                else:
                    stats["incorrect"] += 1
            # CV/Qwen agreement = Qwen picked one of the deterministic candidates
            if sel in r["candidates"]:
                stats["agreement"] += 1
            else:
                stats["disagreement"] += 1
        details.append({**r, "qwen": sel, "audited": ref, "state": rec["resolution"]["state"],
                        "source": rec["resolution"]["source"], "latency_s": rec["latency_s"]})
    n = stats["n_ambiguous"] or 1
    stats["accuracy_on_resolved_with_reference"] = (round(stats["correct"] / (stats["correct"] + stats["incorrect"]), 3)
                                                    if (stats["correct"] + stats["incorrect"]) else None)
    stats["cloud_escalation_rate_pct"] = round(100 * stats["unclear"] / n, 1)   # unresolved rows would go to cloud
    stats["human_review_rate_pct_if_no_cloud"] = round(100 * stats["unclear"] / n, 1)
    (out / "summary.json").write_text(json.dumps({"stats": stats, "rows": details}, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    print(json.dumps(stats, indent=1))
    for d in details:
        print(f"  {d['exam']} q{d['question_id']} r{d['sub_item_id']}: cands={d['candidates']} qwen={d['qwen']} "
              f"audited={d['audited']} state={d['state']} {d['latency_s']}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
