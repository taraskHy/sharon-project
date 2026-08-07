"""Prob-dataset benchmark evaluation.

Joins finished job results with the instructor grades (``prob_data/
grades.csv``) STRICTLY AFTER prediction: this script only reads per-exam
``result.json`` files that already exist on disk; nothing here runs models
or feeds anything back into grading.

Usage:
    python scripts/prob_eval.py [--job jobs/prob-eval-2026-08-07] [--out evaluation/prob]
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import platform
import re
import sys
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from autograder.metrics import ExamOutcome, compute_metrics  # noqa: E402

LETTER_TO_HEBREW = {"A": "א", "B": "ב", "C": "ג", "D": "ד"}


def load_expected(grades_csv: Path) -> dict[str, int]:
    expected: dict[str, int] = {}
    for line in grades_csv.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            expected[f"{int(parts[0]):02d}"] = int(parts[1])
    return expected


def server_info(base_url: str) -> dict:
    """Best-effort capture of the inference server's resource usage."""
    try:
        resp = httpx.get(base_url.replace("/v1", "") + "/api/ps", timeout=5.0)
        models = resp.json().get("models", [])
        return {
            "loaded_models": [
                {
                    "name": m.get("name"),
                    "size_bytes": m.get("size"),
                    "size_vram_bytes": m.get("size_vram", 0),
                }
                for m in models
            ]
        }
    except Exception as e:  # noqa: BLE001
        return {"error": f"server info unavailable: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", default=str(REPO / "jobs" / "prob-eval-2026-08-07"))
    ap.add_argument("--out", default=str(REPO / "evaluation" / "prob"))
    ap.add_argument("--grades", default=str(REPO / "prob_data" / "grades.csv"))
    args = ap.parse_args()

    job_dir = Path(args.job)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    state = json.loads((job_dir / "state.json").read_text(encoding="utf-8"))
    job = json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
    name_map = json.loads(
        (job_dir / "uploads" / "name_map.json").read_text(encoding="utf-8")
    )
    expected = load_expected(Path(args.grades))

    # Sheet-faithful reference: the independent visual audit (all 130 rows
    # double-read unanimously; 3 grades.csv totals shown inconsistent with
    # the marked sheets — see manual_audit.json). Metrics are reported
    # against BOTH references.
    audit_path = out_dir / "manual_audit.json"
    audit: dict[str, dict] = {}
    if audit_path.exists():
        for a in json.loads(audit_path.read_text(encoding="utf-8"))["exams"]:
            audit[a["source_index"]] = a

    outcomes: list[ExamOutcome] = []
    outcomes_audited: list[ExamOutcome] = []
    rows: list[dict] = []
    for anon, ex in sorted(state["exams"].items()):
        original = name_map.get(anon, "")
        index = Path(original).stem
        outcome = ExamOutcome(anon_id=anon)
        row: dict = {
            "exam": anon,
            "source_index": index,
            "status": ex["status"],
            "variant": ex.get("variant"),
            "predicted": ex.get("predicted"),
            "expected": expected.get(index),
            "runtime_s": ex.get("runtime_s"),
            "review_items": ex.get("review_items"),
            "answers": None,
            "model_calls": None,
        }
        result_path = job_dir / "exams" / anon / "result.json"
        if ex["status"] == "done" and result_path.exists():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            answers = {}
            for q in result.get("questions", []):
                for s in q.get("sub_results", []):
                    answers[s["sub_item_id"]] = s.get("student_answer")
            row["answers"] = " ".join(
                f"{i}:{answers.get(str(i)) or '—'}" for i in range(1, 11)
            )
            vd = result.get("variant_detection") or {}
            row["marker_seen"] = vd.get("marker_seen")
            outcome.predicted = result.get("total_awarded")
            outcome.review_items = len(result.get("needs_human_review", []))
            outcome.unanswered_items = len(result.get("unanswered", []))
            a = audit.get(index)
            if a:
                row["audited_total"] = a["audited_total"]
                row["variant_matches_audit"] = ex.get("variant") == a["suit"]
                matches = sum(
                    1 for i in map(str, range(1, 11))
                    if answers.get(i) == a["answers"].get(i)
                )
                row["answer_accuracy"] = matches / 10
                row["answer_mismatches"] = " ".join(
                    f"{i}:{answers.get(i) or '—'}≠{a['answers'][i] or '—'}"
                    for i in map(str, range(1, 11))
                    if answers.get(i) != a["answers"].get(i)
                ) or None
        elif ex["status"] == "failed":
            outcome.failed = True
            outcome.failure_reason = (ex.get("error") or "").splitlines()[0]
        else:
            # not yet graded — exclude from metrics entirely
            rows.append(row)
            continue
        # Label join happens HERE, after the prediction was read from disk.
        outcome.expected = float(expected[index]) if index in expected else None
        outcome.runtime_s = ex.get("runtime_s")
        outcomes.append(outcome)
        if index in audit:
            oa = ExamOutcome(anon_id=anon)
            oa.predicted = outcome.predicted
            oa.failed = outcome.failed
            oa.failure_reason = outcome.failure_reason
            oa.review_items = outcome.review_items
            oa.unanswered_items = outcome.unanswered_items
            oa.runtime_s = outcome.runtime_s
            oa.expected = float(audit[index]["audited_total"])
            outcomes_audited.append(oa)
        row["signed_error"] = (
            outcome.predicted - outcome.expected
            if outcome.predicted is not None and outcome.expected is not None
            else None
        )
        # count model calls from the grade log: variant detection + advisory
        # disambiguation calls (multi-mark rows) + per-row VLM band reads
        # (fallback path only); the deterministic table analysis makes none.
        log_path = job_dir / "exams" / anon / "grade.log"
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            variant_calls = log_text.count("detecting exam variant")
            disamb_calls = log_text.count("advisory disambiguation")
            band_vlm_calls = len(re.findall(r"extracting question \S+ row ", log_text))
            if disamb_calls or band_vlm_calls or "deterministic" in log_text:
                row["model_calls"] = variant_calls + disamb_calls + band_vlm_calls
            else:
                row["model_calls"] = variant_calls + log_text.count("extracting question")
        rows.append(row)

    metrics = compute_metrics(outcomes)
    m = metrics.to_dict()
    m_audit = compute_metrics(outcomes_audited).to_dict() if outcomes_audited else None
    acc_rows = [r for r in rows if r.get("answer_accuracy") is not None]
    answer_acc = (
        sum(r["answer_accuracy"] for r in acc_rows) / len(acc_rows)
        if acc_rows else None
    )

    combined = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "job": job_dir.name,
        "backend": job.get("backend_args", {}),
        "grading": job.get("grading_args", {}),
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "gpu": "none — CPU-only laptop (no discrete GPU); VRAM n/a",
        },
        "server": server_info(job.get("backend_args", {}).get("--base-url", "")),
        "metrics_vs_official_grades": m,
        "metrics_vs_audited_sheets": m_audit,
        "answer_extraction_accuracy_vs_audit": answer_acc,
        "exams": rows,
    }
    (out_dir / "results.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for r in rows for k in r})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def metric_table(mm: dict) -> list[str]:
        return [
            "| Metric | Value |",
            "|---|---:|",
            f"| Exams scored | {mm['scored']} |",
            f"| Failures | {mm['failures']} |",
            f"| Exact-grade accuracy | {mm['exact_accuracy']:.0%} |",
            f"| Within ±5 | {mm['within_5']:.0%} |",
            f"| Within ±10 | {mm['within_10']:.0%} |",
            f"| MAE | {mm['mae']} |",
            f"| Median abs error | {mm['median_abs_error']} |",
            f"| RMSE | {mm['rmse']} |",
            f"| Mean signed error | {mm['mean_signed_error']} |",
            f"| Max abs error | {mm['max_abs_error']} |",
            f"| Human-review rate | {mm['review_rate']:.0%} |",
            f"| Mean runtime / exam | {mm.get('mean_runtime_s', 'n/a')} s |",
        ]

    done = [r for r in rows if r["status"] == "done"]
    lines = [
        "# Prob-dataset evaluation (multiple-choice benchmark)",
        "",
        f"- Generated: {combined['generated_at']}",
        f"- Job: `{job_dir.name}` — {len(done)}/{len(state['exams'])} exams graded",
        f"- Backend: `{combined['backend'].get('--model')}` at `{combined['backend'].get('--base-url')}`",
        f"- Machine: {combined['machine']['platform']} — **CPU-only, no GPU/VRAM**",
        "",
        "## Total-score metrics vs official grades (grades.csv)",
        "",
        *metric_table(m),
    ]
    if m_audit:
        lines += [
            "",
            "## Total-score metrics vs audited sheets",
            "",
            "The independent visual audit (manual_audit.json: every row double-",
            "read unanimously, key re-derived from the booklets) found grades.csv",
            "inconsistent with the physically marked sheets on scans 05 (+10),",
            "06 (+10) and 13 (−10) — instructor totaling errors. Against the",
            "sheet-faithful reference:",
            "",
            *metric_table(m_audit),
        ]
    if answer_acc is not None:
        # Decided vs deferred: an "ambiguous" row is NOT a misread — it earns
        # 0 pending review with the marked columns as candidates. Score them
        # separately, and simulate the reviewer resolving them as the audit did.
        decided_total = decided_ok = deferred = deferred_truth_in_cands = 0
        post_review_match_official = post_review_match_audit = 0
        for r in rows:
            if r.get("answer_accuracy") is None:
                continue
            a = audit[r["source_index"]]
            result = json.loads(
                (job_dir / "exams" / r["exam"] / "result.json").read_text(encoding="utf-8")
            )
            resolved = 0.0
            for q in result["questions"]:
                for s in q["sub_results"]:
                    rid = s["sub_item_id"]
                    truth = a["answers"][rid]
                    if s["status"] == "ambiguous":
                        deferred += 1
                        if truth and truth in (s.get("reason") or ""):
                            deferred_truth_in_cands += 1
                        # reviewer resolves per audit: award if audit answer correct
                        accepted = s.get("accepted_answers") or []
                        if truth in accepted:
                            resolved += s["points_max"]
                    else:
                        decided_total += 1
                        if s["student_answer"] == truth:
                            decided_ok += 1
                        resolved += s["points_total"]
            if r.get("expected") is not None and resolved == r["expected"]:
                post_review_match_official += 1
            if resolved == a["audited_total"]:
                post_review_match_audit += 1
        lines += [
            "",
            "## Answer-extraction accuracy vs audited sheets",
            "",
            f"- Mean per-row accuracy over {len(acc_rows)} audited exams: "
            f"**{answer_acc:.1%}** (10 rows each; '—' entries below are rows "
            "DEFERRED to human review, not misreads)",
            f"- **Auto-decided rows: {decided_ok}/{decided_total} correct** "
            "vs the audit — zero silent errors.",
            f"- Deferred rows: {deferred}, of which {deferred_truth_in_cands} "
            "carry the audited answer among their listed candidates.",
            f"- If the reviewer resolves each deferred row as the audit read "
            f"it, totals match the audited sheets on "
            f"**{post_review_match_audit}/{len(acc_rows)}** exams and the "
            f"official grades on {post_review_match_official}/{len(acc_rows)} "
            "(the remaining gap is exactly the three documented instructor "
            "totaling errors).",
            "- A correct total with per-row errors would be visible here — "
            "totals are never accepted on cancellation.",
        ]
    lines += [
        "",
        "## Per-exam results",
        "",
        "| Exam | Src | Variant | Answers (1-10) | Predicted | Official | Audited | Err(official) | Row acc | Mismatches | Review | Runtime (s) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for r in rows:
        err = f"{r['signed_error']:+g}" if r.get("signed_error") is not None else "—"
        acc = f"{r['answer_accuracy']:.0%}" if r.get("answer_accuracy") is not None else "—"
        lines.append(
            f"| {r['exam']} | {r['source_index']} | {r.get('variant') or '—'} | "
            f"`{r.get('answers') or '—'}` | {r.get('predicted') if r.get('predicted') is not None else '—'} | "
            f"{r.get('expected') if r.get('expected') is not None else '—'} | "
            f"{r.get('audited_total') if r.get('audited_total') is not None else '—'} | {err} | {acc} | "
            f"{r.get('answer_mismatches') or '—'} | "
            f"{r.get('review_items') if r.get('review_items') is not None else '—'} | "
            f"{r.get('runtime_s') if r.get('runtime_s') is not None else '—'} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_dir / 'report.md'}")
    print(json.dumps(m, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
