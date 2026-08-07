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

    outcomes: list[ExamOutcome] = []
    rows: list[dict] = []
    call_design = None
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
        row["signed_error"] = (
            outcome.predicted - outcome.expected
            if outcome.predicted is not None and outcome.expected is not None
            else None
        )
        # count model calls from the grade log: variant detection + one line
        # per band call ("extracting question N row R"); without banding fall
        # back to per-question lines (chunked calls undercount slightly)
        log_path = job_dir / "exams" / anon / "grade.log"
        if log_path.exists():
            log_text = log_path.read_text(encoding="utf-8", errors="replace")
            band_calls = len(re.findall(r"extracting question \S+ row ", log_text))
            variant_calls = log_text.count("detecting exam variant")
            if band_calls:
                row["model_calls"] = variant_calls + band_calls
            else:
                row["model_calls"] = variant_calls + log_text.count("extracting question")
        rows.append(row)

    metrics = compute_metrics(outcomes)
    m = metrics.to_dict()

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
        "metrics": m,
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

    done = [r for r in rows if r["status"] == "done"]
    lines = [
        "# Prob-dataset evaluation (multiple-choice benchmark)",
        "",
        f"- Generated: {combined['generated_at']}",
        f"- Job: `{job_dir.name}` — {len(done)}/{len(state['exams'])} exams graded",
        f"- Backend: `{combined['backend'].get('--model')}` at `{combined['backend'].get('--base-url')}`",
        f"- Machine: {combined['machine']['platform']} — **CPU-only, no GPU/VRAM**",
        "",
        "## Total-score metrics vs instructor grades",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Exams scored | {m['scored']} |",
        f"| Failures | {m['failures']} |",
        f"| Exact-grade accuracy | {m['exact_accuracy']:.0%} |",
        f"| Within ±5 | {m['within_5']:.0%} |",
        f"| Within ±10 | {m['within_10']:.0%} |",
        f"| MAE | {m['mae']} |",
        f"| Median abs error | {m['median_abs_error']} |",
        f"| RMSE | {m['rmse']} |",
        f"| Mean signed error | {m['mean_signed_error']} |",
        f"| Max abs error | {m['max_abs_error']} |",
        f"| Human-review rate | {m['review_rate']:.0%} |",
        f"| Mean runtime / exam | {m.get('mean_runtime_s', 'n/a')} s |",
        "",
        "## Per-exam results",
        "",
        "| Exam | Src | Variant | Answers (1-10) | Predicted | Expected | Error | Review | Runtime (s) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        err = f"{r['signed_error']:+g}" if r.get("signed_error") is not None else "—"
        lines.append(
            f"| {r['exam']} | {r['source_index']} | {r.get('variant') or '—'} | "
            f"`{r.get('answers') or '—'}` | {r.get('predicted') if r.get('predicted') is not None else '—'} | "
            f"{r.get('expected') if r.get('expected') is not None else '—'} | {err} | "
            f"{r.get('review_items') if r.get('review_items') is not None else '—'} | "
            f"{r.get('runtime_s') if r.get('runtime_s') is not None else '—'} |"
        )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_dir / 'report.md'}")
    print(json.dumps(m, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
