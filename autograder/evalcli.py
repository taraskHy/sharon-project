"""Dataset and batch-evaluation commands.

    autograder make-manifests   discover graded exams, build deterministic split
    autograder eval-batch       grade a whole split, compare to instructor grades
    autograder audit-leakage    probe whether a model can read the instructor's
                                grade from (un)masked pages

Leakage policy implemented here:

- exams are referred to by anonymized IDs (``exam-042``) everywhere downstream;
- the model receives page images only — never filenames, paths, or metadata;
- instructor red-ink annotations are masked from page images by default;
- the expected grade is read from the manifest and compared with the
  prediction only AFTER grading completes.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from .backends import create_backend
from .dataset import ExamRecord, assign_split, discover_exams, load_manifest, write_manifests
from .ingest import load_pages
from .masking import mask_pages
from .metrics import ExamOutcome, compute_metrics
from .schema import ExamResult


def _log(msg: str) -> None:
    print(f"[autograder] {msg}", flush=True)


# --------------------------------------------------------------------------
# make-manifests
# --------------------------------------------------------------------------


def cmd_make_manifests(args) -> int:
    report = discover_exams(args.dataset_root)
    if report.malformed:
        _log(f"malformed filenames ({len(report.malformed)}): {', '.join(report.malformed)}")
    if report.duplicate_indices:
        _log(f"duplicate indices ({len(report.duplicate_indices)}): {report.duplicate_indices}")
    if not report.records:
        _log("ERROR: no valid exam files found")
        return 2
    assign_split(report.records, seed=args.seed)
    paths = write_manifests(report, args.manifest_dir, seed=args.seed)
    n_train = sum(1 for r in report.records if r.split == "train")
    n_val = len(report.records) - n_train
    _log(
        f"{len(report.records)} exams -> {n_train} train / {n_val} validation "
        f"(seed={args.seed})"
    )
    for name, p in paths.items():
        _log(f"wrote {name}: {p}")
    warned = [r for r in report.records if r.warnings]
    for r in warned:
        _log(f"data-quality warning for {r.anon_id}: {'; '.join(r.warnings)}")
    return 0


# --------------------------------------------------------------------------
# eval-batch
# --------------------------------------------------------------------------


def _load_split_records(args) -> list[ExamRecord]:
    manifest_dir = Path(args.manifest_dir)
    wanted = ["train", "validation"] if args.split == "all" else [args.split]
    records: list[ExamRecord] = []
    for split in wanted:
        path = manifest_dir / f"{split}_manifest.json"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found — run 'autograder make-manifests' first"
            )
        records.extend(load_manifest(path))
    return records


def _shared_key_path(args, backend, eval_root: Path, max_image_edge: int) -> Path:
    """Parse the answer key once for the whole batch."""
    from .cli import _get_key, _fingerprints, _stored_fingerprints

    key_path = Path(args.key)
    if key_path.suffix.lower() == ".json":
        return key_path  # already structured; per-exam runs load it directly
    shared = eval_root / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    ns = argparse.Namespace(
        key=args.key, rubric=args.rubric, resume=args.resume,
        key_cache_dir=getattr(args, "key_cache_dir", None),
        no_key_cache=getattr(args, "no_key_cache", False),
    )
    current = _fingerprints(ns, backend, max_image_edge, include_exam=False)
    stored = _stored_fingerprints(shared)
    _, key_source = _get_key(
        ns, backend, shared, max_image_edge,
        reusable=args.resume and stored.get("key") == current["key"],
    )
    _log(f"batch answer key source: {key_source}")
    (shared / "fingerprint.json").write_text(json.dumps(current), encoding="utf-8")
    return shared / "answer_key.json"


def cmd_eval_batch(args) -> int:
    from .cli import (
        _fingerprints,
        _stored_fingerprints,
        guard_direct_cloud_backend,
        resolve_config,
        run_grade_pipeline,
    )

    backend_config, max_image_edge, survey_image_edge = resolve_config(args)
    guard_direct_cloud_backend(backend_config)
    backend = create_backend(backend_config)
    eval_root = Path(args.out)
    eval_root.mkdir(parents=True, exist_ok=True)

    records = _load_split_records(args)
    if args.limit:
        records = records[: args.limit]
    _log(f"evaluating {len(records)} exams from split '{args.split}'")

    key_json = _shared_key_path(args, backend, eval_root, max_image_edge)

    # The per-exam runs receive the SHARED parsed key json, so the variant
    # config must be resolved against the ORIGINAL key document's location.
    from .variant import alignment_override_path, variant_config_path

    variant_map = getattr(args, "variant_map", None)
    if not variant_map:
        auto = variant_config_path(args.key)
        if auto.exists():
            variant_map = str(auto)
            _log(f"variant mapping: {variant_map}")
    alignment_map = getattr(args, "alignment_map", None)
    if not alignment_map:
        auto = alignment_override_path(args.key)
        if auto.exists():
            alignment_map = str(auto)
            _log(f"alignment overrides: {alignment_map}")

    outcomes: list[ExamOutcome] = []
    review_cases: list[dict] = []
    for i, record in enumerate(records, start=1):
        _log(f"--- [{i}/{len(records)}] {record.anon_id} ---")
        exam_out = eval_root / "exams" / record.anon_id
        exam_out.mkdir(parents=True, exist_ok=True)
        outcome = ExamOutcome(anon_id=record.anon_id)
        t0 = time.monotonic()
        try:
            source = Path(record.original_path)
            ns = argparse.Namespace(
                key=str(key_json),
                rubric=None,
                resume=args.resume,
                version=args.version,
                exam=str(source),
                key_cache_dir=getattr(args, "key_cache_dir", None),
                no_key_cache=getattr(args, "no_key_cache", False),
                variant_map=variant_map,
                alignment_map=alignment_map,
            )
            # Fast resume: reuse the finished result when inputs are unchanged.
            result: Optional[ExamResult] = None
            result_path = exam_out / "result.json"
            if args.resume and result_path.exists():
                current = _fingerprints(
                    ns, backend, max_image_edge, include_exam=True, exam_path=source,
                    survey_image_edge=survey_image_edge,
                )
                if _stored_fingerprints(exam_out).get("exam") == current["exam"]:
                    _log("resume: reusing finished result")
                    result = ExamResult.model_validate_json(
                        result_path.read_text(encoding="utf-8")
                    )
            if result is None:
                def masked_loader(edge, _src=source, _mask=args.mask):
                    loaded = load_pages(_src, edge)
                    return mask_pages(loaded)[0] if _mask else loaded

                pages = load_pages(source, max_image_edge)
                if args.mask:
                    pages, mask_report = mask_pages(pages)
                    (exam_out / "masking.json").write_text(
                        json.dumps(mask_report.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    record.masking_status = "masked"
                result = run_grade_pipeline(
                    ns,
                    backend,
                    exam_out,
                    max_image_edge,
                    exam_path=source,
                    exam_label=record.anon_id,
                    pages=pages,
                    survey_image_edge=survey_image_edge,
                    page_loader=masked_loader,
                )
            outcome.predicted = result.total_awarded
            outcome.review_items = len(result.needs_human_review)
            outcome.unanswered_items = len(result.unanswered)
            outcome.detected_variant = result.detected_version
            vd = result.variant_detection or {}
            outcome.variant_uncertain = bool(vd.get("uncertain"))
            outcome.key_source = (result.backend_info or {}).get("answer_key_source")
            if result.needs_human_review:
                review_cases.append(
                    {
                        "anon_id": record.anon_id,
                        "items": [r.model_dump() for r in result.needs_human_review],
                    }
                )
        except Exception as e:  # noqa: BLE001 - batch must continue after failures
            outcome.failed = True
            outcome.failure_reason = f"{type(e).__name__}: {e}"
            _log(f"FAILED: {outcome.failure_reason}")
        outcome.runtime_s = time.monotonic() - t0
        # Label comparison happens ONLY here, after grading completed.
        outcome.expected = float(record.expected_grade)
        outcomes.append(outcome)

    metrics = compute_metrics(outcomes)
    _write_batch_reports(eval_root, args, backend, metrics, review_cases)
    _log(
        f"done: {metrics.processed} processed, {metrics.failures} failed, "
        f"MAE={metrics.mae:.2f}, within±5={metrics.within_5:.0%}, "
        f"review rate={metrics.review_rate:.0%}"
    )
    return 0


def _write_batch_reports(eval_root: Path, args, backend, metrics, review_cases) -> None:
    combined = {
        "generated_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "split": args.split,
        "masking_enabled": args.mask,
        "backend": backend.describe(),
        "metrics": metrics.to_dict(),
        "exams": [
            {
                "anon_id": o.anon_id,
                "expected": o.expected,
                "predicted": o.predicted,
                "error": o.error,
                "detected_variant": o.detected_variant,
                "variant_uncertain": o.variant_uncertain,
                "key_source": o.key_source,
                "review_items": o.review_items,
                "unanswered_items": o.unanswered_items,
                "runtime_s": round(o.runtime_s, 1) if o.runtime_s is not None else None,
                "failed": o.failed,
                "failure_reason": o.failure_reason,
            }
            for o in metrics.outcomes
        ],
    }
    (eval_root / "combined_results.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with (eval_root / "combined_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "anon_id", "expected", "predicted", "error", "detected_variant",
                "variant_uncertain", "key_source", "review_items",
                "unanswered_items", "runtime_s", "failed", "failure_reason",
            ],
        )
        writer.writeheader()
        for row in combined["exams"]:
            writer.writerow(row)
    failed = [e for e in combined["exams"] if e["failed"]]
    (eval_root / "failed_exams.json").write_text(
        json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (eval_root / "review_cases.json").write_text(
        json.dumps(review_cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    m = metrics.to_dict()
    lines = [
        f"# Batch evaluation — split '{args.split}'",
        "",
        f"- Generated: {combined['generated_at']}",
        f"- Backend: `{backend.identity}`",
        f"- Masking: {'enabled' if args.mask else 'DISABLED'}",
        f"- Exams processed: {m['processed']} (failed: {m['failures']})",
        "",
        "## Total-score metrics (vs instructor grades)",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Exact accuracy | {m['exact_accuracy']:.0%} |",
        f"| Within ±2 | {m['within_2']:.0%} |",
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
        "Client-side memory is negligible; model memory usage is a property of",
        "the inference server and is documented in docs/deployment.md.",
        "",
        "## Per-exam results",
        "",
        "| Exam | Expected | Predicted | Error | Review items | Runtime (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for e in combined["exams"]:
        if e["failed"]:
            lines.append(f"| {e['anon_id']} | {e['expected']} | FAILED | — | — | — |")
        else:
            lines.append(
                f"| {e['anon_id']} | {e['expected']} | {e['predicted']} | "
                f"{e['error']:+.1f} | {e['review_items']} | {e['runtime_s']} |"
            )
    (eval_root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    _log(f"wrote {eval_root / 'summary.md'}")


# --------------------------------------------------------------------------
# audit-leakage
# --------------------------------------------------------------------------


class GradeProbe(BaseModel):
    """Structured output for the leakage probe."""

    sees_grade_or_scores: bool = Field(
        description="True if any instructor-written grade, score, or points total is readable."
    )
    final_grade_guess: Optional[int] = Field(
        default=None, description="The final grade if readable, else null."
    )
    evidence: str = Field(description="Where on the pages the information was seen.")


_PROBE_SYSTEM = (
    "You are auditing scanned exams for information leakage. Look ONLY for "
    "grades, scores, point totals, or correctness marks written by an "
    "instructor (often red ink). Report what you can actually read; do not "
    "estimate a grade from the student's work quality."
)


def cmd_audit_leakage(args) -> int:
    from .cli import guard_direct_cloud_backend, resolve_config
    from .ingest import labeled_page_blocks

    backend_config, max_image_edge, _ = resolve_config(args)
    guard_direct_cloud_backend(backend_config)
    backend = create_backend(backend_config)
    records = _load_split_records(args)[: args.limit or 5]
    eval_root = Path(args.out)
    eval_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for record in records:
        pages = load_pages(record.original_path, max_image_edge)
        # Probe the pages most likely to carry grade summaries plus answer areas.
        probe_pages = pages[:2] + pages[-3:]
        for variant in ("unmasked", "masked"):
            imgs = probe_pages
            if variant == "masked":
                imgs, _ = mask_pages(probe_pages)
            try:
                probe = backend.parse(
                    system=_PROBE_SYSTEM,
                    content_blocks=labeled_page_blocks(imgs)
                    + [{"type": "text", "text": "Report any readable instructor grade/scores."}],
                    output_model=GradeProbe,
                )
                guessed = probe.final_grade_guess
                leaked = (
                    guessed is not None
                    and abs(guessed - record.expected_grade) <= args.tolerance
                )
                rows.append(
                    {
                        "anon_id": record.anon_id,
                        "variant": variant,
                        "sees_grade_or_scores": probe.sees_grade_or_scores,
                        "grade_guess": guessed,
                        "grade_leaked": leaked,
                        "evidence": probe.evidence,
                    }
                )
                _log(
                    f"{record.anon_id} [{variant}]: sees={probe.sees_grade_or_scores} "
                    f"guess={guessed} leaked={leaked}"
                )
            except Exception as e:  # noqa: BLE001
                rows.append(
                    {"anon_id": record.anon_id, "variant": variant, "error": str(e)}
                )
                _log(f"{record.anon_id} [{variant}]: probe failed: {e}")

    unmasked_leaks = sum(1 for r in rows if r.get("variant") == "unmasked" and r.get("grade_leaked"))
    masked_leaks = sum(1 for r in rows if r.get("variant") == "masked" and r.get("grade_leaked"))
    verdict = {
        "unmasked_leaks": unmasked_leaks,
        "masked_leaks": masked_leaks,
        "conclusion": (
            "MASKING REQUIRED: the model can read instructor grades from unmasked scans"
            if unmasked_leaks > 0
            else "no grade leakage detected on the probed sample (verify with more exams)"
        )
        + (
            "; WARNING: leakage persists AFTER masking — unmasked/red-only masking is "
            "insufficient, do not use these scans as ordinary grading input"
            if masked_leaks > 0
            else ""
        ),
    }
    out_path = eval_root / "leakage_audit.json"
    out_path.write_text(
        json.dumps({"verdict": verdict, "probes": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _log(f"wrote {out_path}")
    _log(verdict["conclusion"])
    return 0 if masked_leaks == 0 else 1


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------


def add_eval_commands(sub, common: argparse.ArgumentParser) -> None:
    from .cli import add_backend_args

    mm = sub.add_parser("make-manifests", help="Discover graded exams and build the split")
    mm.add_argument("--dataset-root", default="test", help="Directory of <index>_<grade> exams")
    mm.add_argument("--manifest-dir", default="datasets")
    mm.add_argument("--seed", type=int, default=42)
    mm.set_defaults(func=cmd_make_manifests)

    eb = sub.add_parser(
        "eval-batch", parents=[common], help="Grade a dataset split and compare to instructor grades"
    )
    eb.add_argument("--manifest-dir", default="datasets")
    eb.add_argument("--split", choices=["train", "validation", "all"], default="validation")
    eb.add_argument("--limit", type=int, default=None, help="Evaluate only the first N exams")
    eb.add_argument("--version", default="auto")
    eb.add_argument(
        "--no-mask", dest="mask", action="store_false", default=True,
        help="Disable instructor-annotation masking (leakage risk — for audits only)",
    )
    eb.set_defaults(func=cmd_eval_batch, out="eval_out")

    al = sub.add_parser(
        "audit-leakage", help="Probe whether the model can read instructor grades from scans"
    )
    add_backend_args(al)
    al.add_argument("--manifest-dir", default="datasets")
    al.add_argument("--split", choices=["train", "validation", "all"], default="train")
    al.add_argument("--limit", type=int, default=5)
    al.add_argument("--tolerance", type=int, default=2)
    al.add_argument("--out", default="eval_out")
    al.set_defaults(func=cmd_audit_leakage)
