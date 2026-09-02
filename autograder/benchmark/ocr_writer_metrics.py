"""Report-time per-writer OCR metrics: CER / WER / omission / digit-sign.

The OCR validation plan requires WRITER-GROUPED CER plus WER-family metrics
(the earlier campaign showed writer-grouped CER is the honest unit). The
frozen ``OcrPrimaryAdapter`` (``ocr-primary-bench-v1``) scores CER only and
is deliberately NOT modified — a scoring change there would bump the adapter
version and detach new runs from the frozen history. This module computes
the richer view AFTER a run, from (reference, hypothesis) pairs, using the
exact frozen metric functions of ``scripts/hebrew_bench_eval.py`` via
``scripts/refaudit.py`` (single namespace — the harness and the frozen
evaluator can never drift).

No model / provider call; pure post-processing of persisted rows.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Optional

from .manifests import REPO_ROOT, BenchmarkManifest

#: ocr-writer-metrics-v1 (2026-09-02): first version. Report-time only —
#: never part of the adapter's frozen scoring.
OCR_WRITER_METRICS_VERSION = "ocr-writer-metrics-v1"

_FNS: dict | None = None


class OcrMetricsError(RuntimeError):
    """Typed refusal: inadmissible reference or malformed pair."""


def _fns() -> dict:
    global _FNS
    if _FNS is None:
        from .manifests import _load_refaudit
        ra = _load_refaudit()
        normalize, lev, word_align = ra._load_metric_fns()
        _FNS = {"normalize": normalize, "lev": lev, "word_align": word_align,
                "digit_op_signature": ra.digit_op_signature}
    return _FNS


def writer_of_case(case_id: str) -> str:
    """Writer token of a handwritten bench item (hl_/hc_ e-writer ids);
    born-digital/printed items group under 'no_writer'."""
    parts = str(case_id).split("_")
    if len(parts) >= 2 and parts[0] in ("hl", "hc") and parts[1].startswith("e"):
        return parts[1]
    return "no_writer"


def pair_metrics(reference: str, hypothesis: Optional[str]) -> dict[str, Any]:
    """Exact per-pair metrics. hypothesis None = a lost line (schema failure
    or refused output) — counted as line loss, never silently skipped."""
    f = _fns()
    if not isinstance(reference, str) or not reference.strip():
        raise OcrMetricsError("empty/invalid reference; refusing to score")
    if hypothesis is None:
        return {"scored": False, "line_lost": True, "cer": None, "wer": None,
                "omission_rate": None, "hallucination_rate": None,
                "digit_sign_error": None}
    g, h = f["normalize"](reference), f["normalize"](hypothesis)
    cer = (f["lev"](g, h) / len(g)) if g else (0.0 if not h else 1.0)
    subs, dels, ins = f["word_align"](g.split(), h.split())
    gt_words = max(len(g.split()), 1)
    hyp_words = max(len(h.split()), 1)
    return {"scored": True, "line_lost": False,
            "cer": round(cer, 4),
            "wer": round((subs + dels + ins) / gt_words, 4),
            "omission_rate": round(dels / gt_words, 4),
            "hallucination_rate": round(ins / hyp_words, 4),
            "digit_sign_error":
                f["digit_op_signature"](reference) != f["digit_op_signature"](hypothesis)}


def writer_metrics(pairs: list[dict]) -> dict[str, Any]:
    """Aggregate per-writer + overall. Each pair:
        {case_id, reference, hypothesis, provenance_valid}
    Inadmissible provenance is a refusal count, never a silent skip."""
    rows = []
    refused = []
    for p in pairs:
        if not p.get("provenance_valid"):
            refused.append(p.get("case_id"))
            continue
        m = pair_metrics(p["reference"], p.get("hypothesis"))
        rows.append({"case_id": p["case_id"],
                     "writer": writer_of_case(p["case_id"]), **m})

    def _block(sub: list[dict]) -> dict:
        scored = [r for r in sub if r["scored"]]
        cers = [r["cer"] for r in scored]
        return {"cases": len(sub), "scored": len(scored),
                "line_loss": sum(1 for r in sub if r["line_lost"]),
                "line_loss_rate": round(sum(1 for r in sub if r["line_lost"])
                                        / len(sub), 4) if sub else None,
                "mean_cer": round(statistics.mean(cers), 4) if cers else None,
                "median_cer": round(statistics.median(cers), 4) if cers else None,
                "max_cer": round(max(cers), 4) if cers else None,
                "mean_wer": round(statistics.mean(
                    [r["wer"] for r in scored]), 4) if scored else None,
                "mean_omission_rate": round(statistics.mean(
                    [r["omission_rate"] for r in scored]), 4) if scored else None,
                "mean_hallucination_rate": round(statistics.mean(
                    [r["hallucination_rate"] for r in scored]), 4) if scored else None,
                "digit_sign_errors": sum(1 for r in scored
                                         if r["digit_sign_error"]),
                "cer_le_5pct": sum(1 for c in cers if c <= 0.05),
                "cer_le_5pct_rate": round(sum(1 for c in cers if c <= 0.05)
                                          / len(cers), 4) if cers else None}

    writers = sorted({r["writer"] for r in rows})
    per_writer = {w: _block([r for r in rows if r["writer"] == w])
                  for w in writers}
    worst = None
    gated = {w: b for w, b in per_writer.items() if b["mean_cer"] is not None}
    if gated:
        worst = max(gated, key=lambda w: gated[w]["mean_cer"])
    return {"metrics_version": OCR_WRITER_METRICS_VERSION,
            "overall": _block(rows),
            "per_writer": per_writer,
            "worst_writer_by_mean_cer": worst,
            "proposed_gate": "per-writer CER <= 5% (every writer, not the "
                             "mean of means) AND zero harmful verdict flips — "
                             "the flip metric needs the frozen local grader "
                             "and is owner-gated separately",
            "refused_invalid_reference": sorted(x for x in refused if x),
            "per_case": sorted(rows, key=lambda r: r["case_id"])}


def pairs_from_run(run_dir: Path | str, manifest: BenchmarkManifest) -> list[dict]:
    """Build scoring pairs from a completed run's outputs.jsonl + the frozen
    manifest labels (references never leave the evaluation side)."""
    run_dir = Path(run_dir)
    by_id = {c.case_id: c for c in manifest.cases}
    pairs = []
    outputs = run_dir / "outputs.jsonl"
    if not outputs.exists():
        raise OcrMetricsError(f"no outputs.jsonl under {run_dir}")
    for i, line in enumerate(outputs.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            raise OcrMetricsError(f"malformed output row at line {i}: {e}") from e
        case = by_id.get(row.get("case_id"))
        if case is None:
            raise OcrMetricsError(f"output row {row.get('case_id')!r} not in "
                                  "the frozen manifest")
        hyp = None
        out = row.get("output") or {}
        if isinstance(out, dict):
            hyp = out.get("transcription")
        pairs.append({"case_id": case.case_id,
                      "reference": case.label.get("reference"),
                      "hypothesis": hyp,
                      "provenance_valid": case.label.get("provenance_valid")})
    return pairs


__all__ = ["OCR_WRITER_METRICS_VERSION", "OcrMetricsError", "pair_metrics",
           "writer_metrics", "writer_of_case", "pairs_from_run"]
