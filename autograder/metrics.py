"""Total-score evaluation metrics (pure Python, offline-testable)."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field


@dataclass
class ExamOutcome:
    anon_id: str
    expected: float | None = None
    predicted: float | None = None
    review_items: int = 0
    unanswered_items: int = 0
    runtime_s: float | None = None
    failed: bool = False
    failure_reason: str | None = None
    detected_variant: str | None = None
    variant_uncertain: bool = False
    key_source: str | None = None  # cache | parsed | resume | json

    @property
    def error(self) -> float | None:
        if self.failed or self.expected is None or self.predicted is None:
            return None
        return self.predicted - self.expected


@dataclass
class Metrics:
    processed: int = 0
    failures: int = 0
    scored: int = 0
    exact: float = 0.0
    within_2: float = 0.0
    within_5: float = 0.0
    within_10: float = 0.0
    mae: float = 0.0
    median_ae: float = 0.0
    rmse: float = 0.0
    mean_signed_error: float = 0.0
    max_abs_error: float = 0.0
    review_rate: float = 0.0
    mean_runtime_s: float | None = None
    outcomes: list[ExamOutcome] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {
            "processed": self.processed,
            "failures": self.failures,
            "scored": self.scored,
            "exact_accuracy": round(self.exact, 4),
            "within_2": round(self.within_2, 4),
            "within_5": round(self.within_5, 4),
            "within_10": round(self.within_10, 4),
            "mae": round(self.mae, 3),
            "median_abs_error": round(self.median_ae, 3),
            "rmse": round(self.rmse, 3),
            "mean_signed_error": round(self.mean_signed_error, 3),
            "max_abs_error": round(self.max_abs_error, 3),
            "review_rate": round(self.review_rate, 4),
        }
        if self.mean_runtime_s is not None:
            d["mean_runtime_s"] = round(self.mean_runtime_s, 1)
        return d


def compute_metrics(outcomes: list[ExamOutcome]) -> Metrics:
    m = Metrics(outcomes=outcomes)
    m.processed = len(outcomes)
    m.failures = sum(1 for o in outcomes if o.failed)
    errors = [o.error for o in outcomes if o.error is not None]
    m.scored = len(errors)
    if errors:
        abs_errors = [abs(e) for e in errors]
        m.exact = sum(1 for e in abs_errors if e == 0) / len(errors)
        m.within_2 = sum(1 for e in abs_errors if e <= 2) / len(errors)
        m.within_5 = sum(1 for e in abs_errors if e <= 5) / len(errors)
        m.within_10 = sum(1 for e in abs_errors if e <= 10) / len(errors)
        m.mae = sum(abs_errors) / len(errors)
        m.median_ae = statistics.median(abs_errors)
        m.rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
        m.mean_signed_error = sum(errors) / len(errors)
        m.max_abs_error = max(abs_errors)
    graded = [o for o in outcomes if not o.failed]
    if graded:
        m.review_rate = sum(1 for o in graded if o.review_items > 0) / len(graded)
    runtimes = [o.runtime_s for o in graded if o.runtime_s is not None]
    if runtimes:
        m.mean_runtime_s = sum(runtimes) / len(runtimes)
    return m
