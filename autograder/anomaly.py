"""Batch-level anomaly detection.

Every model and every deterministic stage can be locally self-consistent
while the WHOLE batch is wrong: the wrong template was loaded, the answer key
column is shifted, a variant mapping is inverted, one question's crop
coordinates moved, one question's extraction broke. Per-item confidence
cannot see any of that — it is only visible across students.

This module compares each question/page/template/variant against the rest of
the same batch and emits ``BatchWarning``s. It NEVER changes a grade and
never resolves anything: it tells the lecturer where to look, once, instead
of letting a systemic failure become fifty independent review items (see
``reviewqueue`` for the grouping that consumes these warnings).

Everything is deterministic arithmetic over already-persisted per-item
records — no model, no network. Small batches are under-powered by
construction, so nothing fires below ``min_exams``/``min_affected``.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

SEVERITY = ("info", "warning", "critical")


@dataclass
class ItemObservation:
    """One graded (or attempted) sub-item, as persisted by the pipeline."""

    exam_id: str
    question_id: str
    sub_item_id: str = ""
    variant: Optional[str] = None
    template: Optional[str] = None
    page: Optional[int] = None
    blank: bool = False              # nothing was written / no mark found
    crop_failed: bool = False        # the crop could not be produced or was INVALID
    ambiguous_mc: bool = False       # deterministic MC could not decide
    ocr_failed: bool = False         # transcription unusable / OCR_UNRESOLVED
    ocr_chars: Optional[int] = None
    review: bool = False
    review_reason: Optional[str] = None
    grade_invalid: bool = False      # grader output failed deterministic validation
    score: Optional[float] = None
    max_score: Optional[float] = None
    alignment_failed: bool = False   # printed -> canonical mapping unresolved
    template_mismatch: bool = False


@dataclass
class ExamObservation:
    """One student exam's package-level facts."""

    exam_id: str
    variant: Optional[str] = None
    variant_unknown: bool = False
    template: Optional[str] = None
    page_count: Optional[int] = None
    page_count_mismatch: bool = False
    alignment_failed: bool = False


@dataclass
class BatchWarning:
    code: str
    severity: str
    scope: str                       # question | page | template | variant | batch
    scope_id: str
    affected_items: int
    affected_students: int
    explanation: str
    signals: dict[str, Any] = field(default_factory=dict)
    sample_exam_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)

    def covers(self, item: ItemObservation) -> bool:
        """True when this warning plausibly explains that item's review —
        used to group many reviews under one systemic cause."""
        if self.scope == "question":
            return item.question_id == self.scope_id
        if self.scope == "page":
            return str(item.page) == self.scope_id
        if self.scope == "template":
            return (item.template or "") == self.scope_id
        if self.scope == "variant":
            return (item.variant or "") == self.scope_id
        return self.scope == "batch"


@dataclass
class AnomalyConfig:
    min_exams: int = 5               # below this a batch cannot support a claim
    min_affected: int = 3            # never fire on one or two items
    rate_floor: float = 0.30         # an outlier must also be absolutely high
    rate_multiple: float = 2.5       # ...and this many times the rest of the batch
    rate_margin: float = 0.25        # ...or this far above the rest in absolute terms
    unknown_variant_rate: float = 0.15
    variant_dominance: float = 0.85
    variant_expected_tolerance: float = 0.25
    degenerate_share: float = 0.90   # share of students at 0 % or 100 % on one question
    length_ratio: float = 0.35       # question mean OCR length vs batch mean
    cluster_rate: float = 0.30       # template/variant-level clustering rate


# --------------------------------------------------------------------------


def _rate(n: int, d: int) -> float:
    return (n / d) if d else 0.0


def _is_outlier(rate: float, others: list[float], cfg: AnomalyConfig) -> bool:
    """Absolutely high AND clearly apart from the rest of the batch."""
    if rate < cfg.rate_floor:
        return False
    if not others:
        return False
    base = statistics.median(others)
    return rate >= max(base * cfg.rate_multiple, base + cfg.rate_margin)


def _students(items: Iterable[ItemObservation]) -> list[str]:
    return sorted({i.exam_id for i in items})


def _warn(code: str, severity: str, scope: str, scope_id: str,
          affected: list[ItemObservation], explanation: str, **signals) -> BatchWarning:
    st = _students(affected)
    return BatchWarning(code=code, severity=severity, scope=scope, scope_id=scope_id,
                        affected_items=len(affected), affected_students=len(st),
                        explanation=explanation, signals=signals, sample_exam_ids=st[:5])


def _by(items: Iterable[ItemObservation], attr: str) -> dict[str, list[ItemObservation]]:
    out: dict[str, list[ItemObservation]] = defaultdict(list)
    for i in items:
        v = getattr(i, attr)
        if v is not None and v != "":
            out[str(v)].append(i)
    return dict(out)


def _flag_rate_outliers(groups: dict[str, list[ItemObservation]], predicate, *, code: str,
                        scope: str, severity: str, cfg: AnomalyConfig, wording: str) -> list[BatchWarning]:
    rates = {g: _rate(sum(1 for i in its if predicate(i)), len(its)) for g, its in groups.items()}
    out: list[BatchWarning] = []
    for g, its in sorted(groups.items()):
        hit = [i for i in its if predicate(i)]
        if len(hit) < cfg.min_affected:
            continue
        others = [r for k, r in rates.items() if k != g]
        if not _is_outlier(rates[g], others, cfg):
            continue
        base = statistics.median(others) if others else 0.0
        out.append(_warn(code, severity, scope, g, hit,
                         wording.format(scope_id=g, rate=rates[g], base=base),
                         rate=round(rates[g], 3), batch_rate=round(base, 3),
                         group_size=len(its)))
    return out


# --------------------------------------------------------------------------


def detect_batch_anomalies(items: Iterable[ItemObservation],
                           exams: Iterable[ExamObservation] = (),
                           *, expected_variant_distribution: dict[str, float] | None = None,
                           config: AnomalyConfig | None = None) -> list[BatchWarning]:
    """Return every batch-level warning, most severe first. Never mutates
    anything and never decides a grade."""
    cfg = config or AnomalyConfig()
    items = list(items)
    exams = list(exams)
    n_exams = len({*(e.exam_id for e in exams), *(i.exam_id for i in items)})
    if n_exams < cfg.min_exams:
        return []

    out: list[BatchWarning] = []
    by_q = _by(items, "question_id")
    by_page = _by(items, "page")
    by_template = _by(items, "template")
    by_variant = _by(items, "variant")

    # -- extraction ---------------------------------------------------------
    out += _flag_rate_outliers(by_q, lambda i: i.blank, code="QUESTION_BLANK_RATE_SPIKE",
                               scope="question", severity="critical", cfg=cfg,
                               wording=("question {scope_id} is blank for {rate:.0%} of students "
                                        "against {base:.0%} elsewhere in this batch — a shifted crop "
                                        "or a broken extraction looks more likely than the cohort"))
    out += _flag_rate_outliers(by_page, lambda i: i.crop_failed, code="CROP_FAILURE_CLUSTER",
                               scope="page", severity="critical", cfg=cfg,
                               wording=("page {scope_id} fails to produce a usable crop for {rate:.0%} "
                                        "of students against {base:.0%} on other pages"))
    out += _flag_rate_outliers(by_q, lambda i: i.ambiguous_mc, code="MC_AMBIGUITY_SPIKE",
                               scope="question", severity="warning", cfg=cfg,
                               wording=("question {scope_id} produces undecidable marks for {rate:.0%} "
                                        "of students against {base:.0%} elsewhere — check the column "
                                        "geometry for this question"))

    # -- OCR ----------------------------------------------------------------
    out += _flag_rate_outliers(by_q, lambda i: i.ocr_failed, code="OCR_FAILURE_CLUSTER",
                               scope="question", severity="warning", cfg=cfg,
                               wording=("question {scope_id} fails transcription for {rate:.0%} of "
                                        "students against {base:.0%} elsewhere"))
    out += _flag_rate_outliers(by_template, lambda i: i.ocr_failed, code="OCR_FAILURE_CLUSTER",
                               scope="template", severity="warning", cfg=cfg,
                               wording=("template {scope_id} fails transcription for {rate:.0%} of its "
                                        "items against {base:.0%} on other templates"))
    out += _ocr_length_anomalies(by_q, cfg)

    # -- grading ------------------------------------------------------------
    out += _degenerate_scores(by_q, cfg)
    out += _flag_rate_outliers(by_q, lambda i: i.review, code="GRADE_REVIEW_CLUSTER",
                               scope="question", severity="warning", cfg=cfg,
                               wording=("{rate:.0%} of question {scope_id} needs human review against "
                                        "{base:.0%} elsewhere — one cause probably explains all of them"))
    out += _flag_rate_outliers(by_template, lambda i: i.grade_invalid, code="GRADE_INVALID_CLUSTER",
                               scope="template", severity="warning", cfg=cfg,
                               wording=("grader output fails validation on {rate:.0%} of template "
                                        "{scope_id} against {base:.0%} elsewhere"))
    out += _flag_rate_outliers(by_variant, lambda i: i.grade_invalid, code="GRADE_INVALID_CLUSTER",
                               scope="variant", severity="warning", cfg=cfg,
                               wording=("grader output fails validation on {rate:.0%} of variant "
                                        "{scope_id} against {base:.0%} elsewhere"))

    # -- alignment / package ------------------------------------------------
    out += _alignment_clusters(items, by_q, cfg)
    out += _variant_anomalies(exams, items, expected_variant_distribution, cfg)
    out += _template_mismatch(exams, items, cfg)

    order = {s: n for n, s in enumerate(reversed(SEVERITY))}
    out.sort(key=lambda w: (order.get(w.severity, 9), -w.affected_students, w.code))
    return out


def _ocr_length_anomalies(by_q: dict[str, list[ItemObservation]], cfg: AnomalyConfig) -> list[BatchWarning]:
    means = {}
    for q, its in by_q.items():
        lens = [i.ocr_chars for i in its if i.ocr_chars is not None]
        if len(lens) >= cfg.min_affected:
            means[q] = statistics.mean(lens)
    if len(means) < 3:
        return []
    overall = statistics.median(means.values())
    out = []
    for q, m in sorted(means.items()):
        if overall > 0 and m < overall * cfg.length_ratio:
            its = [i for i in by_q[q] if i.ocr_chars is not None]
            out.append(_warn("OCR_LENGTH_ANOMALY", "info", "question", q, its,
                             f"question {q} transcribes {m:.0f} characters on average against "
                             f"{overall:.0f} elsewhere — the answer area may be mis-cropped",
                             mean_chars=round(m, 1), batch_mean_chars=round(overall, 1)))
    return out


def _degenerate_scores(by_q: dict[str, list[ItemObservation]], cfg: AnomalyConfig) -> list[BatchWarning]:
    out = []
    for q, its in sorted(by_q.items()):
        scored = [i for i in its if i.score is not None and i.max_score]
        if len(scored) < max(cfg.min_affected, 5):
            continue
        zero = sum(1 for i in scored if i.score <= 1e-9)
        full = sum(1 for i in scored if abs(i.score - i.max_score) <= 1e-9)
        for n, label in ((zero, "zero"), (full, "full")):
            share = _rate(n, len(scored))
            if share >= cfg.degenerate_share:
                out.append(_warn("QUESTION_SCORE_DEGENERATE", "critical", "question", q, scored,
                                 f"{share:.0%} of students score {label} marks on question {q} — "
                                 "check the answer-key column, the variant mapping and the rubric "
                                 "gate before accepting these grades",
                                 share=round(share, 3), kind=label, n=len(scored)))
    return out


def _alignment_clusters(items: list[ItemObservation], by_q: dict[str, list[ItemObservation]],
                        cfg: AnomalyConfig) -> list[BatchWarning]:
    out = []
    for q, its in sorted(by_q.items()):
        hit = [i for i in its if i.alignment_failed]
        if len(hit) < cfg.min_affected:
            continue
        if _rate(len(hit), len(its)) >= cfg.cluster_rate:
            out.append(_warn("ALIGNMENT_FAILURE_CLUSTER", "critical", "question", q, hit,
                             f"{len(hit)} students share the same unresolved printed-to-canonical "
                             f"mapping on question {q} — resolve the alignment once instead of "
                             "reviewing each student",
                             rate=round(_rate(len(hit), len(its)), 3)))
    return out


def _variant_anomalies(exams: list[ExamObservation], items: list[ItemObservation],
                       expected: dict[str, float] | None, cfg: AnomalyConfig) -> list[BatchWarning]:
    if not exams:
        return []
    out = []
    unknown = [e for e in exams if e.variant_unknown or not e.variant]
    rate = _rate(len(unknown), len(exams))
    if rate > cfg.unknown_variant_rate and len(unknown) >= cfg.min_affected:
        ids = sorted(e.exam_id for e in unknown)
        out.append(BatchWarning("VARIANT_UNRESOLVED_RATE", "critical", "batch", "*",
                                len(unknown), len(ids),
                                f"the variant marker is unresolved on {rate:.0%} of the batch — the "
                                "marker catalogue or the marker page is probably wrong for this "
                                "package; fix it once at package level",
                                {"rate": round(rate, 3), "n_exams": len(exams)}, ids[:5]))
    known = [e for e in exams if e.variant and not e.variant_unknown]
    if not known:
        return out
    counts = Counter(e.variant for e in known)
    shares = {v: n / len(known) for v, n in counts.items()}
    exam_items = defaultdict(list)
    for i in items:
        exam_items[i.exam_id].append(i)
    for v, share in sorted(shares.items()):
        exp = (expected or {}).get(v)
        bad_expected = exp is not None and abs(share - exp) > cfg.variant_expected_tolerance
        bad_dominant = (expected is None and len(shares) > 1 and share >= cfg.variant_dominance
                        and len(known) >= cfg.min_exams)
        if not (bad_expected or bad_dominant):
            continue
        affected = [i for e in known if e.variant == v for i in exam_items.get(e.exam_id, [])]
        detail = (f"expected about {exp:.0%}" if exp is not None
                  else f"{len(shares)} variants are configured")
        w = _warn("VARIANT_DISTRIBUTION_ANOMALY", "warning", "variant", v, affected,
                  f"{share:.0%} of this batch was detected as {v} ({detail}) — a marker that matches "
                  "too easily, or a mis-scanned marker page, produces exactly this",
                  share=round(share, 3), expected=exp, n_exams=len(known))
        w.affected_students = counts[v]
        w.sample_exam_ids = sorted(e.exam_id for e in known if e.variant == v)[:5]
        out.append(w)
    # a variant configured in the package but never seen at all
    for v, exp in sorted((expected or {}).items()):
        if exp > 0 and v not in counts:
            out.append(BatchWarning("VARIANT_DISTRIBUTION_ANOMALY", "warning", "variant", v, 0, 0,
                                    f"variant {v} is configured (expected about {exp:.0%}) but was "
                                    "not detected on a single exam in this batch",
                                    {"share": 0.0, "expected": exp}, []))
    return out


def _template_mismatch(exams: list[ExamObservation], items: list[ItemObservation],
                       cfg: AnomalyConfig) -> list[BatchWarning]:
    # "no template recorded anywhere" means the pipeline does not track
    # templates for this package — that is silence, not a mismatch.
    templates_tracked = any(e.template for e in exams)
    out = []
    for code, pred, wording in (
        ("PAGE_COUNT_MISMATCH_CLUSTER", lambda e: e.page_count_mismatch,
         "{n} exams ({rate:.0%}) do not have the page structure this package expects"),
        ("TEMPLATE_MISMATCH_CLUSTER", lambda e: templates_tracked and e.template is None,
         "{n} exams ({rate:.0%}) could not be matched to any known page template"),
        ("ALIGNMENT_FAILURE_CLUSTER", lambda e: e.alignment_failed,
         "{n} exams ({rate:.0%}) share an unresolved question alignment"),
    ):
        hit = [e for e in exams if pred(e)]
        rate = _rate(len(hit), len(exams))
        if len(hit) >= cfg.min_affected and rate >= cfg.cluster_rate:
            ids = sorted(e.exam_id for e in hit)
            out.append(BatchWarning(code, "critical", "batch", "*", len(hit), len(ids),
                                    wording.format(n=len(hit), rate=rate)
                                    + " — one package-level fix resolves all of them",
                                    {"rate": round(rate, 3), "n_exams": len(exams)}, ids[:5]))
    return out


# --------------------------------------------------------------------------


def explained_by(warnings: list[BatchWarning], item: ItemObservation) -> Optional[BatchWarning]:
    """The most severe warning that plausibly explains this item's review."""
    order = {s: n for n, s in enumerate(reversed(SEVERITY))}
    hits = [w for w in warnings if w.covers(item)]
    return min(hits, key=lambda w: order.get(w.severity, 9)) if hits else None
