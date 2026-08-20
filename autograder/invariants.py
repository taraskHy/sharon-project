"""Deterministic grade invariants.

Arithmetic and structural score validity are NEVER delegated to a model.
Two levels:

``check_question_invariants``  one grader output against its QuestionGradingPack
``check_exam_invariants``      a finished ExamResult against the AnswerKey

Violations mark the result INVALID and route it to escalation — they are not
silently corrected. The ONE documented exception is purely deterministic
arithmetic: when every credited rubric item declares its points, the question
score is a sum with no judgement in it, so ``repair_arithmetic`` may recompute
it. That repair is opt-in and never runs inside validation; the same holds for
``recompute_exam_totals``, which recomputes an exam total from already-decided
per-question scores (the totals are defined as that sum — the model never
supplies them).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

_EPS = 1e-6


@dataclass
class InvariantReport:
    ok: bool
    problems: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)
    repairable_total: Optional[float] = None

    def as_dict(self) -> dict:
        return {"ok": self.ok, "problems": list(self.problems), "checked": list(self.checked),
                "repairable_total": self.repairable_total}


def _add(rep: InvariantReport, name: str, ok: bool, msg: str) -> None:
    rep.checked.append(name)
    if not ok:
        rep.problems.append(msg)
        rep.ok = False


# --------------------------------------------------------------------------
# question level (one model grading output)
# --------------------------------------------------------------------------


def check_question_invariants(g: Any, pack: Any, *, selection_correct: bool | None = None) -> InvariantReport:
    """``g`` is duck-typed (escalation.GradeResult): ``.score``, ``.credited()``.
    ``pack`` is a QuestionGradingPack."""
    rep = InvariantReport(True)
    score = float(getattr(g, "score", 0.0) or 0.0)
    max_score = float(getattr(pack, "max_score", 0.0) or 0.0)
    _add(rep, "score_non_negative", score >= -_EPS, f"score {score} is negative")
    _add(rep, "score_within_max", score <= max_score + _EPS,
         f"score {score} exceeds question maximum {max_score}")

    specs = pack.rubric_specs() if hasattr(pack, "rubric_specs") else {}
    credited = list(g.credited()) if hasattr(g, "credited") else []
    ids = [c.id for c in credited]
    _add(rep, "rubric_ids_exist", not specs or all(i in specs for i in ids),
         f"credited rubric ids not in the rubric: {sorted(set(ids) - set(specs))}")
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    _add(rep, "no_duplicate_credit", not dupes, f"rubric items credited more than once: {dupes}")

    credited_set = set(ids)
    for rid in sorted(credited_set):
        spec = specs.get(rid)
        if spec is None:
            continue
        clash = sorted(credited_set & set(getattr(spec, "excludes", []) or []))
        _add(rep, "mutual_exclusion", not clash,
             f"rubric items {rid} and {clash} are declared mutually exclusive but both received credit")
        missing = sorted(set(getattr(spec, "requires", []) or []) - credited_set)
        _add(rep, "prerequisites", not missing,
             f"rubric item {rid} requires {missing}, which did not receive credit")

    # Component sum: only meaningful when every rubric item declares points.
    pts = [getattr(specs.get(i), "points", None) for i in sorted(credited_set)] if specs else []
    if specs and credited_set and all(p is not None for p in pts):
        expected = round(sum(float(p) for p in pts), 6)
        capped = min(expected, max_score)
        rep.repairable_total = capped
        _add(rep, "subscore_sum", abs(score - capped) <= 1e-4,
             f"score {score} does not equal the sum of credited rubric points ({capped})")

    policy = getattr(pack, "grading_policy", "")
    if policy == "wrong_choice_zero" and selection_correct is False:
        _add(rep, "wrong_choice_zero", score <= _EPS,
             f"wrong_choice_zero: wrong selection must score 0, got {score}")
    if policy == "choice_only":
        _add(rep, "choice_only", not credited_set,
             "choice_only: the question is decided by the selection alone, "
             f"but rubric items {sorted(credited_set)} were credited")

    gran = getattr(pack, "score_granularity", None)
    if gran:
        steps = score / float(gran)
        _add(rep, "score_granularity", abs(steps - round(steps)) <= 1e-6,
             f"score {score} is not a multiple of the configured granularity {gran}")
    return rep


def repair_arithmetic(g: Any, pack: Any) -> tuple[float, bool]:
    """The ONE sanctioned deterministic repair: recompute a question score
    from unambiguously declared rubric-item points. Returns (score, repaired).
    Never called by validation — the caller decides explicitly."""
    rep = check_question_invariants(g, pack)
    if rep.repairable_total is None:
        return float(getattr(g, "score", 0.0) or 0.0), False
    return rep.repairable_total, abs(float(g.score) - rep.repairable_total) > 1e-4


# --------------------------------------------------------------------------
# exam level (a finished ExamResult)
# --------------------------------------------------------------------------


def check_exam_invariants(result: Any, key: Any = None) -> InvariantReport:
    """``result`` is a schema.ExamResult (or the equivalent dict shape)."""
    rep = InvariantReport(True)
    questions = _get(result, "questions") or []
    key_max = {q.id: float(q.max_points) for q in (getattr(key, "questions", []) or [])} if key else {}

    total = 0.0
    for q in questions:
        qid = _get(q, "question_id")
        awarded = float(_get(q, "points_awarded") or 0.0)
        qmax = float(_get(q, "points_max") or 0.0)
        subs = _get(q, "sub_results") or _get(q, "sub_items") or []
        raw = 0.0
        for s in subs:
            sel = float(_get(s, "points_selection") or 0.0)
            exp = float(_get(s, "points_explanation") or 0.0)
            tot = float(_get(s, "points_total") or 0.0)
            smax = float(_get(s, "points_max") or 0.0)
            sid = _get(s, "sub_item_id")
            _add(rep, "sub_component_sum", abs(sel + exp - tot) <= 1e-4,
                 f"q{qid}/{sid}: components {sel}+{exp} != total {tot}")
            _add(rep, "sub_non_negative", tot >= -_EPS, f"q{qid}/{sid}: negative points {tot}")
            _add(rep, "sub_within_max", tot <= smax + 1e-4,
                 f"q{qid}/{sid}: {tot} exceeds sub-item maximum {smax}")
            raw += tot
        _add(rep, "question_cap", abs(awarded - min(raw, qmax)) <= 1e-4,
             f"q{qid}: awarded {awarded} != min(raw {round(raw, 4)}, cap {qmax})")
        if key_max:
            _add(rep, "question_max_matches_key", abs(qmax - key_max.get(qid, qmax)) <= 1e-6,
                 f"q{qid}: result maximum {qmax} differs from the key's {key_max.get(qid)}")
        total += awarded

    stated_total = float(_get(result, "total_awarded") or 0.0)
    rep.repairable_total = round(total, 4)
    _add(rep, "total_is_sum", abs(stated_total - total) <= 1e-4,
         f"total_awarded {stated_total} != sum of question scores {round(total, 4)}")
    stated_max = float(_get(result, "total_max") or 0.0)
    sum_max = round(sum(float(_get(q, "points_max") or 0.0) for q in questions), 4)
    _add(rep, "total_max_is_sum", abs(stated_max - sum_max) <= 1e-4,
         f"total_max {stated_max} != sum of question maxima {sum_max}")
    return rep


def recompute_exam_totals(result: Any) -> float:
    """Deterministic arithmetic only: the exam total IS the sum of the
    per-question scores. Returns the recomputed value (caller assigns)."""
    return round(sum(float(_get(q, "points_awarded") or 0.0)
                     for q in (_get(result, "questions") or [])), 4)


def _get(obj: Any, name: str):
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
