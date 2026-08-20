"""Pre-run cost / query ESTIMATOR.

Before a grading job starts the lecturer should know roughly how much cloud
work it implies. This module derives that from the exam structure, the
per-question grading policies and the configured routes — with NO provider
call of any kind, and no pricing fetched from the internet.

Everything it returns is an ESTIMATE and is labelled as such. Actual usage
continues to come from the usage ledger; the two are never mixed.

The escalation rates are ASSUMPTIONS, defaulted conservatively and
overridable from an earlier run's measured metrics
(``EscalationAssumptions.from_metrics``). They have not been calibrated
empirically — the estimator reports which source it used so a number can
never be mistaken for a measurement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from .usage import CLOUD_BACKENDS

KIND = "ESTIMATE"
DISCLAIMER = ("ESTIMATE ONLY — derived from exam structure, policies and assumed "
              "escalation rates. Actual usage comes from the usage ledger.")


@dataclass
class EscalationAssumptions:
    """How often each fallback is expected to fire. Assumptions, not data."""

    mc_correct_rate: float = 0.5          # share of selections that are correct
    mc_ambiguous_rate: float = 0.08       # rows the deterministic reader cannot decide
    mc_local_resolution_rate: float = 0.60  # of those, the share the local model settles
    ocr_suspicion_rate: float = 0.25      # transcriptions that trip a suspicion signal
    blank_answer_rate: float = 0.05       # answers with nothing to transcribe
    grade_escalation_rate: float = 0.15   # gradings that fail validation and escalate
    cache_hit_rate: float = 0.0           # exact repeats already answered
    source: str = "default"

    @classmethod
    def from_metrics(cls, metrics: dict | None, **overrides) -> "EscalationAssumptions":
        """Build from an earlier run's measured rates where present. Missing
        keys keep the default, and the source is recorded honestly."""
        m = dict(metrics or {})
        kw: dict[str, Any] = {}
        for key, name in (("mc_ambiguous_rate", "mc_ambiguous_rate"),
                          ("ocr_suspicion_rate", "ocr_suspicion_rate"),
                          ("grade_escalation_rate", "grade_escalation_rate"),
                          ("cache_hit_rate", "cache_hit_rate"),
                          ("mc_local_resolution_rate", "local_resolution_success")):
            if m.get(name) is not None:
                v = float(m[name])
                kw[key] = v / 100.0 if v > 1 else v
        kw.update(overrides)
        return cls(**kw, source="historical" if kw else "default")


@dataclass
class TokenAssumptions:
    """Rough per-call token sizes. Image inputs dominate the OCR/MC tasks."""

    ocr_input: int = 1200
    ocr_output: int = 150
    ocr_verify_input: int = 1300
    ocr_verify_output: int = 60
    grade_input: int = 900
    grade_output: int = 120
    grade_escalate_input: int = 900
    grade_escalate_output: int = 400
    mc_input: int = 800
    mc_output: int = 40


#: call kind -> (task name, input-token field, output-token field)
CALL_KINDS = {
    "ocr": ("ocr_primary", "ocr_input", "ocr_output"),
    "ocr_verify": ("ocr_verify", "ocr_verify_input", "ocr_verify_output"),
    "grade_primary": ("grade_primary", "grade_input", "grade_output"),
    "grade_escalate": ("grade_escalate", "grade_escalate_input", "grade_escalate_output"),
    "mc_resolve_cloud": ("mc_resolve_cloud", "mc_input", "mc_output"),
}


@dataclass
class JobEstimate:
    kind: str = KIND
    disclaimer: str = DISCLAIMER
    exams: int = 0
    questions_total: int = 0
    sub_items_total: int = 0
    sub_items_deterministic: int = 0
    explanations_skipped: int = 0
    estimated_calls: dict[str, float] = field(default_factory=dict)
    estimated_input_tokens: int = 0
    estimated_output_tokens: int = 0
    estimated_cost: Optional[float] = None
    cost_unavailable_reason: Optional[str] = None
    per_exam: dict[str, Any] = field(default_factory=dict)
    assumptions: dict[str, Any] = field(default_factory=dict)
    by_question: list[dict] = field(default_factory=list)

    @property
    def estimated_cloud_calls(self) -> float:
        return round(sum(self.estimated_calls.values()), 1)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["estimated_cloud_calls"] = self.estimated_cloud_calls
        return d


def _question_plan(q, policy: str, a: EscalationAssumptions) -> dict:
    """Per-sub-item expected work for ONE question, as fractions."""
    n = max(len(q.sub_items), 1)
    explanation_relevant = bool(getattr(q, "explanation_required", False)
                                or float(getattr(q, "explanation_weight", 0.0) or 0.0) > 0
                                or str(getattr(q, "type", "")) in (
                                    "selection_with_explanation", "matching_with_explanation", "open"))
    if policy == "choice_only" or not explanation_relevant:
        ocr_share = 0.0
    elif policy == "wrong_choice_zero" or policy == "explanation_required_if_correct":
        ocr_share = a.mc_correct_rate          # only correct selections reach the explanation
    else:
        ocr_share = 1.0
    ocr_share *= (1.0 - a.blank_answer_rate)   # nothing written -> nothing to transcribe
    deterministic = n * (1.0 - ocr_share)
    return {"question_id": q.id, "policy": policy, "sub_items": n,
            "explanation_relevant": explanation_relevant,
            "ocr_items": round(n * ocr_share, 2),
            "deterministic_items": round(deterministic, 2),
            "explanations_skipped": round(deterministic if explanation_relevant else 0.0, 2)}


def estimate_job(*, key, exams: int, policies: dict[str, str] | None = None,
                 gateway=None, pricing: dict[str, dict] | None = None,
                 assumptions: EscalationAssumptions | None = None,
                 tokens: TokenAssumptions | None = None) -> JobEstimate:
    """Estimate the cloud work a job implies. Makes no calls of any kind."""
    a = assumptions or EscalationAssumptions()
    t = tokens or TokenAssumptions()
    policies = policies or {}
    est = JobEstimate(exams=exams, assumptions={**asdict(a), "tokens": asdict(t)})

    plans = [_question_plan(q, policies.get(q.id, "choice_and_explanation_independent"), a)
             for q in key.questions]
    est.by_question = plans
    est.questions_total = len(plans) * exams
    est.sub_items_total = int(sum(p["sub_items"] for p in plans) * exams)
    est.sub_items_deterministic = int(round(sum(p["deterministic_items"] for p in plans) * exams))
    est.explanations_skipped = int(round(sum(p["explanations_skipped"] for p in plans) * exams))

    ocr_items = sum(p["ocr_items"] for p in plans) * exams
    mc_items = sum(p["sub_items"] for p in plans) * exams
    ambiguous = mc_items * a.mc_ambiguous_rate
    keep = 1.0 - a.cache_hit_rate

    calls = {
        "ocr": ocr_items * keep,
        "ocr_verify": ocr_items * a.ocr_suspicion_rate * keep,
        "grade_primary": ocr_items * keep,
        "grade_escalate": ocr_items * a.grade_escalation_rate * keep,
        "mc_resolve_cloud": ambiguous * (1.0 - a.mc_local_resolution_rate) * keep,
    }
    est.estimated_calls = {k: round(v, 1) for k, v in calls.items()}

    in_tok = out_tok = 0.0
    for kind, n in calls.items():
        _, fi, fo = CALL_KINDS[kind]
        in_tok += n * getattr(t, fi)
        out_tok += n * getattr(t, fo)
    est.estimated_input_tokens = int(round(in_tok))
    est.estimated_output_tokens = int(round(out_tok))

    est.estimated_cost, est.cost_unavailable_reason = _estimate_cost(calls, t, gateway, pricing)
    if exams:
        est.per_exam = {"cloud_calls": round(est.estimated_cloud_calls / exams, 2),
                        "input_tokens": int(round(in_tok / exams)),
                        "output_tokens": int(round(out_tok / exams)),
                        "cost": (round(est.estimated_cost / exams, 4)
                                 if est.estimated_cost is not None else None)}
    return est


def _estimate_cost(calls: dict[str, float], t: TokenAssumptions, gateway,
                   pricing: dict[str, dict] | None) -> tuple[Optional[float], Optional[str]]:
    """Cost needs BOTH a route (which model runs the task) and local pricing
    for that model. Without either, call/token estimates stand on their own."""
    if not pricing:
        return None, "no pricing configured locally — showing calls and tokens only"
    if gateway is None:
        return None, "no model configuration supplied — cannot tell which models would run"
    total = 0.0
    missing: list[str] = []
    for kind, n in calls.items():
        task, fi, fo = CALL_KINDS[kind]
        route = gateway.routes.get(task)
        if route is None or not route.enabled:
            continue                              # task not configured: no cloud work
        if route.backend not in CLOUD_BACKENDS:
            continue                              # local model: no marginal cost
        p = pricing.get(route.model)
        if not p:
            missing.append(route.model)
            continue
        total += n * (getattr(t, fi) * float(p.get("input", 0.0))
                      + getattr(t, fo) * float(p.get("output", 0.0))) / 1_000_000.0
    if missing:
        return None, "no local pricing for: " + ", ".join(sorted(set(missing)))
    return round(total, 4), None


def load_pricing(config: dict | None) -> dict[str, dict]:
    """Pricing from the local models.toml ``[pricing]`` table:

        [pricing."vendor/model-slug"]
        input = 0.15     # USD per 1M input tokens
        output = 0.60    # USD per 1M output tokens

    Never fetched from the network; absent table = no cost estimate.
    """
    table = (config or {}).get("pricing") or {}
    return {str(model): {"input": float(v.get("input", 0.0)), "output": float(v.get("output", 0.0))}
            for model, v in table.items() if isinstance(v, dict)}
