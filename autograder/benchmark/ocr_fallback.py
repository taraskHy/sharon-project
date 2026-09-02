"""Pre-registered prospective OCR fallback: ``gemini_then_sonnet_hard_failure_fallback_v1``.

The rule is deliberately narrow. Sonnet is used ONLY when Gemini produced
nothing usable — never because Sonnet's text scores better, looks better, or
pleases a downstream grader. Those are all retrospective criteria that require
the reference, and using them would turn a deployable policy into an oracle
that cannot exist in production.

That constraint is enforced structurally: :func:`select` accepts only the two
outcome classifications and the two candidate texts. It has no parameter through
which a reference, a CER, a split label or a grade could arrive, so it cannot
consult one even by accident. ``tests/test_ocr_fallback.py`` additionally proves
that stripping every evaluation field from the inputs leaves each decision
byte-identical.

Triggers (any one is sufficient):

    provider_content_filter_failure
    provider_other_http_failure
    truncation with no usable output
    json_parse_failure / schema_failure with no usable output
    empty total output
    model_text_refusal (a bare unreadable marker against a readable reference)

A fallback result is not an endorsement: ``fallback_used`` rows carry provenance
and are flagged for review. Using the fallback does NOT authorise production
AUTO acceptance of the result.
"""
from __future__ import annotations

from typing import Any

POLICY_ID = "gemini_then_sonnet_hard_failure_fallback_v1"

#: The candidate slugs are NOT hardcoded here. A selection policy that names its
#: vendors cannot be reused, couples policy to a particular experiment, and (in
#: this repo) would put a vendor model id in production code, which the
#: vendor-independence test forbids. The slugs live in the frozen experiment
#: specification — data, not code — and are passed in.
PRIMARY_ROLE = "primary"
SECONDARY_ROLE = "secondary"

#: The outcome axes that make a primary result unusable. Every one is an
#: objective property of the response itself.
HARD_FAILURE_TRIGGERS: tuple[str, ...] = (
    "provider_content_filter_failure",
    "provider_other_http_failure",
    "truncation",
    "json_parse_failure",
    "schema_failure",
    "model_text_refusal",
)

#: Fields that must never influence the decision. Present only so a test can
#: assert their removal changes nothing.
FORBIDDEN_DECISION_INPUTS: tuple[str, ...] = (
    "reference", "frozen_reference", "cer", "wer", "split", "verdict",
    "grade", "score", "rubric", "official_solution", "expected", "target",
)


def is_hard_failure(outcome: dict[str, Any]) -> bool:
    """True when a candidate produced nothing usable for this crop.

    ``usable_transcription_returned`` is authoritative: anything else is a hard
    failure regardless of which trigger fired, so a new provider failure mode
    added later still routes to the fallback instead of silently passing.
    """
    if outcome.get("usable_transcription_returned") is True:
        return False
    return True


def which_trigger(outcome: dict[str, Any]) -> str | None:
    if outcome.get("usable_transcription_returned") is True:
        return None
    for t in HARD_FAILURE_TRIGGERS:
        if outcome.get(t) is True:
            return t
    return "empty_or_missing_output"


def select(*, case_id: str,
           primary_outcome: dict[str, Any], primary_text: str | None,
           secondary_outcome: dict[str, Any] | None = None,
           secondary_text: str | None = None,
           primary_model: str = PRIMARY_ROLE,
           secondary_model: str = SECONDARY_ROLE) -> dict[str, Any]:
    """Choose the prospective OCR result for one crop.

    No reference, metric, split or grade is a parameter here, by construction.
    """
    if not is_hard_failure(primary_outcome):
        return {"case_id": case_id, "chosen_model": primary_model, "chosen_text": primary_text,
                "fallback_used": False, "trigger": None, "resolved": True,
                "needs_review": False, "provenance": f"{POLICY_ID}:primary"}

    trigger = which_trigger(primary_outcome)
    if secondary_outcome is not None and not is_hard_failure(secondary_outcome):
        return {"case_id": case_id, "chosen_model": secondary_model, "chosen_text": secondary_text,
                "fallback_used": True, "trigger": trigger, "resolved": True,
                # a fallback result is never auto-acceptable on its own
                "needs_review": True, "provenance": f"{POLICY_ID}:fallback[{trigger}]"}

    return {"case_id": case_id, "chosen_model": None, "chosen_text": None,
            "fallback_used": False, "trigger": trigger, "resolved": False,
            "needs_review": True, "provenance": f"{POLICY_ID}:unresolved[{trigger}]"}


def replay(cases: list[str], primary: dict[str, dict], secondary: dict[str, dict],
           primary_text: dict[str, str | None],
           secondary_text: dict[str, str | None],
           primary_model: str = PRIMARY_ROLE,
           secondary_model: str = SECONDARY_ROLE) -> dict[str, Any]:
    """Apply the policy across a population and report coverage."""
    decisions = [select(case_id=c,
                        primary_outcome=primary[c], primary_text=primary_text.get(c),
                        secondary_outcome=secondary.get(c), secondary_text=secondary_text.get(c),
                        primary_model=primary_model, secondary_model=secondary_model)
                 for c in cases]
    n = len(decisions)
    used_primary = sum(1 for d in decisions if d["chosen_model"] == primary_model)
    used_fallback = sum(1 for d in decisions if d["fallback_used"])
    unresolved = sum(1 for d in decisions if not d["resolved"])
    return {
        "policy_id": POLICY_ID,
        "primary_model": primary_model, "secondary_model": secondary_model,
        "intended_crops": n,
        "primary_used": used_primary,
        "fallback_used": used_fallback,
        "unresolved": unresolved,
        "resolved_coverage": f"{n - unresolved}/{n}",
        "resolved_rate": round((n - unresolved) / n, 4) if n else None,
        "needs_review": sum(1 for d in decisions if d["needs_review"]),
        "triggers": {t: sum(1 for d in decisions if d["trigger"] == t)
                     for t in sorted({d["trigger"] for d in decisions if d["trigger"]})},
        "decisions": decisions,
    }


__all__ = ["POLICY_ID", "PRIMARY_ROLE", "SECONDARY_ROLE", "HARD_FAILURE_TRIGGERS",
           "FORBIDDEN_DECISION_INPUTS", "is_hard_failure", "which_trigger",
           "select", "replay"]
