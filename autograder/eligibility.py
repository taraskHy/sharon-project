"""ONE authoritative answer to: does this grading case's explanation need a
HUMAN ground-truth label, or does the exam's grading policy already decide the
score deterministically?

Wraps the production policy gate (``policies.decide_before_ocr``) — nothing
here re-implements policy semantics, and the labeling app must never grow its
own copy. Facts come from the dataset case records themselves (the student's
selected MC option, the key's ``correct_by_version``, the exam version, the
canonical grading policy, the wrong-answer rule) — NEVER from filenames,
historical totals, instructor red marks, or benchmark labels.

Semantics (mirrors ``decide_before_ocr`` exactly where an MC observation
exists):

* ``wrong_choice_zero`` + confidently wrong MC        -> deterministic 0, no human label
* ``explanation_required_if_correct`` + wrong MC + rule zero/selection
                                                      -> deterministic 0, no human label
* ``explanation_required_if_correct`` + wrong MC + rule ``process``
                                                      -> explanation still graded by a human
* ``explanation_can_rescue_wrong_choice``             -> always human-labelable
* ``choice_and_explanation_independent``              -> always human-labelable
* ``choice_only``                                     -> there is no explanation component;
                                                         never part of the human explanation queue
* unresolved / ambiguous MC                           -> NEVER a deterministic zero
* no MC observation at all (``selected`` and any MC state absent — e.g. the
  explanation-only grade_primary cells)               -> ``mc_state="absent"``; never a
                                                         deterministic zero
* accepted option(s) not determinable (no version and no ``default`` entry,
  or an empty accepted list)                          -> the MC cannot safely be judged
                                                         wrong; treated as unresolved
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .policies import MCResolution, POLICIES, decide_before_ocr, normalize_answer

#: stable, typed reason vocabulary (the only values ``reason`` may take)
ELIGIBILITY_REASONS = (
    "mc_correct_explanation_required",   # correct MC; policy demands the explanation be graded
    "wrong_mc_deterministic_zero",       # confidently wrong MC + zero/selection rule -> score 0, no human label
    "wrong_mc_rescue_allowed",           # wrong MC but the explanation can rescue -> human label
    "wrong_mc_process_rule",             # wrong MC but wrong_answer_rule == "process" -> human label
    "independent_explanation",           # explanation graded independently of the MC -> human label
    "mc_unresolved",                     # MC observed but not safely resolvable -> human label, never zero
    "mc_absent",                         # the case carries no MC observation at all -> human label
    "choice_only",                       # policy has no explanation component -> never in the queue
)


@dataclass(frozen=True)
class ExplanationLabelEligibility:
    """The one decision object the dataset builder, the labeling bundle, the
    labeling server and the benchmark importer all share."""

    eligible_for_human_label: bool
    deterministic_score: float | None    # set only when the policy decides the score
    reason: str                          # one of ELIGIBILITY_REASONS
    policy: str
    mc_state: str                        # "correct" | "wrong" | "unresolved" | "absent"
    selected_option: str | None
    accepted_options: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible_for_human_label": self.eligible_for_human_label,
            "deterministic_score": self.deterministic_score,
            "reason": self.reason,
            "policy": self.policy,
            "mc_state": self.mc_state,
            "selected_option": self.selected_option,
            "accepted_options": list(self.accepted_options),
        }


def decide_explanation_label_eligibility(*, policy: str, mc: MCResolution | None,
                                         accepted: list[str] | tuple[str, ...],
                                         points_selection: float, points_max: float,
                                         wrong_answer_rule: str = "zero",
                                         min_confidence: float = 0.9) -> ExplanationLabelEligibility:
    """Authoritative per-case decision. ``mc=None`` means the case carries no
    MC observation at all (not "blank" — a blank answer sheet is a real
    observation with state ``"blank"``)."""
    if policy not in POLICIES:
        raise ValueError(f"unknown grading policy {policy!r}")
    accepted = [a for a in (accepted or []) if a]
    if mc is None:
        if policy == "choice_only":
            return ExplanationLabelEligibility(False, None, "choice_only", policy, "absent", None, tuple(accepted))
        reason = "independent_explanation" if policy == "choice_and_explanation_independent" else "mc_absent"
        return ExplanationLabelEligibility(True, None, reason, policy, "absent", None, tuple(accepted))
    if not accepted:
        # The accepted option set could not be determined (unknown exam version,
        # or a key defect). DELIBERATE deviation from decide_before_ocr, which
        # would count any selection as wrong against an empty list: for LABELING
        # the MC "cannot safely be determined" (task contract: do not assume
        # wrong, do not drop the case), so we force the unresolved path — no
        # deterministic zero is possible here. (The legacy scorer likewise
        # treats an empty accepted set as a key defect needing review,
        # grade.py _grade_sub_item; the production gate's zero on empty
        # accepted is not a semantics we want baked into ground truth.)
        mc = MCResolution(selected=mc.selected, state="unclear", confidence=0.0,
                          source=mc.source, candidates=list(mc.candidates))
    decision = decide_before_ocr(policy=policy, mc=mc, accepted=list(accepted),
                                 points_selection=points_selection, points_max=points_max,
                                 wrong_answer_rule=wrong_answer_rule, min_confidence=min_confidence)
    resolved = mc.resolved(min_confidence)
    if resolved:
        mc_state = "correct" if decision.selection_correct else "wrong"
    else:
        mc_state = "unresolved"
    if decision.action == "score_locally":
        reason = "choice_only" if decision.persist_flag == "deterministic_choice_only" else "wrong_mc_deterministic_zero"
        return ExplanationLabelEligibility(False, decision.score, reason, policy, mc_state,
                                           mc.selected, tuple(accepted))
    if decision.action == "review":                      # choice_only + unresolved MC
        return ExplanationLabelEligibility(False, None, "choice_only", policy, mc_state,
                                           mc.selected, tuple(accepted))
    # action == "ocr_explanation": the explanation genuinely gets graded -> human label
    if policy == "choice_and_explanation_independent":
        reason = "independent_explanation"
    elif not resolved:
        reason = "mc_unresolved"
    elif mc_state == "wrong" and policy == "explanation_can_rescue_wrong_choice":
        reason = "wrong_mc_rescue_allowed"
    elif mc_state == "wrong":                            # explanation_required_if_correct + rule "process"
        reason = "wrong_mc_process_rule"
    else:
        reason = "mc_correct_explanation_required"
    return ExplanationLabelEligibility(True, None, reason, policy, mc_state, mc.selected, tuple(accepted))


# ------------------------------------------------------- dataset adapters --

def _normalize(letter: Any) -> str:
    return normalize_answer(str(letter))


def _accepted_for(pack: dict, sub_item_id: str | None, version: Any) -> list[str]:
    """The official accepted letters for this sub-item under this exam version,
    with production semantics (``grade._accepted``): exact version first, then
    the ``"default"`` entry. No version and no default -> undeterminable ([])."""
    cbv = pack.get("correct_by_version") or {}
    entry = cbv.get(sub_item_id) if sub_item_id else None
    if entry is None and len(cbv) == 1:                  # grade_primary packs are narrowed to one sub-item
        entry = next(iter(cbv.values()))
    entry = entry or {}
    raw = None
    if version is not None:
        raw = entry.get(str(version))
    if raw is None:
        raw = entry.get("default")
    return [_normalize(x) for x in (raw or [])]


def eligibility_for_case(input_row: dict, label_row: dict | None = None, *,
                         wrong_answer_rule: str = "zero",
                         min_confidence: float = 0.9) -> ExplanationLabelEligibility:
    """Eligibility for one declared-dataset grading case (``cases_inputs.jsonl``
    row + its evaluation-side ``cases_labels.jsonl`` row).

    MC facts are taken ONLY from the case record: ``selected``, ``version``,
    optional ``mc_state``/``mc_confidence`` observation fields, and the pack's
    ``correct_by_version``/``grading_policy``/``wrong_answer_rule``."""
    pack = input_row.get("pack") or {}
    policy = pack.get("grading_policy") or "choice_and_explanation_independent"
    rule = pack.get("wrong_answer_rule") or wrong_answer_rule
    sub_item_id = (label_row or {}).get("sub_item_id")
    selected = input_row.get("selected")
    version = input_row.get("version")
    accepted = _accepted_for(pack, sub_item_id, version)
    state = input_row.get("mc_state")
    confidence = input_row.get("mc_confidence")
    norm_selected = _normalize(selected) if selected is not None else None
    if selected is None and state is None:
        mc = None                                        # explanation-only case: no MC observation exists
    elif selected is not None and norm_selected is None:
        # A recorded mark that normalizes to nothing (punctuation/whitespace).
        # Production maps this to state "unclear" (grade._policy_gate /
        # reliability._mc_resolution: answered + empty normalized answer is
        # NOT a single_mark) — never a confident wrong, never a zero.
        mc = MCResolution(selected=None, state="unclear", confidence=0.0, source="dataset")
    else:
        mc = MCResolution(selected=norm_selected,
                          state=state or "single_mark",
                          confidence=float(confidence) if confidence is not None else 1.0,
                          source="dataset")
    max_score = float(pack.get("max_score") or (label_row or {}).get("max_score") or 0.0)
    return decide_explanation_label_eligibility(policy=policy, mc=mc, accepted=accepted,
                                                points_selection=max_score, points_max=max_score,
                                                wrong_answer_rule=rule, min_confidence=min_confidence)


def split_cases(inputs: Iterable[dict], labels_by_id: dict[str, dict], *,
                min_confidence: float = 0.9) -> tuple[list[tuple[dict, ExplanationLabelEligibility]],
                                                      list[tuple[dict, ExplanationLabelEligibility]]]:
    """Partition dataset cases into (human-labelable, policy-decided). Every
    case lands in exactly one list — no silent loss."""
    eligible: list[tuple[dict, ExplanationLabelEligibility]] = []
    excluded: list[tuple[dict, ExplanationLabelEligibility]] = []
    for row in inputs:
        e = eligibility_for_case(row, labels_by_id.get(row.get("case_id")), min_confidence=min_confidence)
        (eligible if e.eligible_for_human_label else excluded).append((row, e))
    return eligible, excluded


def eligibility_counts(decisions: Iterable[ExplanationLabelEligibility]) -> dict[str, Any]:
    """Bundle/manifest accounting block. Guarantees
    ``source_cases == human_labelable + deterministic_zero + excluded_choice_only``."""
    ds = list(decisions)
    by_reason: dict[str, int] = {}
    by_policy: dict[str, dict[str, int]] = {}
    for e in ds:
        by_reason[e.reason] = by_reason.get(e.reason, 0) + 1
        pol = by_policy.setdefault(e.policy, {})
        pol[e.reason] = pol.get(e.reason, 0) + 1
    counts = {
        "source_cases": len(ds),
        "human_labelable": sum(1 for e in ds if e.eligible_for_human_label),
        "deterministic_zero": by_reason.get("wrong_mc_deterministic_zero", 0),
        "excluded_choice_only": by_reason.get("choice_only", 0),
        "mc_unresolved": by_reason.get("mc_unresolved", 0),
        "mc_absent": by_reason.get("mc_absent", 0),
        "wrong_mc_rescue_allowed": by_reason.get("wrong_mc_rescue_allowed", 0),
        "wrong_mc_process_rule": by_reason.get("wrong_mc_process_rule", 0),
        "independent_explanation": by_reason.get("independent_explanation", 0),
        "mc_correct_explanation_required": by_reason.get("mc_correct_explanation_required", 0),
        "by_reason": dict(sorted(by_reason.items())),
        "by_policy": {k: dict(sorted(v.items())) for k, v in sorted(by_policy.items())},
    }
    assert counts["source_cases"] == counts["human_labelable"] + counts["deterministic_zero"] + counts["excluded_choice_only"]
    return counts


__all__ = ["ExplanationLabelEligibility", "ELIGIBILITY_REASONS",
           "decide_explanation_label_eligibility", "eligibility_for_case",
           "split_cases", "eligibility_counts"]
