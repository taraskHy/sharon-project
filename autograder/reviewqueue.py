"""Review reason codes, prioritisation and mechanical grouping.

Three rules govern this module:

1. Every REVIEW item says WHY it exists, with a stable machine-readable code
   and a concise explanation rendered from STRUCTURED facts. No model is
   asked to explain itself in prose.
2. Priority changes the ORDER a lecturer sees cases in. It never changes a
   grade, and it never decides anything (asserted by tests).
3. Cases are grouped only by an EXACT mechanical fingerprint — the same
   unresolved variant marker, the same template mapping, the same alignment,
   the same missing-page structure. Semantically similar student answers are
   never grouped, so one lecturer decision can never leak into another
   student's judged answer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

# --------------------------------------------------------------------------
# §12 reason codes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ReasonSpec:
    code: str
    title: str
    tier: int              # base priority tier (see PRIORITY_TIERS)
    mechanical: bool       # may an identical case be resolved in one decision?


#: tier 0 systemic/package · 1 grade uncertainty · 2 MC · 3 OCR · 4 residual
PRIORITY_TIERS = {0: "package/systemic", 1: "grade uncertainty", 2: "unresolved selection",
                  3: "reading disagreement", 4: "isolated low-impact ambiguity"}

REASONS: dict[str, ReasonSpec] = {s.code: s for s in (
    ReasonSpec("PACKAGE_ANOMALY", "Package-level anomaly", 0, True),
    ReasonSpec("VARIANT_UNRESOLVED", "Exam variant not resolved", 0, True),
    ReasonSpec("ALIGNMENT_UNRESOLVED", "Question alignment not resolved", 0, True),
    ReasonSpec("PROVIDER_FAILED", "Model provider failed", 0, False),
    ReasonSpec("BUDGET_PAUSED", "Paused on budget", 0, False),
    ReasonSpec("GRADE_DISAGREEMENT", "Graders disagree", 1, False),
    ReasonSpec("GRADE_INVALID", "Grader output failed validation", 1, False),
    ReasonSpec("EVIDENCE_INVALID", "Cited evidence is not in the student's answer", 1, False),
    ReasonSpec("GRADE_UNCERTAIN", "Rubric application undecidable", 1, False),
    ReasonSpec("MC_CONFLICT", "Selection resolvers conflict", 2, False),
    ReasonSpec("MC_UNRESOLVED", "Selection undecidable", 2, False),
    ReasonSpec("OCR_PROVIDER_DISAGREEMENT", "Readings disagree", 3, False),
    ReasonSpec("OCR_UNRESOLVED", "Handwriting could not be read", 3, False),
)}

#: Legacy free-text reasons (persisted by earlier runs) -> codes.
_TEXT_RULES: tuple[tuple[str, str], ...] = (
    ("version detection", "VARIANT_UNRESOLVED"),
    ("variant", "VARIANT_UNRESOLVED"),
    ("alignment", "ALIGNMENT_UNRESOLVED"),
    ("evidence absent", "EVIDENCE_INVALID"),
    ("evidence", "EVIDENCE_INVALID"),
    ("budget", "BUDGET_PAUSED"),
    ("provider", "PROVIDER_FAILED"),
    ("verifier disagreement", "OCR_PROVIDER_DISAGREEMENT"),
    ("could not be read", "OCR_UNRESOLVED"),
    ("illegible", "OCR_UNRESOLVED"),
    ("untranscribed", "OCR_UNRESOLVED"),
    ("candidates", "MC_UNRESOLVED"),
    ("multiple live marks", "MC_UNRESOLVED"),
    ("ambiguous", "MC_UNRESOLVED"),
    ("disagreement", "GRADE_DISAGREEMENT"),
    ("outside", "GRADE_INVALID"),
    ("uncertain", "GRADE_UNCERTAIN"),
)


def classify_reason(text: str, kind: str | None = None) -> str:
    """Map a persisted free-text review reason onto a stable code. New code
    paths pass the code directly; this exists so older artefacts and the
    deterministic grader's prose reasons still get a typed code."""
    t = (text or "").lower()
    if t.upper() in REASONS:
        return t.upper()
    for needle, code in _TEXT_RULES:
        if needle in t:
            return code
    return {"mc": "MC_UNRESOLVED", "ocr": "OCR_UNRESOLVED", "variant": "VARIANT_UNRESOLVED",
            "grading": "GRADE_UNCERTAIN"}.get(kind or "", "GRADE_UNCERTAIN")


def render_explanation(code: str, facts: dict[str, Any] | None = None) -> str:
    """A short, structured, human-readable explanation. Deterministic text
    built from recorded facts — never a generated narrative."""
    f = dict(facts or {})
    spec = REASONS.get(code)
    head = spec.title if spec else code
    lines = [f"{code}", "", head]
    def add(label: str, value: Any):
        if value is not None and value != [] and value != "":
            lines.append(f"{label}: {value}")

    if code in ("MC_CONFLICT", "MC_UNRESOLVED"):
        add("Deterministic candidates", ", ".join(f.get("deterministic") or []) or None)
        add("Local resolver", f.get("local"))
        add("Cloud resolver", f.get("cloud"))
        add("State", f.get("state"))
        add("Resolution", f.get("resolution", "unresolved"))
        if f.get("wrong_choice_zero"):
            add("Impact", f"this question scores 0 on a wrong selection ({f.get('points_affected')} pts)")
    elif code in ("OCR_UNRESOLVED", "OCR_PROVIDER_DISAGREEMENT"):
        add("Primary reading", f.get("primary"))
        add("Second reading", f.get("secondary"))
        add("Disputed", ", ".join(f.get("disputed") or []) or None)
        add("Image quality", f.get("image_quality"))
    elif code in ("GRADE_DISAGREEMENT", "GRADE_INVALID", "GRADE_UNCERTAIN", "EVIDENCE_INVALID"):
        if f.get("primary_score") is not None and f.get("max_score") is not None:
            add("Primary", f"{f['primary_score']:g}/{f['max_score']:g}")
        if f.get("escalation_score") is not None and f.get("max_score") is not None:
            add("Escalation", f"{f['escalation_score']:g}/{f['max_score']:g}")
        add("Disputed rubric items", ", ".join(f.get("disputed_rubric_items") or []) or None)
        add("Unsupported evidence for", ", ".join(f.get("fabricated_items") or []) or None)
        add("Validation problems", "; ".join(f.get("problems") or [])[:300] or None)
    elif code in ("VARIANT_UNRESOLVED", "ALIGNMENT_UNRESOLVED", "PACKAGE_ANOMALY"):
        add("Marker seen", f.get("marker_seen"))
        add("Candidates", ", ".join(f.get("candidates") or []) or None)
        add("Affected students", f.get("students_affected"))
        add("Package fact", f.get("fact"))
    else:
        add("Detail", f.get("detail"))
    if f.get("batch_warning"):
        lines += ["", f"Batch warning: {f['batch_warning']}"]
    return "\n".join(lines).rstrip()


# --------------------------------------------------------------------------
# §13 mechanical fingerprints
# --------------------------------------------------------------------------

#: Only these mechanical kinds may be resolved once for every identical case.
MECHANICAL_KINDS = ("variant_marker", "template_mapping", "alignment", "page_structure",
                    "package_fact")


def mechanical_fingerprint(kind: str, **facts: Any) -> Optional[str]:
    """Exact-match fingerprint for a MECHANICAL cause. Returns None for
    anything semantic — similarity between student answers must never
    produce a shared decision."""
    if kind not in MECHANICAL_KINDS:
        return None
    payload = json.dumps({"kind": kind, **{k: facts[k] for k in sorted(facts)}},
                         sort_keys=True, ensure_ascii=False, default=str)
    return f"{kind}:{hashlib.sha256(payload.encode()).hexdigest()[:16]}"


# --------------------------------------------------------------------------
# review cases
# --------------------------------------------------------------------------


@dataclass
class ReviewCase:
    exam_id: str
    question_id: str
    sub_item_id: str
    reason_code: str
    points_affected: float = 0.0
    max_points: float = 0.0
    facts: dict[str, Any] = field(default_factory=dict)
    mechanical_kind: Optional[str] = None
    mechanical_fingerprint: Optional[str] = None
    students_affected: int = 1
    batch_warning_code: Optional[str] = None
    batch_warning_students: int = 0
    wrong_choice_zero: bool = False

    def __post_init__(self):
        if self.mechanical_fingerprint is None and self.mechanical_kind:
            self.mechanical_fingerprint = mechanical_fingerprint(
                self.mechanical_kind, **{k: v for k, v in sorted(self.facts.items())})

    @property
    def reusable(self) -> bool:
        """May one decision be applied to every EXACT matching case?"""
        spec = REASONS.get(self.reason_code)
        return bool(self.mechanical_fingerprint) and bool(spec and spec.mechanical)

    def explanation(self) -> str:
        f = dict(self.facts)
        f.setdefault("points_affected", self.points_affected)
        f.setdefault("wrong_choice_zero", self.wrong_choice_zero)
        if self.batch_warning_code:
            f["batch_warning"] = (f"{self.batch_warning_code} affects "
                                  f"{self.batch_warning_students} students — one decision here "
                                  "probably resolves all of them")
        return render_explanation(self.reason_code, f)

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update({"reusable": self.reusable, "priority": self.priority_key()[:2],
                  "explanation": self.explanation()})
        return d

    # -- §11 prioritisation --------------------------------------------------

    def tier(self, high_point_threshold: float = 4.0) -> int:
        spec = REASONS.get(self.reason_code)
        base = spec.tier if spec else 4
        # a systemic cause outranks everything, whatever the item-level reason
        if self.batch_warning_code and self.batch_warning_students > 1:
            return 0
        if base == 1 and self.points_affected < high_point_threshold:
            return 4          # low-point isolated grading ambiguity sorts last
        if base == 2 and self.wrong_choice_zero:
            return 2
        if base == 2:
            return 4 if self.points_affected < high_point_threshold else 2
        return base

    def priority_key(self, high_point_threshold: float = 4.0) -> tuple:
        return (self.tier(high_point_threshold),
                -max(self.students_affected, self.batch_warning_students),
                -round(self.points_affected, 4),
                self.exam_id, self.question_id, self.sub_item_id)


def prioritize(cases: Iterable[ReviewCase], *, high_point_threshold: float = 4.0) -> list[ReviewCase]:
    """Deterministic review order. Ordering ONLY — no grade is touched."""
    return sorted(cases, key=lambda c: c.priority_key(high_point_threshold))


# --------------------------------------------------------------------------
# §13 grouping
# --------------------------------------------------------------------------


@dataclass
class ReviewGroup:
    fingerprint: Optional[str]
    reason_code: str
    cases: list[ReviewCase]
    apply_to_all_eligible: bool
    scope: str = ""

    @property
    def size(self) -> int:
        return len(self.cases)

    @property
    def students(self) -> list[str]:
        return sorted({c.exam_id for c in self.cases})

    def as_dict(self) -> dict:
        return {"fingerprint": self.fingerprint, "reason_code": self.reason_code,
                "size": self.size, "students": self.students,
                "apply_to_all_eligible": self.apply_to_all_eligible, "scope": self.scope,
                "explanation": self.cases[0].explanation() if self.cases else ""}


def group_cases(cases: Iterable[ReviewCase]) -> list[ReviewGroup]:
    """Group by EXACT mechanical fingerprint. Everything else stays a group
    of one that can never be broadcast."""
    grouped: dict[str, list[ReviewCase]] = {}
    singles: list[ReviewCase] = []
    for c in cases:
        if c.reusable:
            grouped.setdefault(c.mechanical_fingerprint, []).append(c)
        else:
            singles.append(c)
    out = [ReviewGroup(fp, cs[0].reason_code, prioritize(cs), True,
                       scope=f"{cs[0].mechanical_kind} {fp}")
           for fp, cs in grouped.items()]
    out += [ReviewGroup(None, c.reason_code, [c], False, scope="single item") for c in singles]
    out.sort(key=lambda g: g.cases[0].priority_key())
    return out


def apply_scope(group: ReviewGroup) -> dict:
    """The persisted scope of an apply-to-all decision: exactly which cases a
    single lecturer decision covered, and on what fingerprint."""
    if not group.apply_to_all_eligible:
        raise ValueError("apply-to-all is allowed only for exact mechanical groups")
    return {"fingerprint": group.fingerprint, "reason_code": group.reason_code,
            "mechanical_kind": group.cases[0].mechanical_kind,
            "items": [{"exam_id": c.exam_id, "question_id": c.question_id,
                       "sub_item_id": c.sub_item_id} for c in group.cases]}


def queue_summary(cases: Iterable[ReviewCase]) -> dict:
    """What the lecturer sees before opening the queue."""
    cases = list(cases)
    groups = group_cases(cases)
    by_code: dict[str, int] = {}
    for c in cases:
        by_code[c.reason_code] = by_code.get(c.reason_code, 0) + 1
    reusable = [g for g in groups if g.apply_to_all_eligible]
    return {"cases": len(cases), "groups": len(groups),
            "decisions_required": len(groups),
            "cases_absorbed_by_grouping": sum(g.size - 1 for g in reusable),
            "by_reason": dict(sorted(by_code.items())),
            "tiers": {PRIORITY_TIERS[t]: sum(1 for c in cases if c.tier() == t)
                      for t in sorted(PRIORITY_TIERS)}}
