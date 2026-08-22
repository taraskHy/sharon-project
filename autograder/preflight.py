"""Package-level safety check, run ONCE after automatic discovery.

If the package itself is structurally unresolved — a variant with no
alignment, a rubric naming a question the key does not have, two printed
items mapped onto the same canonical id — then grading hundreds of student
exams produces hundreds of review items that all say the same thing.

So the package is validated first, and a structural defect surfaces as ONE
``PACKAGE_SETUP_REQUIRED`` with the exact unresolved facts. This is the
single highest-leverage reduction in lecturer work in the whole system.

Everything here is deterministic structure checking: no model, no student
exam, no I/O beyond what the caller passes in. Variants are handled
generically — an internal id is any stable token; nothing assumes a
particular marker vocabulary.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Optional

from .policies import POLICIES

READY = "READY"
SETUP_REQUIRED = "PACKAGE_SETUP_REQUIRED"

_VALID_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


#: Category per finding code. Blocking findings stop the batch ONCE at
#: package level; non-blocking ones are recorded and grading proceeds.
CATEGORIES = {
    "NO_VARIANTS": "structure", "INVALID_VARIANT_ID": "structure",
    "DUPLICATE_VARIANT_ID": "structure", "VARIANT_NOT_IN_KEY": "structure",
    "ALIGNMENT_UNRESOLVED": "alignment", "ALIGNMENT_UNKNOWN_QUESTION": "alignment",
    "ALIGNMENT_UNKNOWN_SUB_ITEM": "alignment", "ALIGNMENT_INCOMPLETE": "alignment",
    "DUPLICATE_CANONICAL_ASSIGNMENT": "alignment",
    "EMPTY_KEY": "key", "DUPLICATE_QUESTION_ID": "key", "DUPLICATE_SUB_ITEM_ID": "key",
    "MISSING_MAX_SCORE": "key", "MAX_SCORE_INCONSISTENT": "key",
    "MISSING_KEY_ANSWER": "key", "KEY_ANSWER_UNVERIFIED": "key",
    "RUBRIC_UNKNOWN_QUESTION": "key",
    "POLICY_UNKNOWN_QUESTION": "policy", "INVALID_POLICY": "policy",
    "POLICY_MISSING": "policy",
    "MISSING_CROP_REGION": "template", "TEMPLATE_MISSING": "template",
    "TOTAL_SCORE_UNDETERMINED": "structure", "TOTAL_SCORE_MISMATCH": "informational",
    "DISCOVERY_UNRESOLVED": "structure",
    "RAG_UNAVAILABLE": "informational", "VARIANT_DISTRIBUTION_UNUSUAL": "distribution",
}


@dataclass
class PreflightFact:
    code: str
    severity: str                # blocking | warning
    subject: str                 # variant / question / template / package
    subject_id: str
    message: str
    needed: str = ""             # what a human must supply to resolve it
    category: str = ""

    def __post_init__(self):
        self.category = self.category or CATEGORIES.get(self.code, "structure")

    @property
    def blocking(self) -> bool:
        return self.severity == "blocking"

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class PreflightReport:
    status: str = READY
    facts: list[PreflightFact] = field(default_factory=list)
    total_possible_score: Optional[float] = None
    variants: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == READY

    @property
    def blocking(self) -> list[PreflightFact]:
        return [f for f in self.facts if f.severity == "blocking"]

    @property
    def warnings(self) -> list[PreflightFact]:
        return [f for f in self.facts if f.severity == "warning"]

    def as_dict(self) -> dict:
        return {"status": self.status, "ok": self.ok,
                "total_possible_score": self.total_possible_score,
                "variants": list(self.variants), "checked": list(self.checked),
                "blocking": [f.as_dict() for f in self.blocking],
                "warnings": [f.as_dict() for f in self.warnings],
                "by_category": self.by_category()}

    def by_category(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for f in self.facts:
            slot = out.setdefault(f.category, {"blocking": 0, "warning": 0})
            slot[f.severity] = slot.get(f.severity, 0) + 1
        return out

    def summary(self) -> str:
        if self.ok:
            return (f"{READY}: {len(self.variants)} variant(s), total possible score "
                    f"{self.total_possible_score:g}"
                    + (f"; {len(self.warnings)} warning(s)" if self.warnings else ""))
        lines = [f"{SETUP_REQUIRED} — {len(self.blocking)} unresolved package fact(s):"]
        for f in self.blocking:
            lines.append(f"  [{f.code}] {f.subject} {f.subject_id}: {f.message}"
                         + (f"\n      needed: {f.needed}" if f.needed else ""))
        return "\n".join(lines)


def _add(rep: PreflightReport, check: str, code: str, severity: str, subject: str,
         subject_id: str, message: str, needed: str = "") -> None:
    if check not in rep.checked:
        rep.checked.append(check)
    rep.facts.append(PreflightFact(code, severity, subject, subject_id, message, needed))


def preflight_package(*, key, variants: Iterable[str] | None = None,
                      alignment: dict[str, Any] | None = None,
                      policies: dict[str, str] | None = None,
                      rubric_question_ids: Iterable[str] | None = None,
                      template: dict[str, Any] | None = None,
                      unresolved: Iterable[str] = (),
                      required_crops: dict[str, Any] | None = None) -> PreflightReport:
    """Validate one exam package before any student exam is graded.

    ``alignment`` maps variant -> {question_id -> {printed_sub_item: canonical_sub_item}}
    or the sentinel ``"unresolved"`` for a variant whose order is not known.
    """
    rep = PreflightReport()
    key_versions = list(getattr(key, "versions", []) or [])
    vs = list(variants) if variants is not None else key_versions
    rep.variants = vs
    key_qids = [q.id for q in key.questions]

    # -- variants ----------------------------------------------------------
    rep.checked.append("variant_ids_valid")
    if not vs:
        _add(rep, "variant_ids_valid", "NO_VARIANTS", "blocking", "package", "*",
             "the package defines no exam variant at all",
             "confirm the exam has a single variant, or supply the variant markers")
    for v in vs:
        if not isinstance(v, str) or not _VALID_ID.match(v or ""):
            _add(rep, "variant_ids_valid", "INVALID_VARIANT_ID", "blocking", "variant", str(v),
                 f"{v!r} is not a usable internal variant id",
                 "give the variant a stable id such as variant_1")
    dup = [v for v, n in Counter(vs).items() if n > 1]
    if dup:
        _add(rep, "variant_ids_valid", "DUPLICATE_VARIANT_ID", "blocking", "package", ", ".join(map(str, dup)),
             "the same variant id is defined more than once", "remove the duplicate definition")
    unknown = [v for v in vs if key_versions and v not in key_versions]
    if unknown:
        _add(rep, "variant_ids_valid", "VARIANT_NOT_IN_KEY", "blocking", "variant", ", ".join(unknown),
             f"detected variant(s) {unknown} are not versions of the answer key {key_versions}",
             "map each detected variant onto one of the key's versions")

    # -- alignment ---------------------------------------------------------
    rep.checked.append("alignment_complete")
    alignment = alignment or {}
    for v in vs:
        a = alignment.get(v)
        if a is None or a == "unresolved":
            _add(rep, "alignment_complete", "ALIGNMENT_UNRESOLVED", "blocking", "variant", str(v),
                 "the printed-to-canonical question mapping for this variant is unresolved",
                 "confirm this variant prints the key's order, or supply the mapping once")
            continue
        for qid, mapping in (a or {}).items():
            if qid not in key_qids:
                _add(rep, "alignment_complete", "ALIGNMENT_UNKNOWN_QUESTION", "blocking",
                     "variant", str(v), f"alignment refers to question {qid!r}, which the key does not define",
                     "correct the alignment or the answer key")
                continue
            if not isinstance(mapping, dict) or not mapping:
                _add(rep, "alignment_complete", "ALIGNMENT_UNRESOLVED", "blocking", "variant", str(v),
                     f"question {qid} has no usable mapping for this variant",
                     "supply the printed-to-canonical mapping for this question")
                continue
            targets = list(mapping.values())
            dups = sorted({t for t, n in Counter(targets).items() if n > 1})
            if dups:
                _add(rep, "alignment_complete", "DUPLICATE_CANONICAL_ASSIGNMENT", "blocking",
                     "variant", str(v),
                     f"question {qid}: canonical sub-item(s) {dups} are assigned to more than one "
                     "printed item — the mapping is not bijective",
                     "each printed item must map to exactly one canonical item")
            q = next(qq for qq in key.questions if qq.id == qid)
            key_subs = {s.id for s in q.sub_items}
            missing = sorted(key_subs - set(targets))
            extra = sorted(set(targets) - key_subs)
            if extra:
                _add(rep, "alignment_complete", "ALIGNMENT_UNKNOWN_SUB_ITEM", "blocking",
                     "variant", str(v),
                     f"question {qid}: mapping targets {extra} do not exist in the key",
                     "correct the mapping targets")
            if missing:
                _add(rep, "alignment_complete", "ALIGNMENT_INCOMPLETE", "blocking", "variant", str(v),
                     f"question {qid}: canonical sub-item(s) {missing} are not covered by the mapping",
                     "extend the mapping to cover every sub-item of this question")

    # -- key / rubric structure --------------------------------------------
    rep.checked.append("key_structure")
    if not key_qids:
        _add(rep, "key_structure", "EMPTY_KEY", "blocking", "package", "*",
             "the answer key defines no questions", "re-parse or supply the answer key")
    qdup = [q for q, n in Counter(key_qids).items() if n > 1]
    if qdup:
        _add(rep, "key_structure", "DUPLICATE_QUESTION_ID", "blocking", "package", ", ".join(qdup),
             "the answer key defines the same question id more than once",
             "give each question a unique id")
    for q in key.questions:
        sub_ids = [s.id for s in q.sub_items]
        sdup = [s for s, n in Counter(sub_ids).items() if n > 1]
        if sdup:
            _add(rep, "key_structure", "DUPLICATE_SUB_ITEM_ID", "blocking", "question", q.id,
                 f"sub-item id(s) {sdup} appear more than once", "give each sub-item a unique id")
        if q.max_points is None or float(q.max_points) <= 0:
            _add(rep, "key_structure", "MISSING_MAX_SCORE", "blocking", "question", q.id,
                 "the question has no usable maximum score", "set the question's maximum score")
        sub_total = sum(float(s.points or 0) for s in q.sub_items)
        if sub_total and float(q.max_points or 0) > sub_total + 1e-9:
            _add(rep, "key_structure", "MAX_SCORE_INCONSISTENT", "warning", "question", q.id,
                 f"the question maximum ({q.max_points:g}) exceeds the sum of its sub-items "
                 f"({sub_total:g}) — no student can reach it", "check the rubric's point split")
        for v in vs:
            for s in q.sub_items:
                answers = s.correct_by_version.get(v) or s.correct_by_version.get("default")
                if not answers:
                    _add(rep, "key_structure", "MISSING_KEY_ANSWER", "blocking", "question", q.id,
                         f"sub-item {s.id} has no accepted answer for variant {v}",
                         f"supply the correct answer for sub-item {s.id} on variant {v}")
                if v in (getattr(s, "versions_unverified", []) or []):
                    _add(rep, "key_structure", "KEY_ANSWER_UNVERIFIED", "blocking", "question", q.id,
                         f"sub-item {s.id}: the key's answer for variant {v} is not "
                         "deterministically verified",
                         "confirm this value against the official key once")

    if rubric_question_ids is not None:
        rep.checked.append("rubric_question_ids")
        for rq in rubric_question_ids:
            if rq not in key_qids:
                _add(rep, "rubric_question_ids", "RUBRIC_UNKNOWN_QUESTION", "blocking",
                     "question", str(rq),
                     f"the rubric describes question {rq!r}, which the answer key does not define",
                     "align the rubric and the answer key question numbering")

    # -- policies -----------------------------------------------------------
    rep.checked.append("grading_policies")
    policies = policies or {}
    for qid, pol in policies.items():
        if qid not in key_qids:
            _add(rep, "grading_policies", "POLICY_UNKNOWN_QUESTION", "blocking", "question", str(qid),
                 f"a grading policy is configured for question {qid!r}, which does not exist",
                 "remove the policy or correct the question id")
        elif pol not in POLICIES:
            _add(rep, "grading_policies", "INVALID_POLICY", "blocking", "question", str(qid),
                 f"grading policy {pol!r} is not one of {list(POLICIES)}",
                 "choose a supported grading policy")
    for qid in key_qids:
        if policies and qid not in policies:
            _add(rep, "grading_policies", "POLICY_MISSING", "warning", "question", qid,
                 "no grading policy was inferred for this question; the default will be used",
                 "confirm the intended policy for this question")

    # -- templates / crops ---------------------------------------------------
    if required_crops is not None:
        rep.checked.append("required_crops")
        for name, present in required_crops.items():
            if not present:
                _add(rep, "required_crops", "MISSING_CROP_REGION", "blocking", "template", str(name),
                     f"the template requires region {name!r}, which is not defined",
                     "define the region once for this package")
    if template is not None:
        rep.checked.append("template_present")
        if not template:
            _add(rep, "template_present", "TEMPLATE_MISSING", "warning", "template", "*",
                 "no page template was discovered for this package",
                 "confirm the answer-sheet layout once if extraction misreads pages")

    # -- deterministic total -------------------------------------------------
    rep.checked.append("total_score_deterministic")
    try:
        rep.total_possible_score = round(sum(float(q.max_points) for q in key.questions), 4)
    except (TypeError, ValueError):
        rep.total_possible_score = None
        _add(rep, "total_score_deterministic", "TOTAL_SCORE_UNDETERMINED", "blocking", "package", "*",
             "the exam's total possible score cannot be computed from the key",
             "fix the per-question maxima")
    stated = getattr(key, "total_points", None)
    if (rep.total_possible_score is not None and stated
            and abs(float(stated) - rep.total_possible_score) > 1e-6):
        _add(rep, "total_score_deterministic", "TOTAL_SCORE_MISMATCH", "warning", "package", "*",
             f"the key states a total of {float(stated):g} but the questions sum to "
             f"{rep.total_possible_score:g}; the per-question sum is authoritative",
             "check the key's stated total")

    # -- facts discovery could not settle ------------------------------------
    for u in unresolved or ():
        _add(rep, "discovery_unresolved", "DISCOVERY_UNRESOLVED", "blocking", "package", str(u),
             f"automatic discovery could not settle {u!r}",
             "resolve this package fact once")

    rep.status = SETUP_REQUIRED if rep.blocking else READY
    return rep


def alignment_from_discovery(fact_value: Any, variants: Iterable[str], key) -> dict[str, Any]:
    """Normalise the discovery/alignment.json contract into the shape
    ``preflight_package`` checks.

    ``None`` (single version) and ``{"identity": True}`` both mean "this
    variant prints the key's own order", which is complete by construction.
    """
    variants = list(variants)
    qmap = {q.id: {s.id: s.id for s in q.sub_items} for q in key.questions}
    if not fact_value:
        return {v: qmap for v in variants}
    out: dict[str, Any] = {}
    for v in variants:
        entry = fact_value.get(v) if isinstance(fact_value, dict) else None
        if entry is None:
            out[v] = "unresolved"
        elif entry is True or (isinstance(entry, dict) and entry.get("identity") is True):
            # `{"heart": true}` (prob sidecar) and `{"identity": true}` both mean
            # "this variant prints the key's own order"
            out[v] = qmap
        elif isinstance(entry, dict):
            out[v] = {qid: (qmap.get(qid, {}) if (m is True or (isinstance(m, dict) and m.get("identity") is True))
                            else m)
                      for qid, m in entry.items()}
        else:
            out[v] = "unresolved"
    return out


def reviews_avoided(report: PreflightReport, n_exams: int) -> int:
    """How many per-student review items this ONE package warning replaces —
    the number the lecturer would otherwise have had to work through."""
    return max(0, len(report.blocking) * max(n_exams, 0))


# --------------------------------------------------------------------------
# blocking gate (§6): stop ONCE at package level, never per student
# --------------------------------------------------------------------------


class PackageSetupRequired(RuntimeError):
    """The package is structurally unresolved. Raised ONCE, before any student
    exam is graded, so a single defect cannot become one review per student."""

    def __init__(self, report: PreflightReport):
        self.report = report
        super().__init__(report.summary())


def gate_package(report: PreflightReport) -> PreflightReport:
    """Raise on a blocking finding; return the report otherwise. Non-blocking
    findings (unusual-but-valid distribution, optional RAG unavailable,
    informational metadata) never stop grading."""
    if report.blocking:
        raise PackageSetupRequired(report)
    return report


def package_report_for_key(key, key_path=None, *, policies=None,
                           rubric_question_ids=None) -> PreflightReport:
    """Deterministic preflight from the answer key plus whatever sidecars sit
    next to it. Reads files only — no model, no discovery, no network."""
    import json
    from pathlib import Path

    alignment_raw = None
    template = None
    if key_path is not None:
        p = Path(key_path)
        stem = p.with_suffix("") if p.suffix.lower() == ".json" else p
        for suffix, target in (("alignment", "alignment"), ("template", "template")):
            cand = stem.with_name(stem.name + f".{suffix}.json")
            if not cand.exists() and p.name.endswith(".answer_key.json"):
                cand = p.with_name(p.name.replace(".json", f".{suffix}.json"))
            if cand.exists():
                try:
                    data = json.loads(cand.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001 — an unreadable sidecar is simply absent
                    continue
                if target == "alignment":
                    alignment_raw = data
                else:
                    template = data
    versions = list(getattr(key, "versions", []) or [])
    return preflight_package(
        key=key, variants=versions,
        alignment=alignment_from_discovery(alignment_raw, versions, key),
        policies=policies, rubric_question_ids=rubric_question_ids, template=template)
