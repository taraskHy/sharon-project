"""Production canary suites: the MECHANISM only.

The contract this file defines:

    candidate model/config change
        -> run the frozen canary suite
        -> compare against the accepted baseline
        -> promote ONLY if the acceptance rules pass

Three independent suites are supported because the three tasks fail
differently: ``mc_resolver`` (did the same letter come out?), ``ocr``
(did the reading change?), ``grading`` (did the score/rubric decision
change?).

IMPORTANT: nothing here runs a model. ``run_suite`` takes an injected
runner, and no canary suite is populated in this repository — building one
means spending provider calls, which is a separate, deliberate decision on
hardware that can afford it. Until then this module gives the config
contract, the comparison and the promotion rule, all testable on mocks.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

CANARY_KINDS = ("mc_resolver", "ocr", "grading")


@dataclass
class CanaryCase:
    """One frozen input with its accepted answer. Inputs are referenced by
    hash/id, never by a path that could carry identity."""

    id: str
    kind: str
    input_ref: str                      # image hash / item id / pack hash
    expected: dict[str, Any] = field(default_factory=dict)
    note: str = ""


@dataclass
class AcceptanceRules:
    """When may a candidate configuration be promoted?"""

    max_regressions: int = 0            # cases that were right and became wrong
    min_agreement: float = 1.0          # share of cases matching the baseline
    max_score_delta: float = 0.0        # grading: allowed per-case score drift
    allow_new_uncertainty: bool = False  # may the candidate newly say "uncertain"?

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class CanarySuite:
    name: str
    kind: str
    cases: list[CanaryCase] = field(default_factory=list)
    acceptance: AcceptanceRules = field(default_factory=AcceptanceRules)
    baseline_provenance: dict[str, Any] = field(default_factory=dict)
    created: str = ""

    def __post_init__(self):
        if self.kind not in CANARY_KINDS:
            raise ValueError(f"unknown canary kind {self.kind!r} (expected one of {CANARY_KINDS})")

    def as_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CanarySuite":
        return cls(name=d["name"], kind=d["kind"],
                   cases=[CanaryCase(**c) for c in d.get("cases", [])],
                   acceptance=AcceptanceRules(**(d.get("acceptance") or {})),
                   baseline_provenance=dict(d.get("baseline_provenance") or {}),
                   created=d.get("created", ""))


@dataclass
class CanaryVerdict:
    promote: bool
    kind: str
    cases: int = 0
    matched: int = 0
    regressions: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    drift: list[str] = field(default_factory=list)

    @property
    def agreement(self) -> float:
        return round(self.matched / self.cases, 4) if self.cases else 0.0

    def as_dict(self) -> dict:
        d = asdict(self)
        d["agreement"] = self.agreement
        return d


# --------------------------------------------------------------------------


def _same(kind: str, expected: dict, got: dict, rules: AcceptanceRules) -> tuple[bool, str]:
    """Task-specific equivalence. Deliberately strict: a canary exists to
    catch change, not to tolerate it."""
    if kind == "mc_resolver":
        ok = (expected.get("selected") == got.get("selected")
              and expected.get("state", got.get("state")) == got.get("state"))
        return ok, f"selected {expected.get('selected')!r} -> {got.get('selected')!r}"
    if kind == "ocr":
        ok = (expected.get("text") or "").strip() == (got.get("text") or "").strip()
        return ok, "transcription text changed"
    # grading
    delta = abs(float(expected.get("score", 0)) - float(got.get("score", 0)))
    same_items = set(expected.get("rubric_items_met") or []) == set(got.get("rubric_items_met") or [])
    newly_uncertain = bool(got.get("uncertain")) and not bool(expected.get("uncertain"))
    ok = delta <= rules.max_score_delta and same_items and not (
        newly_uncertain and not rules.allow_new_uncertainty)
    why = []
    if delta > rules.max_score_delta:
        why.append(f"score {expected.get('score')} -> {got.get('score')}")
    if not same_items:
        why.append(f"rubric items {sorted(expected.get('rubric_items_met') or [])} -> "
                   f"{sorted(got.get('rubric_items_met') or [])}")
    if newly_uncertain:
        why.append("candidate newly reports uncertainty")
    return ok, "; ".join(why) or "changed"


def compare_to_baseline(suite: CanarySuite, results: dict[str, dict],
                        *, drift: Iterable[str] = ()) -> CanaryVerdict:
    """``results`` maps case id -> the candidate's output for that case."""
    v = CanaryVerdict(promote=False, kind=suite.kind, cases=len(suite.cases), drift=list(drift))
    for case in suite.cases:
        got = results.get(case.id)
        if got is None:
            v.regressions.append(f"{case.id}: no result from the candidate")
            continue
        ok, why = _same(suite.kind, case.expected, got, suite.acceptance)
        if ok:
            v.matched += 1
        else:
            v.regressions.append(f"{case.id}: {why}")
    r = suite.acceptance
    if not suite.cases:
        v.reasons.append("the canary suite is empty — it cannot support a promotion decision")
        return v
    if len(v.regressions) > r.max_regressions:
        v.reasons.append(f"{len(v.regressions)} regressions exceed the allowed {r.max_regressions}")
    if v.agreement < r.min_agreement:
        v.reasons.append(f"agreement {v.agreement:.0%} below the required {r.min_agreement:.0%}")
    v.promote = not v.reasons
    if v.promote and v.drift:
        v.reasons.append("promoted despite configuration drift: " + "; ".join(v.drift))
    return v


def run_suite(suite: CanarySuite, runner: Callable[[CanaryCase], dict]) -> dict[str, dict]:
    """Execute a suite through an INJECTED runner (a mock in tests, a real
    gateway call in production). This module never constructs a provider
    client and never decides that a canary should run."""
    return {c.id: runner(c) for c in suite.cases}


class CanaryStore:
    """Frozen suites + accepted baselines on disk (JSON, human-editable)."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, name: str) -> Path:
        return self.root / f"{name}.json"

    def save(self, suite: CanarySuite) -> Path:
        suite.created = suite.created or time.strftime("%Y-%m-%d %H:%M:%S")
        p = self._p(suite.name)
        p.write_text(json.dumps(suite.as_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        return p

    def load(self, name: str) -> Optional[CanarySuite]:
        p = self._p(name)
        if not p.exists():
            return None
        return CanarySuite.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def list_suites(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.json"))

    def record_verdict(self, name: str, verdict: CanaryVerdict,
                       candidate_provenance: dict | None = None) -> None:
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "suite": name,
               "verdict": verdict.as_dict(), "candidate": candidate_provenance or {}}
        with (self.root / "verdicts.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
