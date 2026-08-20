"""Decision records (§14) + full local early-exit accounting (§15).

One compact record per question route answers, without reading application
logs:

    "Why did this question receive this grade without human review?"
    "Which stages ran, which were skipped, and why?"
    "What did we NOT have to send to a cloud model, and what saved it?"

The record is written by the pipeline as it goes (``DecisionTrace``) and
frozen at the end (``DecisionRecord``). ``EarlyExitLedger`` aggregates the
saved work across a batch so the local-first optimisation stays first-class
and measurable instead of anecdotal.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

FINAL_STATES = ("AUTO", "ESCALATE", "REVIEW", "PAUSED")

#: Why model/cloud work was not needed. Each maps to real avoided calls.
SKIP_REASONS = (
    "deterministic_mc",              # confident deterministic MC resolution
    "choice_only",                   # policy: the selection decides the score
    "wrong_choice_zero",             # policy: wrong selection -> 0, stop
    "explanation_not_required",      # the rubric puts no points on the explanation
    "cache_hit",                     # exact request fingerprint already answered
    "reused_variant_decision",       # identical marker fingerprint decided earlier
    "reused_layout_decision",        # identical template/alignment fingerprint
    "deterministic_package_parse",   # the key/text layer settled it
    "local_resolver",                # a local model resolved it; no cloud call
    "persisted_result",              # a previous successful run's result
    "blank_crop",                    # nothing to transcribe (image triage)
    "no_suspicion_signal",           # the reading tripped no suspicion signal
    "ocr_unresolved",                # the reading is not trusted: grading not attempted
    "no_grading_pack",               # no question pack available for this question
)

_AVOIDED_KINDS = ("ocr", "grading", "mc", "cloud")


@dataclass
class StageRecord:
    stage: str
    status: str                      # executed | skipped | failed
    reason: str = ""
    task: Optional[str] = None
    model: Optional[str] = None
    cache_hit: Optional[bool] = None
    request_id: Optional[str] = None
    usage: dict[str, Any] = field(default_factory=dict)
    latency_s: Optional[float] = None
    cloud: bool = False              # did this stage call a REMOTE provider?
    skip_reason: Optional[str] = None
    avoided: dict[str, int] = field(default_factory=dict)


@dataclass
class DecisionRecord:
    exam_id: str
    question_id: str
    sub_item_id: str = ""
    item_id: str = ""                         # anonymous internal id (privacy.py)
    final_state: str = "AUTO"
    reason_code: str = "AUTO"
    reason: str = ""
    grading_policy: Optional[str] = None
    rag_policy: Optional[str] = None
    variant: Optional[str] = None
    variant_source: Optional[str] = None      # deterministic|local|cloud|human|reused
    alignment_source: Optional[str] = None
    pack_hash: Optional[str] = None
    mc_route: Optional[str] = None            # how the selection was settled
    ocr_status: Optional[str] = None          # signals.OCRStatus
    grade_status: Optional[str] = None        # signals.GradeStatus
    evidence: Optional[dict] = None           # evidence.EvidenceValidation.as_dict()
    invariants: Optional[dict] = None         # invariants.InvariantReport.as_dict()
    escalation: Optional[dict] = None         # {stage, outcome, score_delta, problems}
    proposed_score: Optional[float] = None    # the grader's number (never authoritative)
    stages: list[StageRecord] = field(default_factory=list)
    deterministic_decisions: list[str] = field(default_factory=list)
    signals: dict[str, Any] = field(default_factory=dict)
    points_awarded: Optional[float] = None
    points_max: Optional[float] = None
    ts: str = ""

    # -- queries -------------------------------------------------------------

    def executed(self) -> list[StageRecord]:
        return [s for s in self.stages if s.status == "executed"]

    def skipped(self) -> list[StageRecord]:
        return [s for s in self.stages if s.status == "skipped"]

    def why_skipped(self, stage: str) -> Optional[str]:
        for s in self.stages:
            if s.stage == stage and s.status == "skipped":
                return s.skip_reason or s.reason
        return None

    def model_calls(self) -> list[dict]:
        return [{"stage": s.stage, "task": s.task, "model": s.model, "cache_hit": s.cache_hit,
                 "request_id": s.request_id,
                 "tokens": s.usage.get("total_tokens"), "cost": s.usage.get("reported_cost")}
                for s in self.executed() if s.task]

    def avoided(self) -> dict[str, int]:
        out = {k: 0 for k in _AVOIDED_KINDS}
        for s in self.stages:
            for k, n in (s.avoided or {}).items():
                out[k] = out.get(k, 0) + int(n)
        return out

    def skip_reasons(self) -> list[str]:
        return [s.skip_reason for s in self.skipped() if s.skip_reason]

    @property
    def fully_local(self) -> bool:
        """True when no REMOTE provider call decided this item. A cache hit
        costs nothing and does not disqualify the item."""
        return not any(s.cloud and not s.cache_hit for s in self.executed())

    def as_dict(self) -> dict:
        d = asdict(self)
        d.update({"model_calls": self.model_calls(), "avoided": self.avoided(),
                  "skip_reasons": self.skip_reasons()})
        return d

    def explain(self) -> str:
        """The compact 'why was this auto-graded?' view."""
        lines = [f"{self.exam_id} q{self.question_id}"
                 + (f"/{self.sub_item_id}" if self.sub_item_id else "")
                 + f" -> {self.final_state} ({self.reason_code})"]
        if self.reason:
            lines.append(f"  reason: {self.reason}")
        if self.grading_policy:
            lines.append(f"  grading policy: {self.grading_policy}"
                         + (f" · RAG policy: {self.rag_policy}" if self.rag_policy else ""))
        if self.ocr_status or self.grade_status:
            lines.append(f"  status: OCR {self.ocr_status or '—'} · grading "
                         f"{self.grade_status or '—'}"
                         + (f" · selection {self.mc_route}" if self.mc_route else ""))
        if self.evidence:
            ev = self.evidence
            lines.append(f"  evidence: {len(ev.get('verified') or [])} verified, "
                         f"{len(ev.get('fabricated') or [])} unsupported, "
                         f"{len(ev.get('missing') or [])} missing")
        if self.invariants is not None and not self.invariants.get("ok", True):
            lines.append(f"  invariants FAILED: {'; '.join(self.invariants.get('problems', []))[:200]}")
        if self.escalation:
            e = self.escalation
            lines.append(f"  escalation: stage {e.get('stage')} -> {e.get('outcome')}"
                         + (f" (score delta {e.get('score_delta')})"
                            if e.get("score_delta") is not None else ""))
        if self.variant:
            lines.append(f"  variant: {self.variant} (source: {self.variant_source or '?'}"
                         + (f", alignment: {self.alignment_source}" if self.alignment_source else "") + ")")
        for d in self.deterministic_decisions:
            lines.append(f"  deterministic: {d}")
        for s in self.stages:
            if s.status == "executed":
                bits = [s.task or s.stage]
                if s.model:
                    bits.append(s.model)
                if s.cache_hit:
                    bits.append("cache hit")
                if s.request_id:
                    bits.append(f"req {s.request_id}")
                lines.append(f"  ran {s.stage}: " + ", ".join(bits))
            elif s.status == "skipped":
                lines.append(f"  skipped {s.stage}: {s.skip_reason or s.reason}")
            else:
                lines.append(f"  FAILED {s.stage}: {s.reason}")
        if self.points_max is not None:
            lines.append(f"  score: {self.points_awarded}/{self.points_max}")
        return "\n".join(lines)


class DecisionTrace:
    """Mutable builder used while the pipeline runs one question."""

    def __init__(self, exam_id: str, question_id: str, sub_item_id: str = "", **kw):
        self.record = DecisionRecord(exam_id=exam_id, question_id=question_id,
                                     sub_item_id=sub_item_id, **kw)

    # -- stage events --------------------------------------------------------

    def executed(self, stage: str, *, task: str | None = None, model: str | None = None,
                 cache_hit: bool | None = None, usage: dict | None = None,
                 request_id: str | None = None, latency_s: float | None = None,
                 cloud: bool = False, reason: str = "") -> "DecisionTrace":
        self.record.stages.append(StageRecord(stage, "executed", reason, task, model, cache_hit,
                                              request_id, dict(usage or {}), latency_s, cloud))
        return self

    def skipped(self, stage: str, skip_reason: str, *, detail: str = "",
                avoided: dict[str, int] | None = None) -> "DecisionTrace":
        if skip_reason not in SKIP_REASONS:
            raise ValueError(f"unknown skip reason {skip_reason!r}")
        self.record.stages.append(StageRecord(stage, "skipped", detail, skip_reason=skip_reason,
                                              avoided=dict(avoided or {})))
        return self

    def failed(self, stage: str, reason: str, *, task: str | None = None) -> "DecisionTrace":
        self.record.stages.append(StageRecord(stage, "failed", reason, task))
        return self

    def deterministic(self, what: str) -> "DecisionTrace":
        self.record.deterministic_decisions.append(what)
        return self

    def signals(self, signals: Any) -> "DecisionTrace":
        self.record.signals = signals.as_dict() if hasattr(signals, "as_dict") else dict(signals or {})
        return self

    def package(self, *, variant=None, variant_source=None, alignment_source=None,
                grading_policy=None, pack_hash=None, rag_policy=None) -> "DecisionTrace":
        r = self.record
        r.variant = variant if variant is not None else r.variant
        r.variant_source = variant_source or r.variant_source
        r.alignment_source = alignment_source or r.alignment_source
        r.grading_policy = grading_policy or r.grading_policy
        r.pack_hash = pack_hash or r.pack_hash
        r.rag_policy = rag_policy or r.rag_policy
        return self

    def statuses(self, *, mc_route=None, ocr_status=None, grade_status=None,
                 evidence=None, invariants=None, escalation=None,
                 proposed_score=None) -> "DecisionTrace":
        """Typed outcomes of the route's checks, recorded as they happen."""
        r = self.record
        r.mc_route = mc_route or r.mc_route
        r.ocr_status = ocr_status or r.ocr_status
        r.grade_status = grade_status or r.grade_status
        r.evidence = evidence if evidence is not None else r.evidence
        r.invariants = invariants if invariants is not None else r.invariants
        r.escalation = escalation if escalation is not None else r.escalation
        if proposed_score is not None:
            r.proposed_score = proposed_score
        return self

    def finish(self, final_state: str, reason_code: str, reason: str = "", *,
               points_awarded: float | None = None, points_max: float | None = None) -> DecisionRecord:
        if final_state not in FINAL_STATES:
            raise ValueError(f"unknown final state {final_state!r}")
        r = self.record
        r.final_state, r.reason_code, r.reason = final_state, reason_code, reason
        r.points_awarded, r.points_max = points_awarded, points_max
        r.ts = r.ts or time.strftime("%Y-%m-%d %H:%M:%S")
        return r


# --------------------------------------------------------------------------
# §15 early-exit accounting
# --------------------------------------------------------------------------


class EarlyExitLedger:
    """Aggregate of the model work the local-first path avoided."""

    def __init__(self):
        self.records: list[DecisionRecord] = []

    def add(self, record: DecisionRecord) -> DecisionRecord:
        self.records.append(record)
        return record

    def extend(self, records: Iterable[DecisionRecord]) -> None:
        for r in records:
            self.add(r)

    def as_dict(self) -> dict:
        n = len(self.records) or 1
        by_reason: dict[str, int] = {}
        avoided = {k: 0 for k in _AVOIDED_KINDS}
        explanations_skipped = 0
        reviews_avoided = 0
        cache_hits = 0
        for r in self.records:
            for s in r.stages:
                if s.status == "skipped" and s.skip_reason:
                    by_reason[s.skip_reason] = by_reason.get(s.skip_reason, 0) + 1
                    if s.stage in ("ocr_explanation", "explanation_judge", "ocr_primary"):
                        explanations_skipped += 1
                for k, v in (s.avoided or {}).items():
                    avoided[k] = avoided.get(k, 0) + int(v)
                if s.status == "executed" and s.cache_hit:
                    cache_hits += 1
            if r.final_state == "AUTO" and any(
                    s.status == "executed" and s.stage.startswith(("grade_escalate", "ocr_verify"))
                    for s in r.stages):
                reviews_avoided += 1     # escalation resolved what would have been a REVIEW
        fully_local = sum(1 for r in self.records if r.fully_local)
        return {
            "questions": len(self.records),
            "explanations_skipped": explanations_skipped,
            "ocr_calls_avoided": avoided.get("ocr", 0),
            "grading_calls_avoided": avoided.get("grading", 0),
            "mc_calls_avoided": avoided.get("mc", 0),
            "cloud_calls_avoided": avoided.get("cloud", 0),
            "review_cases_avoided": reviews_avoided,
            "cache_hits": cache_hits,
            "fully_local_questions": fully_local,
            "pct_graded_fully_locally": round(100 * fully_local / n, 1),
            "by_skip_reason": dict(sorted(by_reason.items())),
            "by_final_state": {s: sum(1 for r in self.records if r.final_state == s)
                               for s in FINAL_STATES},
        }


class DecisionTraceStore:
    """Append-only JSONL of decision records (numbers and ids; no page text)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: DecisionRecord) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.as_dict(), ensure_ascii=False, default=str) + "\n")

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def find(self, exam_id: str, question_id: str, sub_item_id: str = "") -> Optional[dict]:
        for d in self.read():
            if (d.get("exam_id") == exam_id and d.get("question_id") == question_id
                    and (not sub_item_id or d.get("sub_item_id") == sub_item_id)):
                return d
        return None


def record_from_early_exit(entry: dict, exam_id: str = "") -> DecisionRecord:
    """Bridge from ``grade.early_exit_log()`` entries to a decision record, so
    the already-validated policy gate contributes to the same accounting."""
    flag = (entry.get("flag") or "").replace("deterministic_", "")
    skip = {"choice_only": "choice_only", "zero_wrong_choice": "wrong_choice_zero"}.get(
        flag, "deterministic_mc")
    t = DecisionTrace(exam_id, entry.get("question_id", ""), entry.get("sub_item_id", ""),
                      grading_policy=entry.get("policy"))
    t.deterministic(entry.get("reason", ""))
    t.skipped("ocr_explanation", skip, detail=entry.get("reason", ""),
              avoided={"ocr": 1, "grading": 1, "cloud": 2})
    return t.finish("AUTO", "AUTO", entry.get("reason", ""))
