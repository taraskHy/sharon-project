"""Testable back-end for the human REVIEW UI + the OpenRouter settings tab.

Review items are assembled from persisted job artefacts into three fast
shapes (OCR / MC / GRADING), each carrying exactly the evidence the
lecturer needs for a one-click decision. Resolutions persist to
<exam_dir>/review_resolutions.json (never mutating result.json in place),
and are re-applied by the pipeline on resume. VARIANT/LAYOUT decisions
that are exactly reusable expose apply-to-all; semantic grading decisions
never do.
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

from .reviewqueue import ReviewCase, classify_reason, group_cases, prioritize, queue_summary


@dataclass
class ReviewItem:
    kind: Literal["ocr", "mc", "grading", "variant"]
    exam_id: str
    question_id: str
    sub_item_id: str
    reason: str
    crop_png_b64: Optional[str] = None
    # OCR
    primary_transcription: Optional[str] = None
    secondary_transcription: Optional[str] = None
    disputed_regions: list[str] = field(default_factory=list)
    # MC
    deterministic_candidate: list[str] = field(default_factory=list)
    local_candidate: Optional[str] = None
    cloud_candidate: Optional[str] = None
    # grading
    selected_option: Optional[str] = None
    proposed_score: Optional[float] = None
    max_score: Optional[float] = None
    rubric_items: list[str] = field(default_factory=list)
    grading_evidence: Optional[str] = None
    question_context: Optional[str] = None
    # variant
    apply_to_all_eligible: bool = False
    options: list[str] = field(default_factory=list)   # one-click choices
    # queue metadata (reviewqueue: stable code, priority, mechanical grouping)
    reason_code: str = ""
    explanation: str = ""
    points_affected: float = 0.0
    priority_tier: int = 4
    group_fingerprint: Optional[str] = None
    batch_warning_code: Optional[str] = None
    batch_warning_students: int = 0


def build_review_items(exam_id: str, result: dict, extraction: dict | None = None,
                       chain_traces: dict | None = None, packs: dict | None = None,
                       crops: dict | None = None, warnings: list | None = None) -> list[ReviewItem]:
    """Assemble review items from a persisted result (+ optional extraction,
    MC chain traces, grading packs, crop bytes keyed by (q, sub), and the
    batch warnings that may explain several of them at once). Returned in
    review PRIORITY order (reviewqueue) — ordering only, never a grade."""
    items: list[ReviewItem] = []
    ext_index = {}
    for q in (extraction or {}).get("questions", []):
        for s in q.get("sub_items", []):
            ext_index[(q["question_id"], s["sub_item_id"])] = s
    crops = crops or {}
    for r in result.get("needs_human_review", []):
        qid, sid, reason = r.get("question_id"), r.get("sub_item_id"), r.get("reason", "")
        if qid == "*" and "version detection" in reason:
            items.append(ReviewItem("variant", exam_id, "*", "*", reason,
                                    apply_to_all_eligible=True,
                                    options=list(result.get("variant_detection", {}).get("options", []) or [])))
            continue
        se = ext_index.get((qid, sid), {})
        crop = crops.get((qid, sid))
        b64 = base64.standard_b64encode(crop).decode() if crop else None
        if se.get("status") == "ambiguous" or "multiple live marks" in reason or "resolution chain" in reason:
            trace = (chain_traces or {}).get((qid, sid), {})
            items.append(ReviewItem("mc", exam_id, qid, sid, reason, crop_png_b64=b64,
                                    deterministic_candidate=list(se.get("candidate_answers", [])),
                                    local_candidate=trace.get("local"), cloud_candidate=trace.get("cloud"),
                                    options=list(se.get("candidate_answers", [])) + ["blank", "unclear"]))
        elif "unintelligible" in reason or "could not be read" in reason or "verifier" in reason:
            items.append(ReviewItem("ocr", exam_id, qid, sid, reason, crop_png_b64=b64,
                                    primary_transcription=se.get("explanation_transcription"),
                                    secondary_transcription=(chain_traces or {}).get(("ocr", qid, sid)),
                                    disputed_regions=list((chain_traces or {}).get(("ocr_regions", qid, sid), [])),
                                    options=["accept primary", "accept secondary", "mark unreadable"]))
        else:
            pack = (packs or {}).get(qid)
            row = next((x for qq in result.get("questions", []) if qq.get("question_id") == qid
                        for x in qq.get("sub_items", []) if x.get("sub_item_id") == sid), {})
            items.append(ReviewItem("grading", exam_id, qid, sid, reason, crop_png_b64=b64,
                                    selected_option=row.get("student_answer"),
                                    proposed_score=row.get("points_total"), max_score=row.get("points_max"),
                                    rubric_items=list(pack.rubric_item_ids()) if pack else [],
                                    grading_evidence=(row.get("reason") or "")[:200],
                                    question_context=(pack.to_grader_context(include_solution=False)[:800] if pack else None),
                                    primary_transcription=row.get("explanation_transcription"),
                                    options=["accept proposed", "set score", "mark unintelligible"]))
    for it in items:
        annotate_item(it, warnings=warnings)
    items.sort(key=lambda i: (i.priority_tier, -i.batch_warning_students, -i.points_affected,
                              i.question_id, i.sub_item_id))
    return items


def annotate_item(item: ReviewItem, warnings: list | None = None) -> ReviewItem:
    """Attach the stable reason code, the structured explanation, the priority
    tier and the mechanical group fingerprint (see ``reviewqueue``)."""
    case = to_case(item, warnings=warnings)
    item.reason_code = case.reason_code
    item.explanation = case.explanation()
    item.points_affected = case.points_affected
    item.priority_tier = case.tier()
    item.group_fingerprint = case.mechanical_fingerprint
    item.batch_warning_code = case.batch_warning_code
    item.batch_warning_students = case.batch_warning_students
    item.apply_to_all_eligible = case.reusable
    return item


def to_case(item: ReviewItem, warnings: list | None = None) -> "ReviewCase":
    """Map one UI review item onto the typed queue case."""
    code = classify_reason(item.reason, item.kind)
    facts: dict = {}
    kind = None
    if item.kind == "mc":
        facts = {"deterministic": list(item.deterministic_candidate),
                 "local": item.local_candidate, "cloud": item.cloud_candidate}
        if item.local_candidate and item.cloud_candidate and item.local_candidate != item.cloud_candidate:
            code = "MC_CONFLICT"
    elif item.kind == "ocr":
        facts = {"primary": item.primary_transcription, "secondary": item.secondary_transcription,
                 "disputed": list(item.disputed_regions)}
        if item.secondary_transcription and item.secondary_transcription != item.primary_transcription:
            code = "OCR_PROVIDER_DISAGREEMENT"
    elif item.kind == "variant":
        kind = "variant_marker"
        facts = {"candidates": list(item.options), "fact": item.reason[:200]}
    else:
        facts = {"primary_score": item.proposed_score, "max_score": item.max_score,
                 "disputed_rubric_items": list(item.rubric_items),
                 "problems": [item.grading_evidence] if item.grading_evidence else []}
    points = float(item.max_score or 0.0) - float(item.proposed_score or 0.0) \
        if item.max_score is not None else 0.0
    case = ReviewCase(exam_id=item.exam_id, question_id=item.question_id,
                      sub_item_id=item.sub_item_id, reason_code=code,
                      points_affected=max(points, 0.0), max_points=float(item.max_score or 0.0),
                      facts=facts, mechanical_kind=kind)
    for w in (warnings or []):
        covers = (getattr(w, "scope", "") == "batch"
                  or (getattr(w, "scope", "") == "question"
                      and getattr(w, "scope_id", None) == item.question_id))
        if covers and getattr(w, "affected_students", 0) > case.batch_warning_students:
            case.batch_warning_code = getattr(w, "code", None)
            case.batch_warning_students = int(getattr(w, "affected_students", 0))
    return case


# ------------------------------------------------------------ resolutions --


class ResolutionStore:
    """Per-exam human resolutions; applied on resume; never edits result.json."""

    def __init__(self, exam_dir: str | Path):
        self.path = Path(exam_dir) / "review_resolutions.json"

    def load(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def resolve(self, question_id: str, sub_item_id: str, *, decision: str, value=None,
                by: str = "lecturer") -> dict:
        d = self.load()
        d[f"{question_id}:{sub_item_id}"] = {"decision": decision, "value": value, "by": by,
                                             "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        return d

    def apply_to_all(self, job_dir: str | Path, kind: str, question_id: str, sub_item_id: str,
                     *, decision: str, value=None, fingerprint: str | None = None,
                     scope: dict | None = None) -> int:
        """Only VARIANT/LAYOUT decisions may be broadcast; semantic grading
        decisions are refused (raises ValueError).

        ``scope`` (from ``reviewqueue.apply_scope``) restricts the broadcast to
        the exact matching mechanical cases and is persisted verbatim, so it is
        always auditable which decision covered which items and on what
        fingerprint."""
        if kind not in ("variant", "layout"):
            raise ValueError("apply-to-all is allowed only for variant/layout decisions")
        targets = None
        if scope:
            targets = {i["exam_id"] for i in scope.get("items", [])}
            fingerprint = fingerprint or scope.get("fingerprint")
        n, applied = 0, []
        for exam_dir in sorted(Path(job_dir).glob("exams/*")):
            if not exam_dir.is_dir() or (targets is not None and exam_dir.name not in targets):
                continue
            ResolutionStore(exam_dir).resolve(question_id, sub_item_id, decision=decision, value=value,
                                              by="lecturer:apply-to-all")
            applied.append(exam_dir.name)
            n += 1
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind, "decision": decision,
               "value": value, "fingerprint": fingerprint, "question_id": question_id,
               "sub_item_id": sub_item_id, "applied_to": applied, "scope": scope}
        p = Path(job_dir) / "apply_to_all.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return n


# ------------------------------------------------------ settings summary ----


def review_queue(items: list[ReviewItem]) -> dict:
    """What the REVIEW tab renders: prioritised items, mechanical groups, and
    how many decisions the lecturer actually has to make."""
    cases = [to_case(i) for i in items]
    groups = group_cases(cases)
    return {"items": [i for i in items],
            "summary": queue_summary(cases),
            "groups": [g.as_dict() for g in groups]}


def settings_summary(gateway=None, ledger=None, budget=None, cache=None,
                     openrouter_key_present: bool = False) -> dict:
    """Everything the settings tab shows. NEVER includes the key value."""
    tasks = gateway.describe() if gateway is not None else {}
    cloud_tasks = {t: r for t, r in tasks.items() if r.get("backend") == "openrouter"}
    return {
        "openrouter_enabled": bool(cloud_tasks),
        "key_present": openrouter_key_present,
        "tasks": tasks,
        "usage": ledger.aggregate() if ledger is not None else None,
        "cache": cache.stats() if cache is not None else None,
        "budget": budget.snapshot() if budget is not None else None,
    }


def test_connection(gateway, task: str = "grade_primary") -> dict:
    """Minimal-token connectivity probe: the backend's health check only —
    no completion request, no paid tokens."""
    try:
        be = gateway.backend_for(task)
        rep = be.health_check()
        return {"ok": rep.ok, "detail": rep.detail, "backend": rep.backend, "model": rep.model}
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        return {"ok": False, "detail": msg[:300], "backend": None, "model": None}


# ------------------------------------------------- batch view (UI backend) --


def load_job_results(job_dir: str | Path) -> dict[str, dict]:
    """Every persisted result.json in a job, keyed by internal exam id."""
    out: dict[str, dict] = {}
    for exam_dir in sorted(Path(job_dir).glob("exams/*")):
        p = exam_dir / "result.json"
        if p.is_file():
            try:
                out[exam_dir.name] = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 — a partial write is simply not there yet
                continue
    return out


def observations_from_results(results: dict[str, dict], extractions: dict[str, dict] | None = None):
    """Map persisted artefacts onto the batch-anomaly observation shapes."""
    from .anomaly import ExamObservation, ItemObservation

    items, exams = [], []
    extractions = extractions or {}
    for exam_id, result in results.items():
        variant = result.get("detected_version")
        detection = (result.get("version_detection") or "")
        exams.append(ExamObservation(
            exam_id=exam_id, variant=variant,
            variant_unknown=not variant or "UNCERTAIN" in detection,
            template=(result.get("backend_info") or {}).get("template")))
        pages = {}
        for qx in (extractions.get(exam_id) or {}).get("questions", []):
            for s in qx.get("sub_items", []):
                pages[(qx["question_id"], s["sub_item_id"])] = s.get("source_page")
        for q in result.get("questions", []):
            for s in q.get("sub_results", []):
                reason = (s.get("reason") or "").lower()
                text = s.get("explanation_transcription") or ""
                items.append(ItemObservation(
                    exam_id=exam_id, question_id=q.get("question_id", ""),
                    sub_item_id=s.get("sub_item_id", ""), variant=variant,
                    page=pages.get((q.get("question_id"), s.get("sub_item_id"))),
                    blank=s.get("status") == "unanswered",
                    ambiguous_mc=s.get("status") == "ambiguous",
                    ocr_failed=("could not be read" in reason or "illegible" in reason),
                    ocr_chars=len(text) if text else 0,
                    review=bool(s.get("needs_review")), review_reason=s.get("reason"),
                    score=s.get("points_total"), max_score=s.get("points_max")))
    return items, exams


def batch_overview(results: dict[str, dict], extractions: dict[str, dict] | None = None,
                   expected_variant_distribution: dict[str, float] | None = None) -> dict:
    """Everything the batch view shows: anomaly warnings first (one systemic
    warning can explain many individual reviews), then the prioritised,
    grouped review queue."""
    from .anomaly import detect_batch_anomalies

    items, exams = observations_from_results(results, extractions)
    warnings = detect_batch_anomalies(items, exams,
                                      expected_variant_distribution=expected_variant_distribution)
    review_items: list[ReviewItem] = []
    for exam_id, result in results.items():
        review_items += build_review_items(exam_id, result, (extractions or {}).get(exam_id),
                                           warnings=warnings)
    cases = [to_case(i, warnings=warnings) for i in review_items]
    return {"warnings": [w.as_dict() for w in warnings],
            "review_items": review_items,
            "summary": queue_summary(cases),
            "groups": [g.as_dict() for g in group_cases(cases)],
            "exams": len(results), "items": len(items)}


def decision_trace_for(exam_dir: str | Path, result: dict, question_id: str,
                       sub_item_id: str) -> str:
    """The compact "why did this get this grade?" view for ONE item.

    Prefers a recorded decision trace (decisions.jsonl). Falls back to the
    persisted result, which is always available, and says so — it never
    invents a route the pipeline did not record.
    """
    from .trace import DecisionRecord, DecisionTraceStore, StageRecord

    store = DecisionTraceStore(Path(exam_dir) / "decisions.jsonl")
    rows = store.read()
    # Stable lookup. Records are written with exam_id = exam label OR the exam
    # file stem, while this UI knows the output DIRECTORY name — an ad-hoc run
    # whose --out basename differs would hide its trace (audit finding). The
    # per-exam decisions.jsonl lives inside the exam dir, so try every stable
    # identifier, and when the file carries a single exam_id, accept it.
    candidates = [Path(exam_dir).name]
    exam_file = (result or {}).get("exam_file")
    if exam_file:
        candidates += [Path(str(exam_file)).stem, str(exam_file)]
    file_ids = {d.get("exam_id") for d in rows}
    if len(file_ids) == 1:
        candidates += list(file_ids)
    rec = None
    for cand in dict.fromkeys(candidates):
        # LAST match wins: a re-graded exam may leave superseded records; the
        # newest one describes the run that produced the current result.json.
        rec = next((d for d in reversed(rows)
                    if d.get("exam_id") == cand and d.get("question_id") == question_id
                    and (not sub_item_id or d.get("sub_item_id") == sub_item_id)), None)
        if rec:
            break
    if rec:
        stages = [StageRecord(**{k: v for k, v in s.items()
                                 if k in StageRecord.__dataclass_fields__})
                  for s in rec.get("stages", [])]
        text = DecisionRecord(**{**{k: v for k, v in rec.items()
                                    if k in DecisionRecord.__dataclass_fields__},
                                 "stages": stages}).explain()
        if (Path(exam_dir) / "shadow_comparison.json").exists():
            # The recorded route belongs to the SHADOW run: it is a proposal,
            # never the student's grade — label it so it cannot be confused
            # with the authoritative legacy result in result.json.
            return ("SHADOW / NON-AUTHORITATIVE\n"
                    "(this trace records the shadow reliability route's proposal; "
                    "the student's actual recorded grade is the legacy result in "
                    "result.json)\n\n" + text)
        return text
    row = next((s for q in result.get("questions", []) if q.get("question_id") == question_id
                for s in q.get("sub_results", []) if s.get("sub_item_id") == sub_item_id), None)
    if row is None:
        return "no decision record and no graded row for this item"
    state = "REVIEW" if row.get("needs_review") else "AUTO"
    lines = ["RECONSTRUCTED SUMMARY - NOT AN EXECUTION TRACE",
             "(this run recorded no decision trace: the fields below are read back "
             "from the persisted result, not from the route that produced it)",
             "",
             f"{Path(exam_dir).name} q{question_id}/{sub_item_id} -> {state}",
             f"  score: {row.get('points_total')}/{row.get('points_max')}",
             f"  selection: {row.get('student_answer')} (accepted: {row.get('accepted_answers')})",
             f"  reason: {(row.get('reason') or '')[:300]}"]
    ev = row.get("explanation_evaluation") or {}
    if ev:
        lines.append(f"  explanation verdict: {ev.get('verdict')}")
    return "\n".join(lines)
