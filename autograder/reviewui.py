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


def build_review_items(exam_id: str, result: dict, extraction: dict | None = None,
                       chain_traces: dict | None = None, packs: dict | None = None,
                       crops: dict | None = None) -> list[ReviewItem]:
    """Assemble review items from a persisted result (+ optional extraction,
    MC chain traces, grading packs, and crop bytes keyed by (q, sub))."""
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
    return items


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
                     *, decision: str, value=None) -> int:
        """Only VARIANT/LAYOUT decisions may be broadcast; semantic grading
        decisions are refused (raises ValueError)."""
        if kind not in ("variant", "layout"):
            raise ValueError("apply-to-all is allowed only for variant/layout decisions")
        n = 0
        for exam_dir in sorted(Path(job_dir).glob("exams/*")):
            if exam_dir.is_dir():
                ResolutionStore(exam_dir).resolve(question_id, sub_item_id, decision=decision, value=value,
                                                  by="lecturer:apply-to-all")
                n += 1
        return n


# ------------------------------------------------------ settings summary ----


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
