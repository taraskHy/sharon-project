"""SEEN-46 blind human-review campaign app.

A NEW human-reference campaign on top of the proven labeling infrastructure
(labeling_app.db.LabelDB / bundle.Bundle / backup): independent friends grade
the explanation quality of the 46 already-seen cases BLIND — they never see
the original instructor score, the local model's output, audit decisions,
split names, derivability, or each other's decisions, for the entire
campaign. Two independent reviews per case; disagreements go to a separate
adjudicated_human_reference; no source ever overwrites another.

The original instructor grade is a REFERENCE source (recorded, immutable,
compared later) — not infallible truth. Human reviewers are not automatically
infallible either. HELD_OUT is structurally absent: the bundle builder
refuses any case outside the frozen 46-case campaign manifest.

Decisions:  invalid | partially_valid | valid   (score 0 | M/2 | M)
            + confidence high|medium|low + issue flag + optional note.
The verdict is stored as the mapped score (the DB's agreement rule therefore
compares VERDICTS); confidence/issue/text travel in the label note as JSON
and never affect agreement.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

CAMPAIGN = "seen46_2026-08-28"
SCHEMA_VERSION = 1
PORT_DEFAULT = 8790

VERDICTS = ("invalid", "partially_valid", "valid")
CONFIDENCES = ("high", "medium", "low")
ISSUES = ("none", "transcription_evidence", "rubric_official_solution",
          "genuinely_ambiguous", "needs_source_page")

#: fields that must NEVER appear in a reviewer's pre-decision payload —
#: enforced by tests over the LIVE payload, not just by UI hiding
BLIND_FORBIDDEN_FIELDS = (
    "instructor", "actual_instructor_score", "instructor_derived_verdict",
    "model", "model_verdict", "model_score", "predicted_verdict", "justification",
    "audit", "audit_flag", "human_decision",
    "split", "DEV", "CALIBRATION", "HELD_OUT",
    "derivable", "strict", "expected", "agreement", "consensus",
    "final", "adjudicat",
)


def verdict_to_score(verdict: str, max_score: float) -> float:
    return {"invalid": 0.0, "partially_valid": max_score / 2, "valid": max_score}[verdict]


def score_to_verdict(score: float, max_score: float) -> str | None:
    for v in VERDICTS:
        if abs(verdict_to_score(v, max_score) - float(score)) < 1e-9:
            return v
    return None


def default_data_dir() -> Path:
    env = os.environ.get("REVIEW46_DATA_DIR")
    if env:
        return Path(env)
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / "autograder" / "review46"


def live_db_path() -> Path:
    """The deployment's review DB — a test must never open it (mirrors the
    labeling app's live-DB barrier; resolved WITHOUT REVIEW46_DATA_DIR on
    purpose so a redirected test cannot move the guard)."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return (Path(base) / "autograder" / "review46" / "labels.db").resolve()


def assert_not_live_review_db(path: Path) -> None:
    if not (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("PYTEST_VERSION")):
        return
    if str(os.environ.get("REVIEW46_ALLOW_LIVE_DB", "")).strip().lower() in ("1", "true", "yes", "on"):
        return
    live = live_db_path()
    try:
        resolved = Path(os.path.normpath(os.path.realpath(str(path))))
    except OSError:
        return
    same = resolved == live
    if not same:
        try:
            same = live.exists() and Path(path).exists() and os.path.samefile(str(path), str(live))
        except OSError:
            same = False
    if same:
        raise RuntimeError(
            f"refusing to open the LIVE review46 database from a test ({resolved}); "
            "point the test at tmp_path (REVIEW46_ALLOW_LIVE_DB=1 only for deliberate "
            "read-only forensics)")


def decision_note(verdict: str, confidence: str, issue: str, text: str) -> str:
    return json.dumps({"verdict": verdict, "confidence": confidence, "issue": issue,
                       "text": text}, ensure_ascii=False, sort_keys=True)


def parse_note(note: str | None) -> dict:
    try:
        d = json.loads(note or "")
        return d if isinstance(d, dict) else {}
    except Exception:  # noqa: BLE001 — a legacy/free-text note is not an error
        return {"text": note or ""}


__all__ = ["CAMPAIGN", "SCHEMA_VERSION", "PORT_DEFAULT", "VERDICTS", "CONFIDENCES", "ISSUES",
           "BLIND_FORBIDDEN_FIELDS", "verdict_to_score", "score_to_verdict",
           "default_data_dir", "live_db_path", "assert_not_live_review_db",
           "decision_note", "parse_note"]
