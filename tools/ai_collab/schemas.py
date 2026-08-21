"""Structured handoff (Claude -> orchestrator) and review (reviewer ->
orchestrator) documents: validation, normalization, lenient JSON parsing.

Validation is intentionally dependency-free (no jsonschema): it returns a
list of human-readable problems; an empty list means valid.
"""

from __future__ import annotations

import json

from .states import HANDOFF_STATUSES, SEVERITIES, VERDICTS, V_CHANGES_REQUIRED

# Documented schema shown to Claude verbatim (spec section 9).
HANDOFF_SCHEMA_DOC = """{
  "task_id": "<string, the task id you were given>",
  "round": <integer round number you were given>,
  "status": "READY_FOR_REVIEW" | "BLOCKED" | "USER_APPROVAL_REQUIRED",
  "summary": "<what you did / why you are blocked, 3-15 lines>",
  "files_changed": ["relative/path", ...],
  "tests": {"commands": ["..."], "passed": <int>, "failed": <int>},
  "architecture_changes": ["..."],
  "known_gaps": ["..."],
  "questions_for_reviewer": ["..."]
}"""

# Documented schema shown to the reviewer verbatim (spec section 11).
REVIEW_SCHEMA_DOC = """{
  "verdict": "APPROVED" | "CHANGES_REQUIRED" | "BLOCKED",
  "summary": "<3-10 line overall assessment>",
  "findings": [
    {
      "id": "F1",
      "severity": "critical" | "high" | "medium" | "low",
      "category": "correctness|integration|regression|security|privacy|resources|provider-boundary|tests|architecture|requirements",
      "file": "relative/path or empty",
      "line_or_symbol": "line number, function or symbol, or empty",
      "issue": "<one or two sentences: what is wrong>",
      "evidence": "<short quote/reference from the diff or files>",
      "requested_change": "<the concrete, bounded change you want>"
    }
  ],
  "approved_scope": ["<parts that are fine as-is>"],
  "tests_requested": ["<tests Claude should add or run>"],
  "context_requests": ["relative/path of source files you need to see, only if strictly necessary"]
}"""

_STR_LIST_FIELDS_HANDOFF = (
    "files_changed",
    "architecture_changes",
    "known_gaps",
    "questions_for_reviewer",
)


def _check_str_list(obj: dict, key: str, problems: list[str]) -> None:
    value = obj.get(key)
    if value is None:
        return
    if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
        problems.append(f"{key} must be a list of strings")


def validate_handoff(obj) -> list[str]:
    problems: list[str] = []
    if not isinstance(obj, dict):
        return ["handoff must be a JSON object"]
    if not isinstance(obj.get("task_id"), str) or not obj.get("task_id"):
        problems.append("task_id (non-empty string) is required")
    if not isinstance(obj.get("round"), int) or isinstance(obj.get("round"), bool):
        problems.append("round (integer) is required")
    status = obj.get("status")
    if status not in HANDOFF_STATUSES:
        problems.append(f"status must be one of {sorted(HANDOFF_STATUSES)}")
    if not isinstance(obj.get("summary"), str) or not obj.get("summary", "").strip():
        problems.append("summary (non-empty string) is required")
    for key in _STR_LIST_FIELDS_HANDOFF:
        _check_str_list(obj, key, problems)
    tests = obj.get("tests")
    if tests is not None:
        if not isinstance(tests, dict):
            problems.append("tests must be an object")
        else:
            if "commands" in tests:
                _check_str_list(tests, "commands", problems)
            for key in ("passed", "failed"):
                value = tests.get(key)
                if value is not None and (
                    not isinstance(value, int) or isinstance(value, bool)
                ):
                    problems.append(f"tests.{key} must be an integer")
    for key in ("base_commit", "head_commit"):
        value = obj.get(key)
        if value is not None and not isinstance(value, str):
            problems.append(f"{key} must be a string")
    return problems


def normalize_handoff(obj: dict) -> dict:
    out = dict(obj)
    for key in _STR_LIST_FIELDS_HANDOFF:
        out.setdefault(key, [])
    tests = out.get("tests") or {}
    tests.setdefault("commands", [])
    tests.setdefault("passed", 0)
    tests.setdefault("failed", 0)
    out["tests"] = tests
    out.setdefault("base_commit", "")
    out.setdefault("head_commit", "")
    return out


def validate_review(obj) -> list[str]:
    problems: list[str] = []
    if not isinstance(obj, dict):
        return ["review must be a JSON object"]
    verdict = obj.get("verdict")
    if not isinstance(verdict, str) or verdict.upper() not in VERDICTS:
        problems.append(f"verdict must be one of {sorted(VERDICTS)}")
        verdict = None
    else:
        verdict = verdict.upper()
    if not isinstance(obj.get("summary"), str) or not obj.get("summary", "").strip():
        problems.append("summary (non-empty string) is required")
    findings = obj.get("findings", [])
    if not isinstance(findings, list):
        problems.append("findings must be a list")
        findings = []
    for i, finding in enumerate(findings):
        where = f"findings[{i}]"
        if not isinstance(finding, dict):
            problems.append(f"{where} must be an object")
            continue
        if not isinstance(finding.get("id"), str) or not finding.get("id"):
            problems.append(f"{where}.id (non-empty string) is required")
        if finding.get("severity") not in SEVERITIES:
            problems.append(f"{where}.severity must be one of {list(SEVERITIES)}")
        for key in ("issue", "requested_change"):
            if not isinstance(finding.get(key), str) or not finding.get(key, "").strip():
                problems.append(f"{where}.{key} (non-empty string) is required")
        for key in ("category", "file", "line_or_symbol", "evidence"):
            value = finding.get(key)
            if value is not None and not isinstance(value, str):
                problems.append(f"{where}.{key} must be a string")
    if verdict == V_CHANGES_REQUIRED and not findings:
        problems.append("verdict CHANGES_REQUIRED requires at least one finding")
    for key in ("approved_scope", "tests_requested", "context_requests"):
        _check_str_list(obj, key, problems)
    return problems


def normalize_review(obj: dict) -> dict:
    out = dict(obj)
    out["verdict"] = str(out.get("verdict", "")).upper()
    out.setdefault("findings", [])
    for finding in out["findings"]:
        for key in ("category", "file", "line_or_symbol", "evidence"):
            finding.setdefault(key, "")
    for key in ("approved_scope", "tests_requested", "context_requests"):
        out.setdefault(key, [])
    return out


def parse_json_lenient(text: str):
    """Parse a JSON object out of *text*, tolerating fences and surrounding prose.

    Returns the parsed object or None. Deterministic: first balanced top-level
    object wins.
    """
    if not text:
        return None
    candidate = text.strip()
    # Strip a single markdown fence if the whole payload is fenced.
    if candidate.startswith("```"):
        first_newline = candidate.find("\n")
        if first_newline != -1 and candidate.rstrip().endswith("```"):
            candidate = candidate[first_newline + 1 :].rstrip()
            candidate = candidate[: candidate.rfind("```")].strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass
    # Scan for the first balanced {...} block, string-aware.
    start = candidate.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(candidate)):
            ch = candidate[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(candidate[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = candidate.find("{", start + 1)
    return None
