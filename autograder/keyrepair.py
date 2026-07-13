"""Deterministic decoding/verification of per-version answers in the key.

The exam family's answer key encodes each sub-item's per-version answers as
a letter group like ``F/F/G`` whose positions follow the key's own legend
("colours are R,B,G for A1,A2,A3"). Model parses of this encoding proved
unreliable (columns flattened to one letter), and the key document is
born-digital — so wherever the PDF **text layer** carries those groups, this
module decodes them deterministically and overrides the model's columns.

What cannot be verified deterministically (e.g. multiple-choice answers
encoded ONLY by highlight colour, which the text layer does not carry) is
either taken from a one-time explicit override file
(``<key stem>.versions-override.json``) or marked ``versions_unverified`` on
the sub-item, which the scorer turns into a human-review flag for exams of
the affected versions. Nothing is guessed, and the variant is never chosen
from student answers.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .schema import AnswerKey

_HEADER = re.compile(r"שאלה\s+מספר\s+(\d+)")


def load_key_text(key_path: str | Path) -> str:
    """The document's text layer only — no image rendering (fast; used to
    verify cached keys without paying the full page render)."""
    import fitz

    path = Path(key_path)
    if path.suffix.lower() == ".json" or not path.exists():
        return ""
    try:
        with fitz.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)
    except (RuntimeError, ValueError):
        return ""


def _group_pattern(n_versions: int) -> re.Pattern:
    # Option letters are A-I (matching answers use up to nine labeled
    # pyramids); the restriction also excludes stray axis labels ("ציר X").
    # PDF text extraction of RTL pages sometimes splits a group across
    # lines ("G" / "/F/F"), so whitespace incl. newlines is allowed around
    # the slashes; the count-vs-sub-items check still guards misdetection.
    return re.compile(r"\b([A-I](?:\s*/\s*[A-I]){%d})\b" % (n_versions - 1))


def _normalize_group(group: str) -> str:
    return re.sub(r"\s+", "", group)


def question_segments(key_text: str) -> dict[str, str]:
    """Split the key's text layer into per-question segments by the printed
    question headers, in document order."""
    matches = list(_HEADER.finditer(key_text))
    segments: dict[str, str] = {}
    for i, m in enumerate(matches):
        qid = m.group(1)
        end = matches[i + 1].start() if i + 1 < len(matches) else len(key_text)
        # Later headers of the same number (page repeats) extend the segment.
        segments[qid] = segments.get(qid, "") + key_text[m.start() : end]
    return segments


def override_path(key_path: str | Path) -> Path:
    key_path = Path(key_path)
    return key_path.with_name(key_path.stem + ".versions-override.json")


def load_overrides(key_path: str | Path) -> dict:
    """Optional one-time explicit mapping supplied by the operator:
    ``{"<question_id>": {"<sub_item_id>": {"A2": ["B"], ...}, ...}, ...}``.
    Entries are treated as verified."""
    path = override_path(key_path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain an object of question overrides")
    return data


def repair_key_versions(
    key: AnswerKey,
    key_text: str,
    expected_versions: list[str],
    overrides: dict | None = None,
) -> dict:
    """Decode/verify per-version answers in place. Returns an audit report:

    ``{"repaired": [...], "verified": [...], "unverified": [...],
       "overridden": [...], "notes": [...]}``

    Position order follows ``key.versions`` (the legend order the parser
    read); its SET must equal ``expected_versions``.
    """
    report = {"repaired": [], "verified": [], "unverified": [], "overridden": [], "notes": []}
    if not expected_versions:
        return report
    if sorted(key.versions) != sorted(expected_versions):
        report["notes"].append(
            f"key versions {key.versions} do not match expected {expected_versions}; "
            "repair skipped"
        )
        return report

    order = list(key.versions)
    n = len(order)
    pattern = _group_pattern(n)
    segments = question_segments(key_text)
    overrides = overrides or {}

    for q in key.questions:
        q_over = overrides.get(q.id, {})
        segment = segments.get(q.id, "")
        groups = pattern.findall(segment) if segment else []
        # Operator-overridden items are consumed first; positional decode
        # then only has to cover the remaining items (a fragmented/unreadable
        # group can be supplied via the override without forfeiting the
        # deterministic decode of every other item).
        positional_items = [s for s in q.sub_items if s.id not in q_over]
        by_position = groups if len(groups) == len(positional_items) else None
        if segment and by_position is None and groups:
            report["notes"].append(
                f"question {q.id}: {len(groups)} letter groups found for "
                f"{len(positional_items)} non-overridden sub-items — "
                "positional decode unsafe, skipped"
            )
        idx = -1
        for s in q.sub_items:
            if s.id in q_over:
                for v, answers in q_over[s.id].items():
                    s.correct_by_version[v] = list(answers)
                s.versions_unverified = []
                report["overridden"].append(f"{q.id}.{s.id}")
                continue
            idx += 1
            if by_position is not None:
                letters = _normalize_group(by_position[idx]).split("/")
                decoded = {v: [letters[i]] for i, v in enumerate(order)}
                if all(
                    sorted(s.correct_by_version.get(v, [])) == sorted(decoded[v])
                    for v in order
                ):
                    report["verified"].append(f"{q.id}.{s.id}")
                else:
                    # Keep any extra accepted alternatives the model found for
                    # a version whose primary letter matches; otherwise the
                    # deterministic letters win outright.
                    for v in order:
                        prev = s.correct_by_version.get(v, [])
                        s.correct_by_version[v] = (
                            sorted(set(prev) | set(decoded[v]))
                            if decoded[v][0] in prev
                            else decoded[v]
                        )
                    report["repaired"].append(f"{q.id}.{s.id}")
                s.versions_unverified = []
            else:
                # No deterministic source for this sub-item. If the model gave
                # every version the same answers we cannot tell decode
                # flattening from genuine agreement — mark every version
                # unverified. If versions differ, the model at least decoded
                # SOMETHING; still unverified (colour-only), but note it.
                s.versions_unverified = list(order)
                report["unverified"].append(f"{q.id}.{s.id}")

    return report
