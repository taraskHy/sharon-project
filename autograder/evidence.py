"""Evidence-grounded grading: a rubric item may not be credited on an
unsupported semantic assertion.

The grader returns, per rubric item, whether it is met and — for items whose
rubric spec requires it — a SHORT span copied from the frozen student
transcription. This module verifies deterministically that every such span
actually occurs in the transcription.

Two rules make this safe:

1. The student transcription is evidence and stays byte-for-byte immutable.
   Verification builds a temporary normalised COPY; nothing is ever written
   back (``test_transcription_not_mutated``).
2. Normalisation covers only harmless protocol differences — unicode form,
   bidi control characters, Hebrew niqqud, whitespace runs, quote/dash
   variants, Latin case, and the quoting punctuation a grader wraps a span
   in. It never reorders, drops or rewrites letters, so a fabricated
   sentence can never normalise into a real one.

A failed check does NOT correct the grade and does NOT touch the
transcription: the grading result is marked invalid and routed to grading
escalation (see ``escalation.validate_grade`` / ``signals.route``).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

#: Evidence spans are quotes, not explanations — keep them short.
MAX_EVIDENCE_CHARS = 200

#: Bidi/format controls and zero-width joiners: pure protocol noise in RTL text.
_INVISIBLE = re.compile(r"[​-‏‪-‮⁦-⁩﻿­]")
#: Hebrew points/cantillation — present or absent depending on the transcriber.
_HEBREW_MARKS = re.compile(r"[֑-ׇֽֿׁׂׅׄ]")
_WHITESPACE = re.compile(r"\s+")
#: Quote/apostrophe/dash variants that differ only by keyboard or font.
_UNIFY = {
    "׳": "'", "‘": "'", "’": "'", "ʼ": "'", "`": "'",
    "״": '"', "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-", "‐": "-", "‑": "-",
}
#: Punctuation a grader wraps a quoted span in — stripped from the EVIDENCE only.
_QUOTE_EDGE = " \t\n\r'\"()[]{}<>«»„“”‘’…,.;:!?-–—"


def normalize_for_evidence(text: str) -> str:
    """Return a comparison copy of ``text``. Never mutates the input."""
    s = unicodedata.normalize("NFKC", text or "")
    s = _INVISIBLE.sub("", s)
    s = _HEBREW_MARKS.sub("", s)
    s = "".join(_UNIFY.get(ch, ch) for ch in s)
    s = _WHITESPACE.sub(" ", s).strip()
    return s.casefold()


def evidence_supported(evidence: str, transcription: str) -> bool:
    """True when ``evidence`` occurs verbatim (modulo protocol differences)
    in ``transcription``. Empty evidence is never 'supported'."""
    e = normalize_for_evidence((evidence or "").strip(_QUOTE_EDGE))
    if not e:
        return False
    return e in normalize_for_evidence(transcription or "")


# --------------------------------------------------------------------------
# rubric-item level validation
# --------------------------------------------------------------------------


@dataclass
class CreditedItem:
    """One rubric item the grader awarded credit for."""

    id: str
    evidence: Optional[str] = None


@dataclass
class EvidenceValidation:
    ok: bool
    problems: list[str] = field(default_factory=list)
    checked: int = 0
    verified: list[str] = field(default_factory=list)
    fabricated: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    #: credit was awarded with nothing verifiable behind it (fail-closed rule)
    ungrounded_credit: bool = False

    def as_dict(self) -> dict:
        return {"ok": self.ok, "checked": self.checked, "verified": list(self.verified),
                "fabricated": list(self.fabricated), "missing": list(self.missing),
                "ungrounded_credit": self.ungrounded_credit,
                "problems": list(self.problems)}


def _requires_evidence(spec: Any) -> bool:
    """Duck-typed: a rubric spec may declare ``requires_evidence=False`` for
    items that are legitimately gradeable without a quoted span (e.g. "the
    student left the item blank"). Unknown items default to REQUIRED — the
    validation is never weakened globally."""
    if spec is None:
        return True
    return bool(getattr(spec, "requires_evidence", True))


def validate_evidence(*, credited: Iterable[CreditedItem], transcription: Optional[str],
                      specs: dict[str, Any] | None = None,
                      policy: str = "required",
                      credit_awarded: bool = False,
                      max_chars: int = MAX_EVIDENCE_CHARS) -> EvidenceValidation:
    """Deterministic check of every credited rubric item.

    policy: ``required`` (default) — credited items must cite a span that
    exists, unless their spec opts out; ``optional`` — cited spans are still
    verified, but a missing span is not a problem; ``disabled`` — no check
    (only for packs with no semantic rubric, e.g. choice_only).

    ``credit_awarded`` closes the rule. Iterating credited items alone is
    open at the bottom: a grader that awards a positive score while marking
    NO rubric item met has nothing to iterate, so it passed validation and
    went straight to AUTO. That is backwards — an ungrounded assertion of
    merit is exactly what "evidence required" exists to catch, and the models
    that most needed catching were the ones citing least. Measured over the
    26-case DEV population: 19/19 of one candidate's credit-awarding grades
    carried no verified span at all, and every one of them was AUTO.

    So under ``required``: credit must rest on at least one VERIFIED span,
    unless a credited item's spec explicitly opted out of evidence. No credit
    (score 0) demands no grounding — the grader is explaining an absence.
    """
    v = EvidenceValidation(True)
    if policy == "disabled":
        return v
    specs = specs or {}
    exempt: list[str] = []
    seen: set[str] = set()
    for item in credited:
        if item.id in seen:
            v.problems.append(f"rubric item {item.id} credited more than once")
            continue
        seen.add(item.id)
        spec = specs.get(item.id)
        text = (item.evidence or "").strip()
        if not text:
            if policy == "required" and _requires_evidence(spec):
                v.missing.append(item.id)
                v.problems.append(
                    f"rubric item {item.id} credited without student evidence")
            else:
                # the spec legitimately allows credit with no quoted span
                exempt.append(item.id)
            continue
        if len(text) > max_chars + 20:
            v.problems.append(f"rubric item {item.id} evidence exceeds {max_chars} characters")
        v.checked += 1
        if transcription is None:
            v.problems.append(
                f"rubric item {item.id} cites evidence but no transcription is available to verify it")
            continue
        if evidence_supported(text, transcription):
            v.verified.append(item.id)
        else:
            v.fabricated.append(item.id)
            v.problems.append(
                f"rubric item {item.id} cites evidence absent from the student "
                f"transcription: {text[:60]!r}")
    # FAIL CLOSED: credit that rests on nothing verifiable is not AUTO-able.
    #
    # ...but only where grounding is EXPRESSIBLE. A pack with no rubric items
    # declares nothing to cite, so no grader could ever satisfy the rule and
    # every positive score would be permanently REVIEW — a demand the model
    # cannot meet is not a safety check, it is a deadlock. Requiring `specs`
    # keeps the rule where it bites (packs that DO define rubric items) and
    # silent where it cannot.
    if policy == "required" and credit_awarded and specs and not v.verified and not exempt:
        v.ungrounded_credit = True
        # Wording note: this string travels into traces, which are asserted to
        # carry no student text or identity vocabulary. Keep it generic.
        v.problems.append(
            "credit awarded with no verified evidence span "
            "(no rubric item cites text that occurs in the transcription)")
    v.ok = not v.problems
    return v
