"""Automatic per-variant question/sub-item alignment (reordering discovery).

Variants of an exam may reorder questions or sub-items. This module derives
the permutation automatically and emits it in the EXISTING alignment
contract that variant.alignment_from_override validates (per variant:
{"identity": true} or {question_id: {printed_id: key_id, ...}}), so the
frozen pipeline consumes it unchanged.

Resolution ladder per (variant, question):
  1. deterministic text/fingerprint matching — canonical (key) sub-item
     prompts vs the variant booklet's numbered items (normalized token
     Jaccard + digit/Latin-token agreement); accepted only when the
     assignment is a clean bijection with unambiguous margins;
  2. deterministic structural matching — identical item count and
     identical text ⇒ identity; identical count with a unique best
     assignment ⇒ permutation;
  3. local model (task align_resolve) proposes a permutation from the two
     text lists;
  4. targeted cloud (align_resolve_cloud);
  5. human.
Never guesses: an ambiguous assignment escalates rather than being
emitted. Variant ids are opaque strings; nothing here knows what a marker
looks like.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field

from .schema import AnswerKey, KeyQuestion

# --------------------------------------------------------- text tooling ------

_TOKEN = re.compile(r"[A-Za-z0-9]+|[֐-׿]+", re.UNICODE)
_ITEM_HEAD = re.compile(r"^\s*(?:\(?([0-9]{1,3}|[א-ת]{1,2}|[A-Za-z])[.)\]:\-]\s+)(.+)$")


def _norm_tokens(s: str) -> set[str]:
    return {t.lower() for t in _TOKEN.findall(s or "") if len(t) > 1}


def _sim(a: str, b: str) -> float:
    ta, tb = _norm_tokens(a), _norm_tokens(b)
    if not ta or not tb:
        return 0.0
    j = len(ta & tb) / len(ta | tb)
    # short technical/numeric tokens carry disproportionate identity
    da = {t for t in ta if re.search(r"\d|[A-Za-z]", t)}
    db = {t for t in tb if re.search(r"\d|[A-Za-z]", t)}
    if da or db:
        j = 0.7 * j + 0.3 * (len(da & db) / max(len(da | db), 1))
    return j


def split_numbered_items(block: str) -> list[tuple[str, str]]:
    """Parse '1. text' / '(א) text' style lines into (printed_id, text)."""
    out = []
    for line in (block or "").splitlines():
        m = _ITEM_HEAD.match(line)
        if m:
            out.append((m.group(1), m.group(2).strip()))
        elif out and line.strip():
            pid, txt = out[-1]
            out[-1] = (pid, txt + " " + line.strip())   # continuation line
    return out


# ------------------------------------------------------------ results ------


@dataclass
class QuestionAlignmentResult:
    question_id: str
    mapping: Optional[dict[str, str]]     # printed_id -> key_id (None = unresolved)
    identity: bool
    source: str                            # deterministic_text | deterministic_structure | local_model | cloud_model | unresolved
    evidence: str = ""
    margin: float = 0.0


@dataclass
class VariantAlignmentResult:
    variant: str
    questions: dict[str, QuestionAlignmentResult] = field(default_factory=dict)

    def resolved(self) -> bool:
        return all(q.mapping is not None for q in self.questions.values())

    def to_contract_entry(self) -> dict | None:
        """The alignment.json entry for this variant, or None if unresolved."""
        if not self.resolved():
            return None
        if all(q.identity for q in self.questions.values()):
            return {"identity": True}
        return {qid: ({"identity": True} if q.identity else dict(q.mapping)) for qid, q in self.questions.items()}


# ---------------------------------------------------- deterministic stage ---


def deterministic_align_question(q: KeyQuestion, printed_items: list[tuple[str, str]],
                                 *, min_score: float = 0.35, min_margin: float = 0.15) -> QuestionAlignmentResult:
    """Text-fingerprint assignment key sub-items <- printed items."""
    key_items = [(s.id, s.prompt or "") for s in q.sub_items]
    if len(printed_items) != len(key_items):
        return QuestionAlignmentResult(q.id, None, False, "unresolved",
                                       f"item count differs (key {len(key_items)} vs printed {len(printed_items)})")
    # identical text in identical order -> identity (structural fast path)
    if all(_sim(kt, pt) >= 0.999 for (_, kt), (_, pt) in zip(key_items, printed_items)) and key_items:
        return QuestionAlignmentResult(q.id, {p: k for (k, _), (p, _) in zip(key_items, printed_items)},
                                       True, "deterministic_structure", "identical text and order", 1.0)
    # greedy-unique assignment with margin checks (n is small; O(n^2))
    scores = {(p, k): _sim(kt, pt) for (k, kt) in key_items for (p, pt) in printed_items}
    mapping: dict[str, str] = {}
    used_keys: set[str] = set()
    worst_margin = 1.0
    for p, pt in printed_items:
        cands = sorted(((scores[(p, k)], k) for (k, _) in key_items if k not in used_keys), reverse=True)
        if not cands:
            break
        best, k = cands[0]
        second = cands[1][0] if len(cands) > 1 else 0.0
        if best < min_score:
            return QuestionAlignmentResult(q.id, None, False, "unresolved",
                                           f"printed item {p!r} matches no key sub-item (best {best:.2f})")
        if best - second < min_margin:
            return QuestionAlignmentResult(q.id, None, False, "unresolved",
                                           f"printed item {p!r} ambiguous between key items (margin {best-second:.2f})")
        mapping[p] = k
        used_keys.add(k)
        worst_margin = min(worst_margin, best - second)
    if len(mapping) != len(key_items) or set(mapping.values()) != {k for k, _ in key_items}:
        return QuestionAlignmentResult(q.id, None, False, "unresolved", "assignment is not a bijection")
    identity = all(p == k for p, k in mapping.items())
    return QuestionAlignmentResult(q.id, mapping, identity, "deterministic_text",
                                   f"unique text assignment (worst margin {worst_margin:.2f})", worst_margin)


# --------------------------------------------------------- model stages -----


class PermutationProposal(BaseModel):
    question_id: str
    printed_to_key: dict[str, str] = Field(description="printed item id -> canonical key sub-item id")
    confident: bool = False
    notes: Optional[str] = None


ALIGN_SYSTEM = (
    "You are given the canonical list of sub-items of ONE exam question (id + "
    "text) and the same question as printed in another exam variant (printed "
    "id + text), possibly reordered or lightly reworded. Return the mapping "
    "printed id -> canonical id as a complete one-to-one assignment. Set "
    "confident=false if any pairing is not clearly determined by the texts. "
    "Reply with ONLY the JSON object."
)


def model_align_question(q: KeyQuestion, printed_items: list[tuple[str, str]], gateway, *,
                         variant: str, meta: dict | None = None,
                         local_task: str = "align_resolve", cloud_task: str = "align_resolve_cloud") -> QuestionAlignmentResult:
    key_ids = {s.id for s in q.sub_items}
    printed_ids = {p for p, _ in printed_items}
    text = ("Canonical sub-items:\n" + "\n".join(f"  {s.id}: {s.prompt}" for s in q.sub_items)
            + "\nPrinted in this variant:\n" + "\n".join(f"  {p}: {t}" for p, t in printed_items))
    for task, src in ((local_task, "local_model"), (cloud_task, "cloud_model")):
        try:
            gateway.route(task)
        except Exception:  # noqa: BLE001
            continue
        try:
            prop = gateway.call(task=task, system=ALIGN_SYSTEM,
                                content_blocks=[{"type": "text", "text": text}],
                                output_model=PermutationProposal,
                                meta={**(meta or {}), "stage": "discovery", "question_id": q.id}).value
        except Exception:  # noqa: BLE001
            continue
        m = {str(k): str(v) for k, v in prop.printed_to_key.items()}
        if prop.confident and set(m) == printed_ids and set(m.values()) == key_ids and len(set(m.values())) == len(m):
            return QuestionAlignmentResult(q.id, m, all(p == k for p, k in m.items()), src,
                                           f"complete confident permutation from {src}")
    return QuestionAlignmentResult(q.id, None, False, "unresolved", "no confident complete permutation from models")


# ------------------------------------------------------------ orchestrate ---


def align_variant(key: AnswerKey, variant: str, printed_by_question: dict[str, list[tuple[str, str]]],
                  gateway=None, meta: dict | None = None) -> VariantAlignmentResult:
    """printed_by_question: question_id -> [(printed_id, text)] as read from
    THIS variant's booklet (text layer or OCR of printed pages)."""
    out = VariantAlignmentResult(variant)
    for q in key.questions:
        printed = printed_by_question.get(q.id)
        if printed is None:
            out.questions[q.id] = QuestionAlignmentResult(q.id, None, False, "unresolved",
                                                          "no printed items available for this question")
            continue
        r = deterministic_align_question(q, printed)
        if r.mapping is None and gateway is not None:
            r = model_align_question(q, printed, gateway, variant=variant, meta=meta)
        out.questions[q.id] = r
    return out


def alignment_contract(results: dict[str, VariantAlignmentResult]) -> tuple[dict, list[str]]:
    """Merge per-variant results into an alignment.json dict + list of
    unresolved variants (which must escalate to human, never be guessed)."""
    contract: dict = {}
    unresolved: list[str] = []
    for v, res in results.items():
        entry = res.to_contract_entry()
        if entry is None:
            unresolved.append(v)
        else:
            contract[v] = entry
    return contract, unresolved
