"""Exact reuse of settled decisions — and a hard refusal of semantic reuse.

Reusing a lecturer's decision is safe exactly when the underlying issue is
MECHANICALLY IDENTICAL:

    same variant marker fingerprint      -> reuse the marker -> variant mapping
    same package/template fingerprint    -> reuse the template
    same question + rubric fingerprint   -> reuse the QuestionGradingPack
    same exact image hash                -> reuse the model result for it
    same deterministic alignment print   -> reuse the alignment

It is NOT safe when the only thing shared is meaning. "This answer looks
like one that got 7/10, so give it 7/10" is an uncontrolled semantic
nearest-neighbour grader built out of review corrections; it is refused
here (``SemanticReuseRefused``). If such a system is ever wanted it must be
a separate, explicitly evaluated component — not a side effect of the
review queue.

Every reuse is recorded with what was matched, who decided it, and when.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

#: Kinds where an EXACT fingerprint match makes reuse mechanically sound.
EXACT_KINDS = ("variant_marker", "package_template", "question_pack", "image", "alignment",
               "page_structure")

#: Kinds that would amount to learning grades from similar answers.
SEMANTIC_KINDS = ("answer_similarity", "explanation_similarity", "score_neighbour",
                  "student_answer", "embedding")


class SemanticReuseRefused(ValueError):
    """Refused: this would grade one student from another student's answer."""


def _digest(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False,
                                     default=str).encode("utf-8")).hexdigest()[:16]


def exact_fingerprint(kind: str, **facts: Any) -> str:
    """Fingerprint for a mechanically identical situation."""
    if kind in SEMANTIC_KINDS:
        raise SemanticReuseRefused(
            f"{kind!r} is a semantic similarity, not a mechanical identity — "
            "reusing a grade across it is not permitted")
    if kind not in EXACT_KINDS:
        raise ValueError(f"unknown reuse kind {kind!r}")
    return f"{kind}:{_digest({'kind': kind, **facts})}"


def image_fingerprint(data: bytes) -> str:
    """Byte-exact image identity: one changed pixel is a different image."""
    return "image:" + hashlib.sha256(data or b"").hexdigest()[:16]


@dataclass
class ReuseEntry:
    kind: str
    fingerprint: str
    decision: Any
    by: str = "system"
    ts: str = ""
    scope: dict[str, Any] = field(default_factory=dict)
    hits: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


class ExactReuseStore:
    """Persisted fingerprint -> settled decision. Exact matches only."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 — a corrupt store is an empty store
                self._data = {}
        self.hits = 0
        self.misses = 0

    def _save(self) -> None:
        self.path.write_text(json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8")

    def put(self, kind: str, fingerprint: str, decision: Any, *, by: str = "system",
            scope: dict | None = None) -> ReuseEntry:
        if kind in SEMANTIC_KINDS:
            raise SemanticReuseRefused(f"refusing to store a {kind!r} decision for reuse")
        if not fingerprint.startswith(f"{kind}:"):
            raise ValueError(f"fingerprint {fingerprint!r} does not belong to kind {kind!r}")
        entry = ReuseEntry(kind, fingerprint, decision, by,
                           time.strftime("%Y-%m-%d %H:%M:%S"), dict(scope or {}))
        self._data[fingerprint] = entry.as_dict()
        self._save()
        return entry

    def get(self, kind: str, fingerprint: str) -> Optional[ReuseEntry]:
        rec = self._data.get(fingerprint)
        if not rec or rec.get("kind") != kind:
            self.misses += 1
            return None
        rec["hits"] = int(rec.get("hits", 0)) + 1
        self._save()
        self.hits += 1
        return ReuseEntry(**rec)

    def lookup(self, kind: str, **facts: Any) -> Optional[ReuseEntry]:
        return self.get(kind, exact_fingerprint(kind, **facts))

    def stats(self) -> dict:
        n = self.hits + self.misses
        return {"entries": len(self._data), "hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / n, 4) if n else None,
                "by_kind": {k: sum(1 for v in self._data.values() if v.get("kind") == k)
                            for k in EXACT_KINDS if any(v.get("kind") == k for v in self._data.values())}}


def reuse_grade_by_similarity(*_a, **_kw):
    """Deliberately not implemented. Present so the refusal is explicit and
    greppable rather than an absence someone fills in later."""
    raise SemanticReuseRefused(
        "grading one student from another student's similar answer is not permitted; "
        "any such system must be built and evaluated separately")
