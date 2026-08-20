"""QuestionGradingPack — built ONCE per exam question, reused for every
student, and the ONLY grading context a cloud grader ever receives.

Contents (all provenance-tagged, versioned by content hash):
- question id / answer-free question text / type / max score
- correct MC option(s) per version (from the verified key)
- explicit lecturer rubric + scoring rules + grading policy
- official grading solution where allowed (key.reference_explanation)
- a SMALL, budgeted set of local course-RAG evidence chunks
  (bge-m3 index; top_k default 2; hard character budget) — SUPPLEMENTAL:
  the rubric/solution is primary.

Course RAG here is GRADING-side context only. It never touches OCR text,
student wording, or the selected MC option (see tests).
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .schema import AnswerKey, KeyQuestion

DEFAULT_RAG_TOP_K = 2
DEFAULT_RAG_CHAR_BUDGET = 1200

#: Grading-side RAG policies (see docs). RAG_ALWAYS is today's behaviour and
#: stays the default: which policy is actually most efficient is an EMPIRICAL
#: question that has not been measured yet, and guessing here would freeze an
#: unvalidated choice into production.
RAG_POLICIES = ("RAG_DISABLED", "RAG_ALWAYS", "RAG_ON_UNCERTAIN", "RAG_ON_ESCALATION")


def _hash_text(value) -> str:
    if not isinstance(value, str):
        value = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


@dataclass
class RagEvidence:
    chunk_id: str
    source: str
    page: int | None
    similarity: float
    text: str


@dataclass
class RubricItemSpec:
    """One declared rubric item.

    ``requires_evidence`` is the ONLY sanctioned way to grade an item without
    a quoted span of the student's own words (the span check itself lives in
    ``evidence.py`` — the pack never holds student text): the exemption is
    declared per item here, never by weakening the check globally.
    ``excludes``/``requires`` declare mutual exclusion and prerequisites,
    enforced deterministically by ``invariants.py``.
    """

    id: str
    text: str
    points: float | None = None
    requires_evidence: bool = True
    excludes: list[str] = field(default_factory=list)
    requires: list[str] = field(default_factory=list)
    kind: str = "semantic"        # semantic | deterministic


@dataclass
class QuestionGradingPack:
    question_id: str
    question_text: str                    # answer-free (title + sub-item prompts)
    question_type: str
    max_score: float
    correct_by_version: dict[str, dict[str, list[str]]]   # sub_item -> version -> letters
    rubric: list[str]                     # explicit lecturer rubric lines
    scoring_rules: list[str]
    grading_policy: str                   # see policies.py
    official_solution: dict[str, str]     # sub_item -> reference_explanation (where allowed)
    rubric_items: list[RubricItemSpec] = field(default_factory=list)
    evidence_policy: str = "required"     # required | optional | disabled (see evidence.py)
    score_granularity: float | None = None   # e.g. 0.5 -> only half-point scores are valid
    rag_evidence: list[RagEvidence] = field(default_factory=list)
    rag_config: dict[str, Any] = field(default_factory=dict)
    rag_policy: str = "RAG_ALWAYS"
    # -- audit fields (derived from content; see refresh_audit) --------------
    question_text_hash: str = ""
    rubric_hash: str = ""
    solution_hash: str = ""
    rag_index_fingerprint: str | None = None
    rag_chars: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)
    version: str = "v1"
    hash: str = ""

    def refresh_audit(self) -> None:
        """Recompute the derived audit hashes. Any change to the question
        text, the rubric, the official solution or the retrieved context
        changes them — and therefore the pack hash and every cache key
        derived from it."""
        self.question_text_hash = _hash_text(self.question_text)
        self.rubric_hash = _hash_text([asdict(s) for s in self.rubric_specs().values()])
        self.solution_hash = _hash_text(self.official_solution)
        self.rag_chars = sum(len(e.text) for e in self.rag_evidence)
        self.rag_index_fingerprint = (self.rag_config or {}).get("index_config_hash")

    def audit(self) -> dict:
        """Everything needed to explain what this pack was built from."""
        self.refresh_audit()
        return {
            "question_id": self.question_id, "pack_version": self.version, "pack_hash": self.hash,
            "question_text_hash": self.question_text_hash, "rubric_hash": self.rubric_hash,
            "solution_hash": self.solution_hash, "grading_policy": self.grading_policy,
            "evidence_policy": self.evidence_policy, "rag_policy": self.rag_policy,
            "rag_chunk_ids": [e.chunk_id for e in self.rag_evidence],
            "rag_scores": [round(e.similarity, 4) for e in self.rag_evidence],
            "rag_sources": sorted({e.source for e in self.rag_evidence}),
            "rag_index_fingerprint": self.rag_index_fingerprint,
            "rag_chars": self.rag_chars,
            "rag_tokens_estimate": round(self.rag_chars / 4),
            "rubric_item_ids": self.rubric_item_ids(),
            "provenance": dict(self.provenance),
        }

    def compute_hash(self) -> str:
        self.refresh_audit()
        d = asdict(self)
        d.pop("hash", None)
        d.pop("provenance", None)   # provenance is descriptive; content decides identity
        self.hash = hashlib.sha256(json.dumps(d, sort_keys=True, ensure_ascii=False,
                                              default=str).encode()).hexdigest()[:16]
        return self.hash

    def to_grader_context(self, include_solution: bool = True) -> str:
        """Compact text block for the grader prompt — small by design."""
        lines = [f"Question {self.question_id} ({self.question_type}, max {self.max_score:g} pts):",
                 self.question_text.strip()]
        specs = self.rubric_specs()
        if specs:
            lines.append("Rubric:")
            for s in specs.values():
                need = "" if s.requires_evidence else "  (no quoted span needed)"
                lines.append(f"  {s.id}: {s.text}{need}")
        if self.scoring_rules:
            lines.append("Scoring rules: " + " | ".join(self.scoring_rules))
        if include_solution and self.official_solution:
            lines.append("Official solution notes:")
            lines += [f"  [{k}] {v}" for k, v in self.official_solution.items()]
        if self.rag_evidence:
            lines.append("Course context (supplemental — rubric/solution take precedence):")
            lines += [f"  <{e.chunk_id}|{e.source}> {e.text}" for e in self.rag_evidence]
        return "\n".join(lines)

    def rubric_specs(self) -> dict[str, RubricItemSpec]:
        """Declared rubric items, or R1..Rn derived from the plain rubric
        lines. Derived items require evidence — the exemption must be
        declared, never assumed."""
        if self.rubric_items:
            return {s.id: s for s in self.rubric_items}
        return {f"R{i+1}": RubricItemSpec(id=f"R{i+1}", text=r)
                for i, r in enumerate(self.rubric)}

    def rubric_item_ids(self) -> list[str]:
        return list(self.rubric_specs())

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "QuestionGradingPack":
        d = json.loads(text)
        d["rag_evidence"] = [RagEvidence(**e) for e in d.get("rag_evidence", [])]
        d["rubric_items"] = [RubricItemSpec(**s) for s in d.get("rubric_items", [])]
        return cls(**d)


# ---------------------------------------------------------------- building ----


def _rubric_from_key(q: KeyQuestion, general_rules: list[str]) -> tuple[list[str], list[str]]:
    rubric: list[str] = []
    rules: list[str] = list(general_rules)
    if q.grading_notes:
        for line in str(q.grading_notes).splitlines():
            line = line.strip(" -•*\t")
            if line:
                rubric.append(line)
    if q.explanation_required:
        rules.append(f"explanation weight {q.explanation_weight:g}")
    return rubric, rules


def build_pack(key: AnswerKey, q: KeyQuestion, *, grading_policy: str,
               course_id: str | None = None,
               retrieve: Callable[..., list[dict]] | None = None,
               embed_fn: Callable | None = None,
               rag_top_k: int = DEFAULT_RAG_TOP_K,
               rag_char_budget: int = DEFAULT_RAG_CHAR_BUDGET,
               include_solution: bool = True,
               source_label: str = "answer_key",
               rag_policy: str = "RAG_ALWAYS",
               rag_index_fingerprint: str | None = None) -> QuestionGradingPack:
    """Assemble one pack. ``retrieve`` is injected (courses.retrieve or a
    test double) so this module never imports network/model code."""
    question_text = q.title.strip()
    prompts = [s.prompt.strip() for s in q.sub_items if s.prompt and s.prompt.strip()]
    if prompts and len(prompts) <= 12:
        question_text += "\n" + "\n".join(f"- ({s.id}) {s.prompt.strip()}" for s in q.sub_items if s.prompt)
    rubric, rules = _rubric_from_key(q, key.general_rules)
    # choice_only questions carry no semantic rubric decisions, so there is
    # nothing to quote from the student's writing (and rubric items on such a
    # pack are already a validation error elsewhere).
    evidence_policy = "disabled" if grading_policy == "choice_only" else "required"
    pack = QuestionGradingPack(
        question_id=q.id,
        question_text=question_text,
        question_type=str(q.type.value if hasattr(q.type, "value") else q.type),
        max_score=float(q.max_points),
        correct_by_version={s.id: dict(s.correct_by_version) for s in q.sub_items},
        rubric=rubric,
        scoring_rules=rules,
        grading_policy=grading_policy,
        rubric_items=[RubricItemSpec(id=f"R{i+1}", text=r) for i, r in enumerate(rubric)],
        evidence_policy=evidence_policy,
        official_solution={s.id: s.reference_explanation for s in q.sub_items
                           if include_solution and s.reference_explanation},
        provenance={"key_source": source_label, "exam_title": key.exam_title,
                    "built": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "course_id": course_id},
    )
    if rag_policy not in RAG_POLICIES:
        raise ValueError(f"unknown RAG policy {rag_policy!r} (expected one of {RAG_POLICIES})")
    pack.rag_policy = rag_policy
    # RAG_ALWAYS embeds the course context once, at build time. The lazy
    # policies leave the pack context-free and retrieve only when grading
    # actually needs help (see attach_rag) — no retrieval on the easy majority.
    if rag_policy == "RAG_ALWAYS":
        attach_rag(pack, course_id=course_id, retrieve=retrieve, embed_fn=embed_fn,
                   rag_top_k=rag_top_k, rag_char_budget=rag_char_budget,
                   index_fingerprint=rag_index_fingerprint, in_place=True)
    elif course_id:
        pack.rag_config = {"course_id": course_id, "top_k": rag_top_k,
                           "char_budget": rag_char_budget, "chars_used": 0,
                           "index_config_hash": rag_index_fingerprint, "deferred": True}
    pack.compute_hash()
    return pack


def rag_query(pack: QuestionGradingPack) -> str:
    """The retrieval query for a pack: question text + rubric ONLY. The
    student's own words are never part of it — grading-side retrieval must
    not be steered by what the student happened to write."""
    return "\n".join([pack.question_text, *pack.rubric])[:1500]


def attach_rag(pack: QuestionGradingPack, *, course_id: str | None,
               retrieve: Callable[..., list[dict]] | None, embed_fn: Callable | None = None,
               rag_top_k: int = DEFAULT_RAG_TOP_K,
               rag_char_budget: int = DEFAULT_RAG_CHAR_BUDGET,
               index_fingerprint: str | None = None,
               in_place: bool = False) -> QuestionGradingPack:
    """Attach a SMALL, budgeted, provenance-tagged set of course chunks.

    Returns a COPY with a new hash unless ``in_place``. Used at build time
    under RAG_ALWAYS and lazily under RAG_ON_UNCERTAIN / RAG_ON_ESCALATION.
    """
    target = pack if in_place else copy.deepcopy(pack)
    if not (course_id and retrieve is not None and rag_top_k > 0):
        return target
    query = rag_query(target)
    try:
        hits = retrieve(course_id, query, rag_top_k, embed_fn) if embed_fn else retrieve(course_id, query, rag_top_k)
    except TypeError:
        hits = retrieve(course_id, query, rag_top_k)
    target.rag_evidence = []
    used = 0
    for h in hits:
        text = (h.get("text") or "").strip()
        room = rag_char_budget - used
        if room <= 1:
            break
        if len(text) > room:
            text = text[:room - 1].rstrip() + "…"   # ellipsis counted in budget
        used += len(text)
        target.rag_evidence.append(RagEvidence(
            chunk_id=h["chunk_id"], source=h.get("source", "?"), page=h.get("page"),
            similarity=float(h.get("similarity", 0.0)), text=text))
    target.rag_config = {"course_id": course_id, "top_k": rag_top_k,
                         "char_budget": rag_char_budget, "chars_used": used,
                         "index_config_hash": index_fingerprint}
    if not in_place:
        target.compute_hash()
    return target


def build_all_packs(key: AnswerKey, policies: dict[str, str], **kw) -> dict[str, QuestionGradingPack]:
    """policies: question_id -> grading policy name (see policies.py)."""
    return {q.id: build_pack(key, q, grading_policy=policies.get(q.id, "choice_and_explanation_independent"), **kw)
            for q in key.questions}


# ------------------------------------------------------------ persistence ----


class PackStore:
    """Persist packs once per exam question; invalidate on key/rubric/course
    index change via the pack hash + a store-level source fingerprint."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, packs: dict[str, QuestionGradingPack], source_fingerprint: str) -> None:
        for qid, p in packs.items():
            (self.root / f"q{qid}.json").write_text(p.to_json(), encoding="utf-8")
        (self.root / "manifest.json").write_text(json.dumps({
            "source_fingerprint": source_fingerprint,
            "packs": {qid: p.hash for qid, p in packs.items()},
            "built": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1), encoding="utf-8")

    def load(self, source_fingerprint: str) -> dict[str, QuestionGradingPack] | None:
        m = self.root / "manifest.json"
        if not m.exists():
            return None
        man = json.loads(m.read_text(encoding="utf-8"))
        if man.get("source_fingerprint") != source_fingerprint:
            return None   # key/rubric/course index changed -> rebuild
        out = {}
        for qid in man["packs"]:
            p = self.root / f"q{qid}.json"
            if not p.exists():
                return None
            out[qid] = QuestionGradingPack.from_json(p.read_text(encoding="utf-8"))
        return out


def source_fingerprint(key_bytes: bytes, course_index_hash: str | None,
                       policies: dict[str, str], rag_top_k: int, rag_char_budget: int,
                       pack_version: str = "v1", rag_policy: str = "RAG_ALWAYS") -> str:
    """Any change to the key, the course index, the grading policies or the
    retrieval configuration invalidates every pack built from them."""
    return hashlib.sha256(json.dumps({
        "key": hashlib.sha256(key_bytes).hexdigest(),
        "index": course_index_hash, "policies": dict(sorted(policies.items())),
        "top_k": rag_top_k, "budget": rag_char_budget, "version": pack_version,
        "rag_policy": rag_policy,
    }, sort_keys=True).encode()).hexdigest()[:16]
