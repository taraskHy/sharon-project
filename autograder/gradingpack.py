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

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .schema import AnswerKey, KeyQuestion

DEFAULT_RAG_TOP_K = 2
DEFAULT_RAG_CHAR_BUDGET = 1200


@dataclass
class RagEvidence:
    chunk_id: str
    source: str
    page: int | None
    similarity: float
    text: str


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
    rag_evidence: list[RagEvidence] = field(default_factory=list)
    rag_config: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    version: str = "v1"
    hash: str = ""

    def compute_hash(self) -> str:
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
        if self.rubric:
            lines.append("Rubric:")
            lines += [f"  R{i+1}: {r}" for i, r in enumerate(self.rubric)]
        if self.scoring_rules:
            lines.append("Scoring rules: " + " | ".join(self.scoring_rules))
        if include_solution and self.official_solution:
            lines.append("Official solution notes:")
            lines += [f"  [{k}] {v}" for k, v in self.official_solution.items()]
        if self.rag_evidence:
            lines.append("Course context (supplemental — rubric/solution take precedence):")
            lines += [f"  <{e.chunk_id}|{e.source}> {e.text}" for e in self.rag_evidence]
        return "\n".join(lines)

    def rubric_item_ids(self) -> list[str]:
        return [f"R{i+1}" for i in range(len(self.rubric))]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=1)

    @classmethod
    def from_json(cls, text: str) -> "QuestionGradingPack":
        d = json.loads(text)
        d["rag_evidence"] = [RagEvidence(**e) for e in d.get("rag_evidence", [])]
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
               source_label: str = "answer_key") -> QuestionGradingPack:
    """Assemble one pack. ``retrieve`` is injected (courses.retrieve or a
    test double) so this module never imports network/model code."""
    question_text = q.title.strip()
    prompts = [s.prompt.strip() for s in q.sub_items if s.prompt and s.prompt.strip()]
    if prompts and len(prompts) <= 12:
        question_text += "\n" + "\n".join(f"- ({s.id}) {s.prompt.strip()}" for s in q.sub_items if s.prompt)
    rubric, rules = _rubric_from_key(q, key.general_rules)
    pack = QuestionGradingPack(
        question_id=q.id,
        question_text=question_text,
        question_type=str(q.type.value if hasattr(q.type, "value") else q.type),
        max_score=float(q.max_points),
        correct_by_version={s.id: dict(s.correct_by_version) for s in q.sub_items},
        rubric=rubric,
        scoring_rules=rules,
        grading_policy=grading_policy,
        official_solution={s.id: s.reference_explanation for s in q.sub_items
                           if include_solution and s.reference_explanation},
        provenance={"key_source": source_label, "exam_title": key.exam_title,
                    "built": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "course_id": course_id},
    )
    if course_id and retrieve is not None and rag_top_k > 0:
        query = "\n".join([q.title, *rubric, *[s.prompt for s in q.sub_items if s.prompt]])[:1500]
        try:
            hits = retrieve(course_id, query, rag_top_k, embed_fn) if embed_fn else retrieve(course_id, query, rag_top_k)
        except TypeError:
            hits = retrieve(course_id, query, rag_top_k)
        used = 0
        for h in hits:
            text = (h.get("text") or "").strip()
            room = rag_char_budget - used
            if room <= 1:
                break
            if len(text) > room:
                text = text[:room - 1].rstrip() + "…"   # ellipsis counted in budget
            used += len(text)
            pack.rag_evidence.append(RagEvidence(
                chunk_id=h["chunk_id"], source=h.get("source", "?"), page=h.get("page"),
                similarity=float(h.get("similarity", 0.0)), text=text))
        pack.rag_config = {"course_id": course_id, "top_k": rag_top_k,
                           "char_budget": rag_char_budget, "chars_used": used,
                           "index_config_hash": None}
    pack.compute_hash()
    return pack


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
                       pack_version: str = "v1") -> str:
    return hashlib.sha256(json.dumps({
        "key": hashlib.sha256(key_bytes).hexdigest(),
        "index": course_index_hash, "policies": dict(sorted(policies.items())),
        "top_k": rag_top_k, "budget": rag_char_budget, "version": pack_version,
    }, sort_keys=True).encode()).hexdigest()[:16]
