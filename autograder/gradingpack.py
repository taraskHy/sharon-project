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

#: Pack schema version — part of every source fingerprint, so a schema change
#: rebuilds persisted packs instead of half-reading them.
PACK_SCHEMA_VERSION = "v2"
#: Version of the retrieval-query/budget rules (rag_query + the char budget
#: discipline). Bump when either changes so cached packs rebuild.
RETRIEVAL_CONFIG_VERSION = "r1"

#: Grading-side RAG policies (see docs). The DEFAULT IS RAG_DISABLED: the
#: benefit of grading-side retrieval has not been measured, it costs input
#: tokens on every grading call, and no unmeasured optional context should be
#: sent silently. Retrieval is opt-in per package until the A/B decides it.
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
    text_hash: str = ""              # content hash of the included excerpt


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


#: The EXACT section headers ``to_grader_context`` emits, shared with the
#: cloud boundary's payload tripwire (cloudboundary.forbidden_cloud_markers):
#: any of these appearing in an outbound cloud-OCR payload proves grading
#: material leaked into it. Deriving the rendered context and the tripwire
#: from one tuple means they cannot drift apart.
CONTEXT_HEADERS: tuple[str, ...] = (
    "Rubric:",
    "Scoring rules: ",
    "Official solution notes:",
    "Course context (supplemental — rubric/solution take precedence):",
)


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
    #: wrong-answer rule for explanation_required_if_correct ("zero" |
    #: "selection" | "process"); None -> the production default ("zero").
    #: Carried into dataset packs so label eligibility can honor "process".
    wrong_answer_rule: str | None = None
    rag_evidence: list[RagEvidence] = field(default_factory=list)
    #: Chunks retrieved ONCE at pack preparation for the lazy policies
    #: (RAG_ON_UNCERTAIN / RAG_ON_ESCALATION). They are NOT part of the grader
    #: context until the policy activates them (activate_rag) — preparation is
    #: a free local search; activation is what adds provider input tokens.
    rag_prepared: list[RagEvidence] = field(default_factory=list)
    #: Whether retrieval infrastructure (course id + retriever + index) was
    #: available when this pack was built. False means an optional-RAG policy
    #: degrades to no-RAG grading — never to REVIEW by itself.
    rag_available: bool | None = None
    rag_config: dict[str, Any] = field(default_factory=dict)
    rag_policy: str = "RAG_DISABLED"
    # -- audit fields (derived from content; see refresh_audit) --------------
    question_text_hash: str = ""
    rubric_hash: str = ""
    solution_hash: str = ""
    rag_index_fingerprint: str | None = None
    rag_chars: int = 0
    rag_tokens_est: int = 0
    provenance: dict[str, Any] = field(default_factory=dict)
    version: str = PACK_SCHEMA_VERSION
    hash: str = ""

    def refresh_audit(self) -> None:
        """Recompute the derived audit hashes. Any change to the question
        text, the rubric, the official solution or the retrieved context
        changes them — and therefore the pack hash and every cache key
        derived from it."""
        self.question_text_hash = _hash_text(self.question_text)
        self.rubric_hash = _hash_text([asdict(s) for s in self.rubric_specs().values()])
        self.solution_hash = _hash_text(self.official_solution)
        for e in (*self.rag_evidence, *self.rag_prepared):
            e.text_hash = e.text_hash or _hash_text(e.text)
        self.rag_chars = sum(len(e.text) for e in self.rag_evidence)
        self.rag_tokens_est = round(self.rag_chars / 4)
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
            "rag_prepared_chunk_ids": [e.chunk_id for e in self.rag_prepared],
            "rag_available": self.rag_available,
            "rag_scores": [round(e.similarity, 4) for e in self.rag_evidence],
            "rag_sources": sorted({e.source for e in self.rag_evidence}),
            "rag_text_hashes": [e.text_hash for e in self.rag_evidence],
            "rag_index_fingerprint": self.rag_index_fingerprint,
            "retrieval_version": RETRIEVAL_CONFIG_VERSION,
            "rag_chars": self.rag_chars,
            "rag_tokens_estimate": self.rag_tokens_est,
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

    def to_grader_context(self, include_solution: bool = True,
                          include_scoring_rules: bool = True) -> str:
        """Compact text block for the grader prompt — small by design.

        ``include_scoring_rules=False`` drops the final-score composition rules
        (e.g. "explanation weight 0", "no credit for an answer without an
        explanation"). They describe how the student's TOTAL for the sub-item
        is assembled from the selection and the explanation — a downstream,
        deterministic step. An explanation judge that reads them starts
        reasoning about the selection instead of the text in front of it, which
        is exactly what the grade-v2 prompt did.
        """
        lines = [f"Question {self.question_id} ({self.question_type}, max {self.max_score:g} pts):",
                 self.question_text.strip()]
        specs = self.rubric_specs()
        if specs:
            lines.append(CONTEXT_HEADERS[0])                       # Rubric:
            for s in specs.values():
                need = "" if s.requires_evidence else "  (no quoted span needed)"
                lines.append(f"  {s.id}: {s.text}{need}")
        if self.scoring_rules and include_scoring_rules:
            lines.append(CONTEXT_HEADERS[1] + " | ".join(self.scoring_rules))
        if include_solution and self.official_solution:
            lines.append(CONTEXT_HEADERS[2])                       # Official solution notes:
            lines += [f"  [{k}] {v}" for k, v in self.official_solution.items()]
        if self.rag_evidence:
            lines.append(CONTEXT_HEADERS[3])                       # Course context ...
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
        d["rag_prepared"] = [RagEvidence(**e) for e in d.get("rag_prepared", [])]
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
               rag_policy: str = "RAG_DISABLED",
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
    # policies PREPARE the chunks once at pack preparation (a free LOCAL
    # search) but keep the grader context empty — activation (activate_rag)
    # is what later adds provider input tokens, and only where the policy
    # says so. RAG_DISABLED performs no retrieval of any kind.
    can_retrieve = bool(course_id and retrieve is not None and rag_top_k > 0)
    if rag_policy == "RAG_ALWAYS":
        pack.rag_available = can_retrieve
        attach_rag(pack, course_id=course_id, retrieve=retrieve, embed_fn=embed_fn,
                   rag_top_k=rag_top_k, rag_char_budget=rag_char_budget,
                   index_fingerprint=rag_index_fingerprint, in_place=True)
    elif rag_policy in ("RAG_ON_UNCERTAIN", "RAG_ON_ESCALATION"):
        pack.rag_available = can_retrieve
        pack.rag_config = {"course_id": course_id, "top_k": rag_top_k,
                           "char_budget": rag_char_budget, "chars_used": 0,
                           "index_config_hash": rag_index_fingerprint, "deferred": True}
        if can_retrieve:
            prepare_rag(pack, course_id=course_id, retrieve=retrieve, embed_fn=embed_fn,
                        rag_top_k=rag_top_k, rag_char_budget=rag_char_budget)
    pack.compute_hash()
    return pack


def rag_query(pack: QuestionGradingPack) -> str:
    """The retrieval query for a pack: question text + rubric + official
    solution ONLY — stable grading context. The student's own words are never
    part of it: grading-side retrieval must not be steered by what the
    student happened to write (a bad reading would fetch the wrong material
    and bias the grade)."""
    return "\n".join([pack.question_text, *pack.rubric,
                      *pack.official_solution.values()])[:1500]


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
    evidence, used = _retrieve_budgeted(target, course_id=course_id, retrieve=retrieve,
                                        embed_fn=embed_fn, rag_top_k=rag_top_k,
                                        rag_char_budget=rag_char_budget)
    target.rag_evidence = evidence
    target.rag_config = {"course_id": course_id, "top_k": rag_top_k,
                         "char_budget": rag_char_budget, "chars_used": used,
                         "index_config_hash": index_fingerprint}
    if not in_place:
        target.compute_hash()
    return target


def _retrieve_budgeted(pack: QuestionGradingPack, *, course_id: str, retrieve: Callable,
                       embed_fn: Callable | None, rag_top_k: int,
                       rag_char_budget: int) -> tuple[list[RagEvidence], int]:
    """One local retrieval, deterministically selected/truncated to the hard
    character budget, provenance preserved (chunk id, source, page, score,
    excerpt hash)."""
    query = rag_query(pack)
    try:
        hits = retrieve(course_id, query, rag_top_k, embed_fn) if embed_fn else retrieve(course_id, query, rag_top_k)
    except TypeError:
        hits = retrieve(course_id, query, rag_top_k)
    out: list[RagEvidence] = []
    used = 0
    for h in hits:
        text = (h.get("text") or "").strip()
        room = rag_char_budget - used
        if room <= 1:
            break
        if len(text) > room:
            text = text[:room - 1].rstrip() + "…"   # ellipsis counted in budget
        used += len(text)
        out.append(RagEvidence(
            chunk_id=h["chunk_id"], source=h.get("source", "?"), page=h.get("page"),
            similarity=float(h.get("similarity", 0.0)), text=text,
            text_hash=_hash_text(text)))
    return out, used


def prepare_rag(pack: QuestionGradingPack, *, course_id: str, retrieve: Callable,
                embed_fn: Callable | None = None,
                rag_top_k: int = DEFAULT_RAG_TOP_K,
                rag_char_budget: int = DEFAULT_RAG_CHAR_BUDGET) -> None:
    """PREPARATION (free, local): retrieve the question-level chunks once and
    cache them on the pack WITHOUT adding them to the grader context. The
    lazy policies then activate this cache instead of searching again per
    student. Preparation never costs provider tokens."""
    evidence, _used = _retrieve_budgeted(pack, course_id=course_id, retrieve=retrieve,
                                         embed_fn=embed_fn, rag_top_k=rag_top_k,
                                         rag_char_budget=rag_char_budget)
    pack.rag_prepared = evidence
    pack.rag_config = {**(pack.rag_config or {}), "prepared_chunks": len(evidence)}


def activate_rag(pack: QuestionGradingPack) -> QuestionGradingPack:
    """ACTIVATION: the injected ``rag_attach`` used at the live seam. Returns
    a COPY whose grader context includes the chunks prepared at pack
    preparation (new hash — cache entries never collide with the base pack).
    With nothing prepared (no course/retriever/index), returns the pack
    UNCHANGED: an optional-RAG policy degrades to no-RAG grading, never to a
    failure or a REVIEW by itself."""
    if pack.rag_policy not in ("RAG_ON_UNCERTAIN", "RAG_ON_ESCALATION"):
        return pack
    if not pack.rag_prepared:
        return pack
    target = copy.deepcopy(pack)
    target.rag_evidence = list(target.rag_prepared)
    target.rag_config = {**(target.rag_config or {}),
                         "chars_used": sum(len(e.text) for e in target.rag_evidence)}
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
        for qid, want_hash in man["packs"].items():
            p = self.root / f"q{qid}.json"
            if not p.exists():
                return None
            pack = QuestionGradingPack.from_json(p.read_text(encoding="utf-8"))
            if want_hash and pack.hash != want_hash:
                # A pack file that does not match the manifest it sits under
                # (interrupted save, foreign write beneath a shared root) must
                # never be silently graded with — rebuild instead.
                return None
            out[qid] = pack
        return out


def source_fingerprint(key_bytes: bytes, course_index_hash: str | None,
                       policies: dict[str, str], rag_top_k: int, rag_char_budget: int,
                       pack_version: str = PACK_SCHEMA_VERSION,
                       rag_policy: str = "RAG_DISABLED") -> str:
    """Any change to the key, the course index, the grading policies, the
    retrieval configuration, or the pack/retrieval schema versions
    invalidates every pack built from them."""
    return hashlib.sha256(json.dumps({
        "key": hashlib.sha256(key_bytes).hexdigest(),
        "index": course_index_hash, "policies": dict(sorted(policies.items())),
        "top_k": rag_top_k, "budget": rag_char_budget, "version": pack_version,
        "retrieval_version": RETRIEVAL_CONFIG_VERSION,
        "rag_policy": rag_policy,
    }, sort_keys=True).encode()).hexdigest()[:16]
