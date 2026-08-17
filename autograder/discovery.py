"""Automatic exam-package discovery: variants, markers, layout, policies.

Goal: the lecturer uploads exam + key/rubric/solution and never writes a
"heart -> variant A" mapping by hand. Resolution ladder per fact:

    deterministic package analysis (text layer / key structure)
        -> local model (task variant_resolve / policy_infer)
        -> targeted cloud resolver (variant_resolve_cloud)
        -> human only if unresolved

Everything discovered is EMITTED IN THE EXISTING CONTRACTS the validated
pipeline already consumes — <key stem>.variants.json (markers -> variant
ids that equal AnswerKey.versions), <key stem>.alignment.json (identity or
operator-style mapping), <key stem>.template.json (ExamTemplate) — so
variant.decide_version, extract.py banding and grade.py are unchanged.
Resolved catalogs persist and are reused for every identical package
(fingerprint = key bytes + exam booklet bytes), so the lecturer is never
asked twice.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .policies import infer_policy_from_key
from .schema import AnswerKey

# ------------------------------------------------------------ contracts -----


class MarkerCatalog(BaseModel):
    """What a resolver (local or cloud) returns about variant markers."""

    n_variants: int
    markers: list[dict] = Field(
        description="[{id, variant, description, aliases?}] with variant ids exactly matching the key versions")
    marker_page: int = 1
    marker_kind: str = "marker"
    marker_location_hint: Optional[str] = None
    identical_question_order: bool = True
    confident: bool = False
    notes: Optional[str] = None


class PolicyInference(BaseModel):
    question_id: str
    policy: Literal["choice_only", "wrong_choice_zero", "explanation_required_if_correct",
                    "explanation_can_rescue_wrong_choice", "choice_and_explanation_independent"]
    confident: bool = False
    evidence: str = ""


@dataclass
class DiscoveryFact:
    value: object
    source: Literal["deterministic", "local_model", "cloud_model", "human", "unresolved"]
    evidence: str = ""


@dataclass
class DiscoveryResult:
    package_fingerprint: str
    versions: DiscoveryFact
    variants_config: DiscoveryFact          # dict in variants.json shape, or None
    alignment: DiscoveryFact                # dict in alignment.json shape, or None
    template: DiscoveryFact                 # dict in template.json shape, or None
    policies: dict[str, DiscoveryFact] = field(default_factory=dict)
    needs_human: list[str] = field(default_factory=list)
    built: str = ""

    def unresolved(self) -> list[str]:
        out = list(self.needs_human)
        for name, f in (("variants", self.variants_config), ("alignment", self.alignment),
                        ("template", self.template)):
            if f.source == "unresolved" and name not in out:
                out.append(name)
        return out


def package_fingerprint(*parts: bytes | str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8") if isinstance(p, str) else p)
    return h.hexdigest()[:16]


# --------------------------------------------------- deterministic stage ----

_VERSION_PATTERNS = [
    re.compile(r"(?:נוסח|גרסה|טופס|version|form)\s*[:\-]?\s*([A-Za-z0-9א-ת]{1,3})", re.IGNORECASE),
]
_SUIT_WORDS = {"heart": ["♡", "♥", "לב"], "spade": ["♠", "עלה"], "diamond": ["◇", "♦", "מעוין"],
               "club": ["♣", "תלתן"]}


def deterministic_versions(key: AnswerKey, exam_text_layer: str = "") -> DiscoveryFact:
    """Version ids: the verified key is authoritative when it has >1 version."""
    if key.versions and key.versions != ["default"]:
        return DiscoveryFact(list(key.versions), "deterministic", "AnswerKey.versions")
    found = []
    for pat in _VERSION_PATTERNS:
        found += [m.group(1) for m in pat.finditer(exam_text_layer or "")]
    found = sorted(set(found))
    if len(found) >= 2:
        return DiscoveryFact(found, "deterministic", f"exam text layer version labels {found}")
    return DiscoveryFact(["default"], "deterministic", "single-version package (no version labels)")


def deterministic_markers(versions: list[str], exam_text_layer: str = "") -> DiscoveryFact:
    """Only when markers are literally recoverable: version ids that ARE
    printable labels present in the text layer (e.g. 'A1'..'A3', suit
    words). Icon-only markers cannot be derived deterministically -> None."""
    if len(versions) <= 1:
        return DiscoveryFact(None, "deterministic", "single version: no markers needed")
    text = exam_text_layer or ""
    markers = {}
    for v in versions:
        aliases = _SUIT_WORDS.get(v.lower(), [])
        if re.search(rf"(?<![A-Za-z0-9]){re.escape(v)}(?![A-Za-z0-9])", text) or any(a in text for a in aliases):
            markers[v] = {"variant": v, "description": f"printed version label '{v}'", "aliases": aliases}
    if len(markers) == len(versions):
        cfg = {"marker_kind": "printed version label", "marker_page": 1, "markers": markers,
               "mapping_source": {"derived": "deterministic: version labels found in the exam text layer"}}
        return DiscoveryFact(cfg, "deterministic", "all version labels present in text layer")
    return DiscoveryFact(None, "unresolved", "markers not literally present in text layer (icons?)")


def deterministic_template(key: AnswerKey, exam_text_layer: str = "", answer_sheet_hint_page: int | None = None) -> DiscoveryFact:
    """Mode from key structure; answer-sheet rule from a detected 'mark only
    here' instruction; else defaults to structural detection."""
    modes = set()
    for q in key.questions:
        modes.add("with_explanation" if (q.explanation_required or q.explanation_weight > 0) else "multiple_choice")
    mode = "multiple_choice" if modes == {"multiple_choice"} else "with_explanation" if modes == {"with_explanation"} else "mixed"
    qm = {q.id: ("with_explanation" if (q.explanation_required or q.explanation_weight > 0) else "multiple_choice")
          for q in key.questions} if mode == "mixed" else {}
    sheet_rule, pages, evidence = "detected", [], "no fixed-sheet instruction found"
    if answer_sheet_hint_page or re.search(r"(סמנו כאן בלבד|mark (only )?here|answer sheet)", exam_text_layer or "", re.I):
        sheet_rule, pages = "fixed_pages", [answer_sheet_hint_page or 1]
        evidence = "fixed answer-sheet instruction detected"
    tpl = {"template_id": f"auto-{package_fingerprint(key.exam_title, mode)}", "name": key.exam_title,
           "mode": mode, "question_modes": qm, "answer_sheet_rule": sheet_rule,
           "answer_sheet_pages": pages, "answer_table_banding": sheet_rule == "fixed_pages" and mode == "multiple_choice",
           "answer_table_columns_rtl": ["A", "B", "C", "D"], "booklet_answers_not_graded": sheet_rule == "fixed_pages",
           "policy_evidence": evidence}
    return DiscoveryFact(tpl, "deterministic", evidence)


def deterministic_policies(key: AnswerKey) -> dict[str, DiscoveryFact]:
    out = {}
    for q in key.questions:
        pol, ev = infer_policy_from_key(q.explanation_required, q.explanation_weight, q.grading_notes)
        out[q.id] = DiscoveryFact(pol, "deterministic" if pol else "unresolved", ev)
    return out


# ------------------------------------------------------- model stages -------

VARIANT_RESOLVE_SYSTEM = (
    "You inspect the FIRST page(s) of an exam booklet to catalogue how exam "
    "variants are marked (icons, letters, codes). Report the number of "
    "variants, one marker per variant with a precise visual description "
    "usable to recognise it later, the page it appears on, and whether the "
    "question order is identical across variants if that is visible. Use the "
    "provided version ids EXACTLY as the variant values. Never invent "
    "markers you cannot see. Reply with ONLY the JSON object."
)

POLICY_INFER_SYSTEM = (
    "You read the grading instructions of ONE exam question (rubric text, "
    "scoring notes) and classify its grading policy: choice_only, "
    "wrong_choice_zero, explanation_required_if_correct, "
    "explanation_can_rescue_wrong_choice, or choice_and_explanation_independent. "
    "Set confident=false unless the text states the rule explicitly. Reply "
    "with ONLY the JSON object."
)


def _catalog_to_variants_config(cat: MarkerCatalog, versions: list[str]) -> dict | None:
    if not cat.markers or {m.get("variant") for m in cat.markers} != set(versions):
        return None
    markers = {}
    for m in cat.markers:
        mid = re.sub(r"[^A-Za-z0-9_\-]", "_", str(m.get("id") or m["variant"])).lower()
        markers[mid] = {"variant": m["variant"], "description": m.get("description", ""),
                        "aliases": list(m.get("aliases") or [])}
    return {"marker_kind": cat.marker_kind, "marker_page": cat.marker_page,
            **({"marker_location_hint": cat.marker_location_hint} if cat.marker_location_hint else {}),
            "markers": markers,
            "mapping_source": {"derived": "automatic discovery (model-catalogued markers; verify once)"}}


def resolve_markers_with_models(*, versions: list[str], cover_png_b64: str, gateway,
                                meta: dict | None = None,
                                local_task: str = "variant_resolve",
                                cloud_task: str = "variant_resolve_cloud") -> DiscoveryFact:
    blocks = [{"type": "text", "text": f"Version ids: {versions}. Catalogue the variant markers."},
              {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": cover_png_b64}}]
    for task, src in ((local_task, "local_model"), (cloud_task, "cloud_model")):
        try:
            gateway.route(task)
        except Exception:  # noqa: BLE001
            continue
        try:
            cat = gateway.call(task=task, system=VARIANT_RESOLVE_SYSTEM, content_blocks=blocks,
                               output_model=MarkerCatalog, meta={**(meta or {}), "stage": "discovery"}).value
        except Exception:  # noqa: BLE001
            continue
        cfg = _catalog_to_variants_config(cat, versions)
        if cfg is not None and cat.confident:
            return DiscoveryFact(cfg, src, f"{cat.n_variants} markers catalogued by {src}")
    return DiscoveryFact(None, "unresolved", "no model produced a confident, complete marker catalogue")


def resolve_policy_with_models(*, question_id: str, rubric_text: str, gateway,
                               meta: dict | None = None, local_task: str = "policy_infer",
                               cloud_task: str = "policy_infer_cloud") -> DiscoveryFact:
    blocks = [{"type": "text", "text": f"Question {question_id} grading instructions:\n{rubric_text[:2000]}"}]
    for task, src in ((local_task, "local_model"), (cloud_task, "cloud_model")):
        try:
            gateway.route(task)
        except Exception:  # noqa: BLE001
            continue
        try:
            inf = gateway.call(task=task, system=POLICY_INFER_SYSTEM, content_blocks=blocks,
                               output_model=PolicyInference, meta={**(meta or {}), "stage": "discovery"}).value
        except Exception:  # noqa: BLE001
            continue
        if inf.confident:
            return DiscoveryFact(inf.policy, src, inf.evidence)
    return DiscoveryFact(None, "unresolved", "no confident policy inference")


# ------------------------------------------------------------ orchestration --


def discover_package(*, key: AnswerKey, key_bytes: bytes, exam_bytes: bytes | None = None,
                     exam_text_layer: str = "", cover_png_b64: str | None = None,
                     rubric_texts: dict[str, str] | None = None, gateway=None,
                     meta: dict | None = None) -> DiscoveryResult:
    fp = package_fingerprint(key_bytes, exam_bytes or b"")
    versions = deterministic_versions(key, exam_text_layer)
    vlist = list(versions.value)
    markers = deterministic_markers(vlist, exam_text_layer)
    if markers.source == "unresolved" and gateway is not None and cover_png_b64:
        markers = resolve_markers_with_models(versions=vlist, cover_png_b64=cover_png_b64,
                                              gateway=gateway, meta=meta)
    if len(vlist) > 1:
        alignment = DiscoveryFact({v: {"identity": True} for v in vlist}, "deterministic",
                                  "identity alignment assumed; the pipeline's derived-alignment "
                                  "check review-flags any variant whose printed order differs")
    else:
        alignment = DiscoveryFact(None, "deterministic", "single version: no alignment needed")
    template = deterministic_template(key, exam_text_layer)
    policies = deterministic_policies(key)
    if gateway is not None:
        for qid, fact in list(policies.items()):
            if fact.source == "unresolved":
                rt = (rubric_texts or {}).get(qid) or ""
                if rt:
                    policies[qid] = resolve_policy_with_models(question_id=qid, rubric_text=rt,
                                                               gateway=gateway, meta=meta)
    needs_human = [f"policy:{qid}" for qid, f in policies.items() if f.source == "unresolved"]
    if markers.source == "unresolved" and len(vlist) > 1:
        needs_human.append("variants")
    return DiscoveryResult(package_fingerprint=fp, versions=versions, variants_config=markers,
                           alignment=alignment, template=template, policies=policies,
                           needs_human=needs_human, built=time.strftime("%Y-%m-%d %H:%M:%S"))


# ------------------------------------------------------------ persistence ----


class VariantCatalogStore:
    """Resolved discovery per package fingerprint; reused for identical exams."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, fp: str) -> Path:
        return self.root / f"{fp}.json"

    def load(self, fp: str) -> dict | None:
        p = self._p(fp)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def save(self, result: DiscoveryResult, human_overrides: dict | None = None) -> None:
        d = asdict(result)
        d["human_overrides"] = human_overrides or {}
        self._p(result.package_fingerprint).write_text(json.dumps(d, ensure_ascii=False, indent=1),
                                                       encoding="utf-8")

    def apply_human(self, fp: str, **facts) -> dict:
        """Record a human resolution (e.g. variants_config=...) once; reused
        for every identical package thereafter."""
        d = self.load(fp) or {"package_fingerprint": fp}
        d.setdefault("human_overrides", {}).update(facts)
        for k, v in facts.items():
            if k in d and isinstance(d[k], dict) and "source" in d[k]:
                d[k] = {"value": v, "source": "human", "evidence": "lecturer resolution"}
        d["needs_human"] = [n for n in d.get("needs_human", []) if n not in facts and n != "variants"]
        self._p(fp).write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
        return d


def write_sidecars(result_or_dict, key_path: str | Path) -> list[Path]:
    """Emit the discovered facts as the EXISTING sidecar contracts next to
    the key: <stem>.variants.json / .alignment.json / .template.json.
    Existing files are never overwritten (a lecturer's manual mapping wins)."""
    key_path = Path(key_path)
    d = asdict(result_or_dict) if not isinstance(result_or_dict, dict) else result_or_dict
    written = []
    for name, fact in (("variants", d["variants_config"]), ("alignment", d["alignment"]),
                       ("template", d["template"])):
        val = fact.get("value") if isinstance(fact, dict) else fact
        if not val:
            continue
        p = key_path.with_name(key_path.stem + f".{name}.json")
        if p.exists():
            continue
        p.write_text(json.dumps(val, ensure_ascii=False, indent=1), encoding="utf-8")
        written.append(p)
    return written
