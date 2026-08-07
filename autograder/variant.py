"""Marker-based exam-variant detection and per-variant question alignment.

Some exam families print several VARIANTS of the same exam and identify each
variant by a symbol on the cover page (here: a flower). The variant decides
which answer-key column applies, and — because variants also SHUFFLE the
order of questions and options — how the student's printed sub-item numbers
map onto the key's canonical numbering.

Hard rules, enforced structurally:

- The variant is decided by the cover marker plus an AUTHORITATIVE
  marker-to-variant mapping shipped next to the answer key
  (``<key>.variants.json``). It is NEVER inferred from the student's
  answers, the instructor's grade, or whichever key scores highest — the
  detection call receives the cover image and marker descriptions only.
- A missing/cropped/illegible/ambiguous marker produces an UNCERTAIN
  decision that is routed to human review; a variant is still chosen for
  provisional output (the mapping's first variant, deterministically — a
  documented arbitrary choice, not a score-derived one).
- Alignment maps PRINTED sub-item numbers to KEY sub-item ids by matching
  printed question CONTENT to the key's item prompts. It never sees student
  answers or the key's correct answers.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from .backends import VisionBackend
from .grade import VersionDecision
from .ingest import PageImage, image_block
from .prompts import ALIGNMENT_SYSTEM, VARIANT_DETECT_SYSTEM
from .schema import (
    AnswerKey,
    ExamSurvey,
    KeyQuestion,
    QuestionExtraction,
    VariantAlignment,
    VariantDetection,
)


# --------------------------------------------------------------------------
# marker configuration (authoritative mapping, shipped next to the key)
# --------------------------------------------------------------------------


class VariantConfigError(ValueError):
    pass


def variant_config_path(key_path: str | Path) -> Path:
    key_path = Path(key_path)
    return key_path.with_name(key_path.stem + ".variants.json")


def load_variant_config(key_path: str | Path, explicit: str | Path | None = None) -> dict | None:
    """Load the marker→variant mapping. Returns None when the exam family
    has no marker config (legacy answer-agreement detection applies)."""
    path = Path(explicit) if explicit else variant_config_path(key_path)
    if not path.exists():
        return None
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        raise VariantConfigError(f"variant config {path} is unreadable: {e}") from e
    markers = cfg.get("markers")
    if not isinstance(markers, dict) or not markers:
        raise VariantConfigError(f"variant config {path} has no 'markers' table")
    for name, entry in markers.items():
        if not isinstance(entry, dict) or "variant" not in entry or "description" not in entry:
            raise VariantConfigError(
                f"variant config {path}: marker {name!r} needs 'variant' and 'description'"
            )
    cfg["_path"] = str(path)
    return cfg


def config_fingerprint(cfg: dict) -> str:
    stable = {k: v for k, v in cfg.items() if not k.startswith("_")}
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


# --------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------


# Bump when the deterministic resolution logic changes: it enters the
# prompt-version hash so stage fingerprints invalidate (a resolver change
# can change the effective variant of past runs).
RESOLVER_VERSION = "marker-resolver-v2-description-fallback"


def _norm_tokens(text: str) -> set[str]:
    return {t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if len(t) > 2}


def resolve_marker_name(name: str | None, cfg: dict, seen: str = "") -> str | None:
    """Resolve a model-reported marker to its canonical id.

    Models sometimes echo a human alias, or perceive the marker correctly
    but copy the catalogue DESCRIPTION instead of the abstract id (observed
    live: marker_seen = the daisy description verbatim, matched_marker
    null), or describe the sighting in their own words while leaving
    matched_marker null (observed live: marker_seen = 'a diamond outline
    symbol', matched_marker null — scan 24). Resolution order, all
    deterministic:
    1. exact canonical id; 2. declared alias; 3. the reported text names
    exactly ONE canonical id or alias as a word; 4. unique description
    match — the reported text must cover ≥80 % of exactly ONE catalogue
    description's tokens. Ambiguity always resolves to nothing."""
    if name in cfg["markers"]:
        return name
    for canonical, entry in cfg["markers"].items():
        if name is not None and name in entry.get("aliases", []):
            return canonical
    reported = _norm_tokens(f"{name or ''} {seen}")
    reported_sing = {t.rstrip("s") for t in reported}
    if reported:
        named = []
        for canonical, entry in cfg["markers"].items():
            words = _norm_tokens(canonical)
            for alias in entry.get("aliases", []):
                words |= _norm_tokens(alias)
            if {w.rstrip("s") for w in words} & reported_sing:
                named.append(canonical)
        if len(set(named)) == 1:
            return named[0]
        matches = []
        for canonical, entry in cfg["markers"].items():
            desc = _norm_tokens(entry["description"])
            if desc and len(desc & reported) / len(desc) >= 0.8:
                matches.append(canonical)
        if len(matches) == 1:
            return matches[0]
    return name  # unknown — decide_version treats it as unmatched


def detect_variant(
    llm: VisionBackend, pages: list[PageImage], cfg: dict
) -> VariantDetection:
    page_num = int(cfg.get("marker_page", 1))
    page = next((p for p in pages if p.page_number == page_num), pages[0])
    catalogue = "\n".join(
        f"- {name}: {entry['description']}" for name, entry in cfg["markers"].items()
    )
    blocks = [
        {
            "type": "text",
            "text": (
                "Marker catalogue (match against EXACTLY one, or none):\n"
                + catalogue
                + (
                    f"\nExpected location: {cfg['marker_location_hint']}"
                    if cfg.get("marker_location_hint")
                    else ""
                )
            ),
        },
        image_block(page),
        {"type": "text", "text": "Identify the variant marker on this cover page now."},
    ]
    return llm.parse(
        system=VARIANT_DETECT_SYSTEM,
        content_blocks=blocks,
        output_model=VariantDetection,
        max_tokens=800,
    )


def decide_version(
    detection: VariantDetection, cfg: dict, key: AnswerKey
) -> tuple[VersionDecision, dict]:
    """Deterministically turn a detection into a VersionDecision + an
    auditable record. Never consults answers."""
    mapping = {name: entry["variant"] for name, entry in cfg["markers"].items()}
    source = cfg.get("mapping_source", {})
    record = {
        "marker_kind": cfg.get("marker_kind", "marker"),
        "marker_seen": detection.marker_seen,
        "matched_marker": detection.matched_marker,
        "confident": detection.confident,
        "page": int(cfg.get("marker_page", 1)),
        "page_region": detection.page_region,
        "obstruction_note": detection.obstruction_note,
        "mapping_from_authoritative_config": True,
        "mapping_source": source.get("derived", "variant config"),
        "config_path": cfg.get("_path"),
        "config_fingerprint": config_fingerprint(cfg),
    }

    marker = resolve_marker_name(detection.matched_marker, cfg, seen=detection.marker_seen)
    record["matched_marker"] = marker
    if detection.matched_marker != marker:
        record["marker_reported"] = detection.matched_marker
    if marker in mapping and detection.confident:
        variant = mapping[marker]
        if variant not in key.versions:
            record["error"] = f"mapped variant {variant!r} not among key versions {key.versions}"
            return (
                VersionDecision(
                    version=key.versions[0],
                    description=(
                        f"marker {marker!r} maps to {variant!r} which the key does not "
                        f"define — provisional {key.versions[0]}; human review required"
                    ),
                    uncertain=True,
                ),
                record,
            )
        return (
            VersionDecision(
                version=variant,
                description=(
                    f"cover-page {record['marker_kind']} {marker!r} ({detection.page_region}) "
                    f"→ variant {variant} per {record['mapping_source']}"
                ),
                uncertain=False,
            ),
            record,
        )

    fallback = sorted(mapping.values())[0]
    why = (
        f"marker unmatched (saw: {detection.marker_seen!r})"
        if marker is None
        else f"marker {marker!r} matched without confidence"
        if marker in mapping
        else f"model named unknown marker {marker!r}"
    )
    record["fallback_variant"] = fallback
    return (
        VersionDecision(
            version=fallback,
            description=(
                f"variant marker could not be determined ({why}); provisional "
                f"variant {fallback} chosen deterministically (first in mapping, "
                "NOT score-based) — human review required"
            ),
            uncertain=True,
        ),
        record,
    )


# --------------------------------------------------------------------------
# per-variant question alignment (printed numbering -> key numbering)
# --------------------------------------------------------------------------


def alignment_override_path(key_path: str | Path) -> Path:
    key_path = Path(key_path)
    return key_path.with_name(key_path.stem + ".alignment.json")


def load_alignment_overrides(key_path: str | Path, explicit: str | Path | None = None) -> dict | None:
    path = Path(explicit) if explicit else alignment_override_path(key_path)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise VariantConfigError(f"{path} must contain a per-variant object")
    data["_path"] = str(path)
    return data


def _canon_printed(s: str) -> str:
    """Normalize a printed sub-item id: models emit artefacts like '.1'."""
    import re as _re

    return _re.sub(r"^[^0-9A-Za-z]+|[^0-9A-Za-z]+$", "", s.strip())


def alignment_from_override(key: AnswerKey, variant: str, data: dict) -> VariantAlignment | None:
    """Build the operator-verified alignment for ``variant``. Returns None
    when the file has no entry for it. Invalid operator data is a HARD error
    — it is configuration, not model output."""
    from .schema import QuestionAlignmentEntry

    entry = data.get(variant)
    if entry is None:
        return None
    questions = []
    if entry is True or (isinstance(entry, dict) and entry.get("identity") is True):
        return identity_alignment(key, variant)
    for q in key.questions:
        q_map = entry.get(q.id)
        if q_map is None:
            raise VariantConfigError(
                f"alignment override for {variant} lacks question {q.id}"
            )
        if isinstance(q_map, dict) and q_map.get("identity") is True:
            questions.append(
                QuestionAlignmentEntry(
                    question_id=q.id,
                    printed_to_key={s.id: s.id for s in q.sub_items},
                    identical_order=True,
                )
            )
            continue
        mapping = {_canon_printed(k): str(v) for k, v in q_map.items()}
        questions.append(
            QuestionAlignmentEntry(
                question_id=q.id,
                printed_to_key=mapping,
                identical_order=all(k == v for k, v in mapping.items()),
            )
        )
    alignment = VariantAlignment(
        variant=variant,
        questions=questions,
        confident=True,
        notes=f"operator-verified alignment from {data.get('_path')}",
    )
    problems = validate_alignment(key, alignment)
    if problems:
        raise VariantConfigError(
            f"alignment override for {variant} is invalid: {'; '.join(problems)}"
        )
    return alignment


def _question_pages(qid: str, survey: ExamSurvey, pages: list[PageImage]) -> list[PageImage]:
    """The QUESTION pages for qid (booklet, not answer sheets)."""
    nums = {
        p.page_number
        for p in survey.pages
        if qid in p.question_ids and p.page_kind in ("question_or_instructions", "mixed")
    }
    return [p for p in pages if p.page_number in nums]


def alignment_fingerprint(key_fp: str, variant: str) -> str:
    h = hashlib.sha256()
    from .schema import VariantAlignment as _VA

    h.update(key_fp.encode())
    h.update(variant.encode())
    h.update(hashlib.sha256(ALIGNMENT_SYSTEM.encode("utf-8")).hexdigest().encode())
    h.update(json.dumps(_VA.model_json_schema(), sort_keys=True).encode())
    return h.hexdigest()


def load_cached_alignment(cache_dir: Path, fingerprint: str) -> VariantAlignment | None:
    path = cache_dir / f"align_{fingerprint[:32]}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            return None
        return VariantAlignment.model_validate(payload["alignment"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValidationError):
        return None


def store_cached_alignment(cache_dir: Path, fingerprint: str, alignment: VariantAlignment) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"align_{fingerprint[:32]}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "alignment": alignment.model_dump(),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    tmp.replace(path)
    return path


ALIGN_CHUNK_SIZE = 10  # sub-items per derivation call (one 20-item call collapsed live)


def derive_alignment(
    llm: VisionBackend,
    key: AnswerKey,
    survey: ExamSurvey,
    pages: list[PageImage],
    variant: str,
) -> VariantAlignment:
    """Model-derived printed→key mapping, one bounded call per question
    (large questions in chunks of ALIGN_CHUNK_SIZE key items). Content
    matching only — no answers are sent. The result is treated as
    UNVERIFIED by the pipeline regardless of validity."""
    from .schema import QuestionAlignmentEntry

    questions: list[QuestionAlignmentEntry] = []
    confident = True
    notes: list[str] = []
    for q in key.questions:
        qpages = _question_pages(q.id, survey, pages)
        if not qpages:
            sheet = set(survey.answer_sheet_policy.authoritative_pages)
            qpages = [p for p in pages if p.page_number not in sheet]
        mapping: dict[str, str] = {}
        chunks = [
            q.sub_items[i : i + ALIGN_CHUNK_SIZE]
            for i in range(0, len(q.sub_items), ALIGN_CHUNK_SIZE)
        ]
        for chunk in chunks:
            part = _derive_question_chunk(llm, key, q, chunk, qpages, variant)
            entry = next((e for e in part.questions if e.question_id == q.id), None)
            if entry is None:
                confident = False
                notes.append(f"question {q.id}: chunk returned no entry")
                continue
            mapping.update(normalized_mapping(entry))
            if not part.confident:
                confident = False
                if part.notes:
                    notes.append(part.notes)
        questions.append(
            QuestionAlignmentEntry(
                question_id=q.id,
                printed_to_key=mapping,
                identical_order=all(k == v for k, v in mapping.items()) and bool(mapping),
            )
        )
    return VariantAlignment(
        variant=variant,
        questions=questions,
        confident=confident,
        notes="; ".join(notes) if notes else None,
    )


def _derive_question_chunk(
    llm: VisionBackend,
    key: AnswerKey,
    q: KeyQuestion,
    chunk,
    qpages: list[PageImage],
    variant: str,
) -> VariantAlignment:
    ids = [s.id for s in chunk]
    blocks: list[dict] = [
        {
            "type": "text",
            "text": (
                f"Answer key question {q.id} ({q.title}) — the canonical "
                "sub-items to locate (ids and prompts only, no answers):\n"
                + json.dumps(
                    [{"id": s.id, "prompt": s.prompt} for s in chunk],
                    ensure_ascii=False,
                    indent=1,
                )
            ),
        }
    ]
    for p in qpages:
        blocks.append({"type": "text", "text": f"--- Page {p.page_number} ---"})
        blocks.append(image_block(p))
    blocks.append(
        {
            "type": "text",
            "text": (
                f"This exam form is variant {variant!r}. For EACH of the "
                f"{len(ids)} key sub-items above (ids {', '.join(ids)}), find "
                "the question with that CONTENT on these pages and report the "
                "number PRINTED next to it. Output one questions[] entry for "
                f"question {q.id!r} whose printed_to_key maps each PRINTED "
                "number to its key id — ONLY for these key ids; other items "
                "are handled separately."
            ),
        }
    )
    return llm.parse(
        system=ALIGNMENT_SYSTEM,
        content_blocks=blocks,
        output_model=VariantAlignment,
        max_tokens=2500,
    )


def validate_alignment(key: AnswerKey, alignment: VariantAlignment) -> list[str]:
    """Deterministic checks. Returns problem descriptions (empty = usable).
    An unusable alignment must never be silently applied."""
    problems: list[str] = []
    by_q = {e.question_id: e for e in alignment.questions}
    for q in key.questions:
        entry = by_q.get(q.id)
        if entry is None:
            problems.append(f"question {q.id}: no alignment entry")
            continue
        key_ids = [s.id for s in q.sub_items]
        mapping = normalized_mapping(entry)
        if any(not k for k in mapping):
            problems.append(f"question {q.id}: empty printed id after normalization")
        targets = list(mapping.values())
        missing = [k for k in key_ids if k not in targets]
        dupes = sorted({t for t in targets if targets.count(t) > 1})
        extra = [t for t in targets if t not in key_ids]
        if missing:
            problems.append(f"question {q.id}: key items unmapped: {missing}")
        if dupes:
            problems.append(f"question {q.id}: duplicate targets: {dupes}")
        if extra:
            problems.append(f"question {q.id}: unknown key ids: {extra}")
        if len(mapping) != len(entry.printed_to_key):
            problems.append(
                f"question {q.id}: printed ids collide after normalization"
            )
        if len(mapping) != len(key_ids):
            problems.append(
                f"question {q.id}: {len(mapping)} printed items mapped, "
                f"key has {len(key_ids)}"
            )
    return problems


def identity_alignment(key: AnswerKey, variant: str) -> VariantAlignment:
    from .schema import QuestionAlignmentEntry

    return VariantAlignment(
        variant=variant,
        questions=[
            QuestionAlignmentEntry(
                question_id=q.id,
                printed_to_key={s.id: s.id for s in q.sub_items},
                identical_order=True,
            )
            for q in key.questions
        ],
        confident=True,
        notes="identity (no alignment derivation ran)",
    )


def normalized_mapping(entry) -> dict[str, str]:
    """The entry's printed→key map with printed ids normalized (models emit
    artefacts like '.1'; operator files may have stray spaces)."""
    return {_canon_printed(k): str(v) for k, v in entry.printed_to_key.items()}


def printed_view(q: KeyQuestion, entry) -> KeyQuestion:
    """The key question relabeled into this variant's PRINTED numbering, so
    extraction works in the numbering the student actually saw. Prompts
    follow the content: printed number N gets the prompt of the key item it
    maps to."""
    view = q.model_copy(deep=True)
    by_key_id = {s.id: s for s in q.sub_items}
    printed_items = []
    for printed, key_id in sorted(normalized_mapping(entry).items(), key=lambda kv: _num(kv[0])):
        src = by_key_id[key_id]
        item = src.model_copy(deep=True)
        item.id = printed
        printed_items.append(item)
    view.sub_items = printed_items
    return view


def remap_extraction(qx: QuestionExtraction, entry) -> QuestionExtraction:
    """Rewrite extraction sub-item ids from printed numbering back to key
    numbering (provenance keeps the printed number)."""
    mapping = normalized_mapping(entry)
    for s in qx.sub_items:
        printed = _canon_printed(s.sub_item_id)
        key_id = mapping.get(printed)
        if key_id is None:
            continue  # reconciliation will flag it against the key
        if key_id != printed:
            s.source_region = (
                f"{s.source_region} (printed #{printed})"
                if s.source_region
                else f"printed #{printed}"
            )
        s.sub_item_id = key_id
    return qx


def _num(s: str):
    try:
        return (0, int(s))
    except ValueError:
        return (1, s)
