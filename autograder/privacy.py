"""Privacy minimisation for provider requests.

A cloud provider needs the crop or the answer text and the rubric. It does
not need to know WHOSE exam this is, what the file was called, or where it
sits on disk. Intake already anonymises exam ids (``jobs.intake_exams``);
this module makes the guarantee structural at the point every request is
built:

- request construction is a WHITELIST — only the fields listed here reach a
  provider payload, so a new identifying field cannot leak by being added
  upstream;
- OCR sends the required crop plus an anonymous item id, nothing else;
- grading sends the anonymous item id, the question/rubric context, the
  frozen student text, the selected option, and the cited evidence;
- ``scan_for_identifiers`` is an audit hook used by the tests to prove that
  fabricated identifying metadata never appears in a payload;
- log/ledger records carry the internal item id, never the original filename
  or a filesystem path.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import Any, Iterable, Optional

#: Metadata keys that may accompany a provider request (routing/accounting only).
#: exam_id is the ANONYMIZED internal id (jobs intake / safe_log_name) — never
#: a raw filename. The rag_* keys are numbers-only RAG accounting.
PROVIDER_META_ALLOWED = frozenset({
    "job_id", "exam_id", "item_id", "question_id", "sub_item_id", "stage", "task",
    "pack_hash", "prompt_version", "attempt", "variant",
    "rag_policy", "rag_chars", "rag_chunks",
})

#: Metadata keys that must never reach a provider (and never a log line).
IDENTITY_KEYS = frozenset({
    "student_name", "name", "full_name", "student_id", "id_number", "national_id",
    "email", "phone", "school", "university", "institution", "class", "course_section",
    "original_name", "original_filename", "filename", "file", "path", "filepath",
    "source_path", "upload_name", "batch_name", "exam_file", "author", "owner",
})

_PATHISH = re.compile(r"(?:[A-Za-z]:[\\/])|(?:\\\\)|(?:/(?:home|users|mnt|var)/)", re.I)


class PrivacyError(ValueError):
    """A request could not be built without leaking identifying data."""


# --------------------------------------------------------------------------
# anonymous ids
# --------------------------------------------------------------------------


def anonymous_item_id(*parts: Any, salt: str = "") -> str:
    """A stable, non-reversible internal id for one gradeable item.

    Built from ALREADY anonymous parts (job id, internal exam id, question,
    sub-item). It is stable across runs so caching, reuse and the usage
    ledger still line up, and it carries no student identity.
    """
    payload = "|".join(str(p) for p in parts)
    return "item-" + hashlib.sha256((salt + payload).encode("utf-8")).hexdigest()[:16]


def safe_log_name(name: str | PurePath) -> str:
    """The loggable form of a file reference: its stem only when that stem is
    already an internal id, otherwise an opaque digest. Never a path."""
    stem = PurePath(str(name)).stem
    if re.fullmatch(r"(exam[-_]?\d+|item-[0-9a-f]{6,}|q\d+)", stem, re.I):
        return stem
    return "file-" + hashlib.sha256(str(name).encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------------------------
# scrubbing + auditing
# --------------------------------------------------------------------------


def scrub_meta(meta: dict | None) -> dict:
    """Whitelist filter for request/ledger metadata."""
    return {k: v for k, v in (meta or {}).items() if k in PROVIDER_META_ALLOWED}


def _walk(obj: Any, skip_keys: tuple[str, ...] = ()):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield ("key", str(k))
            if str(k) in skip_keys:
                continue          # e.g. base64 image payloads: opaque, not text
            yield from _walk(v, skip_keys)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v, skip_keys)
    elif isinstance(obj, str):
        yield ("value", obj)
    elif obj is not None:
        yield ("value", str(obj))


def scan_for_identifiers(payload: Any, identifiers: Iterable[str] = (),
                         skip_keys: tuple[str, ...] = ("data",)) -> list[str]:
    """Every identity leak found in ``payload``: forbidden keys, any of the
    caller-supplied identifier strings, and anything that looks like a
    filesystem path. Used by the privacy tests and by the pre-send assertion."""
    found: list[str] = []
    needles = [str(s) for s in identifiers if str(s).strip()]
    for kind, text in _walk(payload, skip_keys):
        if kind == "key" and text.lower() in IDENTITY_KEYS:
            found.append(f"forbidden key {text!r}")
            continue
        if kind == "value":
            for n in needles:
                if n and n.lower() in text.lower():
                    found.append(f"identifier {n!r} present in payload")
            if _PATHISH.search(text):
                found.append(f"filesystem path in payload: {text[:60]!r}")
    return sorted(set(found))


def assert_anonymous(payload: Any, identifiers: Iterable[str] = ()) -> None:
    problems = scan_for_identifiers(payload, identifiers)
    if problems:
        raise PrivacyError("; ".join(problems))


def scan_blocks(content_blocks: list[dict]) -> tuple[list[str], list[str]]:
    """Pre-send check on the blocks that actually reach a provider.

    Returns (hard, soft): ``hard`` are forbidden metadata keys, which are
    never legitimate inside a content block and must abort the request;
    ``soft`` are path-like strings, which are recorded as warnings because a
    student could conceivably have written one by hand.
    """
    hard, soft = [], []
    for kind, text in _walk(content_blocks, ("data",)):
        if kind == "key" and text.lower() in IDENTITY_KEYS:
            hard.append(f"forbidden key {text!r} in a provider content block")
        elif kind == "value" and _PATHISH.search(text):
            soft.append(f"path-like string in a provider content block: {text[:60]!r}")
    return sorted(set(hard)), sorted(set(soft))


# --------------------------------------------------------------------------
# request construction (the only sanctioned way to build a provider payload)
# --------------------------------------------------------------------------


@dataclass
class ProviderRequest:
    """What actually goes to a provider: nothing else is added downstream."""

    task: str
    content_blocks: list[dict]
    meta: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"task": self.task, "content_blocks": self.content_blocks, "meta": self.meta}


def build_ocr_request(*, item_id: str, crop_png_b64: str, task: str = "ocr_primary",
                      instruction: str = "Transcribe the handwriting in this image verbatim.",
                      meta: dict | None = None) -> ProviderRequest:
    """OCR gets the crop and an anonymous item id — nothing about the student,
    the file, the batch or the other pages of the exam."""
    if not crop_png_b64:
        raise PrivacyError("an OCR request needs a crop")
    blocks = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": crop_png_b64}},
        {"type": "text", "text": f"[{item_id}] {instruction}"},
    ]
    return ProviderRequest(task, blocks, {**scrub_meta(meta), "item_id": item_id})


def build_grading_request(*, item_id: str, question_context: str, transcription: str,
                          selected_option: str | None = None, evidence: str | None = None,
                          task: str = "grade_primary", meta: dict | None = None) -> ProviderRequest:
    """Grading gets the anonymous item id, the question/rubric context, the
    frozen student text, the selected option and any cited evidence."""
    parts = [f"[{item_id}]", question_context.strip()]
    if selected_option:
        parts.append(f"Student selected option: {selected_option}")
    parts.append("Student answer (verbatim, do not rewrite):\n---\n" + (transcription or "") + "\n---")
    if evidence:
        parts.append(f"Evidence under review: {evidence}")
    return ProviderRequest(task, [{"type": "text", "text": "\n".join(parts)}],
                           {**scrub_meta(meta), "item_id": item_id})


def safe_ledger_entry(entry: dict) -> dict:
    """Ledger/log records: identifying keys dropped, file references reduced
    to internal ids."""
    out = {}
    for k, v in (entry or {}).items():
        if k.lower() in IDENTITY_KEYS:
            if k.lower() in ("exam_file", "file", "filename", "original_name", "path"):
                out["item_ref"] = safe_log_name(v) if v else None
            continue
        if isinstance(v, str) and _PATHISH.search(v):
            out[k] = safe_log_name(v)
            continue
        out[k] = v
    return out
