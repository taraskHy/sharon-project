"""Persistent cache for parsed answer keys.

Parsing an answer-key document is the single most expensive model call in
the pipeline (~20K prompt tokens; ≈10 minutes on the local GPU). The parsed
result depends only on the key document, the rubric, the model/backend
configuration, the parsing prompt, the render settings and the output
schema — so it is cached on disk keyed by a fingerprint over exactly those
inputs and reused across runs and across exams in a batch.

Invalidation is automatic: the fingerprint embeds a hash of the parser's
system prompt text and of the ``AnswerKey`` JSON schema, so editing either
in the source invalidates every stale entry without manual version bumps.
Corrupted, truncated or schema-incompatible cache files are rejected and
re-parsed; they are never trusted and never crash a run.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from .schema import AnswerKey

CACHE_FORMAT = 1  # bump when the cache file layout itself changes


def default_cache_dir() -> Path:
    env = os.environ.get("GRADER_KEY_CACHE")
    if env:
        return Path(env)
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "autograder" / "key_cache"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "autograder" / "key_cache"


def key_fingerprint(
    *,
    key_bytes_hash: str,
    rubric_text: str | None,
    backend_description: dict,
    max_image_edge: int,
    parser_prompt: str,
) -> str:
    """Fingerprint of everything that can change the PARSE itself.

    ``backend_description`` is ``backend.describe()`` — backend type, model
    tag, base_url, structured mode and generation parameters. The prompt and
    schema enter as content hashes so source edits self-invalidate.

    The operator version-override file is deliberately NOT part of this
    fingerprint: overrides are re-applied deterministically on EVERY load
    (including cache hits), so editing them must not force a re-parse of the
    unchanged document — they invalidate the per-exam GRADING fingerprints
    instead (see cli._fingerprints).
    """
    h = hashlib.sha256()
    parts = {
        "format": CACHE_FORMAT,
        "key_bytes": key_bytes_hash,
        "rubric": rubric_text or "",
        "backend": backend_description,
        "max_image_edge": max_image_edge,
        "parser_prompt_sha": hashlib.sha256(parser_prompt.encode("utf-8")).hexdigest(),
        "schema_sha": hashlib.sha256(
            json.dumps(AnswerKey.model_json_schema(), sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    h.update(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    return h.hexdigest()


def load_cached_key(cache_dir: Path, fingerprint: str) -> AnswerKey | None:
    """Return the cached AnswerKey, or None when absent/corrupt/incompatible."""
    path = cache_dir / f"{fingerprint}.json"
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("fingerprint") != fingerprint:
            return None  # file renamed/copied around — do not trust it
        return AnswerKey.model_validate(payload["answer_key"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValidationError):
        return None


def store_cached_key(
    cache_dir: Path, fingerprint: str, key: AnswerKey, components_note: dict | None = None
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{fingerprint}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "format": CACHE_FORMAT,
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "components": components_note or {},
                "answer_key": key.model_dump(),
            },
            ensure_ascii=False,
            indent=1,
        ),
        encoding="utf-8",
    )
    tmp.replace(path)  # atomic-ish: never leaves a half-written cache entry
    return path
