"""Small shared helpers: hashing, token estimation, atomic JSON, truncation."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def sha256_json(obj) -> str:
    canon = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_text(canon)


def est_tokens(text: str) -> int:
    """Crude, deterministic token estimate (chars/4). Used only for budgets."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, path)


def read_text(path: Path, max_bytes: int | None = None) -> str:
    data = Path(path).read_bytes()
    if max_bytes is not None and len(data) > max_bytes:
        data = data[:max_bytes]
    return data.decode("utf-8", errors="replace")


def truncate_middle(text: str, max_chars: int, label: str = "") -> tuple[str, bool]:
    """Deterministically truncate keeping the head and tail of *text*."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    head = int(max_chars * 0.6)
    tail = int(max_chars * 0.25)
    omitted = len(text) - head - tail
    marker = f"\n[... TRUNCATED by orchestrator: {omitted} chars omitted {label}...]\n"
    return text[:head] + marker + text[len(text) - tail :], True


def truncate_tail(text: str, max_chars: int, label: str = "") -> tuple[str, bool]:
    """Keep the END of *text* (useful for test output where failures come last)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    omitted = len(text) - max_chars
    marker = f"[... TRUNCATED by orchestrator: first {omitted} chars omitted {label}...]\n"
    return marker + text[len(text) - max_chars :], True


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    slug = _SLUG_RE.sub("-", name.lower()).strip("-")
    return slug or "task"
