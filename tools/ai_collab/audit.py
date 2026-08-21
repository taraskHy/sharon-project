"""Append-only audit log (JSONL) per run: timestamps, models, usage, cost,
cache hits, round transitions, stop reason. Developer-tooling auditability —
never student data, never API keys.
"""

from __future__ import annotations

import json
from pathlib import Path

from .util import now_iso


def log_event(audit_path: Path, event: str, **fields) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": now_iso(), "event": event}
    record.update(fields)
    with open(audit_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, sort_keys=False) + "\n")
