"""Owner grading labels — the evaluation-side human labels for the grading
benchmarks, kept in a SEPARATE, incrementally written file next to the frozen
dataset (never inside it):

    <datasets_root>/grade_primary/owner_labels.json
        {"version": 1, "entries": {case_id: {score, rubric_met, note, status, labeled_at}}}

Rules: originals are never touched (the frozen cases_inputs/labels files are
immutable); every decision is written atomically and can be revisited; only
``status == "confirmed"`` entries become scoring labels; ``merge_owner_labels``
is the single place the manifest loader reads them. The UI is
scripts/grade_label_ui.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

OWNER_LABELS_FILENAME = "owner_labels.json"
STATUSES = ("confirmed", "skipped")


class OwnerLabelError(RuntimeError):
    pass


class OwnerLabelStore:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = Path(dataset_dir)
        self.path = self.dataset_dir / OWNER_LABELS_FILENAME
        self._lock = threading.Lock()
        self._data: dict[str, Any] = {"version": 1, "entries": {}}
        self._reload()

    # -- persistence -----------------------------------------------------------
    def _reload(self) -> None:
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
            self._data.setdefault("entries", {})

    def _atomic_write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f"{self.path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
        try:
            tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=1), encoding="utf-8", newline="\n")
            os.replace(tmp, self.path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    # -- api ---------------------------------------------------------------------
    @property
    def entries(self) -> dict[str, dict]:
        return dict(self._data.get("entries", {}))

    def get(self, case_id: str) -> dict | None:
        return self._data.get("entries", {}).get(case_id)

    def record(self, case_id: str, *, score: float | None, max_score: float | None = None,
               rubric_met: list[str] | None = None, note: str = "", status: str = "confirmed",
               now: str | None = None) -> dict:
        if status not in STATUSES:
            raise OwnerLabelError(f"unknown status {status!r}")
        if status == "confirmed":
            if score is None:
                raise OwnerLabelError("a confirmed label needs a final score")
            score = float(score)
            if score < 0 or (max_score is not None and score > float(max_score) + 1e-9):
                raise OwnerLabelError(f"score {score} outside 0..{max_score}")
        with self._lock:
            self._reload()            # merge concurrent writers (second UI tab)
            entry = {"score": (float(score) if score is not None else None),
                     "rubric_met": list(rubric_met or []), "note": note or "", "status": status,
                     "labeled_at": now or time.strftime("%Y-%m-%d %H:%M:%S")}
            self._data.setdefault("entries", {})[case_id] = entry
            self._atomic_write()
        return entry

    def reset(self, case_id: str) -> None:
        with self._lock:
            self._reload()
            self._data.get("entries", {}).pop(case_id, None)
            self._atomic_write()

    def summary(self, case_ids: list[str]) -> dict[str, Any]:
        e = self._data.get("entries", {})
        confirmed = [c for c in case_ids if e.get(c, {}).get("status") == "confirmed"]
        skipped = [c for c in case_ids if e.get(c, {}).get("status") == "skipped"]
        remaining = [c for c in case_ids if c not in e]
        return {"total": len(case_ids), "confirmed": len(confirmed), "skipped": len(skipped),
                "remaining": len(remaining), "remaining_ids": remaining, "path": str(self.path)}

    def sha256(self) -> str | None:
        if not self.path.exists():
            return None
        return hashlib.sha256(self.path.read_bytes()).hexdigest()


def merge_owner_labels(labels_by_id: dict[str, dict], store: OwnerLabelStore) -> int:
    """Write confirmed owner decisions into the evaluation-side label dicts
    (score, rubric_met, owner_note, owner_status). Returns the count merged."""
    n = 0
    for cid, entry in store.entries.items():
        lab = labels_by_id.get(cid)
        if lab is None:
            continue
        lab["owner_status"] = entry.get("status")
        if entry.get("status") == "confirmed":
            lab["score"] = entry.get("score")
            lab["rubric_met"] = list(entry.get("rubric_met") or []) or None
            lab["owner_note"] = entry.get("note") or None
            n += 1
    return n


__all__ = ["OWNER_LABELS_FILENAME", "OwnerLabelStore", "OwnerLabelError", "merge_owner_labels"]
