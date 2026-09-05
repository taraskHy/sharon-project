"""Deterministic request fingerprint + local result cache for the gateway.

Fingerprint inputs (any change invalidates): task, backend, model, prompt
version, generation parameters (max_tokens, temperature, reasoning, extra
generation knobs), the system prompt hash, every text block hash, every
image hash, the output schema hash, and any caller-declared pack hash
(e.g. QuestionGradingPack.hash) passed in ``meta``. Successful validated
results only are stored — transient failures never enter the cache.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def _h(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def fingerprint(route, system: str, content_blocks: list[dict],
                output_model: type[BaseModel], max_tokens: int | None,
                meta: dict | None = None) -> str:
    """The SEMANTIC REQUEST IDENTITY: may a stored response be reused here?

    Derived from the EFFECTIVE backend configuration (see
    ``autograder.routeidentity``), not from a hand-maintained field list. The
    old list omitted ``provider``, so a provider-pinned request could be served
    a response produced by an UNPINNED run — which is exactly what happened to
    five of eight cases in the V1 alt-candidate screen.

    The identity is VERSIONED, so keys written under the old scheme can never
    be read by the corrected one. Historical entries stay on disk untouched.
    """
    from .routeidentity import semantic_request_identity, wire_response_format

    # The CANONICAL WIRE SCHEMA — the response_format block as transmitted,
    # including the strict transform and the schema name — not the raw
    # model_json_schema(), which is only a proxy for it.
    base = semantic_request_identity(
        route, system=system, content_blocks=content_blocks,
        schema=wire_response_format(route, output_model), max_tokens=max_tokens)
    pack = (meta or {}).get("pack_hash")
    return base if pack is None else _h(json.dumps({"base": base, "pack_hash": pack},
                                                   sort_keys=True))


class RequestCache:
    """Content-addressed JSON store: <root>/<fp[:2]>/<fp>.json."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.hits = 0
        self.misses = 0

    def _path(self, fp: str) -> Path:
        return self.root / fp[:2] / f"{fp}.json"

    def get(self, fp: str, output_model: type[BaseModel]):
        p = self._path(fp)
        if not p.exists():
            self.misses += 1
            return None
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            obj = output_model.model_validate(rec["value"])
        except Exception:  # noqa: BLE001 — corrupt entry = miss, never a crash
            self.misses += 1
            return None
        self.hits += 1
        return obj

    def put(self, fp: str, value: BaseModel, meta: dict | None = None) -> None:
        p = self._path(fp)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "fingerprint": fp,
            "stored": time.strftime("%Y-%m-%d %H:%M:%S"),
            "meta": meta or {},
            "value": value.model_dump(mode="json"),
        }, ensure_ascii=False), encoding="utf-8")

    def stats(self) -> dict:
        n = self.hits + self.misses
        return {"hits": self.hits, "misses": self.misses,
                "hit_rate": round(self.hits / n, 4) if n else None}
