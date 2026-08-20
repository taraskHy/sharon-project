"""MC resolution chain: deterministic CV -> local model -> cloud model -> REVIEW.

The validated deterministic extractor stays PRIMARY and untouched. This
module only handles rows the extractor could not decide (multiple live
marks / weak evidence). Each stage receives ONLY the small row-band crop
plus the candidate letters, must return the structured MCRead below, and
is consulted lazily — the local model is loaded only when a row is
actually uncertain (Ollama's normal idle unload applies; no preloading).

Agreement rules (never guess):
- deterministic candidates + local read agree on ONE letter -> resolved
  (source "agreement", strong evidence);
- local read confident single_mark within candidates -> resolved (local);
- otherwise -> cloud read; if it agrees with local OR is a confident
  single_mark within candidates -> resolved (cloud);
- any conflict / unclear / multiple_marks at the end -> REVIEW.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

from .policies import MCResolution

MC_RESOLVER_SYSTEM = (
    "You are looking at ONE row of a printed multiple-choice answer table "
    "cropped from a scanned exam. Cells contain either nothing or a student "
    "mark (X, tick, fill, circle). A scribbled-out mark is cancelled. Report "
    "ONLY what is visible: the final selected option letter if exactly one "
    "clean mark remains, the candidate letters you can see marks in, and the "
    "state. Never guess. Reply with ONLY the JSON object."
)


class MCRead(BaseModel):
    """Structured MC read (the contract every resolver stage returns)."""

    selected: Optional[str] = Field(default=None, description="single final letter or null")
    candidates: list[str] = Field(default_factory=list)
    state: Literal["single_mark", "multiple_marks", "erased", "blank", "unclear"]
    confidence: Literal["high", "medium", "low"]


CONF = {"high": 0.95, "medium": 0.7, "low": 0.4}


@dataclass
class ChainTrace:
    stages: list[dict] = field(default_factory=list)

    def add(self, stage: str, **kw):
        self.stages.append({"stage": stage, **kw})


def _prompt_blocks(png: bytes, letters: list[str], candidates: list[str]) -> list[dict]:
    return [
        {"type": "text", "text": (
            f"Option columns (right-to-left in the image): {', '.join(letters)}. "
            f"Deterministic ink analysis found marks in: {', '.join(candidates)}. "
            "Decide only among those unless the analysis clearly missed a mark.")},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                     "data": base64.standard_b64encode(png).decode()}},
    ]


def _read_ok(read: MCRead, candidates: list[str]) -> bool:
    return (read.state == "single_mark" and read.selected is not None
            and read.selected in candidates and read.confidence in ("high", "medium"))


def resolve_row(*, band_png: bytes, letters: list[str], candidates: list[str],
                gateway=None, meta: dict | None = None,
                local_task: str = "mc_resolve", cloud_task: str = "mc_resolve_cloud",
                allow_cloud: bool = True) -> tuple[MCResolution, ChainTrace]:
    """Run the chain for one uncertain row. ``gateway`` is a ModelGateway (or
    None -> immediate REVIEW, preserving today's behavior)."""
    trace = ChainTrace()
    trace.add("deterministic", candidates=list(candidates), state="multiple_marks")
    # A crop that is not a decodable image cannot be read by any model: spend
    # no calls discovering that (imagequality; deterministic, ~1 ms).
    from .imagequality import triage_crop

    quality = triage_crop(band_png)
    if quality.status == "INVALID":
        trace.add("image_quality", status=quality.status, detail=quality.detail)
        return MCResolution(None, "unclear", 0.0, "review", list(candidates)), trace
    if gateway is None:
        return MCResolution(None, "multiple_marks", 0.0, "review", list(candidates)), trace

    def _call(task):
        try:
            gateway.route(task)
        except Exception:  # noqa: BLE001 — task not configured
            return None, "not_configured"
        try:
            res = gateway.call(task=task, system=MC_RESOLVER_SYSTEM,
                               content_blocks=_prompt_blocks(band_png, letters, candidates),
                               output_model=MCRead, meta={**(meta or {}), "stage": "mc_resolve"})
            return res.value, "ok"
        except Exception as e:  # noqa: BLE001 — a failed stage never crashes grading
            return None, f"error: {type(e).__name__}"

    local, lstat = _call(local_task)
    trace.add("local", status=lstat, read=(local.model_dump() if local else None))
    if local is not None:
        if _read_ok(local, candidates):
            src = "agreement" if len(candidates) == 1 and local.selected == candidates[0] else "local_model"
            return MCResolution(local.selected, "single_mark", CONF[local.confidence], src, list(candidates)), trace
        if local.state == "blank" and local.confidence == "high" and not candidates:
            return MCResolution(None, "blank", 0.95, "local_model", []), trace

    if not allow_cloud:
        return MCResolution(None, "unclear", 0.0, "review", list(candidates)), trace
    cloud, cstat = _call(cloud_task)
    trace.add("cloud", status=cstat, read=(cloud.model_dump() if cloud else None))
    if cloud is not None and _read_ok(cloud, candidates):
        if local is not None and local.state == "single_mark" and local.selected not in (None, cloud.selected):
            # local and cloud disagree on a letter -> do NOT guess
            trace.add("conflict", local=local.selected, cloud=cloud.selected)
            return MCResolution(None, "unclear", 0.0, "review", list(candidates)), trace
        src = "agreement" if (local is not None and local.selected == cloud.selected) else "cloud_model"
        return MCResolution(cloud.selected, "single_mark", CONF[cloud.confidence], src, list(candidates)), trace
    return MCResolution(None, "unclear", 0.0, "review", list(candidates)), trace


class MCResolverStats:
    """Review-minimization counters for the MC chain."""

    def __init__(self):
        self.rows = 0
        self.deterministic = 0
        self.local_calls = 0
        self.local_resolved = 0
        self.cloud_calls = 0
        self.cloud_resolved = 0
        self.review = 0

    def observe(self, res: MCResolution, trace: ChainTrace, deterministic_only: bool):
        self.rows += 1
        if deterministic_only:
            self.deterministic += 1
            return
        stages = {s["stage"]: s for s in trace.stages}
        if "local" in stages and stages["local"].get("status") == "ok":
            self.local_calls += 1
        if "cloud" in stages and stages["cloud"].get("status") == "ok":
            self.cloud_calls += 1
        if res.source in ("local_model", "agreement") and "cloud" not in stages:
            self.local_resolved += 1
        elif res.source in ("cloud_model", "agreement") and "cloud" in stages:
            self.cloud_resolved += 1
        elif res.source == "review":
            self.review += 1

    def as_dict(self) -> dict:
        n = self.rows or 1
        return {"rows": self.rows, "deterministic_pct": round(100 * self.deterministic / n, 1),
                "local_fallback_rate": round(100 * self.local_calls / n, 1),
                "local_resolution_success": (round(100 * self.local_resolved / self.local_calls, 1)
                                             if self.local_calls else None),
                "cloud_mc_escalation_rate": round(100 * self.cloud_calls / n, 1),
                "review_rows": self.review}
