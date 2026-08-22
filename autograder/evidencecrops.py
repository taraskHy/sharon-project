"""Explanation evidence crops — the production OCR-verifier input contract.

The OCR verifier (task ``ocr_verify``) judges ONE handwritten line/cell image
against ONE candidate transcription. In production it therefore needs, per
(question, sub-item), a trustworthy crop of exactly the region where the
student wrote that explanation. This module is the explicit interface for
that crop producer — and records, honestly, that production does not have
one yet:

* ``PageRegion`` in schema.py is descriptive only (model-reported, never
  calibrated against real scans) and ``tablecrop`` covers MC answer rows,
  not free-text explanation areas. There is NO generic, calibrated
  explanation-region geometry (docs/ocr-verifier-audit.md, item UNWIRED).
* Inventing coordinates, or silently sending the FULL PAGE as "the crop",
  would hand the verifier an image that does not match the candidate line:
  a false SUPPORTED is then likely and would look like evidence. That is
  the one failure the verifier exists to prevent.

So the production provider is ``UnavailableCropProvider``: it returns
``UNAVAILABLE`` for every item with the reason recorded, and the reliability
route then behaves fail-closed — a suspicious reading with no evidence crop
becomes REVIEW ("suspicious; no evidence crop available"), never AUTO and
never a verifier call over an arbitrary region (escalation.escalate_ocr).

Everything around the missing producer is wired and tested: the interface,
deterministic quality triage of any crop that IS supplied
(imagequality.triage_crop via reliability._crop_quality), verifier routing
through the gateway (cache, budget, ledger, privacy scan), the trace stage,
and the typed review reason. When a calibrated crop producer exists it
plugs in here, and nothing downstream changes.

Fallback behaviour (documented contract):
    crop AVAILABLE   -> triage -> verifier call (if ocr_verify is usable)
    crop UNAVAILABLE -> no verifier call; suspicious items -> REVIEW with
                        reason "suspicious; no evidence crop available";
                        unsuspicious items proceed to grading as before
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

CROP_AVAILABLE = "AVAILABLE"
CROP_UNAVAILABLE = "UNAVAILABLE"

#: Why production has no crop producer today (single source of truth; the
#: readiness report, the GUI diagnostics and the grade log all quote it).
PRODUCTION_UNAVAILABLE_REASON = (
    "no calibrated per-question explanation-region geometry exists in production "
    "(PageRegion is descriptive only; tablecrop covers MC rows); coordinates are "
    "not invented and the full page is never sent as a crop"
)


@dataclass(frozen=True)
class CropResult:
    """Outcome of asking the provider for one (question, sub-item) crop."""

    status: str                      # CROP_AVAILABLE | CROP_UNAVAILABLE
    png_b64: str | None = None       # base64 PNG when AVAILABLE
    reason: str = ""                 # why UNAVAILABLE (human-readable)
    source: str = ""                 # producer identity, e.g. "unavailable", "benchmark_fixture"
    geometry: dict | None = None     # producer-specific provenance (page, bbox, calibration id)

    @property
    def available(self) -> bool:
        return self.status == CROP_AVAILABLE and bool(self.png_b64)


class ExplanationCropProvider(Protocol):
    """The crop-producer interface. Implementations must never fabricate a
    region: return UNAVAILABLE with a reason instead."""

    name: str

    def crop(self, question_id: str, sub_item_id: str) -> CropResult: ...

    def describe(self) -> dict: ...


@dataclass
class UnavailableCropProvider:
    """The explicit production default: every crop is UNAVAILABLE."""

    reason: str = PRODUCTION_UNAVAILABLE_REASON
    name: str = "unavailable"

    def crop(self, question_id: str, sub_item_id: str) -> CropResult:
        return CropResult(CROP_UNAVAILABLE, None, self.reason, self.name)

    def describe(self) -> dict:
        return {"provider": self.name, "status": CROP_UNAVAILABLE, "reason": self.reason,
                "fallback": "fail-closed: suspicious readings -> REVIEW; no verifier call"}


@dataclass
class StaticCropProvider:
    """Crops supplied explicitly by the caller (tests, benchmark fixtures, a
    future calibrated producer that pre-computes crops). Keys are
    (question_id, sub_item_id); values are base64 PNG strings."""

    crops: dict[tuple[str, str], str] = field(default_factory=dict)
    name: str = "static"
    geometry: dict[tuple[str, str], dict] = field(default_factory=dict)

    def crop(self, question_id: str, sub_item_id: str) -> CropResult:
        b64 = self.crops.get((question_id, sub_item_id))
        if not b64:
            return CropResult(CROP_UNAVAILABLE, None,
                              "no crop supplied for this item", self.name)
        return CropResult(CROP_AVAILABLE, b64, "", self.name,
                          self.geometry.get((question_id, sub_item_id)))

    def describe(self) -> dict:
        return {"provider": self.name, "status": CROP_AVAILABLE if self.crops else CROP_UNAVAILABLE,
                "items": len(self.crops)}


def production_crop_provider() -> ExplanationCropProvider:
    """The provider the grading pipeline uses. Deliberately UNAVAILABLE until
    a calibrated explanation-region producer exists; see the module docstring.
    Never returns a full-page provider."""
    return UnavailableCropProvider()


def collect_crops(provider: ExplanationCropProvider, key) -> tuple[dict[tuple[str, str], str], dict]:
    """Ask the provider for every explanation-bearing (question, sub-item) of
    ``key`` (an AnswerKey). Returns the crops dict the reliability route
    consumes (ONLY available crops) plus an availability report for the
    trace/log/GUI: counts and the first reason."""
    crops: dict[tuple[str, str], str] = {}
    unavailable = 0
    reasons: dict[str, int] = {}
    for q in getattr(key, "questions", []) or []:
        for s in getattr(q, "sub_items", []) or []:
            res = provider.crop(q.id, s.id)
            if res.available:
                crops[(q.id, s.id)] = res.png_b64  # type: ignore[assignment]
            else:
                unavailable += 1
                reasons[res.reason] = reasons.get(res.reason, 0) + 1
    report = {
        **provider.describe(),
        "items_with_crop": len(crops),
        "items_without_crop": unavailable,
        "unavailable_reasons": reasons,
    }
    return crops, report


__all__ = ["CROP_AVAILABLE", "CROP_UNAVAILABLE", "PRODUCTION_UNAVAILABLE_REASON",
           "CropResult", "ExplanationCropProvider", "UnavailableCropProvider",
           "StaticCropProvider", "production_crop_provider", "collect_crops"]
