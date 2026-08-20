"""Deterministic pre-OCR image triage.

A cloud OCR call on a blank, clipped, or empty crop buys nothing: the model
will confabulate rather than report "there is nothing here" (measured, see
evaluation/hebrew_transcription_loop.md). This stage answers "is there
readable content in this crop at all?" with plain geometry and pixel
statistics — NO model, no network, ~1 ms per crop.

Thresholds are deliberately conservative: legitimate messy handwriting must
pass. The failure mode we accept is "we sent a bad crop to OCR anyway"; the
failure mode we refuse is "we rejected a real answer as blank".

Where the caller can cheaply produce a better image (re-render the page at a
higher dpi, widen the crop margin, use an alternate page rendering), the
result says so via ``recommended_action`` and ``triage_with_recovery`` runs
those deterministic retries BEFORE anything escalates to a human.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

# --- thresholds (conservative; see module docstring) -----------------------
DARK = 160                  # gray value below which a pixel counts as ink/print
BLANK_INK_FRACTION = 0.0025  # below this the crop is effectively empty
FLOODED_INK_FRACTION = 0.75  # above this the crop is a black smear, not writing
MIN_SIDE_PX = 12            # smaller than this cannot hold a legible glyph
MAX_ASPECT = 80.0           # a 80:1 sliver is a geometry bug, not a text line
LOW_CONTRAST_RANGE = 35     # median paper level minus the mean of the darkest 0.5%
BORDER_PX = 2               # thickness of the border band tested for clipping
BORDER_INK_FRACTION = 0.35  # ink filling this much of a border band = clipped
SKEW_DEGREES = 20.0         # principal-axis tilt beyond this is extreme

STATUSES = ("OK", "BLANK", "LOW_CONTRAST", "CLIPPED", "EXTREME_SKEW", "SUSPICIOUS_CROP", "INVALID")

@dataclass
class ImageQualityResult:
    status: str
    signals: dict[str, Any] = field(default_factory=dict)
    recoverable: bool = False
    recommended_action: str = "proceed"     # proceed|expand_crop|rerender|skip_ocr|escalate
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "OK"

    def as_dict(self) -> dict:
        return {"status": self.status, "signals": dict(self.signals),
                "recoverable": self.recoverable, "recommended_action": self.recommended_action,
                "detail": self.detail}


def _decode_gray(png: bytes) -> Optional[np.ndarray]:
    """PNG bytes -> 2-D uint8 grayscale, or None when undecodable."""
    if not png:
        return None
    try:
        import fitz
        pix = fitz.Pixmap(png)
        a = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    except Exception:  # noqa: BLE001 — any decode failure is INVALID, never a crash
        return None
    if a.size == 0:
        return None
    return a[:, :, :3].mean(axis=2).astype(np.uint8) if a.shape[2] >= 3 else a[:, :, 0].copy()


def _skew_degrees(mask: np.ndarray) -> Optional[float]:
    """Tilt of the ink's principal axis, in degrees from horizontal."""
    ys, xs = np.nonzero(mask)
    if xs.size < 30:
        return None
    x = xs.astype(np.float64) - xs.mean()
    y = ys.astype(np.float64) - ys.mean()
    cov = np.cov(np.vstack([x, y]))
    if not np.all(np.isfinite(cov)):
        return None
    vals, vecs = np.linalg.eigh(cov)
    if vals[1] <= 0 or vals[1] / max(vals[0], 1e-9) < 3.0:
        # A round blob has no meaningful principal axis — do not claim skew.
        return None
    vx, vy = vecs[:, 1]
    return float(abs(np.degrees(np.arctan2(vy, vx))) % 180.0)


def triage_crop(png: bytes, *, expected_size: tuple[int, int] | None = None,
                expect_content: bool = True) -> ImageQualityResult:
    """Classify ONE crop. ``expected_size`` (w, h), when known from the page
    template, catches rendering/geometry failures that pixel statistics alone
    cannot see."""
    gray = _decode_gray(png)
    if gray is None:
        return ImageQualityResult("INVALID", {"bytes": len(png or b"")}, True, "rerender",
                                  "the crop is missing or not a decodable image")
    h, w = gray.shape
    sig: dict[str, Any] = {"width": int(w), "height": int(h)}
    if w < MIN_SIDE_PX or h < MIN_SIDE_PX:
        return ImageQualityResult("SUSPICIOUS_CROP", sig, True, "rerender",
                                  f"crop is {w}x{h}px — too small to hold legible writing")
    aspect = max(w / h, h / w)
    sig["aspect"] = round(aspect, 2)
    if aspect > MAX_ASPECT:
        return ImageQualityResult("SUSPICIOUS_CROP", sig, True, "rerender",
                                  f"crop aspect ratio {aspect:.0f}:1 indicates a geometry error")
    if expected_size:
        ew, eh = expected_size
        drift = max(abs(w - ew) / max(ew, 1), abs(h - eh) / max(eh, 1))
        sig["size_drift"] = round(drift, 3)
        if drift > 0.5:
            return ImageQualityResult("SUSPICIOUS_CROP", sig, True, "rerender",
                                      f"crop is {w}x{h}px where the template expects {ew}x{eh}px")

    mask = gray < DARK
    ink = float(mask.mean())
    p2, p98 = (float(x) for x in np.percentile(gray, [2, 98]))
    # Ink-vs-paper separation, NOT the histogram spread: sparse handwriting on
    # clean paper has a p2==p98 histogram while being perfectly readable.
    k = max(20, int(0.005 * gray.size))
    darkest = float(np.sort(gray, axis=None)[:k].mean())
    paper = float(np.median(gray))
    contrast = paper - darkest
    sig.update({"ink_fraction": round(ink, 5), "gray_p2": p2, "gray_p98": p98,
                "ink_level": round(darkest, 1), "paper_level": round(paper, 1),
                "contrast_range": round(contrast, 1), "mean_gray": round(float(gray.mean()), 1)})

    if ink > FLOODED_INK_FRACTION:
        return ImageQualityResult("SUSPICIOUS_CROP", sig, True, "rerender",
                                  f"{ink:.0%} of the crop is dark — a scan/render artefact, not writing")
    if ink < BLANK_INK_FRACTION:
        # Nothing to transcribe: an OCR call here can only invent text.
        return ImageQualityResult("BLANK", sig, False, "skip_ocr" if expect_content else "proceed",
                                  f"only {ink:.3%} of the crop carries ink")
    if contrast < LOW_CONTRAST_RANGE:
        return ImageQualityResult("LOW_CONTRAST", sig, True, "rerender",
                                  f"ink/paper separation {contrast:.0f} is too flat to read reliably")

    top = float(mask[:BORDER_PX, :].mean()) if h > 2 * BORDER_PX else 0.0
    bottom = float(mask[-BORDER_PX:, :].mean()) if h > 2 * BORDER_PX else 0.0
    left = float(mask[:, :BORDER_PX].mean()) if w > 2 * BORDER_PX else 0.0
    right = float(mask[:, -BORDER_PX:].mean()) if w > 2 * BORDER_PX else 0.0
    sig["border_ink"] = {"top": round(top, 3), "bottom": round(bottom, 3),
                         "left": round(left, 3), "right": round(right, 3)}
    clipped = [side for side, v in sig["border_ink"].items() if v > BORDER_INK_FRACTION]
    if clipped:
        return ImageQualityResult("CLIPPED", sig, True, "expand_crop",
                                  f"ink runs off the {', '.join(clipped)} edge(s) — content is cut off")

    skew = _skew_degrees(mask)
    sig["skew_degrees"] = None if skew is None else round(skew, 1)
    if skew is not None and SKEW_DEGREES < skew < 180 - SKEW_DEGREES:
        return ImageQualityResult("EXTREME_SKEW", sig, True, "rerender",
                                  f"writing is tilted {skew:.0f}° from horizontal")
    return ImageQualityResult("OK", sig, False, "proceed", "")


def should_call_ocr(result: ImageQualityResult) -> bool:
    """False when a model call cannot possibly help (missing/blank crop)."""
    return result.recommended_action not in ("skip_ocr",) and result.status not in ("BLANK", "INVALID")


def triage_with_recovery(png: bytes, *, rerender: Callable[[], bytes] | None = None,
                         expand_crop: Callable[[], bytes] | None = None,
                         expected_size: tuple[int, int] | None = None,
                         expect_content: bool = True,
                         max_attempts: int = 3) -> tuple[bytes, ImageQualityResult, list[dict]]:
    """Try the cheap deterministic recoveries before anything escalates.

    ``rerender``/``expand_crop`` are injected by the caller (the existing page
    rendering / crop geometry code); this module never renders anything itself.
    Returns the best image found, its verdict, and the attempt trail.
    """
    trail: list[dict] = []
    current = png
    result = triage_crop(current, expected_size=expected_size, expect_content=expect_content)
    trail.append({"attempt": 0, "action": "initial", "status": result.status})
    tried: set[str] = set()
    for n in range(1, max_attempts):
        if result.ok or not result.recoverable:
            break
        action = result.recommended_action
        fn = {"expand_crop": expand_crop, "rerender": rerender}.get(action)
        if fn is None or action in tried:
            break
        tried.add(action)
        try:
            candidate = fn()
        except Exception as e:  # noqa: BLE001 — a failed recovery is just no recovery
            trail.append({"attempt": n, "action": action, "status": "recovery_failed",
                          "error": type(e).__name__})
            break
        cand_result = triage_crop(candidate, expected_size=None, expect_content=expect_content)
        trail.append({"attempt": n, "action": action, "status": cand_result.status})
        if cand_result.ok or _rank(cand_result) < _rank(result):
            current, result = candidate, cand_result
    if not result.ok and not result.recoverable:
        result.recommended_action = "skip_ocr" if result.status == "BLANK" else "escalate"
    elif not result.ok:
        result.recommended_action = "escalate"
    return current, result, trail


def _rank(r: ImageQualityResult) -> int:
    """Lower is better. Used only to keep the best of two deterministic renders."""
    return {"OK": 0, "EXTREME_SKEW": 1, "LOW_CONTRAST": 2, "CLIPPED": 3,
            "SUSPICIOUS_CROP": 4, "BLANK": 5, "INVALID": 6}.get(r.status, 9)
