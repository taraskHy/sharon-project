"""Deterministic pre-OCR image triage (§4). Synthetic crops only — no
batch processing, no model, no network."""

from __future__ import annotations

import numpy as np
import pytest

from autograder.imagequality import (ImageQualityResult, should_call_ocr, triage_crop,
                                     triage_with_recovery)
from autograder.tablecrop import _encode_png_gray


def png(arr: np.ndarray) -> bytes:
    return _encode_png_gray(np.clip(arr, 0, 255).astype(np.uint8))


def blank(w=160, h=48, value=255) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


def handwriting(w=240, h=60, strokes=14, thickness=3, value=40) -> np.ndarray:
    """A synthetic 'handwritten line': short dark strokes on white paper."""
    a = blank(w, h)
    rng = np.random.default_rng(7)
    for i in range(strokes):
        x = 8 + i * ((w - 20) // strokes)
        y = 14 + int(rng.integers(0, 8))
        a[y:y + 20, x:x + thickness] = value
        a[y + 10:y + 10 + thickness, x:x + 8] = value
    return a


# ---------------------------------------------------------------- valid ------


def test_legitimate_handwriting_passes():
    r = triage_crop(png(handwriting()))
    assert r.status == "OK" and r.ok and r.recommended_action == "proceed"
    assert should_call_ocr(r)


def test_messy_faint_but_readable_handwriting_is_not_rejected():
    """Conservative by design: gray ink on off-white paper still passes."""
    a = handwriting(value=95)
    a[a == 255] = 238
    r = triage_crop(png(a))
    assert r.status == "OK", r.as_dict()


# ---------------------------------------------------------------- blank ------


def test_blank_crop_is_detected_and_skips_ocr():
    r = triage_crop(png(blank()))
    assert r.status == "BLANK" and not r.recoverable
    assert r.recommended_action == "skip_ocr" and not should_call_ocr(r)


def test_suspiciously_empty_crop_with_a_speck_is_still_blank():
    a = blank()
    a[10:12, 10:12] = 0          # a dust speck: ~0.05% ink
    assert triage_crop(png(a)).status == "BLANK"


# --------------------------------------------------------- contrast/clip -----


def test_low_contrast_crop_is_flagged_recoverable():
    a = handwriting(value=150)
    a[a == 255] = 170            # ink and paper nearly the same gray
    r = triage_crop(png(a))
    assert r.status == "LOW_CONTRAST" and r.recoverable and r.recommended_action == "rerender"


def test_clipped_content_is_detected_at_the_boundary():
    a = handwriting()
    a[:2, :] = 20                # ink running along the whole top edge
    r = triage_crop(png(a))
    assert r.status == "CLIPPED" and r.signals["border_ink"]["top"] > 0.35
    assert r.recommended_action == "expand_crop"


def test_flooded_and_tiny_crops_are_suspicious():
    assert triage_crop(png(blank(value=10))).status == "SUSPICIOUS_CROP"
    assert triage_crop(png(handwriting(w=200, h=2))).status == "SUSPICIOUS_CROP"
    wide = triage_crop(png(blank(w=1000, h=8)))
    assert wide.status == "SUSPICIOUS_CROP"


def test_crop_size_drift_against_the_template_is_suspicious():
    r = triage_crop(png(handwriting(w=240, h=60)), expected_size=(600, 60))
    assert r.status == "SUSPICIOUS_CROP" and "template expects" in r.detail


def test_invalid_bytes_never_raise():
    for bad in (b"", b"not-a-png", None):
        r = triage_crop(bad)
        assert r.status == "INVALID" and not should_call_ocr(r)


def test_extreme_skew_is_detected():
    a = blank(200, 200)
    for t in range(150):         # a stroke at ~45 degrees
        a[20 + t, 20 + t:23 + t] = 0
    r = triage_crop(png(a))
    assert r.status == "EXTREME_SKEW"


# ------------------------------------------------------------- recovery ------


def test_deterministic_recovery_runs_before_escalation():
    bad = png(handwriting()[:, :].copy())
    a = handwriting()
    a[:2, :] = 20
    clipped = png(a)
    calls = {"expand": 0}

    def expand():
        calls["expand"] += 1
        return bad                     # the wider crop is clean

    img, result, trail = triage_with_recovery(clipped, expand_crop=expand)
    assert calls["expand"] == 1 and result.status == "OK" and img == bad
    assert [t["action"] for t in trail] == ["initial", "expand_crop"]


def test_recovery_that_does_not_help_escalates_once():
    a = handwriting()
    a[:2, :] = 20
    clipped = png(a)
    img, result, trail = triage_with_recovery(clipped, expand_crop=lambda: clipped)
    assert result.status == "CLIPPED" and result.recommended_action == "escalate"
    assert len(trail) == 2                      # one attempt, then stop (no loop)


def test_blank_crop_never_triggers_a_recovery_or_an_ocr_call():
    calls = {"n": 0}

    def rerender():
        calls["n"] += 1
        return png(blank())

    img, result, trail = triage_with_recovery(png(blank()), rerender=rerender)
    assert calls["n"] == 0 and result.status == "BLANK"
    assert result.recommended_action == "skip_ocr" and not should_call_ocr(result)


def test_missing_crop_is_recovered_by_rerender_without_any_model_call():
    good = png(handwriting())
    img, result, trail = triage_with_recovery(b"", rerender=lambda: good)
    assert result.status == "OK" and img == good
