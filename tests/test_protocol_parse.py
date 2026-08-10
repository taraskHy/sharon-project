"""Regression tests for the declared-envelope protocol parser.

Covers the live bug: Gemini/qwen outputs truncated at max tokens leaked
the {"transcription": ... wrapper into the scored transcription via the
parser fallback (found by the 2026-08-11 fidelity audit).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "mbr", Path(__file__).resolve().parents[1] / "scripts" / "m2_bench_run.py"
)
mbr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mbr)

parse = mbr.parse_declared_envelope


def test_complete_envelope_parses():
    text, ops = parse('{"transcription": "שלום עולם"}')
    assert text == "שלום עולם"
    assert "json_envelope_parse" in ops


def test_fenced_complete_envelope():
    text, ops = parse('```json\n{"transcription": "טקסט"}\n```')
    assert text == "טקסט"
    assert "code_fence_strip" in ops and "json_envelope_parse" in ops


def test_truncated_envelope_strips_wrapper():
    # the live bug: output cut before the closing quote/brace
    text, ops = parse('{"transcription": "ניתן לראות שמספר הפיקסלים בכל עוצמה')
    assert text == "ניתן לראות שמספר הפיקסלים בכל עוצמה"
    assert "truncated_envelope_prefix_strip" in ops


def test_fenced_truncated_envelope():
    text, ops = parse('```json\n{"transcription": "התדר הנמוך נעלם')
    assert text == "התדר הנמוך נעלם"
    assert "code_fence_strip" in ops and "truncated_envelope_prefix_strip" in ops


def test_trailing_quote_and_close_artifacts():
    text, _ = parse('{"transcription": "גם לא ישאר DC."')
    assert text == "גם לא ישאר DC."
    text2, _ = parse('{"transcription": "אבג"}')
    assert text2 == "אבג"


def test_dangling_escape_stripped():
    text, ops = parse('{"transcription": "בבי\\')
    assert text == "בבי"
    assert "dangling_escape_strip" in ops


def test_escapes_unescaped_cleanly():
    text, ops = parse('{"transcription": "שורה אחת\\nשורה שתיים עם \\"מרכאות\\"')
    assert text == 'שורה אחת\nשורה שתיים עם "מרכאות"'
    assert "json_string_unescape" in ops


def test_plain_text_returns_none():
    # no declared envelope -> not this parser's business (caller falls back)
    text, ops = parse("סתם טקסט חופשי בלי מעטפת")
    assert text is None
    assert ops == []


def test_never_invents_content():
    # unbalanced interior escape: falls back to prefix-stripped fragment,
    # never fabricates or drops payload characters beyond the artifacts
    raw = '{"transcription": "טקסט עם escape שבור \\x בסוף'
    text, ops = parse(raw)
    assert text is not None
    assert text.startswith("טקסט עם escape שבור")
    assert "no_unescape_applied" in ops
