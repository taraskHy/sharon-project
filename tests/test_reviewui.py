"""Review-UI back-end + settings summary — offline."""

from __future__ import annotations

import json

import pytest

from autograder.backends.mock import MockBackend
from autograder.gateway import ModelGateway
from autograder.reviewui import (ResolutionStore, build_review_items, settings_summary,
                                 test_connection as probe_connection)

RESULT = {
    "needs_human_review": [
        {"question_id": "*", "sub_item_id": "*", "reason": "exam version detection is uncertain: ..."},
        {"question_id": "3", "sub_item_id": "7", "reason": "multiple live marks; candidates for human review"},
        {"question_id": "1", "sub_item_id": "2", "reason": "explanation could not be read reliably"},
        {"question_id": "1", "sub_item_id": "4", "reason": "grader uncertain after escalation"},
    ],
    "questions": [{"question_id": "1", "sub_items": [
        {"sub_item_id": "4", "student_answer": "F", "points_total": 2, "points_max": 4,
         "reason": "partial rubric", "explanation_transcription": "טקסט"}]}],
}
EXTRACTION = {"questions": [
    {"question_id": "3", "sub_items": [{"sub_item_id": "7", "status": "ambiguous", "candidate_answers": ["B", "D"]}]},
    {"question_id": "1", "sub_items": [{"sub_item_id": "2", "status": "answered",
                                        "explanation_transcription": "קריאה ראשית"}]},
]}


def test_review_items_shapes_and_evidence():
    items = build_review_items("exam-001", RESULT, EXTRACTION,
                               chain_traces={("3", "7"): {"local": "B", "cloud": "D"},
                                             ("ocr", "1", "2"): "קריאה משנית"},
                               crops={("3", "7"): b"\x89PNGmc"})
    kinds = {(i.kind, i.question_id, i.sub_item_id): i for i in items}
    v = kinds[("variant", "*", "*")]
    assert v.apply_to_all_eligible
    mc = kinds[("mc", "3", "7")]
    assert mc.deterministic_candidate == ["B", "D"] and mc.local_candidate == "B" and mc.cloud_candidate == "D"
    assert mc.crop_png_b64 and mc.options[:2] == ["B", "D"]
    ocr = kinds[("ocr", "1", "2")]
    assert ocr.primary_transcription == "קריאה ראשית" and ocr.secondary_transcription == "קריאה משנית"
    g = kinds[("grading", "1", "4")]
    assert g.selected_option == "F" and g.proposed_score == 2 and g.max_score == 4
    assert not g.apply_to_all_eligible


def test_resolution_store_and_apply_to_all_rules(tmp_path):
    job = tmp_path / "job"
    for e in ("exam-001", "exam-002"):
        (job / "exams" / e).mkdir(parents=True)
    rs = ResolutionStore(job / "exams" / "exam-001")
    rs.resolve("3", "7", decision="B")
    assert rs.load()["3:7"]["decision"] == "B"
    assert not (job / "exams" / "exam-001" / "result.json").exists()   # never edits result.json
    n = rs.apply_to_all(job, "variant", "*", "*", decision="heart")
    assert n == 2 and ResolutionStore(job / "exams" / "exam-002").load()["*:*"]["decision"] == "heart"
    with pytest.raises(ValueError):
        rs.apply_to_all(job, "grading", "1", "4", decision="accept proposed")


def test_settings_summary_never_contains_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-SECRET")
    gw = ModelGateway.from_dict({"models": {"grade_primary": {"backend": "mock", "model": "m"}}},
                                backend_factory=lambda c: MockBackend(config=c))
    s = settings_summary(gateway=gw, openrouter_key_present=True)
    assert s["key_present"] is True and "SECRET" not in json.dumps(s)
    assert s["tasks"]["grade_primary"]["model"] == "m"
    p = probe_connection(gw)
    assert p["ok"] is True and "SECRET" not in json.dumps(p)
