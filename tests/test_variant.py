"""Marker-based variant detection and per-variant question alignment.

The exam family prints three variants identified by a cover-page flower.
Requirements under test (see docs/variants.md):
- each catalogued flower selects its mapped variant;
- the variant is NEVER chosen from student answers or by score maximisation;
- an unclear/missing flower produces an uncertain decision routed to review;
- different flowers grade against different key columns;
- cached/resumed artefacts of one variant are not reused for another;
- instructor ink on the cover must not enter the decision;
- alignment maps printed numbering to key numbering and is validated
  deterministically, with a flagged identity fallback.
"""

import argparse
import json

import pytest

from autograder.backends import BackendConfig
from autograder.backends.mock import MockBackend
from autograder.cli import _fingerprints, run_grade_pipeline
from autograder.config import GraderConfig
from autograder.grade import grade_exam
from autograder.key_parser import save_answer_key
from autograder.schema import (
    ExamExtraction,
    KeyQuestion,
    KeySubItem,
    QuestionAlignmentEntry,
    QuestionExtraction,
    SubItemExtraction,
    VariantAlignment,
    VariantDetection,
)
from autograder.variant import (
    decide_version,
    detect_variant,
    identity_alignment,
    load_variant_config,
    printed_view,
    remap_extraction,
    validate_alignment,
)
from tests.conftest import make_pdf
from tests.test_grade import make_key


CFG = {
    "marker_kind": "flower",
    "marker_page": 1,
    "markers": {
        "variant_symbol_a1": {
            "variant": "A1",
            "aliases": ["four_petal_clover"],
            "description": "four petals in an X",
        },
        "variant_symbol_a2": {
            "variant": "A2",
            "aliases": ["five_petal_star"],
            "description": "five pointed petals",
        },
        "variant_symbol_a3": {
            "variant": "A3",
            "aliases": ["many_petal_daisy"],
            "description": "many small petals",
        },
    },
    "mapping_source": {"derived": "unit-test mapping"},
    "_path": "unit-test",
}


def detection(marker, confident=True, seen="a flower"):
    return VariantDetection(
        marker_seen=seen,
        matched_marker=marker,
        confident=confident,
        page_region="bottom third",
    )


# --------------------------------------------------------------------------
# marker -> variant decision
# --------------------------------------------------------------------------


def test_each_flower_selects_its_variant():
    key = make_key()
    for marker, variant in [
        ("variant_symbol_a1", "A1"),
        ("variant_symbol_a2", "A2"),
        ("variant_symbol_a3", "A3"),
    ]:
        decision, record = decide_version(detection(marker), CFG, key)
        assert decision.version == variant
        assert not decision.uncertain
        assert record["matched_marker"] == marker
        assert record["mapping_source"] == "unit-test mapping"


def test_alias_names_resolve_to_canonical_ids():
    """The model may echo a human-readable alias; the decision resolves it
    to the canonical variant_symbol_* id and records both."""
    key = make_key()
    decision, record = decide_version(detection("five_petal_star"), CFG, key)
    assert decision.version == "A2"
    assert not decision.uncertain
    assert record["matched_marker"] == "variant_symbol_a2"
    assert record["marker_reported"] == "five_petal_star"


def test_unclear_flower_is_uncertain_and_routed_to_review():
    key = make_key()
    for det in [
        detection(None, seen="smudged symbol"),
        detection("variant_symbol_a1", confident=False),
        detection("unknown_rose"),
    ]:
        decision, record = decide_version(det, CFG, key)
        assert decision.uncertain, det
        assert "review" in decision.description
        # The fallback is the alphabetically first variant — a documented
        # deterministic choice, never derived from answers or scores.
        assert decision.version == "A1"
        assert record.get("fallback_variant") == "A1"

    # The uncertain decision must surface as a review item in grading.
    extraction = ExamExtraction(
        questions=[
            QuestionExtraction(
                question_id=q.id,
                source_pages=[1],
                authoritative_source="test",
                sub_items=[
                    SubItemExtraction(
                        sub_item_id=s.id,
                        status="unanswered",
                        interpretation_rationale="empty",
                        confidence=1.0,
                    )
                    for s in q.sub_items
                ],
            )
            for q in key.questions
        ]
    )
    decision, _ = decide_version(detection(None), CFG, key)
    result = grade_exam(
        key, extraction, {}, decision, GraderConfig(),
        exam_file="e.pdf", graded_at="t", model="mock:m",
    )
    assert any("version" in r.reason.lower() for r in result.needs_human_review)


def test_variant_never_chosen_by_score_maximisation():
    """The decision function receives NO extraction/answers at all, and a
    rigged extraction that would score perfectly under A3 still grades under
    the flower's A1 key column."""
    from autograder.schema import ExplanationEvaluation

    key = make_key()
    q1 = key.question("1")
    # Answers 100% correct under A3, mostly wrong under A1 (the fixture's
    # matching question has strongly version-dependent letters).
    rigged = ExamExtraction(
        questions=[
            QuestionExtraction(
                question_id="1",
                source_pages=[11],
                authoritative_source="answer sheet",
                sub_items=[
                    SubItemExtraction(
                        sub_item_id=s.id,
                        status="answered",
                        final_answer=s.correct_by_version["A3"][0],
                        answer_origin="answer_sheet",
                        explanation_transcription="נימוק כלשהו",
                        explanation_legibility="full",
                        interpretation_rationale="rigged",
                        confidence=1.0,
                    )
                    for s in q1.sub_items
                ],
            ),
            QuestionExtraction(
                question_id="3",
                source_pages=[13],
                authoritative_source="answer table",
                sub_items=[
                    SubItemExtraction(
                        sub_item_id=s.id,
                        status="unanswered",
                        interpretation_rationale="empty",
                        confidence=1.0,
                    )
                    for s in key.question("3").sub_items
                ],
            ),
        ]
    )
    judgements = {
        "1": {
            s.id: ExplanationEvaluation(sub_item_id=s.id, verdict="valid", reasoning="ok")
            for s in q1.sub_items
        }
    }
    decision, _ = decide_version(detection("variant_symbol_a1"), CFG, key)
    assert decision.version == "A1"
    result = grade_exam(
        key, rigged, judgements, decision, GraderConfig(),
        exam_file="e.pdf", graded_at="t", model="mock:m",
    )
    q1_result = next(q for q in result.questions if q.question_id == "1")
    item4 = next(s for s in q1_result.sub_results if s.sub_item_id == "4")
    # A1's key column applies (H), not A3's (E) that the rig would prefer.
    assert item4.accepted_answers == ["H"]
    assert item4.student_answer == "E" and item4.selection_correct is False
    # A score-maximising selector would have scored 32/32 under A3.
    assert q1_result.points_awarded < q1_result.points_max / 2


def test_detection_call_sees_only_cover_and_catalogue():
    backend = MockBackend(
        config=BackendConfig(backend="mock", model="m"),
        responder=lambda model, system, blocks: detection("variant_symbol_a2"),
    )
    from autograder.ingest import PageImage

    pages = [
        PageImage(page_number=1, png_bytes=b"cover", width=5, height=5, text=""),
        PageImage(page_number=2, png_bytes=b"quest", width=5, height=5, text=""),
    ]
    detect_variant(backend, pages, CFG)
    call = backend.calls[0]
    images = [b for b in call.content_blocks if b.get("type") == "image"]
    assert len(images) == 1, "exactly the cover page image, nothing else"
    text = call.all_text()
    assert "five pointed petals" in text
    assert "answer" not in text.lower().replace("answers or answer keys", ""), (
        "no answer material may reach variant detection"
    )
    assert "IGNORE all ink added by hand" in call.system


# --------------------------------------------------------------------------
# alignment
# --------------------------------------------------------------------------


def _shuffled_entry() -> QuestionAlignmentEntry:
    # Variant prints key item 16 at position 20, 18 at 16, 20 at 18 (cycle).
    mapping = {str(i): str(i) for i in range(1, 21)}
    mapping["20"], mapping["16"], mapping["18"] = "16", "18", "20"
    return QuestionAlignmentEntry(question_id="3", printed_to_key=mapping)


def test_printed_view_and_remap_roundtrip():
    key = make_key()
    q3 = key.question("3")
    entry = _shuffled_entry()
    view = printed_view(q3, entry)
    # The view's printed item 20 carries the key item 16's prompt.
    v20 = next(s for s in view.sub_items if s.id == "20")
    assert v20.prompt == next(s for s in q3.sub_items if s.id == "16").prompt

    qx = QuestionExtraction(
        question_id="3",
        source_pages=[13],
        authoritative_source="table",
        sub_items=[
            SubItemExtraction(
                sub_item_id="20",
                status="answered",
                final_answer="C",
                answer_origin="answer_sheet",
                interpretation_rationale="row 20 read",
                confidence=1.0,
            )
        ],
    )
    remap_extraction(qx, entry)
    assert qx.sub_items[0].sub_item_id == "16", "printed row 20 scores as key item 16"
    assert "printed #20" in qx.sub_items[0].source_region


def test_validate_alignment_rejects_incomplete_and_duplicate_maps():
    key = make_key()
    good = identity_alignment(key, "A1")
    assert validate_alignment(key, good) == []

    entry = next(e for e in good.questions if e.question_id == "3")
    entry.printed_to_key["1"] = "2"  # now '2' twice, '1' never
    problems = validate_alignment(key, good)
    assert any("unmapped" in p for p in problems)
    assert any("duplicate" in p for p in problems)

    missing_q = VariantAlignment(variant="A2", questions=[])
    problems = validate_alignment(key, missing_q)
    assert any("no alignment entry" in p for p in problems)


# --------------------------------------------------------------------------
# end-to-end through the pipeline (mock backend) + fingerprints
# --------------------------------------------------------------------------


def _pipeline_fixtures(key):
    from tests.test_offline_pipeline import (
        _fixture_extraction,
        _fixture_judgement,
        _fixture_survey,
    )

    return {
        "ExamSurvey": _fixture_survey(),
        "QuestionExtraction": _fixture_extraction(),
        "ExplanationJudgement": _fixture_judgement(),
        "VariantDetection": detection("variant_symbol_a2"),
        "VariantAlignment": identity_alignment(key, "A2"),
    }


def test_pipeline_grades_with_detected_variant_and_records_it(tmp_path, no_network):
    key = make_key()
    key_path = tmp_path / "answer_key.json"
    save_answer_key(key, key_path)
    (tmp_path / "answer_key.variants.json").write_text(
        json.dumps({k: v for k, v in CFG.items() if not k.startswith("_")}),
        encoding="utf-8",
    )
    exam = make_pdf(tmp_path / "01_50.pdf")
    fixtures = _pipeline_fixtures(key)
    backend = MockBackend(
        config=BackendConfig(backend="mock", model="m"),
        responder=lambda model, system, blocks: fixtures[model.__name__].model_copy(deep=True),
    )
    ns = argparse.Namespace(
        key=str(key_path), rubric=None, resume=False, version="auto", exam=str(exam),
        key_cache_dir=str(tmp_path / "cache"), no_key_cache=False, variant_map=None,
    )
    result = run_grade_pipeline(ns, backend, tmp_path / "out", 800, exam_path=exam)
    assert result.detected_version == "A2"
    assert result.variant_detection["matched_marker"] == "variant_symbol_a2"
    assert result.variant_detection["selected_variant"] == "A2"
    assert result.variant_detection["uncertain"] is False
    assert result.variant_detection["page"] == 1
    assert result.variant_detection["mapping_from_authoritative_config"] is True
    assert "unit-test mapping" in result.variant_detection["mapping_source"]
    assert "flower" in result.version_detection
    # Grading really used A2's answer column: fixture extraction answers were
    # built for A1 (see test_offline_pipeline), so item 4 (H under A1, A under
    # A2) must be judged wrong here.
    q1 = next(q for q in result.questions if q.question_id == "1")
    item4 = next(s for s in q1.sub_results if s.sub_item_id == "4")
    assert item4.accepted_answers == ["A"], "A2's key column applies"


def test_two_exams_with_different_flowers_use_different_key_columns(tmp_path, no_network):
    key = make_key()
    key_path = tmp_path / "answer_key.json"
    save_answer_key(key, key_path)
    (tmp_path / "answer_key.variants.json").write_text(
        json.dumps({k: v for k, v in CFG.items() if not k.startswith("_")}),
        encoding="utf-8",
    )
    results = {}
    for marker, variant in [("variant_symbol_a1", "A1"), ("variant_symbol_a3", "A3")]:
        exam = make_pdf(tmp_path / f"exam_{variant}.pdf")
        fixtures = _pipeline_fixtures(key)
        fixtures["VariantDetection"] = detection(marker)
        fixtures["VariantAlignment"] = identity_alignment(key, variant)
        backend = MockBackend(
            config=BackendConfig(backend="mock", model="m"),
            responder=lambda model, system, blocks, fx=fixtures: fx[model.__name__].model_copy(deep=True),
        )
        ns = argparse.Namespace(
            key=str(key_path), rubric=None, resume=False, version="auto", exam=str(exam),
            key_cache_dir=str(tmp_path / "cache"), no_key_cache=False, variant_map=None,
        )
        results[variant] = run_grade_pipeline(
            ns, backend, tmp_path / f"out_{variant}", 800, exam_path=exam
        )
    assert results["A1"].detected_version == "A1"
    assert results["A3"].detected_version == "A3"
    # Same extraction fixture, different key columns -> different accepted
    # answers on the version-dependent item 4 of question 1 (H vs E).
    a1_item = next(
        s for q in results["A1"].questions if q.question_id == "1" for s in q.sub_results if s.sub_item_id == "4"
    )
    a3_item = next(
        s for q in results["A3"].questions if q.question_id == "1" for s in q.sub_results if s.sub_item_id == "4"
    )
    assert a1_item.accepted_answers != a3_item.accepted_answers


def test_variant_context_enters_fingerprints(tmp_path, no_network):
    """Changing the marker config or the pinned version invalidates resume:
    artefacts graded under one variant interpretation are never reused for
    another."""
    key = make_key()
    key_path = tmp_path / "answer_key.json"
    save_answer_key(key, key_path)
    exam = make_pdf(tmp_path / "01_50.pdf")
    backend = MockBackend(config=BackendConfig(backend="mock", model="m"))

    def fp(version="auto", cfg=None):
        cfg_path = tmp_path / "answer_key.variants.json"
        if cfg is None:
            if cfg_path.exists():
                cfg_path.unlink()
        else:
            cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
        ns = argparse.Namespace(
            key=str(key_path), rubric=None, exam=str(exam), version=version,
            variant_map=None,
        )
        return _fingerprints(ns, backend, 800, include_exam=True, exam_path=exam)

    base_cfg = {k: v for k, v in CFG.items() if not k.startswith("_")}
    fp_none = fp()
    fp_cfg = fp(cfg=base_cfg)
    assert fp_none["exam"] != fp_cfg["exam"], "adding a marker config invalidates"

    changed = json.loads(json.dumps(base_cfg))
    changed["markers"]["variant_symbol_a1"]["variant"] = "A2"
    fp_changed = fp(cfg=changed)
    assert fp_cfg["exam"] != fp_changed["exam"], "editing the mapping invalidates"

    fp_pin_a2 = fp(version="A2", cfg=base_cfg)
    fp_pin_a3 = fp(version="A3", cfg=base_cfg)
    assert fp_pin_a2["exam"] != fp_pin_a3["exam"], "different pinned variants never share artefacts"
    assert fp_pin_a2["exam"] != fp_cfg["exam"]
