"""Grading-side RAG at the live reliability seam: policy payload boundaries,
OCR safety, evidence separation, privacy, and accounting. Mocked gateway and
injected retriever; no provider, no network, no index build.
"""

from __future__ import annotations

import json

from autograder.escalation import GradeResult, OCRVerifyResult, RubricItemGrade
from autograder.gradingpack import activate_rag, build_all_packs, rag_query
from autograder.privacy import scan_for_identifiers
from autograder.reliability import ReliabilityConfig, run_reliability_judging
from autograder.schema import ExamExtraction, QuestionExtraction, SubItemExtraction
from tests.test_grade import make_key
from tests.test_grading_modes import FakeRuntime

TRANSCRIPTION = "התדרים הגבוהים נשמרים בתמונה לאחר הסינון"
CHUNK_TEXT = "פירמידת לפלסיאן שומרת את התדרים הגבוהים בכל רמה"
CHUNKS = [{"chunk_id": "c0", "source": "lecture3.pdf", "page": 4, "similarity": 0.81,
           "text": CHUNK_TEXT},
          {"chunk_id": "c1", "source": "lecture3.pdf", "page": 5, "similarity": 0.62,
           "text": "פירמידה גאוסיאנית מחליקה את התמונה"}]


def _retrieve(course_id, query, top_k, embed_fn=None):
    return CHUNKS[:top_k]


def _ext(transcription=TRANSCRIPTION, legibility="full"):
    return ExamExtraction(questions=[QuestionExtraction(
        question_id="1", source_pages=[1], authoritative_source="sheet",
        sub_items=[SubItemExtraction(sub_item_id="1", status="answered", final_answer="F",
                                     explanation_transcription=transcription,
                                     explanation_legibility=legibility,
                                     interpretation_rationale="", confidence=1.0)])])


def _packs(rag_policy, retrieve=_retrieve):
    key = make_key()
    key.questions[0].grading_notes = "identifies that high frequencies survive"
    return key, build_all_packs(key, {}, course_id="CV", retrieve=retrieve,
                                rag_policy=rag_policy, rag_index_fingerprint="idx-a")


def _judge(tmp_path, key, packs, responses, extraction=None):
    rt = FakeRuntime(tmp_path, responses)
    run = run_reliability_judging(key=key, extraction=extraction or _ext(), version="A1",
                                  config=ReliabilityConfig(mode="reliability"),
                                  gateway=rt.gateway, packs=packs, exam_id="exam-001",
                                  rag_attach=activate_rag)
    return run, rt


def _texts(rt, task, call=0):
    return "\n".join(b.get("text", "") for b in rt.blocks[task][call]
                     if isinstance(b, dict))


CLEAN = GradeResult(score=2, rubric_items=[
    RubricItemGrade(id="R1", met=True, student_evidence="התדרים הגבוהים נשמרים")])


def test_rag_disabled_payload_carries_no_course_context(tmp_path):
    key, packs = _packs("RAG_DISABLED")
    run, rt = _judge(tmp_path, key, packs, {"grade_primary": [CLEAN]})
    assert rt.calls == {"grade_primary": 1}
    assert "Course context" not in _texts(rt, "grade_primary")
    assert CHUNK_TEXT not in _texts(rt, "grade_primary")
    rec = run.records[0]
    assert rec.rag["policy"] == "RAG_DISABLED" and rec.rag["used"] is False


def test_rag_always_payload_includes_the_prepared_context(tmp_path):
    key, packs = _packs("RAG_ALWAYS")
    run, rt = _judge(tmp_path, key, packs, {"grade_primary": [CLEAN]})
    body = _texts(rt, "grade_primary")
    assert "Course context" in body and CHUNK_TEXT in body
    rec = run.records[0]
    assert rec.rag["used"] is True and rec.rag["chunk_ids"] == ["c0", "c1"]
    assert rec.rag["chars"] > 0


def test_rag_on_uncertain_clean_primary_stays_rag_free(tmp_path):
    key, packs = _packs("RAG_ON_UNCERTAIN")
    run, rt = _judge(tmp_path, key, packs, {"grade_primary": [CLEAN]})
    assert rt.calls == {"grade_primary": 1}
    assert "Course context" not in _texts(rt, "grade_primary")
    assert run.records[0].rag["used"] is False
    assert run.decisions[0].final_state == "AUTO"


def test_rag_on_uncertain_activates_cached_context_on_unclean_primary(tmp_path):
    key, packs = _packs("RAG_ON_UNCERTAIN")
    unsure = GradeResult(score=2, uncertain=True)
    run, rt = _judge(tmp_path, key, packs,
                     {"grade_primary": [unsure, CLEAN], "grade_escalate": [CLEAN]})
    assert rt.calls["grade_primary"] == 2                      # primary + RAG retry
    assert "Course context" not in _texts(rt, "grade_primary", 0)
    retry = _texts(rt, "grade_primary", 1)
    assert "Course context" in retry and CHUNK_TEXT in retry
    rec = run.records[0]
    assert rec.rag["used"] is True and rec.rag["chunk_ids"] == ["c0", "c1"]
    assert any(s.stage == "grading_rag" and s.status == "executed" for s in rec.stages)
    assert run.decisions[0].final_state == "AUTO"


def test_rag_on_escalation_gives_context_to_the_escalation_stage_only(tmp_path):
    key, packs = _packs("RAG_ON_ESCALATION")
    unsure = GradeResult(score=2, uncertain=True)
    run, rt = _judge(tmp_path, key, packs,
                     {"grade_primary": [unsure], "grade_escalate": [CLEAN]})
    assert "Course context" not in _texts(rt, "grade_primary")
    esc = _texts(rt, "grade_escalate")
    assert "Course context" in esc and CHUNK_TEXT in esc
    assert run.decisions[0].final_state == "AUTO"


def test_unresolved_ocr_never_reaches_grading_rag(tmp_path):
    """OCR_UNRESOLVED routes to OCR/REVIEW logic; a bad reading is never sent
    to grading RAG so the grader could 'reconstruct' course terminology."""
    key, packs = _packs("RAG_ALWAYS")
    ext = _ext(transcription=None, legibility="illegible")
    run, rt = _judge(tmp_path, key, packs,
                     {"grade_primary": [CLEAN],
                      "ocr_verify": [OCRVerifyResult(verdict="supported")]},
                     extraction=ext)
    assert rt.total_calls == 0                                # no grader, no RAG, no verify
    assert run.decisions[0].final_state == "REVIEW"
    assert run.decisions[0].reason_code == "OCR_UNRESOLVED"


def test_rag_text_cannot_serve_as_student_evidence(tmp_path):
    """A span that exists only in the COURSE CHUNK is fabricated evidence:
    course material can prove correctness, never what the student wrote."""
    key, packs = _packs("RAG_ALWAYS")
    cheating = GradeResult(score=2, rubric_items=[
        RubricItemGrade(id="R1", met=True,
                        student_evidence="פירמידת לפלסיאן שומרת את התדרים")])  # from c0, not the student
    run, rt = _judge(tmp_path, key, packs,
                     {"grade_primary": [cheating], "grade_escalate": [cheating]})
    rec = run.records[0]
    assert rec.evidence["fabricated"] >= 1
    assert run.decisions[0].final_state == "REVIEW"
    assert run.decisions[0].reason_code in ("EVIDENCE_INVALID", "GRADE_DISAGREEMENT")


def test_transcription_is_immutable_through_the_rag_route(tmp_path):
    key, packs = _packs("RAG_ALWAYS")
    ext = _ext()
    before = ext.questions[0].sub_items[0].explanation_transcription
    run, rt = _judge(tmp_path, key, packs, {"grade_primary": [CLEAN]}, extraction=ext)
    assert ext.questions[0].sub_items[0].explanation_transcription == before
    # and the grader was told exactly the frozen text
    assert before in _texts(rt, "grade_primary")


def test_retrieval_query_is_question_level_and_identity_free(tmp_path):
    key, packs = _packs("RAG_ON_UNCERTAIN")
    q = rag_query(packs["1"])
    fake_identity = ["Dana Cohen", "205551234", "dana@uni.ac.il", "C:\\scans\\dana.pdf"]
    assert scan_for_identifiers({"query": q}, fake_identity) == []
    assert TRANSCRIPTION not in q                             # never student words
    assert "Match operations" in q                            # question text
    assert "reference reasoning" in q                         # official solution included


def test_grader_payload_is_identity_free(tmp_path):
    key, packs = _packs("RAG_ALWAYS")
    run, rt = _judge(tmp_path, key, packs, {"grade_primary": [CLEAN]})
    payload = rt.blocks["grade_primary"][0]
    fake_identity = ["Dana Cohen", "205551234", "dana@uni.ac.il"]
    assert scan_for_identifiers(payload, fake_identity) == []


def test_accounting_separates_rag_overhead_and_early_exit_savings(tmp_path):
    key, packs = _packs("RAG_ALWAYS")
    run, rt = _judge(tmp_path, key, packs, {"grade_primary": [CLEAN]})
    acc = run.accounting()
    assert acc["items_with_rag"] == 1 and acc["rag_chars_total"] > 0
    assert acc["rag_tokens_est_total"] == round(acc["rag_chars_total"] / 4)
    # ledger rows carry the numbers-only RAG meta for token attribution
    rows = [json.loads(l) for l in
            (tmp_path / "usage.jsonl").read_text(encoding="utf-8").splitlines()]
    grade_rows = [r for r in rows if r.get("task") == "grade_primary"]
    assert grade_rows and grade_rows[0]["rag_chars"] > 0
    assert grade_rows[0]["rag_policy"] == "RAG_ALWAYS" and grade_rows[0]["pack_hash"]

    # early-exit savings: a wrong_choice_zero item skips OCR/RAG/grader/escalation
    wrong_key = make_key()
    for q in wrong_key.questions:
        for s in q.sub_items:
            s.correct_by_version = {v: ["Z"] for v in wrong_key.versions}
    rt2 = FakeRuntime(tmp_path / "z2", {"grade_primary": [CLEAN]})
    run2 = run_reliability_judging(
        key=wrong_key, extraction=_ext(), version="A1",
        config=ReliabilityConfig(mode="reliability"), gateway=rt2.gateway,
        packs=build_all_packs(wrong_key, {"1": "wrong_choice_zero"}),
        policies={"1": "wrong_choice_zero"}, exam_id="exam-001")
    assert rt2.total_calls == 0
    acc2 = run2.accounting()
    assert acc2["ocr_skipped_by_early_exit"] == 1
    assert acc2["rag_skipped_by_early_exit"] == 1
    assert acc2["grader_skipped_by_early_exit"] == 1
    assert acc2["escalation_skipped_by_early_exit"] == 1
