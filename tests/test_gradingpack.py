"""QuestionGradingPack + grading-side RAG — offline tests (injected retriever)."""

from __future__ import annotations

import json

from autograder.gradingpack import (PackStore, build_all_packs, build_pack,
                                    source_fingerprint)
from tests.test_grade import make_key


def fake_retrieve_factory(chunks):
    calls = []

    def retrieve(course_id, query, top_k, embed_fn=None):
        calls.append({"course_id": course_id, "query": query, "top_k": top_k})
        return chunks[:top_k]

    retrieve.calls = calls
    return retrieve


CHUNKS = [
    {"chunk_id": f"c{i}", "source": "notes.pdf", "page": i, "similarity": 0.9 - i * 0.1,
     "text": ("סעיף " + str(i) + " " + "מידע על פירמידות ותדרים " * 40)}
    for i in range(6)
]


def test_pack_content_and_hash_stability():
    key = make_key()
    q1 = key.questions[0]
    p1 = build_pack(key, q1, grading_policy="explanation_required_if_correct")
    p2 = build_pack(key, q1, grading_policy="explanation_required_if_correct")
    assert p1.question_id == "1" and p1.max_score == 32
    assert p1.correct_by_version["1"]["A1"] == ["F"]
    assert p1.official_solution["1"].startswith("reference reasoning")
    assert p1.hash and p1.hash == p2.hash            # deterministic
    assert "correct_by_version" not in p1.question_text  # answer-free question text
    ctx = p1.to_grader_context()
    assert "Question 1" in ctx and "Rubric" not in ctx  # no grading_notes in fixture -> no rubric block
    p3 = build_pack(key, q1, grading_policy="wrong_choice_zero")
    assert p3.hash != p1.hash                          # policy is part of identity


def test_rag_top_k_and_char_budget_respected():
    key = make_key()
    q1 = key.questions[0]
    r = fake_retrieve_factory(CHUNKS)
    p = build_pack(key, q1, grading_policy="choice_and_explanation_independent",
                   course_id="CV", retrieve=r, rag_top_k=2, rag_char_budget=500,
                   rag_policy="RAG_ALWAYS")            # retrieval is opt-in (default disabled)
    assert r.calls[0]["top_k"] == 2 and r.calls[0]["course_id"] == "CV"
    assert len(p.rag_evidence) <= 2
    assert sum(len(e.text) for e in p.rag_evidence) <= 500     # ellipsis counted
    assert p.rag_config["chars_used"] <= 500
    assert p.provenance["course_id"] == "CV" and p.rag_evidence[0].chunk_id == "c0"
    # rubric/solution are primary: they appear before the course context block
    ctx = p.to_grader_context()
    assert ctx.index("Official solution") < ctx.index("Course context")


def test_no_rag_text_reaches_ocr_repair_path():
    """Structural guarantee: the OCR-repair arm and the grading pack never
    import each other; RAG evidence lives only inside the grading pack."""
    import inspect
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    rag_ocr_src = (repo / "scripts" / "m2_rag_ocr.py").read_text(encoding="utf-8")
    assert "gradingpack" not in rag_ocr_src and "QuestionGradingPack" not in rag_ocr_src
    from autograder import gradingpack
    src = inspect.getsource(gradingpack)
    for forbidden in ("raw_text", "suggested_text", "transcription", "repair"):
        assert forbidden not in src   # the pack has no notion of student OCR text at all


def test_store_reuse_and_source_change_invalidates(tmp_path):
    key = make_key()
    r = fake_retrieve_factory(CHUNKS)
    policies = {q.id: "choice_and_explanation_independent" for q in key.questions}
    packs = build_all_packs(key, policies, course_id="CV", retrieve=r, rag_top_k=1,
                            rag_policy="RAG_ALWAYS")
    store = PackStore(tmp_path / "packs")
    fp = source_fingerprint(b"KEYBYTES", "idx-abc", policies, 1, 1200)
    store.save(packs, fp)
    loaded = store.load(fp)
    assert loaded is not None and loaded["1"].hash == packs["1"].hash
    assert loaded["1"].rag_evidence[0].chunk_id == "c0"
    # course index rebuilt -> different fingerprint -> packs must be rebuilt
    assert store.load(source_fingerprint(b"KEYBYTES", "idx-NEW", policies, 1, 1200)) is None
    # key changed
    assert store.load(source_fingerprint(b"KEYBYTES2", "idx-abc", policies, 1, 1200)) is None
    # policy changed
    assert store.load(source_fingerprint(b"KEYBYTES", "idx-abc", {**policies, "1": "wrong_choice_zero"}, 1, 1200)) is None
    # persisted JSON round-trips
    d = json.loads((tmp_path / "packs" / "q1.json").read_text(encoding="utf-8"))
    assert d["hash"] == packs["1"].hash and d["rag_config"]["top_k"] == 1
