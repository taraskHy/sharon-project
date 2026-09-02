"""Drift protection + canary contract (§8), grading-RAG policies (§16),
QuestionGradingPack auditability (§17). Mocked gateway; no provider calls,
no index build."""

from __future__ import annotations

import json

import pytest

from autograder.backends.mock import MockBackend
from autograder.canary import (AcceptanceRules, CanaryCase, CanaryStore, CanarySuite,
                               compare_to_baseline, run_suite)
from autograder.escalation import GradeResult, RubricItemGrade, escalate_grade
from autograder.gateway import ModelGateway
from autograder.gradingpack import (RAG_POLICIES, activate_rag, attach_rag, build_pack,
                                    rag_query, source_fingerprint)
from autograder.provenance import DecisionProvenance, drift_between, provenance_from_call
from autograder.cloudboundary import research_authorization
from tests.test_escalation import _gw
from tests.test_grade import make_key

TRANSCRIPTION = "התדרים הגבוהים נשמרים בתמונה לאחר הסינון"

CHUNKS = [{"chunk_id": "c0", "source": "lecture3.pdf", "page": 4, "similarity": 0.81,
           "text": "פירמידת לפלסיאן שומרת את התדרים הגבוהים"},
          {"chunk_id": "c1", "source": "lecture3.pdf", "page": 5, "similarity": 0.62,
           "text": "פירמידה גאוסיאנית מחליקה את התמונה"}]


def fake_retrieve(calls):
    def _r(course_id, query, top_k, embed_fn=None):
        calls.append({"course_id": course_id, "query": query, "top_k": top_k})
        return CHUNKS[:top_k]
    return _r


def pack(policy="RAG_ALWAYS", retrieve=None, course_id="CV", grading_policy="choice_and_explanation_independent"):
    key = make_key()
    p = build_pack(key, key.questions[0], grading_policy=grading_policy, course_id=course_id,
                   retrieve=retrieve, rag_top_k=2, rag_policy=policy,
                   rag_index_fingerprint="idx-abc")
    p.rubric_items = []
    p.rubric = ["identifies that high frequencies survive"]
    p.compute_hash()
    return p


# -------------------------------------------------------------- §8 provenance --


def _call_result():
    gw = ModelGateway.from_dict(
        {"models": {"grade_primary": {"backend": "openrouter", "model": "vendor/grade-1",
                                      "prompt_version": "grade-v7",
                                      "reasoning": {"effort": "low"}}}},
        backend_factory=lambda c: MockBackend(config=c, responder=lambda *a: GradeResult(score=2)),
        execution_mode="research",   # cloud-shaped grading route as a provenance vehicle
        research_auth=research_authorization("test:provenance", tasks=["grade_primary"],
                                             models=["vendor/grade-1"]))
    be = gw.backend_for("grade_primary")
    be.last_usage = {"provider": "SomeProvider", "model": "vendor/grade-1-2026-05",
                     "request_id": "req-9", "generation_id": "gen-9", "total_tokens": 400}
    return gw.call(task="grade_primary", system="SYSTEM PROMPT",
                   content_blocks=[{"type": "text", "text": "grade this"}],
                   output_model=GradeResult)


def test_provenance_records_what_was_asked_and_what_answered():
    res = _call_result()
    p = provenance_from_call(res, system="SYSTEM PROMPT",
                             content_blocks=[{"type": "text", "text": "grade this"}],
                             output_model=GradeResult, pack_hash="pk1", schema_version="grade-schema-v1")
    assert p.requested_model == "vendor/grade-1" and p.actual_model == "vendor/grade-1-2026-05"
    assert p.actual_provider == "SomeProvider" and p.generation_id == "gen-9"
    assert p.prompt_version == "grade-v7" and p.prompt_hash and p.schema_hash
    assert p.decoding["temperature"] == 0.0 and p.reasoning == {"effort": "low"}
    assert p.input_hashes[0].startswith("text:") and p.pack_hash == "pk1"


def test_provenance_never_carries_a_secret():
    import os
    os.environ["CANARY_SECRET_MODEL"] = "vendor/x"
    gw = ModelGateway.from_dict({"models": {"t": {"backend": "openrouter", "model": "${CANARY_SECRET_MODEL}",
                                                  "extra_generation": {"api_key": "sk-SECRET",
                                                                       "top_p": 1.0}}}},
                                backend_factory=lambda c: MockBackend(
                                    config=c, responder=lambda *a: GradeResult(score=1)),
                                execution_mode="research",
                                research_auth=research_authorization(
                                    "test:provenance-secret", tasks=["t"], models=["vendor/x"]))
    res = gw.call(task="t", system="s", content_blocks=[{"type": "text", "text": "x"}],
                  output_model=GradeResult)
    p = provenance_from_call(res, system="s", content_blocks=[], output_model=GradeResult)
    assert "SECRET" not in json.dumps(p.as_dict()) and p.decoding["top_p"] == 1.0


def test_configuration_fingerprint_changes_with_the_model_or_prompt():
    a = DecisionProvenance(task="grade_primary", requested_model="m1", backend="openrouter",
                           prompt_version="v1", prompt_hash="h1", schema_hash="s1")
    same = DecisionProvenance(task="grade_primary", requested_model="m1", backend="openrouter",
                              prompt_version="v1", prompt_hash="h1", schema_hash="s1")
    other = DecisionProvenance(task="grade_primary", requested_model="m2", backend="openrouter",
                               prompt_version="v1", prompt_hash="h1", schema_hash="s1")
    assert a.fingerprint() == same.fingerprint() != other.fingerprint()
    assert drift_between(a, other) == ["requested_model: 'm1' -> 'm2'"]


# ------------------------------------------------------------------ §8 canary --


def mc_suite():
    return CanarySuite(name="mc-frozen", kind="mc_resolver", cases=[
        CanaryCase(id="row1", kind="mc_resolver", input_ref="image:aaa",
                   expected={"selected": "B", "state": "single_mark"}),
        CanaryCase(id="row2", kind="mc_resolver", input_ref="image:bbb",
                   expected={"selected": "C", "state": "single_mark"}),
    ])


def test_canary_promotes_only_an_identical_candidate():
    suite = mc_suite()
    same = run_suite(suite, lambda c: dict(c.expected))
    v = compare_to_baseline(suite, same)
    assert v.promote and v.agreement == 1.0 and not v.regressions


def test_canary_blocks_a_regressing_candidate():
    suite = mc_suite()
    got = {"row1": {"selected": "B", "state": "single_mark"},
           "row2": {"selected": "D", "state": "single_mark"}}
    v = compare_to_baseline(suite, got)
    assert not v.promote and v.regressions == ["row2: selected 'C' -> 'D'"]
    assert any("regressions exceed" in r for r in v.reasons)


def test_grading_canary_respects_score_tolerance_and_new_uncertainty():
    suite = CanarySuite(name="grade-frozen", kind="grading",
                        acceptance=AcceptanceRules(max_score_delta=0.5),
                        cases=[CanaryCase(id="g1", kind="grading", input_ref="pack:1",
                                          expected={"score": 3.0, "rubric_items_met": ["R1"]})])
    assert compare_to_baseline(suite, {"g1": {"score": 3.4, "rubric_items_met": ["R1"]}}).promote
    assert not compare_to_baseline(suite, {"g1": {"score": 4.0, "rubric_items_met": ["R1"]}}).promote
    assert not compare_to_baseline(suite, {"g1": {"score": 3.0, "rubric_items_met": ["R1", "R2"]}}).promote
    assert not compare_to_baseline(suite, {"g1": {"score": 3.0, "rubric_items_met": ["R1"],
                                                  "uncertain": True}}).promote


def test_an_empty_suite_can_never_authorise_a_promotion():
    v = compare_to_baseline(CanarySuite(name="empty", kind="ocr"), {})
    assert not v.promote and "empty" in v.reasons[0]


def test_missing_candidate_results_count_as_regressions():
    v = compare_to_baseline(mc_suite(), {"row1": {"selected": "B", "state": "single_mark"}})
    assert not v.promote and "no result" in v.regressions[0]


def test_suites_persist_per_task_and_verdicts_are_logged(tmp_path):
    store = CanaryStore(tmp_path / "canary")
    store.save(mc_suite())
    store.save(CanarySuite(name="ocr-frozen", kind="ocr"))
    assert store.list_suites() == ["mc-frozen", "ocr-frozen"]
    loaded = store.load("mc-frozen")
    assert loaded.kind == "mc_resolver" and len(loaded.cases) == 2
    store.record_verdict("mc-frozen", compare_to_baseline(loaded, run_suite(loaded, lambda c: dict(c.expected))),
                         candidate_provenance={"requested_model": "vendor/new"})
    log = (tmp_path / "canary" / "verdicts.jsonl").read_text(encoding="utf-8")
    assert "vendor/new" in log and '"promote": true' in log


def test_unknown_canary_kinds_are_rejected():
    with pytest.raises(ValueError):
        CanarySuite(name="x", kind="vibes")


def test_no_canary_suite_is_shipped_populated():
    """The mechanism ships; the data does not. Populating a canary costs
    provider calls and is a separate, deliberate decision."""
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    assert not list(repo.glob("**/canary/*.json"))


# ----------------------------------------------------------- §16 RAG policies --


def test_rag_disabled_never_retrieves():
    calls = []
    p = pack("RAG_DISABLED", fake_retrieve(calls))
    assert p.rag_evidence == [] and calls == []
    gw, n = _gw({"grade_primary": [GradeResult(score=9)],       # invalid -> would escalate
                 "grade_escalate": [GradeResult(score=2)]})
    escalate_grade(pack=p, selected="F", transcription=TRANSCRIPTION, version="A1",
                   selection_correct=True, gateway=gw,
                   rag_attach=lambda pk: attach_rag(pk, course_id="CV", retrieve=fake_retrieve(calls)))
    assert calls == []                                           # policy wins over availability


def test_rag_always_embeds_context_at_build_time():
    calls = []
    p = pack("RAG_ALWAYS", fake_retrieve(calls))
    assert len(calls) == 1 and [e.chunk_id for e in p.rag_evidence] == ["c0", "c1"]
    assert "Course context" in p.to_grader_context()


def test_rag_on_uncertain_prepares_at_build_and_activates_only_when_unclean():
    calls = []
    p = pack("RAG_ON_UNCERTAIN", fake_retrieve(calls))
    # PREPARATION: exactly one free LOCAL retrieval at pack build. The grader
    # context stays empty — no provider tokens are spent by preparation.
    assert len(calls) == 1
    assert p.rag_evidence == [] and [e.chunk_id for e in p.rag_prepared] == ["c0", "c1"]
    assert "Course context" not in p.to_grader_context()
    clean = GradeResult(score=2, rubric_items=[
        RubricItemGrade(id="R1", met=True, student_evidence="התדרים הגבוהים נשמרים")])
    gw, n = _gw({"grade_primary": [clean]})
    d = escalate_grade(pack=p, selected="F", transcription=TRANSCRIPTION, version="A1",
                       selection_correct=True, gateway=gw, rag_attach=activate_rag)
    assert d.outcome == "auto" and len(calls) == 1               # easy case: no activation

    gw2, n2 = _gw({"grade_primary": [GradeResult(score=99), clean], "grade_escalate": [clean]})
    d2 = escalate_grade(pack=p, selected="F", transcription=TRANSCRIPTION, version="A1",
                        selection_correct=True, gateway=gw2, rag_attach=activate_rag)
    assert len(calls) == 1                                        # ACTIVATION reuses the cache
    assert d2.stage == "primary_rag" and d2.outcome == "auto"
    assert d2.signals.rag_used and n2["grade_escalate"] == 0     # RAG resolved it before escalating
    assert d2.rag_chunk_ids == ["c0", "c1"] and d2.rag_chars > 0


def test_rag_on_escalation_gives_context_only_to_the_escalation_grader():
    calls = []
    p = pack("RAG_ON_ESCALATION", fake_retrieve(calls))
    assert len(calls) == 1 and p.rag_evidence == []              # prepared, not active
    clean = GradeResult(score=2, rubric_items=[
        RubricItemGrade(id="R1", met=True, student_evidence="התדרים הגבוהים נשמרים")])
    unsure = GradeResult(score=2, uncertain=True)
    gw, n = _gw({"grade_primary": [unsure], "grade_escalate": [clean]})
    d = escalate_grade(pack=p, selected="F", transcription=TRANSCRIPTION, version="A1",
                       selection_correct=True, gateway=gw, rag_attach=activate_rag)
    assert len(calls) == 1 and n["grade_primary"] == 1 and n["grade_escalate"] == 1
    assert d.outcome == "auto" and d.stage == "escalated" and d.signals.rag_used


def test_optional_rag_unavailable_degrades_to_no_rag_grading():
    """RAG_ON_UNCERTAIN with no course/retriever: the retry is skipped, the
    flow continues to escalation, and nothing REVIEWs because of retrieval."""
    p = pack("RAG_ON_UNCERTAIN", None, course_id=None)
    assert p.rag_available is False and p.rag_prepared == []
    clean = GradeResult(score=2, rubric_items=[
        RubricItemGrade(id="R1", met=True, student_evidence="התדרים הגבוהים נשמרים")])
    unsure = GradeResult(score=2, uncertain=True)
    gw, n = _gw({"grade_primary": [unsure], "grade_escalate": [clean]})
    d = escalate_grade(pack=p, selected="F", transcription=TRANSCRIPTION, version="A1",
                       selection_correct=True, gateway=gw, rag_attach=activate_rag)
    assert d.outcome == "auto" and d.stage == "escalated"        # resolved WITHOUT RAG
    assert d.signals.rag_available is False and not d.signals.rag_used
    assert n["grade_primary"] == 1                               # no pointless RAG retry


def test_retrieval_is_never_steered_by_the_student_words():
    p = pack("RAG_ON_UNCERTAIN", None)
    q = rag_query(p)
    assert TRANSCRIPTION not in q and "Match operations" in q
    calls = []
    attach_rag(p, course_id="CV", retrieve=fake_retrieve(calls))
    assert TRANSCRIPTION not in calls[0]["query"]


def test_unknown_rag_policy_is_rejected():
    key = make_key()
    with pytest.raises(ValueError):
        build_pack(key, key.questions[0], grading_policy="choice_only", rag_policy="RAG_MAYBE")
    assert set(RAG_POLICIES) == {"RAG_DISABLED", "RAG_ALWAYS", "RAG_ON_UNCERTAIN", "RAG_ON_ESCALATION"}


def test_attaching_rag_produces_a_new_pack_identity():
    calls = []
    p = pack("RAG_ON_UNCERTAIN", None)
    before = p.hash
    enriched = attach_rag(p, course_id="CV", retrieve=fake_retrieve(calls))
    assert p.hash == before and p.rag_evidence == []             # the original is untouched
    assert enriched.hash != before and enriched.rag_chars > 0


# ------------------------------------------------------- §17 pack audit trail --


def test_pack_audit_records_every_source_it_was_built_from():
    calls = []
    p = pack("RAG_ALWAYS", fake_retrieve(calls))
    a = p.audit()
    assert a["question_id"] == "1" and a["pack_hash"] == p.hash and a["pack_version"] == "v2"
    assert a["question_text_hash"] and a["rubric_hash"] and a["solution_hash"]
    assert a["rag_policy"] == "RAG_ALWAYS" and a["rag_chunk_ids"] == ["c0", "c1"]
    assert a["rag_scores"] == [0.81, 0.62] and a["rag_sources"] == ["lecture3.pdf"]
    assert a["rag_index_fingerprint"] == "idx-abc" and a["rag_chars"] > 0
    assert a["rag_tokens_estimate"] == round(a["rag_chars"] / 4)
    assert a["grading_policy"] and a["evidence_policy"] and a["rubric_item_ids"]


def test_changing_the_rubric_or_solution_changes_the_pack_identity():
    p = pack("RAG_DISABLED", None)
    h0, rubric0, sol0 = p.hash, p.rubric_hash, p.solution_hash
    p.rubric = ["a different rubric line"]
    p.rubric_items = []
    p.compute_hash()
    assert p.hash != h0 and p.rubric_hash != rubric0
    p.official_solution = {"1": "a different official solution"}
    p.compute_hash()
    assert p.solution_hash != sol0


def test_changing_retrieval_configuration_invalidates_the_pack_store_fingerprint():
    pol = {"1": "choice_only"}
    base = source_fingerprint(b"KEY", "idx-a", pol, 2, 1200)
    assert base != source_fingerprint(b"KEY", "idx-b", pol, 2, 1200)        # course source
    assert base != source_fingerprint(b"KEY", "idx-a", pol, 3, 1200)        # top_k
    assert base != source_fingerprint(b"KEY", "idx-a", pol, 2, 900)         # budget
    assert base != source_fingerprint(b"KEY", "idx-a", {"1": "wrong_choice_zero"}, 2, 1200)
    assert base != source_fingerprint(b"KEY2", "idx-a", pol, 2, 1200)       # key/rubric
    assert base != source_fingerprint(b"KEY", "idx-a", pol, 2, 1200, rag_policy="RAG_ALWAYS")


def test_a_pack_is_reusable_across_students():
    """The pack is per QUESTION, not per student: nothing student-specific
    can enter it."""
    p = pack("RAG_ALWAYS", fake_retrieve([]))
    blob = json.dumps(p.to_json(), ensure_ascii=False)
    assert TRANSCRIPTION not in blob and "exam-" not in blob
    assert p.audit()["question_id"] == "1"


# ------------------------------------------- §4 default: retrieval is OFF ----


def test_the_default_rag_policy_is_disabled_and_retrieves_nothing():
    """An unspecified policy must not silently send course context: the
    benefit is unmeasured and it costs input tokens on every grading call."""
    from autograder.gradingpack import QuestionGradingPack

    calls = []
    key = make_key()
    p = build_pack(key, key.questions[0], grading_policy="choice_and_explanation_independent",
                   course_id="CV", retrieve=fake_retrieve(calls))     # no rag_policy given
    assert p.rag_policy == "RAG_DISABLED" and calls == []
    assert p.rag_evidence == [] and p.rag_chars == 0 and p.rag_config == {}
    assert "Course context" not in p.to_grader_context()
    # the dataclass default itself, independent of the builder
    assert QuestionGradingPack.__dataclass_fields__["rag_policy"].default == "RAG_DISABLED"


def test_no_course_context_reaches_the_grader_under_the_default(tmp_path):
    """End of the chain: nothing retrieved, nothing in the prompt."""
    from autograder.gradingpack import build_all_packs
    from autograder.reliability import ReliabilityConfig, run_reliability_judging
    from autograder.schema import ExamExtraction, QuestionExtraction, SubItemExtraction
    from tests.test_grading_modes import FakeRuntime

    key = make_key()
    ext = ExamExtraction(questions=[QuestionExtraction(
        question_id="1", source_pages=[1], authoritative_source="sheet",
        sub_items=[SubItemExtraction(sub_item_id="1", status="answered", final_answer="F",
                                     explanation_transcription=TRANSCRIPTION,
                                     explanation_legibility="full",
                                     interpretation_rationale="", confidence=1.0)])])
    rt = FakeRuntime(tmp_path, {"grade_primary": [GradeResult(score=4.0)]})
    calls = []
    run_reliability_judging(key=key, extraction=ext, version="A1",
                            config=ReliabilityConfig(mode="reliability"), gateway=rt.gateway,
                            packs=build_all_packs(key, {}), exam_id="exam-001",
                            rag_attach=lambda pk: attach_rag(pk, course_id="CV",
                                                             retrieve=fake_retrieve(calls)))
    assert calls == []                                  # the policy wins over availability
    prompt = rt.blocks["grade_primary"][0][0]["text"]
    assert "Course context" not in prompt and "לפלסיאן" not in prompt
