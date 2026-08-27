"""Production grading is LOCAL: routing, failure semantics, and token-saving.

Covers the §13 items about the grading side: OpenRouter-configured-for-OCR
still grades locally; a dead/malformed local grader parks the item for REVIEW
(typed LOCAL_GRADER_UNAVAILABLE) instead of calling any cloud; deterministic
early exits and the request cache keep provider calls at zero. No network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from autograder.backends import BackendConfig
from autograder.backends.mock import MockBackend
from autograder.escalation import GRADE_SYSTEM, GradeResult, escalate_grade
from autograder.gateway import ModelGateway
from autograder.gradingpack import QuestionGradingPack
from autograder.policies import MCResolution, decide_before_ocr
from autograder.prompts import EXPLANATION_OCR_SYSTEM
from autograder.requestcache import RequestCache
from autograder.schema import ExplanationTranscription
from autograder.usage import UsageLedger

LOCAL_URL = "http://localhost:11434/v1"


def _pack(**kw):
    d = dict(question_id="1", question_text="שאלה", question_type="multiple_choice",
             max_score=4.0, correct_by_version={}, rubric=["הסבר את הרעיון המרכזי"],
             scoring_rules=[], grading_policy="choice_plus_explanation",
             official_solution={}, evidence_policy="disabled")
    d.update(kw)
    return QuestionGradingPack(**d)


class _CountingGateway:
    """Real ModelGateway over mocks, counting constructed backends and calls
    per (task, effective destination)."""

    def __init__(self, tmp_path: Path, routes: dict, responders: dict):
        self.built: list[BackendConfig] = []
        self.calls: list[str] = []
        outer = self

        def factory(cfg):
            outer.built.append(cfg)

            def responder(model, system, blocks):
                outer.calls.append(cfg.model)
                fn = responders.get(cfg.model)
                if fn is None:
                    raise AssertionError(f"unexpected call to backend model {cfg.model}")
                return fn(system, blocks)
            return MockBackend(config=cfg, responder=responder)

        self.gw = ModelGateway.from_dict({"models": routes}, backend_factory=factory,
                                         cache=RequestCache(tmp_path / "cache"),
                                         ledger=UsageLedger(tmp_path / "usage.jsonl"))


def test_openrouter_configured_for_ocr_grading_still_runs_locally(tmp_path):
    """§13-7: an OpenRouter OCR credential/route changes nothing about where
    grading executes. The grading call reaches the LOCAL backend; the
    OpenRouter backend is never even constructed for it."""
    cg = _CountingGateway(
        tmp_path,
        routes={
            "ocr_primary": {"backend": "openrouter", "model": "vendor/ocr"},
            "grade_primary": {"backend": "ollama", "base_url": LOCAL_URL, "model": "local-grader"},
        },
        responders={"local-grader": lambda s, b: GradeResult(score=4.0)})
    d = escalate_grade(pack=_pack(), selected="A", transcription="הסבר נכון", version="default",
                       selection_correct=True, gateway=cg.gw)
    assert d.outcome == "auto" and d.result.score == 4.0
    assert cg.calls == ["local-grader"]
    assert all(c.backend != "openrouter" for c in cg.built), \
        "no OpenRouter backend may be constructed by a grading call"
    row = cg.gw.ledger.entries()[-1]
    assert row["task"] == "grade_primary" and row["cloud"] is False


def test_local_grader_down_is_review_never_cloud(tmp_path):
    """§13-10: unavailability of the local grader -> REVIEW. Nothing tries a
    different backend; the ledger shows the failed local attempt only."""
    def _dead(system, blocks):
        raise RuntimeError("connection refused")
    cg = _CountingGateway(
        tmp_path,
        routes={
            "ocr_primary": {"backend": "openrouter", "model": "vendor/ocr"},
            "grade_primary": {"backend": "ollama", "base_url": LOCAL_URL, "model": "local-grader"},
        },
        responders={"local-grader": _dead})
    d = escalate_grade(pack=_pack(), selected="A", transcription="טקסט", version="default",
                       selection_correct=True, gateway=cg.gw)
    assert d.outcome == "review" and d.result is None
    assert cg.calls == ["local-grader"], "exactly one (local) attempt; no fallback anywhere"


def test_reliability_reason_code_for_a_dead_local_grader_is_typed():
    """The REVIEW created for a dead LOCAL grading route carries
    LOCAL_GRADER_UNAVAILABLE (a registered review reason), not the generic
    provider code."""
    from autograder.reliability import _route_is_cloud
    from autograder.reviewqueue import REASONS

    assert "LOCAL_GRADER_UNAVAILABLE" in REASONS
    gw = ModelGateway.from_dict({"models": {
        "grade_primary": {"backend": "ollama", "base_url": LOCAL_URL, "model": "q"}}})
    assert _route_is_cloud(gw, "grade_primary") is False
    gw2 = ModelGateway.from_dict({"models": {
        "grade_primary": {"backend": "openrouter", "model": "vendor/m"}}},
        execution_mode="research")
    assert _route_is_cloud(gw2, "grade_primary") is True


def test_malformed_local_grade_is_review_not_a_fabricated_grade(tmp_path):
    """§13-11: a structurally valid but rubric-invalid grade (score out of
    range) fails deterministic validation; with no escalation route the item
    is REVIEW and the invalid score is only a proposal, never an AUTO."""
    cg = _CountingGateway(
        tmp_path,
        routes={"grade_primary": {"backend": "ollama", "base_url": LOCAL_URL, "model": "local-grader"}},
        responders={"local-grader": lambda s, b: GradeResult(score=99.0)})
    d = escalate_grade(pack=_pack(), selected="A", transcription="טקסט", version="default",
                       selection_correct=True, gateway=cg.gw)
    assert d.outcome == "review"


def test_wrong_choice_deterministic_zero_needs_no_ocr_and_no_grading():
    """§13-12: a confidently wrong selection under wrong_choice_zero scores
    locally — OCR calls 0, grading calls 0 (the decision happens BEFORE any
    model work)."""
    mc = MCResolution(selected="B", state="single_mark", confidence=0.99,
                      source="deterministic", candidates=["B"])
    gate = decide_before_ocr(policy="wrong_choice_zero", mc=mc, accepted=["A"],
                             points_selection=4.0, points_max=4.0, min_confidence=0.9)
    assert gate.action == "score_locally"


def test_choice_only_never_pays_for_explanation_ocr_or_grading():
    """§13-13."""
    mc = MCResolution(selected="A", state="single_mark", confidence=0.99,
                      source="deterministic", candidates=["A"])
    gate = decide_before_ocr(policy="choice_only", mc=mc, accepted=["A"],
                             points_selection=4.0, points_max=4.0, min_confidence=0.9)
    assert gate.action == "score_locally"


def test_an_exact_ocr_cache_hit_makes_no_provider_call(tmp_path):
    """§13-14: the second identical OCR request is served from the local
    request cache; the backend sees exactly one call."""
    cg = _CountingGateway(
        tmp_path,
        routes={"ocr_primary": {"backend": "openrouter", "model": "vendor/ocr"}},
        responders={"vendor/ocr": lambda s, b: ExplanationTranscription(
            sub_item_id="1", transcription="שלום", legibility="full")})
    blocks = [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAA"}}]
    r1 = cg.gw.call(task="ocr_primary", system=EXPLANATION_OCR_SYSTEM, content_blocks=blocks,
                    output_model=ExplanationTranscription, meta={"job_id": "j"})
    r2 = cg.gw.call(task="ocr_primary", system=EXPLANATION_OCR_SYSTEM, content_blocks=blocks,
                    output_model=ExplanationTranscription, meta={"job_id": "j"})
    assert not r1.cache_hit and r2.cache_hit
    assert cg.calls == ["vendor/ocr"], "one provider call; the retry is a cache hit"


def test_grading_failures_are_not_cached_as_results(tmp_path):
    calls = {"n": 0}

    def flaky(system, blocks):
        calls["n"] += 1
        raise RuntimeError("down")
    cg = _CountingGateway(
        tmp_path,
        routes={"grade_primary": {"backend": "ollama", "base_url": LOCAL_URL, "model": "g"}},
        responders={"g": flaky})
    for _ in (1, 2):
        with pytest.raises(Exception):
            cg.gw.call(task="grade_primary", system=GRADE_SYSTEM,
                       content_blocks=[{"type": "text", "text": "grade"}],
                       output_model=GradeResult, meta={})
    assert calls["n"] == 2, "a failure is never served back from the cache"


def test_local_grading_needs_no_network(tmp_path, monkeypatch):
    """§13-20: with sockets disabled outright, the local-grading code path
    (mock transport) completes end to end."""
    import socket

    def _no(*a, **kw):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", _no)
    cg = _CountingGateway(
        tmp_path,
        routes={"grade_primary": {"backend": "ollama", "base_url": LOCAL_URL, "model": "g"}},
        responders={"g": lambda s, b: GradeResult(score=4.0)})
    d = escalate_grade(pack=_pack(), selected="A", transcription="הסבר", version="default",
                       selection_correct=True, gateway=cg.gw)
    assert d.outcome == "auto"


def test_dead_local_grader_reason_code_end_to_end(tmp_path):
    """Through the full reliability route: a grade_primary route whose local
    backend cannot be reached parks the item as REVIEW /
    LOCAL_GRADER_UNAVAILABLE — even though escalate_grade absorbs the
    exception internally — and no other backend is consulted."""
    from autograder.gradingpack import build_all_packs
    from autograder.reliability import ReliabilityConfig, run_reliability_judging
    from autograder.schema import ExamExtraction, QuestionExtraction, SubItemExtraction
    from tests.test_grade import make_key

    key = make_key()
    ext = ExamExtraction(questions=[QuestionExtraction(
        question_id="1", source_pages=[1], authoritative_source="sheet",
        sub_items=[SubItemExtraction(sub_item_id="1", status="answered", final_answer="F",
                                     explanation_transcription="התדרים הגבוהים נשמרים בתמונה",
                                     explanation_legibility="full",
                                     interpretation_rationale="", confidence=1.0)])])
    built = []

    def factory(cfg):
        built.append(cfg)

        def responder(model, system, blocks):
            raise RuntimeError("connection refused")
        return MockBackend(config=cfg, responder=responder)

    gw = ModelGateway.from_dict(
        {"models": {"grade_primary": {"backend": "ollama", "base_url": LOCAL_URL,
                                      "model": "local-grader"}}},
        backend_factory=factory)
    run = run_reliability_judging(key=key, extraction=ext, version="A1",
                                  config=ReliabilityConfig(mode="reliability"),
                                  gateway=gw, packs=build_all_packs(key, {}),
                                  exam_id="exam-001")
    d = run.decisions[0]
    assert d.final_state == "REVIEW"
    assert d.reason_code == "LOCAL_GRADER_UNAVAILABLE"
    assert len(built) == 1 and built[0].backend == "ollama_native"


def test_verify_agreement_requires_zero_token_differences():
    """A single disagreeing token — e.g. an added Hebrew negation — must fail
    the AUTO gate even when the char-level similarity clears the floor."""
    from autograder.escalation import (OCRVerifyTranscription, compare_transcriptions,
                                       escalate_ocr)

    primary = "התדרים הגבוהים נשמרים בתמונה לאחר הסינון"
    negated = "התדרים הגבוהים לא נשמרים בתמונה לאחר הסינון"
    assert compare_transcriptions(primary, negated)["similarity"] >= 0.95

    def factory(cfg):
        return MockBackend(config=cfg, responder=lambda m, s, b: OCRVerifyTranscription(
            transcription=negated, legibility="full"))

    gw = ModelGateway.from_dict(
        {"models": {"ocr_verify": {"backend": "mock", "model": "v"}}}, backend_factory=factory)
    d = escalate_ocr(transcription=primary, crop_png_b64="AAA=", gateway=gw,
                     extra_suspicion=["self_declared_partial"])   # force the verifier
    assert d.outcome == "review", "one added negation token must never AUTO"
    assert d.verify["additions"] == 1
