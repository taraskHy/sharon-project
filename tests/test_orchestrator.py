"""End-to-end plumbing + resume semantics — offline, mocked."""

from __future__ import annotations

import json

from autograder.backends import BackendError
from autograder.backends.mock import MockBackend
from autograder.orchestrator import (handle_model_failure, install_hooks, openrouter_configured,
                                     prepare_exam_package, setup_from_config)
from autograder.usage import BudgetExceeded, BudgetLimits
from autograder.cloudboundary import research_authorization
from tests.test_grade import make_key


def _cfg(tmp_path):
    p = tmp_path / "models.toml"
    p.write_text('[models.mc_resolve]\nbackend="mock"\nmodel="local"\n'
                 '[models.grade_primary]\nbackend="mock"\nmodel="cloud"\n', encoding="utf-8")
    return p


def test_setup_and_hooks_install_and_uninstall(tmp_path):
    rt = setup_from_config(_cfg(tmp_path), tmp_path / "state",
                           backend_factory=lambda c: MockBackend(config=c))
    from autograder import extract, grade
    install_hooks(rt, {"1": "wrong_choice_zero"})
    assert extract.get_mc_resolver() is not None
    install_hooks(None, None)
    assert extract.get_mc_resolver() is None
    assert not openrouter_configured(rt)   # mock-only config, no key


def test_prepare_exam_package_persists_and_reuses(tmp_path):
    key = make_key()
    keyp = tmp_path / "pkg" / "exam.answer_key.json"
    keyp.parent.mkdir()
    keyp.write_text("{}", encoding="utf-8")
    calls = {"n": 0}

    def retrieve(course_id, query, top_k, embed_fn=None):
        calls["n"] += 1
        return [{"chunk_id": "c1", "source": "s", "page": 1, "similarity": 0.8, "text": "קטע"}]

    out1 = prepare_exam_package(None, key=key, key_bytes=b"K1", key_path=keyp,
                                exam_text_layer="נוסח A1 נוסח A2 נוסח A3", course_id="CV",
                                course_index_hash="idx1", retrieve=retrieve, rag_policy="RAG_ALWAYS",
                                    packages_root=tmp_path / "state")
    assert set(out1["packs"]) == {q.id for q in key.questions}
    assert (keyp.with_name("exam.answer_key.variants.json")).exists()   # emitted sidecar
    n_first = calls["n"]
    out2 = prepare_exam_package(None, key=key, key_bytes=b"K1", key_path=keyp,
                                exam_text_layer="נוסח A1 נוסח A2 נוסח A3", course_id="CV",
                                course_index_hash="idx1", retrieve=retrieve, rag_policy="RAG_ALWAYS",
                                    packages_root=tmp_path / "state")
    assert calls["n"] == n_first                    # packs reused, no re-retrieval
    assert out2["package_fingerprint"] == out1["package_fingerprint"]
    out3 = prepare_exam_package(None, key=key, key_bytes=b"K1", key_path=keyp,
                                exam_text_layer="נוסח A1 נוסח A2 נוסח A3", course_id="CV",
                                course_index_hash="idx2", retrieve=retrieve, rag_policy="RAG_ALWAYS",
                                packages_root=tmp_path / "state")
    assert calls["n"] > n_first                     # course index changed -> packs rebuilt


def test_packages_do_not_retrieve_course_context_by_default(tmp_path):
    """Grading-side retrieval is opt-in: an unspecified RAG policy sends no
    course context and performs no retrieval at all."""
    key = make_key()
    keyp = tmp_path / "pkg" / "exam.answer_key.json"
    keyp.parent.mkdir()
    keyp.write_text("{}", encoding="utf-8")
    calls = {"n": 0}

    def retrieve(course_id, query, top_k, embed_fn=None):
        calls["n"] += 1
        return [{"chunk_id": "c1", "source": "s", "page": 1, "similarity": 0.8, "text": "קטע"}]

    out = prepare_exam_package(None, key=key, key_bytes=b"K1", key_path=keyp,
                               course_id="CV", course_index_hash="idx1", retrieve=retrieve,
                               packages_root=tmp_path / "state")
    assert calls["n"] == 0
    for pack in out["packs"].values():
        assert pack.rag_policy == "RAG_DISABLED"
        assert pack.rag_evidence == [] and pack.rag_chars == 0
        assert "Course context" not in pack.to_grader_context()


def test_model_failure_pauses_not_destroys(tmp_path):
    job = tmp_path / "job"
    job.mkdir()
    rec = handle_model_failure(BudgetExceeded("hard budget cost reached"), job, "exam-003", "grade")
    assert rec["action"] == "pause_job" and (job / "pause.request").exists()
    rec2 = handle_model_failure(BackendError("HTTP 429 from OpenRouter"), job, "exam-004", "ocr")
    assert rec2["action"] == "pause_item"
    rec3 = handle_model_failure(ValueError("bad key json"), job, "exam-005", "key")
    assert rec3["action"] == "fail_item"
    lines = (job / "model_failures.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3 and json.loads(lines[0])["exam_id"] == "exam-003"


def test_rerun_resumes_missing_items_only(tmp_path):
    """The gateway cache makes a rerun free for already-answered requests:
    a failed job resumed later re-issues ONLY the calls that never succeeded."""
    from pydantic import BaseModel
    from autograder.gateway import ModelGateway
    from autograder.requestcache import RequestCache

    class Out(BaseModel):
        text: str

    provider_calls = {"n": 0}
    fail_next = {"flag": False}

    def factory(cfg):
        def responder(model, system, blocks):
            if fail_next["flag"]:
                fail_next["flag"] = False
                raise BackendError("OpenRouter unreachable")
            provider_calls["n"] += 1
            return Out(text="ok")
        return MockBackend(config=cfg, responder=responder)

    def gw():
        # research mode: the cache-resume mechanics are what is under test; a
        # production gateway would refuse the cloud-shaped grading route
        # outright (tests/test_cloud_boundary.py).
        return ModelGateway.from_dict({"models": {"grade_primary": {"backend": "openrouter", "model": "m"}}},
                                      backend_factory=factory, cache=RequestCache(tmp_path / "c"),
                                      execution_mode="research",
                                      research_auth=research_authorization(
                                          "test:orchestrator", tasks=["grade_primary"],
                                          models=["m"]))

    g1 = gw()
    g1.call(task="grade_primary", system="s", content_blocks=[{"type": "text", "text": "exam1"}], output_model=Out)
    fail_next["flag"] = True
    try:
        g1.call(task="grade_primary", system="s", content_blocks=[{"type": "text", "text": "exam2"}], output_model=Out)
    except BackendError:
        pass
    assert provider_calls["n"] == 1
    # "later": fresh process, same cache -> exam1 is a hit, exam2 is the only new call
    g2 = gw()
    r1 = g2.call(task="grade_primary", system="s", content_blocks=[{"type": "text", "text": "exam1"}], output_model=Out)
    r2 = g2.call(task="grade_primary", system="s", content_blocks=[{"type": "text", "text": "exam2"}], output_model=Out)
    assert r1.cache_hit is True and r2.cache_hit is False and provider_calls["n"] == 2
