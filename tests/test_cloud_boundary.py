"""The production cloud boundary: the cloud is for OCR transcription ONLY.

This is the 2026-08 product decision, enforced in code (cloudboundary.py) at
the single choke point every provider request passes through — never by
models.toml alone. Classification is by EFFECTIVE backend + URL; research
mode is explicit and benchmark-only. No provider is contacted anywhere in
this file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autograder.backends import BackendConfig
from autograder.backends.mock import MockBackend
from autograder.cloudboundary import (CLOUD_OCR_ALLOWLIST, CloudBoundaryError,
                                      approved_cloud_ocr_systems, check_cloud_call,
                                      forbidden_cloud_markers, is_remote_route)
from autograder.escalation import (GRADE_SYSTEM, OCR_VERIFY_INDEPENDENT_SYSTEM,
                                   OCR_VERIFY_SYSTEM, GradeResult)
from autograder.gateway import GatewayConfigError, ModelGateway
from autograder.prompts import EXPLANATION_OCR_SYSTEM
from autograder.requestcache import RequestCache
from autograder.schema import ExplanationTranscription
from autograder.usage import UsageLedger

REPO = Path(__file__).resolve().parents[1]
OPENROUTER_URL = "https://openrouter.ai/api/v1"
LOCAL_URL = "http://localhost:11434/v1"
REMOTE_OPENAI_URL = "https://api.groq.com/openai/v1"


# --------------------------------------------------------------------------
# the allowlist itself
# --------------------------------------------------------------------------


def test_the_allowlist_is_minimal_and_explicit():
    assert CLOUD_OCR_ALLOWLIST == frozenset({"ocr_primary", "ocr_verify"})


def test_remote_classification_by_effective_destination():
    assert is_remote_route("openrouter", None)
    assert is_remote_route("anthropic", None)
    assert is_remote_route("openai", OPENROUTER_URL)
    assert is_remote_route("openai", REMOTE_OPENAI_URL)
    # locality, not backend name, decides for URL-based backends
    assert not is_remote_route("openai", LOCAL_URL)
    assert not is_remote_route("openai", "http://192.168.1.20:8000/v1")
    assert not is_remote_route("ollama", None)
    assert not is_remote_route("ollama_native", LOCAL_URL)
    assert not is_remote_route("mock", None)
    # an ollama-native backend pointed OFF-machine is remote here, even though
    # usage.is_cloud_route (billing) classifies it as local ollama
    assert is_remote_route("ollama_native", "http://203.0.113.5:11434")


# --------------------------------------------------------------------------
# §2 task layer: what may / may not cross
# --------------------------------------------------------------------------


def _check(task, backend="openrouter", base_url=None, mode="production", **kw):
    return check_cloud_call(task=task, backend=backend, base_url=base_url,
                            execution_mode=mode, **kw)


def test_openrouter_ocr_primary_is_allowed():
    _check("ocr_primary", system=EXPLANATION_OCR_SYSTEM,
           content_blocks=[{"type": "image", "source": {"type": "base64",
                                                        "media_type": "image/png", "data": "x"}}])


def test_openrouter_ocr_verify_is_allowed_only_under_the_independent_contract():
    _check("ocr_verify", system=OCR_VERIFY_INDEPENDENT_SYSTEM,
           content_blocks=[{"type": "image", "source": {"type": "base64",
                                                        "media_type": "image/png", "data": "x"}}])
    # the legacy fidelity-verdict prompt (shows the verifier the primary
    # reading) is research-only and NOT an approved production contract
    with pytest.raises(CloudBoundaryError) as e:
        _check("ocr_verify", system=OCR_VERIFY_SYSTEM)
    assert e.value.code == "UNREGISTERED_OCR_PROMPT"


@pytest.mark.parametrize("task", ["grade_primary", "grade_escalate", "mc_resolve_cloud",
                                  "variant_resolve_cloud", "align_resolve_cloud",
                                  "policy_infer_cloud", "grading_rag", "anything_new"])
def test_every_non_ocr_task_is_refused_to_openrouter(task):
    with pytest.raises(CloudBoundaryError) as e:
        _check(task)
    assert e.value.code == "CLOUD_TASK_FORBIDDEN"
    assert "--research" in str(e.value) or "research" in str(e.value)


@pytest.mark.parametrize("backend,base_url", [
    ("openai", REMOTE_OPENAI_URL),         # remote OpenAI-compatible endpoint
    ("openai", OPENROUTER_URL),            # openrouter behind an openai route
    ("anthropic", None),                   # Anthropic direct
    ("ollama_native", "http://203.0.113.5:11434"),   # remote 'ollama'
])
def test_remote_grading_endpoints_are_rejected_whatever_the_backend_name(backend, base_url):
    with pytest.raises(CloudBoundaryError):
        _check("grade_primary", backend=backend, base_url=base_url)


@pytest.mark.parametrize("backend,base_url", [
    ("ollama", LOCAL_URL),
    ("ollama_native", None),
    ("openai", LOCAL_URL),
    ("openai", "http://192.168.1.20:8000/v1"),
    ("mock", None),
])
def test_local_and_mock_grading_backends_are_allowed(backend, base_url):
    _check("grade_primary", backend=backend, base_url=base_url,
           system=GRADE_SYSTEM)          # local grading may carry the grading prompt


def test_research_mode_is_an_explicit_bypass_and_bad_modes_are_refused():
    _check("grade_primary", mode="research")
    with pytest.raises(CloudBoundaryError):
        _check("grade_primary", mode="prod")     # typo'd mode never silently allows


# --------------------------------------------------------------------------
# §2/§13 payload layer: OCR requests must carry no grading material
# --------------------------------------------------------------------------


def test_the_approved_prompt_registry_is_exactly_the_known_ocr_contracts():
    """The two production OCR contracts + the six frozen m2-strict-v1 bench
    transcription prompts (registered 2026-09-02 for the pre-registered OCR
    validation campaign). Nothing else — and never the legacy fidelity-verdict
    prompt, which sees the primary reading."""
    from autograder.benchmark.roles import _load_historical_prompts
    from autograder.escalation import OCR_VERIFY_SYSTEM
    bench = _load_historical_prompts()
    assert set(bench) == {"handwritten_line", "handwritten_cell", "printed_rtl",
                          "mixed_he_en", "formula_printed",
                          "option_row_association"}
    assert approved_cloud_ocr_systems() == frozenset(
        {EXPLANATION_OCR_SYSTEM, OCR_VERIFY_INDEPENDENT_SYSTEM}
        | set(bench.values()))
    assert OCR_VERIFY_SYSTEM not in approved_cloud_ocr_systems()


def test_an_ocr_task_name_cannot_smuggle_the_grading_prompt():
    with pytest.raises(CloudBoundaryError) as e:
        _check("ocr_primary", system=GRADE_SYSTEM)
    assert e.value.code == "UNREGISTERED_OCR_PROMPT"


@pytest.mark.parametrize("marker_i", range(4))
def test_no_rubric_solution_or_rag_header_may_ride_an_ocr_payload(marker_i):
    from autograder.gradingpack import CONTEXT_HEADERS
    marker = CONTEXT_HEADERS[marker_i]
    with pytest.raises(CloudBoundaryError) as e:
        _check("ocr_primary", system=EXPLANATION_OCR_SYSTEM,
               content_blocks=[{"type": "text", "text": f"context\n{marker} sneaky"}])
    assert e.value.code == "GRADING_CONTENT_IN_OCR_PAYLOAD"


def test_the_tripwire_markers_are_the_real_rendered_headers():
    """The markers cannot drift from what the pack actually renders."""
    from autograder.gradingpack import CONTEXT_HEADERS, QuestionGradingPack, RagEvidence
    pack = QuestionGradingPack(
        question_id="1", question_text="q", question_type="multiple_choice", max_score=4.0,
        correct_by_version={}, rubric=["r1"], scoring_rules=["rule"], grading_policy="choice_plus_explanation",
        official_solution={"1": "sol"},
        rag_evidence=[RagEvidence(chunk_id="c1", source="s", text="chunk", page=1, similarity=0.9)])
    ctx = pack.to_grader_context()
    for h in CONTEXT_HEADERS:
        assert h in ctx, h
    markers = forbidden_cloud_markers()
    assert sum(1 for m in markers if m in ctx) >= len(CONTEXT_HEADERS)


def test_a_full_grading_context_never_passes_the_boundary():
    from autograder.gradingpack import QuestionGradingPack
    pack = QuestionGradingPack(
        question_id="1", question_text="שאלה", question_type="multiple_choice", max_score=4.0,
        correct_by_version={}, rubric=["הסבר נכון"], scoring_rules=[],
        grading_policy="choice_plus_explanation", official_solution={"1": "פתרון"})
    with pytest.raises(CloudBoundaryError):
        _check("ocr_primary", system=EXPLANATION_OCR_SYSTEM,
               content_blocks=[{"type": "text", "text": pack.to_grader_context()}])


def test_scores_and_grades_have_no_transport_to_the_cloud():
    """Instructor scores/grades live only in grading packs, labels and
    results — none of which the two OCR request builders touch. The payload
    check plus the prompt registry close the deliberate-misuse path; this
    test pins the honest-path shape: the real production OCR payload carries
    only images and locate-the-writing text."""
    # the real lazy-OCR block shapes (extract.lazy_explanation_ocr)
    blocks = [
        {"type": "text", "text": "Question 2: title\nType: multiple_choice\nSub-items (2):"},
        {"type": "text", "text": "Relevant scan pages follow (1 pages)."},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "x"}},
        {"type": "text", "text": "Transcribe the student's written explanation for sub-item 1 "
                                 "of question 2 now. Report sub_item_id '1'."},
    ]
    _check("ocr_primary", system=EXPLANATION_OCR_SYSTEM, content_blocks=blocks)


# --------------------------------------------------------------------------
# gateway integration: enforced BEFORE cache / budget / serialization
# --------------------------------------------------------------------------


def _gw(tmp_path, routes, mode="production"):
    built = []

    def factory(cfg):
        built.append(cfg)
        return MockBackend(config=cfg, responder=lambda model, system, blocks: GradeResult(score=1.0))

    gw = ModelGateway.from_dict({"models": routes}, backend_factory=factory,
                                cache=RequestCache(tmp_path / "cache"),
                                ledger=UsageLedger(tmp_path / "usage.jsonl"),
                                execution_mode=mode)
    return gw, built


def test_gateway_call_refuses_cloud_grading_before_any_backend_or_cache_work(tmp_path):
    gw, built = _gw(tmp_path, {"grade_primary": {"backend": "openrouter", "model": "vendor/m"}})
    with pytest.raises(CloudBoundaryError):
        gw.call(task="grade_primary", system=GRADE_SYSTEM,
                content_blocks=[{"type": "text", "text": "grade this"}],
                output_model=GradeResult, meta={"job_id": "j"})
    assert built == [], "no backend may even be constructed for a refused route"
    assert not any((tmp_path / "cache").rglob("*.json")), "nothing was cached"
    assert gw.ledger.entries() == [], "nothing was recorded as attempted"


def test_gateway_call_allows_cloud_ocr_and_local_grading(tmp_path):
    gw, _ = _gw(tmp_path, {
        "ocr_primary": {"backend": "openrouter", "model": "vendor/ocr"},
        "grade_primary": {"backend": "ollama", "base_url": LOCAL_URL, "model": "qwen3-vl:8b-instruct"},
    })
    r1 = gw.call(task="ocr_primary", system=EXPLANATION_OCR_SYSTEM,
                 content_blocks=[{"type": "image", "source": {"type": "base64",
                                                              "media_type": "image/png", "data": "x"}}],
                 output_model=GradeResult, meta={"job_id": "j"})
    r2 = gw.call(task="grade_primary", system=GRADE_SYSTEM,
                 content_blocks=[{"type": "text", "text": "grade"}],
                 output_model=GradeResult, meta={"job_id": "j"})
    assert r1.value.score == 1.0 and r2.value.score == 1.0


def test_pointing_grading_at_openrouter_through_an_openai_route_changes_nothing(tmp_path):
    """models.toml cannot override the rule — the guard is not configuration."""
    gw, built = _gw(tmp_path, {"grade_primary": {"backend": "openai", "base_url": OPENROUTER_URL,
                                                 "model": "vendor/m"}})
    with pytest.raises(CloudBoundaryError):
        gw.call(task="grade_primary", system=GRADE_SYSTEM,
                content_blocks=[{"type": "text", "text": "grade"}],
                output_model=GradeResult, meta={})
    assert built == []


def test_research_mode_must_be_named_at_construction(tmp_path):
    gw, _ = _gw(tmp_path, {"grade_primary": {"backend": "openrouter", "model": "vendor/m"}},
                mode="research")
    res = gw.call(task="grade_primary", system=GRADE_SYSTEM,
                  content_blocks=[{"type": "text", "text": "grade"}],
                  output_model=GradeResult, meta={})
    assert res.value.score == 1.0
    with pytest.raises(GatewayConfigError):
        ModelGateway.from_dict({"models": {"t": {"backend": "mock", "model": "m"}}},
                               execution_mode="researchy")


def test_describe_marks_blocked_roles_for_the_gui(tmp_path):
    gw, _ = _gw(tmp_path, {
        "ocr_primary": {"backend": "openrouter", "model": "vendor/ocr"},
        "grade_primary": {"backend": "openrouter", "model": "vendor/m"},
        "mc_resolve": {"backend": "ollama", "base_url": LOCAL_URL, "model": "q"},
    })
    d = gw.describe()
    assert d["ocr_primary"]["remote"] and not d["ocr_primary"]["blocked_in_production"]
    assert d["grade_primary"]["remote"] and d["grade_primary"]["blocked_in_production"]
    assert not d["mc_resolve"]["remote"] and not d["mc_resolve"]["blocked_in_production"]


# --------------------------------------------------------------------------
# no fallback can flip local grading into cloud grading
# --------------------------------------------------------------------------


def test_a_failing_local_grader_never_falls_back_to_a_cloud_backend(tmp_path):
    """escalate_grade on a dead local route -> REVIEW; no second backend is
    consulted, no cloud route exists to consult."""
    from autograder.escalation import escalate_grade
    from autograder.gradingpack import QuestionGradingPack

    built = []

    def factory(cfg):
        built.append(cfg)

        def responder(model, system, blocks):
            raise RuntimeError("local grader down")
        return MockBackend(config=cfg, responder=responder)

    gw = ModelGateway.from_dict(
        {"models": {"grade_primary": {"backend": "ollama", "base_url": LOCAL_URL, "model": "q"}}},
        backend_factory=factory)
    pack = QuestionGradingPack(question_id="1", question_text="q", question_type="multiple_choice",
                               max_score=4.0, correct_by_version={}, rubric=["r"], scoring_rules=[],
                               grading_policy="choice_plus_explanation", official_solution={})
    d = escalate_grade(pack=pack, selected="A", transcription="טקסט", version="default",
                       selection_correct=True, gateway=gw)
    assert d.outcome == "review" and d.result is None
    assert len(built) == 1 and built[0].backend == "ollama_native"


def test_production_gateways_default_to_production_mode(tmp_path):
    from autograder.orchestrator import setup_from_config
    cfg = tmp_path / "models.toml"
    cfg.write_text('[models.grade_primary]\nbackend = "mock"\nmodel = "m"\n', encoding="utf-8")
    rt = setup_from_config(cfg, tmp_path / "state")
    assert rt.gateway.execution_mode == "production"


# --------------------------------------------------------------------------
# configuration: the shipped examples encode the architecture
# --------------------------------------------------------------------------


def test_the_example_config_routes_grading_locally_and_ocr_to_the_cloud():
    import tomllib
    data = tomllib.loads((REPO / "models.example.toml").read_text(encoding="utf-8"))
    models = data["models"]
    for t in ("ocr_primary", "ocr_verify"):
        assert models[t]["backend"] == "openrouter", t
    for t in ("grade_primary", "grade_escalate"):
        assert models[t]["backend"] == "ollama", t
        assert not is_remote_route("ollama", models[t].get("base_url")), t
        assert models[t]["model"] == "UNSELECTED", f"{t}: no local winner has been benchmarked yet"
        assert models[t]["prompt_version"] == "grade-v4-charitable"
    assert "mc_resolve_cloud" not in models and "variant_resolve_cloud" not in models


def test_the_research_config_is_explicitly_marked_non_production():
    text = (REPO / "models.research.example.toml").read_text(encoding="utf-8")
    assert "NOT A PRODUCTION CONFIGURATION" in text
    assert "--research" in text
    import tomllib
    data = tomllib.loads(text)
    assert data["models"]["grade_primary"]["backend"] == "openrouter"


def test_no_model_slug_is_hardcoded_in_the_boundary_or_gateway():
    for f in ("autograder/cloudboundary.py", "autograder/gateway.py"):
        text = (REPO / f).read_text(encoding="utf-8")
        for slug in ("gemini", "claude-sonnet", "gpt-", "qwen"):
            assert slug not in text.lower(), (f, slug)


# --------------------------------------------------------------------------
# §13-21: no production GUI action can serialize a cloud grading request
# --------------------------------------------------------------------------


def test_no_gui_or_pipeline_path_opts_into_research_mode():
    """The GUI's only model-calling action spawns `autograder run-job` ->
    `autograder grade`, whose gateway comes from orchestrator.setup_from_config
    (production mode, boundary armed) and whose legacy direct path refuses
    cloud outright (guard_direct_cloud_backend). Research mode is reachable
    ONLY from the bench CLI flag — pinned here at the source level: no
    production module ever passes execution_mode."""
    for f in ("autograder/webui.py", "autograder/orchestrator.py", "autograder/cli.py",
              "autograder/jobs.py", "autograder/reliability.py", "autograder/escalation.py",
              "autograder/extract.py", "autograder/mcresolve.py"):
        text = (REPO / f).read_text(encoding="utf-8")
        assert 'execution_mode="research"' not in text, f
        assert "execution_mode='research'" not in text, f
    # and the bench runner passes it only from the spec flag
    runner = (REPO / "autograder/benchmark/runner.py").read_text(encoding="utf-8")
    assert 'execution_mode="research" if spec.research else "production"' in runner


def test_unpriced_paid_ocr_with_a_cost_ceiling_refuses_at_runtime_construction(tmp_path):
    """§12: a cost ceiling is unenforceable pre-call for an unpriced model
    (predicted $0 passes every check), so the production runtime refuses to
    come up until the OCR model has a [pricing] entry."""
    from autograder.orchestrator import setup_from_config
    cfg = tmp_path / "models.toml"
    cfg.write_text(
        '[models.ocr_primary]\nbackend = "openrouter"\nmodel = "vendor/ocr"\n'
        '[budget]\nenabled = true\nmax_cost_total = 10.0\n', encoding="utf-8")
    with pytest.raises(GatewayConfigError, match="pricing"):
        setup_from_config(cfg, tmp_path / "state")
    # priced -> fine
    cfg.write_text(
        '[models.ocr_primary]\nbackend = "openrouter"\nmodel = "vendor/ocr"\n'
        '[budget]\nenabled = true\nmax_cost_total = 10.0\n'
        '[pricing."vendor/ocr"]\ninput = 0.15\noutput = 0.6\n', encoding="utf-8")
    rt = setup_from_config(cfg, tmp_path / "state")
    assert rt.gateway.execution_mode == "production"
    # no ceiling -> the estimator is optional and construction succeeds
    cfg.write_text('[models.ocr_primary]\nbackend = "openrouter"\nmodel = "vendor/ocr"\n',
                   encoding="utf-8")
    assert setup_from_config(cfg, tmp_path / "state2")
