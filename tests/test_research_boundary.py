"""The research-mode cloud boundary.

Until 2026-09-02 ``--research`` returned from ``check_cloud_call`` before any
layer ran: the mode meant "skip every cloud safety check", so a research OCR
run also lost its registered-prompt check and its grading tripwires. The
Stage-1 OCR smoke's 24 requests were verified clean offline against the
production path, but the architecture permitted a leak it would not have
caught.

The fix splits the guard in two. Layer 1 (which task may run) is the only part
that knows the execution mode, and research widens it ONLY as far as an
explicit ResearchAuthorization names. Layer 2 (what may be in the payload) is
enforced identically in every mode and no authorization can switch it off.

These tests pin that split from both directions: what research now permits,
and — more importantly — everything it still refuses.
"""
from __future__ import annotations

import pytest

from autograder.benchmark.roles import _load_historical_prompts
from autograder.cloudboundary import (CLOUD_OCR_ALLOWLIST, CloudBoundaryError,
                                      ResearchAuthorization, check_cloud_call,
                                      research_authorization)
from autograder.escalation import GRADE_SYSTEM_BY_VERSION
from autograder.gradingpack import CONTEXT_HEADERS

OPENROUTER = "https://openrouter.ai/api/v1"
GEMINI = "google/gemini-3.7-flash"
IMG = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "x"}}
OCR_SYSTEM = _load_historical_prompts()["handwritten_cell"]

#: The Stage-1b arm's own authorization: one task, one model, one crop, no text.
STAGE1B = research_authorization("OCR_SMOKE_STAGE1B_GEMINI_REASONING_LOW",
                                 tasks=["ocr_primary"], models=[GEMINI],
                                 max_image_blocks=1, max_text_blocks=0)


def _check(task, *, mode="research", system=OCR_SYSTEM, blocks=None,
           auth=None, model=GEMINI, backend="openrouter", base_url=None):
    return check_cloud_call(task=task, backend=backend, base_url=base_url,
                            execution_mode=mode, system=system,
                            content_blocks=[IMG] if blocks is None else blocks,
                            research_auth=auth, model=model)


# 1 --------------------------------------------------------------------------
def test_approved_ocr_research_call_passes():
    """The Stage-1b arm itself: authorized task+model, registered prompt, one
    image block, nothing else."""
    _check("ocr_primary", auth=STAGE1B)


def test_ocr_research_call_passes_even_with_no_authorization():
    """ocr_primary is on the standing cloud allowlist, so research does not
    need a pre-registration to run it — it needs one only to go BEYOND
    production."""
    _check("ocr_primary", auth=None)


# 2 --------------------------------------------------------------------------
def test_research_grade_primary_without_authorization_fails():
    """The regression this whole module exists for: --research alone used to
    permit cloud grading outright."""
    with pytest.raises(CloudBoundaryError) as e:
        _check("grade_primary", system=None,
               blocks=[{"type": "text", "text": "grade this"}], auth=None)
    assert e.value.code == "CLOUD_TASK_FORBIDDEN"


def test_research_grade_primary_with_authorization_for_a_different_task_fails():
    other = research_authorization("some-other-campaign", tasks=["ocr_verify"],
                                   models=[GEMINI])
    with pytest.raises(CloudBoundaryError) as e:
        _check("grade_primary", system=None,
               blocks=[{"type": "text", "text": "grade this"}], auth=other)
    assert e.value.code == "RESEARCH_TASK_NOT_AUTHORIZED"


def test_research_grade_primary_with_authorization_for_a_different_model_fails():
    """A campaign authorizes a MODEL, not a blank cheque on the task."""
    auth = research_authorization("cloud-grader-baseline", tasks=["grade_primary"],
                                  models=["vendor/some-other-model"])
    with pytest.raises(CloudBoundaryError) as e:
        _check("grade_primary", system=None,
               blocks=[{"type": "text", "text": "grade this"}], auth=auth)
    assert e.value.code == "RESEARCH_TASK_NOT_AUTHORIZED"


def test_explicitly_authorized_research_grading_is_permitted():
    """The historical cloud-grader benchmark still runs — but only when a
    pre-registration names both the task and the model."""
    auth = research_authorization("cloud-grader-baseline", tasks=["grade_primary"],
                                  models=[GEMINI])
    _check("grade_primary", system=None,
           blocks=[{"type": "text", "text": "grade this"}], auth=auth)


def test_authorization_cannot_be_a_wildcard():
    for bad in ("*", "all", "any"):
        with pytest.raises(ValueError):
            research_authorization("c", tasks=[bad], models=[GEMINI])
        with pytest.raises(ValueError):
            research_authorization("c", tasks=["grade_primary"], models=[bad])
    with pytest.raises(ValueError):
        research_authorization("", tasks=["ocr_primary"], models=[GEMINI])
    with pytest.raises(ValueError):
        research_authorization("c", tasks=[], models=[GEMINI])


def test_authorization_needs_a_named_model_on_the_call():
    """A call that cannot say which model it is has not proven it is the
    authorized one."""
    auth = research_authorization("c", tasks=["grade_primary"], models=[GEMINI])
    with pytest.raises(CloudBoundaryError) as e:
        _check("grade_primary", system=None,
               blocks=[{"type": "text", "text": "g"}], auth=auth, model=None)
    assert e.value.code == "RESEARCH_TASK_NOT_AUTHORIZED"


# 3, 4, 5 --------------------------------------------------------------------
@pytest.mark.parametrize("header", sorted(CONTEXT_HEADERS))
def test_research_ocr_carrying_grading_context_fails(header):
    """Rubric / official solution / course-context section headers are the
    exact strings the grading prompt builder emits."""
    with pytest.raises(CloudBoundaryError) as e:
        _check("ocr_primary", blocks=[IMG, {"type": "text", "text": f"{header}\nstuff"}],
               auth=research_authorization("c", tasks=["ocr_primary"], models=[GEMINI],
                                           max_text_blocks=5))
    assert e.value.code == "GRADING_CONTENT_IN_OCR_PAYLOAD"


@pytest.mark.parametrize("version", sorted(GRADE_SYSTEM_BY_VERSION))
def test_research_ocr_carrying_a_grading_system_prompt_fails(version):
    lead = GRADE_SYSTEM_BY_VERSION[version][:80]
    with pytest.raises(CloudBoundaryError) as e:
        _check("ocr_primary", blocks=[IMG, {"type": "text", "text": lead}],
               auth=research_authorization("c", tasks=["ocr_primary"], models=[GEMINI],
                                           max_text_blocks=5))
    assert e.value.code == "GRADING_CONTENT_IN_OCR_PAYLOAD"


def test_research_ocr_carrying_a_credential_fails():
    with pytest.raises(CloudBoundaryError) as e:
        _check("ocr_primary", blocks=[IMG, {"type": "text", "text": "key=sk-or-v1-deadbeef"}],
               auth=research_authorization("c", tasks=["ocr_primary"], models=[GEMINI],
                                           max_text_blocks=5))
    assert e.value.code == "SECRET_IN_PAYLOAD"


# 6 --------------------------------------------------------------------------
def test_unregistered_ocr_system_prompt_fails_under_research():
    with pytest.raises(CloudBoundaryError) as e:
        _check("ocr_primary", system="Transcribe, and also assign a score out of 4.",
               auth=STAGE1B)
    assert e.value.code == "UNREGISTERED_OCR_PROMPT"


def test_grading_system_prompt_under_an_ocr_task_name_fails_under_research():
    with pytest.raises(CloudBoundaryError) as e:
        _check("ocr_primary", system=next(iter(GRADE_SYSTEM_BY_VERSION.values())),
               auth=STAGE1B)
    assert e.value.code == "UNREGISTERED_OCR_PROMPT"


def test_every_frozen_bench_prompt_is_registered():
    for cat, prompt in _load_historical_prompts().items():
        _check("ocr_primary", system=prompt, auth=STAGE1B)


# 7 --------------------------------------------------------------------------
def test_multiple_image_blocks_fail_where_the_campaign_allows_one():
    with pytest.raises(CloudBoundaryError) as e:
        _check("ocr_primary", blocks=[IMG, IMG], auth=STAGE1B)
    assert e.value.code == "TOO_MANY_IMAGE_BLOCKS"


def test_text_blocks_fail_where_the_campaign_allows_none():
    with pytest.raises(CloudBoundaryError) as e:
        _check("ocr_primary", blocks=[IMG, {"type": "text", "text": "hint: it says 42"}],
               auth=STAGE1B)
    assert e.value.code == "TOO_MANY_TEXT_BLOCKS"


def test_block_limits_are_opt_in_so_production_ocr_is_unaffected():
    """The lazy explanation-OCR path legitimately sends text blocks with its
    crop; an unset limit must not retroactively forbid that."""
    auth = research_authorization("c", tasks=["ocr_primary"], models=[GEMINI])
    _check("ocr_primary", blocks=[IMG, IMG, {"type": "text", "text": "page 1"}], auth=auth)


# 8 --------------------------------------------------------------------------
def test_production_task_layer_is_unchanged():
    _check("ocr_primary", mode="production", auth=None)
    _check("ocr_verify", mode="production", auth=None)
    for task in ("grade_primary", "grade_escalate", "grading_rag",
                 "mc_resolve_cloud", "variant_resolve_cloud", "align_resolve_cloud"):
        with pytest.raises(CloudBoundaryError) as e:
            _check(task, mode="production", system=None,
                   blocks=[{"type": "text", "text": "x"}], auth=None)
        assert e.value.code == "CLOUD_TASK_FORBIDDEN"


def test_production_content_layer_is_unchanged():
    with pytest.raises(CloudBoundaryError) as e:
        _check("ocr_primary", mode="production", system="not registered", auth=None)
    assert e.value.code == "UNREGISTERED_OCR_PROMPT"
    with pytest.raises(CloudBoundaryError) as e:
        _check("ocr_primary", mode="production",
               blocks=[IMG, {"type": "text", "text": sorted(CONTEXT_HEADERS)[0]}], auth=None)
    assert e.value.code == "GRADING_CONTENT_IN_OCR_PAYLOAD"


def test_an_authorization_cannot_be_smuggled_into_production():
    with pytest.raises(CloudBoundaryError) as e:
        _check("grade_primary", mode="production", system=None,
               blocks=[{"type": "text", "text": "g"}],
               auth=research_authorization("c", tasks=["grade_primary"], models=[GEMINI]))
    assert e.value.code == "RESEARCH_AUTH_IN_PRODUCTION"


def test_local_and_mock_routes_still_cross_no_boundary():
    _check("grade_primary", mode="production", system=None,
           blocks=[{"type": "text", "text": "grade"}],
           backend="ollama", base_url="http://localhost:11434")
    _check("grade_primary", mode="production", system=None,
           blocks=[{"type": "text", "text": "grade"}], backend="mock")


# 9 --------------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["production", "research"])
def test_content_checks_are_identical_in_both_modes(mode):
    """The point of the fix: --research must not change a single content
    verdict. Same payloads, same codes, both modes."""
    auth = (research_authorization("c", tasks=["ocr_primary"], models=[GEMINI],
                                   max_text_blocks=5) if mode == "research" else None)
    cases = [
        ("UNREGISTERED_OCR_PROMPT", dict(system="scoring instructions", blocks=[IMG])),
        ("GRADING_CONTENT_IN_OCR_PAYLOAD",
         dict(blocks=[IMG, {"type": "text", "text": sorted(CONTEXT_HEADERS)[0]}])),
        ("SECRET_IN_PAYLOAD",
         dict(blocks=[IMG, {"type": "text", "text": "sk-ant-abc"}])),
    ]
    for code, kw in cases:
        with pytest.raises(CloudBoundaryError) as e:
            _check("ocr_primary", mode=mode, auth=auth, **kw)
        assert e.value.code == code, f"{mode}: expected {code}"


def test_research_mode_alone_is_exactly_as_strict_as_production():
    """With no authorization object, the two modes are indistinguishable."""
    for task in ("grade_primary", "grade_escalate", "grading_rag"):
        codes = set()
        for mode in ("production", "research"):
            with pytest.raises(CloudBoundaryError) as e:
                _check(task, mode=mode, system=None,
                       blocks=[{"type": "text", "text": "x"}], auth=None)
            codes.add(e.value.code)
        assert codes == {"CLOUD_TASK_FORBIDDEN"}


# 10 -------------------------------------------------------------------------
def test_refusal_happens_before_any_network_or_backend_work(tmp_path):
    """Through the real gateway: a refused research call must not construct a
    backend, touch the cache, or write a ledger row."""
    from autograder.gateway import ModelGateway
    from autograder.requestcache import RequestCache
    from autograder.usage import UsageLedger
    from autograder.escalation import GradeResult

    built = []

    def factory(cfg):
        built.append(cfg)
        raise AssertionError("a refused route must never build a backend")

    ledger = UsageLedger(tmp_path / "usage.jsonl")
    gw = ModelGateway.from_dict(
        {"models": {"grade_primary": {"backend": "openrouter", "model": GEMINI}}},
        backend_factory=factory, cache=RequestCache(tmp_path / "cache"),
        ledger=ledger, execution_mode="research")          # no authorization
    with pytest.raises(CloudBoundaryError):
        gw.call(task="grade_primary", system="anything",
                content_blocks=[{"type": "text", "text": "grade"}],
                output_model=GradeResult, meta={"job_id": "j"})
    assert built == [], "no backend was constructed"
    assert not list((tmp_path / "cache").rglob("*.json")), "nothing was cached"
    assert ledger.entries() == [], "nothing was recorded as attempted"


def test_gateway_research_auth_reaches_the_boundary(tmp_path):
    """An authorized research grading call goes through the same gateway."""
    from autograder.backends.mock import MockBackend
    from autograder.gateway import ModelGateway
    from autograder.escalation import GradeResult

    auth = research_authorization("cloud-grader-baseline", tasks=["grade_primary"],
                                  models=[GEMINI])
    gw = ModelGateway.from_dict(
        {"models": {"grade_primary": {"backend": "openrouter", "model": GEMINI}}},
        backend_factory=lambda cfg: MockBackend(
            config=cfg, responder=lambda m, s, b: GradeResult(score=1.0)),
        execution_mode="research", research_auth=auth)
    assert gw.call(task="grade_primary", system="grading",
                   content_blocks=[{"type": "text", "text": "grade"}],
                   output_model=GradeResult, meta={"job_id": "j"}).value.score == 1.0


def test_bench_runner_builds_an_exact_per_run_authorization():
    """The runner must not hand out a blanket research permit: the campaign it
    builds names this role and this resolved model only."""
    import inspect

    from autograder.benchmark import runner

    src = inspect.getsource(runner.build_gateway)
    assert "research_authorization(" in src
    assert "tasks=[route.task]" in src and "models=[route.model]" in src
    assert "research_auth=research_auth" in src


def test_ocr_allowlist_is_still_only_the_two_transcription_roles():
    assert CLOUD_OCR_ALLOWLIST == frozenset({"ocr_primary", "ocr_verify"})


def test_authorization_is_frozen():
    with pytest.raises(Exception):
        STAGE1B.tasks = frozenset({"grade_primary"})   # type: ignore[misc]
    assert isinstance(STAGE1B, ResearchAuthorization)
