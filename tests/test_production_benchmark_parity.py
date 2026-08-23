"""The benchmark must select the model production will actually run.

GRADE_PRIMARY is chosen by running `GradeAdapter` over the benchmark cases, and
then deployed as the `grade_primary` route that `reliability.py` calls through
`escalate_grade`. If those two build different requests, the campaign measures
one implementation and the deployment runs another — the model with the best
benchmark number is not necessarily the model that behaves best in production,
and nothing in the pipeline would report the discrepancy.

These tests pin the equivalence at the seam where it can silently drift: the
system prompt, the content blocks, the response schema, the prompt version and
the output-token budget. They are offline — a mock gateway records what
production WOULD send; no provider is contacted.

The legacy judge (`grade.judge_all`, `--grading-mode legacy`) is a different
implementation with a different schema and is NOT what the campaign selects;
that is asserted here too, so the distinction stays explicit in the test suite
rather than living only in docs/model-roles.md.
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

from autograder.benchmark.roles import GradeAdapter, pack_from_inputs
from autograder.escalation import GRADE_SYSTEM, GradeResult, escalate_grade
from tests.test_escalation import _gw, _pack

REPO = Path(__file__).resolve().parents[1]


class _Recorder:
    """A gateway stand-in that captures the call instead of making it."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def call(self, **kw):
        self.calls.append(dict(kw))
        return self.inner.call(**kw)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def _real_case() -> dict:
    """The FIRST GRADE_PRIMARY case as it is stored — the exact input shape the
    campaign will send, not a hand-built approximation."""
    line = (REPO / "evaluation" / "model_selection" / "datasets" / "grade_primary"
            / "cases_inputs.jsonl").read_text(encoding="utf-8").splitlines()[0]
    return json.loads(line)


def _production_call(pack, *, selected, transcription, version):
    gw, _ = _gw({"grade_primary": [GradeResult(score=4)], "grade_escalate": [GradeResult(score=4)]})
    rec = _Recorder(gw)
    escalate_grade(pack=pack, selected=selected, transcription=transcription, version=version,
                   selection_correct=True, gateway=rec)
    assert rec.calls, "production made no grade_primary call"
    return rec.calls[0]


def test_the_benchmark_sends_the_request_production_sends():
    """Same pack, same selection, same transcription -> same bytes on the wire."""
    inputs = _real_case()
    pack = pack_from_inputs(inputs["pack"])          # what BOTH sides grade against
    bench = GradeAdapter().build_request(dict(inputs), REPO)
    prod = _production_call(pack, selected=inputs.get("selected"),
                            transcription=inputs["transcription"], version=inputs.get("version"))

    assert bench.system == prod["system"] == GRADE_SYSTEM
    assert bench.content_blocks == prod["content_blocks"], (
        "the benchmark and production build different grading prompts — the campaign would "
        "select a model on one prompt and deploy it on another")
    assert bench.output_model is prod["output_model"] is GradeResult
    assert prod["task"] == "grade_primary" == GradeAdapter().task


def test_parity_holds_for_a_synthetic_pack_too():
    """The real case has selected=None today; cover a correct selection as well,
    so the parity is not an accident of one case's shape."""
    pack = _pack()
    d = {"question_id": pack.question_id, "question_text": pack.question_text,
         "question_type": pack.question_type, "max_score": pack.max_score,
         "correct_by_version": pack.correct_by_version, "rubric": list(pack.rubric),
         "scoring_rules": list(pack.scoring_rules), "grading_policy": pack.grading_policy,
         "official_solution": dict(pack.official_solution),
         "rubric_items": [ri.__dict__ if not hasattr(ri, "model_dump") else ri.model_dump()
                          for ri in pack.rubric_items],
         "evidence_policy": pack.evidence_policy, "score_granularity": pack.score_granularity}
    inputs = {"case_id": "c1", "pack": d, "selected": "F",
              "transcription": "ניתן לראות שהתדרים הגבוהים נשמרים", "version": "A1"}
    bench = GradeAdapter().build_request(inputs, REPO)
    prod = _production_call(pack_from_inputs(d), selected="F",
                            transcription=inputs["transcription"], version="A1")
    assert bench.content_blocks == prod["content_blocks"]
    assert bench.system == prod["system"]


def test_the_task_name_matches_so_the_selected_route_is_the_deployed_route():
    assert GradeAdapter("grade_primary").task == "grade_primary"
    assert GradeAdapter("grade_escalate").task == "grade_escalate"


def _model_configs():
    """Every models.toml that ships or runs here.

    The tracked template is the authority — the live `models.toml` is gitignored
    (it holds machine-local slugs and the mutable [pricing] table), so a test
    that only read the live file would pass locally and never run on a fresh
    clone.
    """
    out = {"models.example.toml": tomllib.loads(
        (REPO / "models.example.toml").read_text(encoding="utf-8"))}
    live = REPO / "models.toml"
    if live.exists():
        out["models.toml"] = tomllib.loads(live.read_text(encoding="utf-8"))
    return out


def test_the_output_budget_is_the_same_in_the_campaign_and_in_production():
    """A cap difference is invisible until a long answer truncates.

    Production calls `gateway.call` WITHOUT max_tokens, so the backend falls
    back to the route's configured value (`max_tokens or self.config.max_tokens`);
    the benchmark passes `Request.max_tokens` explicitly. Those two numbers come
    from different files and drifted apart once already (300 vs 600).
    """
    for name, cfg in _model_configs().items():
        for task in ("grade_primary", "grade_escalate"):
            configured = cfg["models"][task]["max_tokens"]
            assert configured == GradeAdapter.default_max_tokens, (
                f"{name} [{task}].max_tokens={configured} but the benchmark grants "
                f"{GradeAdapter.default_max_tokens}: the campaign would select on one output budget "
                "and production would run on another")


def test_the_prompt_version_is_the_same_in_the_campaign_and_in_production():
    for name, cfg in _model_configs().items():
        assert cfg["models"]["grade_primary"]["prompt_version"] == GradeAdapter.prompt_version, name


def test_the_campaign_does_not_select_the_legacy_judge():
    """`--grading-mode legacy` grades with a different schema entirely.

    It is still the CLI default, so this is the one production route the
    benchmark result must not be read as covering.
    """
    from autograder.grade import ExplanationEvaluation

    assert ExplanationEvaluation is not GradeResult
    assert set(ExplanationEvaluation.model_fields) != set(GradeResult.model_fields)
    src = (REPO / "autograder" / "grade.py").read_text(encoding="utf-8")
    assert "gateway" not in src.split("def judge_all")[1][:2000], (
        "judge_all now goes through the gateway; the legacy/gateway distinction this test "
        "records has changed and the model-selection docs need updating")


def test_production_still_reaches_the_gateway_through_escalate_grade():
    """Guards the wiring the parity argument rests on: reliability mode calls
    `escalate_grade`, which calls `gateway.call(task='grade_primary')`."""
    src = (REPO / "autograder" / "reliability.py").read_text(encoding="utf-8")
    assert "escalate_grade(" in src
    esc = (REPO / "autograder" / "escalation.py").read_text(encoding="utf-8")
    assert 'gateway.call(task=primary_task' in esc
