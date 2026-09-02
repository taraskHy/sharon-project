"""A provider refusal is not a schema failure.

``score()`` has always been handed the error string and ignored it, labelling
every no-output case ``schema_failure``. The OCR Stage-1c arm showed what that
costs: its ``metrics.json`` reported ``schema_failures: 3`` when all three
losses were provider content-filter refusals and the model's structured output
had been valid on every request it was allowed to answer. Anyone reading that
file would have concluded the candidate could not hold a JSON schema — the
opposite of what happened.

Scored metrics are untouched by the fix (a case with no output is unscored
either way), so ``adapter_version`` is deliberately NOT bumped: bumping it would
change every config hash and break comparability with the frozen Stage-1/1b runs
over a naming correction.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from autograder.benchmark.manifests import load_manifest
from autograder.benchmark.roles import adapter_for, classify_no_output

S1C = Path("evaluation/model_selection/runs_stage1c/ocr_primary/"
           "dev__smoke__all__google-gemini-3.7-flash__45297cdd83")
S1B = Path("evaluation/model_selection/runs/ocr_primary/"
           "dev__smoke__all__google-gemini-3.7-flash__45297cdd83")
STAGE1_GEMINI = Path("evaluation/model_selection/runs/ocr_primary/"
                     "dev__smoke__all__google-gemini-3.7-flash__feceaa6084")


@pytest.mark.parametrize("error,schema,provider", [
    ("the backend refused this request (content_filter)", False, True),
    ("HTTP 400 from backend: Reasoning is mandatory for this endpoint", False, True),
    ("output was truncated at max_tokens=600 (finish_reason=length)", False, True),
    ("backend unreachable after 3 attempts: timeout", False, True),
    ("model output failed BenchTranscription validation after 1 attempt(s)", True, False),
    ("Invalid JSON: EOF while parsing a string at line 1 column 46", True, False),
    (None, True, False),
    ("", True, False),
])
def test_classification(error, schema, provider):
    assert classify_no_output(error) == (schema, provider)


def test_a_refusal_and_a_schema_failure_are_never_both_true():
    for e in ("content_filter", "validation failed", None, "HTTP 500"):
        s, p = classify_no_output(e)
        assert not (s and p)
        assert s or p, "a no-output case must be attributed to something"


@pytest.fixture(scope="module")
def manifest():
    return load_manifest("ocr_primary")


def _score_run(d, manifest):
    if not (d / "outputs.jsonl").exists():
        pytest.skip(f"{d} not present")
    ad = adapter_for("ocr_primary")
    by = {c.case_id: c for c in manifest.cases}
    rows = [json.loads(l) for l in (d / "outputs.jsonl").read_text(encoding="utf-8").splitlines()
            if l.strip()]
    return [ad.score(by[r["case_id"]], r.get("output") if r.get("ok") else None, r.get("error"))
            for r in rows], rows


def test_stage1c_content_filter_refusals_are_provider_failures(manifest):
    """The concrete regression: 3 refusals, 0 schema failures."""
    scored, _ = _score_run(S1C, manifest)
    assert sum(1 for r in scored if r["provider_failure"]) == 3
    assert sum(1 for r in scored if r["schema_failure"]) == 0
    assert sum(1 for r in scored if r["no_output"]) == 3
    assert {r["skip_reason"] for r in scored if r["no_output"]} == {"provider_failure"}


def test_stage1c_aggregate_no_longer_claims_schema_failures(manifest):
    scored, rows = _score_run(S1C, manifest)
    agg = adapter_for("ocr_primary").aggregate(scored, rows)
    assert agg["schema_failures"] == 0, "Gemini held the schema on every answered request"
    assert agg["provider_failures"] == 3
    assert agg["no_output_cases"] == 3


def test_stage1_gemini_http400s_are_provider_failures_not_schema_failures(manifest):
    """The Stage-1 arm was rejected before inference; the model never spoke."""
    scored, _ = _score_run(STAGE1_GEMINI, manifest)
    assert sum(1 for r in scored if r["provider_failure"]) == 8
    assert sum(1 for r in scored if r["schema_failure"]) == 0


def test_stage1b_truncations_are_provider_failures_and_the_json_eof_is_schema(manifest):
    """Stage-1b lost 2 cases to the cap and 1 to unparseable output — different causes."""
    scored, _ = _score_run(S1B, manifest)
    assert sum(1 for r in scored if r["provider_failure"]) == 2
    assert sum(1 for r in scored if r["schema_failure"]) == 1


def test_scored_metrics_are_unchanged_by_the_taxonomy_fix(manifest):
    """CER/usable rates must be identical to the committed run metrics — the fix
    renames failures, it does not rescore anything."""
    scored, rows = _score_run(S1C, manifest)
    agg = adapter_for("ocr_primary").aggregate(scored, rows)
    committed = json.loads((S1C / "metrics.json").read_text(encoding="utf-8"))
    assert agg["overall"]["mean_cer"] == committed["overall"]["mean_cer"]
    assert agg["overall"]["median_cer"] == committed["overall"]["median_cer"]
    assert agg["overall"]["cases"] == committed["overall"]["cases"]


def test_successful_cases_carry_no_failure_label(manifest):
    scored, _ = _score_run(S1C, manifest)
    for r in scored:
        if r["scored"]:
            assert not r["schema_failure"] and not r["provider_failure"] and not r["no_output"]
