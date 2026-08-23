"""An unpriced model must not be able to spend money.

`predicted_call_cost` returns 0.0 for a model missing from the local [pricing]
table. That silence is the failure: every pre-call budget check would pass, and
the $10 ceiling could only react AFTER the provider had already charged. So a
live cloud run of an unpriced candidate is refused outright.

No model, network or OCR calls.
"""
from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from autograder.benchmark.runner import UnpricedCandidate, require_priced_candidate
from autograder.usage import predicted_call_cost

REPO = Path(__file__).resolve().parents[1]


class _Route:
    def __init__(self, model, max_tokens=600):
        self.model, self.max_tokens = model, max_tokens


def test_an_unpriced_model_estimates_zero_which_is_why_it_must_not_run():
    """The premise: this is exactly how the ceiling would be bypassed."""
    assert predicted_call_cost(_Route("who/knows"), "x" * 8000, [], {"other/model": {"input": 1, "output": 1}}) == 0.0
    assert predicted_call_cost(_Route("who/knows"), "x" * 8000, [], None) == 0.0


@pytest.mark.parametrize("pricing", [
    None, {}, {"other/model": {"input": 1.0, "output": 2.0}},
    {"m": {}}, {"m": {"input": 0, "output": 2.0}}, {"m": {"input": 2.0, "output": 0}},
    {"m": "not-a-dict"},
])
def test_every_shape_of_missing_price_is_refused(pricing):
    with pytest.raises(UnpricedCandidate, match="no usable entry"):
        require_priced_candidate("m", pricing)


def test_a_priced_model_is_allowed():
    assert require_priced_candidate("m", {"m": {"input": 0.2, "output": 1.2}}) is None


def test_the_refusal_names_the_fix():
    with pytest.raises(UnpricedCandidate) as e:
        require_priced_candidate("vendor/slug", {})
    msg = str(e.value)
    assert "models.toml" in msg and "--models-config" in msg and "vendor/slug" in msg


def test_every_configured_grading_candidate_is_priced():
    """The campaign cannot start with a candidate this machine cannot price."""
    pricing = tomllib.loads((REPO / "models.toml").read_text(encoding="utf-8")).get("pricing") or {}
    cands = tomllib.loads((REPO / "evaluation" / "model_selection" / "candidates.toml").read_text(encoding="utf-8"))
    for role in ("grade_primary", "grade_escalate"):
        for slug in cands["roles"][role]["candidates"]:
            require_priced_candidate(slug, pricing)          # raises if any is unpriced


def test_the_prices_are_positive_and_output_costs_at_least_input():
    """A transcription slip that swapped or zeroed a column would silently
    shrink every estimate; output is never cheaper than input for these."""
    pricing = tomllib.loads((REPO / "models.toml").read_text(encoding="utf-8"))["pricing"]
    assert len(pricing) >= 5
    for slug, p in pricing.items():
        assert p["input"] > 0 and p["output"] > 0, slug
        assert p["output"] >= p["input"], slug


def test_the_estimate_is_a_real_number_for_a_real_grade_primary_request():
    """End to end: a priced candidate produces a non-zero dollar figure for the
    actual benchmark request, which is what the ceiling needs."""
    dataset = REPO / "evaluation" / "model_selection" / "datasets" / "grade_primary"
    if not (dataset / "manifest.json").exists():
        pytest.skip("grade_primary dataset not built")
    import json
    from autograder.benchmark.roles import GradeAdapter
    rows = [json.loads(l) for l in (dataset / "cases_inputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    labels = {json.loads(l)["case_id"]: json.loads(l)
              for l in (dataset / "cases_labels.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    ad = GradeAdapter("grade_primary")
    req = ad.build_request(rows[0], labels[rows[0]["case_id"]])
    pricing = tomllib.loads((REPO / "models.toml").read_text(encoding="utf-8"))["pricing"]
    for slug in pricing:
        cost = predicted_call_cost(_Route(slug, ad.default_max_tokens), req.system, req.content_blocks, pricing)
        assert cost > 0, slug
        assert cost < 0.05, f"{slug}: a single grading call should be well under 5 cents, got {cost}"
