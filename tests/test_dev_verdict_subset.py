"""The frozen DEV verdict-evaluable population — offline.

DEV holds 32 grade_primary cases but only 26 carry a derivable explanation
verdict in a class that has ground-truth support. Running the raw split would
pay for 6 cases with no ground truth and quietly change the denominator of
every metric, so the population is pre-registered and hash-verified exactly
like a smoke subset.

No provider is contacted anywhere in this file.
"""

from __future__ import annotations

import json

import pytest

from autograder.benchmark.manifests import load_manifest
from autograder.benchmark.subsets import (
    DEFAULT_SUBSET_ROOT,
    SubsetError,
    freeze_subset,
    load_subset,
    propose_subset,
    subset_case_ids,
    subset_path,
)

ROLE, NAME = "grade_primary", "dev_verdict"


@pytest.fixture(scope="module")
def manifest():
    return load_manifest(ROLE)


@pytest.fixture(scope="module")
def frozen(manifest):
    try:
        return load_subset(ROLE, NAME, manifest)
    except SubsetError as e:
        pytest.skip(f"not frozen in this checkout: {e}")


# ------------------------------------------------------------- the selection ----


def test_it_is_the_26_derivable_dev_cases(frozen):
    assert frozen["case_count"] == 26
    assert len(frozen["cases"]) == 26


def test_the_class_distribution_is_22_valid_and_4_partially_valid(frozen):
    assert frozen["class_distribution"] == {"valid": 22, "partially_valid": 4}


def test_the_invalid_class_is_absent_by_construction(frozen):
    """Not an oversight: no authoritative invalid example exists in any split."""
    assert "invalid" not in frozen["class_distribution"]
    assert all(c["verdict"] in ("valid", "partially_valid") for c in frozen["cases"])


def test_every_case_is_dev(manifest, frozen):
    by_id = {c.case_id: c for c in manifest.cases}
    for row in frozen["cases"]:
        assert row["split"] == "DEV"
        assert by_id[row["case_id"]].split == "DEV"


def test_no_calibration_or_held_out_leaks_in(manifest, frozen):
    ids = {c["case_id"] for c in frozen["cases"]}
    for c in manifest.cases:
        if c.split in ("CALIBRATION", "HELD_OUT"):
            assert c.case_id not in ids, f"{c.case_id} ({c.split}) is in a DEV population"


def test_every_excluded_case_has_a_recorded_reason(frozen):
    assert len(frozen["excluded"]) == 6
    for row in frozen["excluded"]:
        assert row["reason"], row
    # all six are the wrong-selection zeros
    assert all(r["reason"] == "zero_because_selection_wrong_explanation_never_scored"
               for r in frozen["excluded"])


def test_selection_plus_exclusions_account_for_the_whole_split(manifest, frozen):
    dev = manifest.by_split("DEV")
    assert len(frozen["cases"]) + len(frozen["excluded"]) == len(dev) == 32


# ------------------------------------------------------------- provenance ----


def test_it_records_what_it_was_frozen_against(frozen):
    for key in ("manifest_hashes", "git_commit", "prompt_version", "adapter_version",
                "candidate_configs", "frozen_at", "rules_version", "why"):
        assert frozen.get(key), key
    assert frozen["prompt_version"] == "grade-v3"
    assert frozen["adapter_version"] == "grade-bench-v2"


def test_it_records_the_frozen_candidate_configuration(frozen):
    cfgs = frozen["candidate_configs"]
    assert cfgs["google/gemini-3.7-flash"]["reasoning"] == {"effort": "low"}
    assert cfgs["google/gemini-3.7-flash"]["max_tokens"] == 1200
    for slug in ("openai/gpt-5.6-luna-pro", "anthropic/claude-sonnet-5"):
        assert cfgs[slug]["reasoning"] == {"effort": "none"}
        assert cfgs[slug]["max_tokens"] == 600


def test_every_row_carries_its_label_provenance(frozen):
    for row in frozen["cases"]:
        assert row["verdict_reason"] in (
            "full_credit_implies_valid", "partial_credit_implies_partially_valid")
        assert row["instructor_final_score"] in (2.0, 4.0)
        assert row["writer"] and row["question_id"] and row["sub_item_id"]


def test_the_manifest_hashes_match_the_live_dataset(manifest, frozen):
    if frozen["manifest_hashes"] == dict(manifest.hashes):
        return
    # the dataset moved after the subset froze; that is acceptable ONLY when
    # the manifest's owner-confirmed revision chain explains the walk from the
    # frozen hashes to the live ones (e.g. the 2026-09-01 row transposition)
    from prerepair import chain_end, manifest as live_manifest
    revs = [r for r in live_manifest()["revisions"] if r.get("owner_confirmed")]
    assert chain_end(revs, frozen["manifest_hashes"]["inputs_sha256"], "inputs") \
        == manifest.hashes["inputs_sha256"]
    assert chain_end(revs, frozen["manifest_hashes"]["labels_sha256"], "labels") \
        == manifest.hashes["labels_sha256"]


# ------------------------------------------------------------- the guards ----


def test_a_tampered_file_is_refused(tmp_path, manifest, frozen):
    d = json.loads(json.dumps(frozen))
    d["cases"][0]["case_id"] = "e999_q9_r9"
    root = tmp_path / "subsets"
    root.mkdir()
    (root / f"{ROLE}__{NAME}.json").write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(SubsetError, match="hash mismatch"):
        load_subset(ROLE, NAME, manifest, root)


def test_a_verdict_that_changed_since_freezing_is_refused(tmp_path, manifest, frozen):
    """If a relabelling moved a case's ground truth, the frozen population is
    no longer describing the same experiment."""
    from autograder.benchmark.subsets import _selection_hash

    d = json.loads(json.dumps(frozen))
    row = next(c for c in d["cases"] if c["verdict"] == "valid")
    row["verdict"] = "partially_valid"
    d["selection_sha256"] = _selection_hash(d["cases"])     # attacker fixes the hash
    root = tmp_path / "subsets"
    root.mkdir()
    (root / f"{ROLE}__{NAME}.json").write_text(json.dumps(d), encoding="utf-8")
    with pytest.raises(SubsetError, match="verdict changed"):
        load_subset(ROLE, NAME, manifest, root)


def test_freezing_refuses_to_overwrite(tmp_path, manifest):
    freeze_subset(ROLE, NAME, manifest, tmp_path, expect_count=26)
    with pytest.raises(SubsetError, match="never re-selected"):
        freeze_subset(ROLE, NAME, manifest, tmp_path, expect_count=26)


def test_freezing_refuses_a_population_that_does_not_match_the_plan(tmp_path, manifest):
    """A population that silently changed size between plan and freeze is
    exactly what a pre-registration exists to prevent."""
    with pytest.raises(SubsetError, match="expected 25 cases"):
        freeze_subset(ROLE, NAME, manifest, tmp_path, expect_count=25)


def test_freezing_refuses_a_distribution_that_does_not_match_the_plan(tmp_path, manifest):
    with pytest.raises(SubsetError, match="class distribution"):
        freeze_subset(ROLE, NAME, manifest, tmp_path, expect_count=26,
                      expect_distribution={"valid": 20, "partially_valid": 6})


def test_the_proposal_is_reproducible_from_the_dataset_alone(manifest, frozen):
    fresh = propose_subset(ROLE, NAME, manifest)
    assert fresh["selection_sha256"] == frozen["selection_sha256"]
    assert [c["case_id"] for c in fresh["cases"]] == [c["case_id"] for c in frozen["cases"]]


def test_an_unknown_subset_name_is_refused(manifest):
    with pytest.raises(SubsetError, match="no subset rule"):
        propose_subset(ROLE, "does_not_exist", manifest)


# --------------------------------------------------------------- the runner ----


def test_the_runner_resolves_the_same_ids(manifest, frozen):
    assert sorted(subset_case_ids(ROLE, NAME, manifest)) == \
        sorted(c["case_id"] for c in frozen["cases"])


def test_the_runner_refuses_the_subset_on_a_non_dev_split(manifest):
    from autograder.benchmark.runner import RunSpec, run_benchmark

    spec = RunSpec(role=ROLE, split="calibration", subset=NAME, candidate="vendor/m",
                   dry_run=True, allow_unlisted=True)
    with pytest.raises(ValueError, match="DEV-only"):
        run_benchmark(spec, manifest=manifest)


def test_smoke_and_dev_verdict_are_different_populations(manifest, frozen):
    """The two-case smoke manifest must never stand in for the full DEV run."""
    from autograder.benchmark.smoke import smoke_case_ids

    smoke = set(smoke_case_ids(ROLE, manifest))
    full = set(subset_case_ids(ROLE, NAME, manifest))
    assert smoke < full          # proper subset
    assert len(smoke) == 2 and len(full) == 26
