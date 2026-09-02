"""``ocr-neutral-v2`` is a ONE-VARIABLE change to ``m2-strict-v1``.

The experiment it serves (OCR_PROMPT_V2_NEUTRAL_FRAMING) is only interpretable
if exactly one thing differs between control and treatment: the framing
sentence. Everything that defines OCR behaviour — the exact-copy rule, the
unreadable markers, the struck-through protocol, the RTL ordering rule and the
JSON contract — must be carried over byte-for-byte.

These tests make that a property of the code rather than a claim in a document,
and they also pin the things a "neutral" rewrite could plausibly break: telling
the model to include annotations, weakening exact transcription, or changing the
output schema.
"""
from __future__ import annotations

import hashlib

import pytest

from autograder.benchmark.roles import (OCR_PROMPT_VERSIONS, _load_historical_prompts,
                                        adapter_for, load_ocr_prompts)
from autograder.cloudboundary import approved_cloud_ocr_systems

RULES_MARKER = "\nRules:"
CATS = ("handwritten_line", "handwritten_cell")


@pytest.fixture(scope="module")
def v1():
    return load_ocr_prompts("m2-strict-v1")


@pytest.fixture(scope="module")
def v2():
    return load_ocr_prompts("ocr-neutral-v2")


# ---- the one-variable guarantee ------------------------------------------

@pytest.mark.parametrize("cat", CATS)
def test_the_rules_block_is_byte_identical(cat, v1, v2):
    """Everything from 'Rules:' onward is carried over verbatim."""
    a = v1[cat][v1[cat].index(RULES_MARKER):]
    b = v2[cat][v2[cat].index(RULES_MARKER):]
    assert a == b
    assert hashlib.sha256(a.encode()).hexdigest() == hashlib.sha256(b.encode()).hexdigest()


@pytest.mark.parametrize("cat", CATS)
def test_only_the_first_line_differs(cat, v1, v2):
    a, b = v1[cat].split("\n"), v2[cat].split("\n")
    assert len(a) == len(b), "line count must not change"
    differing = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    assert differing == [0], f"expected only line 0 to differ, got {differing}"


@pytest.mark.parametrize("cat", CATS)
def test_the_full_ocr_contract_survives(cat, v2):
    """Every behavioural clause the campaign depends on."""
    t = v2[cat]
    for clause in (
        "Copy EXACTLY the visible text, word by word.",
        "Never complete, paraphrase, correct, or invent words.",
        "Fidelity over fluency.",
        "If a single word is unreadable, write [?] in its place.",
        "If everything is unreadable, output [unreadable].",
        "Text struck through by the writer is cancelled - skip it.",
        "Hebrew is written right-to-left; output the text in normal logical order",
        'Reply with ONLY a JSON object: {"transcription": "<the text>"}',
    ):
        assert clause in t, f"{cat}: lost {clause!r}"


@pytest.mark.parametrize("cat", CATS)
def test_the_exam_and_grading_framing_is_gone(cat, v2):
    """The single intended variable, from the other direction."""
    low = v2[cat].lower()
    for word in ("exam", "instructor", "red instructor", "answer sheet", "answer cell",
                 "grade", "grading", "marks are final", "score"):
        assert word not in low, f"{cat}: still frames the task with {word!r}"


@pytest.mark.parametrize("cat", CATS)
def test_v1_still_contains_the_framing_it_is_the_control_for(cat, v1):
    """If the control ever loses its framing, the experiment is meaningless."""
    low = v1[cat].lower()
    assert "exam" in low
    if cat == "handwritten_cell":
        assert "red instructor ink" in low


# ---- what a "neutral" rewrite could plausibly break -----------------------

def test_the_annotation_exclusion_instruction_is_preserved_not_dropped(v2):
    """'Ignore any red instructor ink' becomes a neutral description of the same
    visual fact. Dropping it outright would invite grading annotations into the
    transcription, which is what annotation_inclusion_error measures."""
    cell = v2["handwritten_cell"]
    assert "Ignore any marks written in a different colour of ink" in cell
    assert "ignore" in cell.lower()


@pytest.mark.parametrize("cat", CATS)
def test_the_prompt_never_asks_for_annotations_to_be_included(cat, v2):
    t = v2[cat].lower()
    for bad in ("include any marks", "transcribe the marks", "include annotations",
                "include corrections", "include checkmarks", "include the score"):
        assert bad not in t


@pytest.mark.parametrize("cat", CATS)
def test_exact_transcription_is_not_weakened(cat, v2):
    t = v2[cat].lower()
    for softener in ("summarize", "summarise", "clean up", "correct the spelling",
                     "fix errors", "if unclear, guess", "approximate", "best effort"):
        assert softener not in t


# ---- identity, registration and plumbing ----------------------------------

def test_v2_covers_only_the_categories_the_population_uses(v2):
    assert set(v2) == set(CATS), (
        "the 32-crop population is 16 line + 16 cell; v2 is deliberately scoped to "
        "those, and build_request raises loudly for anything else")


def test_an_out_of_scope_category_fails_loudly(v2):
    ad = adapter_for("ocr_primary", "ocr-neutral-v2")
    with pytest.raises(KeyError, match="ocr-neutral-v2"):
        ad.build_request({"case_id": "x", "image": "y.png", "category": "formula_printed"},
                         __import__("pathlib").Path("."))


def test_prompt_hashes_differ_from_the_control(v1, v2):
    """Different prompt bytes -> different request fingerprint -> the neutral run
    cannot silently reuse cached control responses."""
    for cat in CATS:
        assert hashlib.sha256(v1[cat].encode()).hexdigest() != \
            hashlib.sha256(v2[cat].encode()).hexdigest()


def test_both_versions_are_registered_cloud_ocr_prompts(v1, v2):
    approved = approved_cloud_ocr_systems()
    for cat in CATS:
        assert v1[cat] in approved
        assert v2[cat] in approved, "the boundary would refuse the treatment arm"


def test_adapter_pins_the_version_without_moving_adapter_version():
    """Scoring is unchanged, so adapter_version must NOT move — otherwise the
    treatment stops being comparable to the control."""
    control = adapter_for("ocr_primary")
    treat = adapter_for("ocr_primary", "ocr-neutral-v2")
    assert control.prompt_version == "m2-strict-v1"
    assert treat.prompt_version == "ocr-neutral-v2"
    assert control.adapter_version == treat.adapter_version == "ocr-primary-bench-v1"
    assert control.default_max_tokens == treat.default_max_tokens
    assert control.model_visible_fields == treat.model_visible_fields


def test_unknown_prompt_version_is_refused():
    with pytest.raises(ValueError):
        load_ocr_prompts("something-else")
    with pytest.raises(ValueError):
        adapter_for("ocr_primary", "ocr-neutral-v3")


def test_version_registry_is_explicit():
    assert OCR_PROMPT_VERSIONS == ("m2-strict-v1", "ocr-neutral-v2")


def test_the_frozen_v1_prompts_are_untouched():
    """The control must be byte-identical to what the paired run used."""
    import json
    from pathlib import Path
    contract = Path("evaluation/model_selection/runs/ocr_primary/"
                    "OCR_SMOKE_STAGE1_CONTRACT_EXEC_2026-09-02.json")
    if not contract.exists():
        pytest.skip("stage-1 contract artifact not present")
    recorded = json.loads(contract.read_text(encoding="utf-8"))["prompt_sha256_by_category"]
    now = {k: hashlib.sha256(v.encode()).hexdigest()
           for k, v in _load_historical_prompts().items()}
    assert now == recorded
