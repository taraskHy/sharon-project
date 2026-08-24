"""Strict structured-output schema conformance — offline.

On 2026-08-24 every ``openai/gpt-5.6-luna-pro`` call of the GRADE_PRIMARY
smoke run died with a pre-inference HTTP 400:

    Invalid schema for response_format 'GradeResult': In context=(),
    'additionalProperties' is required to be supplied and to be false.

The candidate produced zero tokens and could not be evaluated at all. The
defect was not in one hand-written literal — pydantic never emits
``additionalProperties``, so EVERY output model we send was invalid for
strict providers. These tests pin the central fix and audit every model that
actually goes out over a structured-output request.

No provider is contacted anywhere in this file.
"""

from __future__ import annotations

import json
from typing import Optional

import httpx
import pytest
from pydantic import BaseModel, Field

from autograder.backends import BackendConfig
from autograder.backends.openai_compat import OpenAICompatBackend
from autograder.strictschema import schema_violations, strict_json_schema


# ------------------------------------------------------- the real models ----


#: every class named at an ``output_model=`` call site, with its module
_OUTPUT_MODELS = {
    "AnswerKey": "autograder.schema",
    "BandRowExtraction": "autograder.schema",
    "BenchTranscription": "autograder.benchmark.roles",
    "ExamSurvey": "autograder.schema",
    "ExplanationJudgement": "autograder.schema",
    "ExplanationTranscription": "autograder.schema",
    "GradeProbe": "autograder.evalcli",
    "GradeResult": "autograder.escalation",
    "MCRead": "autograder.mcresolve",
    "MarkDisambiguation": "autograder.schema",
    "MarkerCatalog": "autograder.discovery",
    "OCRVerifyResult": "autograder.escalation",
    "PermutationProposal": "autograder.alignment",
    "PolicyInference": "autograder.discovery",
    "QuestionExtraction": "autograder.schema",
    "RubricItemGrade": "autograder.escalation",
    "SheetCloseRead": "autograder.schema",
    "VariantAlignment": "autograder.schema",
    "VariantDetection": "autograder.schema",
}


def _output_models():
    """Every Pydantic model handed to a backend as ``output_model``.

    The audit is over ALL of them, not just the one that happened to fail:
    the defect is in the generator, so every model shares it.
    """
    import importlib

    out = {}
    for name, module in _OUTPUT_MODELS.items():
        out[name] = getattr(importlib.import_module(module), name)
    return out


@pytest.mark.parametrize("name", sorted(_output_models()))
def test_every_output_model_is_strict_compatible_after_transform(name):
    model = _output_models()[name]
    fixed = strict_json_schema(model.model_json_schema())
    assert schema_violations(fixed) == [], f"{name} still violates the strict contract"


def test_graderesult_specifically_closes_root_and_nested_objects():
    """The exact schema that Luna rejected."""
    from autograder.escalation import GradeResult

    raw = GradeResult.model_json_schema()
    before = schema_violations(raw)
    assert any("additionalProperties" in v for v in before), (
        "regression guard: pydantic is expected to emit an OPEN schema, which "
        "is precisely why the transform must exist")

    fixed = strict_json_schema(raw)
    assert fixed["additionalProperties"] is False                      # root
    assert fixed["$defs"]["RubricItemGrade"]["additionalProperties"] is False   # nested
    # rubric items live behind a $ref inside an array — the nested object is
    # reached through $defs, and every property of it is now required
    assert set(fixed["$defs"]["RubricItemGrade"]["required"]) == {
        "id", "met", "student_evidence"}
    assert set(fixed["required"]) == {
        "score", "rubric_items", "rubric_items_met", "uncertain", "evidence"}


def test_optional_fields_become_explicitly_nullable_not_dropped():
    """Optionality must survive as an admissible null, never as omission."""
    from autograder.escalation import GradeResult

    fixed = strict_json_schema(GradeResult.model_json_schema())
    ev = fixed["properties"]["evidence"]
    assert {"type": "null"} in ev["anyOf"], "Optional[str] must still accept null"
    items = fixed["properties"]["rubric_items"]
    assert "null" in items["type"], "a defaulted list must still accept null"


# ---------------------------------------------------- transform mechanics ----


def test_transform_is_pure_and_does_not_mutate_input():
    raw = {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]}
    snapshot = json.dumps(raw, sort_keys=True)
    strict_json_schema(raw)
    assert json.dumps(raw, sort_keys=True) == snapshot


def test_recurses_through_defs_arrays_and_unions():
    schema = {
        "type": "object",
        "properties": {
            "rows": {"type": "array", "items": {"type": "object",
                                                "properties": {"x": {"type": "integer"}},
                                                "required": ["x"]}},
            "choice": {"anyOf": [
                {"type": "object", "properties": {"y": {"type": "string"}}, "required": ["y"]},
                {"type": "null"},
            ]},
        },
        "required": ["rows", "choice"],
        "$defs": {"Nested": {"type": "object", "properties": {"z": {"type": "boolean"}},
                             "required": ["z"]}},
    }
    fixed = strict_json_schema(schema)
    assert fixed["additionalProperties"] is False
    assert fixed["properties"]["rows"]["items"]["additionalProperties"] is False
    assert fixed["properties"]["choice"]["anyOf"][0]["additionalProperties"] is False
    assert fixed["$defs"]["Nested"]["additionalProperties"] is False
    assert schema_violations(fixed) == []


def test_object_without_explicit_type_is_still_closed():
    """Pydantic emits property-only nodes for some nested models."""
    fixed = strict_json_schema({"properties": {"a": {"type": "string"}}, "required": ["a"]})
    assert fixed["additionalProperties"] is False


def test_nullable_object_type_list_is_treated_as_an_object():
    fixed = strict_json_schema(
        {"type": ["object", "null"], "properties": {"a": {"type": "string"}}, "required": ["a"]})
    assert fixed["additionalProperties"] is False


def test_deliberate_additionalproperties_subschema_is_preserved():
    """An author who constrained extra properties on purpose keeps them."""
    schema = {"type": "object", "properties": {}, "additionalProperties": {"type": "string"}}
    fixed = strict_json_schema(schema)
    assert fixed["additionalProperties"] == {"type": "string"}
    assert schema_violations(fixed) == []


def test_ref_valued_optional_is_wrapped_not_corrupted():
    schema = {"type": "object",
              "properties": {"child": {"$ref": "#/$defs/C"}},
              "required": [],
              "$defs": {"C": {"type": "object", "properties": {"a": {"type": "string"}},
                              "required": ["a"]}}}
    fixed = strict_json_schema(schema)
    child = fixed["properties"]["child"]
    assert child["anyOf"] == [{"$ref": "#/$defs/C"}, {"type": "null"}]
    assert "$ref" not in child, "a $ref must not keep siblings that strict validators reject"


def test_require_all_false_applies_only_the_closed_object_rule():
    raw = {"type": "object", "properties": {"a": {"type": "string"}}, "required": []}
    fixed = strict_json_schema(raw, require_all=False)
    assert fixed["additionalProperties"] is False
    assert fixed["required"] == []
    assert schema_violations(fixed, require_all=False) == []


# ------------------------------------------------- what goes over the wire ----


class _Out(BaseModel):
    score: float
    note: Optional[str] = Field(default=None)


def _backend(**cfg):
    return OpenAICompatBackend(
        BackendConfig(backend="openai", model="m", base_url="http://x/v1", **cfg),
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))


def test_request_payload_carries_the_strict_schema():
    b = _backend()
    payload = b._build_payload([{"role": "user", "content": "hi"}], _Out, 100)
    sent = payload["response_format"]["json_schema"]["schema"]
    assert schema_violations(sent) == []
    assert sent["additionalProperties"] is False


def test_prompt_copy_and_response_format_copy_are_identical():
    """The model must never be shown a different schema from the one the
    provider enforces."""
    b = _backend()
    payload = b._build_payload([{"role": "user", "content": "hi"}], _Out, 100)
    sent = payload["response_format"]["json_schema"]["schema"]
    assert b.schema_for(_Out) == sent


def test_strict_schema_can_be_disabled_for_a_server_that_rejects_it():
    b = _backend(strict_schema=False)
    assert b.schema_for(_Out) == _Out.model_json_schema()
    assert schema_violations(b.schema_for(_Out)) != []


def test_openrouter_inherits_the_strict_transform(monkeypatch):
    from autograder.backends.openrouter import OpenRouterBackend

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-not-a-real-key")
    b = OpenRouterBackend(
        BackendConfig(backend="openrouter", model="openai/gpt-5.6-luna-pro"),
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json={})))
    payload = b._build_payload([{"role": "user", "content": "hi"}], _Out, 100)
    assert schema_violations(payload["response_format"]["json_schema"]["schema"]) == []
