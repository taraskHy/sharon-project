"""Strict-provider-compatible JSON Schema.

OpenAI-family structured output (and Azure's hosting of it, which is what
OpenRouter routed ``openai/gpt-5.6-luna-pro`` to on 2026-08-24) validates the
schema BEFORE running the model and rejects the request outright when it does
not meet the strict contract:

    Invalid schema for response_format 'GradeResult': In context=(),
    'additionalProperties' is required to be supplied and to be false.

That rejection is a pre-inference HTTP 400: no tokens, no charge, no result —
the candidate simply cannot be benchmarked. Pydantic's ``model_json_schema()``
never emits ``additionalProperties``, so EVERY object we send is invalid for
those providers.

The strict contract has two rules, and a schema that satisfies both is also
valid for permissive servers (Ollama, vLLM, TGI), which simply ignore the
extra constraints:

1. every object node carries ``additionalProperties: false``;
2. every declared property is listed in ``required`` — optionality is
   expressed by admitting ``null`` in the type, never by omission.

This module implements both, once, for any schema — it is deliberately NOT a
patch to one generated literal, because the same defect exists in every model
we send (GradeResult, RubricItemGrade, the OCR and MC schemas, and anything
added later).
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

#: Keys whose value is itself a schema.
_SCHEMA_VALUED = ("items", "contains", "not", "if", "then", "else",
                  "propertyNames", "additionalItems", "unevaluatedItems")
#: Keys whose value is a LIST of schemas.
_SCHEMA_LIST_VALUED = ("anyOf", "oneOf", "allOf", "prefixItems")
#: Keys whose value is a MAPPING of name -> schema.
_SCHEMA_MAP_VALUED = ("properties", "$defs", "definitions", "patternProperties")


def _is_object_node(node: dict) -> bool:
    """True when this node describes a JSON object.

    ``type`` may be a string or a list (``["object", "null"]``). A node with
    ``properties`` but no ``type`` is still an object — pydantic emits that
    shape for some nested models.
    """
    t = node.get("type")
    if isinstance(t, str):
        return t == "object"
    if isinstance(t, list):
        return "object" in t
    return "properties" in node


def _walk(node: Any, *, require_all: bool) -> Any:
    if isinstance(node, list):
        return [_walk(n, require_all=require_all) for n in node]
    if not isinstance(node, dict):
        return node

    out = dict(node)

    for key in _SCHEMA_MAP_VALUED:
        if isinstance(out.get(key), dict):
            out[key] = {k: _walk(v, require_all=require_all) for k, v in out[key].items()}
    for key in _SCHEMA_LIST_VALUED:
        if isinstance(out.get(key), list):
            out[key] = [_walk(v, require_all=require_all) for v in out[key]]
    for key in _SCHEMA_VALUED:
        if isinstance(out.get(key), (dict, list)):
            out[key] = _walk(out[key], require_all=require_all)
    # additionalProperties may itself be a schema; only a bare True/absent
    # value is the permissive default we must close.
    if isinstance(out.get("additionalProperties"), dict):
        out["additionalProperties"] = _walk(out["additionalProperties"],
                                            require_all=require_all)

    if _is_object_node(out):
        # Rule 1 — close the object. An explicit sub-schema is left alone:
        # the author asked for constrained extra properties on purpose.
        if not isinstance(out.get("additionalProperties"), dict):
            out["additionalProperties"] = False
        if require_all:
            # Rule 2 — every property is required; a property that used to be
            # optional (absent from `required`, or carrying a default) becomes
            # explicitly nullable so "not supplied" stays expressible.
            props = out.get("properties")
            if isinstance(props, dict) and props:
                already = list(out.get("required") or [])
                missing = [k for k in props if k not in already]
                for name in missing:
                    props[name] = _make_nullable(props[name])
                out["required"] = already + missing
    return out


def _make_nullable(schema: Any) -> Any:
    """Admit ``null`` without disturbing the rest of the sub-schema."""
    if not isinstance(schema, dict):
        return schema
    s = dict(schema)
    if "anyOf" in s and isinstance(s["anyOf"], list):
        if not any(isinstance(b, dict) and b.get("type") == "null" for b in s["anyOf"]):
            s["anyOf"] = list(s["anyOf"]) + [{"type": "null"}]
        return s
    t = s.get("type")
    if isinstance(t, str):
        if t != "null":
            s["type"] = [t, "null"]
        return s
    if isinstance(t, list):
        if "null" not in t:
            s["type"] = list(t) + ["null"]
        return s
    if "$ref" in s:
        # a $ref cannot carry a sibling type; wrap it instead
        ref = s.pop("$ref")
        s["anyOf"] = [{"$ref": ref}, {"type": "null"}]
        return s
    return s


def strict_json_schema(schema: dict, *, require_all: bool = True) -> dict:
    """Return ``schema`` rewritten to satisfy strict structured-output providers.

    Pure function — the input is never mutated. ``require_all=False`` applies
    only rule 1 (``additionalProperties: false``), for servers that enforce
    the closed-object rule but are happy with genuinely optional fields.
    """
    return _walk(deepcopy(schema), require_all=require_all)


def schema_violations(schema: dict, *, require_all: bool = True) -> list[str]:
    """Every strict-contract breach in ``schema``, as human-readable paths.

    Empty list == the schema is acceptable to a strict provider. Used by the
    offline conformance tests so a newly added model cannot reintroduce the
    defect unnoticed.
    """
    problems: list[str] = []

    def visit(node: Any, path: str) -> None:
        if isinstance(node, list):
            for i, n in enumerate(node):
                visit(n, f"{path}[{i}]")
            return
        if not isinstance(node, dict):
            return
        if _is_object_node(node):
            ap = node.get("additionalProperties")
            if ap is not False and not isinstance(ap, dict):
                problems.append(f"{path or '<root>'}: additionalProperties is not false")
            if require_all and isinstance(node.get("properties"), dict):
                required = set(node.get("required") or [])
                for name in node["properties"]:
                    if name not in required:
                        problems.append(f"{path or '<root>'}.{name}: not in required")
        for key in _SCHEMA_MAP_VALUED:
            if isinstance(node.get(key), dict):
                for k, v in node[key].items():
                    visit(v, f"{path}.{key}.{k}" if path else f"{key}.{k}")
        for key in _SCHEMA_LIST_VALUED:
            if isinstance(node.get(key), list):
                visit(node[key], f"{path}.{key}" if path else key)
        for key in _SCHEMA_VALUED:
            if isinstance(node.get(key), (dict, list)):
                visit(node[key], f"{path}.{key}" if path else key)
        if isinstance(node.get("additionalProperties"), dict):
            visit(node["additionalProperties"],
                  f"{path}.additionalProperties" if path else "additionalProperties")

    visit(schema, "")
    return problems


__all__ = ["strict_json_schema", "schema_violations"]
