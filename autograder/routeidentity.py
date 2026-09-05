"""Canonical, secret-free identities for a route — derived, not hand-listed.

The V1 incident (2026-09-04) was caused by a hand-maintained field list.
``TaskRoute.fingerprint_fields()`` enumerated the route knobs that mattered and
simply did not mention ``provider``. A provider-pinned arm therefore produced
the *same* run id and the *same* request-cache key as the unpinned
configuration, and five of eight cases in a "pinned" arm were served from an
earlier UNPINNED run's cache. Appending one more name to that list would fix
the symptom and leave the mechanism — the next field someone forgets — intact.

So identity is derived from the EFFECTIVE backend configuration, after
``TaskRoute.to_backend_config()`` has folded in reasoning, provider routing and
every candidate override. Whatever actually reaches the wire is what identity is
computed from. A knob that changes the request but is never folded into the
effective config cannot silently escape, because there is no separate list to
forget it in.

Two identities, deliberately distinct
-------------------------------------

They answer different questions and must not be overloaded onto one hash:

``semantic_request_identity`` — *may this stored response be reused for this
request?* Covers everything that changes WHAT the provider returns: the served
model and route (including the provider pin and fallback policy), the decoding
configuration, the response-format contract, and the request content itself.

``experiment_identity`` — *is this the same experiment?* Adds what changes the
MEANING of a run without changing any single response: the operational retry
policy, the cache policy, the adapter and prompt versions, the dataset manifest
and the case set. Two runs can be semantically cache-compatible per request and
still be different experiments; the V1 incident is exactly that case.

Secrets and volatile values are excluded by construction: the API key is never
read (only ``api_key_env``, a variable NAME, exists on the config, and even that
is dropped), and no timestamp, counter or wall-clock value participates.

Versioning
----------

``CACHE_IDENTITY_VERSION`` is part of the hashed payload, so a key computed
under the old, incomplete scheme can never collide with a corrected one.
Historical cache entries stay on disk, untouched and simply unreachable by the
new keys — they are evidence of what previous runs did, and deleting them would
destroy that record.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any

#: Bumped when the identity DERIVATION changes.
#:   v1 — the historical TaskRoute.fingerprint_fields() list, which omitted
#:        `provider`; a pinned arm shared a key with the unpinned one.
#:   v2 — derived from the effective backend config, but digesting the RAW
#:        model_json_schema() rather than the schema actually transmitted.
#:   v3 — digests the CANONICAL WIRE SCHEMA: the response_format block as sent,
#:        including the strict transform and the schema name.
#:   v4 — canonicalises base_url to the EFFECTIVE endpoint, so an unset base_url
#:        and the explicitly spelled default hash identically.
CACHE_IDENTITY_VERSION = 4

#: Effective-config fields that change WHAT the provider returns.
#: `extra_generation` carries reasoning AND the provider routing object once
#: to_backend_config() has run, which is why derivation happens after it.
SEMANTIC_FIELDS = (
    "backend", "model", "base_url",
    "structured_mode", "strict_schema",
    "max_tokens", "temperature",
    "extra_generation",
)

#: Additional effective-config fields that change the EXPERIMENT but not the
#: content of any single successful response.
EXPERIMENT_ONLY_FIELDS = (
    "transport_retries", "validation_retries", "timeout_s",
)

#: Route-level fields that are NOT decoding knobs and therefore never reach
#: BackendConfig, but which DO define identity. `prompt_version` is the
#: important one: a version bump must invalidate a cached response even when
#: the rendered system text happens to be byte-identical, because the version
#: is the thing the experiment is comparing. Deriving purely from the effective
#: config silently dropped it — caught by test_changed_model_prompt_image_or_pack_invalidates.
ROUTE_LEVEL_FIELDS = ("task", "prompt_version")

#: Never part of any identity. `api_key_env` is only a variable NAME and the key
#: itself is never read here, but the name is dropped too so an identity can
#: never become a channel for credential configuration. `concurrency` is a local
#: scheduling detail that changes nothing about a request.
EXCLUDED_FIELDS = frozenset({"api_key_env", "api_key", "concurrency"})


class IdentityError(RuntimeError):
    pass


def _h(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def _canonical(value: Any) -> Any:
    """Order-independent canonical form, so two equivalent configurations
    built by different paths hash identically."""
    if isinstance(value, dict):
        return {k: _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        # provider `order` is SEMANTICALLY ORDERED — preference order changes
        # which provider serves the request — so sequences are NOT sorted.
        return [_canonical(v) for v in value]
    return value


#: Backend -> the endpoint actually used when base_url is left unset. Identity
#: must key on the EFFECTIVE endpoint, not on whether the caller happened to
#: spell it out: a hand-built route carrying the explicit OpenRouter URL and a
#: build_route() route leaving it None reach the same endpoint and must hash
#: identically. Not canonicalising this made every V3 arm's frozen identity
#: differ from the identity it actually ran under.
_BACKEND_DEFAULT_BASE_URL = {
    "openrouter": "https://openrouter.ai/api/v1",
}


def canonical_base_url(backend: str | None, base_url: str | None) -> str | None:
    """The effective endpoint. None resolves to the backend's own default where
    one is known; a trailing slash is not a different endpoint."""
    url = base_url or _BACKEND_DEFAULT_BASE_URL.get(str(backend or "").lower())
    return url.rstrip("/") if isinstance(url, str) else url


def effective_config_fields(route: Any) -> dict[str, Any]:
    """The request-affecting fields of the EFFECTIVE backend configuration.

    Derived by actually running ``to_backend_config()``, so candidate
    overrides, reasoning and provider routing are all already folded in.
    """
    cfg = route.to_backend_config() if hasattr(route, "to_backend_config") else route
    out: dict[str, Any] = {}
    for name in SEMANTIC_FIELDS + EXPERIMENT_ONLY_FIELDS:
        if name in EXCLUDED_FIELDS:
            continue
        out[name] = _canonical(getattr(cfg, name, None))
    # route-level identity that never reaches BackendConfig
    for name in ROUTE_LEVEL_FIELDS:
        out[name] = _canonical(getattr(route, name, None))
    # canonicalise the ENDPOINT, so "unset" and "spelled out" agree
    out["base_url"] = canonical_base_url(out.get("backend"), out.get("base_url"))
    leaked = EXCLUDED_FIELDS & set(out)
    if leaked:                                  # defensive: cannot happen above
        raise IdentityError(f"excluded field(s) reached the identity: {sorted(leaked)}")
    return out


def route_semantic_fields(route: Any) -> dict[str, Any]:
    """Just the semantic (response-affecting) half of the effective config."""
    full = effective_config_fields(route)
    keep = set(SEMANTIC_FIELDS) | set(ROUTE_LEVEL_FIELDS)
    return {k: v for k, v in full.items() if k in keep}


def wire_response_format(route: Any, output_model: Any) -> dict[str, Any]:
    """The response_format block AS TRANSMITTED.

    Identity must key on what actually goes on the wire, not on a proxy for it.
    Hashing the RAW ``model_json_schema()`` was such a proxy: the backend sends
    ``strict_json_schema(raw)`` under a ``name``, so a change to the strict
    transform, or a differently-named model with identical fields, would alter
    the transmitted payload while leaving a raw-schema digest unmoved. That is
    the same class of defect as the omitted ``provider`` field, one level down.
    """
    from .strictschema import strict_json_schema

    cfg = route.to_backend_config() if hasattr(route, "to_backend_config") else route
    raw = output_model.model_json_schema() if hasattr(output_model, "model_json_schema") else output_model
    strict = bool(getattr(cfg, "strict_schema", True))
    mode = getattr(cfg, "structured_mode", "json_schema")
    schema = strict_json_schema(raw) if strict else raw
    return {
        "structured_mode": mode,
        # transmitted as response_format.json_schema.name
        "name": getattr(output_model, "__name__", None),
        "schema": schema,
    }


def semantic_request_identity(route: Any, *, system: str, content_blocks: list[dict],
                              schema: dict, max_tokens: int | None) -> str:
    """May a stored response be reused for this request?

    Includes the request content, so it is a per-request key. Versioned.
    """
    blocks = []
    for b in content_blocks or []:
        if b.get("type") == "text":
            blocks.append(["text", _h(b.get("text", ""))])
        elif b.get("type") == "image":
            blocks.append(["image", _h((b.get("source") or {}).get("data", ""))])
        else:
            blocks.append([str(b.get("type")), _h(b)])
    return _h({
        "identity_version": CACHE_IDENTITY_VERSION,
        "kind": "semantic_request",
        "route": route_semantic_fields(route),
        "max_tokens_override": max_tokens,
        "system": _h(system or ""),
        "schema": _h(schema or {}),
        "blocks": blocks,
    })


def experiment_identity(route: Any, *, extra: dict[str, Any] | None = None) -> str:
    """Is this the same experiment? Route-level only — no per-case content."""
    return _h({
        "identity_version": CACHE_IDENTITY_VERSION,
        "kind": "experiment",
        "effective_config": effective_config_fields(route),
        "extra": _canonical(extra or {}),
    })


def identity_report(route: Any) -> dict[str, Any]:
    """Human-readable identity breakdown for a freeze artifact."""
    eff = effective_config_fields(route)
    prov = (eff.get("extra_generation") or {}).get("provider") or {}
    return {
        "identity_version": CACHE_IDENTITY_VERSION,
        "derived_from": "TaskRoute.to_backend_config() — the EFFECTIVE configuration",
        "semantic_fields": list(SEMANTIC_FIELDS) + list(ROUTE_LEVEL_FIELDS),
        "experiment_only_fields": list(EXPERIMENT_ONLY_FIELDS),
        "excluded_fields": sorted(EXCLUDED_FIELDS),
        "effective_config": eff,
        "provider_order": prov.get("order"),
        "allow_fallbacks": prov.get("allow_fallbacks"),
        "experiment_identity": experiment_identity(route),
    }


__all__ = ["CACHE_IDENTITY_VERSION", "SEMANTIC_FIELDS", "EXPERIMENT_ONLY_FIELDS",
           "ROUTE_LEVEL_FIELDS", "canonical_base_url",
           "EXCLUDED_FIELDS", "IdentityError", "effective_config_fields",
           "route_semantic_fields", "semantic_request_identity", "experiment_identity",
           "identity_report"]


def identities_from_argv(argv: list[str], *, output_model: Any = None,
                         system: str | None = None,
                         content_blocks: list[dict] | None = None,
                         max_tokens: int | None = None) -> dict[str, Any]:
    """Derive identities through the REAL runtime path.

    ``argv -> build_parser -> _spec_from_args -> build_route ->
    to_backend_config -> identities``

    A freeze must not maintain a parallel hand-built ``TaskRoute``: V3 froze
    identities from one and executed under another, and the two differed by
    ``base_url`` alone. Anything that computes a frozen identity should call
    this, so the frozen value is by construction the value execution produces.
    """
    from .benchmark.cli import _spec_from_args
    from .benchmark.registry import load_registry
    from .benchmark.roles import adapter_for
    from .benchmark.runner import build_route
    from .cli import build_parser

    args = build_parser().parse_args(argv)
    spec = _spec_from_args(args, dry_run=False)
    # Reproduce the runner EXACTLY: it loads the candidate registry (so declared
    # candidate_overrides apply) and seeds max_tokens from the adapter default.
    # Passing registry=None instead silently picked up models.toml's production
    # defaults — a parallel path, which is the very thing this helper exists to
    # eliminate.
    registry = load_registry(spec.registry_path)
    adapter = adapter_for(spec.role, prompt_version=spec.prompt_version)
    route = build_route(spec, spec.candidate, spec.prompt_version or "m2-strict-v1",
                        max_tokens or adapter.default_max_tokens, registry=registry)
    out: dict[str, Any] = {
        "experiment_identity": experiment_identity(route),
        "effective_config": effective_config_fields(route),
        "spec": {
            "candidate": spec.candidate, "prompt_version": spec.prompt_version,
            "provider": spec.provider, "transport_retries": spec.transport_retries,
            "cache_policy": getattr(spec, "cache_policy", "use"),
            "subset": spec.subset, "split": spec.split, "research": spec.research,
        },
    }
    if output_model is not None and content_blocks is not None:
        out["semantic_request_identity"] = semantic_request_identity(
            route, system=system or "", content_blocks=content_blocks,
            schema=wire_response_format(route, output_model), max_tokens=max_tokens)
        out["wire_response_format"] = wire_response_format(route, output_model)
    return out


class IdentityMismatch(RuntimeError):
    """A runtime route does not match the frozen arm it claims to be."""


def assert_identity_matches(*, route: Any, frozen_experiment_identity: str,
                            arm_id: str | None = None) -> dict[str, Any]:
    """Hard equality gate, intended to run BEFORE any cache read or send.

    Raises with a field-level, secret-free diff. A differing field is never
    waved through as harmless: V3 continued past exactly such a mismatch and the
    arm it measured was not the arm that was frozen.
    """
    actual = experiment_identity(route)
    if actual == frozen_experiment_identity:
        return {"arm_id": arm_id, "match": True, "identity": actual}
    eff = effective_config_fields(route)
    raise IdentityMismatch(
        f"arm {arm_id!r}: runtime route identity {actual} does not equal the frozen "
        f"{frozen_experiment_identity}. ZERO cache reads and ZERO requests were made. "
        f"Runtime effective configuration: {json.dumps(eff, sort_keys=True, default=str)}")
