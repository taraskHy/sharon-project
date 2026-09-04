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

#: Bumped when the identity DERIVATION changes. v1 = the historical
#: TaskRoute.fingerprint_fields() list that omitted `provider`.
CACHE_IDENTITY_VERSION = 2

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
    leaked = EXCLUDED_FIELDS & set(out)
    if leaked:                                  # defensive: cannot happen above
        raise IdentityError(f"excluded field(s) reached the identity: {sorted(leaked)}")
    return out


def route_semantic_fields(route: Any) -> dict[str, Any]:
    """Just the semantic (response-affecting) half of the effective config."""
    full = effective_config_fields(route)
    keep = set(SEMANTIC_FIELDS) | set(ROUTE_LEVEL_FIELDS)
    return {k: v for k, v in full.items() if k in keep}


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
           "ROUTE_LEVEL_FIELDS",
           "EXCLUDED_FIELDS", "IdentityError", "effective_config_fields",
           "route_semantic_fields", "semantic_request_identity", "experiment_identity",
           "identity_report"]
