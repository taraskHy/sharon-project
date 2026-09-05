"""OCR_PROVIDER_METADATA_CAPTURE_PROTOCOL_V1 — executable capture + acceptance.

Why this module exists
----------------------

V4 was blocked, and correctly so, because it named a provider-catalogue capture
a "prerequisite" without saying what the capture was or what would make it pass.
A prerequisite stated in prose is not a preregistered procedure: whatever the
catalogue returned would have been judged after it was seen.

So the criteria live here, as code, and are frozen and committed *before* the
first request. Nothing in this module inspects a live response to decide what
"acceptable" means — the thresholds, the required slugs, the price comparison
and every failure action are constants, and :func:`evaluate_acceptance` is a
pure function of a captured snapshot.

Safety contract
---------------

These are PUBLIC metadata reads. The protocol forbids sending an Authorization
header, forbids any request body, and forbids POST. If an endpoint were to
demand a credential, the correct outcome is to stop, not to authenticate: a
metadata request that needs a key is not the public read this protocol
authorizes.

Cost direction
--------------

When a live price is LOWER than the frozen one, the frozen (higher) price is
retained. A screen's conservative maximum must never shrink because a vendor is
running a promotion — the ceiling is a commitment, not a forecast.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from typing import Any

PROTOCOL_ID = "OCR_PROVIDER_METADATA_CAPTURE_PROTOCOL_V1"

# ---------------------------------------------------------------- endpoints --
#: Identified from PRESERVED artifacts (candidate-discovery + catalog snapshot),
#: never from a new live response.
BASE = "https://openrouter.ai/api/v1"
ENDPOINT_PROVIDERS = f"{BASE}/providers"
ENDPOINT_MODELS = f"{BASE}/models"
ENDPOINT_MODEL_ENDPOINTS = f"{BASE}/models/{{slug}}/endpoints"

HTTP_METHOD = "GET"
REQUEST_BODY = None
TIMEOUT_S = 60.0
#: Metadata-only retry: transport/5xx may be retried this many times. A non-200
#: that is not retryable is a failure, never a reason to try a different URL.
MAX_RETRIES = 2
FOLLOW_REDIRECTS = False
PERMITTED_CONTENT_TYPES = ("application/json",)
ALLOWED_RESPONSE_HEADERS = ("content-type", "content-length", "date", "server",
                            "x-request-id", "request-id")
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%f%z"   # UTC ISO-8601 with offset

#: Never sent. Present so the prohibition is machine-checkable.
FORBIDDEN_REQUEST_HEADERS = ("authorization", "x-api-key", "api-key", "cookie")

# ------------------------------------------------------- required identities --
REQUIRED_PROVIDER_SLUGS = ("google-ai-studio", "google-vertex", "alibaba")

#: The preserved mapping V4/V3 were frozen against. A live catalogue must remain
#: COMPATIBLE with these; a change here is a rejection, not an update.
PRESERVED_MAPPING = {
    "google-ai-studio": "Google AI Studio",
    "google-vertex": "Google",
}

REQUIRED_ARMS = (
    {"arm_id": "gemini_pinned_ai_studio", "model": "google/gemini-3.7-flash",
     "provider_slug": "google-ai-studio"},
    {"arm_id": "gemini_pinned_vertex", "model": "google/gemini-3.7-flash",
     "provider_slug": "google-vertex"},
    {"arm_id": "qwen3_vl_235b_pinned_alibaba", "model": "qwen/qwen3-vl-235b-a22b-instruct",
     "provider_slug": "alibaba"},
)

# ------------------------------------------------------------------ pricing --
#: Frozen prices (USD per 1M tokens) and the measured per-case token profile the
#: conservative maximum is computed from. These are the V3/V4 figures.
FROZEN_PRICES = {
    "google/gemini-3.7-flash": {"input_per_m": 0.75, "output_per_m": 3.75},
    "qwen/qwen3-vl-235b-a22b-instruct": {"input_per_m": 0.21, "output_per_m": 1.90},
}
#: input tokens assumed per case, and max_tokens (the conservative output cap).
TOKEN_PROFILE = {
    "google/gemini-3.7-flash": {"input_tokens": 1388, "max_tokens": 1000},
    "qwen/qwen3-vl-235b-a22b-instruct": {"input_tokens": 3000, "max_tokens": 1000},
}
CASES_PER_ARM = 8
FROZEN_CAMPAIGN_MAXIMUM_USD = 0.096896
L0_USD = 0.71783254
FAMILY_HARD_ABS_USD = 0.82323229

ACCEPTED = "ACCEPTED"
FAILED = "METADATA_PREREQUISITE_FAILED"


class MetadataProtocolError(RuntimeError):
    pass


def utc_now() -> str:
    return time.strftime(TIMESTAMP_FORMAT, time.gmtime()).replace("+0000", "+0000") or ""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def safe_headers(headers: Any) -> dict[str, str]:
    out = {}
    try:
        items = headers.items()
    except AttributeError:
        return out
    for k, v in items:
        lk = str(k).lower()
        if lk in FORBIDDEN_REQUEST_HEADERS:
            continue
        if lk in ALLOWED_RESPONSE_HEADERS:
            out[lk] = str(v)
    return out


def _norm_name(s: str | None) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).casefold()


def _price_per_m(value: Any) -> float | None:
    """OpenRouter reports per-TOKEN strings; convert to per-million."""
    try:
        return float(value) * 1_000_000.0
    except (TypeError, ValueError):
        return None


def conservative_arm_cost(model: str, input_per_m: float, output_per_m: float) -> float:
    p = TOKEN_PROFILE[model]
    per_case = (p["input_tokens"] * input_per_m / 1_000_000.0
                + p["max_tokens"] * output_per_m / 1_000_000.0)
    return round(per_case * CASES_PER_ARM, 8)


# --------------------------------------------------------------- acceptance --

def evaluate_acceptance(snapshot: dict[str, Any]) -> dict[str, Any]:
    """PURE. Decide ACCEPTED / METADATA_PREREQUISITE_FAILED from a snapshot.

    ``snapshot`` carries the parsed ``providers``, ``models`` and
    ``model_endpoints`` sections plus each request's transport result. Every
    threshold used here is a module constant frozen before capture.
    """
    reasons: list[dict[str, Any]] = []

    def fail(code, detail, **extra):
        reasons.append({"ok": False, "code": code, "detail": detail, **extra})

    def okc(code, detail, **extra):
        reasons.append({"ok": True, "code": code, "detail": detail, **extra})

    # ---- transport / archival -------------------------------------------
    for name, req in (snapshot.get("requests") or {}).items():
        if req.get("http_status") != 200:
            fail("HTTP_NOT_200", f"{name}: HTTP {req.get('http_status')}", request=name)
        ct = (req.get("headers") or {}).get("content-type", "")
        if not any(ct.startswith(t) for t in PERMITTED_CONTENT_TYPES):
            fail("BAD_CONTENT_TYPE", f"{name}: content-type {ct!r}", request=name)
        if not req.get("raw_body_sha256") or not req.get("archived"):
            fail("ARCHIVE_FAILURE", f"{name}: raw body was not archived", request=name)
        if req.get("parse_error"):
            fail("PARSE_FAILURE", f"{name}: {req['parse_error']}", request=name)

    # ---- provider catalogue ---------------------------------------------
    provs = snapshot.get("providers") or []
    by_slug: dict[str, list[str]] = {}
    for p in provs:
        slug, name = p.get("slug"), p.get("name")
        if isinstance(slug, str):
            by_slug.setdefault(slug, []).append(name)
    for slug in REQUIRED_PROVIDER_SLUGS:
        entries = by_slug.get(slug)
        if not entries:
            fail("SLUG_ABSENT", f"required provider slug {slug!r} is not in the catalogue",
                 slug=slug)
            continue
        if len(entries) != 1:
            fail("SLUG_NOT_UNIQUE", f"slug {slug!r} appears {len(entries)} times", slug=slug)
            continue
        name = entries[0]
        if not isinstance(name, str) or not name.strip():
            fail("EMPTY_DISPLAY_NAME", f"slug {slug!r} has no display name", slug=slug)
            continue
        okc("SLUG_RESOLVED", f"{slug!r} -> {name!r}", slug=slug, display_name=name)
        # reverse ambiguity, reported explicitly
        owners = sorted({s for s, ns in by_slug.items()
                         if any(_norm_name(n) == _norm_name(name) for n in ns)})
        if len(owners) > 1:
            fail("REVERSE_AMBIGUOUS",
                 f"display name {name!r} maps to multiple slugs {owners}",
                 slug=slug, display_name=name, owners=owners)
        # compatibility with the preserved mapping
        if slug in PRESERVED_MAPPING:
            if _norm_name(name) != _norm_name(PRESERVED_MAPPING[slug]):
                fail("PRESERVED_MAPPING_CHANGED",
                     f"{slug!r} was {PRESERVED_MAPPING[slug]!r}, catalogue now says {name!r}",
                     slug=slug)
            else:
                okc("PRESERVED_MAPPING_COMPATIBLE", f"{slug!r} still {name!r}", slug=slug)

    # ---- model endpoints -------------------------------------------------
    eps = snapshot.get("model_endpoints") or {}
    for arm in REQUIRED_ARMS:
        model, slug = arm["model"], arm["provider_slug"]
        info = eps.get(model)
        if not info:
            fail("MODEL_ENDPOINTS_MISSING", f"no endpoint data for {model!r}", model=model)
            continue
        if info.get("canonical_slug") and info["canonical_slug"] != model:
            fail("MODEL_ALIAS_REDIRECT",
                 f"{model!r} resolves to {info['canonical_slug']!r}", model=model)
        names = {_norm_name(e.get("provider_name")) for e in (info.get("endpoints") or [])}
        want = _norm_name(by_slug.get(slug, [None])[0]) if by_slug.get(slug) else None
        if not want or want not in names:
            fail("ARM_ROUTE_UNAVAILABLE",
                 f"{model!r} is not served by {slug!r} in the captured endpoints",
                 model=model, slug=slug, observed=sorted(n for n in names if n))
        else:
            okc("ARM_ROUTE_AVAILABLE", f"{model!r} available via {slug!r}",
                model=model, slug=slug)
        caps = info.get("capabilities") or {}
        if caps.get("image_input") is False:
            fail("IMAGE_INPUT_LOST", f"{model!r} no longer reports image input", model=model)
        if caps.get("structured_outputs") is False:
            fail("STRUCTURED_OUTPUT_LOST", f"{model!r} no longer reports structured outputs",
                 model=model)

    # ---- pricing ---------------------------------------------------------
    prices = snapshot.get("prices") or {}
    effective: dict[str, dict[str, float]] = {}
    for model, frozen in FROZEN_PRICES.items():
        live = prices.get(model)
        if not live or live.get("input_per_m") is None or live.get("output_per_m") is None:
            fail("PRICE_FIELDS_MISSING", f"{model!r}: price fields absent", model=model)
            continue
        eff_in = max(frozen["input_per_m"], live["input_per_m"])
        eff_out = max(frozen["output_per_m"], live["output_per_m"])
        effective[model] = {"input_per_m": eff_in, "output_per_m": eff_out,
                            "frozen": frozen, "live": live,
                            "retained_frozen_because_live_is_lower":
                                bool(live["input_per_m"] < frozen["input_per_m"]
                                     or live["output_per_m"] < frozen["output_per_m"])}
        okc("PRICE_CAPTURED", f"{model!r} live {live} vs frozen {frozen}", model=model)

    arm_costs, total = {}, 0.0
    if len(effective) == len(FROZEN_PRICES):
        for arm in REQUIRED_ARMS:
            m = arm["model"]
            c = conservative_arm_cost(m, effective[m]["input_per_m"], effective[m]["output_per_m"])
            arm_costs[arm["arm_id"]] = c
            total += c
        total = round(total, 8)
        if total > FROZEN_CAMPAIGN_MAXIMUM_USD:
            fail("CAMPAIGN_MAXIMUM_EXCEEDED",
                 f"recomputed conservative maximum ${total:.8f} exceeds the frozen "
                 f"${FROZEN_CAMPAIGN_MAXIMUM_USD}", recomputed=total)
        else:
            okc("CAMPAIGN_MAXIMUM_WITHIN_FROZEN", f"${total:.8f} <= ${FROZEN_CAMPAIGN_MAXIMUM_USD}",
                recomputed=total)
        if round(L0_USD + total, 8) > FAMILY_HARD_ABS_USD:
            fail("FAMILY_HARD_LIMIT_EXCEEDED",
                 f"L0 + maximum = ${L0_USD + total:.8f} exceeds ${FAMILY_HARD_ABS_USD}")
        else:
            okc("FITS_FAMILY_HARD_LIMIT",
                f"L0 + maximum = ${L0_USD + total:.8f} <= ${FAMILY_HARD_ABS_USD}")

    failures = [r for r in reasons if not r["ok"]]
    return {
        "protocol": PROTOCOL_ID,
        "result": FAILED if failures else ACCEPTED,
        "failure_count": len(failures),
        "reasons": reasons,
        "resolved_mapping": {s: (by_slug.get(s) or [None])[0] for s in REQUIRED_PROVIDER_SLUGS},
        "effective_prices_per_m": effective,
        "conservative_arm_costs_usd": arm_costs,
        "conservative_campaign_maximum_usd": round(total, 8) if arm_costs else None,
        "frozen_campaign_maximum_usd": FROZEN_CAMPAIGN_MAXIMUM_USD,
        "L0_usd": L0_USD, "family_hard_abs_usd": FAMILY_HARD_ABS_USD,
        "remaining_headroom_usd": (round(FAMILY_HARD_ABS_USD - L0_USD - total, 8)
                                   if arm_costs else None),
    }


def protocol_document() -> dict[str, Any]:
    """The frozen protocol, exactly as the artifact records it."""
    return {
        "protocol": PROTOCOL_ID,
        "endpoints": {"providers": ENDPOINT_PROVIDERS, "models": ENDPOINT_MODELS,
                      "model_endpoints_template": ENDPOINT_MODEL_ENDPOINTS},
        "endpoint_provenance": ("identified from PRESERVED artifacts "
                                "(OCR_CANDIDATE_DISCOVERY_2026-09-03.json, "
                                "OCR_OPENROUTER_CATALOG_SNAPSHOT_2026-09-03.json), never from a "
                                "new live response"),
        "http_method": HTTP_METHOD, "request_body": REQUEST_BODY,
        "timeout_s": TIMEOUT_S, "max_retries": MAX_RETRIES,
        "retry_scope": "transport errors and retryable 5xx only; never a different URL",
        "follow_redirects": FOLLOW_REDIRECTS,
        "permitted_content_types": list(PERMITTED_CONTENT_TYPES),
        "allowed_response_headers": list(ALLOWED_RESPONSE_HEADERS),
        "forbidden_request_headers": list(FORBIDDEN_REQUEST_HEADERS),
        "unauthenticated": ("metadata requests MUST be unauthenticated. If a credential is "
                            "required, STOP rather than sending it."),
        "no_payload_content": ("no exam content, crops, prompts, references or credentials are "
                               "transmitted; there is no request body at all"),
        "raw_body_preservation": ("the exact response text is archived verbatim and its SHA-256 "
                                  "recorded; the parsed result is stored SEPARATELY"),
        "timestamp_format": TIMESTAMP_FORMAT,
        "json_parsing": ("strict json.loads of the raw body; any exception is PARSE_FAILURE and "
                         "the protocol fails"),
        "fields": {
            "provider_slug": "providers[].slug",
            "provider_display_name": "providers[].name",
            "model_endpoint_provider": "models/{slug}/endpoints -> data.endpoints[].provider_name",
            "model_canonical_slug": "data.canonical_slug",
            "availability": "presence of a matching endpoints[] entry for the pinned provider",
            "price_input": "models[].pricing.prompt (per TOKEN)",
            "price_output": "models[].pricing.completion (per TOKEN)",
            "image_input": "models[].architecture.input_modalities contains 'image'",
            "structured_outputs": "models[].supported_parameters contains 'structured_outputs'",
        },
        "normalization": ("display names compared after whitespace collapse + casefold ONLY, and "
                          "only after canonical slug lookup; slugs are never normalised into "
                          "display names"),
        "mapping_uniqueness": ("each required slug must appear exactly once with a non-empty "
                               "display name; reverse ambiguity (one display name owned by "
                               "multiple slugs) is reported explicitly and fails"),
        "price_rules": {
            "unit_conversion": "per-token strings multiplied by 1e6 to per-million",
            "effective_price": "max(frozen, live) per field — a lower live price NEVER lowers the "
                               "conservative maximum",
            "conservative_arm_cost": "8 cases x (input_tokens*input_per_m + max_tokens*output_per_m)",
            "token_profile": TOKEN_PROFILE,
            "frozen_prices_per_m": FROZEN_PRICES,
        },
        "acceptance_outcomes": {"pass": ACCEPTED, "fail": FAILED},
        "rejection_triggers": [
            "HTTP status other than 200", "unexpected content type", "archive failure",
            "JSON parse failure", "required slug absent", "slug not unique",
            "empty display name", "reverse-ambiguous display name",
            "preserved Google mapping changed", "model endpoints missing",
            "model alias/canonical redirect", "arm route unavailable at the pinned provider",
            "image input lost", "structured output lost", "price fields missing",
            "recomputed conservative maximum above the frozen maximum",
            "L0 + maximum above the family hard limit",
        ],
        "failure_action": ("record METADATA_PREREQUISITE_FAILED immutably, make no inference "
                           "calls, invent no mappings, do NOT drop Qwen, and do NOT freeze an "
                           "executable V5"),
        "gates_never_weakened": ("output-token assumptions and advancement gates are constants "
                                 "here; they may not be relaxed to make a budget fit"),
        "required_provider_slugs": list(REQUIRED_PROVIDER_SLUGS),
        "preserved_mapping": dict(PRESERVED_MAPPING),
        "required_arms": [dict(a) for a in REQUIRED_ARMS],
        "frozen_campaign_maximum_usd": FROZEN_CAMPAIGN_MAXIMUM_USD,
        "L0_usd": L0_USD, "family_hard_abs_usd": FAMILY_HARD_ABS_USD,
    }


__all__ = ["PROTOCOL_ID", "ENDPOINT_PROVIDERS", "ENDPOINT_MODELS", "ENDPOINT_MODEL_ENDPOINTS",
           "HTTP_METHOD", "TIMEOUT_S", "MAX_RETRIES", "FOLLOW_REDIRECTS",
           "PERMITTED_CONTENT_TYPES", "ALLOWED_RESPONSE_HEADERS", "FORBIDDEN_REQUEST_HEADERS",
           "REQUIRED_PROVIDER_SLUGS", "PRESERVED_MAPPING", "REQUIRED_ARMS", "FROZEN_PRICES",
           "TOKEN_PROFILE", "CASES_PER_ARM", "FROZEN_CAMPAIGN_MAXIMUM_USD", "L0_USD",
           "FAMILY_HARD_ABS_USD", "ACCEPTED", "FAILED", "MetadataProtocolError",
           "evaluate_acceptance", "protocol_document", "conservative_arm_cost",
           "safe_headers", "sha256_text"]
