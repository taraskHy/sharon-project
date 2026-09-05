"""Immutable, sanitized preservation of provider HTTP responses.

The prompt-v2 post-mortem found the gap this module closes. Every one of the 27
content-filtered Gemini generations returned HTTP 200 with no ``usage`` block
and no ``provider`` field, and the request cache stored only the *parsed*,
schema-validated value — never the body. So the single question that mattered
("which serving provider filtered this crop?") was unanswerable after the fact,
and could not be recovered at any price.

The fix is to archive the raw body at the HTTP boundary, before parsing can
fail and before a filtered response is discarded as "no usable output".

Three rules hold this together:

**Sanitize by allowlist, never by blocklist.** Only headers named in
:data:`SAFE_HEADERS` are kept. A header we have not explicitly reasoned about
is dropped, so a provider adding a new authenticated echo header cannot leak a
credential into an artifact by default.

**Attribution is three separate fields, never two.** ``requested_provider`` is
what we asked for; ``observed_provider`` is what the response *explicitly*
said; ``provider_attribution_status`` says whether those can be compared. When
a body does not name its provider, ``observed_provider`` is ``None`` and the
status is ``UNKNOWN`` — it is never back-filled from the request. Inferring the
observed provider from the requested route is exactly the reasoning error that
would have manufactured a false answer to the prompt-v2 question.

**Artifacts are write-once.** A filename that already exists is never
overwritten; the collision is surfaced instead, because silently replacing an
archived response would destroy the audit trail it exists to provide.

What this module does and does not guarantee
--------------------------------------------

It does NOT guarantee that a provider call can never be lost. No software can
promise that once a response has already arrived: the provider has billed us,
and a storage failure at that moment cannot be undone by retrying, buffering or
any other local action.

The guarantee is narrower and actually achievable:

* a provider response is never *silently* accepted without its raw evidence —
  if the archive write fails, :class:`ArchiveFailure` is raised, the parsed
  result is NOT returned, and no further attempt or case is sent;
* a best-effort independent marker (``<archive>.ARCHIVE_FAILURE``) and a
  tainted billing event record what was lost, so the gap is visible in the
  audit trail rather than inferred from a hole in it.

In short: loss is possible, but silent loss is not.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

#: Response headers safe to retain. Diagnostic identifiers and content
#: metadata only — nothing that carries or echoes a credential.
SAFE_HEADERS: frozenset[str] = frozenset({
    "content-type", "content-length", "date",
    "x-request-id", "x-requestid", "request-id",
    "openrouter-id", "x-openrouter-id",
    "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset",
    "retry-after", "server", "x-provider", "x-served-by",
})

#: Header names that must never be archived even if someone adds them to the
#: allowlist by mistake. Belt and braces: the allowlist is the real control.
FORBIDDEN_HEADERS: frozenset[str] = frozenset({
    "authorization", "proxy-authorization", "cookie", "set-cookie",
    "x-api-key", "api-key", "openai-api-key", "x-goog-api-key",
})

#: Attribution statuses.
EXPLICIT = "EXPLICIT"      # the body named the serving provider
UNKNOWN = "UNKNOWN"        # the body did not; observed_provider stays None

#: Secret-shaped tokens scrubbed from any archived text as a last resort.
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"or-v1-[A-Za-z0-9]{8,}"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._\-]{8,}"),
    # api_key / "api-key" / apiKey, JSON or YAML style, quoted or bare
    re.compile(r"(?i)\"?api[_-]?key\"?\s*[:=]\s*\"?[A-Za-z0-9._\-]{6,}\"?"),
)


def redact_secrets(text: str) -> str:
    """Replace secret-shaped substrings with a marker.

    The archive should never contain a credential in the first place — bodies
    are provider output, not our request. This runs anyway because an error
    body can echo the request, and an archive is the worst possible place to
    discover that we were wrong about that.
    """
    out = text
    for pat in _SECRET_PATTERNS:
        out = pat.sub("[REDACTED]", out)
    return out


def safe_headers(headers: Any) -> dict[str, str]:
    """Allowlisted, lowercased response headers."""
    out: dict[str, str] = {}
    try:
        items = headers.items()
    except AttributeError:
        return out
    for k, v in items:
        lk = str(k).lower()
        if lk in FORBIDDEN_HEADERS:
            continue
        if lk in SAFE_HEADERS:
            out[lk] = redact_secrets(str(v))
    return out


def observed_provider_of(body: dict | None) -> tuple[str | None, str]:
    """Extract the provider the response EXPLICITLY names.

    Returns ``(observed_provider, provider_attribution_status)``. A body that
    does not name a provider yields ``(None, UNKNOWN)`` — never a guess, and
    never the requested route echoed back.
    """
    if not isinstance(body, dict):
        return None, UNKNOWN
    val = body.get("provider")
    if isinstance(val, str) and val.strip():
        return val.strip(), EXPLICIT
    return None, UNKNOWN


def requested_route_of(payload: dict | None) -> dict[str, Any]:
    """The provider-routing intent of an outbound payload, for the record."""
    prov = (payload or {}).get("provider")
    if not isinstance(prov, dict):
        return {"requested_provider": None, "requested_provider_order": None,
                "allow_fallbacks": None, "route_pinned": False}
    order = prov.get("order")
    order = list(order) if isinstance(order, list) else None
    allow = prov.get("allow_fallbacks")
    # A pin is only a pin when exactly one provider is named AND fallbacks are
    # off; anything else can be served by someone else without notice.
    pinned = bool(order and len(order) == 1 and allow is False)
    return {"requested_provider": (order[0] if order and len(order) == 1 else None),
            "requested_provider_order": order,
            "allow_fallbacks": allow,
            "route_pinned": pinned}


class RouteViolation(RuntimeError):
    """The response explicitly named a provider we did not pin to."""


class ArchiveFailure(RuntimeError):
    """The raw response could not be archived.

    This FAILS CLOSED. The response has already arrived and has already been
    billed — that cannot be undone — but proceeding would mean accepting a
    parsed result whose evidence was never recorded, and continuing to spend
    on further attempts while blind. Raising stops the arm instead.

    The honest guarantee is therefore NOT "a provider call can never be lost".
    It is: a provider call is never *silently* accepted without its raw
    evidence. If the archive write fails, the run stops and says so.
    """


def check_route(requested: dict[str, Any], observed_provider: str | None,
                status: str) -> dict[str, Any]:
    """Compare intent against what the response explicitly reported.

    A violation requires an EXPLICIT observation that disagrees. UNKNOWN is
    never a violation — it is the absence of evidence, and treating it as a
    breach would make every historical filtered response look like one.
    """
    verdict = {
        "route_pinned": requested.get("route_pinned", False),
        "requested_provider": requested.get("requested_provider"),
        "observed_provider": observed_provider,
        "provider_attribution_status": status,
        "violation": False,
        "detail": None,
    }
    if not requested.get("route_pinned"):
        verdict["detail"] = "route was not pinned; no provider guarantee to check"
        return verdict
    if status != EXPLICIT:
        verdict["detail"] = ("response did not name a provider; the pin cannot be "
                             "confirmed from this response (not treated as a violation)")
        return verdict

    # CANONICAL MAPPING, never string normalisation. A slug and a display name
    # are different namespaces: OpenRouter's `google-vertex` reports as
    # `Google`, and normalising both sides declared a correctly honoured pin a
    # violation, halting the V3 campaign. `google-ai-studio` / `Google AI Studio`
    # matched only by coincidence.
    from .providermap import (COMPLIANT, VIOLATION, load_provider_map, match_provider)

    want = requested.get("requested_provider")
    try:
        m = match_provider(requested_slug=want, observed_provider=observed_provider,
                           pmap=load_provider_map())
    except Exception as e:  # noqa: BLE001 — a missing map must not silently pass a route
        verdict["violation"] = False
        verdict["result"] = "UNKNOWN_NO_PROVIDER_MAP"
        verdict["detail"] = f"provider mapping unavailable ({type(e).__name__}: {e}); UNKNOWN"
        return verdict
    verdict["result"] = m["result"]
    verdict["expected_display_names"] = m["expected_display_names"]
    verdict["slug_mapping_status"] = m["slug_mapping_status"]
    verdict["violation"] = (m["result"] == VIOLATION)
    verdict["detail"] = m["detail"]
    return verdict


def _norm(name: str | None) -> str:
    """Compare provider identifiers tolerantly across slug/display spellings
    ('google-ai-studio' vs 'Google AI Studio')."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


@dataclass
class RawResponseRecord:
    """One archived provider response. Parsed outcome is stored SEPARATELY
    from the raw body so a parsing change can never rewrite what arrived."""
    ts: str
    http_status: int | None
    requested_model: str | None
    requested_provider: str | None
    requested_provider_order: list | None
    allow_fallbacks: bool | None
    route_pinned: bool
    observed_provider: str | None
    provider_attribution_status: str
    content_type: str | None
    raw_body: str | None
    raw_body_sha256: str | None
    raw_body_bytes: int | None
    raw_body_truncated: bool
    headers: dict[str, str] = field(default_factory=dict)
    parsed_outcome: dict[str, Any] = field(default_factory=dict)
    route_check: dict[str, Any] = field(default_factory=dict)
    attempt: int | None = None
    task: str | None = None
    case_id: str | None = None
    error: str | None = None
    # ---- stable correlation identifiers (never timestamps or file order) ----
    campaign_id: str | None = None
    arm_id: str | None = None
    logical_request_id: str | None = None
    attempt_id: str | None = None
    retry_index: int | None = None
    case_hash: str | None = None
    ledger_entry_id: str | None = None

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        body = json.dumps({k: v for k, v in d.items() if k != "content_sha256"},
                          ensure_ascii=False, sort_keys=True, default=str)
        d["content_sha256"] = hashlib.sha256(body.encode()).hexdigest()
        return d


#: bodies larger than this are truncated in the artifact (the sha256 and byte
#: count always describe the FULL body, so truncation stays detectable)
MAX_BODY_CHARS = 200_000


def build_record(*, payload: dict | None, http_status: int | None,
                 raw_text: str | None, headers: Any = None,
                 parsed_body: dict | None = None,
                 parsed_outcome: dict[str, Any] | None = None,
                 attempt: int | None = None, task: str | None = None,
                 case_id: str | None = None, error: str | None = None,
                 campaign_id: str | None = None, arm_id: str | None = None,
                 logical_request_id: str | None = None, attempt_id: str | None = None,
                 retry_index: int | None = None, ledger_entry_id: str | None = None,
                 now: str | None = None) -> RawResponseRecord:
    """Assemble one sanitized record. Pure: no I/O, no network."""
    requested = requested_route_of(payload)
    observed, status = observed_provider_of(parsed_body)
    full = raw_text if isinstance(raw_text, str) else None
    sha = hashlib.sha256(full.encode("utf-8")).hexdigest() if full is not None else None
    nbytes = len(full.encode("utf-8")) if full is not None else None
    truncated = bool(full is not None and len(full) > MAX_BODY_CHARS)
    kept = redact_secrets(full[:MAX_BODY_CHARS]) if full is not None else None
    hdrs = safe_headers(headers)
    return RawResponseRecord(
        ts=now or time.strftime("%Y-%m-%d %H:%M:%S"),
        http_status=http_status,
        requested_model=(payload or {}).get("model"),
        requested_provider=requested["requested_provider"],
        requested_provider_order=requested["requested_provider_order"],
        allow_fallbacks=requested["allow_fallbacks"],
        route_pinned=requested["route_pinned"],
        observed_provider=observed,
        provider_attribution_status=status,
        content_type=hdrs.get("content-type"),
        raw_body=kept,
        raw_body_sha256=sha,
        raw_body_bytes=nbytes,
        raw_body_truncated=truncated,
        headers=hdrs,
        parsed_outcome=dict(parsed_outcome or {}),
        route_check=check_route(requested, observed, status),
        attempt=attempt, task=task, case_id=case_id, error=error,
        campaign_id=campaign_id, arm_id=arm_id,
        logical_request_id=logical_request_id, attempt_id=attempt_id,
        retry_index=retry_index,
        case_hash=(hashlib.sha256(case_id.encode()).hexdigest()[:16] if case_id else None),
        ledger_entry_id=ledger_entry_id or attempt_id,
    )


class RawResponseArchive:
    """Write-once JSONL archive of :class:`RawResponseRecord`.

    Append-only by construction: records are added, never edited, and the file
    is opened in append mode so a crash cannot truncate earlier entries.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.written = 0

    def append(self, rec: RawResponseRecord) -> dict[str, Any]:
        doc = rec.to_json()
        line = json.dumps(doc, ensure_ascii=False, default=str)
        with self.path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(line + "\n")
        self.written += 1
        return doc

    def records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines()
                if l.strip()]


__all__ = ["SAFE_HEADERS", "FORBIDDEN_HEADERS", "EXPLICIT", "UNKNOWN", "MAX_BODY_CHARS",
           "RawResponseRecord", "RawResponseArchive", "RouteViolation", "ArchiveFailure",
           "build_record", "redact_secrets", "safe_headers", "observed_provider_of",
           "requested_route_of", "check_route"]
