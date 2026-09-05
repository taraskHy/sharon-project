"""Canonical OpenRouter provider slug <-> display-name mapping.

Why this module exists
----------------------

The V3 campaign was halted on its second arm by a route violation that was not
real. The arm pinned the slug ``google-vertex``; the response reported the
provider as ``Google``; and the comparison — which lowercased both sides and
stripped punctuation — concluded they were different providers. They are not.
OpenRouter's slug for Vertex *is* ``google-vertex`` and its display name *is*
``Google``.

Worse, the first arm passed only by luck: ``google-ai-studio`` and
``Google AI Studio`` happen to normalise to the same string. So the check was
never sound; it was coincidentally right once and confidently wrong once.

String normalisation cannot decide provider identity. A slug and a display name
are different namespaces, and the relationship between them is *data*, not a
transformation. This module holds that data, sourced from a preserved artifact
rather than from memory, and records which entries are actually evidenced.

Verification status is part of the contract
-------------------------------------------

An entry is ``VERIFIED`` only when a preserved artifact records the mapping.
``UNVERIFIED`` entries exist so a pin can still be *declared* without pretending
its attribution can be confirmed: an observed value for an unverified slug is
reported ``UNKNOWN_UNVERIFIED_SLUG`` — never silently compliant, and never a
violation either. Absence of evidence is not evidence of a breach; it is also
not permission to assume success.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The artifact the Google mappings are read from. It recorded OpenRouter's own
#: /providers response during the 2026-09-03 route-forensics session.
SOURCE_ARTIFACT = Path("evaluation/model_selection/runs/ocr_primary/"
                       "OCR_PROVIDER_ROUTE_FORENSICS_2026-09-03.json")

#: Attribution / compliance results.
COMPLIANT = "COMPLIANT"
VIOLATION = "VIOLATION"
UNKNOWN = "UNKNOWN"                                   # response named no provider
UNKNOWN_AMBIGUOUS = "UNKNOWN_AMBIGUOUS"               # name maps to >1 slug
UNKNOWN_UNRECOGNISED = "UNKNOWN_UNRECOGNISED"         # name maps to no known slug
UNKNOWN_UNVERIFIED_SLUG = "UNKNOWN_UNVERIFIED_SLUG"   # slug has no evidenced mapping

VERIFIED = "VERIFIED"
UNVERIFIED = "UNVERIFIED"


class ProviderMapError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderEntry:
    slug: str
    display_names: tuple[str, ...]
    status: str
    evidence: str


def _parse_forensics(path: Path) -> dict[str, ProviderEntry]:
    """Read 'Display Name (slug)' pairs recorded verbatim in the artifact."""
    doc = json.loads(path.read_text(encoding="utf-8"))
    listed = (doc.get("does_openrouter_offer_multiple_routes_for_this_exact_model") or {}
              ).get("distinct_providers") or []
    out: dict[str, ProviderEntry] = {}
    for item in listed:
        m = re.match(r"^\s*(.+?)\s*\(([^)]+)\)\s*$", str(item))
        if not m:
            continue
        name, slug = m.group(1).strip(), m.group(2).strip()
        out[slug] = ProviderEntry(
            slug=slug, display_names=(name,), status=VERIFIED,
            evidence=f"{path.name} distinct_providers entry {item!r}")
    return out


#: Slugs this project pins that have NO preserved slug->display-name evidence.
#: They are declared so a pin remains expressible, but their attribution can
#: never be confirmed until a provider catalogue is captured and committed.
UNVERIFIED_SLUGS: dict[str, str] = {
    "alibaba": ("no preserved artifact records this slug's display name. The 2026-09-03 "
                "discovery session read it from OpenRouter's /providers endpoint but never "
                "persisted the response; every surviving mention is prose in a freeze "
                "narrative, which is not evidence of the wire value."),
}


def source_digest(path: Path | None = None) -> dict[str, Any]:
    p = Path(path or SOURCE_ARTIFACT)
    doc = json.loads(p.read_text(encoding="utf-8"))
    return {
        "artifact": str(p).replace("\\", "/"),
        "artifact_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "recorded_content_sha256": doc.get("content_sha256"),
        "captured_at": doc.get("created_at"),
        "source": "OpenRouter GET /api/v1/providers, recorded during route forensics",
    }


def load_provider_map(path: Path | None = None) -> dict[str, ProviderEntry]:
    p = Path(path or SOURCE_ARTIFACT)
    if not p.exists():
        raise ProviderMapError(f"preserved provider artifact missing: {p}")
    m = _parse_forensics(p)
    if not m:
        raise ProviderMapError(f"no provider mappings could be read from {p}")
    for slug, why in UNVERIFIED_SLUGS.items():
        m.setdefault(slug, ProviderEntry(slug=slug, display_names=(), status=UNVERIFIED,
                                         evidence=why))
    return m


def _norm(text: str | None) -> str:
    """Case/whitespace normalisation ONLY — applied after canonical mapping, never
    as a substitute for it."""
    return re.sub(r"\s+", " ", (text or "").strip()).casefold()


def slug_for_display_name(name: str, pmap: dict[str, ProviderEntry]) -> list[str]:
    """Every slug whose evidenced display names include ``name``."""
    n = _norm(name)
    return sorted(e.slug for e in pmap.values()
                  if e.status == VERIFIED and any(_norm(d) == n for d in e.display_names))


def match_provider(*, requested_slug: str | None, observed_provider: str | None,
                   pmap: dict[str, ProviderEntry] | None = None) -> dict[str, Any]:
    """Decide route compliance from the canonical mapping.

    Returns the four fields kept deliberately distinct: the requested slug, the
    display names that slug is expected to report, what was actually observed,
    and the resulting status.
    """
    pmap = pmap if pmap is not None else load_provider_map()
    entry = pmap.get(requested_slug) if requested_slug else None
    out: dict[str, Any] = {
        "requested_provider_slug": requested_slug,
        "expected_display_names": list(entry.display_names) if entry else [],
        "observed_provider": observed_provider,
        "slug_mapping_status": (entry.status if entry else UNKNOWN_UNRECOGNISED),
        "result": None,
        "detail": None,
    }
    if not requested_slug:
        out["result"] = UNKNOWN
        out["detail"] = "no provider pin requested"
        return out
    if observed_provider is None or not str(observed_provider).strip():
        out["result"] = UNKNOWN
        out["detail"] = "the response named no provider; the pin cannot be confirmed"
        return out
    if entry is None or entry.status != VERIFIED:
        out["result"] = UNKNOWN_UNVERIFIED_SLUG
        out["detail"] = (f"no evidenced display name for slug {requested_slug!r}; attribution "
                         f"cannot be confirmed or refuted")
        return out

    if any(_norm(d) == _norm(observed_provider) for d in entry.display_names):
        out["result"] = COMPLIANT
        out["detail"] = (f"observed {observed_provider!r} is the evidenced display name of "
                         f"{requested_slug!r}")
        return out

    owners = slug_for_display_name(observed_provider, pmap)
    if len(owners) == 1:
        out["result"] = VIOLATION
        out["detail"] = (f"pinned {requested_slug!r} but the response reported "
                         f"{observed_provider!r}, which is the display name of {owners[0]!r}")
    elif len(owners) > 1:
        out["result"] = UNKNOWN_AMBIGUOUS
        out["detail"] = (f"observed {observed_provider!r} maps to more than one slug "
                         f"({owners}); attribution is ambiguous, not compliant")
    else:
        out["result"] = UNKNOWN_UNRECOGNISED
        out["detail"] = (f"observed {observed_provider!r} matches no evidenced display name; "
                         f"attribution is unknown, NOT silently compliant")
    return out


__all__ = ["SOURCE_ARTIFACT", "COMPLIANT", "VIOLATION", "UNKNOWN", "UNKNOWN_AMBIGUOUS",
           "UNKNOWN_UNRECOGNISED", "UNKNOWN_UNVERIFIED_SLUG", "VERIFIED", "UNVERIFIED",
           "ProviderEntry", "ProviderMapError", "load_provider_map", "match_provider",
           "slug_for_display_name", "source_digest", "UNVERIFIED_SLUGS"]
