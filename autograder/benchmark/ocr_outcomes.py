"""Explicit OCR outcome taxonomy.

A single ``success`` or ``refusal`` field cannot describe what happened to an
OCR request, and collapsing them has already misled this project twice:

* Stage-1c's ``metrics.json`` reported ``schema_failures: 3`` when all three
  losses were provider-side content_filter outcomes and the model's structured
  output had been valid on every request it answered.
* The Stage-1c summary reported Gemini ``Refusals = 0`` on the same arm where
  three requests came back content-filtered. That is defensible ONLY if
  "refusal" means *the model wrote a refusal*, which is not how a reader parses
  a column called Refusals.

So every row is classified along ten INDEPENDENT axes. They are deliberately
not mutually exclusive: a row can be provider-completed AND a model-text
refusal AND not a usable transcription, and that combination is exactly the
case a single boolean hides.

    provider_http_response_received an HTTP response body arrived at all
    provider_request_completed      the provider returned CONTENT for us to use
                                    (a body arrived AND it was not a provider-side
                                    refusal/failure). A content_filter finish is a
                                    received body but NOT a completed request —
                                    naming both is the point of this module
    provider_content_filter_failure finish_reason == content_filter
    provider_other_http_failure     HTTP/transport error, no response body
    model_text_refusal              the model answered with only an unreadable
                                    marker against a readable reference
    usable_transcription_returned   schema-valid text that is not a bare marker
    fabrication_detected            NOT decided here — semantic, human-assigned
    truncation                      finish_reason == length
    json_parse_failure              a body arrived but was not parseable JSON
    schema_failure                  parsed JSON that did not satisfy the schema
    total_line_loss                 no usable text for this crop, any cause

``usable_transcription_returned`` is the one that belongs in a coverage
numerator. ``provider_request_completed`` is NOT: an HTTP 200 carrying
"[unreadable]" is a completed request and a lost line.
"""
from __future__ import annotations

from typing import Any

#: Markers the frozen m2-strict-v1 prompts instruct the model to emit when it
#: cannot read something. A response consisting only of one of these is the
#: model declining to transcribe, not a transcription.
UNREADABLE_MARKERS: tuple[str, ...] = ("[unreadable]", "[?]")

#: The Hebrew marker used inside audited references themselves. A reference that
#: contains it is not fully readable, so a model echoing uncertainty there is
#: not refusing — it agrees with the human auditor.
REFERENCE_UNREADABLE_MARKERS: tuple[str, ...] = ("[לא קריא]", "לא קריא")

OUTCOME_FIELDS: tuple[str, ...] = (
    "provider_http_response_received",
    "provider_request_completed",
    "provider_content_filter_failure",
    "provider_other_http_failure",
    "model_text_refusal",
    "usable_transcription_returned",
    "fabrication_detected",
    "truncation",
    "json_parse_failure",
    "schema_failure",
    "total_line_loss",
)


def reference_is_readable(reference: str | None) -> bool:
    """A reference that itself says '[לא קריא]' is not fully readable."""
    if not reference or not reference.strip():
        return False
    return not any(m in reference for m in REFERENCE_UNREADABLE_MARKERS)


def is_bare_marker(text: str | None) -> bool:
    return bool(text is not None and text.strip() in UNREADABLE_MARKERS)


def classify_row(row: dict, reference: str | None = None) -> dict[str, Any]:
    """Classify one ``outputs.jsonl`` row along all ten axes.

    ``row`` is the raw runner record — never a derived metrics file, which is
    where the earlier mislabelling lived.
    """
    err = str(row.get("error") or "")
    low = err.lower()
    out = (row.get("output") or {}) if row.get("output") else {}
    text = out.get("transcription")

    content_filter = "content_filter" in low
    truncation = "truncated at max_tokens" in low or "finish_reason=length" in low
    json_parse = ("invalid json" in low or "eof while parsing" in low
                  or "expecting value" in low)
    # a validation error that is not a raw-JSON problem is a schema problem
    schema = ("validation" in low and not json_parse)
    http_other = bool(err) and not (content_filter or truncation or json_parse or schema)

    # A BODY arrived for: any success, a content_filter finish, a truncation
    # finish, and any parse/schema problem. It did not for transport/HTTP errors.
    body_received = (bool(text is not None) or content_filter or truncation
                     or json_parse or schema)
    # The request COMPLETED only if that body carried content we could use at
    # all. A content_filter finish received a body and completed nothing — the
    # distinction Stage-1c's "Refusals = 0" line blurred.
    completed = body_received and not (content_filter or http_other)

    ref_readable = reference_is_readable(reference)
    refusal = bool(text is not None and is_bare_marker(text) and ref_readable)
    usable = bool(text is not None and text.strip() != "" and not is_bare_marker(text))

    return {
        "provider_http_response_received": body_received,
        "provider_request_completed": completed,
        "provider_content_filter_failure": content_filter,
        "provider_other_http_failure": http_other,
        "model_text_refusal": refusal,
        "usable_transcription_returned": usable,
        # Semantic, and never inferred from a distance metric. Callers overlay a
        # human/auditor decision; nothing here guesses it.
        "fabrication_detected": None,
        "truncation": truncation,
        "json_parse_failure": json_parse,
        "schema_failure": schema,
        "total_line_loss": not usable,
        "failure_detail": err[:200] or None,
    }


def summarize(classified: dict[str, dict], case_ids=None) -> dict[str, Any]:
    """Counts over an INTENDED denominator, with coverage stated as a fraction."""
    ids = list(case_ids) if case_ids is not None else list(classified)
    rows = [classified[c] for c in ids if c in classified]
    n = len(rows)
    counts = {f: sum(1 for r in rows if r.get(f) is True) for f in OUTCOME_FIELDS}
    counts["intended_crops"] = n
    counts["usable_coverage"] = f"{counts['usable_transcription_returned']}/{n}"
    counts["usable_rate"] = (round(counts["usable_transcription_returned"] / n, 4) if n else None)
    counts["provider_completed_coverage"] = f"{counts['provider_request_completed']}/{n}"
    counts["hard_provider_failures"] = sum(
        1 for r in rows if r["provider_content_filter_failure"] or r["provider_other_http_failure"]
        or r["truncation"] or r["json_parse_failure"] or r["schema_failure"])
    counts["case_ids"] = ids
    return counts


def coverage_line(counts: dict) -> str:
    """The one-line phrasing that may not be shortened in a report."""
    return (f"usable {counts['usable_coverage']} · "
            f"provider-filter {counts['provider_content_filter_failure']} · "
            f"model-text refusals {counts['model_text_refusal']} · "
            f"other provider failures {counts['provider_other_http_failure']} · "
            f"truncation {counts['truncation']} · "
            f"parse/schema {counts['json_parse_failure'] + counts['schema_failure']}")


__all__ = ["UNREADABLE_MARKERS", "REFERENCE_UNREADABLE_MARKERS", "OUTCOME_FIELDS",
           "reference_is_readable", "is_bare_marker", "classify_row", "summarize",
           "coverage_line"]
