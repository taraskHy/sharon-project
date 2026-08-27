"""The production cloud boundary: OpenRouter (or any remote endpoint) may be
used ONLY for OCR transcription. Grading never leaves the machine.

This is a PRODUCT decision (2026-08), not a configuration default:

    production + remote provider + task not in CLOUD_OCR_ALLOWLIST
        -> hard failure BEFORE request serialization / network access

The guard is enforced at the single choke point every provider request
passes through (``ModelGateway.call``), and it classifies by the EFFECTIVE
destination — backend name AND base URL — never by what models.toml claims:
an "openai" route pointed at openrouter.ai is OpenRouter, an "ollama" route
pointed at a public host is remote. Editing models.toml therefore cannot
re-enable cloud grading; only the explicit research execution mode can, and
that mode exists solely for the historical cloud-grader benchmarks
(research baselines — see docs/architecture.md).

Three independent layers, all deny-by-default:

1. TASK LAYER — a remote request is allowed only for a task in
   ``CLOUD_OCR_ALLOWLIST`` (minimal and explicit: the two OCR transcription
   roles). Grading, RAG-assisted grading, MC/variant/alignment resolution,
   policy inference and anything added later are refused by omission.
2. PROMPT LAYER — an allowed OCR task may carry only a REGISTERED OCR
   system prompt (exact string match). ``gateway.call(task="ocr_primary",
   system=GRADE_SYSTEM, ...)`` is refused: an OCR role name must not become
   a tunnel for grading content.
3. PAYLOAD LAYER — the content blocks of an allowed OCR request must not
   contain grading material. The tripwire scans for the exact section
   headers the grading prompt builder emits (rubric / official solution /
   course-context markers) plus the grading system prompt itself.

The boundary never inspects, stores, or transmits anything: it raises
``CloudBoundaryError`` before the backend serializes the request.
"""

from __future__ import annotations

from .usage import CLOUD_BACKENDS, _is_local_url, effective_provider

#: The ONLY tasks a production configuration may route to a remote provider.
#: Minimal and explicit — extend only with a deliberate, documented product
#: decision, never to make a configuration "work".
CLOUD_OCR_ALLOWLIST: frozenset[str] = frozenset({"ocr_primary", "ocr_verify"})

#: Execution modes. "production" is the default everywhere; "research" exists
#: only for the historical cloud-grader benchmark commands and must be set
#: explicitly per run (autograder bench ... --research).
EXECUTION_MODES: tuple[str, ...] = ("production", "research")

PRODUCTION = "production"
RESEARCH = "research"


class CloudBoundaryError(RuntimeError):
    """A production request would have reached a remote provider outside the
    cloud-OCR contract. Raised before serialization; nothing was sent."""

    def __init__(self, message: str, *, code: str, task: str | None = None):
        super().__init__(message)
        self.code = code
        self.task = task


def is_remote_route(backend: str | None, base_url: str | None) -> bool:
    """Would a request on this route leave the machine / private network?

    Stricter than ``usage.is_cloud_route``: an Ollama-native backend pointed
    at a PUBLIC host is remote here (nothing may leave, paid or not), while
    localhost/LAN endpoints of any backend are local. ``mock`` never leaves
    the process.
    """
    provider = effective_provider(backend, base_url)
    if provider == "mock":
        return False
    if provider in CLOUD_BACKENDS:
        return True
    # openai-compat, ollama-native, or anything URL-based: locality decides.
    # No URL means the backend's own default endpoint (localhost for ollama).
    return not _is_local_url(base_url)


# --------------------------------------------------------------------------
# The approved cloud-OCR contract
# --------------------------------------------------------------------------

def approved_cloud_ocr_systems() -> frozenset[str]:
    """The registered OCR system prompts that may accompany a remote OCR
    request. Imported lazily (prompts/escalation import gateway helpers).

    The registry is intentionally tiny: the lazy explanation transcriber and
    the INDEPENDENT verify transcriber (image -> its own exact reading;
    agreement is computed locally). The legacy fidelity-verdict prompt
    (escalation.OCR_VERIFY_SYSTEM) is deliberately NOT registered: it shows
    the verifier the primary reading, and survives only for the historical B2
    research benchmark under research mode. A new cloud OCR prompt must be
    added HERE, in code review — not smuggled through a task name.
    """
    from .escalation import OCR_VERIFY_INDEPENDENT_SYSTEM
    from .prompts import EXPLANATION_OCR_SYSTEM

    return frozenset({EXPLANATION_OCR_SYSTEM, OCR_VERIFY_INDEPENDENT_SYSTEM})


def forbidden_cloud_markers() -> tuple[str, ...]:
    """Exact strings whose presence in an outbound OCR payload proves grading
    material leaked into it. These are the section headers ``grade_prompt``
    emits plus the opening line of EVERY registered grading system prompt
    (derived from the registry, so a new prompt version is covered
    automatically) — deterministic tripwires, not heuristics."""
    from .escalation import GRADE_SYSTEM_BY_VERSION
    from .gradingpack import CONTEXT_HEADERS

    return tuple(CONTEXT_HEADERS) + tuple(
        p[:80] for p in GRADE_SYSTEM_BY_VERSION.values())


def _block_texts(content_blocks: list[dict] | None):
    for b in content_blocks or []:
        if isinstance(b, dict) and b.get("type") == "text":
            t = b.get("text")
            # a malformed non-string text still gets scanned (stringified)
            # rather than slipping past the tripwire unexamined
            yield t if isinstance(t, str) else repr(t)


def check_cloud_call(*, task: str, backend: str | None, base_url: str | None,
                     execution_mode: str, system: str | None = None,
                     content_blocks: list[dict] | None = None) -> None:
    """The production guard. Call BEFORE building/serializing any provider
    request. Raises ``CloudBoundaryError``; returns None when allowed.

    Research mode bypasses the boundary — it exists for the explicitly
    invoked cloud-grader benchmarks only, and every other construction site
    defaults to production.
    """
    if execution_mode not in EXECUTION_MODES:
        raise CloudBoundaryError(
            f"unknown execution mode {execution_mode!r} (expected one of "
            f"{list(EXECUTION_MODES)})", code="BAD_MODE", task=task)
    if execution_mode == RESEARCH:
        return
    if not is_remote_route(backend, base_url):
        return                                # local/mock: no boundary to cross
    provider = effective_provider(backend, base_url)
    if task not in CLOUD_OCR_ALLOWLIST:
        raise CloudBoundaryError(
            f"task {task!r} must not reach the remote provider {provider!r} in "
            "production: only OCR transcription "
            f"({', '.join(sorted(CLOUD_OCR_ALLOWLIST))}) may use the cloud. "
            "Grading runs locally (route the task to a local backend in "
            "models.toml). Historical cloud-grading benchmarks require the "
            "explicit research mode (autograder bench ... --research).",
            code="CLOUD_TASK_FORBIDDEN", task=task)
    if system is not None and system not in approved_cloud_ocr_systems():
        raise CloudBoundaryError(
            f"task {task!r}: the system prompt is not a registered cloud-OCR "
            "prompt. An OCR task name must not carry non-OCR instructions to "
            "a remote provider.", code="UNREGISTERED_OCR_PROMPT", task=task)
    markers = forbidden_cloud_markers()
    for text in _block_texts(content_blocks):
        for marker in markers:
            if marker and marker in text:
                raise CloudBoundaryError(
                    f"task {task!r}: outbound OCR payload contains grading "
                    f"material (matched a registered grading marker). Rubrics, "
                    "official solutions and course context never leave the "
                    "machine.", code="GRADING_CONTENT_IN_OCR_PAYLOAD", task=task)


__all__ = ["CLOUD_OCR_ALLOWLIST", "EXECUTION_MODES", "PRODUCTION", "RESEARCH",
           "CloudBoundaryError", "approved_cloud_ocr_systems",
           "forbidden_cloud_markers", "check_cloud_call", "is_remote_route"]
