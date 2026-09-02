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
re-enable cloud grading; only an explicit, pre-registered research
authorization can, and only for the task and model it names.

TWO LAYERS. Only the first one knows which execution mode it is in.

LAYER 1 — TASK / EXPERIMENT AUTHORIZATION (mode-dependent)

    production : a remote request is allowed only for a task in
                 ``CLOUD_OCR_ALLOWLIST`` (the two OCR transcription roles).
                 Grading, RAG-assisted grading, MC/variant/alignment
                 resolution, policy inference and anything added later are
                 refused by omission.
    research   : the same allowlist, PLUS whatever an explicit
                 ``ResearchAuthorization`` names — exact campaign, exact
                 task, exact model, no wildcards. ``--research`` on its own
                 authorizes nothing: with no authorization object it is
                 exactly as strict as production.

LAYER 2 — CONTENT / PAYLOAD SAFETY (always, every mode, not negotiable)

    * PROMPT — an OCR task may carry only a REGISTERED OCR system prompt
      (exact string match). ``gateway.call(task="ocr_primary",
      system=GRADE_SYSTEM, ...)`` is refused: an OCR role name must not
      become a tunnel for grading content.
    * TRIPWIRES — the content blocks of an OCR request must not contain
      grading material. The scan matches the exact section headers the
      grading prompt builder emits (rubric / official solution /
      course-context markers) plus every registered grading system prompt.
    * SECRETS — no block may carry anything credential-shaped, on any task.
    * BLOCK SHAPE — a campaign's declared image/text block limits.

Why the split: until 2026-09-02 ``--research`` returned from
``check_cloud_call`` before ANY layer ran, so a research OCR run silently
lost its registered-prompt check and its grading tripwires too. Widening
"which experiment may run" must never widen "what may be in the payload".
The Stage-1 OCR smoke's 24 requests were verified clean offline against the
production path, but the architecture allowed a leak it did not catch.

The boundary never inspects, stores, or transmits anything: it raises
``CloudBoundaryError`` before the backend serializes the request.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    The registry is intentionally tiny: the lazy explanation transcriber, the
    INDEPENDENT verify transcriber (image -> its own exact reading; agreement
    is computed locally), and — since 2026-09-02, for the pre-registered OCR
    validation campaign — the six frozen m2-strict-v1 bench transcription
    prompts, recovered byte-identically from the historical script (pure
    exact-transcription instructions; the payload tripwires and the task
    allowlist still apply unchanged). The legacy fidelity-verdict prompt
    (escalation.OCR_VERIFY_SYSTEM) is deliberately NOT registered: it shows
    the verifier the primary reading, and survives only for the historical B2
    research benchmark under research mode. A new cloud OCR prompt must be
    added HERE, in code review — not smuggled through a task name.
    """
    from .escalation import OCR_VERIFY_INDEPENDENT_SYSTEM
    from .prompts import EXPLANATION_OCR_SYSTEM

    registered = {EXPLANATION_OCR_SYSTEM, OCR_VERIFY_INDEPENDENT_SYSTEM}
    try:
        from .benchmark.roles import OCR_PROMPT_VERSIONS, load_ocr_prompts
        # Every registered OCR prompt VERSION, not just the historical set:
        # m2-strict-v1 (frozen) and ocr-neutral-v2 (the one-variable framing
        # treatment, whose rules block is copied verbatim from v1). A new
        # version must be added to OCR_PROMPT_VERSIONS in code review, exactly
        # as a new prompt had to be added here before.
        for _v in OCR_PROMPT_VERSIONS:
            registered |= set(load_ocr_prompts(_v).values())
    except Exception:  # noqa: BLE001 — fail closed: recovery failure just
        pass           # leaves the bench prompts unregistered
    return frozenset(registered)


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


#: Credential shapes that must never appear in an outbound payload. A key in a
#: content block is a leak whatever the task or the mode, so this scan is part
#: of the always-on content layer rather than any campaign's contract.
_SECRET_PATTERNS: tuple[tuple[str, str], ...] = (
    ("openrouter_key", "sk-or-"),
    ("openai_key", "sk-proj-"),
    ("anthropic_key", "sk-ant-"),
    ("aws_key", "AKIA"),
    ("bearer_header", "Authorization: Bearer "),
    ("private_key", "-----BEGIN "),
)


@dataclass(frozen=True)
class ResearchAuthorization:
    """An EXPLICIT, pre-registered permission to make a research call that
    production would refuse.

    Research mode is not a skeleton key. Before this existed, ``--research``
    returned from ``check_cloud_call`` before any layer ran, so an OCR research
    run also lost its registered-prompt check and its grading tripwires — the
    mode meant "skip every cloud safety check", which is never what an
    experiment needs. An authorization instead widens exactly ONE layer (which
    task, on which model) and widens nothing else.

    Every field is exact and finite; there is deliberately no wildcard. The
    block limits are the campaign's own payload shape (the Stage-1 OCR smoke
    sends one crop and no text at all), and default to unbounded so that
    authorizing a task cannot silently loosen a limit a caller never set.
    """

    campaign: str
    tasks: frozenset[str]
    models: frozenset[str]
    max_image_blocks: int | None = None
    max_text_blocks: int | None = None

    def authorizes(self, task: str, model: str | None) -> bool:
        if task not in self.tasks:
            return False
        # A campaign names its models; a call that cannot say which model it is
        # about has not proven it is the authorized one.
        return model is not None and model in self.models


def research_authorization(campaign: str, *, tasks, models,
                           max_image_blocks: int | None = None,
                           max_text_blocks: int | None = None) -> ResearchAuthorization:
    """Build a ResearchAuthorization, refusing the shapes that would make it
    meaningless (no campaign id, no task, no model, or a wildcard)."""
    t, m = frozenset(tasks), frozenset(models)
    if not campaign or not campaign.strip():
        raise ValueError("a research authorization must name its campaign")
    if not t or not m:
        raise ValueError("a research authorization must name at least one task and one model")
    for bad in ("*", "all", "any"):
        if bad in t or bad in m:
            raise ValueError(
                f"{bad!r} is not a task or model: research authorization is exact by design")
    return ResearchAuthorization(campaign=campaign.strip(), tasks=t, models=m,
                                 max_image_blocks=max_image_blocks,
                                 max_text_blocks=max_text_blocks)


def _authorize_task(*, task: str, provider: str, execution_mode: str,
                    research_auth: "ResearchAuthorization | None",
                    model: str | None) -> None:
    """LAYER 1 — may this task reach this remote provider at all?

    Production and research differ ONLY here, and only by what an explicit
    pre-registration names.
    """
    if task in CLOUD_OCR_ALLOWLIST:
        return                       # the standing cloud-OCR contract, both modes
    if execution_mode == RESEARCH and research_auth is not None:
        if research_auth.authorizes(task, model):
            return
        raise CloudBoundaryError(
            f"task {task!r} on model {model!r} is not covered by research "
            f"authorization {research_auth.campaign!r} (authorized tasks: "
            f"{sorted(research_auth.tasks)}; models: {sorted(research_auth.models)}). "
            "Research mode widens the task layer only as far as an explicit "
            "pre-registration says.",
            code="RESEARCH_TASK_NOT_AUTHORIZED", task=task)
    hint = ("Grading runs locally (route the task to a local backend in "
            "models.toml). A historical cloud-grading benchmark needs research "
            "mode AND an explicit ResearchAuthorization naming this task and model.")
    raise CloudBoundaryError(
        f"task {task!r} must not reach the remote provider {provider!r} in "
        f"{execution_mode}: only OCR transcription "
        f"({', '.join(sorted(CLOUD_OCR_ALLOWLIST))}) may use the cloud. " + hint,
        code="CLOUD_TASK_FORBIDDEN", task=task)


def _check_payload(*, task: str, system: str | None,
                   content_blocks: list[dict] | None,
                   research_auth: "ResearchAuthorization | None") -> None:
    """LAYER 2 — content safety. ALWAYS enforced, in every execution mode.

    No authorization can switch this off: a campaign may say which task runs,
    never that a rubric may ride along with a crop.
    """
    blocks = content_blocks or []
    # -- secrets: never outbound, whatever the task ---------------------------
    for text in _block_texts(blocks) if blocks else ():
        for name, pat in _SECRET_PATTERNS:
            if pat in text:
                raise CloudBoundaryError(
                    f"task {task!r}: outbound payload contains what looks like a "
                    f"credential ({name}). Nothing that resembles a secret leaves "
                    "the machine.", code="SECRET_IN_PAYLOAD", task=task)
    # -- campaign block-shape limits -----------------------------------------
    if research_auth is not None:
        n_img = sum(1 for b in blocks if isinstance(b, dict) and b.get("type") == "image")
        n_txt = sum(1 for b in blocks if isinstance(b, dict) and b.get("type") == "text")
        if research_auth.max_image_blocks is not None and n_img > research_auth.max_image_blocks:
            raise CloudBoundaryError(
                f"task {task!r}: {n_img} image blocks, but campaign "
                f"{research_auth.campaign!r} authorizes at most "
                f"{research_auth.max_image_blocks}. Sending a second page is a "
                "different experiment.", code="TOO_MANY_IMAGE_BLOCKS", task=task)
        if research_auth.max_text_blocks is not None and n_txt > research_auth.max_text_blocks:
            raise CloudBoundaryError(
                f"task {task!r}: {n_txt} text blocks, but campaign "
                f"{research_auth.campaign!r} authorizes at most "
                f"{research_auth.max_text_blocks}.",
                code="TOO_MANY_TEXT_BLOCKS", task=task)
    # -- the OCR content contract --------------------------------------------
    # Scoped to OCR tasks: an authorized cloud-grading benchmark legitimately
    # CARRIES rubric text, so scanning it for grading markers would be
    # incoherent. What must never happen is grading material travelling under
    # an OCR task name — and that is exactly what these two layers stop.
    if task not in CLOUD_OCR_ALLOWLIST:
        return
    if system is not None and system not in approved_cloud_ocr_systems():
        raise CloudBoundaryError(
            f"task {task!r}: the system prompt is not a registered cloud-OCR "
            "prompt. An OCR task name must not carry non-OCR instructions to "
            "a remote provider.", code="UNREGISTERED_OCR_PROMPT", task=task)
    markers = forbidden_cloud_markers()
    for text in _block_texts(blocks):
        for marker in markers:
            if marker and marker in text:
                raise CloudBoundaryError(
                    f"task {task!r}: outbound OCR payload contains grading "
                    f"material (matched a registered grading marker). Rubrics, "
                    "official solutions and course context never leave the "
                    "machine.", code="GRADING_CONTENT_IN_OCR_PAYLOAD", task=task)


def check_cloud_call(*, task: str, backend: str | None, base_url: str | None,
                     execution_mode: str, system: str | None = None,
                     content_blocks: list[dict] | None = None,
                     research_auth: "ResearchAuthorization | None" = None,
                     model: str | None = None) -> None:
    """The guard. Call BEFORE building/serializing any provider request.
    Raises ``CloudBoundaryError``; returns None when allowed.

    Two layers, and only the first one knows what mode it is in:

    1. TASK/EXPERIMENT AUTHORIZATION — production allows the cloud-OCR
       allowlist; research allows that plus whatever an explicit
       ``ResearchAuthorization`` names (exact task AND exact model).
       ``--research`` with no authorization is exactly as strict as production.
    2. CONTENT/PAYLOAD SAFETY — registered OCR prompt, grading tripwires,
       secret scan and the campaign's block-shape limits. Enforced in EVERY
       mode; no authorization can disable it.
    """
    if execution_mode not in EXECUTION_MODES:
        raise CloudBoundaryError(
            f"unknown execution mode {execution_mode!r} (expected one of "
            f"{list(EXECUTION_MODES)})", code="BAD_MODE", task=task)
    if research_auth is not None and execution_mode != RESEARCH:
        raise CloudBoundaryError(
            "a research authorization was supplied outside research mode; "
            "production is never widened by a pre-registration.",
            code="RESEARCH_AUTH_IN_PRODUCTION", task=task)
    if not is_remote_route(backend, base_url):
        return                                # local/mock: no boundary to cross
    _authorize_task(task=task, provider=effective_provider(backend, base_url),
                    execution_mode=execution_mode, research_auth=research_auth,
                    model=model)
    _check_payload(task=task, system=system, content_blocks=content_blocks,
                   research_auth=research_auth)


__all__ = ["CLOUD_OCR_ALLOWLIST", "EXECUTION_MODES", "PRODUCTION", "RESEARCH",
           "CloudBoundaryError", "ResearchAuthorization", "research_authorization",
           "approved_cloud_ocr_systems", "forbidden_cloud_markers",
           "check_cloud_call", "is_remote_route"]
