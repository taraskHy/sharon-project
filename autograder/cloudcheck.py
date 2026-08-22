"""Human-readable explanations for "the cloud is not ready" failures.

Every cloud-dependent operation (grading in reliability mode, a benchmark
run, a settings probe) can fail for one of a few well-known, *expected*
reasons while the product is still being set up:

* the role has no selected model yet (``model = "UNSELECTED"`` or an unset
  ``${ENV}`` slug)                      -> "grade_primary model is not selected"
* the OpenRouter credential is missing -> "OpenRouter credential is not configured"
* the task is absent/disabled in models.toml
* the experiment budget is exhausted

These are product states, not programming errors, so the CLI and the GUI
must explain them in one sentence instead of printing a stack trace. This
module turns the underlying exceptions (GatewayConfigError, BackendError,
BudgetExceeded, ...) into that sentence. It never calls a provider and never
reads the credential's value — only whether it is set.
"""
from __future__ import annotations

import os
import re

from .backends.openrouter import OPENROUTER_KEY_ENV

_UNSELECTED_RE = re.compile(r"task '([^']+)' is UNSELECTED")
_NO_MODEL_RE = re.compile(r"task '([^']+)' has no model configured")
_NO_ROUTE_RE = re.compile(r"no route configured for task '([^']+)'")
_DISABLED_RE = re.compile(r"task '([^']+)' is disabled")


class CloudNotReady(RuntimeError):
    """A cloud-dependent operation cannot run yet; ``str(exc)`` is the
    one-sentence, user-facing explanation. ``code`` is stable for the GUI."""

    def __init__(self, message: str, *, code: str, task: str | None = None):
        super().__init__(message)
        self.code = code
        self.task = task


def openrouter_credential_present() -> bool:
    """True iff the OpenRouter credential is set in the environment. The
    value itself is never read into application state, logged or shown."""
    return bool(os.environ.get(OPENROUTER_KEY_ENV))


def explain_cloud_error(exc: BaseException) -> CloudNotReady | None:
    """Map a low-level exception to a CloudNotReady explanation, or None when
    the exception is not one of the expected setup states (let it propagate).
    """
    text = str(exc)
    name = type(exc).__name__
    m = _UNSELECTED_RE.search(text)
    if m:
        return CloudNotReady(f"{m.group(1)} model is not selected", code="UNSELECTED",
                             task=m.group(1))
    m = _NO_MODEL_RE.search(text)
    if m:
        return CloudNotReady(
            f"{m.group(1)} model is not selected (its ${{ENV}} slug is unset)",
            code="UNSELECTED", task=m.group(1))
    m = _NO_ROUTE_RE.search(text)
    if m:
        return CloudNotReady(f"{m.group(1)} is not configured in models.toml",
                             code="NO_ROUTE", task=m.group(1))
    m = _DISABLED_RE.search(text)
    if m:
        return CloudNotReady(f"{m.group(1)} is disabled in models.toml",
                             code="DISABLED", task=m.group(1))
    if OPENROUTER_KEY_ENV in text or "OpenRouter backend requires" in text:
        return CloudNotReady(
            "OpenRouter credential is not configured "
            f"(set {OPENROUTER_KEY_ENV} in the environment; it is never stored in a file)",
            code="NO_CREDENTIAL")
    if name == "BudgetExceeded" or "hard budget" in text:
        return CloudNotReady(f"experiment budget exhausted: {text}", code="BUDGET")
    return None


def require_cloud_task(gateway, task: str) -> None:
    """Raise CloudNotReady with a friendly sentence when ``task`` cannot be
    used right now (UNSELECTED / unconfigured / disabled / no credential).
    Makes NO network call: the OpenRouter credential is checked by presence
    only; no backend object is constructed."""
    try:
        route = gateway.route(task)
    except Exception as e:  # noqa: BLE001 — translated below
        friendly = explain_cloud_error(e)
        if friendly is not None:
            raise friendly from e
        raise
    from .usage import is_cloud_route

    if is_cloud_route(route.backend, route.base_url) and not openrouter_credential_present():
        raise CloudNotReady(
            "OpenRouter credential is not configured "
            f"(set {OPENROUTER_KEY_ENV} in the environment; it is never stored in a file)",
            code="NO_CREDENTIAL", task=task)


__all__ = ["CloudNotReady", "explain_cloud_error", "require_cloud_task",
           "openrouter_credential_present"]
