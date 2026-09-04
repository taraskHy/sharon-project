"""The OCR decision registry: which model/prompt/route configurations have been
measured and RULED OUT, so a dropped arm cannot be re-run by accident.

This is a *research* guard, not production routing. It answers one question at
run time: has this exact (model, prompt_version, provider-pin) already been
measured and dropped? A dropped arm still runs — but only when the caller names
the new experiment that authorizes it, so the re-run is a deliberate, recorded
act rather than a slip.

Route identity matters. ``google/gemini-3.7-flash`` under automatic OpenRouter
routing is NOT the same configuration as the same slug pinned to a single
serving provider: the catalog exposes six endpoints across two distinct
providers (Google Vertex and Google AI Studio) for that slug, the two 32-crop
arms were served by a mix of both without control, and no content-filtered row
in any arm records which provider produced it. A pinned route is therefore a
genuinely different experiment, and the registry says so explicitly rather than
letting a pin silently inherit a drop — or silently evade one.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path("evaluation/model_selection/policies/ocr_decision_registry.json")

#: statuses that forbid an unauthorized live run
BLOCKING = ("DROP", "DROP_AS_PRIMARY_ROUTE", "REJECTED")
#: a status that permits running only as an explicitly labelled control
CONTROL_ONLY = "HISTORICAL_CONTROL_ONLY"


class DroppedConfiguration(RuntimeError):
    """Raised when a live run would execute a configuration already dropped."""


class RegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class Decision:
    entry_id: str
    status: str
    model: str
    prompt_version: str
    provider_pin: str | None
    reason: str
    evidence: str

    @property
    def blocking(self) -> bool:
        return self.status in BLOCKING


def _canonical(doc: dict, field: str = "content_sha256") -> str:
    body = json.dumps({k: v for k, v in doc.items() if k != field},
                      ensure_ascii=False, indent=1, sort_keys=True, default=str)
    return hashlib.sha256(body.encode()).hexdigest()


def load_registry(path: Path | str | None = None) -> dict[str, Any]:
    """Load and self-hash-verify the registry. A tampered registry is refused
    outright — a guard that can be edited without detection is not a guard."""
    p = Path(path or REGISTRY_PATH)
    if not p.exists():
        raise RegistryError(f"OCR decision registry missing: {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    if _canonical(doc) != doc.get("content_sha256"):
        raise RegistryError(f"OCR decision registry failed its self-hash check: {p}")
    return doc


def _matches(entry: dict, model: str, prompt_version: str | None,
             provider_pin: str | None) -> bool:
    m = entry["match"]
    if m["model"] != model:
        return False
    if m["prompt_version"] != "*" and m["prompt_version"] != prompt_version:
        return False
    want = m["provider_pin"]
    if want == "*":
        return True
    # null in the registry means "automatic routing only" — an explicit pin is
    # a different configuration and does not match.
    return want == provider_pin


def decisions_for(model: str, prompt_version: str | None = None,
                  provider_pin: str | None = None,
                  registry: dict | None = None) -> list[Decision]:
    doc = registry if registry is not None else load_registry()
    out = []
    for e in doc["entries"]:
        if _matches(e, model, prompt_version, provider_pin):
            out.append(Decision(entry_id=e["id"], status=e["status"], model=e["match"]["model"],
                                prompt_version=e["match"]["prompt_version"],
                                provider_pin=e["match"]["provider_pin"],
                                reason=e["reason"], evidence=e["evidence"]))
    return out


def provider_pin_of(route: Any) -> str | None:
    """The single pinned provider slug of a route, or None for automatic
    routing. A pin with fallbacks enabled, or more than one provider in the
    order, is NOT deterministic and is reported as unpinned.

    The routing object lives in one of two places depending on how far down the
    stack the route has travelled. ``TaskRoute`` carries it as a TOP-LEVEL
    ``provider`` field and only folds it into ``extra_generation`` inside
    ``to_backend_config()``. Reading only ``extra_generation`` therefore made a
    genuinely pinned TaskRoute look like automatic routing — which is how the
    first live attempt of the alt-candidate screen was refused as a dropped
    auto-route arm. Both shapes are accepted, top level first.
    """
    prov = None
    if isinstance(route, dict):
        prov = route.get("provider")
        if not isinstance(prov, dict):
            eg = route.get("extra_generation")
            prov = eg.get("provider") if isinstance(eg, dict) else None
    else:
        prov = getattr(route, "provider", None)
        if not isinstance(prov, dict):
            eg = getattr(route, "extra_generation", None)
            prov = eg.get("provider") if isinstance(eg, dict) else None
    if not isinstance(prov, dict):
        return None
    order = prov.get("order")
    if not isinstance(order, list) or len(order) != 1:
        return None
    if prov.get("allow_fallbacks", True):
        return None            # a fallback can silently change the provider
    return str(order[0])


def assert_selectable(model: str, prompt_version: str | None, provider_pin: str | None,
                      *, authorized_experiment: str | None = None,
                      registry: dict | None = None) -> list[Decision]:
    """Refuse a live run of a dropped configuration unless an explicit new
    experiment authorizes it. Returns the (possibly empty) decisions that
    matched, so the caller can record them."""
    hits = decisions_for(model, prompt_version, provider_pin, registry=registry)
    blocking = [d for d in hits if d.blocking]
    if blocking and not authorized_experiment:
        d = blocking[0]
        pin = provider_pin or "automatic routing"
        raise DroppedConfiguration(
            f"{model!r} with prompt {prompt_version!r} on {pin} is recorded as "
            f"{d.status} in the OCR decision registry ({d.entry_id}): {d.reason} "
            f"Re-running it requires an explicit research override naming the new "
            f"experiment that authorizes it.")
    return hits


def current_winner(registry: dict | None = None) -> str | None:
    """The selected production OCR route, or None. A dropped configuration can
    never be returned here — that invariant is the point of the registry."""
    doc = registry if registry is not None else load_registry()
    w = doc.get("current_winner")
    if w is None:
        return None
    hits = decisions_for(w.get("model"), w.get("prompt_version"), w.get("provider_pin"),
                         registry=doc)
    if any(d.blocking for d in hits):
        raise RegistryError(
            f"registry is inconsistent: current_winner {w.get('model')!r} is also "
            f"recorded as dropped. Refusing to report a dropped arm as the winner.")
    return w.get("model")


__all__ = ["Decision", "DroppedConfiguration", "RegistryError", "BLOCKING", "CONTROL_ONLY",
           "load_registry", "decisions_for", "assert_selectable", "current_winner",
           "provider_pin_of", "REGISTRY_PATH"]
