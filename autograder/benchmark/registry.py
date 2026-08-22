"""Candidate registry — evaluation/model_selection/candidates.toml as DATA.

The registry lists, per role, the candidate slugs to benchmark and the
campaign budget. It is editable without code changes and carries no prices:
pricing is a mutable external fact that lives (for the local estimator only)
in models.toml [pricing], never in application logic. Winners are NOT
recorded here — every cloud role stays UNSELECTED until an owner reads a
benchmark report and sets the task's model in models.toml.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .manifests import REPO_ROOT, ROLES

DEFAULT_REGISTRY_PATH = REPO_ROOT / "evaluation" / "model_selection" / "candidates.toml"


@dataclass
class RoleCandidates:
    role: str
    status: str                      # UNSELECTED | SELECTED_LOCAL | ...
    gateway_task: str | None
    env_slug: str | None
    candidates: list[str] = field(default_factory=list)
    vision: bool = False
    benchmark: str = ""
    note: str = ""


@dataclass
class CandidateRegistry:
    path: Path
    version: int
    updated: str
    rule: str
    roles: dict[str, RoleCandidates]
    experiment_total_usd: float | None
    warn_usd: float | None

    def for_role(self, role: str) -> RoleCandidates:
        try:
            return self.roles[role]
        except KeyError:
            raise KeyError(f"role {role!r} is not in {self.path} "
                           f"(known: {sorted(self.roles)})") from None

    def is_listed(self, role: str, slug: str) -> bool:
        return slug in self.for_role(role).candidates

    def unselected_roles(self) -> list[str]:
        return sorted(r for r, rc in self.roles.items() if rc.status == "UNSELECTED")

    def summary(self) -> dict[str, Any]:
        return {
            "path": str(self.path), "version": self.version, "updated": self.updated,
            "budget": {"experiment_total_usd": self.experiment_total_usd, "warn_usd": self.warn_usd},
            "roles": {r: {"status": rc.status, "gateway_task": rc.gateway_task,
                          "candidates": list(rc.candidates), "benchmark": rc.benchmark}
                      for r, rc in self.roles.items()},
        }


def load_registry(path: Path | str = DEFAULT_REGISTRY_PATH) -> CandidateRegistry:
    p = Path(path)
    data = tomllib.loads(p.read_text(encoding="utf-8"))
    meta = data.get("meta", {})
    budget = data.get("budget", {})
    roles: dict[str, RoleCandidates] = {}
    for name, sec in (data.get("roles") or {}).items():
        roles[name] = RoleCandidates(
            role=name, status=str(sec.get("status", "UNSELECTED")),
            gateway_task=sec.get("gateway_task"), env_slug=sec.get("env_slug"),
            candidates=[str(c) for c in sec.get("candidates", [])],
            vision=bool(sec.get("vision", False)), benchmark=str(sec.get("benchmark", "")),
            note=str(sec.get("note", "")))
    for r in ROLES:
        if r not in roles:
            # A benchmark role missing from the registry is treated as
            # UNSELECTED with no candidates — the runner will refuse it.
            roles[r] = RoleCandidates(role=r, status="UNSELECTED", gateway_task=None, env_slug=None)
    return CandidateRegistry(
        path=p, version=int(meta.get("version", 0)), updated=str(meta.get("updated", "")),
        rule=str(meta.get("rule", "")), roles=roles,
        experiment_total_usd=(float(budget["experiment_total_usd"])
                              if "experiment_total_usd" in budget else None),
        warn_usd=(float(budget["warn_usd"]) if "warn_usd" in budget else None))


__all__ = ["DEFAULT_REGISTRY_PATH", "RoleCandidates", "CandidateRegistry", "load_registry"]
