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
    #: candidate slug -> route parameter overrides for THIS role.
    #:
    #: Providers do not offer identical inference controls. Forcing one
    #: configuration on all of them either excludes a deployable model or
    #: benchmarks it in a state it cannot actually run in. The benchmark selects
    #: the best model/CONFIGURATION pair, so a documented asymmetry is data, not
    #: a bug — but it must be declared here (never inferred at runtime) so it
    #: travels into the run fingerprint and the cost prediction.
    candidate_overrides: dict = field(default_factory=dict)

    def overrides_for(self, slug: str) -> dict:
        return dict(self.candidate_overrides.get(slug) or {})


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
                          "candidates": list(rc.candidates), "benchmark": rc.benchmark,
                          "candidate_overrides": rc.candidate_overrides}
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
            note=str(sec.get("note", "")),
            candidate_overrides={k: dict(v) for k, v in
                                 (sec.get("candidate_overrides") or {}).items()})
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
