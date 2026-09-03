"""One immutable spend envelope shared by every arm of a multi-arm campaign.

The failure this prevents is subtle and expensive. The benchmark ceilings are
CUMULATIVE — they are compared against the whole persisted ledger — so the
natural way to express "this campaign may spend $0.12" is
``--hard-usd $(ledger_now + 0.12)``. Computed that way, arm 1 gets
``L0 + 0.12``; but by the time arm 2 starts, ``ledger_now`` has grown to
``L0 + spend1``, and arm 2 is handed ``L0 + spend1 + 0.12``. Each arm silently
receives a fresh allowance and a three-arm screen can spend three times its
authorization while every individual arm passes its own gate.

The fix is to compute the thresholds ONCE, from a starting ledger ``L0``
captured before the first arm, and persist them. Every arm then loads the same
absolute numbers. ``L0`` is immutable: reopening the manifest never recomputes
it from a later ledger.

The manifest is self-hashed, and :func:`load_campaign_budget` verifies that
hash, so the envelope cannot be widened mid-campaign by editing the file.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class CampaignBudgetError(RuntimeError):
    pass


class CampaignBudgetExceeded(RuntimeError):
    """A request could cross the campaign's fixed hard threshold."""


def _seal(doc: dict, field: str = "content_sha256") -> dict:
    body = json.dumps({k: v for k, v in doc.items() if k != field},
                      ensure_ascii=False, indent=1, sort_keys=True, default=str)
    doc[field] = hashlib.sha256(body.encode()).hexdigest()
    return doc


@dataclass(frozen=True)
class CampaignBudget:
    campaign: str
    experiment_sha256: str
    starting_ledger_usd: float          # L0 — captured once, never recomputed
    warning_increment_usd: float        # the campaign WARNING increment
    hard_increment_usd: float           # the campaign HARD increment
    warn_usd: float                     # absolute: L0 + warning increment
    hard_usd: float                     # absolute: L0 + hard increment
    predicted_arm_costs: dict[str, float]
    path: Path | None = None

    @property
    def predicted_campaign_worst_case_usd(self) -> float:
        return round(sum(self.predicted_arm_costs.values()), 6)

    def remaining_usd(self, ledger_now: float) -> float:
        return round(self.hard_usd - ledger_now, 8)

    def check(self, *, ledger_now: float, max_request_cost_usd: float) -> None:
        """Reserve the MAXIMUM this request could cost and refuse if that
        would cross the fixed hard threshold.

        The reservation is deliberately the worst case, not an expectation: a
        ceiling that admits a request on the basis of its average cost has not
        bounded anything.
        """
        projected = float(ledger_now) + float(max_request_cost_usd or 0.0)
        if projected > self.hard_usd:
            raise CampaignBudgetExceeded(
                f"campaign {self.campaign!r}: this request could bring cumulative spend to "
                f"${projected:.6f}, crossing the fixed hard threshold ${self.hard_usd:.6f} "
                f"(L0 ${self.starting_ledger_usd:.6f} + ${self.hard_increment_usd:.2f}). "
                f"The threshold is fixed for the whole campaign and is NOT recomputed per arm.")

    def warning_state(self, ledger_now: float) -> str:
        if ledger_now >= self.hard_usd:
            return "HARD"
        if ledger_now >= self.warn_usd:
            return "WARNING"
        return "OK"

    def to_json(self) -> dict[str, Any]:
        return _seal({
            "artifact": "campaign_budget",
            "campaign": self.campaign,
            "experiment_sha256": self.experiment_sha256,
            "starting_ledger_usd_L0": self.starting_ledger_usd,
            "warning_increment_usd": self.warning_increment_usd,
            "hard_increment_usd": self.hard_increment_usd,
            "warn_usd_absolute": self.warn_usd,
            "hard_usd_absolute": self.hard_usd,
            "predicted_arm_costs_usd": self.predicted_arm_costs,
            "predicted_campaign_worst_case_usd": self.predicted_campaign_worst_case_usd,
            "immutability_rule":
                "L0 is captured ONCE, before the first arm. warn_usd and hard_usd are absolute "
                "and are reused verbatim by every arm. They are never recomputed from a later "
                "ledger — that is what would hand each arm a fresh allowance.",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        })


def create_campaign_budget(*, campaign: str, experiment_sha256: str,
                           starting_ledger_usd: float,
                           warning_increment_usd: float, hard_increment_usd: float,
                           predicted_arm_costs: dict[str, float],
                           path: str | Path) -> CampaignBudget:
    """Create and persist the envelope. Refuses to overwrite an existing one:
    a campaign gets exactly one L0."""
    p = Path(path)
    if p.exists():
        raise CampaignBudgetError(
            f"campaign budget already exists at {p}; L0 is immutable and a campaign gets exactly "
            "one. Delete it deliberately if you are genuinely starting a new campaign.")
    if hard_increment_usd < warning_increment_usd:
        raise CampaignBudgetError("hard increment must be >= warning increment")
    b = CampaignBudget(
        campaign=campaign, experiment_sha256=experiment_sha256,
        starting_ledger_usd=round(float(starting_ledger_usd), 8),
        warning_increment_usd=float(warning_increment_usd),
        hard_increment_usd=float(hard_increment_usd),
        warn_usd=round(float(starting_ledger_usd) + float(warning_increment_usd), 8),
        hard_usd=round(float(starting_ledger_usd) + float(hard_increment_usd), 8),
        predicted_arm_costs=dict(predicted_arm_costs), path=p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(b.to_json(), ensure_ascii=False, indent=1, default=str),
                 encoding="utf-8", newline="\n")
    return b


def load_campaign_budget(path: str | Path) -> CampaignBudget:
    """Load and self-hash-verify. The thresholds come from the file exactly as
    written; nothing here consults the current ledger."""
    p = Path(path)
    if not p.exists():
        raise CampaignBudgetError(f"campaign budget manifest missing: {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    check = _seal({k: v for k, v in doc.items() if k != "content_sha256"})
    if check["content_sha256"] != doc.get("content_sha256"):
        raise CampaignBudgetError(
            f"campaign budget manifest failed its self-hash check: {p}. The spend envelope "
            "cannot be widened mid-campaign.")
    b = CampaignBudget(
        campaign=doc["campaign"], experiment_sha256=doc["experiment_sha256"],
        starting_ledger_usd=doc["starting_ledger_usd_L0"],
        warning_increment_usd=doc["warning_increment_usd"],
        hard_increment_usd=doc["hard_increment_usd"],
        warn_usd=doc["warn_usd_absolute"], hard_usd=doc["hard_usd_absolute"],
        predicted_arm_costs=doc.get("predicted_arm_costs_usd") or {}, path=p)
    # the absolute thresholds must still equal L0 + increments
    if round(b.starting_ledger_usd + b.hard_increment_usd, 8) != round(b.hard_usd, 8):
        raise CampaignBudgetError("hard_usd is not L0 + hard increment; manifest is inconsistent")
    if round(b.starting_ledger_usd + b.warning_increment_usd, 8) != round(b.warn_usd, 8):
        raise CampaignBudgetError("warn_usd is not L0 + warning increment; manifest is inconsistent")
    return b


__all__ = ["CampaignBudget", "CampaignBudgetError", "CampaignBudgetExceeded",
           "create_campaign_budget", "load_campaign_budget"]
