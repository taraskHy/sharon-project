"""Hard collaboration-run budgets (spec section 14).

Every limit is checked BEFORE a reviewer call (call count, projected input
tokens, already-consumed output tokens / cost) and usage is recorded after.
Reaching any hard limit stops the run with BUDGET_EXHAUSTED — never a silent
continue. Cache hits consume no budget.

Token numbers are provider-reported when available, otherwise the
deterministic chars/4 estimate; both are recorded as-is in the audit log.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from .config import BudgetCfg


@dataclass
class BudgetState:
    reviewer_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "BudgetState":
        data = data or {}
        return cls(
            reviewer_calls=int(data.get("reviewer_calls", 0)),
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            cost_usd=float(data.get("cost_usd", 0.0)),
        )


class BudgetTracker:
    def __init__(self, cfg: BudgetCfg, state: BudgetState):
        self.cfg = cfg
        self.state = state

    def check_next_call(self, est_input_tokens: int) -> str | None:
        """Return a stop reason if the NEXT reviewer call would break a limit."""
        cfg, st = self.cfg, self.state
        if cfg.max_reviewer_calls > 0 and st.reviewer_calls + 1 > cfg.max_reviewer_calls:
            return (
                f"max_reviewer_calls reached ({st.reviewer_calls}/"
                f"{cfg.max_reviewer_calls})"
            )
        if (
            cfg.max_input_tokens > 0
            and st.input_tokens + est_input_tokens > cfg.max_input_tokens
        ):
            return (
                f"max_input_tokens would be exceeded ({st.input_tokens} used + "
                f"{est_input_tokens} needed > {cfg.max_input_tokens})"
            )
        if cfg.max_output_tokens > 0 and st.output_tokens >= cfg.max_output_tokens:
            return (
                f"max_output_tokens reached ({st.output_tokens}/"
                f"{cfg.max_output_tokens})"
            )
        if cfg.max_cost_usd > 0 and st.cost_usd >= cfg.max_cost_usd:
            return f"max_cost_usd reached ({st.cost_usd:.4f}/{cfg.max_cost_usd})"
        return None

    def record_call(
        self, input_tokens: int, output_tokens: int, cost_usd: float | None
    ) -> None:
        self.state.reviewer_calls += 1
        self.state.input_tokens += max(0, int(input_tokens))
        self.state.output_tokens += max(0, int(output_tokens))
        if cost_usd:
            self.state.cost_usd += float(cost_usd)

    def post_call_exceeded(self) -> str | None:
        """A limit crossed by the call that just completed (checked after)."""
        cfg, st = self.cfg, self.state
        if cfg.max_output_tokens > 0 and st.output_tokens >= cfg.max_output_tokens:
            return (
                f"max_output_tokens reached ({st.output_tokens}/"
                f"{cfg.max_output_tokens})"
            )
        if cfg.max_cost_usd > 0 and st.cost_usd >= cfg.max_cost_usd:
            return f"max_cost_usd reached ({st.cost_usd:.4f}/{cfg.max_cost_usd})"
        return None
