"""Exact binomial uncertainty for rare severe grading errors.

Zero observed catastrophic errors (invalid -> valid full credit) over a small
number of actually-invalid cases is NOT evidence that the true rate is zero.
This module quantifies exactly what the seen data can and cannot rule out,
with deterministic pure-Python mathematics (no scipy dependency).

Formulas (documented per the pre-registration requirement):

- One-sided exact (Clopper-Pearson) upper bound for k successes in n trials
  at confidence 1-alpha: the smallest p_u such that
      P(X <= k | n, p_u) <= alpha.
  For k = 0 this closes to  p_u = 1 - alpha**(1/n)  ("rule of three" exact
  form). For k > 0 it is found by deterministic bisection on the binomial
  CDF (monotone in p), to within BISECTION_TOL.

- One-sided exact lower bound: the largest p_l such that
      P(X >= k | n, p_l) <= alpha;  p_l = 0 when k = 0.

- Two-sided (1-alpha) Clopper-Pearson interval: [lower(alpha/2), upper(alpha/2)].

- Minimum sample size for a zero-event demonstration: the smallest n with
      upper(0, n, alpha) <= bound   <=>   (1 - bound)**n <= alpha
      <=>   n >= ln(alpha) / ln(1 - bound).

Everything here is deterministic; there is no randomness and no model call.
"""
from __future__ import annotations

import math

RARE_EVENT_MATH_VERSION = "rare-events-v1"
BISECTION_TOL = 1e-10


def binomial_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), computed with exact log terms."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    total = 0.0
    log_p, log_q = math.log(p), math.log(1.0 - p)
    for i in range(0, k + 1):
        total += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1)
                          - math.lgamma(n - i + 1) + i * log_p + (n - i) * log_q)
    return min(total, 1.0)


def exact_upper_bound(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided exact (Clopper-Pearson) upper bound at confidence 1-alpha."""
    if not (0 <= k <= n) or n <= 0 or not (0.0 < alpha < 1.0):
        raise ValueError(f"invalid inputs k={k} n={n} alpha={alpha}")
    if k == n:
        return 1.0
    if k == 0:
        return 1.0 - alpha ** (1.0 / n)
    lo, hi = k / n, 1.0
    while hi - lo > BISECTION_TOL:
        mid = (lo + hi) / 2.0
        if binomial_cdf(k, n, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def exact_lower_bound(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided exact (Clopper-Pearson) lower bound at confidence 1-alpha."""
    if not (0 <= k <= n) or n <= 0 or not (0.0 < alpha < 1.0):
        raise ValueError(f"invalid inputs k={k} n={n} alpha={alpha}")
    if k == 0:
        return 0.0
    if k == n:
        return alpha ** (1.0 / n)
    lo, hi = 0.0, k / n
    while hi - lo > BISECTION_TOL:
        mid = (lo + hi) / 2.0
        # P(X >= k | p) = 1 - P(X <= k-1 | p), increasing in p
        if 1.0 - binomial_cdf(k - 1, n, mid) > alpha:
            hi = mid
        else:
            lo = mid
    return lo


def two_sided_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided (1-alpha) exact Clopper-Pearson interval."""
    return (exact_lower_bound(k, n, alpha / 2.0),
            exact_upper_bound(k, n, alpha / 2.0))


def min_n_for_zero_event_bound(bound: float, alpha: float = 0.05) -> int:
    """Smallest n such that observing ZERO events yields a one-sided
    (1-alpha) upper bound <= `bound`."""
    if not (0.0 < bound < 1.0) or not (0.0 < alpha < 1.0):
        raise ValueError(f"invalid inputs bound={bound} alpha={alpha}")
    n = math.ceil(math.log(alpha) / math.log(1.0 - bound))
    # guard against floating-point edge: verify and adjust deterministically
    while exact_upper_bound(0, n, alpha) > bound:
        n += 1
    while n > 1 and exact_upper_bound(0, n - 1, alpha) <= bound:
        n -= 1
    return n


def severe_event_report(k: int, n: int, alpha: float = 0.05) -> dict:
    """A complete uncertainty record for one severe-event cell."""
    return {
        "math_version": RARE_EVENT_MATH_VERSION,
        "observed": k,
        "denominator": n,
        "point_estimate": round(k / n, 6) if n else None,
        "one_sided_upper_95": round(exact_upper_bound(k, n, alpha), 6),
        "two_sided_95": [round(x, 6) for x in two_sided_interval(k, n, alpha)],
        "alpha": alpha,
        "note": ("zero observed events does NOT demonstrate a zero rate; "
                 "the upper bound is what the data can actually exclude"),
    }
