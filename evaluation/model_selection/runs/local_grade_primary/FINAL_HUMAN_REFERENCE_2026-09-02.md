# Final human reference — FROZEN (2026-09-02 01:19:58)

sha256 `ce78aed115633883…` — 46 cases; sources 22 consensus / 22 adjudicated / 2 owner-repaired (distinct forever).

Class distribution: valid **28**, partially_valid **13**, invalid **5**.
The invalid class is now MEASURED on seen data (5 cases: e003_q2_r4, e003_q2_r6, e004_q1_r3, e004_q2_r4, e004_q2_r5); it remains UNMEASURED on HELD_OUT.

Baseline (one-pass 8B) exact per-class:

| class | recall | precision | F1 |
|---|---|---|---|
| valid | 24/28 = 0.8571 | 24/28 = 0.8571 | 0.8571 |
| partially_valid | 6/13 = 0.4615 | 6/12 = 0.5 | 0.48 |
| invalid | 1/5 = 0.2 | 1/6 = 0.1667 | 0.1818 |

exact 31/46, macro-F1 0.5063, balanced accuracy 0.5062, overgrades 8, undergrades 7.

Redundant final on e004_q2_r2: reopen = provenance-only cleanup, no numeric change (verdict 'valid' from consensus either way).
