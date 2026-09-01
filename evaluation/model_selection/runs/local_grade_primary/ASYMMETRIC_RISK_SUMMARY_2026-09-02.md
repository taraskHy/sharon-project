# Asymmetric-risk summary — 2026-09-02

Policy `asymmetric_grading_risk_v1` v1 `11e65e79e0f36cf6…`; reference `ce78aed115633883…`; git `cfe7a02a0145`. Zero-inference analysis; seen development data only.

## Strict weighted risk (46 cases)

| arm | total | mean | inv->val | pv->val | over-loss | under-loss | exact |
|---|---|---|---|---|---|---|---|
| baseline_8b_one_pass | 43 | 0.9348 | 0 | 4 | 32 | 11 | 31/46 |
| arm_a_q8_0 | 40 | 0.8696 | 0 | 3 | 24 | 16 | 28/46 |
| arm_b_two_pass | 43 | 0.9348 | 0 | 4 | 32 | 11 | 31/46 |
| always_invalid | 97 | 2.1087 | 0 | 0 | 0 | 97 | 5/46 |
| always_partially_valid | 43 | 0.9348 | 0 | 0 | 15 | 28 | 13/46 |
| always_valid | 125 | 2.7174 | 5 | 13 | 125 | 0 | 28/46 |

**No arm materially beats the best constant policy (`always_partially_valid`, 43).** Margins vs it: {'baseline_8b_one_pass': 0, 'arm_a_q8_0': 3, 'arm_b_two_pass': 0} — the largest is 7.0% — below the 10% material threshold and not robust (see sensitivity).

## Disagreement-aware (clean 31 cases, weight 26.0)

| arm | clean loss | per weight unit |
|---|---|---|
| baseline_8b_one_pass | 20.0 | 0.7692 |
| arm_a_q8_0 | 18.0 | 0.6923 |
| arm_b_two_pass | 20.0 | 0.7692 |

## Sensitivity

72 matrices; winner frequency {'always_partially_valid': 16, 'arm_a_q8_0': 32, 'arm_b_two_pass': 36, 'baseline_8b_one_pass': 36}; a constant policy co-wins in 16 matrices; arm_a beats baseline in only 36/72 (flips on the adjacent-undergrade cost). **Model ranking is NOT stable under the matrix.**

## Release gates (proposed release_gates_asym_v1 — evaluated on baseline_8b_one_pass + HUMAN_DISPUTE_AWARE_C unless noted; ALL estimates seen-data only)

| gate | target | observed | verdict | evidence |
|---|---|---|---|---|
| HARD_FALSE_FULL | confirmed invalid -> automatic valid = 0 | 0 observed (all arms/policies 0; only 5 seen invalid cases — low power) | PASS | PRODUCTION_POLICY_REPLAY |
| SERIOUS_OVERGRADE | automatic partially_valid -> valid <= 2/46 | 2 (AUTO_ALL: 4) | PASS | PRODUCTION_POLICY_REPLAY |
| WEIGHTED_RISK | semantic-layer total risk <= 0.90 x best constant (always_partially_valid = 43) | baseline 43 (needs <= 38.7) | FAIL | ASYMMETRIC_RISK_STRICT |
| UNDERGRADE_CAP | automatic harmful undergrades <= 3/46 | 1 under HUMAN_DISPUTE_AWARE_C (AUTO_ALL: 6) | PASS | PRODUCTION_POLICY_REPLAY |
| GROUNDING | evidence+schema failures <= 2% of cases | 2/46 = 4.3% | FAIL | ASYMMETRIC_RISK_STRICT |
| AUTOMATION_JOINT | AUTO coverage >= 70% AND weighted-risk gate passes | coverage 67.4%; weighted-risk gate fails | FAIL | both artifacts |
| DISAGREEMENT_ROUTING | wide disagreement + active issues -> REVIEW | HUMAN_DISPUTE_AWARE_C routes all 8 wide + 6 issue cases to REVIEW by construction | PASS | PRODUCTION_POLICY_REPLAY (policy property) |
| OCR | production OCR validated separately before end-to-end shipping | not validated (OCR_VALIDATION_PLAN_2026-09-02.md pending) | FAIL | OCR_VALIDATION_PLAN |
| FINAL_TEST | HELD_OUT untouched until grader+matrix+policy+OCR frozen | HELD_OUT untouched (0 exposure in this task); matrix frozen; grader unchanged; decision policy NOT yet frozen; OCR NOT frozen | PASS | this analysis |

5/9 gates pass. The WEIGHTED_RISK, GROUNDING, AUTOMATION_JOINT and OCR gates block production.

## Recommendation (NOT deployed — no models.toml, prompt, or policy change was made)

- semantic layer: keep `baseline_8b_one_pass` (qwen3-vl:8b-instruct Q4, one-pass, grade-v4-charitable-local). arm_a (q8_0) trades overgrades for undergrades with no robust risk win and lower exact agreement. arm_b (two-pass) changes no verdict; under the asymmetric objective its verifier DOES concentrate some serious overgrades into REVIEW (AUTO pv->valid 1 vs 2, AUTO risk 18 vs 23), but at 2x inference and 43.5/100 review workload — a documented lower-risk/lower-coverage alternative, not the primary recommendation.
- risk layer candidate: `HUMAN_DISPUTE_AWARE_C` — AUTO 31/46 (67.4%), REVIEW 15 (32.6/100 explanation cases), AUTO precision 77.4%, AUTO risk 23 (mean 0.7419), false-full 0, pv->valid 2, automatic undergrades 1 — vs AUTO_ALL risk 42 at 95.7% coverage.
- NOT production-ready: the semantic layer does not beat `always_partially_valid` on weighted risk, invalid-recall is 1/5, grounding failures exceed 2%, and OCR is unvalidated. Deploying the risk layer cannot fix the semantic layer.

## Before HELD_OUT

1. improve the semantic layer's invalid/partial discrimination (model or evidence improvements — NOT prompt-tightening as a risk proxy);
2. freeze the AUTO/REVIEW decision policy version;
3. validate the OCR route separately;
4. re-pass gates; only then run HELD_OUT once, with the already-frozen matrix.

Confirmations: new local inference 0; cloud 0; OCR 0; RAG 0; HELD_OUT exposure 0; human references modified 0; instructor grades modified 0; spend $0.
