# Oracle-ensemble analysis (post-hoc, run1 of every config)

Experts: 14 configs; strict cells: 11; hard cells: 5. Fixed rule: run1 output per config; no GT-based selection is proposed for deployment — this is an upper-bound measurement.

## Per-expert vs oracle (strict cells)

| expert | mean CER | usable |
|---|---|---|
| isol2_tsub | 0.786 | 0/11 |
| it4_contrast | 0.789 | 0/11 |
| it5_moe30b | 0.806 | 0/11 |
| isol3_tsub_lines | 0.825 | 0/11 |
| it2_strict_prompt | 0.847 | 0/11 |
| isol0_orig_e002 | 0.879 | 0/11 |
| isol1_blueonly | 0.883 | 0/11 |
| it3_q8_quant | 0.887 | 0/11 |
| it1_baseline_8b | 0.893 | 0/11 |
| isol6_hdd_tsub_lines | 0.936 | 0/11 |
| isol5_hdd_tsub | 0.937 | 0/11 |
| it7_surya | 0.955 | 0/11 |
| it6_hdd_words | 0.963 | 0/11 |
| isol4_hdd_blueonly | 0.977 | 0/11 |
| **oracle (lowest-CER expert per cell)** | **0.717** | **0/11** |

- Best single expert: isol2_tsub (CER 0.786).
- Oracle improvement over best single: 0.070 CER (9% rel).
- Cells where NO expert reaches CER <= 0.5: 11/11 (e002_q1_r1, e002_q1_r2, e002_q1_r3, e002_q1_r4, e002_q1_r5, e002_q1_r6, e002_q1_r7, e002_q1_r8, e002_q2_r2, e002_q2_r3, e002_q2_r8).
- Cells where NO expert reaches usable (CER <= 0.25): 11/11.
- Mean pairwise error correlation (Pearson over per-cell CER): 0.214 (1 = experts fail identically).

## GT-free selection rules (evaluated post-hoc)

- Medoid consensus (closest-to-others output): CER 0.864, usable 0/11.
- Agreement-gated abstention (accept cell iff any expert pair agrees >= tau; medoid on accepted):

| tau | coverage | CER on accepted | usable on accepted |
|---|---|---|---|
| 0.5 | 11/11 | 0.864 | 0 |
| 0.6 | 11/11 | 0.864 | 0 |
| 0.7 | 11/11 | 0.864 | 0 |
| 0.8 | 11/11 | 0.864 | 0 |
| 0.9 | 11/11 | 0.864 | 0 |

## Hard cells (honest-abstention counts)

| cell | experts flagging unreadable (of 14) |
|---|---|
| e002_q2_r1 | 1 |
| e002_q2_r4 | 0 |
| e002_q2_r5 | 0 |
| e002_q2_r6 | 0 |
| e002_q2_r7 | 0 |

## Decision (per the pre-registered gate)

**REJECT ensembling/MoE over these experts; proceed to the
writer-separated fine-tuning pilot.** The pre-registered rejection gate is
met exactly: oracle usable-rate is 0/11 — even a perfect per-cell selector
with access to ground truth gets zero usable cells, and on all 11 strict
cells no expert reaches even CER 0.5. There is no competence to select:
the oracle 9 % relative CER gain over the best single expert
(.786 -> .717) is noise-level diversity among uniformly failing readers.

Two supporting observations:

- The medoid consensus (0.864) is WORSE than the best single expert —
  10 of the 14 experts share the same base VLM, so consensus drags toward
  that family shared errors.
- Agreement-gated abstention cannot work over this pool: some expert pair
  agrees >= 0.9 on every cell (same-family outputs are near-identical at
  temperature 0, right or wrong), so agreement carries no correctness
  signal. Any future abstention mechanism must come from a model own
  calibrated confidence (e.g. CTC posteriors in the fine-tuned
  recognizer), not from inter-expert agreement of correlated models.

No gate was trained; nothing here feeds deployment. Next step remains the
fine-tuning pilot on the annotation package (evaluation/htr_pilot/).
