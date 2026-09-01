# Baseline error audit — 15 errors vs the final human reference

(2026-09-02 01:28:36; flags only, no target altered)

| case | boundary | source | instructor | category | flags |
|---|---|---|---|---|---|
| e002_q1_r5 | partially_valid -> valid | adjudicated | partially_valid | failed_partial_vs_valid_boundary | amb |
| e002_q1_r6 | partially_valid -> valid | adjudicated | partially_valid | failed_partial_vs_valid_boundary | amb,adj |
| e002_q1_r7 | partially_valid -> invalid | adjudicated | valid | failed_invalid_vs_partial_boundary | amb,tx,adj |
| e002_q2_r4 | partially_valid -> invalid | adjudicated | valid | transcription_evidence_issue | tx,adj |
| e002_q2_r5 | valid -> partially_valid | adjudicated | valid | failed_partial_vs_valid_boundary | amb,adj |
| e003_q1_r6 | valid -> partially_valid | adjudicated | valid | failed_partial_vs_valid_boundary | amb,adj |
| e003_q1_r7 | valid -> invalid | adjudicated | valid | model_too_strict | adj |
| e003_q2_r1 | partially_valid -> valid | two_reviewe | valid | failed_partial_vs_valid_boundary | amb,adj |
| e003_q2_r6 | invalid -> partially_valid | two_reviewe | - | failed_invalid_vs_partial_boundary | - |
| e004_q1_r1 | valid -> invalid | adjudicated | valid | adjudication_uncertainty | tx,adj |
| e004_q1_r3 | invalid -> partially_valid | two_reviewe | partially_valid | failed_invalid_vs_partial_boundary | amb |
| e004_q1_r6 | partially_valid -> invalid | adjudicated | partially_valid | model_too_strict | adj |
| e004_q1_r8 | partially_valid -> valid | adjudicated | partially_valid | failed_partial_vs_valid_boundary | adj |
| e004_q2_r4 | invalid -> partially_valid | two_reviewe | - | failed_invalid_vs_partial_boundary | - |
| e004_q2_r5 | invalid -> partially_valid | two_reviewe | - | model_too_generous | - |

## Findings

- errors are BOUNDARY-dominated: 10/15 primary categories are the two verdict boundaries (6 partial-vs-valid, 4 invalid-vs-partial); pure too_generous/too_strict misreads are only 3/15
- errors concentrate where humans also disagreed: 10/15 sit on adjudicated_human_reference cases and 10/15 carry an adjudication-uncertainty flag; 0/15 on the owner-repaired cases
- rubric ambiguity flagged on 7/15 (generic rubric gives no full-vs-partial criterion for terse core-correct answers)
- short terse answers (<40 chars) drive the partial->valid overgrades; the model over-credits a single correct core claim
- writer e004 has the highest error rate (6/14 = 42.9%) vs e002 31.3%, e003 26.7%
- q1 carries 9/15 errors (39.1% error rate) vs q2 6/15 (26.1%)
- one model-side evidence defect found (e002_q1_r7: cited span is official-solution text, not student text - the validator already caught it: REVIEW)
