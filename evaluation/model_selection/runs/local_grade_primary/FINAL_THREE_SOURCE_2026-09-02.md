# Final three-source analysis — SEEN-46 (2026-09-02 01:01:58)

Final human reference (46) = 22 independent two-reviewer consensus + 22 adjudicated_human_reference (disagreements) + 2 owner_adjudicated_after_source_repair (e004_q2_r6/r8). Model = 44 frozen SEEN-46 outputs + 2 corrected-rerun outputs (superseded pair preserved, registered invalid, excluded). Instructor = frozen derived verdicts (derivable cases only). No source is declared universally correct.

- reviewer agreement before adjudication: **22/44 = 50.0%**, Cohen's kappa **0.246**
- A model vs final human reference: **31/46 = 67.4%** (macro-F1 0.5063, balanced acc 0.5062; overgrades 8, undergrades 7)
- B instructor vs final human reference (38 derivable): **31/38 = 81.6%** (instructor more lenient 6, stricter 1)
- C model vs original instructor (38 derivable): **25/38 = 65.8%** (macro-F1 0.4161; overgrades 4, undergrades 9)
- D three-way over 38 cases: all 23, human+model 3, human+instructor 8, model+instructor 2, none 2
- model evidence validation: AUTO 44/46 (95.7%), evidence failures 2, schema failures 0

Repaired cases (owner reference, corrected model output):

| case | owner verdict | corrected model | instructor (derived) | instructor score |
|---|---|---|---|---|
| e004_q2_r6 | valid | valid | valid | 4 |
| e004_q2_r8 | valid | valid | partially_valid | 2 |
