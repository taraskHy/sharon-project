# SEEN-46 reconciliation — blind human consensus vs model vs instructor (2026-09-01)

Campaign complete (92/92 verified) -> owner-confirmed r6/r8 source repair applied -> 4 affected reviews
and 14 affected model outputs marked stale/invalid (preserved, excluded). Current population: 44 cases,
22 with blind-human CONSENSUS, 22 NEEDS_ADJUDICATION, 2 parked pending fresh re-review.

## Reviewer agreement (44 current cases)
- agreement 22/44 = 50.0% | Cohen's kappa **0.2461** (chance-expected 0.3368)
- both-agree by class: {'invalid': 4, 'partially_valid': 4, 'valid': 14}
- by writer: e002 37.5% (6/16), e003 53.3% (8/15), e004 58.3% (7/12), e007 100.0% (1/1)
- confidence: {'medium': 40, 'high': 43, 'low': 5} | issues: {'none': 79, 'rubric_official_solution': 3, 'transcription_evidence': 3, 'genuinely_ambiguous': 3}
- reviewer-vs-reviewer confusion (Erik rows x Or columns): {"invalid": {"invalid": 4, "partially_valid": 2, "valid": 0}, "partially_valid": {"invalid": 3, "partially_valid": 4, "valid": 1}, "valid": {"invalid": 8, "partially_valid": 8, "valid": 14}}

## A. Model vs blind human consensus (n=22)
- agreement **77.3%** (17/22) | balanced 0.5833 | macro-F1 0.5037
- model higher than humans: 5 | lower: 0 (zero undergrades vs consensus)

## B. Instructor vs blind human consensus (n=18 derivable)
- agreement **77.8%** (14/18) | instructor more lenient 4 / more strict 0
- numeric MAE (18 derivable-consensus cases): 0.4444

## C. Model vs instructor (historical, n=36 non-stale derivable)
- agreement 66.7% | model lower 9 / higher 3

## D. Three-way (n=18)
- {"all_agree": 14, "human_model_agree_only": 2, "human_instructor_agree_only": 0, "model_instructor_agree_only": 2, "all_disagree": 0}

## Style/overfit reading (small n; no generalization claim)
- model-human 77.3% vs model-instructor 66.7%: the model agrees MORE with
  independent blind humans than with the instructor - evidence AGAINST simple instructor-style overfit.
- where humans disagree with the instructor (4 cases): model sides with instructor 2, with humans 2 - an even split, no instructor-tracking.
- the instructor is systematically MORE LENIENT than blind consensus (4 vs 0): part of the model's
  historical "harmful downgrades" vs the instructor reflects instructor leniency, not model blindness.
- numbers will be recomputed after the 22 adjudications and the r6/r8 fresh reviews.

_No source is declared universally correct. HELD_OUT remains sealed._
