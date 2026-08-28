# Counterfactual structural replay — NOT actual model performance

Historical run `dev__dev_verdict__all__qwen3-vl-8b-instruct__433146e4b1` replayed READ-ONLY against the new output contract (grade-validation-v2) using the unchanged production matcher. Zero inference. No historical output was mutated or re-scored as real performance; this measures only whether spans the model ALREADY produced would have verified in the correct field.

- evidence failures examined: **25**
- counterfactually structurally valid with a span the model itself isolated (whole field or a quote-delimited fragment): **16/25**
- a 12–200-char verbatim span exists somewhere in the mis-placed field: **25/25**
- still failing without new copying: {'paraphrase_only_no_isolated_quote': 8, 'wrong_source_span': 1}
- field content classes: {'clean_exact_quote': 5, 'exact_quote_wrapped_in_commentary': 11, 'only_paraphrase_prose': 8, 'wrong_source_text': 1}
- over the 200-char limit as submitted: 20/25
- unknown/duplicate rubric ids in history: 0 (rubric_items was empty in every output)
- historical zero verdicts: 1; newly routed to REVIEW under the zero-side grounding rule: 1 (the run's only AUTO decision — a harmful undergrade — would no longer auto-finalize)

| case | content class | model isolated a verified quote | still-fail reason |
|---|---|---|---|
| e002_q1_r2 | clean_exact_quote | no | - |
| e002_q1_r3 | exact_quote_wrapped_in_commentary | yes | - |
| e002_q1_r4 | clean_exact_quote | no | - |
| e002_q1_r5 | clean_exact_quote | no | - |
| e002_q1_r6 | clean_exact_quote | no | - |
| e002_q1_r7 | exact_quote_wrapped_in_commentary | yes | - |
| e002_q1_r8 | exact_quote_wrapped_in_commentary | yes | - |
| e002_q2_r1 | exact_quote_wrapped_in_commentary | yes | - |
| e002_q2_r2 | only_paraphrase_prose | no | paraphrase_only_no_isolated_quote |
| e002_q2_r3 | only_paraphrase_prose | no | paraphrase_only_no_isolated_quote |
| e002_q2_r5 | wrong_source_text | no | wrong_source_span |
| e002_q2_r6 | exact_quote_wrapped_in_commentary | yes | - |
| e002_q2_r7 | exact_quote_wrapped_in_commentary | yes | - |
| e002_q2_r8 | clean_exact_quote | no | - |
| e003_q1_r1 | exact_quote_wrapped_in_commentary | yes | - |
| e003_q1_r4 | exact_quote_wrapped_in_commentary | yes | - |
| e003_q1_r5 | exact_quote_wrapped_in_commentary | yes | - |
| e003_q1_r6 | only_paraphrase_prose | no | paraphrase_only_no_isolated_quote |
| e003_q1_r7 | only_paraphrase_prose | no | paraphrase_only_no_isolated_quote |
| e003_q1_r8 | only_paraphrase_prose | no | paraphrase_only_no_isolated_quote |
| e003_q2_r1 | only_paraphrase_prose | no | paraphrase_only_no_isolated_quote |
| e003_q2_r2 | only_paraphrase_prose | no | paraphrase_only_no_isolated_quote |
| e003_q2_r7 | exact_quote_wrapped_in_commentary | yes | - |
| e003_q2_r8 | exact_quote_wrapped_in_commentary | yes | - |
| e007_q1_r1 | only_paraphrase_prose | no | paraphrase_only_no_isolated_quote |

_Counterfactual structural replay — not actual model performance._
