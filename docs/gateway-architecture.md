# Provider-independent gateway + low-review grading architecture

Status: **implemented, tested offline, NOT yet live** (no OpenRouter request
and no heavy local model has been executed by this code). Enabled only via
`--models-config models.toml`; absent, the validated pipeline is unchanged.

## Modules

| module | role |
|---|---|
| `autograder/gateway.py` | `ModelGateway`: task → backend/model routing from config (`${ENV}` expanded); wraps `VisionBackend.parse()` with cache / ledger / budget. No model ids in code. |
| `autograder/backends/openrouter.py` | `OpenRouterBackend` (subclass of the existing OpenAI-compat backend): key from `OPENROUTER_API_KEY` only, attribution headers, provider routing + reasoning fields, usage/cost capture, `/key` health probe. |
| `autograder/requestcache.py` | deterministic fingerprint (task, model, prompt version, generation params, system/text/image hashes, schema, pack hash) + content-addressed cache; failures never stored. |
| `autograder/usage.py` | `UsageLedger` (JSONL, aggregates) + `BudgetManager` (soft warn / hard pause, local exempt, never downgrades). |
| `autograder/gradingpack.py` | `QuestionGradingPack` built once per question (answer-free text, key answers per version, rubric, rules, policy, official solution, budgeted course-RAG evidence, provenance, hash) + `PackStore`. |
| `autograder/policies.py` | the five grading policies, `MCResolution`, `decide_before_ocr` (pre-OCR gate), deterministic policy inference. |
| `autograder/mcresolve.py` | MC chain deterministic → local (`mc_resolve`) → cloud (`mc_resolve_cloud`) → REVIEW; agreement rules; stats. |
| `autograder/escalation.py` | OCR suspicion signals + verifier escalation; tiny `GradeResult`; deterministic grade validation; primary → escalate → REVIEW; `ReviewMetrics`. |
| `autograder/discovery.py` | package discovery (versions, markers, alignment, template, policies) emitting the EXISTING sidecar contracts; `VariantCatalogStore` with human-resolution reuse. |
| `autograder/reviewui.py` + `webui.py` | settings tab (no key exposure, minimal-token probe) and one-click review with per-exam `ResolutionStore` (apply-to-all only for variant/layout). |
| `autograder/orchestrator.py` | end-to-end plumbing, hooks, failure/pause semantics. |

Hooks into the validated pipeline (both default OFF): `extract.set_mc_resolver`
(single ambiguous-row site of banded extraction) and
`grade.set_grading_policies` (pre-judge gate in `judge_question`).

## Configuration

`models.toml` (gitignored; template `models.example.toml`):

```toml
[defaults]
structured_mode = "json_schema"; temperature = 0.0; timeout_s = 300.0
[models.grade_primary]
backend = "openrouter"; model = "${GRADE_PRIMARY_MODEL}"; max_tokens = 300
reasoning = { effort = "none" }; prompt_version = "grade-v2"
[models.mc_resolve]
backend = "ollama"; base_url = "http://localhost:11434/v1"
model = "qwen3.8:27b-q4_K_M"; extra_generation = { think = false }
```

Environment: `OPENROUTER_API_KEY` (required only when an openrouter task is
enabled AND used); model slugs `OCR_PRIMARY_MODEL`, `OCR_VERIFY_MODEL`,
`GRADE_PRIMARY_MODEL`, `GRADE_ESCALATE_MODEL`, `MC_RESOLVE_CLOUD_MODEL`,
`VARIANT_RESOLVE_CLOUD_MODEL` (an unset slug on an enabled task fails at
gateway construction, loudly).

## Token / query saving mechanisms

1. deterministic-first everywhere (CV extractor, policy gate, suspicion signals);
2. MC early exits (`choice_only`, `wrong_choice_zero`, wrong-choice rules) skip
   explanation OCR + grading entirely, with a persisted flag;
3. local Qwen MC resolver before any cloud MC call (loaded lazily, never preloaded);
4. request cache: identical (task, model, prompt, inputs, pack, params) → zero calls;
5. crops only, never full pages, in every cloud call;
6. tiny structured outputs (`GradeResult`, `MCRead`, `OCRVerifyResult`), evidence ≤ 200 chars;
7. reasoning effort `none` for routine tasks, reserved for `grade_escalate`;
8. grading packs built once per question and reused; RAG evidence top_k=2 / 1200-char budget;
9. OCR verifier only on suspicion; grade escalation only on validation failure;
10. budgets: soft warn / hard pause (no silent downgrade); ledger aggregates.

## Review-minimization mechanisms

Deterministic settlements; agreement rules (CV+local, local+cloud) instead of
guessing; verifier-supported OCR → AUTO; escalated grades that validate and
agree → AUTO; per-exam one-click resolutions; variant/layout resolutions
reused across the job; human REVIEW only for: MC letter conflicts or
unresolvable rows, OCR verifier disagreement, grading disagreement after
escalation, unresolved discovery facts (markers/policies), and any budget/
provider pause left unresumed.

## Lecturer workflow (target)

1. create/select course → 2. upload course material once (index built/
reused) → 3. upload exam + key/rubric/solution (discovery: variants,
template, policies, packs — automatic; unresolved facts shown for optional
correction) → 4. upload student exams → 5. Grade. Totals and only unresolved
cases for review.

## Pre-live small benchmarks (per component, before enabling)

- OpenRouter backend: 1 text + 1 image call on a mock-graded item, verify
  ledger/usage/cost fields and cache hit on repeat (≈ 2 paid calls).
- MC chain: the 10 deferred prob rows from Mission 1 through `mc_resolve`
  (local Qwen, lazy load) → measure resolution rate/agreement; cloud stage
  on the residue only.
- Policy early exits: replay a graded prob batch with policies installed;
  assert identical totals and count judge calls avoided (zero cost).
- OCR escalation: suspicion signals over the frozen gemini_protocol_clean
  records (zero cost) — flag rate must be low on clean reads; then verifier
  on ≤ 20 flagged items.
- Grading pack + grader: 5 items with known reference verdicts through
  `grade_primary`, validation pass rate; escalate on failures only.
- Discovery: run on prob_data (icons) and sample_data (flowers) with local
  model only; compare emitted variants.json to the frozen manual ones.
