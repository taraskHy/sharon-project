# Model roles

Authoritative registry of every model role in the exam autograder, compiled
2026-08-21 from source-verified per-role records (Part A of the model-selection
effort). One rule governs the registry: **roles requiring different
capabilities are never combined** — a vision transcription task, a text-only
grading judge, and an embedding retriever are separate roles with separate
routes, benchmarks, and candidate models, even when one provider could serve
several.

Each role's *production call path* distinguishes the two runtimes:

- **Legacy validated pipeline** — direct `VisionBackend` calls, local-only by
  `guard_direct_cloud_backend`. This is the default and the only validated
  configuration.
- **Gateway runtime** — `ModelGateway` task routes from gitignored
  `models.toml`, opt-in via `--models-config`; hooks default OFF, and every
  gateway call is wrapped privacy scan → request cache → budget → provider →
  ledger.

## Overview

| Role | Gateway task(s) | Vision | Local-first | Frequency | Cost weight | Selection benchmark |
|---|---|---|---|---|---|---|
| KEY_PARSE | none (legacy only) | yes | yes (enforced) | 1/package (cached) | medium latency, high correctness leverage | first-attempt acceptance + keyrepair verified ratio |
| SURVEY | none (legacy only) | yes | yes (validated) | 1/exam | low tokens, high error cost | sheet-page location accuracy (harness needed) |
| OCR_PRIMARY | ocr_primary (lazy OCR only) | yes | partly (marks yes, Hebrew handwriting no) | per question chunk / per item | high | frozen OCR baselines 2026-08-11 |
| OCR_VERIFY | ocr_verify | yes | possible; cloud by convention | 0–1/suspicious item (~25%) | moderate | Signal-2 fidelity protocol (m2_verify) |
| GRADE_PRIMARY | grade_primary (path B) | no | A required local; B config-only | 1/question (A); ~1/written item (B) | high | shadow A/B (shadow_comparison.json) |
| GRADE_ESCALATE | grade_escalate | no | architecturally; doubtful in practice | ~0.15/graded written item | medium | m2_grading harness + shadow A/B |
| MC_RESOLVE_LOCAL | mc_resolve | yes | yes (is the local tier) | ≤1/ambiguous MC row | low $ (latency, GPU) | mc_fallback_bench vs manual audit |
| MC_RESOLVE_CLOUD | mc_resolve_cloud | yes | n/a (last model tier) | ≤1/locally-unresolved row | low, budget-guarded | mc_fallback_bench rerun cloud-enabled (never run) |
| VARIANT_RESOLVE | variant_resolve, variant_resolve_cloud (discovery only) | yes | yes (live path enforced) | 1/exam live; ≤2/package discovery | low cost, high error cost | eval-batch variant columns vs docs/variants.md ground truth |
| ALIGN_RESOLVE | align_resolve, align_resolve_cloud (dormant, no example entries) | gateway no; legacy yes | yes | ≤ceil(items/10)/question per (key, variant) | low cost, high stakes | none — labeled reordered-booklet set needed |
| RAG_EMBED_RETRIEVE | none (outside gateway, local-only) | no | local-only | 1/course build + 1/question per package; 0/student | low | grading_rag_ab.py (not run; decides policy, not embedder) |

## KEY_PARSE

- **Purpose:** Convert the official answer-key/solution document (Hebrew RTL
  PDF with colour/highlight-encoded per-version answers, plus optional rubric
  text file) into the structured `AnswerKey` every downstream stage consumes:
  questions, sub-items, per-version accepted answers, points and caps,
  explanation rules, reference explanations, exam-level grading rules. The
  repair fallback is deterministic (text-layer decode + operator override),
  not a second model.
- **Inputs:** Every key page as a labeled base64 PNG at `max_image_edge`
  (~19.6K prompt tokens measured) + the PDF's embedded text layer per page
  (labeled colour-blind) + optional `--rubric` text + an authoritative
  version-ids hint when the exam family declares variants. System prompt
  `KEY_PARSER_SYSTEM` with version-legend decoding rules and a worked
  synthetic letter-group example (real-key literals were removed by the
  2026-08-21 generalization audit; see docs/generalization.md §0.6).
- **Output schema:** `AnswerKey` (pydantic): exam_title, versions[],
  questions[`KeyQuestion`{id, title, type, max_points,
  sub_items[`KeySubItem`{correct_by_version, points, reference_explanation,
  versions_unverified}], explanation_required, explanation_weight,
  answer_source, grading_notes}], total_points, general_rules;
  max_tokens=16000, observed 7–9K output tokens.
- **Vision required:** Yes — per-version answers are frequently encoded only
  by text colour/highlighting, which the text layer does not carry;
  key_parser.py tells the model "colours/highlighting are NOT visible here —
  read those from the images". The text layer is a supplement and the input
  to the deterministic repair layer.
- **Reasoning required:** Moderate — exhaustive structured extraction plus
  legend decoding. Empirically the hardest task for the local 8B model: the
  live ledger shows 1 structurally accepted parse out of 6 attempts. No
  extended reasoning tokens are configured; it is decode discipline, not
  chain-of-thought.
- **Call frequency:** 1 model call per key document per configuration
  fingerprint (bounded at 2 attempts, no further model retries by policy) —
  once per exam-package build; 0 per exam thereafter via the persistent
  cache. eval-batch parses once for the whole batch (`_shared_key_path`);
  jobs-runner subprocesses share the keycache fingerprint. ~12 min per parse
  on the RTX 2000 Ada at 32K context; cache hit ~30 s including
  re-verification.
- **Local-first:** Yes — local is currently the only production option:
  `guard_direct_cloud_backend` (called by cmd_parse_key, cmd_grade, and both
  evalcli entry points) refuses any cloud provider on the direct path, and
  key parsing has no gateway route. Proven live on qwen3-vl:8b-instruct
  (Ollama, 32K ctx), flaky at first attempt; the deterministic gate +
  text-layer repair make the local parse safe.
- **Cloud necessity:** No cloud path exists. A stronger model would mainly
  raise first-attempt acceptance, and the key contains no student content, so
  a cloud route would be privacy-lighter than grading — but it would require
  adding a `ModelGateway` task route. Deterministic verification makes cloud
  non-essential for version columns; it cannot rescue structurally wrong
  question/points/rubric extraction, where a better model still helps.
- **Production call path:** cli.cmd_parse_key / run_grade_pipeline (also
  evalcli `_shared_key_path`) → `_get_key` (autograder/cli.py:295) → keycache
  lookup → `parse_answer_key` (autograder/key_parser.py:14) →
  `VisionBackend.parse` on the legacy direct backend (local/mock only).
  Acceptance gate `_key_version_problems` (flattening detector included);
  rejected candidates saved as answer_key.rejected-N.json. Runs before the
  gateway runtime block — even with `--models-config` the parse never touches
  the gateway.
- **ModelGateway route:** None — no `[models.key_parse]` in
  models.example.toml. Related but distinct: gateway tasks
  `policy_infer`/`policy_infer_cloud` (discovery.resolve_policy_with_models)
  are implemented, but their only callers are discover_package /
  prepare_exam_package — tests only. Production policies come from
  `deterministic_policies(key)` or a `--grading-policies` JSON override; the
  `--rubric` file is model-read only as text inside this same key-parse call.
- **Fallback path:** (1) one re-parse with the same model; (2) deterministic
  rescue — `keyrepair.repair_key_versions` decodes per-version letter groups
  from the born-digital text layer, overriding model columns on disagreement,
  re-applied on every cache hit; (3) operator `<key stem>.versions-override.json`;
  (4) colour-only items without override marked `versions_unverified` →
  human-review flags; (5) required versions still unsatisfied → `BackendError`
  hard stop ("refusing to grade with a defective key"). Defective parses are
  never cached.
- **Cacheability:** keycache — persistent on-disk cache (default
  `%LOCALAPPDATA%/autograder/key_cache` or `GRADER_KEY_CACHE`;
  `--key-cache-dir` / `--no-key-cache`), keyed sha256 over key bytes + rubric
  text + backend.describe() + max_image_edge + `KEY_PARSER_SYSTEM` sha +
  AnswerKey schema sha, so prompt/schema/model edits self-invalidate; hits
  re-validated and re-repaired on every load. Also per-run out/answer_key.json
  with `--resume`; a parsed .json passed as `--key` skips the model entirely.
  The gateway requestcache is not involved.
- **Relative cost importance:** Medium as one-time latency (~20K prompt
  tokens, ~12 min local, needs the 32K-context configuration), low aggregate
  — amortized to ~zero per exam. Real weight is correctness leverage: a
  defective key silently misgrades every exam, which is why the pipeline
  spends validation and deterministic repair rather than more model calls.
- **Selection benchmark:** No dedicated automated benchmark. Assets: the
  key-parse attempt ledger (evaluation/performance.md;
  docs/validation/smoke-2026-07-13-strongpc-diagnosis.md) and the per-parse
  repair report. Select by (a) first-attempt acceptance under
  `_key_version_problems` on real key documents, (b) exact agreement of
  `correct_by_version` with the deterministic text-layer decode (keyrepair
  "verified" vs "repaired" ratio — free ground truth), (c) points/caps/rubric
  fidelity audited against the key PDF; end-to-end eval-batch agreement
  (evaluation/results.json) remains the downstream arbiter.
- **Evidence:** autograder/key_parser.py:14-57; autograder/cli.py:295-439;
  autograder/keyrepair.py:1-174; evaluation/performance.md:12-45.

## SURVEY

- **Purpose:** One document-level pass over all pages of a student scan at
  reduced resolution (default 640px long edge): classifies every page,
  locates dedicated answer-sheet pages and fills the answer-sheet policy,
  transcribes handwritten marking conventions verbatim, separates student
  ink from grader ink, states authoritative answer locations, reports version
  hints. Explicitly does not grade. Drives the SHEET_CLOSEREAD page set,
  extraction page routing (`_pages_for_question`) and context, and authority
  enforcement. Wrong sheet location silently misgrades whole questions.
- **Inputs:** `SURVEY_SYSTEM` (prompts.py:70-133); a key-orientation text
  block (question ids/titles/types/points/answer_source, deliberately no
  answers); "--- Page N ---" labels interleaved with base64 PNGs of every
  page at `survey_image_edge` (640; ingest.py:112-130). Images are red-ink
  masked when masking is on: `grade` requires `--mask` (default OFF);
  eval-batch masks by default; masking survives the downscale.
- **Output schema:** `ExamSurvey` (schema.py:262-287): pages[`PageInfo`],
  answer_sheet_policy (`AnswerSheetPolicy`: authoritative_pages,
  booklet_answers_not_graded, policy_source), marking_conventions, student
  ink / grader-annotation descriptions, authoritative_answer_locations,
  version_hints; max_tokens=16000. Persisted as survey.json after the
  deterministic `merge_closeread` folds in the full-resolution close-read —
  survey.json is the single merged source downstream.
- **Vision required:** Yes — the whole payload after key orientation is page
  images: page classification, table-layout detection, handwritten-note and
  ink-colour discrimination.
- **Reasoning required:** Moderate structured document understanding, not
  chain-of-thought. Temperature 0.0, json_schema constrained decoding on the
  legacy backend; no reasoning-effort knob applies (those exist only on
  gateway routes).
- **Call frequency:** 1 call per exam (whole document in one request). 0 when
  (a) an exam template with answer_sheet_rule='fixed_pages' substitutes the
  deterministic `synthesized_survey`, or (b) `--resume` hits the stage
  fingerprint. Triggers 0–1 companion SHEET_CLOSEREAD call (separate sibling
  pass: 1400px, sheet pages only, max_tokens=6000). Not per-question; not
  part of package build.
- **Local-first:** Yes — the validated production configuration.
  SURVEY_IMAGE_LONG_EDGE=640 was chosen so a 13-page exam fits an ~8K local
  context. Measured live: ~1.5–2.5 min for 13 pages on qwen3-vl:8b-instruct
  (evaluation/performance.md:25).
- **Cloud necessity:** No — validated on a local 8B VLM and sized for local
  contexts. Cloud is reachable only by pointing the legacy BackendConfig at
  anthropic/openrouter; the survey sends the ENTIRE document (identity pages
  and all handwriting) and, being a direct VisionBackend call, bypasses the
  gateway's privacy check (gateway.py:213-215 applies only to gateway.call).
  The known anthropic dev-backend bypass applies here.
- **Production call path:** Legacy pipeline only: load_pages → optional
  mask_pages → downscale_pages(survey_image_edge) →
  `survey.survey_exam(backend, survey_pages, key)` (cli.py:716) →
  `VisionBackend.parse(system=SURVEY_SYSTEM, output_model=ExamSurvey)`. Even
  when `--models-config` enables the gateway runtime, the survey stage still
  runs on the legacy backend.
- **ModelGateway route:** None — no survey task exists in gateway.py,
  models.example.toml, or orchestrator.py. Gateway features (privacy check,
  budget, ledger, requestcache) do not apply to this role today; migration
  would require a new task route.
- **Fallback path:** (1) deterministic `synthesized_survey` for
  answer_sheet_rule='fixed_pages' (zero model calls); (2) resume reuse of
  fingerprint-matched survey.json; (3) no answer sheets found → close-read
  skipped, booklet answers authoritative; (4) no model fallback by policy —
  backends raise `BackendError` and never fall back to a different model
  silently; (5) `merge_closeread` deterministically corrects sheet-page
  findings, ignores hallucinated page numbers, drops instructor score
  fractions.
- **Cacheability:** Not the gateway requestcache (that wraps only
  gateway.call). The cache is the per-stage resume fingerprint: survey.json +
  fingerprint.json 'survey', keyed on exam hash + key hash +
  backend.describe() + a prompts/schema hash including `SURVEY_SYSTEM` and
  the ExamSurvey schema + `survey_image_edge` + variant config, key
  overrides, alignment map, version pin, and template fingerprint.
- **Relative cost importance:** Low-to-moderate tokens (~1.5–2.5 min of the
  measured ~10–14 min/exam local runtime; the 640px sizing exists to keep it
  cheap), HIGH error cost — its output routes everything downstream and
  authority enforcement trusts its booklet_answers_not_graded flag. Optimize
  sheet-location/convention accuracy, not tokens.
- **Selection benchmark:** No automated survey metric exists. Grounding:
  evaluation/representative_exam_audit.md (sheet routing traced to the
  survey, line 79; resume/swap-suspect runs, line 102) and
  performance.md:25. Select by answer-sheet page-location accuracy (exact
  page-set match vs operator-audited survey.json), answer_sheet_policy
  correctness, and marking-convention recall on operator-annotated exams
  (evaluation/annotation_priority_queue.csv; e007–e012 next) — a small
  harness would need building; nothing scores ExamSurvey outputs today.
- **Evidence:** autograder/survey.py:1-62,137-188; autograder/prompts.py:70-133;
  autograder/cli.py:694-743; evaluation/performance.md:25,31.

## OCR_PRIMARY

- **Purpose:** Per-question extraction/transcription of student final answers
  — marks, selected options, candidate answers, and (eagerly or lazily) the
  handwritten explanation text — blind to the answer key (question structure
  sent without correct answers, extract.py:81-83). Three production shapes:
  (1) whole-page per-question extraction in chunks of ≤8 sub-items
  (`EXTRACTION_SYSTEM`); (2) per-row band reads of a cropped answer-table row
  (`BAND_EXTRACTION_SYSTEM`) as fallback when deterministic CV fails; (3)
  lazy per-item explanation OCR (`EXPLANATION_OCR_SYSTEM`) run only after the
  MC/policy gate proves a transcription is needed. NOT this role: fixed-table
  MC mark reading, which is deterministic CV (tablecrop + per-cell ink
  analysis, zero model calls) with the model reduced to an advisory proposal
  that never decides the grade.
- **Inputs:** Whole-page: key structure without answers, a pruned survey
  context, labeled page images selected cheapest-first (sheet pages +
  convention notes; else question pages; else whole document,
  extract.py:24-55). Band: one PNG crop of header + single row. Lazy OCR:
  structure + relevant pages + target sub_item_id. Pages pre-masked of red
  instructor ink under `--mask`. Gateway calls pass scan_blocks (identity
  keys abort); `privacy.build_ocr_request` defines the crop +
  anonymous-item-id payload.
- **Output schema:** `QuestionExtraction`{question_id, source_pages,
  authoritative_source, answer_sheet_status,
  sub_items[`SubItemExtraction`{status, answer_origin, final_answer,
  candidate_answers, explanation_transcription, explanation_legibility
  (none|full|partial|illegible|deferred), marks_observed, confidence,
  uncertainty_note}], notes}; band path wraps `BandRowExtraction`
  {printed_row_number, row} for a deterministic registration check; lazy OCR
  returns `ExplanationTranscription`{sub_item_id, transcription, legibility}.
  `MarkDisambiguation` is advisory-only, never decides a grade.
- **Vision required:** Yes — every variant reads scanned pages or PNG crops;
  Hebrew/RTL handwriting with mixed English terms is the norm.
- **Reasoning required:** No — faithful transcription, explicitly not judging
  ("Do NOT grade", "Never guess"); the route sets reasoning effort "none"
  ("routine OCR: no reasoning"). Documented failure modes are visual:
  template collapse on 20-row single calls, RTL column confusion, mark
  hallucination.
- **Call frequency:** Eager (legacy default): ceil(n_sub_items/8) calls per
  question per exam (EXTRACTION_CHUNK_SIZE=8; max_tokens=16000 per call).
  Banded template path: 0 per row when deterministic ink analysis succeeds;
  ~1 advisory call per multi-mark row; per-row VLM fallback 1/sub-item. Lazy
  OCR (reliability mode): ≤1 gateway ocr_primary call per gradeable sub-item
  surviving `decide_before_ocr` — deterministically settled items pay zero;
  estimator ocr_share = mc_correct_rate × (1 − blank_rate). Zero at package
  build.
- **Local-first:** Partly, already implemented: (a) fixed-table MC extraction
  is deterministic CV (validated against 130 independently audited rows); (b)
  the whole-page path runs on the legacy direct local VisionBackend — the
  validated production configuration; (c) Hebrew handwritten explanation
  transcription is where local fails: local Qwen 8B handwritten CER
  ~0.80–0.90 ("unusable"), and fine-tuned local HTR collapses on held-out
  writers (CER ~0.73–0.80) — a local-first lazy-OCR route is not currently
  viable for explanations.
- **Cloud necessity:** Only for the handwritten-explanation slice, on quality
  grounds: frozen baselines show Gemini 3 Flash protocol-clean mean CER 0.287
  (gate-20) vs ML Kit 0.664 and local Qwen ~0.81 full-set / ~0.90
  handwritten. Mark/selection extraction does not need cloud. Direct cloud is
  refused on the legacy path; the ocr_primary route is opt-in configuration
  in gitignored models.toml.
- **Production call path:** Legacy validated pipeline (default): cli grade →
  extract_exam → extract_question → `_extract_chunk` → VisionBackend.parse
  (extract.py:173-178); the banded opt-in first tries deterministic tablecrop
  and only falls back to model reads. Gateway runtime (opt-in
  `--models-config`, `--grading-mode reliability` with a configured
  ocr_primary route): extraction runs transcribe_explanations=False so
  gradeable explanations are structurally marked legibility="deferred";
  run_reliability_judging step 1b → `lazy_explanation_ocr` →
  gateway.call(task="ocr_primary", system=EXPLANATION_OCR_SYSTEM); the
  transcription is written back and frozen to extraction.json. Shadow mode
  stays eager.
- **ModelGateway route:** `ocr_primary` — [models.ocr_primary]
  backend=openrouter, model=${OCR_PRIMARY_MODEL} (unset → loud
  GatewayConfigError), max_tokens=400, reasoning={effort="none"},
  prompt_version="ocr-v1". Only the lazy explanation OCR uses this route; the
  eager and band extractions still go through the legacy direct backend —
  not routed, cached, budgeted, or privacy-scanned by the gateway.
- **Fallback path:** No ocr_primary route → lazy OCR disabled, validated
  eager extraction runs unchanged. Deferred item with no ocr_fn →
  REVIEW/OCR_UNRESOLVED, never silently graded. Provider exception →
  REVIEW/PROVIDER_FAILED; BudgetExceeded → PAUSED/BUDGET_PAUSED. Banded:
  TableCropError → per-row VLM reads → whole-page extraction; band
  registration mismatch deterministically forces status=ambiguous, confidence
  0. Missing/duplicate sub-items reconciled to ambiguous/human review;
  degenerate uniform/cyclic patterns trip a collapse tripwire (confidence
  capped 0.5, review flagged, answers never changed).
- **Cacheability:** Lazy OCR: requestcache deterministic fingerprint (route
  fields, prompt_version, system/text/image block hashes, schema hash,
  pack_hash); cacheable defaults true. Eager/band: no per-request cache —
  stage-level extraction.json reuse under `--resume` only when the extraction
  fingerprint (exam bytes + variant + "|deferred-ocr" flag) matches. Lazily
  produced transcriptions are frozen into extraction.json (created once,
  immutable).
- **Relative cost importance:** HIGH — the dominant per-exam token consumer
  (eager chunks carry full page images at max_tokens=16000; the estimator's
  ocr_input 1200 is the largest routine input class). The lazy-OCR +
  decide_before_ocr design exists to cut this cost; avoided calls are
  accounted in decision traces; the budget manager pauses rather than
  degrades.
- **Selection benchmark:** Handwriting slice: the FROZEN canonical OCR
  baselines of 2026-08-11 (evaluation/ocr_baselines.md) — CER/WER and
  usable≤0.25/≤0.50 rates on hebrew_bench gate-20 and eligible-32 crops with
  owner-verified ground truth; future experiments MUST compare on identical
  item subsets; plus fixed-judge decision preservation on the 12-cell subset.
  Mark/selection slice: the 130 audited answer-table rows and end-to-end
  eval-batch agreement with instructor grades on the deterministic manifests.
- **Evidence:** autograder/extract.py:127-178,354-372,736-780;
  models.example.toml:17-22; evaluation/ocr_baselines.md;
  autograder/policies.py:65-83.

## OCR_VERIFY

- **Purpose:** Image-grounded transcription-fidelity verifier: a second,
  independent read of ONE handwritten line/cell crop that judges only whether
  the primary OCR transcription faithfully matches the visible handwriting
  (omissions, additions, substitutions) — never solving, correcting, or
  inferring intent. The only mechanism besides deterministic validation that
  can turn a suspicious read back into AUTO; disagreement, low confidence,
  any reported issue, or unavailability all resolve to REVIEW
  (OCR_UNRESOLVED). Consulted only when deterministic suspicion signals fire,
  never for grading difficulty, never for crops that failed image quality.
- **Inputs:** `OCR_VERIFY_SYSTEM` + two blocks: the base64 PNG crop and
  "Proposed transcription: … Check fidelity now." Deliberately NO answer key,
  rubric, printed question, or course context — the frozen protocol withholds
  semantics so the verifier judges pixels, not meaning. Meta: item_id,
  question_id, stage='escalation'.
- **Output schema:** `OCRVerifyResult` (defined in autograder/escalation.py,
  NOT schema.py): verdict supported|review; omissions / substitutions /
  additions lists; confidence high|medium|low. AUTO only if
  verdict=='supported' AND confidence high/medium AND all three lists empty;
  anything else → REVIEW. Verdict + confidence persist into `OCRSignals` for
  future calibration; model confidence alone is never sufficient for AUTO.
- **Vision required:** Yes — every call embeds the crop; without
  crop_png_b64 the verifier is skipped and the item goes to REVIEW.
- **Reasoning required:** Low — route sets reasoning={effort='low'},
  max_tokens 400; a constrained pixel-vs-text comparison.
- **Call frequency:** 0–1 per suspicious written sub-item; zero for clean
  reads. Estimator: ocr_items × ocr_suspicion_rate (default 0.25) per exam.
  Zero at package build; zero in the default legacy mode — runs only under
  `--grading-mode reliability|shadow`.
- **Local-first:** Architecturally yes (the gateway is provider-independent;
  'ollama' is a first-class TaskRoute backend), but the shipped example
  routes to OpenRouter and the reliability trace hard-codes cloud=True when
  recording an executed verify. Design intent favors a read independent of
  the primary OCR model, which argues against reusing the same local model.
- **Cloud necessity:** Not architecturally; the production convention is
  cloud: [models.ocr_verify] backend='openrouter', model=${OCR_VERIFY_MODEL};
  the benchmark verifier arm was gemini-3-flash-preview. The role's value
  depends on independence from (and ideally strength beyond) the primary
  reader, which currently implies a cloud VLM.
- **Production call path:** Gateway runtime only; the legacy pipeline never
  invokes it. cli `--grading-mode reliability|shadow` (+ `--models-config`;
  default 'legacy') → run_reliability_judging → per written sub-item: crop
  quality check → `escalate_ocr` → deterministic `ocr_suspicion`
  (escalation.py:49-71) → only if suspicious and quality OK:
  gateway.call(task='ocr_verify'). Pre-gated by ReliabilityConfig.ocr_verify
  (default True) AND `_route_ok`; executed/skipped recorded in the decision
  trace.
- **ModelGateway route:** `ocr_verify` — [models.ocr_verify]
  backend='openrouter', model='${OCR_VERIFY_MODEL}' (unset → GatewayConfigError
  at load), max_tokens=400, reasoning={effort='low'},
  prompt_version='ocr-verify-v1'. escalate_ocr probes route(task) first and
  converts absence into a REVIEW decision rather than an error.
- **Fallback path:** Fail-safe to REVIEW in every degraded case — never
  another model, never acceptance of the suspicious read: bad crop quality
  (no amount of verifier agreement rescues it), gateway or crop missing,
  route unconfigured/disabled, call exception, verifier disagreement / low
  confidence / any issue list. All map to OCR_UNRESOLVED, which routes to
  OCR-side work or REVIEW, never to a stronger grader.
- **Cacheability:** requestcache fingerprint over route fields
  (prompt_version 'ocr-verify-v1' included), system-prompt hash, per-block
  hashes including the image data, schema, pack_hash; cacheable defaults
  true — identical crop+transcription reruns are cache hits (ledger
  cache_hit=true).
- **Relative cost importance:** Moderate — ~1300 in / 60 out tokens per call
  (image-dominated) on ~25% of written items; skipped verifies credited as
  avoided cloud calls in the trace. Cheaper than grade_escalate but a real
  per-exam cloud cost when suspicion rates rise.
- **Selection benchmark:** The frozen Signal-2 transcription-fidelity
  protocol on evaluation/hebrew_bench_v2: scripts/m2_verify_run.py (committed
  arm gemini-3-flash-preview, temperature 0, a-priori thresholds T1–T4) +
  scripts/m2_verify_analysis.py joining verifier outputs with
  preserved/silent labels from evaluation/m2_grading/gemini3_flash.jsonl —
  reports caught-silent, silent-auto-pass, flagged-preserved per threshold.
  Maximize caught-silent at acceptable flagged-preserved; human-verified
  ground truth at evaluation/hebrew_bench/verified_ground_truth.json. Note:
  the prompt and result model live in escalation.py, not prompts.py/schema.py;
  the trace's hard-coded cloud=True is worth revisiting if the route is ever
  pointed at a local backend.
- **Evidence:** autograder/escalation.py:49-159;
  autograder/reliability.py:218,378-390; models.example.toml:24-29;
  scripts/m2_verify_run.py:1-46.

## GRADE_PRIMARY

Explanation grading (LLM judge of written justifications) with two distinct
production call paths: (A) the legacy judge `judge_question`/`judge_all` in
autograder/grade.py on a direct VisionBackend, and (B) gateway task
`grade_primary` inside the escalation engine (`escalate_grade`,
autograder/escalation.py) fed by QuestionGradingPack context, used by the
reliability/shadow route.

- **Purpose:** Decide whether a student's transcribed written explanation
  expresses the key's/rubric's reasoning, so the deterministic scorer can
  convert that judgement into points. Path A returns a per-sub-item verdict
  (valid / partially_valid / invalid / missing / illegible) plus a
  copying-slip flag; path B returns a tiny structured score proposal with
  per-rubric-item met/not-met and verbatim evidence spans, deterministically
  validated and mapped onto the same taxonomy (`_verdict_from_score`,
  reliability.py:141-152 — "The model never supplies the number itself").
  Final points are always computed by plain-Python `grade_exam`.
- **Inputs:** Path A (one call per question): `JUDGE_SYSTEM` + a single text
  block — question title, rubric grading_notes, and a JSON array of sub-items
  (prompt, accepted answers for the detected version, reference_explanation,
  selected answer, transcription, legibility). No images. Path B (one call
  per sub-item): `GRADE_SYSTEM` + `grade_prompt` = `pack.to_grader_context()`
  (question, rubric ids+text, scoring rules, official solution, and — only
  when a RAG policy activates them — course-context chunks tagged
  supplemental) + correct option(s) + selected option + FROZEN transcription
  + allowed rubric ids + score range. RAG enters via the pack policy (default
  RAG_DISABLED): RAG_ALWAYS bakes chunks at build; RAG_ON_UNCERTAIN retries
  the primary once with `activate_rag` chunks (stage 'primary_rag');
  RAG_ON_ESCALATION feeds only the escalation grader. Retrieval query built
  from question+rubric+solution only, never student words; top_k=2 / 1200
  chars.
- **Output schema:** Path A: `ExplanationJudgement` — list of
  `ExplanationEvaluation`{sub_item_id, verdict, reasoning,
  explanation_matches_different_answer}. Path B: `GradeResult`{score: float,
  rubric_items[{id, met, student_evidence: verbatim span|null}],
  rubric_items_met, uncertain, evidence ≤200 chars|null}, deterministically
  validated by `validate_grade` (score range, rubric-id whitelist,
  MC-consistency under wrong_choice_zero/choice_only, evidence-span
  verification against the frozen transcription, question invariants).
- **Vision required:** No, both paths — text blocks only. Reading the
  handwriting is a separate role; grading consumes the frozen transcription.
- **Reasoning required:** Moderate semantic judgement (Hebrew/English mixing,
  abbreviations, informal phrasing), but the production route deliberately
  configures reasoning effort "none" for the primary; "high" is reserved for
  grade_escalate. The design assumes a routine grader with deliberately tiny
  structured output whose unclean outputs escalate rather than think longer.
- **Call frequency:** Path A: one call per question needing judging with ≥1
  transcribed explanation, per exam (sub-items batched); untranscribed or
  policy-settled sub-items cost nothing. Path B: ~one call per written
  sub-item surviving the MC/policy early exit and OCR-trust checks, plus at
  most one RAG retry per unclean primary. Estimator: grade_primary calls =
  ocr_items × (1 − cache_hit_rate), ~900 in / ~120 out tokens per call.
  Shadow mode runs both paths on the same exam.
- **Local-first:** Yes. Path A is REQUIRED local: `guard_direct_cloud_backend`
  restricts the legacy judge to local backends (Ollama/vLLM/LM Studio/
  llama.cpp) or mocks. Path B is provider-agnostic — "No model identifier is
  hardcoded anywhere in application logic"; a local-only models.toml is valid
  ("OpenRouter is never mandatory"); switching to local is config-only. Known
  limitation: the local 8B judge is weak on explanations, which is why the
  cloud route exists.
- **Cloud necessity:** Not architecturally — only by configuration and
  quality. The example routes grade_primary to OpenRouter
  ${GRADE_PRIMARY_MODEL}; unset env fails loudly. The gateway wraps every
  cloud call (privacy → cache → budget → provider → ledger); student identity
  is barred from payloads. Cloud quality is unmeasured in-repo: the shadow
  A/B that should justify any cloud default has not been run live
  (mock-tested only).
- **Production call path:** Path A (default): cmd_grade → run_grade_pipeline
  → `judge_all` (cli.py:872) → judge_question →
  llm.parse(JUDGE_SYSTEM, max_tokens=16000) → grade_exam; runs in modes
  legacy and shadow. Path B (opt-in): `--models-config` + `--grading-mode
  reliability|shadow` → orchestrator.setup_from_config builds
  Runtime(gateway+RequestCache+UsageLedger+BudgetManager) → packs from
  PackStore → run_reliability_judging → `_decide_item` →
  `escalate_grade(primary_task="grade_primary")` →
  gateway.call(task="grade_primary") → validate_grade → AUTO/REVIEW/PAUSED.
  Reliability mode: the route's evaluations are authoritative; shadow mode:
  legacy stays authoritative and only shadow_comparison.json is written.
  Jobs pass grading_mode through with default 'legacy'.
- **ModelGateway route:** `grade_primary` (default of
  ReliabilityConfig.primary_task) — [models.grade_primary]
  backend="openrouter", model="${GRADE_PRIMARY_MODEL}", max_tokens=300,
  reasoning={effort="none"}, prompt_version="grade-v2". Gated, default OFF:
  without `--models-config` the gateway runtime does not exist;
  `--grading-mode` defaults "legacy" and non-legacy modes hard-require
  `--models-config`. The legacy judge never uses this route.
- **Fallback path:** Path B, in order: primary exception → REVIEW "primary
  grader failed"; BudgetExceeded → item PAUSED, job pauses, never recorded
  as a bad grade. Unclean-but-alive primary → (RAG_ON_UNCERTAIN) one
  RAG-augmented retry, with unavailable RAG degrading silently → grade_escalate
  second read; agreement within score_tolerance 0.5 and identical met-ids (or
  clean resolution of declared uncertainty) → AUTO, else REVIEW "unresolved
  disagreement after escalation"; escalate route unconfigured → REVIEW.
  Human REVIEW is terminal. Path A: judge omissions become verdict
  'illegible' flagged, never guessed; illegible/empty transcriptions are
  routed locally without any call.
- **Cacheability:** Path B: gateway RequestCache keyed by route fields,
  system/block/schema hashes, and pack_hash; cross-student hits are rare (the
  transcription is in the prompt) — the cache mainly serves reruns. Upstream,
  the PackStore caches the grading context once per exam question,
  invalidated by a source fingerprint over key/policies/course-index/RAG
  config/schema versions. Path A: no request cache — only whole-stage reuse
  via `--resume` fingerprints.
- **Relative cost importance:** High — one of the two highest-volume cloud
  tasks (roughly one call per written sub-item, alongside ocr_primary). The
  system minimizes it: policy early-exits skip grading for
  deterministically-settled items, max_tokens capped 300 with reasoning
  "none", RAG off by default ("costs input tokens on every grading call"),
  every call budget-checked and ledgered with RAG accounting separated out.
- **Selection benchmark:** No purpose-built judge-selection dataset yet
  (docs/evaluation.md: agreement labels "none yet"). The designed decision
  artifact is the shadow-mode A/B: shadow_comparison.json with
  exact_score_agreement, mean_abs_delta, review-rate delta against the
  authoritative legacy judge — to be run frozen before any default change.
  Supporting: evaluation/m2_grading_results.csv (holds the judge FIXED to
  measure OCR sufficiency, so it cannot rank judges) and the owner's
  annotation queue (e007–e012) toward real verdict-agreement labels; a
  48-exam frozen final-test protocol is specified but the exams are absent.
  Do not conflate the paths when selecting: A needs a verdict-taxonomy judge
  robust to Hebrew/English mixing, locked local; B needs a per-sub-item
  structured grader copying verbatim spans within a rubric-id whitelist — a
  cheap fast model plus the grade_escalate partner is the intended shape.
- **Evidence:** autograder/grade.py:250-282; autograder/escalation.py:165-235,303-333;
  models.example.toml:31-36; autograder/reliability.py:141-152,544-606.

## GRADE_ESCALATE

- **Purpose:** Second, stronger grading read invoked only after the primary
  grader's output fails deterministic validation (or declares uncertainty)
  and, where enabled, a RAG-assisted primary retry has not resolved it.
  Agreement (or clean resolution of declared uncertainty) yields AUTO with
  the escalated result; anything else yields human REVIEW — safety is never
  lowered. It grades the same frozen transcription and is never used to "fix"
  a doubted OCR reading (evidence-backed OCR_UNRESOLVED items go to REVIEW
  before grading).
- **Inputs:** Text-only, same `grade_prompt` builder as grade_primary:
  QuestionGradingPack context (optionally with RAG chunks under
  RAG_ON_ESCALATION or a RAG_ON_UNCERTAIN retry pack), correct option(s),
  the resolved MC selection, the frozen verbatim transcription, allowed
  rubric ids, score range. Same `GRADE_SYSTEM`. No image blocks.
- **Output schema:** `GradeResult` (score, rubric_items with met + verbatim
  student_evidence, rubric_items_met, uncertain, evidence ≤200 chars),
  re-validated by the same `validate_grade`; the score is only ever a
  PROPOSAL — the final number always comes from the deterministic scorer, and
  `_verdict_from_score` maps the proposal to the verdict taxonomy.
- **Vision required:** No — text blocks only; vision lives in
  ocr_primary/ocr_verify, and an untrusted reading is never handed to this
  grader.
- **Reasoning required:** Yes — the one task the config reserves reasoning
  for: reasoning={effort="high"}, max_tokens 600 (vs "none"/300 for the
  primary); docs/gateway-architecture.md:57. Rubric-grounded partial-credit
  judgment is the workload, not extraction.
- **Call frequency:** Per written sub-item, conditionally: 0 for policy
  early-exits, blanks, unreadable OCR, or a clean primary; exactly 1 when the
  primary (and optional RAG retry) is unclean and the route is configured.
  Estimator default grade_escalation_rate = 0.15 of graded explanation items
  — a few calls per exam. Never at package build.
- **Local-first:** Architecturally yes (backend is configuration only;
  'ollama' accepted). Practically doubtful today: the role exists to be a
  STRONGER, independent grader than the primary, and the local qwen 8B scores
  0.083 decision_match_rate on the m2 grading cells. A strong local model
  could be trialed only against the grading benchmark.
- **Cloud necessity:** Not hard-required by code; the example routes to
  OpenRouter ${GRADE_ESCALATE_MODEL} (unset env fails loudly at gateway
  construction; a missing/disabled route degrades to REVIEW, not a crash).
  Design intent: a model stronger than the primary.
- **Production call path:** Gateway runtime only (legacy never calls it;
  `--grading-mode` defaults 'legacy'). run_reliability_judging →
  `_decide_item` → `escalate_grade` → primary grade_primary call →
  `validate_grade` (score range, unknown rubric ids, reported uncertainty,
  evidence >220 chars, nonzero score on wrong choice under wrong_choice_zero,
  rubric items on choice_only, fabricated/missing evidence spans, invariants)
  → if unclean: optional RAG_ON_UNCERTAIN retry →
  gateway.route('grade_escalate') probe →
  gateway.call(task='grade_escalate', stage='escalation') → re-validate.
  AUTO iff v2.ok AND (|score2−score1| ≤ 0.5 AND identical met-id sets, OR
  primary uncertainty resolved by a clean non-uncertain second read);
  otherwise REVIEW with status GRADE_DISAGREEMENT, keeping the second result
  if it validates else the primary. In shadow mode this is recorded only.
- **ModelGateway route:** `grade_escalate` (default in the escalate_grade
  signature and ReliabilityConfig.escalate_task) — [models.grade_escalate]
  backend=openrouter, model=${GRADE_ESCALATE_MODEL}, max_tokens=600,
  reasoning effort high, prompt_version 'grade-v2', temperature 0.0 from
  [defaults]. Gateway wraps privacy scan, request cache, ledger
  (stage='escalation', pack_hash, rag_* meta), and budget. Config-gated: no
  models.toml entry means gateway.route raises GatewayConfigError and
  escalation is simply unavailable.
- **Fallback path:** (1) route unconfigured/disabled → no call, immediate
  REVIEW "no escalation model configured" with the primary result retained;
  (2) escalation call raises → REVIEW "escalation failed"; (3) BudgetExceeded
  → re-raised as a job-level PAUSE, item PAUSED/BUDGET_PAUSED, never a
  recorded grade; (4) escalation ran but disagreed or re-failed validation →
  REVIEW GRADE_DISAGREEMENT. Every REVIEW/PAUSED item becomes a ReviewItem
  merged into needs_human_review (reliability mode) — human review is the
  terminal fallback. RAG under RAG_ON_ESCALATION degrades silently to no-RAG
  escalation, never to REVIEW.
- **Cacheability:** requestcache (cacheable defaults true; fingerprint over
  task/backend/model/prompt_version/knobs + system/text/schema hashes +
  caller-declared pack_hash). Hit rate near zero across students; exact
  reruns replay free; estimator cache_hit_rate default 0.0.
- **Relative cost importance:** Medium — the most expensive per-call grading
  task (estimator 900 in / 400 out tokens vs 120 out for the primary;
  reasoning high) but fires on only ~15% of graded items; budget ceilings
  pause the job rather than degrade grading. The real lever is upstream:
  policy early-exits and clean primaries avoid it; skipped escalations are
  accounted as avoided cloud calls.
- **Selection benchmark:** None dedicated. Deciding assets: (a)
  evaluation/m2_grading_results.csv + evaluation/m2_grading/ —
  decision_match_rate / safe_rate / verdict_shifts against owner-annotated
  cells (12 cells; qwen 8B ~0.083, gemini3_flash ~0.417); score candidate
  escalation models on the cells where the PRIMARY failed validation —
  selection metric: correctness of escalation-resolved AUTO decisions (a
  confident agreement on a wrong grade is the worst failure) plus residual
  REVIEW rate; (b) the shadow comparison harness (compare_shadow;
  shadow_comparison.json) — the stated gate before any default change.
  Caveats: ReliabilityConfig.grade_escalation (default True) feeds only the
  route_item explanation flag — the actual call is gated by
  gateway.route(escalate_task), so setting it False does not prevent the
  call; and nothing in code enforces that the escalation model differs from
  the primary — independence is configuration discipline only.
- **Evidence:** autograder/escalation.py:246-279,377-422;
  models.example.toml:38-43; autograder/reliability.py:430-455;
  evaluation/m2_grading_results.csv.

## MC_RESOLVE_LOCAL

- **Purpose:** Second stage of the MC resolution chain (deterministic CV →
  local → cloud → REVIEW): a local vision model reads ONE cropped
  answer-table row band that deterministic ink analysis left ambiguous
  (multiple live marks) and tries to identify the single clean final mark
  among the deterministic candidate letters, so the row can auto-resolve
  instead of going to human review. It never overrides the deterministic
  first pass — single-dominant-mark and blank rows are decided without any
  model call.
- **Inputs:** One grayscale PNG band crop (header row stitched above the
  single data row, from tablecrop.answer_table_row_bands) as a base64 image
  block, plus a text block naming the RTL option-column letters and the
  deterministic candidates ("Decide only among those unless the analysis
  clearly missed a mark"), under `MC_RESOLVER_SYSTEM` ("report ONLY what is
  visible… Never guess"). Pre-gated by imagequality.triage_crop (INVALID crop
  = immediate REVIEW, zero calls).
- **Output schema:** `MCRead` {selected: letter|null, candidates, state
  single_mark|multiple_marks|erased|blank|unclear, confidence
  high|medium|low}, wrapped into `policies.MCResolution` (CONF map high=0.95
  / medium=0.7 / low=0.4; source deterministic|local_model|cloud_model|
  agreement|review). extract accepts the chain result only via
  `MCResolution.resolved()` with min_confidence=0.9 — only HIGH-confidence
  local reads (0.95) auto-resolve; medium falls through to cloud/review.
- **Vision required:** Yes — reading a handwritten mark in a scanned table
  row.
- **Reasoning required:** No — constrained perception with an agreement rule.
  The route disables thinking (extra_generation think=false; bench adds
  num_ctx 8192, repeat_penalty 1.0); max_tokens 120.
- **Call frequency:** At most 1 per ambiguous MC row; zero for
  deterministically decided rows. Fires only under template banding
  (`_banding_applies`) AND with a hook installed. Estimator
  mc_ambiguous_rate=0.08 (~1.6 calls/exam for a 20-row table); measured on
  the audited job: 10 ambiguous rows across 13 exams (~0.8/exam). Bench
  ledger: ~385–437 in / 25–51 out tokens, 8.7–33 s per call on qwen3.8:27b
  local.
- **Local-first:** Yes — this IS the local-first stage by design: the chain
  consults the local model lazily (loaded only when a row is actually
  uncertain; Ollama idle-unload applies), escalating to cloud only when the
  local read is absent/low-confidence/unresolved. Route: backend 'ollama'
  mapped to the native /api/chat backend (honors think:false and num_ctx),
  qwen3.8:27b-q4_K_M at localhost:11434.
- **Cloud necessity:** No for this role — the benchmark ran allow_cloud=False
  with a local-only config. If the local stage fails or is unconfigured the
  chain escalates to cloud (if allowed) or REVIEW; correctness never depends
  on cloud because REVIEW absorbs everything unresolved.
- **Production call path:** GATED, DEFAULT OFF. Legacy: `extract._MC_RESOLVER`
  is None → multi-mark rows get an advisory-only `_propose_disambiguation`
  via the direct VisionBackend and stay status='ambiguous'. Gateway (opt-in
  `--models-config`): orchestrator.setup_from_config → install_hooks binds
  `mcresolve.resolve_row(gateway, allow_cloud=allow_cloud_mc default True)`
  via `extract.set_mc_resolver`; extract_question_banded calls the chain for
  rows where deterministic analyze_answer_table found >1 real marks →
  gateway.call(task='mc_resolve') → ollama_native backend.
  resolve_row(gateway=None) returns immediate REVIEW, preserving legacy
  behavior.
- **ModelGateway route:** `mc_resolve` — [models.mc_resolve]
  backend='ollama', base_url http://localhost:11434/v1,
  model qwen3.8:27b-q4_K_M, max_tokens=120, extra_generation={think=false},
  prompt_version='mc-v1'. Gateway wrapper applies (privacy scan,
  requestcache, budget, ledger — classified provider 'ollama', cloud=false).
  An unconfigured/disabled task raises GatewayConfigError, caught by
  resolve_row as 'not_configured' — stage skipped, never a crash.
- **Fallback path:** Stage errors never crash grading (broad except →
  status 'error: <type>'). Local read missing / not single_mark /
  out-of-candidates / medium-low confidence → escalate to 'mc_resolve_cloud'
  when allowed and routed; otherwise MCResolution source='review' → row stays
  'ambiguous' with candidate letters. Special case: local high-confidence
  'blank' with no candidates resolves as blank (0.95). Anti-guess rule: a
  local/cloud letter disagreement goes to REVIEW, not to either model.
- **Cacheability:** requestcache (content-addressed
  gateway_cache/<fp[:2]>/<fp>.json): fingerprint covers route fields, hashed
  system prompt, and the SHA-256 of the band PNG bytes — identical crop
  reruns are free. The bench output contains populated cache entries
  (evaluation/mc_fallback/qwen38_local_v1/cache/).
- **Relative cost importance:** Low in dollars (local; reported_cost null;
  ~410–490 total tokens/call); real costs are latency (8.7–33 s on the 27B)
  and GPU residency. Economic value is upstream: every resolved row avoids a
  cloud mc_resolve_cloud call or a human-review row.
- **Selection benchmark:** scripts/mc_fallback_bench.py on the historically
  ambiguous rows of the audited prob batch, judged against
  evaluation/prob/manual_audit.json (130 audited rows / 13 scans). Metrics:
  accuracy_on_resolved_with_reference, resolution rate,
  cloud_escalation_rate_pct. Existing run
  evaluation/mc_fallback/qwen38_local_v1/summary.json: n_ambiguous=10,
  resolved=1, correct=1 (accuracy 1.0 on resolved), 9 unclear → 90% would
  escalate — safe but weak. Promotion requires: no incorrect resolutions on
  audited rows, and resolution rate materially above the measured 10%.
  Production counters in `MCResolverStats`; canary.py defines a frozen
  'mc_resolver' suite contract, but no suite is populated.
- **Evidence:** autograder/mcresolve.py:1-127; models.example.toml:60-66;
  evaluation/mc_fallback/qwen38_local_v1/summary.json;
  autograder/extract.py:414-472.

## MC_RESOLVE_CLOUD

- **Purpose:** Third and final model stage of the MC chain: a cloud model
  (OpenRouter) re-reads the same ambiguous row-band crop after the local
  model failed to produce a confident in-candidates single-mark read. It cuts
  the human-review pile: it resolves the row only when it agrees with the
  local read or is itself a confident single mark within the deterministic
  candidates; any local/cloud letter conflict is a hard REVIEW ("never
  guess").
- **Inputs:** Identical payload to the local stage — same
  `MC_RESOLVER_SYSTEM`, same text block (RTL letters + candidates), same
  base64 PNG band crop — sent to a different task route. Implicit additional
  input: the local stage's MCRead, used for the agreement/conflict decision.
- **Output schema:** Same `MCRead`; folded into MCResolution with
  source='cloud_model' (or 'agreement' when local.selected == cloud.selected);
  conflict with a local single_mark read of a different letter →
  MCResolution(None, 'unclear', 0.0, 'review'). Same acceptance gate: only
  confidence='high' (0.95) cloud reads flip a row to answered.
- **Vision required:** Yes — same band crop.
- **Reasoning required:** No — reasoning={effort='none'}, max_tokens 120;
  the chain's agreement logic, not model reasoning, provides the safety.
- **Call frequency:** At most 1 per ambiguous row, and only for the subset
  the local stage failed. Estimator: MC sub-items × mc_ambiguous_rate (0.08)
  × (1 − mc_local_resolution_rate, default 0.60) × (1 − cache_hit_rate) —
  ~0.6 calls/exam for a 20-row table under defaults; with the MEASURED local
  resolution of 10%, ~1.4/exam. Zero calls in any recorded run: the only
  benchmark ran allow_cloud=False.
- **Local-first:** Yes and enforced — the cloud stage is structurally last:
  only after the deterministic pass flagged the row AND the local read failed
  `_read_ok`. It can be switched off entirely (install_hooks
  allow_cloud_mc=False, or no route), sending unresolved rows straight to
  review.
- **Cloud necessity:** Not for correctness — human REVIEW is the terminal
  fallback, so the pipeline is complete without cloud. Cloud is necessary
  only to reduce review load; the measured local weakness (1/10 resolved) is
  the argument for it. This stage has never run live —
  configured-but-unexercised (mock/config-tested only).
- **Production call path:** Same gated path as MC_RESOLVE_LOCAL, one step
  further: after the local stage fails `_read_ok` and allow_cloud is True,
  `_call('mc_resolve_cloud')` → gateway.call → openrouter backend. Three
  independent gates must all be open: (1) hooks installed (`--models-config`;
  default OFF), (2) allow_cloud=True (orchestrator default True; bench
  hardcodes False), (3) an enabled [models.mc_resolve_cloud] route — an
  unset ${MC_RESOLVE_CLOUD_MODEL} expands to '' and fails gateway
  construction loudly; omitting the task instead yields 'not_configured' →
  stage silently skipped → REVIEW. OPENROUTER_API_KEY comes from the
  environment only, never the file.
- **ModelGateway route:** `mc_resolve_cloud` — backend='openrouter',
  model='${MC_RESOLVE_CLOUD_MODEL}', max_tokens=120,
  reasoning={effort='none'}, prompt_version='mc-v1', temperature 0.0 from
  [defaults]. Ledger/budget classify it as cloud via
  usage.effective_provider/is_cloud_route (a nominally-'openai' route pointed
  at openrouter.ai also counts as cloud). Privacy guard, request cache,
  BudgetManager check/charge, and usage ledger all apply.
- **Fallback path:** Terminal fallback is human review: cloud error /
  not_configured / unclear / low confidence / out-of-candidates →
  MCResolution(None,'unclear',0.0,'review'), row stays 'ambiguous' with
  candidates. Conflict rule: cloud confident letter ≠ local confident letter
  → trace 'conflict' + REVIEW (neither model wins). Stage exceptions never
  crash grading; orchestrator.handle_model_failure classifies cloud/budget
  failures as PAUSE (deterministic/local results kept).
- **Cacheability:** Same requestcache (fingerprint includes the cloud route's
  model/prompt_version and the band PNG SHA-256); reruns of the same job
  re-serve cloud answers free (ledger records cache_hit). No cross-row
  caching — each band crop is unique ink.
- **Relative cost importance:** Low absolute cost, budget-guarded (~800 in /
  ~40 out tokens estimated; volume is the residue of two prior filters).
  Per-call stakes are high relative to cost — each call decides one student's
  recorded answer — so model quality should dominate price.
- **Selection benchmark:** No live run exists. Decisive asset: the same
  audited ambiguous-row set (evaluation/prob/manual_audit.json via
  scripts/mc_fallback_bench.py) rerun with a cloud-enabled config and
  allow_cloud=True — the script currently hardcodes allow_cloud=False, so a
  small harness change is required — scoring the 9/10 rows the local model
  left unclear. Required metrics: accuracy_on_resolved_with_reference (must
  stay 1.0 — a wrong cloud resolution is worse than review), resolution rate
  on locally-unresolved rows, disagreement rate vs deterministic candidates.
  Monitoring: MCResolverStats.cloud_mc_escalation_rate / cloud_resolved +
  the usage ledger; the frozen 'mc_resolver' canary suite is the promotion
  contract for any cloud model change, but no suite is populated. Score
  candidates on agreement-within-candidates, not raw accuracy alone.
- **Evidence:** autograder/mcresolve.py:14-127; models.example.toml:45-50;
  scripts/mc_fallback_bench.py:7-9,85-88;
  evaluation/mc_fallback/qwen38_local_v1/summary.json:9-13.

## VARIANT_RESOLVE

- **Purpose:** Read the printed variant marker (e.g. a flower symbol) on an
  exam's cover page so the pipeline knows which answer-key column and which
  alignment apply. The model ONLY reports what symbol it sees against a
  supplied marker catalogue; the marker-to-variant decision is deterministic:
  `resolve_marker_name` (exact id → alias → unique discriminative token →
  ≥80% unique description overlap; ambiguity resolves to nothing) then
  `decide_version` via the authoritative `<key>.variants.json`. Variant is
  never inferred from student answers, grades, or best-scoring key. A second
  use of the same role name: gateway tasks variant_resolve /
  variant_resolve_cloud catalogue the markers during one-time package
  discovery (building variants.json) — not wired into any production entry
  point yet.
- **Inputs:** Live detection: one cover-page image (marker_page default 1, at
  pipeline render resolution; red ink masked in batch runs) + a text block
  with the marker catalogue (name: description, optional
  marker_location_hint). No answers, no key content, no student identity.
  Discovery: cover PNG + the list of version ids to catalogue against.
- **Output schema:** Detection: `VariantDetection` {marker_seen,
  matched_marker|null, confident, page_region, obstruction_note} →
  deterministic decide_version → `VersionDecision` {version, description,
  uncertain} plus an audit record persisted as result.variant_detection.
  Discovery: `MarkerCatalog` {n_variants, markers[{id, variant, description,
  aliases?}], marker_page, marker_kind, identical_question_order, confident
  (default false), notes}, converted to the variants.json contract only if
  variant ids exactly cover the key versions AND confident=true.
- **Vision required:** Yes — both calls attach the cover-page image; visual
  shape discrimination (petal count/shape) explicitly forbidden from using
  page text ("Only the printed marker counts").
- **Reasoning required:** Low — single-symbol match against 2–4 described
  candidates with constrained JSON; the deterministic resolver
  (RESOLVER_VERSION 'marker-resolver-v3-discriminative-tokens') absorbs known
  model failure modes (alias echo, description copied verbatim, own-words
  sighting). Local route sets think=false / 300 max_tokens; the detection
  call caps at max_tokens=800.
- **Call frequency:** Live: exactly 1 call per exam run when
  `<key>.variants.json` exists and `--version=auto`; 0 when the operator pins
  `--version` or no config exists (legacy answer-agreement detection then
  applies). The call re-runs on every pipeline invocation — it sits before
  the resume-gated stages and has no reuse wrapper. Discovery: at most 2
  gateway calls (local, then cloud) once per package build; 0 when the
  deterministic text-layer stage recovers printed version labels; results
  persist per package fingerprint.
- **Local-first:** Yes, structurally enforced on the live path:
  detect_variant runs on the legacy direct VisionBackend and
  `guard_direct_cloud_backend` refuses cloud there. The example config runs
  the discovery local task on Ollama qwen3.8:27b-q4_K_M; the discovery ladder
  is deterministic text-layer → local variant_resolve → cloud
  variant_resolve_cloud → human.
- **Cloud necessity:** No for per-exam detection (cloud is impossible there
  without moving the call behind the gateway). For package cataloguing, cloud
  is an optional once-per-package escalation; if both model tiers fail the
  fact goes to needs_human('variants'). The gateway discovery route is
  currently reachable only from tests: prepare_exam_package (the sole
  discover_package caller) is invoked nowhere in cli.py/webui.py, and
  install_hooks wires MC/policy hooks only, never variant.
- **Production call path:** cmd_grade → guard_direct_cloud_backend →
  run_grade_pipeline → load_variant_config (cli.py:676) →
  `detect_variant(backend, pages, cfg)` at cli.py:689 — a DIRECT
  VisionBackend.parse (variant.py:181-186), NOT a gateway call even with
  `--models-config` → deterministic decide_version → the version drives
  alignment choice, extraction fingerprint, and key-column scoring; audit
  record written into result.variant_detection. eval-batch surfaces
  detected_variant / variant_uncertain per exam.
- **ModelGateway route:** `variant_resolve` ([models.variant_resolve]
  backend=ollama, model=qwen3.8:27b-q4_K_M, localhost:11434/v1,
  max_tokens=300, think=false, prompt_version='variant-v1') and
  `variant_resolve_cloud` ([models.variant_resolve_cloud] backend=openrouter,
  model=${VARIANT_RESOLVE_CLOUD_MODEL}, max_tokens=300,
  prompt_version='variant-v1'). Used ONLY by
  discovery.resolve_markers_with_models (VARIANT_RESOLVE_SYSTEM →
  MarkerCatalog); missing routes are skipped silently. The per-exam detection
  call has NO gateway route today — migrating it would require a new task
  route plus the gateway's privacy/budget/ledger wrap.
- **Fallback path:** Deterministic, never score-based: (1) marker matched +
  confident → variant; (2) matched without confidence → provisional variant,
  uncertain=true → human review; (3) unmatched/unknown marker → first variant
  in sorted mapping (documented arbitrary choice), uncertain=true → review;
  (4) mapped variant absent from key.versions → key.versions[0] provisional,
  error recorded; (5) no variants.json → legacy answer-agreement detection
  with its own uncertainty margin; (6) discovery ladder ends in
  DiscoveryFact 'unresolved' → needs_human, with lecturer resolutions
  persisted in VariantCatalogStore and reused.
- **Cacheability:** Live detection: NONE — direct backend.parse bypasses the
  requestcache and re-runs per invocation. Its meaning is still
  fingerprint-protected: VARIANT_DETECT_SYSTEM and RESOLVER_VERSION enter the
  prompts-version hash, the variants.json config_fingerprint enters the
  resume fingerprint, and the selected variant enters the extraction
  fingerprint — artefacts can never be reused across variant interpretations.
  Discovery calls: requestcache + once-per-package persistence in
  VariantCatalogStore + never-overwritten `<key>.variants.json` sidecars.
  (Per-variant ALIGNMENT caching belongs to the alignment role.)
- **Relative cost importance:** Low direct cost (one ≤800-token single-image
  call per exam plus at most two 300-token calls per package build). High
  error cost — a confidently wrong marker read grades the whole exam against
  the wrong key column and ordering; every non-confident outcome routes to
  human review and scores never influence the choice. Optimize
  false-confidence rate, not price.
- **Selection benchmark:** No dedicated eval asset. Available: the
  hand-verified marker→variant ground truth in docs/variants.md:32-43
  (test/002 = many-petal daisy = A3, test/003 = five-petal star = A2,
  sample_data/student_exam.pdf = four-petal clover = A1; anchored 2026-07-13
  by manual inspection, no student answers used) combined with eval-batch's
  detected_variant / variant_uncertain columns. Select by variant accuracy on
  all annotated batch exams with ZERO confident-mismatches (confident=true +
  wrong variant bypasses review — the failure mode), uncertain-rate as
  tiebreaker. A labeled cover-page set under evaluation/ is an open gap.
- **Evidence:** autograder/variant.py:99-186,189-281;
  autograder/cli.py:675-691; models.example.toml:52-56,68-74;
  docs/variants.md:10-43.

## ALIGN_RESOLVE

- **Purpose:** Map the sub-item numbering as PRINTED in one exam-variant
  booklet onto the answer key's canonical ids (variants shuffle question and
  sub-item order), so extraction works in the numbering the student saw and
  scoring remaps back. Two implementations coexist: (a) gateway discovery
  tasks align_resolve / align_resolve_cloud — a text-only permutation
  proposal per (variant, question), invoked only after deterministic
  text/structure matching fails (autograder/alignment.py); (b) the legacy
  pipeline's `derive_alignment` — a vision call per question chunk over the
  variant's printed pages, invoked per graded exam when no operator override
  exists (autograder/variant.py, wired in cli.py).
- **Inputs:** Gateway: ONE text block with the canonical sub-items (id +
  prompt, explicitly no answers) and the variant's printed (printed_id, text)
  list, under `ALIGN_SYSTEM`; printed items come from the booklet's text
  layer or OCR via split_numbered_items. Legacy: key ids+prompts for a chunk
  of ≤10 items, plus PAGE IMAGES of that question's booklet pages
  (survey-selected), under `prompts.ALIGNMENT_SYSTEM`; max_tokens=2500 per
  chunk call.
- **Output schema:** Gateway: `PermutationProposal` {question_id,
  printed_to_key, confident, notes} — accepted only if confident=True AND an
  exact bijection between printed-id and key-id sets. Legacy:
  `VariantAlignment` {variant, questions[{question_id, printed_to_key,
  identical_order}], confident, notes}, checked by `validate_alignment`
  (complete coverage, no duplicate targets, no unknown key ids, no printed-id
  collisions after normalization).
- **Vision required:** Split by path — gateway align_resolve: NO (pure text);
  legacy derive_alignment: YES (rendered page images, reading the numbers
  printed next to matching content).
- **Reasoning required:** Light — bounded content-matching/permutation
  assembly with a strict schema; the deterministic ladder absorbs the easy
  mass and acceptance is gated on the confident flag plus bijection
  validation. No reasoning config exists for an align task (none is defined
  at all).
- **Call frequency:** Gateway: at most 1 local + 1 cloud call per (variant,
  question) that deterministic matching cannot settle, once per PACKAGE
  build, then persisted. In current production wiring this is 0:
  prepare_exam_package calls discover_package WITHOUT
  printed_items_by_variant, so the model stage is dormant (identity assumed,
  review-flag safety nets apply); tests only. Legacy: per (key, variant) —
  ceil(n_sub_items/10) calls per question (ALIGN_CHUNK_SIZE=10; e.g. an
  8+8+20 key = 4 calls); at most once per (key fingerprint, variant) via the
  persistent cache; 0 when an operator `.alignment.json` override exists.
- **Local-first:** Yes by design — the documented ladder is deterministic
  text → deterministic structure → LOCAL align_resolve → align_resolve_cloud
  → human, with the local tier strictly first in model_align_question. The
  legacy path uses whatever single VisionBackend the pipeline is configured
  with (typically local Ollama qwen3-vl).
- **Cloud necessity:** No. align_resolve_cloud is an optional targeted
  escalation; unconfigured or unconfident results become 'unresolved' and
  escalate to HUMAN — never guessed. The legacy path likewise needs no
  cloud: invalid derivations fall back to identity numbering with universal
  review flags.
- **Production call path:** Legacy (the only live model path today): if a
  variant config exists and no operator override matches,
  `derive_alignment(backend, key, survey, pages, variant)` is called DIRECTLY
  on the VisionBackend (cli.py:783), bypassing the gateway entirely; the
  result is validated, cached, and always review-flagged. Gateway:
  discover_package → alignment.align_variant → model_align_question →
  gateway.call(task='align_resolve' then 'align_resolve_cloud', meta
  stage='discovery' + question_id) — reachable only when a caller passes
  printed_items_by_variant, which no production caller does.
- **ModelGateway route:** Tasks `align_resolve` (local tier) and
  `align_resolve_cloud` (cloud tier) — CONFIG-GATED AND UNCONFIGURED BY
  DEFAULT: models.example.toml defines no [models.align_resolve] or
  [models.align_resolve_cloud] section (and gateway.py's conventional task
  list omits them), so gateway.route() raises GatewayConfigError, which
  model_align_question swallows (`except Exception: continue`) — the tier is
  silently skipped and the question escalates. A live deployment must add
  both routes to models.toml. The legacy derive_alignment has NO gateway
  route at all (direct VisionBackend.parse — the known
  legacy-bypasses-gateway gap).
- **Fallback path:** Gateway ladder: (1) deterministic_align_question —
  text-fingerprint bijection with min_score 0.35 / min_margin 0.15, identity
  fast-path; ambiguity or non-bijection → unresolved, not a guess; (2) local
  align_resolve; (3) align_resolve_cloud; each proposal accepted only if
  confident AND a complete one-to-one map; (4) unresolved →
  alignment_contract lists the variant for HUMAN resolution. Legacy: operator
  `.alignment.json` override (hard-validated, wins outright) → persistent
  cache → derive_alignment → validate_alignment; any problem →
  identity_alignment; then EVERY non-operator alignment stamps every sub-item
  with an 'unresolved_alignment' note and caps confidence at 0.5 for human
  review — model-derived alignments are never silently trusted (two live
  failures: an incomplete map; a complete-but-wrong identity claim).
- **Cacheability:** Three real caches: (1) gateway requestcache (cacheable
  defaults True; route fields + prompt_version + system/text/schema hashes);
  (2) package persistence — VariantCatalogStore per package fingerprint (key
  bytes + exam bytes) + the `<key stem>.alignment.json` sidecar (never
  overwritten — a lecturer's manual mapping wins); (3) legacy alignment cache
  `align_<fp[:32]>.json` in keycache.default_cache_dir() (GRADER_KEY_CACHE /
  LOCALAPPDATA autograder/key_cache), fingerprint = key fingerprint + variant
  + sha256(ALIGNMENT_SYSTEM) + VariantAlignment schema — self-invalidates on
  prompt/schema edits; skippable with `--no_key_cache`.
- **Relative cost importance:** Low direct cost (tiny text calls once per
  package; legacy vision calls up to 2500 output tokens amortized once per
  (key, variant) across a whole batch). The binding constraint is not spend
  but the cost of a wrong confident permutation — it silently misgrades every
  student of a shuffled variant, which is why every model-derived alignment
  is review-flagged regardless of validity.
- **Selection benchmark:** None exists — the only artifact,
  evaluation/before_exam002_alignment.json, is an identity-alignment snapshot
  ("identity (no alignment derivation ran)"), not labeled reordering ground
  truth; tests/test_alignment.py covers synthetic permutations only. Needs a
  to-be-built labeled set of real reordered variant booklets (A1/A2/A3
  families), scored on exact-permutation accuracy per (variant, question),
  unresolved rate, and a hard zero tolerance for confident-but-wrong
  permutations — maximize resolved coverage under that zero-wrong constraint.
  Note: the per-tier silent skip of GatewayConfigError means a typo'd or
  missing route degrades to 'unresolved → human' rather than erroring — safe
  but easy to misread as model failure.
- **Evidence:** autograder/alignment.py:9-24,105-186;
  autograder/variant.py:415-570; autograder/cli.py:750-838;
  models.example.toml:17-82.

## RAG_EMBED_RETRIEVE

- **Purpose:** Local course-material RAG for the GRADING side: embed
  lecturer-uploaded course material (PDF/TXT/MD/DOCX) into a persistent
  per-course cosine index, then retrieve top-k chunks per exam question as
  supplemental, budgeted context in QuestionGradingPacks. Strictly separate
  from the frozen OCR-repair RAG arm (scripts/m2_rag_ocr.py); grading RAG
  never touches student text, and answer-key-like material is refused at
  ingestion (filename gate + conservative content screen).
- **Inputs:** Index build: all chunk texts of one course (heading/
  paragraph-aware chunks, target 600 / max 1100 chars, 1-paragraph overlap)
  embedded in ONE batched call. Retrieval: an identity-free question-level
  query = question_text + rubric lines + official solution, hard-capped at
  1,500 chars (`gradingpack.rag_query`). Student OCR text, names, ids, and
  paths are NEVER part of the query — a bad reading must not steer retrieval
  and bias the grade.
- **Output schema:** Embed: float32 ndarray [n_texts, dim] from Ollama POST
  /api/embed, L2-normalized before persist/compare. Retrieve:
  list[{chunk_id, text, source, page, section, similarity}] (cosine top-k).
  Pack-side: `RagEvidence`{chunk_id, source, page, similarity, text,
  text_hash} under rag_evidence (active) or rag_prepared (lazy cache),
  deterministically truncated to a 1,200-char budget (top_k=2 default),
  rendered into the grader prompt as "Course context (supplemental —
  rubric/solution take precedence)".
- **Vision required:** No — text-only embedding (bge-m3); course PDFs are
  parsed to text deterministically via PyMuPDF, not vision-OCR'd.
- **Reasoning required:** No — pure embedding + cosine retrieval. The only
  model-quality dimension is multilingual (Hebrew) embedding quality, which
  is why the default is multilingual bge-m3.
- **Call frequency:** Per course-material change: 1 batched embed call
  (operator-triggered from the WebUI Courses tab). Per exam PACKAGE build:
  exactly 1 query-embed + retrieval per question (RAG_ALWAYS attaches at
  build; lazy policies prepare once at build). Per student / per grading
  call: ZERO — packs are persisted in the PackStore and reused for every
  student; `activate_rag` only copies prepared chunks into a new pack copy.
  Under the DEFAULT policy RAG_DISABLED: zero calls of any kind.
- **Local-first:** Local-ONLY today and should stay so: `ollama_embed_fn`
  posts directly to http://localhost:11434/api/embed with model bge-m3
  (DEFAULT_EMBED_MODEL); no API key, no cloud endpoint anywhere in the path.
  docs/grading-rag.md: "No cloud call, no cloud embedding, no index rebuild
  per question or student."
- **Cloud necessity:** No. Embedding/retrieval never leaves the machine. The
  only cloud cost RAG creates is DOWNSTREAM: activated chunks add OpenRouter
  input tokens to grade_primary/grade_escalate (~1,200 chars ≈ 300 est
  tokens, tracked as rag_chars/rag_tokens_est and in ledger rows).
  Preparation is explicitly free ("Preparation never costs provider tokens");
  use is what costs.
- **Production call path:** Prep (once per course): webui Courses tab →
  courses.add_source (key-like names refused, content screened) → "Build /
  rebuild index" → courses.build_index → parse_source → chunk_parsed →
  ollama_embed_fn → embeddings.npy + chunks.jsonl + index_manifest.json under
  courses/<id>/rag_index. Use (once per package): non-legacy modes →
  source_fingerprint → PackStore.load, else
  build_all_packs(retrieve=courses.retrieve) → attach_rag (RAG_ALWAYS) or
  prepare_rag (lazy) → `_retrieve_budgeted` → courses.retrieve (query embed
  via the default ollama_embed_fn — cli passes no embed_fn). At grading time
  escalate_grade consults the injected rag_attach=activate_rag only where the
  pack's policy says so. Same wiring exists in
  orchestrator.prepare_exam_package for the package-prep flow.
- **ModelGateway route:** NONE — deliberately outside ModelGateway: the
  conventional task list has no embed task and gateway.py contains no
  embedding code; ollama_embed_fn is a direct localhost httpx call, so it
  gets no gateway privacy/cache/budget/ledger wrap. That is acceptable only
  because it is local-only and the query is identity-free; any future cloud
  embedding would have to be brought under the gateway boundary. RAG's
  footprint reaches the ledger indirectly: rag_chars/rag_chunks/rag_policy/
  pack_hash ride on grading-call ledger rows.
- **Fallback path:** Missing index: courses.retrieve returns [] when
  embeddings.npy is absent. Optional-RAG policies degrade gracefully — no
  course/retriever/index or empty retrieval sets rag_available=false and
  grading continues WITHOUT context, never a failure or REVIEW by itself.
  Shadow mode swallows ANY failure (index, embedder, retrieval) so the
  authoritative legacy result is unaffected; reliability mode re-raises a
  hard embedder failure at pack build. WebUI index-build failure surfaces the
  "ollama pull bge-m3" hint. OCR_UNRESOLVED readings never reach grading RAG.
- **Cacheability:** Three layers, all content-addressed, none in the gateway
  requestcache: (1) the persistent per-course index — embeddings.npy +
  chunks.jsonl + index_manifest.json, staleness via config_hash over source
  hashes + embed model + chunk config; parsed/<sha>.json caches
  deterministic extraction; (2) the PackStore — retrieval results frozen into
  persisted packs keyed by source_fingerprint (key bytes, course index hash,
  policies, top_k, char budget, PACK_SCHEMA_VERSION v2,
  RETRIEVAL_CONFIG_VERSION r1, rag_policy), so retrieval never re-runs per
  student; (3) per-chunk content hashes (chunk_id, RagEvidence.text_hash,
  rag_index_fingerprint) so any change rebuilds exactly the right thing.
- **Relative cost importance:** Low in provider dollars (zero cloud tokens
  for embed/retrieve; local GPU/CPU once per course + once per package). The
  economically relevant knobs are downstream: top_k=2 and max 1,200 rag chars
  (~300 est input tokens) per grading call ONLY when a policy activates
  context — which is why the default ships RAG_DISABLED until measured.
  Embedder choice matters for retrieval QUALITY (Hebrew course material),
  not spend.
- **Selection benchmark:** scripts/grading_rag_ab.py — the pre-registered
  frozen paired A/B (5 cells: e002_q1_r2, e003_q1_r1, e003_q1_r5, e004_q1_r1,
  e006_q1_r2): arm A no RAG vs arm B identical plus frozen local bge-m3 top-2
  evidence, joined post-hoc against the frozen reference verdicts in
  evaluation/m2_grading/gemini3_flash.jsonl; plus an escalation-only G
  analysis. Its output dir evaluation/grading_rag_ab_v1/ does not exist — the
  A/B has NOT been run, which is exactly why RAG_DISABLED remains the
  default. Caveat: that benchmark decides the RAG POLICY (does grading
  context help), not the embedder; no retrieval-quality benchmark for the
  embedding model exists — swapping bge-m3 would require creating one over
  the course corpus first.
- **Evidence:** autograder/courses.py:67,289-394;
  autograder/gradingpack.py:29-43,263-296,434-447; docs/grading-rag.md:22-36;
  scripts/grading_rag_ab.py:1-42.

## Cross-role facts

- **Task names in configuration vs code.** models.example.toml defines
  routes for: ocr_primary, ocr_verify, grade_primary, grade_escalate,
  mc_resolve, mc_resolve_cloud, variant_resolve, variant_resolve_cloud,
  policy_infer. Referenced only in code, with no example entry:
  **align_resolve, align_resolve_cloud, policy_infer_cloud** — a live
  deployment must add them to models.toml before those tiers can run.
  (Caveat preserved from the source record: the KEY_PARSE record describes
  policy_infer/policy_infer_cloud as "configured in the example toml" while
  its own evidence lists only policy_infer among the example tasks.)
  gateway.py's conventional task list additionally names package_inspect and
  omits the align tasks. No gateway task exists at all for key parsing, the
  survey, or per-exam variant detection, and RAG embedding is deliberately
  outside the gateway.
- **Universal cloud rule.** Cloud calls go through ModelGateway only:
  `guard_direct_cloud_backend` refuses any cloud provider on the direct
  legacy path (cli.py:16-21, 489-514), and every gateway.call is wrapped
  privacy scan → request cache → budget → provider → ledger
  (gateway.py:207-242). Known caveats recorded above: the anthropic
  dev-backend bypass applies to the survey path, and legacy derive_alignment
  is a direct VisionBackend call outside the gateway wrap.
- **Frequency classes.**
  - Per package (once per key/course/package build, then cached): KEY_PARSE
    (keycache), ALIGN_RESOLVE (legacy per (key, variant) via the persistent
    alignment cache; gateway tier once per package build), VARIANT_RESOLVE
    discovery cataloguing (≤2 calls per package), RAG_EMBED_RETRIEVE (one
    embed per course build + one retrieval per question per package, zero per
    student).
  - Per exam: SURVEY (1 call), VARIANT_RESOLVE live detection (1 uncached
    call), GRADE_PRIMARY path A (1 call per question needing judging).
  - Per item (sub-item / row / suspicious read): OCR_PRIMARY, OCR_VERIFY,
    GRADE_PRIMARY path B, GRADE_ESCALATE, MC_RESOLVE_LOCAL, MC_RESOLVE_CLOUD.
- **Model selection.** Candidate models live in
  evaluation/model_selection/candidates.toml; winners are chosen only by the
  benchmarks in docs/model-selection.md.
