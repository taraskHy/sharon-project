# GUI gap report — current state and gaps for the next UI phase

Date: 2026-08-21

Purpose: source-verified inventory of the existing web UI plus a gap analysis for the
next UI phase. This is explicitly **not a redesign** — no mockups, no new layouts, no
proposed screens. It records what exists (with file:line anchors), what is missing per
target area, and which already-built backend pieces can be wired rather than rebuilt.

Scope note: grading behavior (scoring, policies, model routing semantics) is out of
scope. This document covers UI surfaces and their wiring only.

## Current state

Single Streamlit app: `autograder/webui.py` (1008 lines), launched via `autograder ui`
-> `streamlit run webui.py` (`cli.py:1039-1041`). Two-section sidebar + four tabs
(`webui.py:261-263`): New batch, Jobs & results, Courses (experimental RAG),
Models & OpenRouter. `autograder/reviewui.py` (399 lines) is a headless, testable
backend (review items, resolutions, batch overview, settings summary, decision
traces) — not a screen of its own. Grading runs in a **detached** `autograder run-job`
subprocess (DETACHED_PROCESS on Windows, pid persisted to `runner.pid`;
`webui.py:44-92`); the UI polls disk state via a `st.fragment` every 3 s
(`webui.py:619`). Privacy is a stated design constraint: model-visible inputs never
contain original filenames or private paths, enforced by tests (`webui.py:11-12`,
`docs/ui.md:72-77`).

Per-surface inventory:

- **Sidebar 1 — Model backend** (`webui.py:161-216`): legacy validated pipeline
  config (server URL, model, advanced expander: structured mode, timeout, max tokens,
  image edges), defaults from `grader.toml`; values become CLI flags stored in
  `job.json` and forwarded verbatim to every grading subprocess (`webui.py:218-221`).
  "Check backend" button does a direct `create_backend(...).health_check()` — legacy
  path, not through the gateway (`webui.py:193-203`).
- **Sidebar 2 — Grading route** (`webui.py:223-259`): mode selector
  legacy/reliability/shadow only if `models.toml` exists (default legacy; hard-coded
  legacy with a caption otherwise), plus a RAG policy selectbox (RAG_DISABLED default)
  for non-legacy modes; non-legacy adds `--grading-mode/--models-config/--rag-policy`.
- **New batch tab** (`webui.py:426-600`): single-page form — radio between
  "Configured exam package" (auto-discovered `*.template.json` + key sidecars, with
  variant marker table) and "Upload key & configuration" (key PDF/JSON, optional
  rubric, exam mode incl. a raw-JSON textarea for per-question modes, answer-sheet
  rule with comma-separated pages input, optional variants JSON); multi-upload of
  student exams (PDF/images/ZIP), optional course selectbox for RAG, red-annotation
  mask toggle, "Create job" -> `jobs.create_job` with intake-issue warnings
  (`webui.py:583-600`). No pre-create key-parse preview or validation.
- **Jobs & results tab**: one job at a time via selectbox (`webui.py:613-615`).
  Status panel (`webui.py:619-708`): 3 s auto-refresh, status incl. derived
  "interrupted" (pid dead), 8 metrics, current exam+stage, progress bar, 5 controls
  (Start/Resume, safe Pause, Stop, Refresh, Rebuild combined reports), per-exam
  dataframe. "Package setup" preflight renders READY or blocking-findings table +
  reviews_avoided count (`webui.py:715-746`). "Estimated cloud usage"
  (`webui.py:748-782`): 5 pre-run ESTIMATE metrics + per-call-type breakdown, never
  mixed with actual usage. "Batch checks" (`webui.py:784-831`): anomaly-driven
  severity-coded warnings table, 4 summary metrics, and a read-only prioritized
  grouped "Review queue" expander (priority affects order only). "Exam details"
  (`webui.py:833-993`): 4 metrics, variant evidence, per-item dataframe, quick-review
  expander with one-click option buttons persisting via ResolutionStore to
  `review_resolutions.json` (`webui.py:896-931`), shadow-comparison expander
  (non-authoritative labeled, `webui.py:933-957`), per-item decision-trace expander
  via `reviewui.decision_trace_for` (`webui.py:959-970`), per-exam
  result.json/report.md downloads (`webui.py:972-982`), 30-line grade.log tail for
  unfinished exams (`webui.py:988-993`). Batch downloads: combined CSV/JSON/
  summary.md/reports.zip buttons (`webui.py:995-1008`).
- **Courses tab** (`webui.py:333-419`): create/select course, multi-file upload with
  answer-key/rubric content screening + operator override toggle, index status line,
  per-source remove, "Build / rebuild index" with local Ollama embeddings. Labeled
  experimental.
- **Models & OpenRouter tab** (`webui.py:270-325`): read-only — 3 status metrics,
  task->model table from `gateway.describe()`, usage-ledger metrics (requests,
  tokens, cost, cache-hit rate, % fully local), request-cache stats caption, budget
  table + "spent this process" caption, zero-cost "Test connection" health probe.
  API key is env-only, never displayed; no editing of `models.toml` from the UI.

## Gap analysis

### 1. Dashboard
- Exists: nothing. Jobs tab shows exactly one job at a time (`webui.py:613-615`).
- Missing: any cross-job overview — jobs in flight, total review backlog, recent
  activity, cumulative spend, backend health at a glance.
- Wire, don't rebuild: `reviewui.settings_summary` / `test_connection`
  (`reviewui.py:235-259`) for health/status; the persisted `UsageLedger`
  (`gateway_ledger/usage.jsonl`, `orchestrator.py:52-53`) for cumulative spend;
  `reviewui.batch_overview` per job for backlog counts.

### 2. Exam Setup wizard
- Exists: New batch as one long single-page form (`webui.py:426-600`) with raw-JSON
  textarea and comma-separated page numbers; package preflight only appears AFTER
  job creation, in a different tab (`webui.py:715-746`).
- Missing: stepwise flow, key-parse preview before job creation, setup-time
  validation. Setup errors surface post-creation as blocking preflight findings or
  per-student review items.
- Wire, don't rebuild: the existing `preflight_package` path (rendered at
  `webui.py:715-746`) can run at setup time instead of only post-creation.

### 3. Grading progress
- Exists: 3 s-refresh counters, stage name, one progress bar (`webui.py:619-708`);
  grade.log tail only when no result exists yet (`webui.py:988-993`).
- Missing: ETA/throughput, per-stage progress within an exam, live log streaming for
  a running exam, and any live cloud-spend readout during a run.
- Wire, don't rebuild: per-job spend can be computed from the cross-process
  `usage.jsonl` ledger (`usage.py`, `orchestrator.py:52-53`) rather than in-process
  counters.

### 4. Human Review Queue
- Exists: read-only grouped priority queue expander at batch level
  (`webui.py:822-831`); resolving happens only per-exam via Quick review
  (`webui.py:896-931`) persisted through `ResolutionStore`.
- Missing: a dedicated walk-the-queue flow across exams in priority order; an
  apply-to-all control (backend `ResolutionStore.apply_to_all` at
  `reviewui.py:188-219` — variant/layout only, audited to `apply_to_all.jsonl` — has
  no UI button, only an eligibility caption at `webui.py:929-931`); a numeric input
  flow behind the "set score" option label.
- Defect: `build_review_items` is called without `crops`/`chain_traces`/`packs`/
  `warnings` (`webui.py:904` vs signature at `reviewui.py:60-62`), so review items
  render with **no image crops, no MC chain traces, no grading-pack context** —
  reviewing handwriting requires opening files externally.
- Wire, don't rebuild: `reviewui.review_queue()` (never called by webui),
  `apply_to_all`, and `ReviewItem.crop_png_b64`/`question_context`/
  `disputed_regions` are built and tested; the queue screen is mostly wiring.

### 5. Results/export
- Exists: 4 fixed batch download buttons (CSV/JSON/summary.md/reports.zip,
  `webui.py:995-1008`) plus per-exam result.json/report.md (`webui.py:972-982`).
- Missing: XLSX, LMS/gradebook format, score-distribution or per-question views,
  filtering/sorting beyond raw dataframes, export of review-resolutions/audit trail.
- Wire, don't rebuild: `jobs.combine_outputs` already rebuilds combined artifacts;
  new formats are additional renderers over the same data.

### 6. Advanced settings/diagnostics
- Exists: read-only Models & OpenRouter tab (`webui.py:270-325`); single-item
  decision-trace expander (`webui.py:959-970`); cache stats caption only
  (`webui.py:316-317`).
- Missing: models.toml viewing/editing beyond the task table, cache management,
  ledger browser (aggregate metrics only), bulk decision-trace browser; sidebar
  grader.toml edits are not persisted back to `grader.toml`. Any routing or budget
  change means hand-editing TOML; diagnosing a batch means opening JSONL on disk.
- Wire, don't rebuild: `reviewui.decision_trace_for` (`reviewui.py:334-399`) already
  handles decisions.jsonl / shadow labeling / reconstructed fallback for a browser.

### 7. Model selection/benchmark status
- Exists: none. The task->model table (`webui.py:304-306`) is read-only config
  display; nothing in webui.py references `evaluation/`.
- Missing: per-task model selection in the UI, and any benchmark/agreement evidence
  to justify a model choice — model changes are blind config edits, contradicting
  the project's measure-before-adopt stance.
- Wire, don't rebuild: `evaluation/results.json`, `results.csv`, `report.md`,
  `performance.md`, `error_analysis.md`, and `evaluation/hebrew_bench/` (human
  annotations, per-model output runs) exist on disk and could back a
  benchmark-status-per-role view.

### 8. OpenRouter spend indicator
- Exists: partial — ledger aggregate + budget snapshot in the settings tab only
  (`webui.py:307-323`); pre-run estimate section in the Jobs tab
  (`webui.py:748-782`).
- Defect: the estimate's gateway lookup is dead wiring —
  `st.session_state.get("gateway")` at `webui.py:763` is never assigned anywhere in
  webui.py, so `estimate_job` always receives `gateway=None` and the estimated cost
  renders "—" (cost_unavailable_reason).
- Missing: spend display in the Jobs tab during grading, per-job/per-exam actual
  cost next to results. Budget counters are in-memory per-process
  (`usage.py:200-206,247-261`) while spending happens in the detached run-job
  process — the UI-process snapshot cannot reflect a running batch.
- Wire, don't rebuild: `estimate.py` (`estimate_job`/`load_pricing`) is already
  rendered — setting the gateway lights up the cost figure; actual spend must come
  from the persisted `usage.jsonl` ledger.

## Notes for the next phase

- `docs/ui.md` (77 lines) is stale: it covers only the sidebar backend, New batch,
  and Jobs & results — it omits the Courses tab, Models & OpenRouter tab, grading
  route/RAG policy sidebar, quick review, shadow comparison, preflight, and
  estimates. Refresh it alongside any UI work.
- Spend truth must come from the persisted ledger (`gateway_ledger/usage.jsonl`,
  cross-process), never from in-process `BudgetManager` counters — grading spends in
  a detached subprocess the UI process cannot see.
- Fix the two known defects first: the never-set `st.session_state["gateway"]`
  (`webui.py:763`) killing the cost estimate, and `build_review_items` called
  without crops/chain_traces/packs (`webui.py:904`) leaving review items without
  visual evidence.
- The review-queue backend (grouping, priority tiers, one-decision-covers-all,
  apply-to-all with auditing) is built and tested in `reviewui.py` but unwired; the
  Human Review Queue surface is primarily a wiring task.
- Model-selection/benchmark surfaces should read existing `evaluation/` assets and
  `evaluation/model_selection/candidates.toml` (roles remain UNSELECTED until
  benchmarked) rather than introducing new data stores.
- Preserve existing invariants when wiring: priority affects order only (never a
  grade), resolutions never mutate `result.json`, shadow output stays labeled
  non-authoritative, estimates never mix with actual ledger usage, and the API key
  is never displayed or stored.
