# Claude Vision assisted-annotation experiment

Status: **BLOCKED — no Anthropic API credential on this machine.**
No exam image has been sent anywhere; zero API calls have been made.
Everything up to the first API call is built, tested, and
pre-registered so the benchmark is a two-command run once a credential
exists.

## Privacy pre-flight (confirmed)

This experiment, when run, sends **anonymized handwriting crops to
Anthropic**: cleaned student-ink-only line crops, plus (pass B) the
same student's cleaned explanation-cell image. Payloads contain base64
image bytes and fixed instruction text only — no filenames, sample
ids, grades, student identifiers, instructor scores or ink
(`cell_orig` is never accessed — enforced by test), answer keys,
rubrics, question text, hidden ground truth, or neighboring students'
answers. Full statement: `evaluation/claude_candidates/PROTOCOL.md`.

## Pre-registered design (frozen before any inference)

- 30 owner-verified clean lines, 10 each from writers e004/e005/e006
  (deterministic; excludes the 20 overfit ids; ids-only record:
  `evaluation/claude_candidates/claude_bench_ids.json`).
- Two passes on `claude-opus-4-8` (most capable Opus tier,
  high-resolution vision): A = line crop; B = line crop + cleaned cell
  context. Owner's transcription-only instruction verbatim; adaptive
  thinking; no sampling parameters. Raw responses + model id + prompt
  version + image SHA256s + timestamps + latency + token usage are
  saved per call BEFORE the evaluator (the only GT reader) may run —
  it refuses to score a partial run (tested).
- Owner's acceptance gate, frozen in PROTOCOL.md and pinned by
  `tests/test_claude_candidates.py::test_gate_constants_match_owner_protocol`:
  exact-match ≥ 40 %, median CER ≤ 0.10, major hallucination (≥ 2
  inserted words) on ≤ 5 % of lines, and an A↔B-agreement subset
  (≥ 5 lines) with CER ≤ 0.10. ACCEPT iff at least one pass meets all
  four. On REJECT: no candidates for untouched lines, no app changes
  beyond recording the negative result.

## Safety guarantees already in force (9 tests, suite 163/163)

- `build_request()` accepts only image bytes — there is no code path
  by which a transcription or id can enter a payload.
- Claude outputs live outside the pilot package with `verified: false`;
  the runner cannot write annotation records.
- Verified labels remain overwrite-locked in the app (per-sample
  unlock checkbox, 2026-07-17 campaign).
- The annotation app and lib contain no candidate UI and will fail the
  test suite if one appears without an ACCEPT verdict — candidate
  generation is disabled by default, and a failed benchmark leaves the
  manual workflow byte-identical.

## Context: prior evidence

The same 2026-07-17 overnight campaign benchmarked local candidates
(qwen3-vl:8b strict prompts, overfit-CRNN) on 66 verified lines: 0/198
candidates below CER 0.4 → prefill REJECTED. Claude Opus is a far
stronger vision model, hence this separate owner-authorized test; the
bar (median CER ≤ 0.10) is however ~8× stricter than anything any
model has achieved on this handwriting to date (best local: 0.635 on
a single best line).

## To resume (owner action, one of):

1. `ant auth login` (preferred — no static key), or
2. set `ANTHROPIC_API_KEY` for the session.

Then:

```
.venv\Scripts\python.exe scripts/claude_candidates_run.py generate --config claude_line
.venv\Scripts\python.exe scripts/claude_candidates_run.py generate --config claude_line_cell
.venv\Scripts\python.exe scripts/claude_candidates_eval.py
```

(~60 calls, est. a few dollars on Opus 4.8; resume-safe — already-saved
samples are skipped.) The evaluator prints ACCEPT/REJECT against the
frozen gate and writes `evaluation/claude_annotation_candidate_results.csv`
+ `evaluation/claude_candidates/eval_summary.json`; this report is then
updated with the exact metrics.

## Claude-Code subscription backend attempt (2026-07-17, owner-directed)

Owner declined separately billed API usage; an EXPERIMENTAL backend
(`scripts/claude_code_backend.py`) now drives the same benchmark
through the installed `claude -p` CLI on the Max subscription. The API
backend is unchanged and unused; the new backend hard-refuses to run
if ANTHROPIC_API_KEY is set, so it can never silently bill credits.

Auth verification (non-interactive equivalent of `/status`):
`~/.claude.json` oauthAccount shows `billingType: stripe_subscription`,
`organizationRateLimitTier: default_claude_max_20x` (Claude Max 20x) —
no Console/API-credit auth, no apiKeyHelper, no ANTHROPIC_API_KEY at
session/User/Machine level.

One-crop smoke test (sample e004_q1_r5__l1 as neutral `crop.png`, in a
throwaway sandbox containing ONLY that file; fresh stateless session;
allowlist `Read(crop.png)`; Bash/Write/Edit/Web*/Glob/Grep/Task
disallowed): the CLI executed and returned structured JSON with
`api_key_source: "none"`, but ended `is_error: true` — **"Not logged
in · Please run /login"**. Zero tokens consumed, zero Max usage, the
image was never read. Cause: the desktop app's Max login is not shared
with the standalone `claude.exe`; `~/.claude/.credentials.json` does
not exist. Raw artifacts:
`evaluation/claude_candidates/claude_code_smoke/{raw_stream.jsonl,analysis.json}`.

**Status: the Max subscription cannot serve this automated benchmark
through the current interface UNTIL a one-time CLI login is done by
the owner** (`claude setup-token`, or interactive `claude` → `/login`,
choosing the Claude-account subscription option — not an API key).
After that: re-run the smoke
(`.venv\Scripts\python.exe scripts/claude_code_backend.py smoke`),
and only with owner approval run the 30-line benchmark
(`generate --config ... --owner-approved`). No API fallback will be
used.

## Results — intermediate 10-line stage (2026-07-20): **REJECT**

Owner-approved 10-line retrospective (5 clear / 5 difficult by CRNN
confidence, writers e004+e005+e006; PROTOCOL.md addendum) through the
Max-subscription backend. Hygiene verified on all 20 records:
`api_key_source: none`, `image_read: true`, only the sandboxed
neutral-named crop(s) read, `verified: false`, zero API billing.
Ledger: `early10_results.csv`; summary: `early10_summary.json`; raw
streams retained per call.

| pass (10 lines) | mean CER | median CER | exact | no-edit | minor | major | omit | insert | major-halluc lines |
|---|---|---|---|---|---|---|---|---|---|
| A `claude_line` | .893 | **.799** | 0 | 0 | 0 | 9/10 | .014 | .336 | 6/10 |
| B `claude_line_cell` | .674 | **.703** | 0 | 0 | 0 | 8/10 | .000 | .250 | 5/10 |

- Candidates needing no edit: **0**. Needing minor edits: **0**.
  Taking longer to fix than manual transcription: **10/10 in both
  passes** (est. −25 s per 5 lines vs typing from scratch).
- A↔B agreement range .13–.69 — no line pair even reaches the lowest
  pre-registered threshold (0.8), so agreement identifies no subset
  at all, reliable or otherwise.
- Best single line anywhere: CER .314 (`e006_q1_r3__l1`, pass A) —
  better than any local candidate ever measured (.635), but still 2×
  the minor-edit bar.
- The CRNN-confidence "clear/difficult" split did not track Claude
  difficulty (difficult-group B mean .59 vs clear-group B .76) —
  the proxy is writer/legibility-specific, another sign these signals
  don't transfer.
- Qualitative: Claude produces real Hebrew words, sane line structure,
  honest [לא קריא] tokens, near-zero omissions — but the words are
  largely the WRONG words (insertion rate .25–.34, hallucinated words
  on 5–6 lines of 10). Cell context (B) clearly helps (.67 vs .89) yet
  remains far from usable.
- Max usage consumed: 21 calls total incl. smoke; 20-call benchmark:
  80 input + 4,225 output tokens metered by the CLI, 219 s wall,
  $1.76 API-equivalent — **$0 billed** (subscription).

**Early-rejection gate: ALL FOUR criteria fired** (zero time-saving
candidates; median CER > 0.25 in both passes; major hallucinations
> 5 % of lines in both passes; no reliable agreement subset).
**Verdict: REJECT Claude-assisted annotation.** Per protocol: the
remaining 20 lines were NOT run, the annotation app was NOT modified,
no candidates exist for untouched lines, and no annotation record was
touched. Manual annotation via the priority queue remains the path
(`evaluation/annotation_priority_queue.csv`).
