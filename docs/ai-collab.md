# AI collaboration harness (`tools/ai_collab`)

Bounded, local Claude (implementer) ↔ OpenAI/OpenRouter (reviewer) loop.
**Developer tooling only** — never imported by the autograder runtime, never
part of the grading flow, UI, OCR, RAG or job pipeline. The autograder runs
unchanged when this tool is absent or unconfigured. No student data enters
this system.

## Architecture

```
python -m tools.ai_collab start task.md
        │
        ▼
  Orchestrator (state machine, run.json persisted after every transition)
        │
  round N: 1. Claude adapter implements / applies findings
  (N ≤       - claude_code: headless `claude -p` child process
   max_       - manual:     you drive your own Claude Code session
   rounds)    - mock:       scripted (tests / rehearsal)
           2. orchestrator captures GROUND TRUTH itself:
              base/head commit, changed files, `git diff`, untracked files
              (redaction + deny-list applied before anything is persisted/sent)
           3. orchestrator runs configured test commands, records output
              (one bounded in-round fix attempt on failure)
           4. reviewer adapter (read-only): budget check → cache check →
              OpenRouter/OpenAI chat completion → strict JSON review
           5. verdict: APPROVED → stop | BLOCKED → stop |
              CHANGES_REQUIRED → gate (mode-dependent) → next round
```

Roles: **Claude** is the only agent that edits files, runs tests, commits.
The **reviewer** only ever receives text (task, context, handoff, diff, test
output) and returns JSON; it cannot execute anything. The reviewer is an API
model with a stable local context file — it does **not** inherit any browser
ChatGPT conversation.

## States

Pause states (wait for you): `AWAITING_CLAUDE` (manual mode),
`AWAITING_REVIEW_APPROVAL` (manual), `AWAITING_FIX_APPROVAL`
(manual + semi_auto), `USER_APPROVAL_REQUIRED` (Claude escalated an
out-of-scope problem).

Stop states: `APPROVED`, `CHANGES_REQUIRED`, `BLOCKED`, `MAX_ROUNDS`,
`TEST_FAILURE`, `BUDGET_EXHAUSTED`, `USER_APPROVAL_REQUIRED` (if you stop
there), `ERROR`, `STOPPED`. `final.json` records the final state once.

## Modes

| mode | Claude → reviewer | reviewer → Claude fixes |
|---|---|---|
| `manual` | pause for approval | pause for approval |
| `semi_auto` (default) | automatic | pause for approval |
| `auto_bounded` | automatic | automatic, ≤ max_rounds |

## Install / configure

Nothing to install (stdlib only, Python ≥3.11). Copy
`tools/ai_collab/config.example.toml` → `tools/ai_collab/config.toml`
(gitignored) and adjust. Environment:

- `OPENROUTER_API_KEY` — reviewer backend `openrouter` (priority backend)
- `OPENAI_API_KEY` — reviewer backend `openai`
- `AI_REVIEW_MODEL` — referenced by the default `[reviewer].model = "${AI_REVIEW_MODEL}"`

Keys are read from the environment at call time only; they are never written
to config, run artifacts, logs, or exceptions. To change the reviewer model,
set `AI_REVIEW_MODEL` (or edit `[reviewer].model` to another `${VAR}`).

## CLI

```powershell
.\.venv\Scripts\python.exe -m tools.ai_collab start tools\ai_collab\examples\demo_task.md --dry-run
.\.venv\Scripts\python.exe -m tools.ai_collab start runs\my_task.md [--task-id X] [--mode semi_auto] [--allow-dirty] [--create-branch feature/x]
.\.venv\Scripts\python.exe -m tools.ai_collab status  <task-id>
.\.venv\Scripts\python.exe -m tools.ai_collab continue <task-id>   # resume pause/interruption
.\.venv\Scripts\python.exe -m tools.ai_collab approve <task-id> [--note ...]
.\.venv\Scripts\python.exe -m tools.ai_collab stop    <task-id>
.\.venv\Scripts\python.exe -m tools.ai_collab list
```

`--dry-run` validates config + repo, previews the Claude command line, the
planned loop, and payload sizes — with **zero** API calls, zero child
processes, zero writes. Use it before every first real run.

## Claude Code invocation (verified against installed 2.1.215)

`claude -p --output-format json --permission-mode acceptEdits
--allowedTools <list> [--model m] [--max-budget-usd x]`, prompt via STDIN
(avoids Windows arg-length limits), cwd = repo root, subprocess timeout from
config. The handoff channel is a JSON **file** the prompt instructs Claude to
write (deterministic; the stdout JSON envelope is a fallback). This CLI
version has no `--max-turns`; bounding is timeout + optional
`--max-budget-usd` (print-mode flag). If headless invocation misbehaves, set
`[claude].mode = "manual"`: the orchestrator writes the prompt +
`MANUAL_INSTRUCTIONS.md` and waits for you to produce the handoff with your
own interactive session.

## Structured documents

Claude handoff (`claude_handoff.json`, spec §9): `task_id, round, status
(READY_FOR_REVIEW|BLOCKED|USER_APPROVAL_REQUIRED), summary, files_changed,
tests{commands,passed,failed}, architecture_changes, known_gaps,
questions_for_reviewer`. The orchestrator does not trust it: it captures git
state itself and notes discrepancies.

Review (`review.json`, spec §11): `verdict (APPROVED|CHANGES_REQUIRED|
BLOCKED), summary, findings[{id,severity,category,file,line_or_symbol,issue,
evidence,requested_change}], approved_scope, tests_requested,
context_requests`. `CHANGES_REQUIRED` requires ≥1 finding. A malformed reply
gets one strict-format retry, then `ERROR`. Findings with severity in
`[policy].block_on` (default critical+high) veto approval even if the
reviewer said APPROVED. `context_requests` may trigger at most one follow-up
call per round with sanitized, size-capped source files.

## Budgets (hard stops)

`[budget]`: `max_reviewer_calls`, `max_input_tokens` (estimated chars/4 when
the provider reports none), `max_output_tokens`, `max_cost_usd`
(provider-reported, e.g. OpenRouter `usage.include`). Checked **before**
every reviewer call; crossing any limit stops the run as `BUDGET_EXHAUSTED`.
Cache hits cost nothing. Per-call usage is persisted in
`reviewer_request_meta.json`, `run.json` and `audit.jsonl`.

## Request cache

`tools/ai_collab/cache/` (gitignored). Fingerprint: prompt version, reviewer
model, generation config, task hash, diff hash, context hash, test-output
hash. Identical logical requests reuse the stored review (never paid twice);
only validated reviews are cached.

## Git safety

Refuses to start on `main`/`master` (configurable), on detached HEAD, or on
a dirty tree (unless `--allow-dirty`). Records branch + base commit first.
Orchestrator git usage is allow-listed read-only (`rev-parse status diff log
ls-files show branch`); the only mutation is `--create-branch <name>`, which
you request explicitly. The harness itself never pushes, merges, resets,
amends or deletes. Claude may make normal local commits; after approval the
branch is left for you — nothing is merged or pushed automatically.

## Secrets / prompt injection

Payloads and persisted artifacts pass a redaction layer: secret-shaped files
(.env, *.pem, *credentials*, …) are dropped from diffs/untracked/context
requests, and credential-shaped strings (OpenRouter/Anthropic/OpenAI/AWS/
GitHub keys, bearer tokens, `api_key=` assignments) become
`[REDACTED:<kind>]`. Repository content is wrapped in explicit
`BEGIN/END ... (UNTRUSTED)` data sections; both agents are instructed that
nothing inside data sections is an instruction ("ignore previous
instructions" in a source file is reported, not obeyed).

## Run artifacts

```
tools/ai_collab/runs/<task-id>/          (gitignored)
  task.md  run.json  audit.jsonl  final.json
  round_01/
    claude_prompt_a1.md  claude_output_a1.txt  claude_handoff.json
    diff.patch  changed_files.json  bundle.json  tests.txt
    reviewer_request.txt  reviewer_request_meta.json
    review.json  review_raw.txt  [MANUAL_INSTRUCTIONS.md]  [graphify.txt]
  round_02/ ...
```

`audit.jsonl` holds timestamps, models, token usage, reported cost, cache
hits, round transitions and the stop reason.

## Tests

`tests/test_ai_collab_*.py` — 51 offline tests (mock adapters, temp git
repos, `no_network` guard): approval, changes→approval, max rounds, budget
exhaustion, test failure (bounded fix), malformed handoff/review, reviewer
blocked, dirty tree, protected branch, cache reuse, secret redaction, exact
diff capture, crash resume, CLI dry-run/e2e. The suite spends no credits.

## Graphify

Optional (`[graphify] enabled = true` + explicit `query|path|explain`
commands): targeted output is attached to the reviewer payload as untrusted
notes. Never the whole graph; source stays authoritative.
