# Graphify — local code-graph navigation

Graphify maintains a static AST-derived knowledge graph of this repo for fast
architecture navigation. It is a **navigation aid only** — source code and
tests remain authoritative. The graph cannot see dependency injection, runtime
config, dynamically registered backends, or callbacks, and it indexes tests
alongside production code. Always verify graph findings in the real source.

## Install

Isolated from the project's runtime environment (PyPI package is `graphifyy`,
double-y; the CLI is `graphify`):

```
uv tool install graphifyy
```

On this machine it is installed in a dedicated venv at
`C:\Users\ethan\.local\graphify-venv` with launchers in `C:\Users\ethan\.local\bin`
(version 0.9.48). No autograder dependency was added.

## Initial index

From the repo root (code-only static AST extraction — no LLM, no API key,
no embeddings, no OCR):

```
graphify update .
```

Moderate parallelism is the default; cap it with `GRAPHIFY_MAX_WORKERS=6`.
The initial build indexed 127 code files (~125K words) in ~15 s:
2,008 nodes, 5,822 edges, 94 labeled communities.

## Refresh

After meaningful structural changes (incremental — re-extracts only
new/changed files, no LLM needed):

```
graphify update .
```

If communities look stale after large refactors, re-cluster without LLM
naming: `graphify cluster-only . --no-label`. If a refactor deleted many
files and the update is refused by the shrink guard, add `--force`.
Refresh is **manual only** — no watch mode, no git hooks — to avoid
background CPU and OneDrive churn.

## Query

```
graphify query "path from CLI to explanation grading" --budget 2500
graphify query "who consumes QuestionGradingPack"
graphify path "run_grade_pipeline()" "escalate_grade()"
graphify explain "ModelGateway"
```

Output is a BFS node/edge dump with `src=file:line` per node — use it to find
the right files fast, then read them.

Tips from validation: `explain` on a specific symbol is sharper than a broad
`query`; raise `--budget` when output truncates; test files dominate BFS
output, so separate `tests/` from `autograder/` by the `src=` path. Known
blind spots (verified): injected callables/DI (`gateway=` params, `rag_attach`,
`set_mc_resolver`), subprocess chaining (webui → run-job → grade), runtime
flag gating (`--grading-mode`), and aliased local imports — trace those in
source, not the graph.

## Ignore behavior

`.gitignore` is honored automatically (venvs, `out/`, `eval_out*/`, caches).
`.graphifyignore` additionally excludes:

- data corpora: `evaluation/`, `held-out test set/`, `prob_data/`,
  `sample_data/`, `datasets/`, `%temp%/`
- non-code types: `*.json` (graphify would treat JSON as code; here it is
  exam/eval data), `*.md`/`*.txt` (docs need LLM semantic extraction — out of
  scope for the code-only graph), images/PDF/CSV/logs
- `graphify-out/` itself

The indexed surface is the real architecture: `autograder/` (all 57 modules),
`tests/`, `test/`, `scripts/`, `android/`.

## Generated state

Everything generated lives in `graphify-out/` at the repo root (~9 MB:
`graph.json` ~3 MB, `graph.html` ~2.3 MB, `GRAPH_REPORT.md`, manifest, AST
cache). It is **gitignored** — do not commit it. It changes only on manual
refresh, so OneDrive uploads a few MB per rebuild, nothing per source edit.
If that ever becomes a problem, the `GRAPHIFY_OUT` env var accepts an
absolute path to relocate the state outside OneDrive entirely.

## Workflow for architecture tasks

1. `graphify query` to find likely integration surfaces
2. read the actual source files it points at
3. verify the call path manually (the graph may be stale or incomplete)
4. make changes
5. run tests
6. `graphify update .` after meaningful structural changes
