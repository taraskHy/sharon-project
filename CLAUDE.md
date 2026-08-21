# Claude project notes

## Graphify code graph (navigation aid)

- `graphify-out/graph.json` is a static AST code graph of this repo
  (gitignored; rebuild with `graphify update .` — no LLM involved).
- For architecture questions, query the graph first:
  `graphify query "..."`, `graphify path "A" "B"`, `graphify explain "X"`.
- Then **always verify in real source** before concluding. The graph cannot
  see dependency injection, runtime configuration, dynamically registered
  backends, or callbacks, and it indexes tests alongside production code.
  Source code and tests are authoritative; the graph is only a map.
- Details and refresh workflow: `docs/graphify.md`.
