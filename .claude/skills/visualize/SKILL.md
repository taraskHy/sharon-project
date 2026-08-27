---
name: visualize
description: Build or refresh the interactive architecture flow map of this codebase — pipeline stages as lanes, components as cards, labeled runtime flows (calls, subprocess hops, dependency injection, data stores, LLM/network calls, human-in-the-loop), each backed by file:line evidence, with drill-down to modules. Use this whenever the user wants to see or understand the system's structure or flow — "visualize the flow/model/architecture", "how does it all fit together", "show me the pipeline", "map the system", "what calls what", "update the flow map" — even if they don't say "diagram". Also use it after large refactors to check the map still tells the truth. Not for questions about a single function or file; answer those directly.
---

# Visualize — architecture flow map

Produce `flow-out/flowmap.html`: an interactive system-architecture diagram of
this repo, driven by a curated spec at `docs/flow-spec.json`.

Why this design: a file-import graph is not architecture (the user has one from
graphify and rejected it). The map's value is *semantic* — the runtime story:
entry points → pipeline stages → model backends, including exactly the hops
static analysis can't see (DI seams, subprocess chains, config gating, data
stores, humans). Your reading of the source is the instrument; the scripts
only gather evidence and render.

## Workflow

1. **Scan** (mechanical evidence + drift alarm):

   ```
   python .claude/skills/visualize/scripts/scan_modules.py autograder --out flow-out/scan.json
   ```

   Add more package dirs as positional args if the map should cover them
   (e.g. `labeling_app`). Stdlib-only, no installs.

2. **Author or update `docs/flow-spec.json`** — read
   `references/flow-spec.md` (sibling to this file) for the schema and rules.

   - **First time**: find entry points (scan prints hints), then trace the
     main pipeline in source, end to end. Use scan.json's import/call edges as
     leads and a completeness checklist — never as the diagram. Where
     available, `graphify query/path/explain` helps find surfaces fast.
   - **Updating**: run the renderer first; its warnings list stale modules and
     new, unplaced ones. Read what changed (git log helps), adjust components
     and flows, keep the rest.
   - Chase the graph-invisible flows deliberately: injected callables
     (`gateway=`-style params, registered resolvers/callbacks), subprocess
     command construction, runtime flags choosing between paths, reads/writes
     of persistent stores, external network calls, human review steps.
   - Every flow carries `evidence: [{src: "path:line", note}]` you verified by
     reading that line. If you can't point at code, it doesn't go on the map.

3. **Render**:

   ```
   python .claude/skills/visualize/scripts/render_flowmap.py
   ```

   Fix errors; relay meaningful warnings to the user (drift, uncovered
   modules). Output: `flow-out/flowmap.html` (artifact-ready fragment) and
   `flow-out/flowmap.local.html` (standalone, double-clickable).

4. **Deliver**: publish `flow-out/flowmap.html` as an Artifact titled
   "Autograder Flow Map" — check the artifact list for an existing one with
   that title and pass its `url` so the link stays stable across refreshes.
   Also open it (browser pane) to verify it renders: cards in lanes, edges
   labeled, click-through drawer works. Mention `flowmap.local.html` for
   offline viewing.

## Focused maps

For a deep-dive of one subsystem ("visualize just the model access layer"),
write a variant spec to `docs/flow-spec-<topic>.json` — same schema, fewer
components, more detail (components may then be individual modules) — and
render with:

```
python .claude/skills/visualize/scripts/render_flowmap.py --spec docs/flow-spec-<topic>.json --out-dir flow-out/<topic>
```

Publish focused maps as their own artifacts, named for the subsystem.

## Ground rules

- `docs/flow-spec.json` is committed; `flow-out/` is generated and gitignored.
- The spec evolves incrementally — edit, don't regenerate, so review diffs
  stay meaningful.
- Verify before drawing: the scan and graphify are maps, source is territory
  (CLAUDE.md rule). Evidence lines are the contract with the reader.
- Scripts and template are project-agnostic; only the spec knows this system.
