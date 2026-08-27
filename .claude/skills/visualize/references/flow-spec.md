# flow-spec.json — the curated architecture spec

`docs/flow-spec.json` is the source of truth for the flow map. It is a
**semantic model of the system's runtime architecture**, written and updated by
Claude after reading the source — NOT a dump of import edges. The whole reason
this skill exists is that file-connection graphs don't show architecture: they
miss dependency injection, subprocess boundaries, runtime config gating, and
data stores, and they show 79 files where the reader needs ~20 concepts.

The spec is committed to git so the architecture picture evolves incrementally
and gets reviewed like code. Update it; don't rewrite it from scratch.

## Schema

```json
{
  "title": "Autograder Flow",
  "subtitle": "one line about what the system does",
  "groups": [
    { "id": "entry", "name": "Entry points", "summary": "how work starts" }
  ],
  "components": [
    {
      "id": "gateway",
      "name": "Model gateway",
      "kind": "service",
      "group": "models",
      "summary": "2-3 sentences: responsibility, and anything surprising.",
      "modules": ["autograder.gateway", "autograder.requestcache"]
    }
  ],
  "flows": [
    {
      "from": "webui", "to": "jobs",
      "kind": "subprocess",
      "label": "run-job",
      "summary": "What travels and when. Present tense, one or two sentences.",
      "evidence": [
        { "src": "autograder/webui.py:214", "note": "builds the run-job argv" }
      ]
    }
  ],
  "notes": ["caveats, deliberate omissions, TODOs for the next update"]
}
```

## Field rules

- **groups** are the pipeline stages / lanes, rendered left→right in array
  order. Order them as execution flows (entry → … → output). 4–7 lanes.
- **components** are logical subsystems, not files. 12–25 total; if you have
  more, you're drawing the file graph again — merge helpers into the component
  that owns them. Every component lists the concrete `modules` it owns
  (dotted names as scan.json spells them) — that powers drill-down and lets
  the renderer flag drift.
- **component kinds**: `ui` (a human or system starts work here), `stage`
  (a step in the main pipeline), `service` (support logic used across stages),
  `store` (persistent data: caches, DBs, output dirs), `external` (outside
  process: LLM APIs, other machines), `human` (a person in the loop).
- **flow kinds**: `call` (in-process call), `data` (reads/writes a store),
  `subprocess` (crosses a process boundary), `di` (dependency-injected /
  callback seam — the graph-invisible kind), `http` (network to an external
  service), `human` (a person acts).
- **evidence** is mandatory in spirit: every flow should carry at least one
  `file:line` a reader can open to see the hop happen (the call site, the
  argv construction, the injection point). The renderer warns when it's
  missing. Unverified flows don't belong on the map.
- **direction** = direction of control or data movement, whichever the reader
  needs to follow the story. Feedback edges (right→left) are fine and render
  as arcs over the lanes.

## Keeping it honest

- Renderer warnings are the drift alarm: unknown modules mean the spec went
  stale; uncovered modules mean new code hasn't been placed on the map yet.
- When behavior is gated by runtime config (a flag, a models.toml entry),
  say so in the flow's `summary` and point evidence at the gate.
- Aim for the diagram a good staff engineer would whiteboard: few boxes,
  labeled arrows, no lies.
