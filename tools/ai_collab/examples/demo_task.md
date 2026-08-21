# Demo task (synthetic — used for dry-run rehearsal only)

Add a module docstring example to `tools/ai_collab/util.py::slugify`
documenting its contract (lowercase, non-alphanumerics collapsed to `-`,
never empty). Behavior must not change; no other files may be touched.

Acceptance:
- `slugify("Hello World!")` still returns `hello-world`.
- Existing tests keep passing.
