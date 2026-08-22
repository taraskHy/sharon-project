"""Shared ground-truth grading (labeling) web app — human annotation only.

Runs on the owner's PC (local SQLite, local frozen bundle), is exposed
temporarily through a Cloudflare Tunnel, and is used by friends through an
ordinary browser. It never imports the grading pipeline at runtime and never
calls any model or provider of any kind (zero AI/OCR calls by construction —
see tests/test_labeling_app.py).

    python -m labeling_app build-bundle      # anonymized frozen bundle from the grade_primary dataset
    python -m labeling_app serve             # http://127.0.0.1:8787  (grader page /, admin page /admin)
    python -m labeling_app export            # final_labels.json (FINAL labels only)
    python -m labeling_app backup            # snapshot labels.db + export into a timestamped folder

Modules: db (SQLite schema + optimistic concurrency), bundle (anonymized
item bundle), app (Starlette ASGI app + pages), export, backup, cli.

Schema 2 (2026-08-22): evidence fingerprints — every bundle item carries
``evidence_sha256`` (exactly the crops shown, in order); every label/FINAL
stores the fingerprint it was made against; a rebuilt bundle that changes an
item's evidence makes those labels visibly STALE (re-review required) instead
of silently reusing them. Labels on unchanged items are untouched.
"""
SCHEMA_VERSION = 2
__all__ = ["SCHEMA_VERSION"]
