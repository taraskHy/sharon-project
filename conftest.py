"""Repo-root pytest configuration — the outermost live-data barrier.

tests/conftest.py is only loaded when pytest collects under tests/. Someone
running `pytest labeling_app/`, `pytest --doctest-modules`, or a single file
elsewhere would otherwise get no sandbox at all, and any code calling
`default_data_dir()` would point straight at the owner's deployment.

Worse, fixtures do not run during COLLECTION: a module-level
`LabelDB(default_data_dir() / "labels.db")` executes at import time, before any
autouse fixture. So the redirect is done here in `pytest_configure`, which runs
before collection, rather than in a fixture.

This complements — it does not replace — `labeling_app.db.assert_not_live_database`,
which refuses the deployment path inside LabelDB itself.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

#: env vars that would otherwise resolve to the owner's real %LOCALAPPDATA% data
_REDIRECTED = ("LABELING_DATA_DIR", "GRADER_KEY_CACHE")


def pytest_configure(config):
    """Redirect every machine-local data directory BEFORE collection imports run."""
    root = Path(tempfile.mkdtemp(prefix="pytest-autograder-"))
    for var in _REDIRECTED:
        if not os.environ.get(var):
            target = root / var.lower()
            target.mkdir(parents=True, exist_ok=True)
            os.environ[var] = str(target)
    config._autograder_sandbox = root
