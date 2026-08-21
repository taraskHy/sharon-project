"""Claude independence (Part 0B): the production autograder must install,
run, and test with no Claude Code, no Anthropic tooling, and no ai_collab.

Anthropic remains available ONLY as an explicitly configured optional model
provider (pip install -e .[anthropic] + backend="anthropic" in config).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = sorted((REPO_ROOT / "autograder").rglob("*.py"))


def test_production_code_never_imports_dev_tooling():
    for path in PRODUCTION:
        text = path.read_text(encoding="utf-8")
        assert "ai_collab" not in text, path
        assert "from tools" not in text and "import tools" not in text, path


def test_production_code_never_invokes_a_claude_binary():
    """No production module may shell out to (or search for) a `claude`
    executable. The one sanctioned mention of Claude is the optional
    Anthropic backend's default model id."""
    for path in PRODUCTION:
        if path.name == "anthropic_backend.py":
            continue  # optional provider module: model id mentions allowed
        text = path.read_text(encoding="utf-8").lower()
        assert "claude" not in text, f"unexpected claude reference in {path}"


def test_autograder_imports_without_anthropic_or_claude_tooling():
    """A default import of the CLI + reviewui backend must touch neither the
    anthropic SDK nor any `tools` module, with all Anthropic/Claude env
    stripped."""
    code = (
        "import os, sys\n"
        "for k in list(os.environ):\n"
        "    if k.startswith(('ANTHROPIC', 'CLAUDE')):\n"
        "        del os.environ[k]\n"
        "import autograder.cli, autograder.reviewui, autograder.gateway\n"
        "bad = [m for m in sys.modules\n"
        "       if m == 'anthropic' or m.startswith('anthropic.')\n"
        "       or m == 'tools' or m.startswith('tools.')]\n"
        "assert not bad, bad\n"
        "print('CLEAN')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, cwd=str(REPO_ROOT), timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert "CLEAN" in proc.stdout


def test_no_claude_specific_output_parsing_in_backends():
    """Structured-output parsing must be provider-generic: only the optional
    anthropic_backend module may import the anthropic SDK, and no other
    backend mentions Claude response shapes."""
    for path in sorted((REPO_ROOT / "autograder").rglob("*.py")):
        if path.name == "anthropic_backend.py":
            continue
        text = path.read_text(encoding="utf-8")
        assert "import anthropic" not in text and "from anthropic" not in text, path
    for name in ("openai_compat.py", "ollama_native.py", "openrouter.py",
                 "base.py", "mock.py"):
        text = (REPO_ROOT / "autograder" / "backends" / name).read_text(
            encoding="utf-8").lower()
        assert "claude" not in text, name
    # the factory selects anthropic only behind the explicit config branch
    init_text = (REPO_ROOT / "autograder" / "backends" / "__init__.py").read_text(
        encoding="utf-8")
    assert 'config.backend == "anthropic"' in init_text
