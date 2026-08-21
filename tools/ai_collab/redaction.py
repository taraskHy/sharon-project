"""Secret filtering for everything that leaves the machine or is persisted.

Two layers:

1. Path deny-list: files that look like credential stores are dropped entirely
   from diffs / context (``is_secret_path`` + ``filter_diff``).
2. Text scrubbing: common credential shapes are replaced with
   ``[REDACTED:<kind>]`` markers (``redact_text``).

Applied to reviewer payloads AND to persisted run artifacts (diff.patch,
reviewer_request.txt), so a secret accidentally present in the repo is neither
sent to a provider nor copied into the run directory.
"""

from __future__ import annotations

import fnmatch
import re
from pathlib import PurePosixPath

SECRET_PATH_GLOBS = [
    ".env",
    ".env.*",
    "*.env",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "*.jks",
    "*.keystore",
    "id_rsa*",
    "id_dsa*",
    "id_ecdsa*",
    "id_ed25519*",
    "*.tfstate",
    "*.tfstate.*",
    "credentials*",
    "*credentials*",
    "secret*",
    "*secret*",
    "*.htpasswd",
    "netrc",
    ".netrc",
]

# Ordered: more specific key shapes first, generic assignment last.
SECRET_TEXT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("openrouter_key", re.compile(r"sk-or-[A-Za-z0-9_\-]{16,}")),
    ("anthropic_key", re.compile(r"sk-ant-[A-Za-z0-9_\-]{16,}")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{30,}\b")),
    ("openai_key", re.compile(r"\bsk-[A-Za-z0-9_\-]{18,}\b")),
    ("slack_token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    ("bearer_token", re.compile(r"(?i)\bbearer\s+[A-Za-z0-9_\-\.=]{16,}")),
    (
        "assignment",
        re.compile(
            r"(?i)\b((?:api[_-]?key|apikey|secret|token|passwd|password|authorization)"
            r"\s*[:=]\s*)(['\"]?)((?!\[REDACTED:)[^\s'\"]{8,})\2"
        ),
    ),
]


def is_secret_path(relpath: str, extra_globs: list[str] | None = None) -> bool:
    """True when *relpath* (repo-relative, any separator) matches the deny-list."""
    norm = relpath.replace("\\", "/").lower()
    while norm.startswith("./"):
        norm = norm[2:]
    norm = norm.lstrip("/")
    name = PurePosixPath(norm).name
    for pattern in SECRET_PATH_GLOBS + list(extra_globs or []):
        pat = pattern.lower()
        if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(norm, pat):
            return True
    return False


def redact_text(
    text: str, extra_patterns: list[str] | None = None
) -> tuple[str, dict[str, int]]:
    """Scrub credential-shaped substrings. Returns (redacted_text, counts)."""
    counts: dict[str, int] = {}
    if not text:
        return text, counts

    patterns = list(SECRET_TEXT_PATTERNS)
    for i, raw in enumerate(extra_patterns or []):
        try:
            patterns.append((f"extra_{i}", re.compile(raw)))
        except re.error:
            continue  # a broken user pattern must not break redaction

    for kind, pattern in patterns:
        if kind == "assignment":

            def _sub_assign(match: re.Match) -> str:
                counts[kind] = counts.get(kind, 0) + 1
                return f"{match.group(1)}[REDACTED:{kind}]"

            text = pattern.sub(_sub_assign, text)
            continue

        def _sub(match: re.Match, _kind=kind) -> str:
            counts[_kind] = counts.get(_kind, 0) + 1
            return f"[REDACTED:{_kind}]"

        text = pattern.sub(_sub, text)
    return text, counts


_DIFF_HEADER_RE = re.compile(r"^diff --git a/(.*) b/(.*)$")


def filter_diff(
    diff_text: str, extra_globs: list[str] | None = None
) -> tuple[str, list[str]]:
    """Drop whole per-file sections of a unified diff whose path is secret-like.

    Returns (filtered_diff, excluded_paths). Deterministic: sections keep the
    order git produced; an excluded section is replaced by a one-line marker.
    """
    if not diff_text:
        return diff_text, []
    lines = diff_text.splitlines(keepends=True)
    out: list[str] = []
    excluded: list[str] = []
    skipping = False
    for line in lines:
        match = _DIFF_HEADER_RE.match(line.rstrip("\n"))
        if match is not None:
            path = match.group(2)
            if is_secret_path(path, extra_globs):
                skipping = True
                excluded.append(path)
                out.append(
                    f"[EXCLUDED by orchestrator: {path} matches the secret-file deny list]\n"
                )
                continue
            skipping = False
        if not skipping:
            out.append(line)
    return "".join(out), excluded
