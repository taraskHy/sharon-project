"""Reviewer payload assembly: redaction, deterministic chunking, budgets.

Order of sections is fixed. Trimming is deterministic (per-file caps in git
order, then whole-section caps, then a global cap applied to the largest
optional sections first), so identical inputs always produce byte-identical
payloads — which is what makes the request cache sound.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import json

from .config import CollabConfig
from .git_ops import ChangeSet
from .prompts import (
    data_section,
    reviewer_followup_instructions,
    reviewer_response_instructions,
)
from .redaction import filter_diff, is_secret_path, redact_text
from .util import est_tokens, sha256_text, truncate_middle, truncate_tail


@dataclass
class DiffBundle:
    """Redacted, size-capped view of a ChangeSet (also persisted as artifacts)."""

    diff_text: str = ""
    untracked_text: str = ""
    changed_files: dict = field(default_factory=dict)
    excluded_paths: list[str] = field(default_factory=list)
    truncations: list[str] = field(default_factory=list)
    redaction_counts: dict = field(default_factory=dict)


def _split_diff_sections(diff_text: str) -> list[str]:
    """Split a unified diff into per-file sections, preserving git's order."""
    if not diff_text:
        return []
    sections: list[str] = []
    current: list[str] = []
    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git ") and current:
            sections.append("".join(current))
            current = []
        current.append(line)
    if current:
        sections.append("".join(current))
    return sections


def prepare_diff_bundle(change: ChangeSet, cfg: CollabConfig) -> DiffBundle:
    bundle = DiffBundle()
    extra_globs = cfg.redaction.extra_deny_globs
    extra_patterns = cfg.redaction.extra_patterns

    filtered, excluded = filter_diff(change.diff_text, extra_globs)
    bundle.excluded_paths.extend(excluded)

    # Per-file caps in git order, then the whole-diff cap.
    sections = _split_diff_sections(filtered)
    capped_sections: list[str] = []
    for section in sections:
        capped, truncated = truncate_middle(
            section, cfg.payload.max_file_chars, "(per-file cap) "
        )
        if truncated:
            first_line = section.splitlines()[0] if section.splitlines() else "?"
            bundle.truncations.append(f"per-file cap applied: {first_line.strip()}")
        capped_sections.append(capped)

    assembled = "".join(capped_sections)
    kept: list[str] = []
    used = 0
    omitted_files: list[str] = []
    for section in capped_sections:
        if used + len(section) <= cfg.payload.max_diff_chars:
            kept.append(section)
            used += len(section)
        else:
            first_line = section.splitlines()[0] if section.splitlines() else "?"
            omitted_files.append(first_line.strip())
    if omitted_files:
        bundle.truncations.append(
            f"diff cap: {len(omitted_files)} file section(s) omitted"
        )
        kept.append(
            "\n[OMITTED by orchestrator (diff size cap): "
            + "; ".join(omitted_files[:50])
            + "]\n"
        )
    diff_text = "".join(kept) if omitted_files else assembled

    diff_text, counts = redact_text(diff_text, extra_patterns)
    for kind, n in counts.items():
        bundle.redaction_counts[kind] = bundle.redaction_counts.get(kind, 0) + n
    bundle.diff_text = diff_text

    # Untracked files: deny-list, per-file cap, half-of-diff total cap.
    untracked_parts: list[str] = []
    total_untracked_budget = max(1, cfg.payload.max_diff_chars // 2)
    used = 0
    for relpath in change.untracked:
        if is_secret_path(relpath, extra_globs):
            bundle.excluded_paths.append(relpath)
            untracked_parts.append(
                f"--- untracked: {relpath} [EXCLUDED: secret-file deny list]\n"
            )
            continue
        content = change.untracked_contents.get(relpath, "")
        content, _ = truncate_middle(
            content, cfg.payload.max_untracked_file_chars, "(untracked cap) "
        )
        content, counts = redact_text(content, extra_patterns)
        for kind, n in counts.items():
            bundle.redaction_counts[kind] = bundle.redaction_counts.get(kind, 0) + n
        entry = f"--- untracked (new, not yet committed): {relpath}\n{content}\n"
        if used + len(entry) > total_untracked_budget:
            bundle.truncations.append(f"untracked file omitted (size cap): {relpath}")
            untracked_parts.append(
                f"--- untracked: {relpath} [OMITTED: untracked size cap]\n"
            )
            continue
        untracked_parts.append(entry)
        used += len(entry)
    bundle.untracked_text = "".join(untracked_parts)

    bundle.changed_files = {
        "base": change.base,
        "head": change.head,
        "branch": change.branch,
        "name_status": [[status, path] for status, path in change.name_status],
        "untracked": list(change.untracked),
        "excluded_secret_paths": sorted(set(bundle.excluded_paths)),
    }
    return bundle


@dataclass
class ReviewerPayload:
    system: str
    user: str
    est_input_tokens: int = 0
    hashes: dict = field(default_factory=dict)
    truncations: list[str] = field(default_factory=list)
    section_sizes: dict = field(default_factory=dict)
    redaction_counts: dict = field(default_factory=dict)


# Global-cap shrink order: trim the biggest expendable sections first.
_SHRINK_ORDER = [
    "GRAPHIFY ARCHITECTURE NOTES (UNTRUSTED)",
    "TEST OUTPUT (UNTRUSTED)",
    "UNTRACKED FILES (UNTRUSTED)",
    "REPOSITORY DIFF (UNTRUSTED)",
    "REQUESTED SOURCE FILES (UNTRUSTED)",
    "CLAUDE HANDOFF (UNTRUSTED)",
    "PROJECT REVIEWER CONTEXT",
]


def build_reviewer_payload(
    cfg: CollabConfig,
    system_prompt: str,
    task_text: str,
    context_text: str,
    handoff_obj: dict | None,
    bundle: DiffBundle,
    test_output: str,
    graphify_text: str = "",
    requested_files: dict[str, str] | None = None,
    followup: bool = False,
) -> ReviewerPayload:
    extra_patterns = cfg.redaction.extra_patterns
    truncations: list[str] = list(bundle.truncations)
    redactions: dict = dict(bundle.redaction_counts)

    def _prep(
        text: str, cap: int, tail: bool = False, redact: bool = True
    ) -> str:
        if redact:
            text, counts = redact_text(text, extra_patterns)
            for kind, n in counts.items():
                redactions[kind] = redactions.get(kind, 0) + n
        text, truncated = (
            truncate_tail(text, cap) if tail else truncate_middle(text, cap)
        )
        return text

    handoff_text = (
        json.dumps(handoff_obj, indent=2, ensure_ascii=False)
        if handoff_obj is not None
        else "(no handoff)"
    )
    changed_files_text = json.dumps(bundle.changed_files, indent=2, ensure_ascii=False)

    sections: list[tuple[str, str]] = [
        ("ORIGINAL TASK", _prep(task_text, cfg.payload.max_context_chars)),
        (
            "PROJECT REVIEWER CONTEXT",
            _prep(context_text, cfg.payload.max_context_chars),
        ),
        (
            "CLAUDE HANDOFF (UNTRUSTED)",
            _prep(handoff_text, cfg.payload.max_handoff_chars),
        ),
        ("CHANGED FILES (orchestrator-captured)", changed_files_text),
        ("REPOSITORY DIFF (UNTRUSTED)", bundle.diff_text or "(no changes)"),
    ]
    if bundle.untracked_text:
        sections.append(("UNTRACKED FILES (UNTRUSTED)", bundle.untracked_text))
    sections.append(
        (
            "TEST OUTPUT (UNTRUSTED)",
            _prep(test_output, cfg.payload.max_test_output_chars, tail=True)
            if test_output
            else "(no orchestrator-run test output)",
        )
    )
    if graphify_text:
        sections.append(
            (
                "GRAPHIFY ARCHITECTURE NOTES (UNTRUSTED)",
                _prep(graphify_text, cfg.payload.max_context_chars),
            )
        )
    if requested_files:
        parts = []
        for relpath in sorted(requested_files):
            body = _prep(requested_files[relpath], cfg.payload.max_file_chars)
            parts.append(f"----- FILE: {relpath} -----\n{body}\n")
        sections.append(("REQUESTED SOURCE FILES (UNTRUSTED)", "".join(parts)))

    instructions = (
        reviewer_followup_instructions() if followup else reviewer_response_instructions()
    )

    def _assemble(section_list: list[tuple[str, str]]) -> str:
        rendered = [data_section(name, body) for name, body in section_list]
        rendered.append(data_section("RESPONSE INSTRUCTIONS", instructions))
        return "\n".join(rendered)

    user = _assemble(sections)

    # Global cap: shrink expendable sections in fixed order until it fits.
    if len(user) > cfg.payload.max_total_chars:
        sections_by_name = dict(sections)
        overshoot = len(user) - cfg.payload.max_total_chars
        for name in _SHRINK_ORDER:
            if overshoot <= 0:
                break
            body = sections_by_name.get(name)
            if not body:
                continue
            target = max(500, len(body) - overshoot)
            new_body, truncated = truncate_middle(body, target, "(global cap) ")
            if truncated:
                truncations.append(f"global cap trimmed section: {name}")
                overshoot -= len(body) - len(new_body)
                sections_by_name[name] = new_body
        sections = [(name, sections_by_name[name]) for name, _ in sections]
        user = _assemble(sections)

    payload = ReviewerPayload(
        system=system_prompt,
        user=user,
        est_input_tokens=est_tokens(system_prompt) + est_tokens(user),
        truncations=truncations,
        redaction_counts=redactions,
        section_sizes={name: len(body) for name, body in sections},
    )
    payload.hashes = {
        "task_sha256": sha256_text(task_text),
        "context_sha256": sha256_text(context_text),
        "diff_sha256": sha256_text(bundle.diff_text + "\n" + bundle.untracked_text),
        "tests_sha256": sha256_text(test_output),
        "user_sha256": sha256_text(user),
        "system_sha256": sha256_text(system_prompt),
    }
    return payload


def load_requested_files(
    repo: Path, raw_paths: list[str], cfg: CollabConfig
) -> tuple[dict[str, str], list[str]]:
    """Sanitize reviewer context_requests and load file contents.

    Rules: repo-relative, must exist inside the repo, not secret-like, at most
    ``max_context_files`` files. Returns (files, notes-about-rejects).
    """
    files: dict[str, str] = {}
    notes: list[str] = []
    repo_resolved = repo.resolve()
    for raw in raw_paths[: cfg.reviewer.max_context_files]:
        rel = str(raw).strip().replace("\\", "/")
        if not rel or rel.startswith(("/", "~")) or ":" in rel.split("/")[0]:
            notes.append(f"rejected context request (not repo-relative): {raw!r}")
            continue
        candidate = (repo / rel).resolve()
        if not str(candidate).startswith(str(repo_resolved)):
            notes.append(f"rejected context request (escapes repo): {raw!r}")
            continue
        if is_secret_path(rel, cfg.redaction.extra_deny_globs):
            notes.append(f"rejected context request (secret deny list): {rel}")
            continue
        if not candidate.is_file():
            notes.append(f"rejected context request (not a file): {rel}")
            continue
        try:
            data = candidate.read_bytes()
        except OSError as exc:
            notes.append(f"rejected context request (unreadable): {rel} ({exc})")
            continue
        if b"\x00" in data[:8192]:
            notes.append(f"rejected context request (binary): {rel}")
            continue
        files[rel] = data[: cfg.payload.max_file_chars * 4].decode(
            "utf-8", errors="replace"
        )
    if len(raw_paths) > cfg.reviewer.max_context_files:
        notes.append(
            f"context requests capped at {cfg.reviewer.max_context_files} files"
        )
    return files, notes
