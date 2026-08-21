"""Local review request cache (spec section 15).

Fingerprint = sha256 over a canonical JSON of at least: reviewer model,
prompt version, task hash, diff hash, context hash, test-output hash and the
generation config. Identical logical requests reuse the stored review and
consume zero budget. Only successfully validated reviews are stored, so a
malformed reply is never replayed.
"""

from __future__ import annotations

from pathlib import Path

from .util import now_iso, read_json, sha256_json, write_json_atomic


def ensure_ignored_dir(directory: Path) -> None:
    """Create *directory* with a self-ignoring .gitignore (defense in depth).

    The pattern is a bare ``*`` so the directory's entire content — including
    the .gitignore itself — stays invisible to git status/diff/untracked
    capture even in repositories whose root .gitignore lacks an entry.
    """
    directory.mkdir(parents=True, exist_ok=True)
    gitignore = directory / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text("*\n", encoding="utf-8")


class ReviewCache:
    def __init__(self, directory: Path):
        self.directory = Path(directory)

    @staticmethod
    def fingerprint(parts: dict) -> str:
        return sha256_json(parts)

    def _path(self, fp: str) -> Path:
        return self.directory / f"{fp}.json"

    def get(self, fp: str) -> dict | None:
        path = self._path(fp)
        if not path.is_file():
            return None
        try:
            entry = read_json(path)
        except (OSError, ValueError):
            return None
        if not isinstance(entry, dict) or "review" not in entry:
            return None
        return entry

    def put(self, fp: str, review: dict, meta: dict) -> None:
        ensure_ignored_dir(self.directory)
        write_json_atomic(
            self._path(fp),
            {"review": review, "meta": dict(meta, cached_at=now_iso())},
        )
