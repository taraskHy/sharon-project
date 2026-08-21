"""Exception types for the collaboration harness."""


class CollabError(Exception):
    """Base class for all harness errors."""


class ConfigError(CollabError):
    """Invalid or unusable configuration / CLI arguments."""


class GitError(CollabError):
    """A git command failed or returned something unusable."""


class GitSafetyError(GitError):
    """A git precondition (clean tree, non-protected branch, ...) failed."""


class SchemaError(CollabError):
    """A structured document (handoff / review) failed validation."""


class AdapterError(CollabError):
    """A Claude or reviewer adapter could not complete a call."""
