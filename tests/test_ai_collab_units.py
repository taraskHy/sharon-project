"""Unit tests for tools/ai_collab building blocks (offline, no adapters)."""

from pathlib import Path

import pytest

from _ai_collab_helpers import make_repo
from tools.ai_collab import redaction, schemas
from tools.ai_collab.budget import BudgetState, BudgetTracker
from tools.ai_collab.cache import ReviewCache
from tools.ai_collab.config import (
    BudgetCfg,
    CollabConfig,
    config_snapshot,
    load_config,
)
from tools.ai_collab.errors import ConfigError
from tools.ai_collab.git_ops import ChangeSet
from tools.ai_collab.payload import (
    build_reviewer_payload,
    load_requested_files,
    prepare_diff_bundle,
)
from tools.ai_collab.prompts import reviewer_system_prompt
from tools.ai_collab.util import slugify, truncate_middle, truncate_tail

REPO_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- config
def test_config_defaults_without_file(tmp_path):
    cfg, warnings = load_config(None, tmp_path)
    assert cfg.run.mode == "semi_auto"
    assert cfg.run.max_rounds == 3
    assert cfg.reviewer.backend == "openrouter"
    assert cfg.source_path == ""


def test_example_config_parses(monkeypatch):
    monkeypatch.delenv("AI_REVIEW_MODEL", raising=False)
    example = REPO_ROOT / "tools" / "ai_collab" / "config.example.toml"
    cfg, warnings = load_config(example, REPO_ROOT)
    assert cfg.run.mode == "semi_auto"
    assert cfg.run.max_rounds == 3
    assert cfg.claude.mode == "claude_code"
    assert any("AI_REVIEW_MODEL" in w for w in warnings)
    assert cfg.reviewer.model == "${AI_REVIEW_MODEL}"  # left unresolved


def test_config_env_expansion(monkeypatch, tmp_path):
    monkeypatch.setenv("X_REVIEW_MODEL", "vendor/some-model")
    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text('[reviewer]\nmodel = "${X_REVIEW_MODEL}"\n', encoding="utf-8")
    cfg, warnings = load_config(cfg_file, tmp_path)
    assert cfg.reviewer.model == "vendor/some-model"
    assert not warnings
    # the persisted snapshot keeps the ${VAR} form, never the expansion
    snap = config_snapshot(cfg)
    assert snap["reviewer"]["model"] == "${X_REVIEW_MODEL}"


def test_config_invalid_mode_rejected(tmp_path):
    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text('[run]\nmode = "yolo"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(cfg_file, tmp_path)


def test_config_unknown_key_is_warning_not_error(tmp_path):
    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text('[run]\nmoode = "manual"\n', encoding="utf-8")
    cfg, warnings = load_config(cfg_file, tmp_path)
    assert any("moode" in w for w in warnings)
    assert cfg.run.mode == "semi_auto"


# ------------------------------------------------------------------ redaction
def test_redact_text_scrubs_common_credentials():
    text = "\n".join(
        [
            'OPENROUTER = "sk-or-abcdefghij1234567890"',
            'ANTHROPIC = "sk-ant-abcdefghij1234567890"',
            "AWS AKIAABCDEFGHIJKLMNOP done",
            "gh token ghp_abcdefghijklmnopqrst123456",
            "Authorization: Bearer abcdef1234567890abcdef",
            "api_key = supersecretvalue123",
            "def add(a, b): return a + b",
        ]
    )
    redacted, counts = redaction.redact_text(text)
    assert "sk-or-" not in redacted
    assert "sk-ant-" not in redacted
    assert "AKIAABCDEFGHIJKLMNOP" not in redacted
    assert "ghp_" not in redacted
    assert "supersecretvalue123" not in redacted
    assert "def add(a, b): return a + b" in redacted
    assert counts.get("openrouter_key") == 1
    assert counts.get("anthropic_key") == 1
    assert counts.get("aws_access_key") == 1


def test_is_secret_path():
    for path in (".env", "config/.env.prod", "keys/id_rsa", "certs/server.pem",
                 "my_credentials.json", "app/secrets.toml"):
        assert redaction.is_secret_path(path), path
    for path in ("autograder/grade.py", "tests/test_grade.py", "README.md"):
        assert not redaction.is_secret_path(path), path


def test_filter_diff_drops_secret_file_sections():
    diff = (
        "diff --git a/app.py b/app.py\n--- a/app.py\n+++ b/app.py\n"
        "@@ -1 +1,2 @@\n def f(): pass\n+# ok\n"
        "diff --git a/.env b/.env\n--- a/.env\n+++ b/.env\n"
        "@@ -0,0 +1 @@\n+OPENROUTER_API_KEY=sk-or-abcdefghij1234567890\n"
    )
    filtered, excluded = redaction.filter_diff(diff)
    assert excluded == [".env"]
    assert "OPENROUTER_API_KEY" not in filtered
    assert "+# ok" in filtered
    assert "EXCLUDED by orchestrator" in filtered


# -------------------------------------------------------------------- schemas
def test_handoff_validation():
    good = {
        "task_id": "t",
        "round": 1,
        "status": "READY_FOR_REVIEW",
        "summary": "did it",
    }
    assert schemas.validate_handoff(good) == []
    bad = {"round": "one", "status": "DONE", "summary": ""}
    problems = schemas.validate_handoff(bad)
    assert len(problems) >= 3


def test_review_validation():
    good = {"verdict": "approved", "summary": "fine"}
    assert schemas.validate_review(good) == []
    missing_findings = {"verdict": "CHANGES_REQUIRED", "summary": "nope"}
    assert any("finding" in p for p in schemas.validate_review(missing_findings))
    bad_sev = {
        "verdict": "CHANGES_REQUIRED",
        "summary": "s",
        "findings": [{"id": "F1", "severity": "mega", "issue": "x",
                      "requested_change": "y"}],
    }
    assert any("severity" in p for p in schemas.validate_review(bad_sev))


def test_parse_json_lenient():
    assert schemas.parse_json_lenient('{"a": 1}') == {"a": 1}
    fenced = '```json\n{"a": {"b": "with { brace in string"}}\n```'
    assert schemas.parse_json_lenient(fenced) == {"a": {"b": "with { brace in string"}}
    wrapped = 'Sure! Here is the review:\n{"verdict": "APPROVED", "summary": "ok"}\nHope it helps.'
    assert schemas.parse_json_lenient(wrapped)["verdict"] == "APPROVED"
    assert schemas.parse_json_lenient("no json here") is None


# ---------------------------------------------------------------------- cache
def test_cache_roundtrip_and_fingerprint(tmp_path):
    cache = ReviewCache(tmp_path / "cache")
    fp1 = ReviewCache.fingerprint({"model": "m", "diff_sha256": "abc"})
    fp1_again = ReviewCache.fingerprint({"diff_sha256": "abc", "model": "m"})
    fp2 = ReviewCache.fingerprint({"model": "m", "diff_sha256": "abd"})
    assert fp1 == fp1_again
    assert fp1 != fp2
    assert cache.get(fp1) is None
    cache.put(fp1, {"verdict": "APPROVED"}, {"model": "m"})
    entry = cache.get(fp1)
    assert entry["review"]["verdict"] == "APPROVED"
    assert (tmp_path / "cache" / ".gitignore").read_text(encoding="utf-8") == "*\n"


# --------------------------------------------------------------------- payload
def _mini_cfg() -> CollabConfig:
    cfg = CollabConfig()
    cfg.payload.max_file_chars = 400
    cfg.payload.max_diff_chars = 700
    cfg.payload.max_untracked_file_chars = 200
    return cfg


def test_diff_bundle_caps_and_exclusions():
    cfg = _mini_cfg()
    big_body = "\n".join(f"+line {i} padding padding padding" for i in range(60))
    diff = (
        f"diff --git a/big1.py b/big1.py\n{big_body}\n"
        f"diff --git a/big2.py b/big2.py\n{big_body}\n"
        "diff --git a/.env b/.env\n+SECRET=sk-or-abcdefghij1234567890\n"
    )
    change = ChangeSet(
        base="b", head="h", branch="x",
        name_status=[("M", "big1.py"), ("M", "big2.py"), ("A", ".env")],
        diff_text=diff,
        untracked=["notes.txt", ".env.local"],
        untracked_contents={
            "notes.txt": "hello notes",
            ".env.local": "KEY=sk-or-abcdefghij1234567890",
        },
    )
    bundle = prepare_diff_bundle(change, cfg)
    assert ".env" in bundle.excluded_paths
    assert ".env.local" in bundle.excluded_paths
    assert "sk-or-" not in bundle.diff_text
    assert "sk-or-" not in bundle.untracked_text
    assert "TRUNCATED by orchestrator" in bundle.diff_text  # per-file cap
    assert any("omitted" in t for t in bundle.truncations)  # total diff cap
    assert "hello notes" in bundle.untracked_text
    assert bundle.changed_files["excluded_secret_paths"]


def test_reviewer_payload_sections_and_determinism():
    cfg = _mini_cfg()
    change = ChangeSet(
        base="b", head="h", branch="x",
        name_status=[("M", "app.py")],
        diff_text="diff --git a/app.py b/app.py\n+# improved\n",
    )
    bundle = prepare_diff_bundle(change, cfg)
    system = reviewer_system_prompt(["critical", "high"])

    def build():
        return build_reviewer_payload(
            cfg, system, "task text", "context text",
            {"task_id": "t", "round": 1, "status": "READY_FOR_REVIEW",
             "summary": "s"},
            bundle, "1 passed", "",
        )

    p1, p2 = build(), build()
    for marker in (
        "BEGIN ORIGINAL TASK", "BEGIN PROJECT REVIEWER CONTEXT",
        "BEGIN CLAUDE HANDOFF (UNTRUSTED)", "BEGIN REPOSITORY DIFF (UNTRUSTED)",
        "BEGIN TEST OUTPUT (UNTRUSTED)", "BEGIN RESPONSE INSTRUCTIONS",
    ):
        assert marker in p1.user
    assert "+# improved" in p1.user
    assert p1.est_input_tokens > 0
    assert p1.hashes == p2.hashes  # deterministic -> cache-safe


def test_reviewer_payload_global_cap():
    cfg = _mini_cfg()
    cfg.payload.max_total_chars = 2500
    bundle = prepare_diff_bundle(
        ChangeSet(base="b", head="h", branch="x", diff_text=""), cfg
    )
    system = reviewer_system_prompt(["critical", "high"])
    huge_tests = "F" * 5000
    payload = build_reviewer_payload(
        cfg, system, "task", "ctx", None, bundle, huge_tests, ""
    )
    assert any("global cap" in t for t in payload.truncations)


def test_load_requested_files_sanitization(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pkg").mkdir(parents=True)
    (repo / "pkg" / "ok.py").write_text("print('x')\n", encoding="utf-8")
    (repo / ".env").write_text("KEY=v\n", encoding="utf-8")
    cfg = CollabConfig()
    files, notes = load_requested_files(
        repo,
        ["pkg/ok.py", "../outside.py", "C:/abs.py", ".env", "missing.py"],
        cfg,
    )
    assert list(files) == ["pkg/ok.py"]
    assert len(notes) == 4


# --------------------------------------------------------------------- budget
def test_budget_tracker_limits():
    cfg = BudgetCfg(
        max_reviewer_calls=2, max_input_tokens=100,
        max_output_tokens=50, max_cost_usd=1.0,
    )
    tracker = BudgetTracker(cfg, BudgetState())
    assert tracker.check_next_call(40) is None
    tracker.record_call(40, 10, 0.2)
    assert tracker.check_next_call(70) is not None  # input budget
    assert tracker.check_next_call(40) is None
    tracker.record_call(40, 45, 0.2)
    assert "max_output_tokens" in (tracker.post_call_exceeded() or "")
    assert tracker.check_next_call(1) is not None  # call count exhausted


def test_budget_disabled_limits():
    tracker = BudgetTracker(
        BudgetCfg(max_reviewer_calls=0, max_input_tokens=0,
                  max_output_tokens=0, max_cost_usd=0),
        BudgetState(reviewer_calls=99, input_tokens=10**9,
                    output_tokens=10**9, cost_usd=999.0),
    )
    assert tracker.check_next_call(10**9) is None
    assert tracker.post_call_exceeded() is None


# ----------------------------------------------------------------------- util
def test_truncation_helpers():
    text = "x" * 100
    kept, truncated = truncate_middle(text, 50)
    assert truncated and len(kept) < 100 and "TRUNCATED" in kept
    tail, truncated = truncate_tail("head" + "y" * 100, 40)
    assert truncated and tail.endswith("y" * 30) and "TRUNCATED" in tail
    same, truncated = truncate_middle("short", 50)
    assert same == "short" and not truncated


def test_slugify():
    assert slugify("Hello World!") == "hello-world"
    assert slugify("___") == "task"
