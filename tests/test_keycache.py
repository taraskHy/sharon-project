"""Persistent answer-key cache: one parse per unique configuration, strict
invalidation, and safe rejection of corrupted entries."""

import argparse
import json

from autograder import keycache
from autograder.backends import BackendConfig
from autograder.backends.mock import MockBackend
from autograder.cli import _get_key
from autograder.prompts import KEY_PARSER_SYSTEM
from tests.conftest import make_pdf
from tests.test_grade import make_key


def _backend(counter: dict, model: str = "m1") -> MockBackend:
    def responder(output_model, system, blocks):
        if output_model.__name__ == "AnswerKey":
            counter["parses"] = counter.get("parses", 0) + 1
        return make_key().model_copy(deep=True)

    return MockBackend(config=BackendConfig(backend="mock", model=model), responder=responder)


def _ns(key_path, cache_dir, rubric=None):
    return argparse.Namespace(
        key=str(key_path),
        rubric=str(rubric) if rubric else None,
        resume=False,
        key_cache_dir=str(cache_dir),
        no_key_cache=False,
    )


def _get_key_in(ns, backend, out_dir, max_edge):
    out_dir.mkdir(parents=True, exist_ok=True)  # run_grade_pipeline does this
    return _get_key(ns, backend, out_dir, max_edge, reusable=False)


def test_same_key_parsed_once_across_runs_and_exams(tmp_path, no_network):
    key_pdf = make_pdf(tmp_path / "key.pdf")
    cache = tmp_path / "cache"
    counter = {}
    backend = _backend(counter)

    # Three "exams" (three separate out dirs), same key/config: one parse.
    sources = []
    for i in range(3):
        key, source = _get_key_in(_ns(key_pdf, cache), backend, tmp_path / f"out{i}", 800)
        sources.append(source)
        assert key.exam_title
    assert counter["parses"] == 1
    assert sources == ["parsed", "cache", "cache"]


def test_cache_invalidated_by_each_component(tmp_path, no_network):
    key_pdf = make_pdf(tmp_path / "key.pdf")
    cache = tmp_path / "cache"
    counter = {}

    _get_key_in(_ns(key_pdf, cache), _backend(counter), tmp_path / "o1", 800)
    assert counter["parses"] == 1

    # different key bytes
    other_pdf = make_pdf(tmp_path / "key2.pdf", pages=4)
    _get_key_in(_ns(other_pdf, cache), _backend(counter), tmp_path / "o2", 800)
    assert counter["parses"] == 2

    # different rubric
    rubric = tmp_path / "rubric.txt"
    rubric.write_text("cap at 36", encoding="utf-8")
    _get_key_in(_ns(key_pdf, cache, rubric=rubric), _backend(counter), tmp_path / "o3", 800)
    assert counter["parses"] == 3

    # different model
    _get_key_in(_ns(key_pdf, cache), _backend(counter, model="m2"), tmp_path / "o4", 800)
    assert counter["parses"] == 4

    # different render size
    _get_key_in(_ns(key_pdf, cache), _backend(counter), tmp_path / "o5", 1600)
    assert counter["parses"] == 5

    # unchanged config again: cache hit, no new parse
    _get_key_in(_ns(key_pdf, cache), _backend(counter), tmp_path / "o6", 800)
    assert counter["parses"] == 5


def test_prompt_and_schema_versions_enter_the_fingerprint(no_network):
    base = dict(
        key_bytes_hash="k" * 64,
        rubric_text=None,
        backend_description={"backend": "mock", "model": "m"},
        max_image_edge=800,
    )
    fp1 = keycache.key_fingerprint(parser_prompt=KEY_PARSER_SYSTEM, **base)
    fp2 = keycache.key_fingerprint(parser_prompt=KEY_PARSER_SYSTEM + "\nEDITED.", **base)
    assert fp1 != fp2, "editing the parser prompt must invalidate the cache"

    # The operator override file is deliberately NOT in the parse fingerprint:
    # overrides re-apply deterministically on every load (cache hits too), so
    # editing them never forces a re-parse; they invalidate the per-exam
    # grading fingerprints instead (covered in test_variant fingerprints).


def test_corrupted_cache_entries_are_rejected_and_reparsed(tmp_path, no_network):
    key_pdf = make_pdf(tmp_path / "key.pdf")
    cache = tmp_path / "cache"
    counter = {}
    _get_key_in(_ns(key_pdf, cache), _backend(counter), tmp_path / "o1", 800)
    assert counter["parses"] == 1
    entries = list(cache.glob("*.json"))
    assert len(entries) == 1

    # Truncated JSON
    entries[0].write_text('{"fingerprint": "abc", "answer_key": {', encoding="utf-8")
    _get_key_in(_ns(key_pdf, cache), _backend(counter), tmp_path / "o2", 800)
    assert counter["parses"] == 2, "corrupt cache must trigger a fresh parse"

    # Valid JSON, wrong embedded fingerprint (copied/renamed file)
    payload = json.loads(entries[0].read_text(encoding="utf-8"))
    payload["fingerprint"] = "not-the-right-one"
    entries[0].write_text(json.dumps(payload), encoding="utf-8")
    _get_key_in(_ns(key_pdf, cache), _backend(counter), tmp_path / "o3", 800)
    assert counter["parses"] == 3

    # Schema-invalid answer_key payload
    payload = json.loads(entries[0].read_text(encoding="utf-8"))
    payload["answer_key"] = {"exam_title": 42}
    entries[0].write_text(json.dumps(payload), encoding="utf-8")
    _get_key_in(_ns(key_pdf, cache), _backend(counter), tmp_path / "o4", 800)
    assert counter["parses"] == 4


def _variant_cfg_next_to(key_pdf):
    import json as _json

    cfg = {
        "marker_kind": "flower",
        "markers": {
            "a": {"variant": "A1", "description": "one"},
            "b": {"variant": "A2", "description": "two"},
            "c": {"variant": "A3", "description": "three"},
        },
    }
    path = key_pdf.with_name(key_pdf.stem + ".variants.json")
    path.write_text(_json.dumps(cfg), encoding="utf-8")
    return path


def test_key_parse_missing_versions_is_rejected_then_retried(tmp_path, no_network):
    """A parsed key that lacks the family's declared versions must never be
    used or cached: one bounded re-parse, then a hard error."""
    from autograder.backends.base import BackendError

    key_pdf = make_pdf(tmp_path / "key.pdf")
    _variant_cfg_next_to(key_pdf)
    cache = tmp_path / "cache"

    # First parse returns a defective key (versions=['default']), the retry
    # returns a good one: the good one is used and cached.
    calls = {"n": 0}

    def flaky(model, system, blocks):
        calls["n"] += 1
        good = make_key().model_copy(deep=True)
        if calls["n"] == 1:
            bad = make_key().model_copy(deep=True)
            bad.versions = ["default"]
            return bad
        return good

    backend = MockBackend(config=BackendConfig(backend="mock", model="m"), responder=flaky)
    key, source = _get_key_in(_ns(key_pdf, cache), backend, tmp_path / "o1", 800)
    assert source == "parsed"
    assert sorted(key.versions) == ["A1", "A2", "A3"]
    assert calls["n"] == 2

    # Both attempts defective -> hard error, nothing cached.
    def always_bad(model, system, blocks):
        bad = make_key().model_copy(deep=True)
        bad.versions = ["default"]
        return bad

    cache2 = tmp_path / "cache2"
    backend2 = MockBackend(config=BackendConfig(backend="mock", model="m2"), responder=always_bad)
    try:
        _get_key_in(_ns(key_pdf, cache2), backend2, tmp_path / "o2", 800)
        raise AssertionError("expected BackendError for a defective key")
    except BackendError as e:
        assert "defective key" in str(e)
    assert not cache2.exists() or not list(cache2.glob("*.json"))


def test_key_parser_receives_expected_versions_hint(tmp_path, no_network):
    key_pdf = make_pdf(tmp_path / "key.pdf")
    _variant_cfg_next_to(key_pdf)
    counter = {}
    backend = _backend(counter)
    _get_key_in(_ns(key_pdf, cache_dir=tmp_path / "c"), backend, tmp_path / "o1", 800)
    text = backend.calls[0].all_text()
    assert "A1, A2, A3" in text, "parser must be told the required version ids"
    assert "AUTHORITATIVE NOTE" in text


def test_no_key_cache_flag_disables_reuse(tmp_path, no_network):
    key_pdf = make_pdf(tmp_path / "key.pdf")
    cache = tmp_path / "cache"
    counter = {}
    ns = _ns(key_pdf, cache)
    ns.no_key_cache = True
    _get_key_in(ns, _backend(counter), tmp_path / "o1", 800)
    _get_key_in(ns, _backend(counter), tmp_path / "o2", 800)
    assert counter["parses"] == 2
    assert not cache.exists() or not list(cache.glob("*.json"))
