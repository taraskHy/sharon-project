"""Automatic package/variant discovery — offline, mocked models."""

from __future__ import annotations

import json

from autograder.backends.mock import MockBackend
from autograder.discovery import (MarkerCatalog, PolicyInference, VariantCatalogStore,
                                  deterministic_markers, deterministic_versions,
                                  discover_package, write_sidecars)
from autograder.gateway import ModelGateway
from autograder.variant import load_variant_config
from tests.test_grade import make_key


def _gw(responses_by_task):
    calls = {t: 0 for t in responses_by_task}
    q = {t: list(v) for t, v in responses_by_task.items()}

    def factory(cfg):
        task = cfg.model

        def responder(model, system, blocks):
            calls[task] += 1
            return q[task].pop(0)
        return MockBackend(config=cfg, responder=responder)

    return ModelGateway.from_dict({"models": {t: {"backend": "mock", "model": t} for t in responses_by_task}},
                                  backend_factory=factory), calls


def test_versions_from_key_are_authoritative():
    key = make_key()   # versions A1/A2/A3
    v = deterministic_versions(key, "")
    assert v.value == ["A1", "A2", "A3"] and v.source == "deterministic"


def test_deterministic_markers_from_text_layer_labels():
    m = deterministic_markers(["A1", "A2", "A3"], "בחינה נוסח A1 ... נוסח A2 ... נוסח A3")
    assert m.source == "deterministic" and set(m.value["markers"]) == {"A1", "A2", "A3"}
    m2 = deterministic_markers(["heart", "spade"], "no labels, icons only")
    assert m2.source == "unresolved" and m2.value is None


def test_discovery_emits_existing_contracts_and_persists(tmp_path):
    key = make_key()
    cat = MarkerCatalog(n_variants=3, confident=True, marker_kind="flower", marker_page=1,
                        markers=[{"id": "clover", "variant": "A1", "description": "four petals"},
                                 {"id": "star", "variant": "A2", "description": "five pointed petals"},
                                 {"id": "daisy", "variant": "A3", "description": "many small petals"}])
    gw, calls = _gw({"variant_resolve": [cat],
                     "policy_infer": [PolicyInference(question_id="1", policy="explanation_required_if_correct",
                                                      confident=True, evidence="rubric says so")]})
    res = discover_package(key=key, key_bytes=b"K", exam_bytes=b"E", exam_text_layer="icons only",
                           cover_png_b64="AAA=", rubric_texts={"1": "explanation graded only if correct"},
                           gateway=gw)
    assert res.variants_config.source == "local_model" and calls["variant_resolve"] == 1
    assert res.policies["1"].value == "explanation_required_if_correct" and res.policies["1"].source == "local_model"
    assert res.policies["3"].value == "choice_only" and res.policies["3"].source == "deterministic"
    assert not res.unresolved()
    # sidecars in the EXISTING contracts, loadable by the frozen variant loader
    keyp = tmp_path / "exam.answer_key.json"
    keyp.write_text("{}", encoding="utf-8")
    written = write_sidecars(res, keyp)
    names = {p.name for p in written}
    assert {"exam.answer_key.variants.json", "exam.answer_key.alignment.json", "exam.answer_key.template.json"} <= names
    cfg = load_variant_config(keyp)
    assert {e["variant"] for e in cfg["markers"].values()} == {"A1", "A2", "A3"}
    # persisted catalog reused by fingerprint
    store = VariantCatalogStore(tmp_path / "catalog")
    store.save(res)
    assert store.load(res.package_fingerprint)["variants_config"]["source"] == "local_model"


def test_unresolved_routes_to_human_and_human_resolution_is_reused(tmp_path):
    key = make_key()
    gw, calls = _gw({"variant_resolve": [MarkerCatalog(n_variants=3, confident=False, markers=[])],
                     "variant_resolve_cloud": [MarkerCatalog(n_variants=2, confident=True,
                                                             markers=[{"id": "x", "variant": "A1", "description": "d"}])]})
    res = discover_package(key=key, key_bytes=b"K", exam_bytes=b"E", cover_png_b64="AAA=", gateway=gw)
    assert calls["variant_resolve"] == 1 and calls["variant_resolve_cloud"] == 1   # incomplete cloud catalog rejected
    assert "variants" in res.unresolved()
    store = VariantCatalogStore(tmp_path / "catalog")
    store.save(res)
    manual = {"markers": {"h": {"variant": "A1", "description": "heart"}, "s": {"variant": "A2", "description": "spade"},
                          "d": {"variant": "A3", "description": "diamond"}}}
    d = store.apply_human(res.package_fingerprint, variants_config=manual)
    assert d["variants_config"]["source"] == "human" and "variants" not in d["needs_human"]
    again = store.load(res.package_fingerprint)
    assert again["variants_config"]["value"] == manual   # exact resolution reused, no re-asking


def test_existing_manual_sidecar_is_never_overwritten(tmp_path):
    key = make_key()
    res = discover_package(key=key, key_bytes=b"K", exam_text_layer="נוסח A1 נוסח A2 נוסח A3")
    keyp = tmp_path / "k.json"
    keyp.write_text("{}", encoding="utf-8")
    manual = keyp.with_name("k.variants.json")
    manual.write_text('{"markers": {"m": {"variant": "A1", "description": "manual"}}}', encoding="utf-8")
    write_sidecars(res, keyp)
    assert json.loads(manual.read_text(encoding="utf-8"))["markers"]["m"]["description"] == "manual"


def test_no_gateway_is_purely_deterministic():
    key = make_key()
    res = discover_package(key=key, key_bytes=b"K")
    assert res.template.value["mode"] in ("multiple_choice", "with_explanation", "mixed")
    assert all(f.source in ("deterministic", "unresolved") for f in res.policies.values())
