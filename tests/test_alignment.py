"""Automatic reordered-variant alignment — offline. Generic variant ids only."""

from __future__ import annotations

from autograder.alignment import (PermutationProposal, align_variant, alignment_contract,
                                  deterministic_align_question, split_numbered_items)
from autograder.backends.mock import MockBackend
from autograder.discovery import discover_package
from autograder.gateway import ModelGateway
from autograder.schema import AnswerKey, KeyQuestion, KeySubItem
from autograder.variant import alignment_from_override, validate_alignment


def _key(versions=("V1", "V2")) -> AnswerKey:
    subs = [
        KeySubItem(id="1", prompt="Apply a Gaussian blur with sigma 2 to the image", correct_by_version={v: ["A"] for v in versions}, points=2),
        KeySubItem(id="2", prompt="Compute the DFT magnitude of the row signal", correct_by_version={v: ["B"] for v in versions}, points=2),
        KeySubItem(id="3", prompt="Threshold the histogram at level 128 using Otsu", correct_by_version={v: ["C"] for v in versions}, points=2),
    ]
    q = KeyQuestion(id="1", title="Operations", type="multiple_choice", max_points=6, sub_items=subs)
    return AnswerKey(exam_title="t", versions=list(versions), questions=[q], total_points=6)


def _printed(*texts):
    return [(str(i + 1), t) for i, t in enumerate(texts)]


K = _key()
T1, T2, T3 = [s.prompt for s in K.questions[0].sub_items]


def _gw(local=None, cloud=None):
    calls = {"align_resolve": 0, "align_resolve_cloud": 0}
    q = {"align_resolve": list(local or []), "align_resolve_cloud": list(cloud or [])}

    def factory(cfg):
        def responder(model, system, blocks):
            calls[cfg.model] += 1
            return q[cfg.model].pop(0)
        return MockBackend(config=cfg, responder=responder)

    models = {t: {"backend": "mock", "model": t} for t in q if q[t] or t == "align_resolve"}
    return ModelGateway.from_dict({"models": models}, backend_factory=factory), calls


# 1. A/B textual variants, same ordering (labels differ, text identical)
def test_textual_variants_same_ordering_are_identity():
    r = align_variant(K, "V2", {"1": _printed(T1, T2, T3)})
    q = r.questions["1"]
    assert q.identity and q.source == "deterministic_structure"
    assert r.to_contract_entry() == {"identity": True}


# 2. icon variants: alignment is text-driven, marker kind is irrelevant
def test_icon_variants_alignment_ignores_marker_kind():
    # nothing in the alignment code path references marker/icon concepts;
    # two "icon" variants with identical text align to identity, one reordered aligns to a permutation
    res = {"icon_a": align_variant(K, "icon_a", {"1": _printed(T1, T2, T3)}),
           "icon_b": align_variant(K, "icon_b", {"1": _printed(T2, T3, T1)})}
    contract, unresolved = alignment_contract(res)
    assert not unresolved
    assert contract["icon_a"] == {"identity": True}
    assert contract["icon_b"]["1"] == {"1": "2", "2": "3", "3": "1"}
    import inspect
    from autograder import alignment
    src = inspect.getsource(alignment).lower()
    for banned in ("heart", "spade", "diamond", "club", "flower", "clover", "daisy"):
        assert banned not in src


# 3. no-marker variants identified through question text/order
def test_no_marker_variants_distinguished_by_text_order():
    booklet_a = "1. " + T1 + "\n2. " + T2 + "\n3. " + T3
    booklet_b = "1. " + T3 + "\n2. " + T1 + "\n3. " + T2
    items_a, items_b = split_numbered_items(booklet_a), split_numbered_items(booklet_b)
    ra = align_variant(K, "V1", {"1": items_a})
    rb = align_variant(K, "V2", {"1": items_b})
    assert ra.to_contract_entry() == {"identity": True}
    assert rb.to_contract_entry()["1"] == {"1": "3", "2": "1", "3": "2"}
    # the two variants are told apart purely by their printed order (no marker involved)
    assert ra.to_contract_entry() != rb.to_contract_entry()


# 4. genuine reordered variant aligned automatically, contract validated by the frozen loader
def test_reordered_variant_aligned_and_valid_for_pipeline():
    r = align_variant(K, "V2", {"1": _printed(T3, T1, T2)})
    q = r.questions["1"]
    assert q.mapping == {"1": "3", "2": "1", "3": "2"} and not q.identity and q.source == "deterministic_text"
    contract, unresolved = alignment_contract({"V1": align_variant(K, "V1", {"1": _printed(T1, T2, T3)}), "V2": r})
    assert not unresolved
    va = alignment_from_override(K, "V2", {**contract, "_path": "auto"})
    assert va is not None and validate_alignment(K, va) == []
    assert va.questions[0].printed_to_key == {"1": "3", "2": "1", "3": "2"}
    assert va.questions[0].identical_order is False


# 5. ambiguous reorder escalates instead of guessing (deterministic + local unsure -> cloud -> human)
def test_ambiguous_reorder_escalates_not_guesses():
    key = _key()
    # make two key items textually near-identical so no unique assignment exists
    key.questions[0].sub_items[1].prompt = "Apply a Gaussian blur with sigma 3 to the image"
    ambiguous = _printed("Apply a Gaussian blur with sigma to the image", "Threshold the histogram at level 128 using Otsu",
                         "Apply a Gaussian blur with sigma to the image")
    d = deterministic_align_question(key.questions[0], ambiguous)
    assert d.mapping is None and "ambiguous" in d.evidence or "bijection" in d.evidence
    # local model unsure, cloud unsure -> unresolved -> human (never a guessed contract)
    gw, calls = _gw(local=[PermutationProposal(question_id="1", printed_to_key={"1": "1", "2": "3", "3": "2"}, confident=False)],
                    cloud=[PermutationProposal(question_id="1", printed_to_key={"1": "2", "2": "3"}, confident=True)])  # incomplete
    r = align_variant(key, "V2", {"1": ambiguous}, gateway=gw)
    assert calls["align_resolve"] == 1 and calls["align_resolve_cloud"] == 1
    assert r.questions["1"].mapping is None and r.to_contract_entry() is None
    contract, unresolved = alignment_contract({"V2": r})
    assert unresolved == ["V2"] and "V2" not in contract
    # a confident, complete local proposal IS accepted
    gw2, _ = _gw(local=[PermutationProposal(question_id="1", printed_to_key={"1": "1", "2": "3", "3": "2"}, confident=True)])
    r2 = align_variant(key, "V2", {"1": ambiguous}, gateway=gw2)
    assert r2.questions["1"].source == "local_model" and r2.to_contract_entry()["1"] == {"1": "1", "2": "3", "3": "2"}


def test_discovery_emits_derived_alignment_and_flags_unresolved():
    key = _key()
    res = discover_package(key=key, key_bytes=b"K", exam_text_layer="נוסח V1 נוסח V2",
                           printed_items_by_variant={"V1": {"1": _printed(T1, T2, T3)},
                                                     "V2": {"1": _printed(T2, T3, T1)}})
    assert res.alignment.value["V1"] == {"identity": True}
    assert res.alignment.value["V2"]["1"] == {"1": "2", "2": "3", "3": "1"}
    assert not [n for n in res.needs_human if n.startswith("alignment")]
    res2 = discover_package(key=key, key_bytes=b"K", exam_text_layer="נוסח V1 נוסח V2",
                            printed_items_by_variant={"V1": {"1": _printed(T1, T2, T3)}})   # V2 booklet missing
    assert "alignment:V2" in res2.needs_human
