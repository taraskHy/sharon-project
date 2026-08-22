"""SYNTHETIC_NEAR_MISS — second verifier-benchmark component (proposal tooling).

Purpose: the REAL historical-error benchmark (verifier_bench/selected/) is
dominated by severe OCR errors (only 9 subtle negatives); this layer tests
SMALL but grading-relevant transcription errors a verifier must still
reject. It starts ONLY from the frozen audited references and applies
deterministic, generic, OCR-fidelity corruption rules that are fixed in
this file BEFORE any model output is inspected (RULES_VERSION is persisted).
These are fidelity perturbations, never "corrected answers".

Rules (applied where representable; at most ~2 per image):
  numeric group (grading-relevant, preferred first)
    digit_substitution            one digit d -> (d+1) mod 10
    operator_substitution         one math operator swapped (+<->-, *<->/,
                                  <<->>, ^->*) or removed (=, %)
    decimal_point_corruption      one decimal point removed (0.55 -> 055)
    superscript_subscript_loss    super/subscript digits flattened, x^2 -> x2
  text group
    char_deletion                 one interior letter of a >=4-letter word
    short_token_omission          one short (<=3 char) token dropped
    token_duplication_addition    one short token duplicated ("word word")
Hebrew prefix hyphens (e.g. "ב-High") are connectors, never operators.

Every synthetic case inherits the SPLIT of its source image from the frozen
REAL selected manifest (zero image leakage by construction). Synthetic
cases are kept in a separate component (verifier_bench/synthetic/) and must
be reported separately from REAL cases: REAL -> FAR/FRR/SUPPORTED
precision/REVIEW rate; SYNTHETIC -> FAR overall + FAR by corruption type
(especially number/operator/formula); COMBINED only secondarily.

``propose`` writes verifier_bench/synthetic_near_miss_proposal.json and
prints the report; ``freeze`` (owner-approved) writes verifier_bench/
synthetic/. No model is ever invoked here.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib.util
import json
import os
import re
import sys
import uuid
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


refaudit = _load("refaudit")
vsel = _load("verifier_select")

RULES_VERSION = "synthetic-near-miss-rules-v1 (2026-08-22, fixed before any model output)"
SYNTH_DIRNAME = "synthetic"
PROPOSAL_FILENAME = "synthetic_near_miss_proposal.json"
MAX_PER_IMAGE = 2

NUMERIC_RULES = ("digit_substitution", "operator_substitution",
                 "decimal_point_corruption", "superscript_subscript_loss")
TEXT_RULES = ("char_deletion", "short_token_omission", "token_duplication_addition")
ALL_RULES = NUMERIC_RULES + TEXT_RULES

_OP_SUB = {"+": "-", "-": "+", "*": "/", "/": "*", "<": ">", ">": "<", "^": "*",
           "=": "", "%": ""}
_SUPER = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789"))
_SUB = dict(zip("₀₁₂₃₄₅₆₇₈₉", "0123456789"))
_LETTER_WORD = re.compile(r"^[A-Za-zא-ת]{4,}$")
_DECIMAL = re.compile(r"\d\.\d")
_HEB = re.compile(r"[א-ת]")


class SynthError(RuntimeError):
    pass


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pick(item_id: str, rule: str, n: int) -> int:
    """Deterministic position choice: hash of (item, rule) modulo n."""
    return int(_sha(f"{item_id}::{rule}".encode()), 16) % n if n else 0


def _squeeze(s: str) -> str:
    return re.sub(r"[ \t]{2,}", " ", s).strip()


# ------------------------------------------------------------------ rules ----

def apply_rule(rule: str, text: str, item_id: str) -> str | None:
    """Return the corrupted candidate, or None when the rule is not
    applicable to this reference. Pure and deterministic."""
    if rule == "digit_substitution":
        pos = [i for i, ch in enumerate(text) if ch.isdigit() and ch in "0123456789"]
        if not pos:
            return None
        i = pos[_pick(item_id, rule, len(pos))]
        return text[:i] + str((int(text[i]) + 1) % 10) + text[i + 1:]
    if rule == "operator_substitution":
        pos = [i for i, ch in enumerate(text) if ch in _OP_SUB
               and not (ch == "-" and i > 0 and _HEB.match(text[i - 1]))]
        if not pos:
            return None
        i = pos[_pick(item_id, rule, len(pos))]
        return _squeeze(text[:i] + _OP_SUB[text[i]] + text[i + 1:])
    if rule == "decimal_point_corruption":
        matches = list(_DECIMAL.finditer(text))
        if not matches:
            return None
        m = matches[_pick(item_id, rule, len(matches))]
        i = m.start() + 1
        return text[:i] + text[i + 1:]
    if rule == "superscript_subscript_loss":
        if not (any(ch in _SUPER or ch in _SUB for ch in text) or re.search(r"\^\d", text)):
            return None
        out = "".join(_SUPER.get(ch, _SUB.get(ch, ch)) for ch in text)
        out = re.sub(r"\^(\d)", r"\1", out)
        return out if out != text else None
    tokens = text.split()
    if rule == "char_deletion":
        cand = [k for k, t in enumerate(tokens) if _LETTER_WORD.match(t)]
        if not cand:
            return None
        k = cand[_pick(item_id, rule, len(cand))]
        w = tokens[k]
        j = 1 + _pick(item_id, rule + "#pos", len(w) - 2)   # interior position
        tokens[k] = w[:j] + w[j + 1:]
        return " ".join(tokens)
    if rule == "short_token_omission":
        if len(tokens) < 3:
            return None
        cand = [k for k, t in enumerate(tokens) if len(t.strip(".,;:()[]")) <= 3
                and t.strip(".,;:()[]")]
        if not cand:
            return None
        k = cand[_pick(item_id, rule, len(cand))]
        return " ".join(tokens[:k] + tokens[k + 1:])
    if rule == "token_duplication_addition":
        if len(tokens) < 2:
            return None
        short = [k for k, t in enumerate(tokens) if len(t) <= 4] or list(range(len(tokens)))
        k = short[_pick(item_id, rule, len(short))]
        return " ".join(tokens[:k + 1] + [tokens[k]] + tokens[k + 1:])
    raise SynthError(f"unknown rule {rule!r}")


# ------------------------------------------------------------- building ----

def _load_selected(store) -> dict:
    sel_dir = store.bench_dir / vsel.RAW_DIRNAME / vsel.SELECTED_DIRNAME
    manifest_path = sel_dir / "manifest.json"
    if not manifest_path.exists():
        raise SynthError("the REAL selected benchmark is not frozen yet "
                         "(verifier_bench/selected/manifest.json missing) — synthetic "
                         "cases inherit its split assignment")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    labels = [json.loads(l) for l in
              (sel_dir / "cases_labels.jsonl").read_text(encoding="utf-8").splitlines()
              if l.strip()]
    return {"manifest": manifest, "labels": labels,
            "manifest_sha256": _sha(manifest_path.read_bytes())}


def build_synthetic(store, selected: dict | None = None) -> dict:
    if not refaudit.is_frozen(store):
        raise SynthError("the manual audit is not frozen/current")
    selected = selected or load_selected_with_candidates(store)
    normalize, lev, word_align = refaudit._load_metric_fns()
    sig = refaudit.digit_op_signature

    def key(t: str) -> tuple:
        return (normalize(t), sig(t))

    def cer(ref: str, hyp: str) -> float:
        r, h = normalize(ref), normalize(hyp)
        return (lev(r, h) / len(r)) if r else (0.0 if not h else 1.0)

    item_split = {i: s for s, ids in selected["manifest"]["image_ids_per_split"].items()
                  for i in ids}
    real_neg_keys: dict[str, set] = collections.defaultdict(set)
    for lab in selected["labels"]:
        if lab["polarity"] == "negative":
            real_neg_keys[lab["item_id"]].add(key(_candidate_of(lab)))

    inputs, labels = [], []
    removals = collections.Counter()
    no_safe: list[dict] = []
    per_rule_applicable = collections.Counter()
    for item_id in store.eligible_ids:
        entry = store.entry(item_id)
        if entry["status"] not in ("confirmed", "corrected"):
            continue
        item = store.item(item_id)
        split = item_split.get(item_id)
        if split is None:
            raise SynthError(f"{item_id} has no split in the frozen selected manifest")
        reference = entry["audited_reference"]
        ref_key = key(reference)
        generated: list[tuple[str, str]] = []       # (rule, candidate)
        seen_keys = set()
        for rule in ALL_RULES:
            cand = apply_rule(rule, reference, item_id)
            if cand is None or not cand.strip():
                continue
            per_rule_applicable[rule] += 1
            k = key(cand)
            if k == ref_key:
                removals["no_effect_after_normalization"] += 1
                continue
            if k in real_neg_keys.get(item_id, set()):
                removals["duplicate_of_real_negative"] += 1
                continue
            if k in seen_keys:
                removals["duplicate_synthetic_same_image"] += 1
                continue
            seen_keys.add(k)
            generated.append((rule, cand))
        # selection: first numeric rule, then first text rule; else two text rules
        numeric = [g for g in generated if g[0] in NUMERIC_RULES]
        textual = [g for g in generated if g[0] in TEXT_RULES]
        picks = (numeric[:1] + textual[:1]) if numeric else textual[:2]
        picks = picks[:MAX_PER_IMAGE]
        if not picks:
            no_safe.append({"item_id": item_id, "reference_tokens": len(reference.split()),
                            "reason": "no applicable generic rule produced a distinct "
                                      "candidate"})
            continue
        for rule, cand in picks:
            case_id = _sha(f"synthetic::{item_id}::{rule}::{cand}".encode())[:12]
            kinds = refaudit._error_kinds(normalize, word_align, reference, cand)
            inputs.append({"case_id": case_id, "crop": item["image"],
                           "candidate_transcription": cand})
            labels.append({"case_id": case_id, "item_id": item_id,
                           "writer": item.get("writer"), "split": split,
                           "expected_verdict": "review", "polarity": "negative",
                           "source": "synthetic_near_miss", "corruption_type": rule,
                           "corruption_group": "numeric" if rule in NUMERIC_RULES else "text",
                           "error_kinds": kinds, "cer_vs_audited": round(cer(reference, cand), 4),
                           "audited_reference": reference})
    inputs.sort(key=lambda r: r["case_id"])
    by_id = {l["case_id"]: l for l in labels}
    labels = [by_id[r["case_id"]] for r in inputs]
    report = _report(store, labels, removals, no_safe, per_rule_applicable, selected)
    return {"inputs": inputs, "labels": labels, "report": report}


def _candidate_of(label: dict) -> str:
    # the selected labels file does not carry the candidate text; re-derive
    # the dedup key from the audited reference is wrong — so we look it up
    # from the selected inputs file lazily (loaded once by the caller).
    return label.get("_candidate", "")


def _report(store, labels, removals, no_safe, applicable, selected) -> dict:
    by_split = {}
    for s in ("DEV", "CALIBRATION", "HELD_OUT"):
        rows = [l for l in labels if l["split"] == s]
        by_split[s] = {"cases": len(rows),
                       "images": len({l["item_id"] for l in rows}),
                       "writers": sorted({l["writer"] for l in rows}),
                       "numeric_group_cases": sum(1 for l in rows if l["corruption_group"] == "numeric")}
    image_splits = collections.defaultdict(set)
    for l in labels:
        image_splits[l["item_id"]].add(l["split"])
    per_image = collections.Counter(collections.Counter(l["item_id"] for l in labels).values())
    return {
        "rules_version": RULES_VERSION,
        "rules": list(ALL_RULES),
        "synthetic_cases_total": len(labels),
        "cases_by_corruption_type": dict(collections.Counter(l["corruption_type"] for l in labels)),
        "cases_by_group": dict(collections.Counter(l["corruption_group"] for l in labels)),
        "rule_applicability_counts": dict(applicable),
        "images_covered": len({l["item_id"] for l in labels}),
        "eligible_images": len(store.eligible_ids),
        "writers_covered": dict(sorted(collections.Counter(l["writer"] for l in labels).items())),
        "cases_per_image_distribution": dict(sorted(per_image.items())),
        "by_split": by_split,
        "duplicate_removals": dict(removals),
        "images_without_safe_corruption": no_safe,
        "zero_image_overlap_between_splits": all(len(s) == 1 for s in image_splits.values()),
        "inherits_split_from": {"selected_manifest_sha256": selected["manifest_sha256"],
                                "decision": selected["manifest"].get("decision")},
        "metrics_contract": {
            "REAL": ["FAR", "FRR", "SUPPORTED precision", "REVIEW rate"],
            "SYNTHETIC_NEAR_MISS": ["FAR overall", "FAR by corruption type",
                                    "FAR numeric group (number/operator/formula)"],
            "COMBINED": "secondary only; never hides either source",
        },
    }


def write_proposal(store, synth: dict) -> Path:
    out = store.bench_dir / vsel.RAW_DIRNAME / PROPOSAL_FILENAME
    refaudit._atomic_write_json(out, {
        "_policy": "PROPOSAL ONLY — synthetic near-miss layer is NOT frozen.",
        "generated_at": refaudit._now(), **synth["report"]})
    return out


def freeze_synthetic(store, synth: dict) -> dict:
    if not synth["report"]["zero_image_overlap_between_splits"]:
        raise SynthError("refusing to freeze: image overlap between splits")
    out_dir = store.bench_dir / vsel.RAW_DIRNAME / SYNTH_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"cases_inputs.jsonl": synth["inputs"], "cases_labels.jsonl": synth["labels"]}
    tmps = []
    try:
        for name, rows in paths.items():
            tmp = out_dir / f"{name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            tmps.append((tmp, out_dir / name))
        for tmp, path in tmps:
            os.replace(tmp, path)
    except BaseException:
        for tmp, _ in tmps:
            tmp.unlink(missing_ok=True)
        raise
    manifest = {
        "_policy": ("Frozen SYNTHETIC_NEAR_MISS verifier component. Model-visible: "
                    "cases_inputs.jsonl only (opaque id, crop, candidate). Report "
                    "SEPARATELY from the REAL component (FAR overall + FAR by "
                    "corruption type, esp. numeric); COMBINED only secondarily."),
        "frozen_at": refaudit._now(),
        "report": synth["report"],
        "inputs_sha256": _sha((out_dir / "cases_inputs.jsonl").read_bytes()),
        "labels_sha256": _sha((out_dir / "cases_labels.jsonl").read_bytes()),
        "audit_sha256": refaudit._sha256_json(store.entries_canonical()),
    }
    refaudit._atomic_write_json(out_dir / "manifest.json", manifest)
    checks = "\n".join(f"{_sha((out_dir / n).read_bytes())}  {n}" for n in
                       ("cases_inputs.jsonl", "cases_labels.jsonl", "manifest.json")) + "\n"
    tmp = out_dir / f"CHECKSUMS.{os.getpid()}.tmp"
    tmp.write_text(checks, encoding="utf-8")
    os.replace(tmp, out_dir / "CHECKSUMS.sha256")
    return manifest


def _print(r: dict) -> None:
    print(f"rules: {r['rules_version']}")
    print(f"synthetic cases: {r['synthetic_cases_total']}  by type: {r['cases_by_corruption_type']}")
    print(f"by group: {r['cases_by_group']}   applicable counts: {r['rule_applicability_counts']}")
    print(f"images covered: {r['images_covered']}/{r['eligible_images']}   writers: {r['writers_covered']}")
    print(f"cases per image: {r['cases_per_image_distribution']}")
    for s, v in r["by_split"].items():
        print(f"  {s:12s} cases={v['cases']:4d} images={v['images']:3d} writers={v['writers']} numeric={v['numeric_group_cases']}")
    print(f"duplicate removals: {r['duplicate_removals']}")
    print(f"images without safe corruption: {len(r['images_without_safe_corruption'])} "
          f"{[x['item_id'] for x in r['images_without_safe_corruption']]}")
    print(f"zero image overlap between splits: {r['zero_image_overlap_between_splits']}")


def load_selected_with_candidates(store) -> dict:
    """Selected manifest + labels, with each label carrying its candidate
    text (joined from the model-visible inputs) for dedup purposes."""
    selected = _load_selected(store)
    sel_dir = store.bench_dir / vsel.RAW_DIRNAME / vsel.SELECTED_DIRNAME
    cands = {}
    for l in (sel_dir / "cases_inputs.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            row = json.loads(l)
            cands[row["case_id"]] = row["candidate_transcription"]
    for lab in selected["labels"]:
        lab["_candidate"] = cands.get(lab["case_id"], "")
    return selected


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="SYNTHETIC_NEAR_MISS verifier layer (no model calls)")
    ap.add_argument("--bench-dir", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("propose")
    sub.add_parser("freeze")
    args = ap.parse_args(argv)
    store = refaudit.AuditStore(Path(args.bench_dir) if args.bench_dir else None)
    try:
        synth = build_synthetic(store, load_selected_with_candidates(store))
    except SynthError as exc:
        print(f"REFUSED: {exc}")
        return 2
    _print(synth["report"])
    if args.cmd == "propose":
        print(f"proposal written: {write_proposal(store, synth)}")
        return 0
    m = freeze_synthetic(store, synth)
    print(f"frozen: {store.bench_dir / vsel.RAW_DIRNAME / SYNTH_DIRNAME} inputs_sha256={m['inputs_sha256'][:12]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
