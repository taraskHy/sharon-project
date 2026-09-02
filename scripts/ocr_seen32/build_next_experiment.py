"""Mission L: pre-register exactly one next experiment. NOT EXECUTED, NOT AUTHORIZED."""
import hashlib, json, subprocess, time
from pathlib import Path

from autograder.benchmark.manifests import load_manifest
from autograder.benchmark.roles import _load_historical_prompts, adapter_for
from autograder.benchmark.subsets import load_subset

R = Path("evaluation/model_selection/runs/ocr_primary")
EXP = Path("evaluation/model_selection/experiments")
GEM = "google/gemini-3.7-flash"
man = load_manifest("ocr_primary")
by = {c.case_id: c for c in man.cases}
ad = adapter_for("ocr_primary")
sub = load_subset("ocr_primary", "seen46_ocr_dev", man)
base = json.loads((R / "OCR_SEEN32_BASELINE_2026-09-02.json").read_text(encoding="utf-8"))
res = json.loads((R / "OCR_SEEN32_PAIRED_RESULT_2026-09-02.json").read_text(encoding="utf-8"))
ORDER = base["ordered_case_ids"]

cases = []
for cid in ORDER:
    bc = by[cid]
    cases.append({"case_id": cid, "split": bc.split, "category": bc.meta.get("category"),
                  "writer": bc.meta.get("writer"), "image": bc.inputs["image"],
                  "crop_sha256": hashlib.sha256((man.root / bc.inputs["image"]).read_bytes()).hexdigest(),
                  "reference_sha256": hashlib.sha256(bc.label["reference"].encode()).hexdigest()})

art = {
    "experiment": "OCR_PROMPT_V2_NEUTRAL_FRAMING_GEMINI_SEEN32",
    "status": "PRE-REGISTERED DESIGN — NOT EXECUTED, NOT AUTHORIZED, NO PROVIDER CALLS MADE",
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
    "requires": "explicit owner authorization with a spend ceiling before any execution",

    "question": (
        "Does the exam framing in the frozen m2-strict-v1 prompt drive Gemini's content-filter rate? "
        "On the 32 seen-DEV handwritten crops Gemini returned nothing usable for 18 (56%), of which "
        "10 were provider content-filter outcomes, concentrated on the cell-crop half (50% vs 12.5%). "
        "The prompt those crops carry describes a university exam answer cell and instructs the model "
        "to 'Ignore any red instructor ink'. That framing is a plausible, testable, and so far "
        "UNTESTED contributor."),

    "why_this_experiment_and_not_another": {
        "chosen": "E — a new OCR prompt version under a fresh freeze",
        "A_remaining_21_calibration_gemini_only": (
            "REJECTED. It spends the CALIBRATION half of the seen corpus — a population usable once — "
            "to add precision to a failure rate already bounded at 56% (95% UB 71%) on 32 crops. The "
            "number is already decisive; more of it changes no decision."),
        "B_remaining_21_paired": "REJECTED for the same reason at double the cost.",
        "C_gemini_primary_sonnet_fallback_only": (
            "REJECTED as an experiment: that is a deployment configuration, and this run already "
            "measured it (29/32 coverage, failure-aware CER 0.3560, 9 failure-aware critical errors)."),
        "D_different_provider": (
            "The strongest alternative, and the fallback if this fails. Held back only because we "
            "have not ruled out the prompt as a cause; switching provider first risks moving the "
            "wrong variable and learning nothing about why the filter fires."),
    },

    "design": {
        "population": {
            "subset": "seen46_ocr_dev (unchanged)",
            "subset_selection_sha256": sub["selection_sha256"],
            "manifest_hashes": man.hashes,
            "n": len(ORDER),
            "ordered_case_ids": ORDER,
            "case_order_sha256": hashlib.sha256("\n".join(ORDER).encode()).hexdigest(),
            "cases": cases,
            "composition": base["composition"],
            "held_out": 0, "calibration": 0,
        },
        "arms": [
            {"arm": "control", "prompt_version": "m2-strict-v1",
             "prompt_sha256_by_category": {k: hashlib.sha256(v.encode()).hexdigest()
                                           for k, v in sorted(_load_historical_prompts().items())},
             "model": GEM, "route": base["routes"][GEM]["fingerprint_fields"],
             "requests": 0,
             "source": ("REUSED from the paired 32-crop run committed at "
                        "runs_seen32/.../google-gemini-3.7-flash__c4ae61f634 — NOT re-executed"),
             "observed": {"usable": "14/32", "hard_failures": 18, "content_filter": 10,
                          "successful_only_cer": 0.1155}},
            {"arm": "treatment", "prompt_version": "ocr-neutral-v2",
             "prompt_status": ("TO BE WRITTEN AND FROZEN BEFORE ANY CALL; must be registered in "
                               "approved_cloud_ocr_systems() in code review, and its hash recorded "
                               "here before execution"),
             "prompt_requirements": [
                 "identical output contract: BenchTranscription {transcription: str}",
                 "identical unreadable-marker, struck-through and RTL-order rules",
                 "NO mention of exam, instructor, grading, marks or red ink",
                 "neutral description of the task: transcribe the handwritten Hebrew in this image",
                 "no case-specific instructions of any kind",
             ],
             "model": GEM, "route": "reasoning effort low, max_tokens 1000 (unchanged)",
             "requests": 32},
        ],
        "total_new_provider_requests": 32,
        "adapter_version": ad.adapter_version,
        "schema_sha256": base["schema_sha256"],
        "task": "ocr_primary", "rag": "DISABLED",
    },

    "estimated_cost": {
        "basis": "the control arm's measured cost per attempted crop under the identical route",
        "gemini_per_attempt_usd": round(res["accounting"]["gemini_cost_usd"] / 32, 8),
        "estimated_total_usd": round(res["accounting"]["gemini_cost_usd"], 8),
        "worst_case_predicted_usd": 0.15,
        "note": ("if the new prompt REDUCES content-filter outcomes it will cost MORE than the "
                 "control, because filtered rows are free. A higher bill is a positive signal."),
    },

    "metrics": {
        "primary": "hard-failure count / 32 (control: 18/32)",
        "primary_secondary": "content-filter count / 32 (control: 10/32)",
        "quality_guard": "successful-only mean CER (control: 0.1155)",
        "reported_the_same_way": ("the full 12-axis outcome taxonomy, coverage stated as a fraction "
                                  "beside every successful-only metric, and a one-sided 95% upper "
                                  "bound on the failure rate"),
    },

    "advancement_and_drop_rules_stated_in_advance": {
        "ADOPT_and_rerun_paired": ("hard failures <= 4/32 AND successful-only CER <= 0.20. The prompt "
                                   "was a cause; adopt ocr-neutral-v2 and re-run the paired "
                                   "comparison under it."),
        "DROP_gemini_as_primary": ("hard failures >= 10/32. The prompt is NOT the cause; Gemini's "
                                   "filter behaviour on this content is intrinsic at this price "
                                   "point. Move to option D — a different vision/OCR provider — "
                                   "under a new freeze."),
        "REPORT_ONLY": "anything between 5 and 9 hard failures is reported, not resolved.",
        "quality_veto": ("if successful-only CER exceeds 0.20 the treatment is rejected regardless of "
                         "coverage: a prompt that reads more crops but reads them worse is not a win."),
    },

    "prohibitions": [
        "no CALIBRATION crops", "no HELD_OUT", "no grading of any kind", "no OCR verification",
        "no RAG", "no second model in this arm", "no change to audited references or crops",
        "the control arm is REUSED and must never be re-executed",
        "the new prompt must be frozen and hash-recorded BEFORE the first call",
    ],
}
body = json.dumps(art, ensure_ascii=False, indent=1, sort_keys=True, default=str)
art["experiment_sha256"] = hashlib.sha256(body.encode()).hexdigest()
p = EXP / "OCR_PROMPT_V2_NEUTRAL_FRAMING_2026-09-02.json"
p.write_text(json.dumps(art, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
print("wrote", p)
print("experiment_sha256:", art["experiment_sha256"])
print("requests:", art["design"]["total_new_provider_requests"],
      "| est cost:", art["estimated_cost"]["estimated_total_usd"])
