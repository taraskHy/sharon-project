"""Mission I/L/M/N: failure-mode inventory, next experiment pre-registration,
shipment readiness and the GREEN/YELLOW/RED checklist. ZERO inference."""
import hashlib, json, subprocess, time
from pathlib import Path

R = Path("evaluation/model_selection/runs/ocr_primary")
EXP = Path("evaluation/model_selection/experiments")
res = json.loads((R / "OCR_SEEN32_PAIRED_RESULT_2026-09-02.json").read_text(encoding="utf-8"))
GEM, SON = "google/gemini-3.7-flash", "anthropic/claude-sonnet-5"
git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

# ---------------- I. failure-mode inventory --------------------------------
FM = [
 # failure, detection, handling, user-visible, silent-wrong-grade?, tested, blocking
 ("OCR provider unavailable (HTTP/transport)", "BackendError; classified provider_other_http_failure",
  "case fails; fallback policy routes to secondary; never silently empty", "crop unresolved -> human review",
  "NO", "yes (taxonomy + fallback stress)", "no"),
 ("OCR content filter", "finish_reason=content_filter -> BackendError",
  "hard-failure trigger; secondary attempted; else unresolved", "crop unresolved -> human review",
  "NO", "yes (46 stress tests)", "no"),
 ("OCR model-text refusal ([unreadable])", "bare marker vs readable reference",
  "hard-failure trigger; NOT counted as coverage", "crop routed to fallback/review",
  "NO", "yes", "no"),
 ("OCR truncation at max_tokens", "finish_reason=length -> BackendError",
  "hard-failure trigger; observed 2/32 even at 1000", "crop unresolved -> review",
  "NO", "yes (propagation + taxonomy)", "no"),
 ("OCR JSON parse / schema failure", "pydantic validation; classify_no_output",
  "hard-failure trigger; distinguished from provider failure", "crop unresolved -> review",
  "NO", "yes (failure taxonomy, 15 tests)", "no"),
 ("OCR line loss (no usable text)", "usable_transcription_returned False",
  "total_line_loss; never enters coverage numerator", "crop unresolved -> review",
  "NO", "yes", "no"),
 ("OCR wrong digit / sign / operator / negation", "deterministic comparison vs audited reference "
  "(EVALUATION ONLY - no reference exists in production)",
  "**NOT DETECTABLE IN PRODUCTION**", "silently wrong transcription feeds the grader",
  "**YES**", "measured offline only", "**YES - BLOCKER**"),
 ("OCR fabrication (fluent unrelated text)", "human adjudication only; never inferred",
  "no automatic detection", "silently wrong transcription", "**YES**", "one case adjudicated historically",
  "**YES - BLOCKER**"),
 ("Local grader unavailable", "gateway/backend error on the local route",
  "job fails loudly; no partial grade written", "job error surfaced", "NO", "yes", "no"),
 ("Malformed GradeResult", "grade-validation-v2 structural checks",
  "validation_ok False; decision withheld", "case routed to REVIEW", "NO", "yes", "no"),
 ("Evidence mismatch / ungrounded credit", "evidence_verified vs met_ids; evidence_fabricated",
  "risk engine structural gate -> REVIEW", "case routed to REVIEW", "NO", "yes", "no"),
 ("Stale model result", "STALE_MODEL_OUTPUTS tracking; revision-chain-aware verifiers",
  "stale rows invalidated, reruns owner-gated", "case excluded until rerun", "NO", "yes", "no"),
 ("Source hash mismatch (crop/reference drift)", "sha256 recomputation vs frozen manifest",
  "load_manifest / load_subset raise", "run refuses to start", "NO", "yes", "no"),
 ("Risk policy hash mismatch", "matrix_sha256 in every decision record",
  "decision refuses on unknown matrix", "job error", "NO", "yes", "no"),
 ("DB unavailable", "SQLite open/WAL errors", "request fails loudly", "user sees an error",
  "NO", "yes (db tests)", "no"),
 ("Backup failure", "WAL backup tests", "surfaced; no silent success", "admin-visible",
  "NO", "yes (backup tests)", "no"),
 ("Duplicate submission", "id uniqueness constraints", "rejected/deduped", "user informed",
  "NO", "yes", "no"),
 ("Concurrent review", "row locking / concurrency tests", "serialized", "second reviewer sees fresh state",
  "NO", "yes (concurrency tests)", "no"),
 ("Cloud-grading attempt", "cloudboundary task allowlist (two-layer)",
  "CloudBoundaryError before serialization", "job refuses", "NO", "yes (34 boundary tests)", "no"),
 ("Unknown cloud task", "deny-by-default allowlist", "refused", "job refuses", "NO", "yes", "no"),
 ("API usage-ledger failure", "ledger append per call; reconciliation vs account",
  "run-level reconciliation reported", "admin-visible", "NO", "yes (accounting tests)", "no"),
]
blockers = [f for f in FM if f[6].startswith("**YES")]

# ---------------- L. next experiment ---------------------------------------
hw = res["stratified"]
nxt = {
    "experiment": "OCR_PROMPT_V2_VS_M2STRICT_GEMINI_SONNET_SEEN32",
    "status": "PRE-REGISTERED DESIGN, NOT EXECUTED, NOT AUTHORIZED",
    "chosen_option": "E - a new OCR prompt version under a fresh freeze",
    "why_this_and_not_the_others": {
        "A_remaining_21_calibration_gemini_only": (
            "rejected: it would spend the CALIBRATION half of the seen corpus to re-measure a "
            "failure rate already bounded at 56% (95% UB 71%) on 32 crops. It buys precision on a "
            "number that is already decisive and burns a population we only get to use once."),
        "B_remaining_21_paired": "rejected for the same reason, at double the cost",
        "C_gemini_primary_sonnet_fallback_only": (
            "rejected as an EXPERIMENT because that is a deployment configuration, not a question. "
            "We just measured it: 29/32 coverage, failure-aware CER 0.3560. Running it again on new "
            "crops tells us little that this run did not."),
        "D_different_provider": (
            "a real option and the strongest alternative. Held back only because we have not yet "
            "established that the CURRENT prompt is not itself a cause of the failures - Gemini's "
            "content filter fires on 50% of cell crops under a prompt that says 'Ignore any red "
            "instructor ink' and describes an exam. Changing provider before ruling that out risks "
            "chasing the wrong variable."),
    },
    "hypothesis": (
        "Gemini's content-filter rate is influenced by the exam framing in the frozen m2-strict-v1 "
        "prompt. A minimal, neutrally-worded transcription prompt with identical output contract may "
        "reduce hard failures without changing what is asked of the model."),
    "design": {
        "population": "the same frozen 32-crop seen46_ocr_dev subset, unchanged",
        "arms": [
            {"arm": "control", "prompt": "m2-strict-v1 (frozen)", "models": [GEM],
             "note": "already measured; REUSED from this run, not re-executed"},
            {"arm": "treatment", "prompt": "ocr-neutral-v2 (to be written and frozen BEFORE any call)",
             "models": [GEM], "requests": 32},
        ],
        "requests": 32,
        "estimated_cost_usd": 0.15,
        "primary_metric": "hard-failure count / 32 under the new prompt vs 18/32 under m2-strict-v1",
        "secondary": "successful-only CER must not regress beyond 0.20 (control 0.1155)",
        "decision_rule_stated_in_advance": (
            "if hard failures fall to <= 4/32 AND successful CER stays <= 0.20, adopt the new prompt "
            "and re-run the paired comparison. If hard failures stay >= 10/32, the prompt is not the "
            "cause: drop Gemini as primary and move to option D (a different vision/OCR provider). "
            "Anything between is reported, not resolved."),
        "prohibitions": ["no CALIBRATION crops", "no HELD_OUT", "no grading", "no RAG",
                         "no change to audited references or crops",
                         "the control arm is reused, never re-executed"],
    },
    "why_it_is_cheap_and_decisive": (
        "32 requests, ~$0.15, one variable changed, against a control we already own. It either "
        "rescues the best-quality reader in the comparison or eliminates the prompt as a cause and "
        "sends us to a different provider with evidence."),
}

# ---------------- M. shipment readiness ------------------------------------
ship = {
  "human_in_the_loop": {
    "pipeline": "OCR -> local grader -> risk routing -> AUTO proposals -> REVIEW for uncertain",
    "what_can_ship_now": [
        "the deterministic MC / variant path (no LLM in the loop)",
        "the cloud boundary (two-layer, 34 tests, deny-by-default)",
        "immutable transcription storage and the audit trail",
        "the human review site and adjudication flow",
        "cost accounting with exact ledger/account reconciliation",
    ],
    "blockers": [
        "OCR quality: the best DEPLOYABLE configuration reaches failure-aware CER 0.3560 on "
        "handwriting. A third of crops still need a human to read them.",
        "No end-to-end run has ever fed NEW OCR output into the grader; the grading corpus was built "
        "from frozen evidence. The OCR->grader integration is untested on live OCR.",
        "OCR digit/sign/negation errors are undetectable in production (no reference exists there) "
        "and can silently change an answer's meaning.",
    ],
    "estimated_review_workload": (
        "on this evidence, with the Gemini->Sonnet policy: 3 of 32 crops unresolved (9%) plus 15 "
        "fallback rows flagged for review (47%) = 18 of 32 crops touched by a human, ~56%. That is "
        "not a labour saving yet."),
    "ocr_cost_per_100_exams": {
        "at_5_crops": round(res["cost_projections"]["prospective_fallback_composite"]["projections"]["100_exams_at_5"], 4),
        "at_10_crops": round(res["cost_projections"]["prospective_fallback_composite"]["projections"]["100_exams_at_10"], 4),
        "at_15_crops": round(res["cost_projections"]["prospective_fallback_composite"]["projections"]["100_exams_at_15"], 4),
        "note": "composite fallback rate; trivial next to the human review cost it does not yet remove",
    },
    "engineering_remaining": [
        "wire the OCR fallback policy into the production path (currently experiment-only)",
        "build the crop -> grading-case evidence mapping as a contract, not an id convention",
        "one end-to-end run on new OCR output with a persisted grader result",
    ],
    "evaluation_remaining": [
        "the prompt-v2 experiment above (32 requests)",
        "a grader run consuming real OCR output, to measure how OCR error propagates to grades",
    ],
    "realistic_eta_working_days": {
        "engineering": "5-8 days",
        "evaluation": "3-5 days (mostly owner-gated authorizations and review time)",
        "total_to_a_defensible_human_in_the_loop_pilot": "8-13 working days",
        "confidence": "moderate; the unknown is how much OCR error the grader tolerates, which has "
                      "never been measured",
    },
  },
  "high_automation": {
    "goal": "minimal human review, very low false-full risk, bounded undergrading",
    "blockers": [
        "OCR is the binding constraint, not the grader. Failure-aware CER 0.3560 on handwriting "
        "cannot support unreviewed automation.",
        "Silent OCR digit/sign/negation errors have no production detector.",
        "Grader quality: the frozen risk work found no arm beating always-partial (43 vs 40-43) and "
        "the candidate rule reached AUTO 67.4% with false-full 0 on a 46-case sample.",
        "The invalid-case class is structurally unmeasurable on this dataset (every zero-score DEV "
        "case had a wrong selection), so credit-withholding on a correct choice is unvalidated.",
        "0/5 false-full observations bound the rate only below ~45% - nowhere near a safety claim.",
    ],
    "work_remaining": [
        "an OCR configuration that reaches usable coverage well above 90% with critical-error rates "
        "an order of magnitude below today's",
        "a much larger labelled population for the rare-event classes",
        "a production detector for OCR semantic corruption, or a second independent read",
    ],
    "realistic_eta_weeks": {
        "estimate": "not achievable on the current evidence base; 8-16 weeks minimum IF an OCR "
                    "configuration materially better than anything measured so far is found",
        "honest_statement": ("no ETA here is trustworthy. The blocker is not engineering time, it is "
                             "that no OCR configuration measured to date is good enough, and we do "
                             "not yet know one exists at this price point."),
    },
  },
}

# ---------------- N. GREEN/YELLOW/RED --------------------------------------
CHECK = [
 ("deterministic MC path", "GREEN", "deterministic; no LLM; covered by tests", "none", "keep as is"),
 ("variant detection", "GREEN", "deterministic; frozen dataset of 16", "none", "keep as is"),
 ("cloud boundary", "GREEN", "two-layer authorization; 34 tests; --research no longer bypasses "
  "content safety; verified on the wire for all 64 payloads", "none", "keep as is"),
 ("OCR quality", "RED", "best deployable failure-aware CER 0.3560 on 32 handwritten crops",
  "no configuration is close to unreviewed use", "run the prompt-v2 experiment"),
 ("OCR reliability", "RED", "Gemini 18/32 hard failures; Sonnet 27/32 usable but CER 0.4718",
  "neither model alone is viable; composite reaches 29/32 with 56% human touch",
  "prompt-v2, then a different provider if that fails"),
 ("immutable transcription", "GREEN", "content-hashed artifacts; append-only runs", "none", "keep"),
 ("local grading infrastructure", "GREEN", "runs offline; validation-v2; no cloud path", "none", "keep"),
 ("semantic grading quality", "YELLOW", "no arm beats always-partial on the frozen 46-case set",
  "quality plateau unresolved", "revisit after OCR is fixed - grading on bad OCR is untestable"),
 ("evidence grounding", "YELLOW", "structural checks exist; fabricated-evidence detection in place",
  "not validated against real OCR output", "include in the E2E integration run"),
 ("risk policy", "YELLOW", "risk-engine-v1 SHADOW_READY; matrix hash-pinned; 0/5 false-full bounds "
  "only below ~45%", "sample far too small for a safety claim", "needs a much larger labelled set"),
 ("human review UI", "GREEN", "shared blind review site shipped and used for the 92/92 campaign",
  "none", "keep"),
 ("concurrency", "GREEN", "exhaustive state-space/fuzz/concurrency tests", "none", "keep"),
 ("backup/recovery", "GREEN", "WAL backup tests", "none", "periodic restore drill"),
 ("audit trail", "GREEN", "append-only, hash-chained artifacts throughout", "none", "keep"),
 ("cost accounting", "GREEN", "ledger reconciles to the provider account exactly (rounding 0.0) "
  "across four consecutive experiments", "none", "keep"),
 ("failure handling", "YELLOW", "12-axis OCR taxonomy; fallback fails closed; 46 stress tests",
  "OCR semantic corruption has no production detector", "design a detector or a second read"),
 ("HELD_OUT readiness", "RED", "untouched by design; final-eval path only",
  "nothing may be evaluated on it until OCR and grading are settled", "do not touch"),
 ("packaging/deployment", "YELLOW", "CLI + local app work; no deployment automation",
  "no release pipeline", "out of scope until OCR is resolved"),
]

art = {
    "artifact": "ocr_shipment_readiness_package", "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "git_commit": git, "provider_calls": 0,
    "failure_mode_inventory": {
        "columns": ["failure", "detection", "current handling", "user-visible outcome",
                    "silent wrong grade possible?", "tested?", "blocking?"],
        "rows": FM,
        "release_blockers": [f[0] for f in blockers],
        "blocker_rationale": ("both blockers are the same underlying fact: an OCR transcription can "
                              "be confidently wrong in a way that changes meaning, and in production "
                              "there is no reference to compare against, so nothing detects it."),
    },
    "next_experiment": nxt,
    "shipment_readiness": ship,
    "release_checklist": {"columns": ["component", "status", "evidence", "blocker", "next action"],
                          "rows": CHECK,
                          "counts": {"GREEN": sum(1 for c in CHECK if c[1] == "GREEN"),
                                     "YELLOW": sum(1 for c in CHECK if c[1] == "YELLOW"),
                                     "RED": sum(1 for c in CHECK if c[1] == "RED")}},
}
body = json.dumps(art, ensure_ascii=False, indent=1, default=str)
art["content_sha256"] = hashlib.sha256(body.encode()).hexdigest()
p = R / "OCR_SHIPMENT_READINESS_2026-09-02.json"
p.write_text(json.dumps(art, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
print("wrote", p)
print("release blockers:", art["failure_mode_inventory"]["release_blockers"])
print("checklist:", json.dumps(art["release_checklist"]["counts"]))
print("next experiment:", nxt["experiment"], "-", nxt["chosen_option"])
