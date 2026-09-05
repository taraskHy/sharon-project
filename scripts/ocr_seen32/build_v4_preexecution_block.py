"""V4 blocked before any paid call: the unverified-slug resolution procedure is
not preregistered. ZERO provider calls, ZERO metadata calls."""
import json, hashlib, pathlib, subprocess, time

R = pathlib.Path("evaluation/model_selection/runs/ocr_primary")
E = pathlib.Path("evaluation/model_selection/experiments")
V4 = json.loads((E / "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V4_2026-09-05.json").read_text(encoding="utf-8"))
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
ts = time.strftime("%Y-%m-%d %H:%M:%S")
led = [json.loads(l) for l in pathlib.Path("evaluation/model_selection/state/gateway_ledger/usage.jsonl"
                                           ).read_text(encoding="utf-8").splitlines() if l.strip()]
cum = round(sum(float(r.get("reported_cost") or 0) for r in led
                if r.get("cloud") and not r.get("cache_hit")), 8)

doc = {
 "artifact": "ocr_altscreen_v4_preexecution_block",
 "created_at": ts, "git_commit": commit,
 "provider_calls": 0, "metadata_calls": 0, "additional_spend_usd": 0.0,
 "experiment": "OCR_ALTERNATIVE_CANDIDATE_SCREEN_V4",
 "experiment_sha256": V4["experiment_sha256"],
 "outcome": "BLOCKED BEFORE EXECUTION — V4 REQUIRES IMMUTABLE CLOSURE AND A SUCCESSOR (V5)",
 "v4_not_amended": True, "v4_not_executed": True,
 "budget_manifest_created": False,
 "metadata_get_performed": False,

 "blocker": {
   "rule_applied": ("the authorization states: 'First inspect V4 and confirm it already "
                    "preregisters the exact procedure for resolving an unverified provider slug "
                    "from a newly captured OpenRouter provider catalogue. If that procedure is "
                    "absent or ambiguous, do not amend V4. Make zero paid calls and report that "
                    "V4 requires immutable closure and V5.'"),
   "finding": "ABSENT. V4 carries a one-sentence CAVEAT, not a procedure.",
   "what_v4_actually_says": V4["provider_mapping_caveat"],
   "word_frequency_in_v4": {"procedure": 0, "acceptance": 0, "prerequisite": 1},
   "what_a_procedure_would_have_had_to_freeze_and_does_not": [
     "the exact endpoint to request",
     "the fields that must be preserved (URL, retrieval timestamp, HTTP status, exact raw body, "
     "raw-body SHA-256, sanitized headers, parsed provider entries)",
     "the acceptance criteria: slug exists; display-name mapping is unique; the frozen endpoint "
     "remains available; pricing does not exceed the frozen conservative cost; the Google "
     "mappings remain compatible with the preserved frozen mapping",
     "the action on absence, ambiguity, rename without a unique mapping, or higher pricing",
     "whether resolving the slug is permitted at all without amending the frozen experiment",
   ],
   "why_this_is_not_a_formality": ("without frozen acceptance criteria, whatever a catalogue "
                                   "returned would be judged after seeing it. That is the "
                                   "post-hoc decision-making every freeze in this project exists "
                                   "to prevent, and it is how a 'prerequisite' quietly becomes a "
                                   "rubber stamp."),
 },

 "everything_verified_before_the_block": {
   "starting_state_checks": "19/19 PASS",
   "branch": "initial-prototype",
   "head": "fe78907a3b7a3e31e25006ba2e216ea87093e4e3",
   "worktree": "clean",
   "v4_self_hash_recomputes": True,
   "v3_closure_sha256": "52159942ffefeee8f533bbe7c1fb3ba2c978f270611e6260e9f7fbaffdde3864",
   "v1_v2_v3_artifacts_immutable": True,
   "ledger": {"rows": len(led), "usd": cum},
   "family_limits": {"warning": 0.78323229, "hard": 0.82323229},
   "held_out_access": 0,
   "cache_identity_version": 4,
   "offline_verifier": "100/100 PASS",
   "credential": "present and non-empty; never printed, read, hashed or exposed",

   "identity_verification_offline": {
     "experiment_identities_matched": "3/3",
     "semantic_identities_matched": "24/24 — every case-arm request, not only the three first cases",
     "ai_studio": "749d7c41f17c127e932fc55d347f86ed8d33dee7b26511fecebe5aa858af1cfa",
     "vertex": "9a912512cd1e99de7fb7e3d00fb4fb0be072ace0c70eeff821b60e99fc1091ed",
     "qwen_alibaba": "b22a1a20dc55809957d096aeb554caf4df55857bafd2045f9b4b1a5ec8ad41f7",
     "path": "CLI parser -> build_route() -> to_backend_config() -> payload builder, MockTransport",
   },

   "qwen_payload_inspection": {
     "max_tokens": 1000,
     "reasoning_field": "ABSENT — the key is not transmitted at all, not an object with a value",
     "model": "qwen/qwen3-vl-235b-a22b-instruct",
     "schema_name": "BenchTranscription",
     "provider_order": ["alibaba"],
     "allow_fallbacks": False,
     "transport_retries": 0,
     "verdict": ("the declared candidate_override works: the pre-inference HTTP 400 shape that "
                 "destroyed the Stage-1 Gemini arm is not present"),
   },
 },

 "state_unchanged": {
   "ledger_rows": len(led), "ledger_usd": cum,
   "spend_this_task_usd": 0.0,
   "held_out_access": 0, "grading_calls": 0, "rag_calls": 0,
   "v4_budget_manifest": "NOT CREATED",
   "autograder_code_changed": False,
 },

 "required_before_any_v4_family_execution": [
   "close V4 immutably as SUPERSEDED_BEFORE_EXECUTION (it was never executed; its hash is "
   "preserved)",
   "freeze V5 carrying V4's design unchanged plus a PREREGISTERED unverified-slug resolution "
   "procedure with explicit acceptance criteria and failure actions",
   "OR, if the owner prefers, freeze V5 with the Qwen arm removed, which needs no catalogue "
   "capture at all — the two Gemini route-attribution arms are already fully resolvable from the "
   "preserved mapping",
 ],
 "not_done_because_unauthorized": ["V5 creation", "any V4 amendment", "the metadata GET",
                                   "the budget manifest", "any provider call"],
 "ocr_primary_role_status": "UNSELECTED (unchanged)",
}
body = json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True, default=str)
doc["content_sha256"] = hashlib.sha256(body.encode()).hexdigest()
p = R / "OCR_ALTSCREEN_V4_PREEXECUTION_BLOCK_2026-09-05.json"
p.write_text(json.dumps(doc, ensure_ascii=False, indent=1, default=str), encoding="utf-8", newline="\n")
print("wrote", p)
print("content_sha256:", doc["content_sha256"])
print("ledger unchanged:", len(led), cum)
