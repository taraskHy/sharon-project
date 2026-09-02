# Release readiness — **SHADOW_READY** (2026-09-02 03:14:22)

Gate set `release-gates-asym-v2`; 5/12 gates pass. Production-ready is not a selectable status.

| gate | target | observed | limitation | verdict |
|---|---|---|---|---|
| HARD_FALSE_FULL | confirmed invalid -> automatic valid = 0 observed | 0 across every arm and policy (seen development data only (46 explanation cases)) | only 5 invalid cases: one-sided 95% upper bound 45.1% — observation passes, demonstrated safety does NOT | PASS |
| RARE_EVENT_POWER | enough invalid examples to bound the false-full rate at the owner's chosen threshold (e.g. 5% needs 59, 1% needs 299) | 5 invalid examples available | threshold choice is the owner's; every listed option needs more invalid examples than exist on seen data | FAIL |
| SERIOUS_OVERGRADE | automatic partially_valid -> valid <= 2/46 | prospective_valid_only 4, prospective_noninvalid 4 (the committed 2 was oracle-assisted) | seen development data only (46 explanation cases) | FAIL |
| WEIGHTED_RISK_VS_CONSTANTS | semantic-layer total risk <= 0.90 x best constant (always_partially_valid = 43) | baseline 43 | seen development data only (46 explanation cases) | FAIL |
| UNDERGRADE_CAP | automatic harmful undergrades <= 3/46 | valid_only 0, noninvalid 2 | seen development data only (46 explanation cases) | PASS |
| GROUNDING | evidence+schema failures <= 2% of cases | 2/46 = 4.3% | seen development data only (46 explanation cases) | FAIL |
| AUTOMATION_JOINT | AUTO coverage >= 70% AND weighted-risk gate passes | valid_only 58.7% / noninvalid 84.8%; weighted-risk gate fails | seen development data only (46 explanation cases) | FAIL |
| DISAGREEMENT_ROUTING_ONLINE | wide human disagreement routes to REVIEW in production | IMPOSSIBLE prospectively: reviewer disagreement does not exist at decision time; the engine refuses retrospective policies in production and the oracle tables are marked NOT DEPLOYABLE | needs a future PROSPECTIVE ambiguity signal to recover the oracle gains | FAIL |
| ENGINE_SHADOW_SAFETY | OFF/SHADOW never change the active grade; ACTIVE locked; typed refusals everywhere | 1,152-state exhaustive suite + fuzz + concurrency all green; no production caller of ACTIVE exists | engineering property, fully testable | PASS |
| REPRODUCTION | every load-bearing committed number reproduces from raw artifacts | REPRODUCED: 58 checks, 0 failed | deterministic | PASS |
| OCR | production OCR validated separately before end-to-end shipping | campaign FROZEN (8ce4f5eea7b2…), NOT EXECUTED; 0 OCR calls | hard shipping blocker until run and passed | FAIL |
| FINAL_TEST | HELD_OUT untouched until grader+matrix+policy+OCR frozen | untouched; 0 exposure in every artifact of this campaign | - | PASS |

## Why SHADOW_READY and nothing more

- **SHADOW_READY_criteria**: "engine implemented + OFF/SHADOW proven inert + ACTIVE locked + reproduction clean + HELD_OUT sealed + zero observed false-full"
- **not_READY_FOR_OCR_VALIDATION_because**: ["two named engineering prep items in the OCR freeze are open (seen46-ocr subset registration; per-writer WER scoring)", "owner spend authorization for the campaign is not given"]
- **not_READY_FOR_FINAL_VALIDATION_because**: ["semantic layer does not beat always_partially_valid on weighted risk", "grounding failures 4.3% > 2%", "serious-overgrade gate fails prospectively (4 > 2)", "rare-event power gate fails (5 invalid examples)", "OCR unvalidated", "decision policy not yet frozen"]

Recommended shadow candidate: prospective_noninvalid_v1 (84.8% coverage, risk 34) with prospective_valid_only_v1 (58.7%, risk 20) as the conservative alternative; run BOTH in shadow and decide on shadow evidence
