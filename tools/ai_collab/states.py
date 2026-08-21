"""Run states, verdicts and vocabulary for the bounded collaboration loop.

The run state machine:

    CREATED -> CLAUDE_RUNNING -> (AWAITING_CLAUDE)        manual Claude mode
            -> [tests]        -> (AWAITING_REVIEW_APPROVAL)   manual mode gate
            -> REVIEWING      -> APPROVED | BLOCKED
                               | CHANGES_REQUIRED -> (AWAITING_FIX_APPROVAL)
                                                  -> next round or MAX_ROUNDS

Pause states wait for an explicit user action (`approve` / `continue`).
Terminal states end the run; `final.json` is written exactly once.
"""

# --- run states ------------------------------------------------------------
CREATED = "CREATED"
CLAUDE_RUNNING = "CLAUDE_RUNNING"
AWAITING_CLAUDE = "AWAITING_CLAUDE"  # manual Claude mode: waiting for handoff
AWAITING_REVIEW_APPROVAL = "AWAITING_REVIEW_APPROVAL"  # manual: gate before reviewer
REVIEWING = "REVIEWING"
AWAITING_FIX_APPROVAL = "AWAITING_FIX_APPROVAL"  # manual/semi_auto: gate before fixes
USER_APPROVAL_REQUIRED = "USER_APPROVAL_REQUIRED"  # Claude escalated an out-of-scope issue

# --- terminal states (explicit stop states, section 7 of the spec) ---------
APPROVED = "APPROVED"
CHANGES_REQUIRED = "CHANGES_REQUIRED"  # run ended while changes were still required
BLOCKED = "BLOCKED"
MAX_ROUNDS = "MAX_ROUNDS"
TEST_FAILURE = "TEST_FAILURE"
BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
ERROR = "ERROR"
STOPPED = "STOPPED"  # stopped by the user with no pending verdict

TERMINAL_STATES = frozenset(
    {
        APPROVED,
        CHANGES_REQUIRED,
        BLOCKED,
        MAX_ROUNDS,
        TEST_FAILURE,
        BUDGET_EXHAUSTED,
        ERROR,
        STOPPED,
    }
)

# States in which the orchestrator waits for the user. USER_APPROVAL_REQUIRED
# doubles as a stop state: `approve` resumes the run; `stop` finalizes the run
# with final_state USER_APPROVAL_REQUIRED recorded in final.json.
PAUSE_STATES = frozenset(
    {
        AWAITING_CLAUDE,
        AWAITING_REVIEW_APPROVAL,
        AWAITING_FIX_APPROVAL,
        USER_APPROVAL_REQUIRED,
    }
)

# Live (non-final) states a stored run may legitimately be in.
ACTIVE_STATES = frozenset({CREATED, CLAUDE_RUNNING, REVIEWING}) | PAUSE_STATES

# --- reviewer verdicts -----------------------------------------------------
V_APPROVED = "APPROVED"
V_CHANGES_REQUIRED = "CHANGES_REQUIRED"
V_BLOCKED = "BLOCKED"
VERDICTS = frozenset({V_APPROVED, V_CHANGES_REQUIRED, V_BLOCKED})

# --- Claude handoff statuses ----------------------------------------------
H_READY = "READY_FOR_REVIEW"
H_BLOCKED = "BLOCKED"
H_USER_APPROVAL = "USER_APPROVAL_REQUIRED"
HANDOFF_STATUSES = frozenset({H_READY, H_BLOCKED, H_USER_APPROVAL})

SEVERITIES = ("critical", "high", "medium", "low")

MODES = ("manual", "semi_auto", "auto_bounded")
