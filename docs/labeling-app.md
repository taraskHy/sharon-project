# Shared ground-truth grading app (`labeling_app`)

A small, independent **human** labeling website for creating the ground-truth
grades of the GRADE_PRIMARY / GRADE_ESCALATE benchmark. It runs on the owner's
PC, keeps a local SQLite database, and is exposed *temporarily* through a
Cloudflare Tunnel so friends can grade from any browser. It is **not** the
production student-grading interface and it never calls any model
(OpenRouter / Gemini / Claude / GPT / Qwen / OCR calls = 0 — enforced by
`tests/test_labeling_app.py::test_labeling_app_has_no_ai_or_pipeline_dependency`).

```
strong PC:  labeling_app (Starlette ASGI on uvicorn, 127.0.0.1:8787)
            + labels.db (SQLite, WAL)           %LOCALAPPDATA%\autograder\labeling\
            + anonymized frozen bundle          %LOCALAPPDATA%\autograder\labeling\bundle\
            + cloudflared quick tunnel  ->  https://<random>.trycloudflare.com
friends:    browser only (no repo, Python, VPN, models, credentials)
```

Note: FastAPI is not installed in this environment; the app is written
directly on **Starlette** (FastAPI's ASGI core, already installed) and served
by uvicorn — same deployment shape; switching to FastAPI later is mechanical.

## Data location (keep OUT of OneDrive)

| path | what |
|---|---|
| `%LOCALAPPDATA%\autograder\labeling\labels.db` (+ `-wal`, `-shm` while running) | the live database — authoritative while the server runs |
| `…\labeling\bundle\` | anonymized item bundle (`bundle.json`, `items.json`, `images/`, `pages/` (red-ink-masked source pages), `private/id_map.json`, `private/provenance.json`) |
| `…\labeling\exports\final_labels.json` | last export of FINAL labels |
| `…\labeling\backups\<stamp>\` | snapshots (`labels.db` copy via the SQLite backup API + `final_labels.json` + manifest) |

Override the root with `LABELING_DATA_DIR` or `--data-dir`. Never point the
live database at a OneDrive folder; back up a *closed snapshot* into OneDrive
with `backup --copy-to <OneDrive folder>` instead.

## Commands

```powershell
# once: build the anonymized bundle from the frozen grade_primary dataset (67 cases, 82 images)
.\.venv\Scripts\python.exe -m labeling_app build-bundle

# run locally (grader page http://127.0.0.1:8787/ , admin page http://127.0.0.1:8787/admin)
.\.venv\Scripts\python.exe -m labeling_app serve --port 8787
#   optional: --admin-key SOMETHING  (then admin = /admin?key=SOMETHING)

# export FINAL labels / backup / status
.\.venv\Scripts\python.exe -m labeling_app export
.\.venv\Scripts\python.exe -m labeling_app backup [--copy-to "C:\Users\...\OneDrive\labeling-backups"]
.\.venv\Scripts\python.exe -m labeling_app status

# bring FINAL labels into the benchmark (the pipeline consumes ONLY these)
.\.venv\Scripts\python.exe -m autograder bench import-final-labels --role grade_primary --export "%LOCALAPPDATA%\autograder\labeling\exports\final_labels.json"
```

## Cloudflare Tunnel (not started yet — do this only when you want a session)

1. `winget install Cloudflare.cloudflared` (once).
2. Start the app: `python -m labeling_app serve --port 8787` (binds 127.0.0.1 only).
3. In a second terminal: `cloudflared tunnel --url http://127.0.0.1:8787`
   — cloudflared prints an `https://<random-words>.trycloudflare.com` URL.
4. Send that URL (and `/admin?key=…` only to yourself) to the graders.
5. Graders open it in a browser, type their name, grade.
6. End the session: `Ctrl+C` in the cloudflared terminal (the URL dies), then stop the app.
   Run `python -m labeling_app backup` afterwards.
No router port is opened, no public IP is needed, the PC is never exposed
directly; quick tunnels are unauthenticated by design — the URL is the
credential (privacy was declared a non-priority; set `--admin-key` so only
you reach the admin page).

## Which cases reach the graders (eligibility)

Humans grade an explanation ONLY when the exam's grading policy actually
requires the explanation to be scored. The single source of truth is
`autograder.eligibility` (a thin wrapper over the production policy gate
`policies.decide_before_ocr`; the labeling app never re-implements policy
semantics). Facts come from the dataset case records — the student's selected
MC option, the key's `correct_by_version`, the exam version, the canonical
grading policy and its wrong-answer rule — never from filenames, historical
totals or instructor red marks.

* Confidently wrong MC under `wrong_choice_zero`, or under
  `explanation_required_if_correct` with a zero/selection wrong-answer rule
  → **deterministic score 0**: the case is excluded from the bundle at build
  time (recorded in `bundle/private/excluded.json` and in the bundle's
  `eligibility` counts) and the server refuses label submissions for it.
* `explanation_can_rescue_wrong_choice`, `choice_and_explanation_independent`,
  wrong-answer rule `process`, unresolved/ambiguous MC, or no MC observation
  at all (the current explanation-only grade_primary cells) → **human label**.
  An ambiguous or absent MC never yields a deterministic zero.
* `choice_only` → no explanation component exists; never in the queue.

Enforcement is layered: `build-bundle` excludes ineligible cases from the
human workload; `serve`/`export`/`status` recompute eligibility from the
dataset (`--dataset`, default: the repo grade_primary) so even a STALE bundle
fails safely — `claim_next` skips such items, label submission and admin
FINAL return HTTP 400, and workload/progress counts include only genuinely
human-labelable items. Existing labels on a case that later becomes
ineligible are never deleted: they surface as `INELIGIBLE` /
"obsolete" in the admin summary, the export marks the item
`eligible_for_human_label: false`, and `bench import-final-labels` refuses to
promote it to ground truth (recorded under `ignored_ineligible` in
`final_labels.json`). The deterministic policy result stays authoritative;
GRADE_PRIMARY model accuracy is measured only on human-labelable cases.
Today's frozen grade_primary: 67 source cases, 67 human-labelable, 0
deterministic-zero (all cells are explanation-only under the independent
policy).

## Grader workflow

Name → item (question, rubric, official solution, answer image, frozen
transcription, max score) → score buttons (0.5 steps; keys: digits, `.`, `+`/`-`),
rubric items (`Q W E R T`), note → **Save & next** (`Enter`) / **Skip** (`S`) /
**Flag** (`F`). Progress bar shows the grader's own counts and the overall
coverage. "Review my skipped items" re-serves skipped ones. A grader resuming
sees only *their own* earlier answer.

## Source provenance (context, not ground truth)

Each item carries explicit provenance taken from the upstream records — never
reconstructed from the opaque id:

| field | source | shown to graders |
|---|---|---|
| exam (e.g. `003`) | writer code of the benchmark case (`e003` → exam 003) | yes |
| case id (e.g. `e003_q1_r1`) | dataset case id | yes |
| question / part (row) | dataset case id | yes |
| page number | e002 cells: `evaluation/hebrew_bench/crops_manifest.json`; e003–e007 lines: `evaluation/htr_pilot_sources.json` (q1 → page 11, q2 → page 12) | yes |
| line count | dataset `transcription_items` | yes |
| line bounding box on the page | **not recorded upstream → "unavailable"** | reported as unavailable |
| source PDF filename (`test/003_70.pdf`) | carries the instructor's **total grade** in its name → **private** (`bundle/private/provenance.json`, admin view only) | no |

The grader page shows `Source: Exam 003 · Question 1 · Part r1 · Page 11 ·
Case e003_q1_r1` under the crop and a **[View full source page]** link
(`/api/pages/<item>`). The full page is rendered locally from the PDF with the
instructor's red ink masked (dilated red-dominance mask → white: per-row
ticks/crosses, the question total and margin notes disappear); a page is
served only when ≤ 60 strict-red pixels remain, otherwise "full source page:
unavailable". The crop stays the primary grading evidence. The page never shows
expected labels, model output or other graders' scores (it is the student's
sheet with instructor marks removed).

`tests/test_labeling_provenance.py` proves correspondence: every item's
provenance equals the dataset case (exam/question/row/line count), the page
number comes from the named upstream record, the served page bytes equal a
fresh masked render of that (file, page) with zero strict-red pixels, and the
crop pixels are found inside the unmasked render of that page by normalized
cross-correlation (e002 cell 0.95 vs 0.48 on another exam's page; e003 line
0.56 vs 0.28).

## Multiple graders, agreement, adjudication, FINAL

- `labels` is keyed by `(item_id, grader)` — independent rows; nobody can
  overwrite someone else's label. Double-labeling policy (admin): none / all /
  selected items.
- Grader B never sees grader A's label before submitting (the grader API
  returns only the caller's own label; other labels exist only in admin APIs).
- Two saved labels: identical score + rubric → **AGREEMENT**, else
  **NEEDS_ADJUDICATION**; any flagged label → NEEDS_ADJUDICATION.
- **FINAL** is its own table with provenance (`source` agreement | adjudicated,
  contributing graders, their label revisions, adjudicator, timestamp). A single
  unfinished grader label is never treated as FINAL; the admin either
  "Finalize agreement" or adjudicates by hand (or confirms a single label —
  recorded as adjudicated). FINAL can be reopened.

## Concurrency

SQLite WAL, `busy_timeout` 15 s, short `BEGIN IMMEDIATE` write transactions,
one connection per thread. Optimistic concurrency: every label row has a
`revision`; the client sends `expected_revision` (what it loaded) — a mismatch
is HTTP 409 (`stale: true`) and the page reloads the item. Admin FINAL writes
carry `expected_item_revision` (the item's revision, bumped on every change).

## Export format (`final_labels.json`)

```json
{"schema_version": 1, "kind": "grade_primary_final_labels",
 "bundle_items_sha256": "…", "dataset_inputs_sha256": "…", "content_sha256": "…",
 "final_count": N, "exported_at": "…",
 "items": [{"item_id": "<dataset case id>", "display_id": "<opaque id>", "final_score": 3.0,
            "rubric_decisions": ["R1"], "note": "", "source": "agreement|adjudicated",
            "adjudicator": null, "contributing_graders": ["A","B"], "from_revisions": {"A":1,"B":1},
            "finalized_at": "…", "labels": [{"grader":"A","score":3.0,"rubric_decisions":[],"note":"",
                                             "status":"saved","revision":1,"updated_at":"…"}, …]}]}
```
Items and every list are sorted; `content_sha256` covers `items` so two exports
of the same state are identical. `bench import-final-labels` writes
`datasets/grade_primary/final_labels.json` (FINAL rows only); the manifest
loader merges it (`label_source = final:<source>`) and hashes it into the run
identity.

## Privacy

Item ids are opaque (`g` + 10 hex; the map to dataset case ids lives in
`bundle/private/id_map.json`, never served). Graders see question, rubric,
official solution, the answer crop (handwriting only), the transcription and
the max score — no names, student ids, filenames, splits, writer codes,
repository paths, model outputs, OCR confidence or expected labels. The
server only serves `images/<item_id>_<n>.png` inside the bundle root.

## Schema (SQLite)

`meta`, `items(item_id, max_score, rubric_ids, wanted_labels, revision)`,
`graders(name, created_at, last_seen)`, `claims(item_id, grader, claimed_at,
expires_at)`, `labels(item_id, grader, score, rubric, note, status
saved|skipped|flagged, flag_reason, revision, created_at, updated_at)`,
`final_labels(item_id, score, rubric, note, source, adjudicator,
contributing_graders, from_revisions, finalized_at, schema_version)`,
`events(append-only audit)`.
