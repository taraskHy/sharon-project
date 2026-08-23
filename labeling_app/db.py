"""SQLite store for the shared labeling app.

Design:
* ONE local SQLite file (WAL mode, busy timeout, short write transactions
  with BEGIN IMMEDIATE) — authoritative while the server runs.
* Independent labels: PRIMARY KEY (item_id, grader) — a grader can never
  overwrite another grader's label by construction.
* Optimistic concurrency: every label row and every item carries a
  ``revision``; a save must present the revision it loaded
  (``expected_revision``); a mismatch raises StaleWrite (HTTP 409) — a stale
  save never silently overwrites a newer one.
* FINAL ground truth lives in its own table (``final_labels``) with
  provenance (source agreement|adjudicated, contributing graders, revisions,
  adjudicator, timestamp) — never a random individual label.
* Append-only ``events`` audit trail.
* Evidence fingerprints (schema v2): ``items.evidence_sha256`` is the
  fingerprint of exactly the answer crops the served bundle shows for the
  item; every label and every FINAL stores the fingerprint that was current
  when it was written. When a rebuilt bundle changes an item's evidence the
  item's fingerprint moves (``sync_evidence``) and the labels/FINAL written
  against the old evidence become visibly STALE: they are never deleted, never
  counted as fresh labels, never finalized from agreement, never exported as
  ground truth — the grader (or an adjudicator) must re-review the corrected
  evidence and save again. Labels on items whose evidence did not change are
  untouched.

States (derived, never stored as truth):
    UNLABELED            no saved label
    ASSIGNED             active claim, no saved label
    LABELED              >=1 fresh saved label, fewer than wanted (or wanted=1, not final)
    AGREEMENT            >=2 fresh saved labels, all identical (score + rubric), not final
    NEEDS_ADJUDICATION   >=2 fresh saved labels that differ, or any fresh flagged label, not final
    NEEDS_REVIEW_AFTER_EVIDENCE_CHANGE
                         a saved/flagged label (or the FINAL) was made against evidence
                         that has since changed — re-review required
    FINAL                a fresh final_labels row exists
    INELIGIBLE           the grading policy decides the score; labels are obsolete
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import SCHEMA_VERSION

LABEL_STATUSES = ("saved", "skipped", "flagged")
FINAL_SOURCES = ("agreement", "adjudicated")
#: PROVENANCE of a label's score — how the number was DERIVED. Deliberately
#: independent of ``evidence_sha256``, which records WHICH evidence version was
#: on screen. The two answer different questions and must not be conflated:
#:  * human_independent_grading — the grader judged the student's answer from the
#:    evidence the app displayed. A later evidence correction can invalidate it
#:    (the grader judged something incomplete) -> STALE, re-review required.
#:  * original_instructor_grade — the score was COPIED from the authoritative
#:    original instructor-graded exam, for the whole grading unit. It never
#:    depended on what the app displayed, so repairing the app-visible evidence
#:    does NOT invalidate it.
#:  * adjudicated — an adjudicator's resolution of conflicting labels.
LABEL_SOURCES = ("human_independent_grading", "original_instructor_grade", "adjudicated")
DEFAULT_LABEL_SOURCE = "human_independent_grading"
#: sources whose score comes from OUTSIDE the app's evidence -> never evidence-stale,
#: and never subject to ordinary second-grader agreement (they are not a judgment
#: of the answer, they are a transcription of an authoritative existing grade).
AUTHORITATIVE_LABEL_SOURCES = ("original_instructor_grade",)
CLAIM_TTL_S = 30 * 60
#: item state when a saved/flagged label (or the FINAL) predates an evidence change
STATE_EVIDENCE_REVIEW = "NEEDS_REVIEW_AFTER_EVIDENCE_CHANGE"
#: item state when ground truth is an authoritative original-instructor grade:
#: complete, and NOT awaiting a second independent grader (see LABEL_SOURCES)
STATE_AUTHORITATIVE = "AUTHORITATIVE_GROUND_TRUTH"


class StaleWrite(RuntimeError):
    """The client's revision is older than the stored one (HTTP 409)."""


class StaleEvidence(StaleWrite):
    """The client graded evidence that is no longer the served evidence (HTTP 409)."""


class LabelError(ValueError):
    """Invalid label (score out of range, unknown status, ...)."""


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _is_stale(label_fp: str | None, current_fp: str | None, label_source: str | None = None) -> bool:
    """A label is stale only when BOTH fingerprints are known and differ; an
    unknown fingerprint is reported as unknown, never assumed stale or fresh.

    PROVENANCE-AWARE: a label whose score was copied from the authoritative
    original instructor grading (``AUTHORITATIVE_LABEL_SOURCES``) did not come
    from the app's evidence at all, so repairing that evidence cannot invalidate
    it — it is never stale. Independent human grading still goes stale, because
    that judgment WAS formed from the evidence that changed."""
    if label_source in AUTHORITATIVE_LABEL_SOURCES:
        return False
    return bool(label_fp) and bool(current_fp) and label_fp != current_fp


def _sql_tuple(values) -> str:
    """SQL literal tuple for a STATIC, code-controlled set (never user input)."""
    return "(" + ", ".join("'" + str(v).replace("'", "''") + "'" for v in values) + ")"


DDL = [
    f"""CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)""",
    """CREATE TABLE IF NOT EXISTS items (
        item_id TEXT PRIMARY KEY,
        max_score REAL NOT NULL,
        rubric_ids TEXT NOT NULL DEFAULT '[]',
        wanted_labels INTEGER NOT NULL DEFAULT 1,
        revision INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL,
        eligible INTEGER NOT NULL DEFAULT 1,
        evidence_sha256 TEXT,
        evidence_previous_sha256 TEXT,
        evidence_changed_at TEXT)""",
    """CREATE TABLE IF NOT EXISTS graders (
        name TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        last_seen TEXT NOT NULL)""",
    """CREATE TABLE IF NOT EXISTS claims (
        item_id TEXT NOT NULL,
        grader TEXT NOT NULL,
        claimed_at TEXT NOT NULL,
        expires_at REAL NOT NULL,
        PRIMARY KEY (item_id, grader))""",
    """CREATE TABLE IF NOT EXISTS labels (
        item_id TEXT NOT NULL,
        grader TEXT NOT NULL,
        score REAL,
        rubric TEXT NOT NULL DEFAULT '[]',
        note TEXT NOT NULL DEFAULT '',
        status TEXT NOT NULL,
        flag_reason TEXT NOT NULL DEFAULT '',
        revision INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        evidence_sha256 TEXT,
        label_source TEXT NOT NULL DEFAULT 'human_independent_grading',
        entered_by TEXT NOT NULL DEFAULT '',
        source_ref TEXT NOT NULL DEFAULT '',
        provenance_asserted_by TEXT NOT NULL DEFAULT '',
        provenance_asserted_at TEXT,
        PRIMARY KEY (item_id, grader))""",
    """CREATE TABLE IF NOT EXISTS final_labels (
        item_id TEXT PRIMARY KEY,
        score REAL NOT NULL,
        rubric TEXT NOT NULL DEFAULT '[]',
        note TEXT NOT NULL DEFAULT '',
        source TEXT NOT NULL,
        adjudicator TEXT NOT NULL DEFAULT '',
        contributing_graders TEXT NOT NULL DEFAULT '[]',
        from_revisions TEXT NOT NULL DEFAULT '{}',
        finalized_at TEXT NOT NULL,
        schema_version INTEGER NOT NULL,
        evidence_sha256 TEXT,
        ground_truth_source TEXT NOT NULL DEFAULT 'human_independent_grading')""",
    """CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        grader TEXT NOT NULL,
        action TEXT NOT NULL,
        item_id TEXT,
        revision INTEGER,
        detail TEXT NOT NULL DEFAULT '{}')""",
    """CREATE INDEX IF NOT EXISTS idx_labels_grader ON labels(grader)""",
    """CREATE INDEX IF NOT EXISTS idx_events_item ON events(item_id)""",
]

#: backward-safe column additions for databases created by older schemas
#: (e.g. the strong PC's labels.db): existing rows are never rewritten.
_MIGRATIONS = {
    "items": (("eligible", "INTEGER NOT NULL DEFAULT 1"), ("evidence_sha256", "TEXT"),
              ("evidence_previous_sha256", "TEXT"), ("evidence_changed_at", "TEXT")),
    "labels": (("evidence_sha256", "TEXT"),
               ("label_source", "TEXT NOT NULL DEFAULT 'human_independent_grading'"),
               ("entered_by", "TEXT NOT NULL DEFAULT ''"),
               ("source_ref", "TEXT NOT NULL DEFAULT ''"),
               ("provenance_asserted_by", "TEXT NOT NULL DEFAULT ''"),
               ("provenance_asserted_at", "TEXT")),
    "final_labels": (("evidence_sha256", "TEXT"),
                     ("ground_truth_source", "TEXT NOT NULL DEFAULT 'human_independent_grading'")),
}


#: opt-in for the rare test that deliberately wants the real deployment path
LIVE_DB_OPT_IN = "LABELING_ALLOW_LIVE_DB"


def live_db_path() -> Path:
    """The one database a test must never open: the real deployment's labels.db.

    Resolved WITHOUT LABELING_DATA_DIR on purpose — a test that redirects that
    variable is exactly the safe case, and must not accidentally make the guard
    point somewhere else."""
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return (Path(base) / "autograder" / "labeling" / "labels.db").resolve()


def assert_not_live_database(path: Path) -> None:
    """Refuse to open the live labeling database from a test.

    ``LabelDB.__init__`` is a WRITER: it sets ``journal_mode``, runs DDL and
    migrations, and its connection checkpoints the WAL when it closes. Doing
    that to the deployment's database — especially while the labeling server
    holds it open — can physically corrupt the only copy of the human ground
    truth. Under pytest that is never intentional, so it fails loudly here,
    BEFORE any SQLite connection is opened.

    Production is unaffected: the check only fires when pytest is running.
    A test that genuinely needs the real file (forensics) may set
    ``LABELING_ALLOW_LIVE_DB=1``, and should still open it read-only."""
    if not (os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("PYTEST_VERSION")):
        return
    if os.environ.get(LIVE_DB_OPT_IN):
        return
    try:
        resolved = Path(path).resolve()
    except OSError:                                    # unresolvable path cannot be the live one
        return
    if resolved == live_db_path():
        raise LabelError(
            f"refusing to open the LIVE labeling database from a test ({resolved}). "
            "LabelDB is a writer — it sets journal_mode and runs DDL, and closing it checkpoints "
            "the WAL, which can corrupt the deployment's only copy of the ground truth. Point the "
            "test at tmp_path, or snapshot the live file first with "
            f"labeling_app.backup.snapshot_sqlite and open the copy. Set {LIVE_DB_OPT_IN}=1 only "
            "for deliberate read-only forensics.")


class LabelDB:
    def __init__(self, path: str | Path, *, claim_ttl_s: int = CLAIM_TTL_S):
        assert_not_live_database(path)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.claim_ttl_s = claim_ttl_s
        self._local = threading.local()
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            for ddl in DDL:
                c.execute(ddl)
            for table, cols in _MIGRATIONS.items():
                have = {r["name"] for r in c.execute(f"PRAGMA table_info({table})")}
                for name, decl in cols:
                    if name not in have:
                        c.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
            # write the schema stamp only when it actually changes: opening an
            # up-to-date database (status, backup, export) must not touch the WAL
            cur = c.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
            if cur is None or str(cur[0]) != str(SCHEMA_VERSION):
                c.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))

    # ---------------------------------------------------------------- conn --
    @contextmanager
    def _conn(self, write: bool = False) -> Iterator[sqlite3.Connection]:
        """One connection per thread; writes run inside BEGIN IMMEDIATE so
        they are short, serialized and never interleave half-written."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=15.0, isolation_level=None,
                                   check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=15000")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        if write:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        else:
            yield conn

    def _event(self, c: sqlite3.Connection, grader: str, action: str, item_id: str | None,
               revision: int | None, detail: dict | None = None) -> None:
        c.execute("INSERT INTO events(ts, grader, action, item_id, revision, detail) VALUES (?,?,?,?,?,?)",
                  (_now(), grader, action, item_id, revision, json.dumps(detail or {}, ensure_ascii=False, sort_keys=True)))

    # ---------------------------------------------------------------- items --
    def load_items(self, items: list[dict]) -> int:
        """Register bundle items (idempotent). Returns the number added."""
        added = 0
        with self._conn(write=True) as c:
            for it in items:
                cur = c.execute(
                    "INSERT OR IGNORE INTO items(item_id, max_score, rubric_ids, wanted_labels, revision, created_at) "
                    "VALUES (?,?,?,?,0,?)",
                    (it["item_id"], float(it["max_score"]),
                     json.dumps([r["id"] for r in it.get("rubric_items", [])]), 1, _now()))
                added += cur.rowcount
        return added

    def sync_eligibility(self, bundle_item_ids: list[str], ineligible_item_ids: list[str], *,
                         eligibility_known: bool) -> dict:
        """Reconcile ``items.eligible`` with the served bundle. Diff-aware and
        fail-safe:

        * items no longer in the bundle are RETIRED (eligible=0) — they are not
          part of the current human workload (labels/finals are kept and
          surface as obsolete, never deleted);
        * bundle items get their flag only when eligibility is actually KNOWN
          (explicit bundle flags, or a dataset recompute that really ran).
          Unknown eligibility never flips anything — in particular it can
          never silently erase an earlier ineligible mark.

        Events record only real transitions (no writes, no events when nothing
        changes)."""
        bundle = set(bundle_item_ids or [])
        bad = set(ineligible_item_ids or [])
        changes: dict[str, list[str]] = {"retired": [], "marked_ineligible": [], "restored": []}
        with self._conn(write=True) as c:
            for r in c.execute("SELECT item_id, eligible FROM items").fetchall():
                iid, cur = r["item_id"], int(r["eligible"])
                if iid not in bundle:
                    want, kind = 0, "retired"
                elif not eligibility_known:
                    continue
                elif iid in bad:
                    want, kind = 0, "marked_ineligible"
                else:
                    want, kind = 1, "restored"
                if want != cur:
                    c.execute("UPDATE items SET eligible=? WHERE item_id=?", (want, iid))
                    changes[kind].append(iid)
            if any(changes.values()):
                self._event(c, "system", "sync_eligibility", None, None,
                            {k: sorted(v) for k, v in changes.items() if v})
        return changes

    # ------------------------------------------------------------- evidence --
    def sync_evidence(self, fingerprints: dict[str, str]) -> dict[str, Any]:
        """Register the served bundle's evidence fingerprints and detect
        changes. Deterministic and diff-aware:

        * first registration of an item (no fingerprint on record yet): the
          fingerprint is stored and every existing label/FINAL of that item
          WITHOUT a recorded fingerprint is backfilled with it — the explicit,
          logged assumption being that those labels were made against the
          bundle being registered (which is why ``build-bundle --replace``
          registers the OLD bundle before replacing it);
        * a changed fingerprint: the item's fingerprint moves (previous one and
          the time are kept), its revision bumps (in-flight FINAL writes get
          409), open claims are dropped and an ``evidence_changed`` event names
          the graders whose labels are now stale. Labels/FINALs are never
          modified or deleted — staleness is derived by comparison;
        * an unchanged fingerprint: no write, no event.

        Items not in ``fingerprints`` (not in the served bundle) are left alone."""
        out: dict[str, Any] = {"registered": [], "changed": [], "backfilled_labels": 0, "backfilled_finals": 0}
        with self._conn(write=True) as c:
            ts = _now()
            for r in c.execute("SELECT item_id, evidence_sha256 FROM items ORDER BY item_id").fetchall():
                iid, cur = r["item_id"], r["evidence_sha256"]
                new = fingerprints.get(iid)
                if not new:
                    continue
                if cur is None:
                    c.execute("UPDATE items SET evidence_sha256=? WHERE item_id=?", (new, iid))
                    out["backfilled_labels"] += c.execute(
                        "UPDATE labels SET evidence_sha256=? WHERE item_id=? AND evidence_sha256 IS NULL", (new, iid)).rowcount
                    out["backfilled_finals"] += c.execute(
                        "UPDATE final_labels SET evidence_sha256=? WHERE item_id=? AND evidence_sha256 IS NULL", (new, iid)).rowcount
                    out["registered"].append(iid)
                elif cur != new:
                    c.execute("UPDATE items SET evidence_sha256=?, evidence_previous_sha256=?, evidence_changed_at=?, "
                              "revision=revision+1 WHERE item_id=?", (new, cur, ts, iid))
                    graders = [x["grader"] for x in c.execute(
                        "SELECT grader FROM labels WHERE item_id=? ORDER BY grader", (iid,))]
                    has_final = bool(c.execute("SELECT 1 FROM final_labels WHERE item_id=?", (iid,)).fetchone())
                    c.execute("DELETE FROM claims WHERE item_id=?", (iid,))
                    self._event(c, "system", "evidence_changed", iid, None,
                                {"previous": cur, "current": new, "graders_with_labels": graders,
                                 "final_present": has_final})
                    out["changed"].append({"item_id": iid, "previous": cur, "current": new,
                                           "graders": graders, "final_present": has_final})
            if out["registered"]:
                self._event(c, "system", "evidence_registered", None, None,
                            {"items": len(out["registered"]), "backfilled_labels": out["backfilled_labels"],
                             "backfilled_finals": out["backfilled_finals"],
                             "assumption": "labels without a recorded fingerprint were made against the bundle "
                                           "being registered"})
        out["report"] = self.evidence_report()
        return out

    def evidence_report(self) -> dict[str, Any]:
        """Exact accounting of every label/FINAL against the current evidence:
        preserved (all of them — nothing is ever deleted), fresh, stale
        (fingerprint known and different), unknown (no fingerprint recorded)."""
        with self._conn() as c:
            labels = c.execute(
                "SELECT l.item_id, l.grader, l.status, l.revision, l.updated_at, l.evidence_sha256 AS fp, "
                "l.label_source AS src, l.entered_by, l.source_ref, l.provenance_asserted_by, "
                "i.evidence_sha256 AS cur, i.evidence_changed_at, i.evidence_previous_sha256 "
                "FROM labels l JOIN items i ON i.item_id=l.item_id "
                "ORDER BY l.item_id, l.grader").fetchall()
            finals = c.execute(
                "SELECT f.item_id, f.source, f.ground_truth_source AS gts, f.finalized_at, "
                "f.evidence_sha256 AS fp, i.evidence_sha256 AS cur "
                "FROM final_labels f JOIN items i ON i.item_id=f.item_id ORDER BY f.item_id").fetchall()
            changed = [dict(r) for r in c.execute(
                "SELECT item_id, evidence_previous_sha256, evidence_sha256, evidence_changed_at FROM items "
                "WHERE evidence_changed_at IS NOT NULL ORDER BY item_id")]
        stale = [r for r in labels if _is_stale(r["fp"], r["cur"], r["src"])]
        unknown = [r for r in labels if r["fp"] is None]
        stale_f = [r for r in finals if _is_stale(r["fp"], r["cur"], r["gts"])]
        authoritative = [r for r in labels
                         if (r["src"] or DEFAULT_LABEL_SOURCE) in AUTHORITATIVE_LABEL_SOURCES]
        # Authoritative labels on an item whose evidence WAS repaired: the score
        # stands (it came from the complete original grading), but the repair is
        # still recorded. Two separate facts, reported separately, never merged.
        auth_repaired = [r for r in authoritative
                         if r["evidence_changed_at"] and _is_stale(r["fp"], r["cur"])]

        def _lab(r):
            return {"item_id": r["item_id"], "grader": r["grader"], "status": r["status"], "revision": r["revision"],
                    "updated_at": r["updated_at"], "evidence_changed_at": r["evidence_changed_at"],
                    "label_source": r["src"] or DEFAULT_LABEL_SOURCE}
        return {
            "labels_total": len(labels),
            "labels_preserved": len(labels),                   # nothing is ever deleted
            "labels_fresh": len(labels) - len(stale) - len(unknown),
            "labels_stale": len(stale),
            "labels_unknown_evidence": len(unknown),
            "labels_by_source": {k: sum(1 for r in labels if (r["src"] or DEFAULT_LABEL_SOURCE) == k)
                                 for k in LABEL_SOURCES},
            "labels_authoritative": len(authoritative),
            # authoritative labels whose evidence was repaired: still valid, NOT re-review work
            "authoritative_labels_on_repaired_evidence": [_lab(r) for r in auth_repaired],
            "stale_labels": [_lab(r) for r in stale],
            "unknown_evidence_labels": [_lab(r) for r in unknown],
            "finals_total": len(finals),
            "finals_stale": len(stale_f),
            "stale_finals": [{"item_id": r["item_id"], "source": r["source"], "finalized_at": r["finalized_at"]}
                             for r in stale_f],
            "items_evidence_changed": changed,
        }

    def item_ids(self) -> list[str]:
        with self._conn() as c:
            return [r["item_id"] for r in c.execute("SELECT item_id FROM items ORDER BY item_id")]

    def item(self, item_id: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM items WHERE item_id=?", (item_id,)).fetchone()
            return dict(r) if r else None

    def set_wanted_labels(self, mode: str, item_ids: list[str] | None = None, n: int = 2) -> int:
        """Double-labeling policy: 'all' | 'none' | 'selected' (item_ids)."""
        with self._conn(write=True) as c:
            if mode == "all":
                cur = c.execute("UPDATE items SET wanted_labels=?", (n,))
            elif mode == "none":
                cur = c.execute("UPDATE items SET wanted_labels=1")
            elif mode == "selected":
                c.execute("UPDATE items SET wanted_labels=1")
                cur = c.executemany("UPDATE items SET wanted_labels=? WHERE item_id=?",
                                    [(n, i) for i in (item_ids or [])])
            else:
                raise LabelError(f"unknown double-label mode {mode!r}")
            c.execute("INSERT OR REPLACE INTO meta(key, value) VALUES ('double_label_mode', ?)", (mode,))
            self._event(c, "admin", "set_policy", None, None, {"mode": mode, "items": item_ids or [], "n": n})
            return cur.rowcount

    # -------------------------------------------------------------- graders --
    def touch_grader(self, name: str) -> None:
        name = (name or "").strip()
        if not name or len(name) > 64:
            raise LabelError("grader name must be 1..64 characters")
        with self._conn(write=True) as c:
            c.execute("INSERT INTO graders(name, created_at, last_seen) VALUES (?,?,?) "
                      "ON CONFLICT(name) DO UPDATE SET last_seen=excluded.last_seen", (name, _now(), _now()))

    def graders(self) -> list[str]:
        with self._conn() as c:
            return [r["name"] for r in c.execute("SELECT name FROM graders ORDER BY name")]

    # ---------------------------------------------------------------- claims --
    def _expire_claims(self, c: sqlite3.Connection) -> None:
        c.execute("DELETE FROM claims WHERE expires_at < ?", (time.time(),))

    def claim_next(self, grader: str, *, include_skipped: bool = False) -> str | None:
        """The next item this grader should label: first the grader's OWN labels
        that went stale (the evidence was corrected — re-review), then items
        nobody has labeled, then items that still want another label from
        someone else; items actively claimed by another grader are deferred
        (not hidden). Stale labels never count as existing labels."""
        self.touch_grader(grader)
        with self._conn(write=True) as c:
            self._expire_claims(c)
            rows = c.execute("""
                SELECT i.item_id, i.wanted_labels, i.eligible, i.evidence_sha256 AS cur_fp,
                       (SELECT COUNT(*) FROM labels l WHERE l.item_id=i.item_id AND l.status='saved'
                          AND (l.label_source IN {auth}
                               OR l.evidence_sha256 IS NULL OR i.evidence_sha256 IS NULL
                               OR l.evidence_sha256 = i.evidence_sha256)) AS n_saved,
                       (SELECT COUNT(*) FROM labels l WHERE l.item_id=i.item_id AND l.status='saved'
                          AND l.label_source IN {auth}) AS n_authoritative,
                       (SELECT COUNT(*) FROM labels l WHERE l.item_id=i.item_id AND l.grader=? ) AS mine,
                       (SELECT status FROM labels l WHERE l.item_id=i.item_id AND l.grader=?) AS my_status,
                       (SELECT evidence_sha256 FROM labels l WHERE l.item_id=i.item_id AND l.grader=?) AS my_fp,
                       (SELECT label_source FROM labels l WHERE l.item_id=i.item_id AND l.grader=?) AS my_src,
                       (SELECT COUNT(*) FROM claims k WHERE k.item_id=i.item_id AND k.grader<>?) AS claimed_by_others,
                       (SELECT COUNT(*) FROM final_labels f WHERE f.item_id=i.item_id) AS is_final
                FROM items i ORDER BY i.item_id""".format(auth=_sql_tuple(AUTHORITATIVE_LABEL_SOURCES)),
                (grader, grader, grader, grader, grader)).fetchall()

            def _my_stale(r) -> bool:
                return bool(r["mine"]) and _is_stale(r["my_fp"], r["cur_fp"], r["my_src"])

            def _eligible(r) -> bool:
                if not r["eligible"]:            # policy decides this item's score; no human label wanted
                    return False
                if r["is_final"]:
                    return False
                # Ground truth here is an authoritative original-instructor grade.
                # An independent grading of the answer is not comparable to it, so a
                # second label is never requested to satisfy wanted_labels, and the
                # item is never handed back because its evidence was repaired.
                if r["n_authoritative"]:
                    return False
                if _my_stale(r):                 # my own label predates an evidence change: re-review it
                    return True
                if r["mine"] and not (include_skipped and r["my_status"] == "skipped"):
                    return False
                return r["n_saved"] < r["wanted_labels"]
            pool = [r for r in rows if _eligible(r)]
            if not pool:
                return None
            # priority: (my stale re-reviews) > (no label yet) > (needs another label); unclaimed before claimed-by-others
            pool.sort(key=lambda r: (not _my_stale(r), r["n_saved"] > 0, r["claimed_by_others"] > 0, r["item_id"]))
            pick = pool[0]["item_id"]
            c.execute("INSERT OR REPLACE INTO claims(item_id, grader, claimed_at, expires_at) VALUES (?,?,?,?)",
                      (pick, grader, _now(), time.time() + self.claim_ttl_s))
            self._event(c, grader, "claim", pick, None)
            return pick

    def release_claim(self, item_id: str, grader: str) -> None:
        with self._conn(write=True) as c:
            c.execute("DELETE FROM claims WHERE item_id=? AND grader=?", (item_id, grader))

    # ---------------------------------------------------------------- labels --
    def get_label(self, item_id: str, grader: str) -> dict | None:
        with self._conn() as c:
            r = c.execute("SELECT * FROM labels WHERE item_id=? AND grader=?", (item_id, grader)).fetchone()
            if not r:
                return None
            it = c.execute("SELECT evidence_sha256 FROM items WHERE item_id=?", (item_id,)).fetchone()
            return self._label_dict(r, it["evidence_sha256"] if it else None)

    @staticmethod
    def _label_dict(r: sqlite3.Row, current_fp: str | None = None) -> dict:
        d = dict(r)
        d["rubric"] = json.loads(d.get("rubric") or "[]")
        d["label_source"] = d.get("label_source") or DEFAULT_LABEL_SOURCE
        d["authoritative"] = d["label_source"] in AUTHORITATIVE_LABEL_SOURCES
        # an authoritative (instructor-copied) score never goes stale on an
        # evidence repair; an independent judgment still does
        d["evidence_stale"] = _is_stale(d.get("evidence_sha256"), current_fp, d["label_source"])
        return d

    def save_label(self, item_id: str, grader: str, *, score: float | None, rubric: list[str] | None,
                   note: str = "", status: str = "saved", flag_reason: str = "",
                   expected_revision: int = 0, client_evidence_sha256: str | None = None,
                   label_source: str = DEFAULT_LABEL_SOURCE, entered_by: str = "",
                   source_ref: str = "") -> dict:
        """Create/update THIS grader's label for the item. ``expected_revision``
        is the revision the client loaded (0 = no label yet). Stale → StaleWrite.
        ``client_evidence_sha256`` (what the client's page showed) must equal
        the item's current fingerprint when both are known → StaleEvidence
        otherwise. The label records the CURRENT fingerprint (server truth)."""
        if status not in LABEL_STATUSES:
            raise LabelError(f"unknown status {status!r}")
        if label_source not in LABEL_SOURCES:
            raise LabelError(f"unknown label_source {label_source!r}")
        self.touch_grader(grader)
        with self._conn(write=True) as c:
            it = c.execute("SELECT max_score, rubric_ids, revision, eligible, evidence_sha256 FROM items WHERE item_id=?",
                           (item_id,)).fetchone()
            if it is None:
                raise LabelError(f"unknown item {item_id!r}")
            if not int(it["eligible"]):
                raise LabelError("this item is not eligible for human explanation labeling — the grading "
                                 "policy already decides its score deterministically")
            if c.execute("SELECT 1 FROM final_labels WHERE item_id=?", (item_id,)).fetchone():
                raise LabelError("this item already has a FINAL label; ask the admin to reopen it")
            current_fp = it["evidence_sha256"]
            if client_evidence_sha256 and current_fp and client_evidence_sha256 != current_fp:
                raise StaleEvidence("the evidence shown for this item has changed since you loaded it; "
                                    "reload and re-check the complete answer before grading")
            allowed = set(json.loads(it["rubric_ids"] or "[]"))
            rubric_clean = sorted({str(x) for x in (rubric or [])})
            bad = [x for x in rubric_clean if x not in allowed]
            if bad:
                raise LabelError(f"unknown rubric item(s) {bad}")
            if status == "saved":
                if score is None:
                    raise LabelError("a saved label needs a score")
                score = float(score)
                if score < 0 or score > float(it["max_score"]) + 1e-9:
                    raise LabelError(f"score {score} outside 0..{it['max_score']}")
                if abs(score * 2 - round(score * 2)) > 1e-9:
                    raise LabelError("scores are in 0.5 steps")
            else:
                score = None if status == "skipped" else (float(score) if score is not None else None)
            cur = c.execute("SELECT revision FROM labels WHERE item_id=? AND grader=?", (item_id, grader)).fetchone()
            current = int(cur["revision"]) if cur else 0
            if int(expected_revision) != current:
                raise StaleWrite(f"label revision is {current}, you loaded {expected_revision}; reload the item")
            new_rev = current + 1
            ts = _now()
            if cur:
                c.execute("UPDATE labels SET score=?, rubric=?, note=?, status=?, flag_reason=?, revision=?, updated_at=?, "
                          "evidence_sha256=?, label_source=?, entered_by=?, source_ref=? WHERE item_id=? AND grader=?",
                          (score, json.dumps(rubric_clean), note or "", status, flag_reason or "", new_rev, ts,
                           current_fp, label_source, entered_by or "", source_ref or "", item_id, grader))
            else:
                c.execute("INSERT INTO labels(item_id, grader, score, rubric, note, status, flag_reason, revision, "
                          "created_at, updated_at, evidence_sha256, label_source, entered_by, source_ref) "
                          "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (item_id, grader, score, json.dumps(rubric_clean), note or "", status, flag_reason or "",
                           new_rev, ts, ts, current_fp, label_source, entered_by or "", source_ref or ""))
            c.execute("UPDATE items SET revision=revision+1 WHERE item_id=?", (item_id,))
            c.execute("DELETE FROM claims WHERE item_id=? AND grader=?", (item_id, grader))
            self._event(c, grader, f"label_{status}", item_id, new_rev,
                        {"score": score, "rubric": rubric_clean, "flag_reason": flag_reason or "",
                         "evidence_sha256": current_fp})
            r = c.execute("SELECT * FROM labels WHERE item_id=? AND grader=?", (item_id, grader)).fetchone()
            return self._label_dict(r, current_fp)

    def labels_for_item(self, item_id: str) -> list[dict]:
        with self._conn() as c:
            it = c.execute("SELECT evidence_sha256 FROM items WHERE item_id=?", (item_id,)).fetchone()
            cur_fp = it["evidence_sha256"] if it else None
            return [self._label_dict(r, cur_fp) for r in c.execute(
                "SELECT * FROM labels WHERE item_id=? ORDER BY grader", (item_id,))]

    def my_items(self, grader: str) -> dict[str, list[str]]:
        """This grader's items by status, plus ``stale``: the ones whose
        evidence changed after the grader's label (need re-review)."""
        with self._conn() as c:
            out: dict[str, list[str]] = {"saved": [], "skipped": [], "flagged": [], "stale": []}
            for r in c.execute("SELECT l.item_id, l.status, l.evidence_sha256 AS fp, i.evidence_sha256 AS cur, "
                               "l.label_source AS src "
                               "FROM labels l JOIN items i ON i.item_id=l.item_id WHERE l.grader=? ORDER BY l.item_id",
                               (grader,)):
                out[r["status"]].append(r["item_id"])
                if _is_stale(r["fp"], r["cur"], r["src"]):
                    out["stale"].append(r["item_id"])
            return out

    # ------------------------------------------------------------- overview --
    @staticmethod
    def _agreement(saved: list[dict]) -> bool | None:
        if len(saved) < 2:
            return None
        sig = {(float(l["score"]), tuple(l["rubric"])) for l in saved}
        return len(sig) == 1

    def overview(self, item_id: str) -> dict[str, Any]:
        with self._conn() as c:
            self._expire_claims(c)
            it = c.execute("SELECT * FROM items WHERE item_id=?", (item_id,)).fetchone()
            if it is None:
                raise LabelError(f"unknown item {item_id!r}")
            cur_fp = it["evidence_sha256"]
            labels = [self._label_dict(r, cur_fp) for r in c.execute("SELECT * FROM labels WHERE item_id=? ORDER BY grader", (item_id,))]
            final = c.execute("SELECT * FROM final_labels WHERE item_id=?", (item_id,)).fetchone()
            claims = [dict(r) for r in c.execute("SELECT grader, claimed_at FROM claims WHERE item_id=?", (item_id,))]
        stale = [l for l in labels if l["evidence_stale"] and l["status"] in ("saved", "flagged")]
        saved = [l for l in labels if l["status"] == "saved" and not l["evidence_stale"]]
        flagged = [l for l in labels if l["status"] == "flagged" and not l["evidence_stale"]]
        authoritative = [l for l in saved if l["authoritative"]]
        independent = [l for l in saved if not l["authoritative"]]
        # Ordinary inter-grader agreement compares INDEPENDENT judgments of the
        # student's answer. A score copied from the original instructor grading is
        # not such a judgment, so it is never folded into an agreement calculation.
        agree = self._agreement(independent)
        eligible = bool(it["eligible"]) if "eligible" in it.keys() else True
        # A FINAL is judged by the SAME provenance rule as a label: one whose
        # ground truth is a copied original-instructor grade did not depend on
        # the app's evidence, so repairing that evidence does not invalidate it
        # (final_rows/export already apply this rule — the state machine must
        # not disagree with the export).
        final_stale = bool(final) and _is_stale(final["evidence_sha256"], cur_fp,
                                                (final["ground_truth_source"] if "ground_truth_source" in final.keys()
                                                 else None))
        if not eligible:
            # deterministic policy score is authoritative; any history below is obsolete
            state = "INELIGIBLE"
        elif final and final_stale:
            state = STATE_EVIDENCE_REVIEW          # a FINAL made on superseded evidence is not ground truth
        elif final:
            state = "FINAL"
        elif stale:
            state = STATE_EVIDENCE_REVIEW          # some grader must re-review the corrected evidence
        elif (len(independent) >= 2 and agree is False) or flagged:
            state = "NEEDS_ADJUDICATION"
        elif authoritative:
            # ground truth copied from the authoritative original instructor grading:
            # complete, not awaiting a second independent grader
            state = STATE_AUTHORITATIVE
        elif len(independent) >= 2 and agree:
            state = "AGREEMENT"
        elif saved:
            state = "LABELED"
        elif claims:
            state = "ASSIGNED"
        else:
            state = "UNLABELED"
        fd = dict(final) if final else None
        if fd:
            fd["rubric"] = json.loads(fd.get("rubric") or "[]")
            fd["contributing_graders"] = json.loads(fd.get("contributing_graders") or "[]")
            fd["from_revisions"] = json.loads(fd.get("from_revisions") or "{}")
            fd["obsolete_ineligible"] = not eligible
            fd["evidence_stale"] = final_stale
        return {"item_id": item_id, "revision": it["revision"], "max_score": it["max_score"],
                "wanted_labels": it["wanted_labels"], "state": state, "eligible": eligible,
                "obsolete_labels": (len(labels) if not eligible else 0), "n_saved": len(saved),
                "n_skipped": sum(1 for l in labels if l["status"] == "skipped"), "n_flagged": len(flagged),
                "n_stale": len(stale), "stale_graders": sorted(l["grader"] for l in stale),
                "n_authoritative": len(authoritative), "n_independent": len(independent),
                "authoritative_graders": sorted(l["grader"] for l in authoritative),
                "ground_truth_source": (authoritative[0]["label_source"] if authoritative
                                        else DEFAULT_LABEL_SOURCE),
                "evidence_sha256": cur_fp, "evidence_previous_sha256": it["evidence_previous_sha256"],
                "evidence_changed_at": it["evidence_changed_at"],
                # the evidence for this item was repaired at some point — an audit
                # fact kept independently of whether any label became stale
                "evidence_repaired": bool(it["evidence_changed_at"]),
                "agreement": agree, "claims": claims, "labels": labels, "final": fd}

    def all_overviews(self) -> list[dict]:
        return [self.overview(i) for i in self.item_ids()]

    def summary(self) -> dict[str, Any]:
        ovs = self.all_overviews()
        by_state: dict[str, int] = {}
        for o in ovs:
            by_state[o["state"]] = by_state.get(o["state"], 0) + 1
        per_grader: dict[str, dict] = {}
        with self._conn() as c:
            for r in c.execute("SELECT grader, status, COUNT(*) AS n FROM labels GROUP BY grader, status"):
                per_grader.setdefault(r["grader"], {"saved": 0, "skipped": 0, "flagged": 0, "stale": 0})[r["status"]] = r["n"]
            # Per-grader staleness must use the SAME provenance rule as
            # _is_stale/my_items/progress: a score copied from the original
            # instructor grading never goes stale on an evidence repair, so it
            # is excluded here too (otherwise the admin page would show a
            # grader stale work they do not have).
            for r in c.execute("SELECT l.grader, COUNT(*) AS n FROM labels l JOIN items i ON i.item_id=l.item_id "
                               "WHERE l.evidence_sha256 IS NOT NULL AND i.evidence_sha256 IS NOT NULL "
                               "AND l.evidence_sha256 <> i.evidence_sha256 "
                               "AND l.label_source NOT IN " + _sql_tuple(AUTHORITATIVE_LABEL_SOURCES) +
                               " GROUP BY l.grader"):
                per_grader.setdefault(r["grader"], {"saved": 0, "skipped": 0, "flagged": 0, "stale": 0})["stale"] = r["n"]
            mode = c.execute("SELECT value FROM meta WHERE key='double_label_mode'").fetchone()
        eligible_ovs = [o for o in ovs if o["eligible"]]
        return {
            "total_items": len(ovs),
            "eligible_items": len(eligible_ovs),
            "ineligible_items": len(ovs) - len(eligible_ovs),
            "obsolete_ineligible_labels": sum(o["obsolete_labels"] for o in ovs),
            "obsolete_ineligible_finals": sum(1 for o in ovs if not o["eligible"] and o["final"]),
            "unlabeled": by_state.get("UNLABELED", 0) + by_state.get("ASSIGNED", 0),
            "assigned": by_state.get("ASSIGNED", 0),
            "labels_completed": sum(o["n_saved"] for o in eligible_ovs),
            "singly_labeled": sum(1 for o in eligible_ovs if o["n_saved"] == 1),
            "double_labeled": sum(1 for o in eligible_ovs if o["n_saved"] >= 2),
            "agreements": sum(1 for o in eligible_ovs if o["agreement"] is True),
            "disagreements": sum(1 for o in eligible_ovs if o["agreement"] is False),
            "flagged": sum(1 for o in eligible_ovs if o["n_flagged"]),
            "needs_adjudication": by_state.get("NEEDS_ADJUDICATION", 0),
            "final": by_state.get("FINAL", 0),
            "stale_labels": sum(o["n_stale"] for o in ovs),
            "stale_finals": sum(1 for o in ovs if o["final"] and o["final"].get("evidence_stale")),
            "needs_evidence_review": by_state.get(STATE_EVIDENCE_REVIEW, 0),
            "authoritative_ground_truth": by_state.get(STATE_AUTHORITATIVE, 0),
            "authoritative_labels": sum(o["n_authoritative"] for o in eligible_ovs),
            # items still genuinely awaiting another INDEPENDENT grader (authoritative
            # ground truth is never counted as awaiting a second grading)
            "awaiting_second_label": sum(1 for o in eligible_ovs if not o["final"]
                                         and not o["n_authoritative"]
                                         and o["n_saved"] < o["wanted_labels"]),
            "items_evidence_repaired": sum(1 for o in ovs if o["evidence_repaired"]),
            "items_evidence_changed": sum(1 for o in ovs if o["evidence_changed_at"]),
            "by_state": by_state, "per_grader": per_grader,
            "double_label_mode": (mode["value"] if mode else "none"),
            "graders": self.graders(),
        }

    def progress(self, grader: str) -> dict[str, Any]:
        # Only genuinely human-labelable items count toward the workload:
        # a deterministic-zero item needs zero human labels and is never
        # "unfinished". A grader's own STALE label is unfinished work again.
        ovs = [o for o in self.all_overviews() if o["eligible"]]
        mine = self.my_items(grader)
        mine_set = set(mine["saved"] + mine["skipped"] + mine["flagged"])
        stale_mine = set(mine["stale"])
        # An item whose ground truth is an authoritative original-instructor grade is
        # DONE: it never wants a second independent grading, and an evidence repair
        # never turns it back into work for anyone.
        remaining = sum(1 for o in ovs if not o["final"] and not o["n_authoritative"]
                        and ((o["item_id"] in stale_mine)
                             or (o["n_saved"] < o["wanted_labels"] and o["item_id"] not in mine_set)))
        return {"total_items": len(ovs), "labels_completed": sum(o["n_saved"] for o in ovs),
                "items_with_any_label": sum(1 for o in ovs if o["n_saved"]),
                "my_saved": len(mine["saved"]), "my_skipped": len(mine["skipped"]), "my_flagged": len(mine["flagged"]),
                "my_stale": len(stale_mine),
                "authoritative_items": sum(1 for o in ovs if o["n_authoritative"]),
                "remaining_for_me": remaining}

    def set_label_provenance(self, *, grader: str, label_source: str, entered_by: str = "",
                             asserted_by: str = "", source_refs: dict[str, str] | None = None,
                             item_ids: list[str] | None = None, actor: str = "system",
                             dry_run: bool = False) -> dict[str, Any]:
        """Record the PROVENANCE of a grader's existing labels — how their scores
        were derived — WITHOUT touching the scores themselves.

        This never changes ``score``, ``rubric``, ``status``, ``revision`` or
        ``evidence_sha256``: it only records what the label already was. Staleness
        is then re-derived from provenance by the ordinary rules — no flag is ever
        edited by hand.

        ``asserted_by`` records WHO asserted the provenance (e.g. ``owner``). The
        software does not independently verify that a score matches the original
        instructor's grade; the assertion and its author are stored so the claim is
        never mistaken for a machine-verified fact.

        Guards — a label is SKIPPED (never silently rewritten) unless it exists,
        its status is ``saved`` and it carries a score, and its item is eligible."""
        if label_source not in LABEL_SOURCES:
            raise LabelError(f"unknown label_source {label_source!r}")
        refs = dict(source_refs or {})
        ts = _now()
        applied: list[dict] = []
        skipped: list[dict] = []
        with self._conn(write=True) as c:
            rows = c.execute(
                "SELECT l.item_id, l.grader, l.status, l.score, l.label_source, l.evidence_sha256, "
                "i.eligible, (SELECT COUNT(*) FROM final_labels f WHERE f.item_id=l.item_id) AS is_final "
                "FROM labels l JOIN items i ON i.item_id=l.item_id "
                "WHERE l.grader=? ORDER BY l.item_id", (grader,)).fetchall()
            wanted = set(item_ids) if item_ids is not None else None
            for r in rows:
                iid = r["item_id"]
                if wanted is not None and iid not in wanted:
                    continue
                if r["status"] != "saved":
                    skipped.append({"item_id": iid, "reason": f"status is {r['status']!r}, not 'saved'"})
                    continue
                if r["score"] is None:
                    skipped.append({"item_id": iid, "reason": "label carries no score"})
                    continue
                if not int(r["eligible"]):
                    skipped.append({"item_id": iid, "reason": "item is not eligible for a human label"})
                    continue
                if int(r["is_final"]):
                    # A FINAL already froze this item's ground_truth_source. Rewriting
                    # the label's provenance underneath it would leave the two
                    # disagreeing, so the label is skipped: reopen the item first.
                    skipped.append({"item_id": iid,
                                    "reason": "item already has a FINAL label whose ground_truth_source is frozen; "
                                              "reopen it before changing this label's provenance"})
                    continue
                previous = r["label_source"] or DEFAULT_LABEL_SOURCE
                ref = refs.get(iid, "")
                if not dry_run:
                    c.execute("UPDATE labels SET label_source=?, entered_by=?, source_ref=?, "
                              "provenance_asserted_by=?, provenance_asserted_at=? "
                              "WHERE item_id=? AND grader=?",
                              (label_source, entered_by or grader, ref, asserted_by or "", ts, iid, grader))
                    self._event(c, actor, "label_provenance_set", iid, None,
                                {"grader": grader, "previous_label_source": previous,
                                 "label_source": label_source, "entered_by": entered_by or grader,
                                 "source_ref": ref, "asserted_by": asserted_by or "",
                                 "score_unchanged": r["score"],
                                 "evidence_sha256": r["evidence_sha256"],
                                 "note": "provenance only — score, rubric, status, revision and "
                                         "evidence fingerprint are untouched; asserted, not machine-verified"})
                applied.append({"item_id": iid, "previous_label_source": previous,
                                "label_source": label_source, "source_ref": ref})
            if applied and not dry_run:
                self._event(c, actor, "label_provenance_backfill", None, None,
                            {"grader": grader, "label_source": label_source, "entered_by": entered_by or grader,
                             "asserted_by": asserted_by or "", "labels": len(applied),
                             "skipped": len(skipped), "scores_modified": 0})
        return {"grader": grader, "label_source": label_source, "entered_by": entered_by or grader,
                "asserted_by": asserted_by or "", "dry_run": bool(dry_run),
                "applied": applied, "applied_count": len(applied),
                "skipped": skipped, "skipped_count": len(skipped), "scores_modified": 0}

    def verify_provenance(self) -> dict[str, Any]:
        """READ-ONLY audit of what provenance the stored labels actually carry,
        cross-checked against the audit trail.

        Every ``label_provenance_set`` event recorded the score the label had at
        the moment its provenance was written (``score_unchanged``). Comparing
        that against the score stored now PROVES, from the database itself, that
        recording provenance did not alter any score — and would name any label
        where it did. Also reports, per grader and per source: how many labels
        carry which ``label_source``, whether ``entered_by`` /
        ``provenance_asserted_by`` are populated, and which authoritative labels
        sit on repaired evidence (valid, but the repair stays visible).

        Nothing is written; this never "fixes" anything."""
        with self._conn() as c:
            labels = [dict(r) for r in c.execute(
                "SELECT l.*, i.evidence_sha256 AS cur_fp, i.evidence_previous_sha256 AS prev_fp, "
                "i.evidence_changed_at FROM labels l JOIN items i ON i.item_id=l.item_id ORDER BY l.item_id, l.grader")]
            events = [dict(r) for r in c.execute(
                "SELECT item_id, grader, detail FROM events WHERE action='label_provenance_set' ORDER BY id")]
            backfills = [dict(r) for r in c.execute(
                "SELECT ts, grader, detail FROM events WHERE action='label_provenance_backfill' ORDER BY id")]
        by_key = {(l["item_id"], l["grader"]): l for l in labels}
        score_changed: list[dict] = []
        checked = 0
        for e in events:
            try:
                d = json.loads(e["detail"] or "{}")
            except json.JSONDecodeError:
                continue
            lab = by_key.get((e["item_id"], d.get("grader")))
            if lab is None or "score_unchanged" not in d:
                continue
            checked += 1
            recorded, current = d.get("score_unchanged"), lab.get("score")
            if (recorded is None) != (current is None) or (
                    recorded is not None and abs(float(recorded) - float(current)) > 1e-9):
                score_changed.append({"item_id": e["item_id"], "grader": d.get("grader"),
                                      "score_when_provenance_recorded": recorded, "score_now": current})
        by_source: dict[str, int] = {s: 0 for s in LABEL_SOURCES}
        per_grader: dict[str, dict[str, Any]] = {}
        missing_entered_by, missing_asserted_by = [], []
        for l in labels:
            src = l.get("label_source") or DEFAULT_LABEL_SOURCE
            by_source[src] = by_source.get(src, 0) + 1
            g = per_grader.setdefault(l["grader"], {"labels": 0, "by_source": {}, "entered_by": set(),
                                                    "asserted_by": set(), "revisions": set(), "statuses": set()})
            g["labels"] += 1
            g["by_source"][src] = g["by_source"].get(src, 0) + 1
            g["entered_by"].add(l.get("entered_by") or "")
            g["asserted_by"].add(l.get("provenance_asserted_by") or "")
            g["revisions"].add(l.get("revision"))
            g["statuses"].add(l.get("status"))
            if src in AUTHORITATIVE_LABEL_SOURCES:
                if not (l.get("entered_by") or "").strip():
                    missing_entered_by.append(l["item_id"])
                if not (l.get("provenance_asserted_by") or "").strip():
                    missing_asserted_by.append(l["item_id"])
        for g in per_grader.values():
            for k in ("entered_by", "asserted_by", "revisions", "statuses"):
                g[k] = sorted(x for x in g[k] if x is not None)
        auth_on_repaired = [l["item_id"] for l in labels
                            if (l.get("label_source") or DEFAULT_LABEL_SOURCE) in AUTHORITATIVE_LABEL_SOURCES
                            and l.get("evidence_changed_at")]
        stale = [l["item_id"] for l in labels
                 if _is_stale(l.get("evidence_sha256"), l.get("cur_fp"), l.get("label_source"))
                 and l["status"] in ("saved", "flagged")]
        return {
            "labels_total": len(labels),
            "labels_by_source": by_source,
            "per_grader": per_grader,
            "provenance_events_checked": checked,
            "scores_changed_since_provenance_recorded": score_changed,
            "scores_unchanged": not score_changed,
            "authoritative_missing_entered_by": sorted(missing_entered_by),
            "authoritative_missing_asserted_by": sorted(missing_asserted_by),
            "authoritative_labels_on_repaired_evidence": sorted(auth_on_repaired),
            "stale_labels": sorted(stale),
            "backfill_events": [{"ts": b["ts"], "actor": b["grader"], **json.loads(b["detail"] or "{}")}
                                for b in backfills],
            "assertion_note": ("label_source/entered_by/provenance_asserted_by record an ASSERTION about how a "
                               "score was derived; the software cannot verify a score against the original "
                               "graded paper and never claims to have done so"),
        }

    # ---------------------------------------------------------------- final --
    def set_final(self, item_id: str, *, score: float, rubric: list[str] | None, note: str, source: str,
                  adjudicator: str, expected_item_revision: int) -> dict:
        if source not in FINAL_SOURCES:
            raise LabelError(f"unknown final source {source!r}")
        with self._conn(write=True) as c:
            it = c.execute("SELECT max_score, rubric_ids, revision, eligible, evidence_sha256 FROM items WHERE item_id=?",
                           (item_id,)).fetchone()
            if it is None:
                raise LabelError(f"unknown item {item_id!r}")
            if not int(it["eligible"]):
                raise LabelError("this item is not eligible for a human FINAL label — the deterministic "
                                 "policy score is authoritative")
            if int(it["revision"]) != int(expected_item_revision):
                raise StaleWrite(f"item revision is {it['revision']}, you loaded {expected_item_revision}; reload")
            score = float(score)
            if score < 0 or score > float(it["max_score"]) + 1e-9:
                raise LabelError(f"score {score} outside 0..{it['max_score']}")
            allowed = set(json.loads(it["rubric_ids"] or "[]"))
            rubric_clean = sorted({str(x) for x in (rubric or [])})
            if any(x not in allowed for x in rubric_clean):
                raise LabelError("unknown rubric item in final label")
            cur_fp = it["evidence_sha256"]
            labels = [self._label_dict(r, cur_fp) for r in c.execute(
                "SELECT * FROM labels WHERE item_id=? AND status='saved' ORDER BY grader", (item_id,))]
            if source == "agreement" and any(l["evidence_stale"] for l in labels):
                raise LabelError("cannot finalize from agreement: a contributing label predates an evidence "
                                 "change; the grader must re-review the corrected evidence first")
            # provenance names only labels made against the CURRENT evidence
            fresh = [l for l in labels if not l["evidence_stale"]]
            contributing = sorted(l["grader"] for l in fresh)
            revs = {l["grader"]: l["revision"] for l in fresh}
            ts = _now()
            # provenance of the SCORE this FINAL rests on (distinct from `source`,
            # which records how the FINAL was reached: agreement | adjudicated)
            gts = c.execute("SELECT label_source FROM labels WHERE item_id=? AND status='saved' "
                            "AND label_source IN " + _sql_tuple(AUTHORITATIVE_LABEL_SOURCES) +
                            " ORDER BY grader LIMIT 1", (item_id,)).fetchone()
            ground_truth_source = gts["label_source"] if gts else DEFAULT_LABEL_SOURCE
            c.execute("INSERT OR REPLACE INTO final_labels(item_id, score, rubric, note, source, adjudicator, "
                      "contributing_graders, from_revisions, finalized_at, schema_version, evidence_sha256, "
                      "ground_truth_source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                      (item_id, score, json.dumps(rubric_clean), note or "", source, adjudicator or "",
                       json.dumps(contributing), json.dumps(revs, sort_keys=True), ts, SCHEMA_VERSION, cur_fp,
                       ground_truth_source))
            c.execute("UPDATE items SET revision=revision+1 WHERE item_id=?", (item_id,))
            c.execute("DELETE FROM claims WHERE item_id=?", (item_id,))
            self._event(c, adjudicator or "admin", f"final_{source}", item_id, int(it["revision"]) + 1,
                        {"score": score, "rubric": rubric_clean, "evidence_sha256": cur_fp})
        return self.overview(item_id)["final"]

    def finalize_agreement(self, item_id: str, *, adjudicator: str = "admin",
                           expected_item_revision: int | None = None) -> dict:
        """FINAL from explicit agreement: requires >=2 identical FRESH saved labels."""
        ov = self.overview(item_id)
        if ov["agreement"] is not True:
            raise LabelError("no agreement to finalize (needs two or more identical saved labels made against "
                             "the current evidence)")
        l0 = next(l for l in ov["labels"] if l["status"] == "saved" and not l["evidence_stale"])
        return self.set_final(item_id, score=l0["score"], rubric=l0["rubric"], note="",
                              source="agreement", adjudicator=adjudicator,
                              expected_item_revision=ov["revision"] if expected_item_revision is None else expected_item_revision)

    def reopen(self, item_id: str, *, admin: str = "admin") -> None:
        with self._conn(write=True) as c:
            c.execute("DELETE FROM final_labels WHERE item_id=?", (item_id,))
            c.execute("UPDATE items SET revision=revision+1 WHERE item_id=?", (item_id,))
            self._event(c, admin, "reopen", item_id, None)

    def final_rows(self) -> list[dict]:
        with self._conn() as c:
            rows = []
            for r in c.execute("SELECT f.*, i.evidence_sha256 AS current_evidence_sha256 FROM final_labels f "
                               "JOIN items i ON i.item_id=f.item_id ORDER BY f.item_id"):
                d = dict(r)
                d["rubric"] = json.loads(d["rubric"] or "[]")
                d["contributing_graders"] = json.loads(d["contributing_graders"] or "[]")
                d["from_revisions"] = json.loads(d["from_revisions"] or "{}")
                d["ground_truth_source"] = d.get("ground_truth_source") or DEFAULT_LABEL_SOURCE
                d["authoritative"] = d["ground_truth_source"] in AUTHORITATIVE_LABEL_SOURCES
                d["evidence_stale"] = _is_stale(d.get("evidence_sha256"), d.pop("current_evidence_sha256", None),
                                                d["ground_truth_source"])
                rows.append(d)
            return rows

    def events(self, limit: int = 200) -> list[dict]:
        with self._conn() as c:
            return [dict(r) for r in c.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,))]

    def snapshot_to(self, dest: Path) -> None:
        """Consistent copy of the live database (SQLite online backup API)."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as src:
            dst = sqlite3.connect(str(dest))
            try:
                src.backup(dst)
            finally:
                dst.close()

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


__all__ = ["LabelDB", "StaleWrite", "StaleEvidence", "LabelError",
           "assert_not_live_database", "live_db_path", "LIVE_DB_OPT_IN", "LABEL_STATUSES", "FINAL_SOURCES", "CLAIM_TTL_S",
           "STATE_EVIDENCE_REVIEW", "STATE_AUTHORITATIVE", "LABEL_SOURCES", "DEFAULT_LABEL_SOURCE",
           "AUTHORITATIVE_LABEL_SOURCES"]
