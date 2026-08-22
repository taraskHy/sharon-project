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

States (derived, never stored as truth):
    UNLABELED            no saved label
    ASSIGNED             active claim, no saved label
    LABELED              >=1 saved label, fewer than wanted (or wanted=1, not final)
    AGREEMENT            >=2 saved labels, all identical (score + rubric), not final
    NEEDS_ADJUDICATION   >=2 saved labels that differ, or any flagged label, not final
    FINAL                a final_labels row exists
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from . import SCHEMA_VERSION

LABEL_STATUSES = ("saved", "skipped", "flagged")
FINAL_SOURCES = ("agreement", "adjudicated")
CLAIM_TTL_S = 30 * 60


class StaleWrite(RuntimeError):
    """The client's revision is older than the stored one (HTTP 409)."""


class LabelError(ValueError):
    """Invalid label (score out of range, unknown status, ...)."""


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


DDL = [
    f"""CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)""",
    """CREATE TABLE IF NOT EXISTS items (
        item_id TEXT PRIMARY KEY,
        max_score REAL NOT NULL,
        rubric_ids TEXT NOT NULL DEFAULT '[]',
        wanted_labels INTEGER NOT NULL DEFAULT 1,
        revision INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL)""",
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
        schema_version INTEGER NOT NULL)""",
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


class LabelDB:
    def __init__(self, path: str | Path, *, claim_ttl_s: int = CLAIM_TTL_S):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.claim_ttl_s = claim_ttl_s
        self._local = threading.local()
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            for ddl in DDL:
                c.execute(ddl)
            c.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))

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
        """The next item this grader should label: first items nobody has
        labeled, then items that still want another label from someone else;
        items actively claimed by another grader are deferred (not hidden)."""
        self.touch_grader(grader)
        with self._conn(write=True) as c:
            self._expire_claims(c)
            rows = c.execute("""
                SELECT i.item_id, i.wanted_labels,
                       (SELECT COUNT(*) FROM labels l WHERE l.item_id=i.item_id AND l.status='saved') AS n_saved,
                       (SELECT COUNT(*) FROM labels l WHERE l.item_id=i.item_id AND l.grader=? ) AS mine,
                       (SELECT status FROM labels l WHERE l.item_id=i.item_id AND l.grader=?) AS my_status,
                       (SELECT COUNT(*) FROM claims k WHERE k.item_id=i.item_id AND k.grader<>?) AS claimed_by_others,
                       (SELECT COUNT(*) FROM final_labels f WHERE f.item_id=i.item_id) AS is_final
                FROM items i ORDER BY i.item_id""", (grader, grader, grader)).fetchall()
            def _eligible(r) -> bool:
                if r["is_final"]:
                    return False
                if r["mine"] and not (include_skipped and r["my_status"] == "skipped"):
                    return False
                return r["n_saved"] < r["wanted_labels"]
            pool = [r for r in rows if _eligible(r)]
            if not pool:
                return None
            # priority: (no label yet) > (needs another label); unclaimed before claimed-by-others
            pool.sort(key=lambda r: (r["n_saved"] > 0, r["claimed_by_others"] > 0, r["item_id"]))
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
            return self._label_dict(r) if r else None

    @staticmethod
    def _label_dict(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["rubric"] = json.loads(d.get("rubric") or "[]")
        return d

    def save_label(self, item_id: str, grader: str, *, score: float | None, rubric: list[str] | None,
                   note: str = "", status: str = "saved", flag_reason: str = "",
                   expected_revision: int = 0) -> dict:
        """Create/update THIS grader's label for the item. ``expected_revision``
        is the revision the client loaded (0 = no label yet). Stale → StaleWrite."""
        if status not in LABEL_STATUSES:
            raise LabelError(f"unknown status {status!r}")
        self.touch_grader(grader)
        with self._conn(write=True) as c:
            it = c.execute("SELECT max_score, rubric_ids, revision FROM items WHERE item_id=?", (item_id,)).fetchone()
            if it is None:
                raise LabelError(f"unknown item {item_id!r}")
            if c.execute("SELECT 1 FROM final_labels WHERE item_id=?", (item_id,)).fetchone():
                raise LabelError("this item already has a FINAL label; ask the admin to reopen it")
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
                c.execute("UPDATE labels SET score=?, rubric=?, note=?, status=?, flag_reason=?, revision=?, updated_at=? "
                          "WHERE item_id=? AND grader=?",
                          (score, json.dumps(rubric_clean), note or "", status, flag_reason or "", new_rev, ts,
                           item_id, grader))
            else:
                c.execute("INSERT INTO labels(item_id, grader, score, rubric, note, status, flag_reason, revision, "
                          "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                          (item_id, grader, score, json.dumps(rubric_clean), note or "", status, flag_reason or "",
                           new_rev, ts, ts))
            c.execute("UPDATE items SET revision=revision+1 WHERE item_id=?", (item_id,))
            c.execute("DELETE FROM claims WHERE item_id=? AND grader=?", (item_id, grader))
            self._event(c, grader, f"label_{status}", item_id, new_rev,
                        {"score": score, "rubric": rubric_clean, "flag_reason": flag_reason or ""})
            r = c.execute("SELECT * FROM labels WHERE item_id=? AND grader=?", (item_id, grader)).fetchone()
            return self._label_dict(r)

    def labels_for_item(self, item_id: str) -> list[dict]:
        with self._conn() as c:
            return [self._label_dict(r) for r in c.execute(
                "SELECT * FROM labels WHERE item_id=? ORDER BY grader", (item_id,))]

    def my_items(self, grader: str) -> dict[str, list[str]]:
        with self._conn() as c:
            out: dict[str, list[str]] = {"saved": [], "skipped": [], "flagged": []}
            for r in c.execute("SELECT item_id, status FROM labels WHERE grader=? ORDER BY item_id", (grader,)):
                out[r["status"]].append(r["item_id"])
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
            labels = [self._label_dict(r) for r in c.execute("SELECT * FROM labels WHERE item_id=? ORDER BY grader", (item_id,))]
            final = c.execute("SELECT * FROM final_labels WHERE item_id=?", (item_id,)).fetchone()
            claims = [dict(r) for r in c.execute("SELECT grader, claimed_at FROM claims WHERE item_id=?", (item_id,))]
        saved = [l for l in labels if l["status"] == "saved"]
        flagged = [l for l in labels if l["status"] == "flagged"]
        agree = self._agreement(saved)
        if final:
            state = "FINAL"
        elif (len(saved) >= 2 and agree is False) or flagged:
            state = "NEEDS_ADJUDICATION"
        elif len(saved) >= 2 and agree:
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
        return {"item_id": item_id, "revision": it["revision"], "max_score": it["max_score"],
                "wanted_labels": it["wanted_labels"], "state": state, "n_saved": len(saved),
                "n_skipped": sum(1 for l in labels if l["status"] == "skipped"), "n_flagged": len(flagged),
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
                per_grader.setdefault(r["grader"], {"saved": 0, "skipped": 0, "flagged": 0})[r["status"]] = r["n"]
            mode = c.execute("SELECT value FROM meta WHERE key='double_label_mode'").fetchone()
        return {
            "total_items": len(ovs),
            "unlabeled": by_state.get("UNLABELED", 0) + by_state.get("ASSIGNED", 0),
            "assigned": by_state.get("ASSIGNED", 0),
            "labels_completed": sum(o["n_saved"] for o in ovs),
            "singly_labeled": sum(1 for o in ovs if o["n_saved"] == 1),
            "double_labeled": sum(1 for o in ovs if o["n_saved"] >= 2),
            "agreements": sum(1 for o in ovs if o["agreement"] is True),
            "disagreements": sum(1 for o in ovs if o["agreement"] is False),
            "flagged": sum(1 for o in ovs if o["n_flagged"]),
            "needs_adjudication": by_state.get("NEEDS_ADJUDICATION", 0),
            "final": by_state.get("FINAL", 0),
            "by_state": by_state, "per_grader": per_grader,
            "double_label_mode": (mode["value"] if mode else "none"),
            "graders": self.graders(),
        }

    def progress(self, grader: str) -> dict[str, Any]:
        ovs = self.all_overviews()
        mine = self.my_items(grader)
        remaining = sum(1 for o in ovs if o["state"] != "FINAL" and o["n_saved"] < o["wanted_labels"]
                        and o["item_id"] not in set(mine["saved"] + mine["skipped"] + mine["flagged"]))
        return {"total_items": len(ovs), "labels_completed": sum(o["n_saved"] for o in ovs),
                "items_with_any_label": sum(1 for o in ovs if o["n_saved"]),
                "my_saved": len(mine["saved"]), "my_skipped": len(mine["skipped"]), "my_flagged": len(mine["flagged"]),
                "remaining_for_me": remaining}

    # ---------------------------------------------------------------- final --
    def set_final(self, item_id: str, *, score: float, rubric: list[str] | None, note: str, source: str,
                  adjudicator: str, expected_item_revision: int) -> dict:
        if source not in FINAL_SOURCES:
            raise LabelError(f"unknown final source {source!r}")
        with self._conn(write=True) as c:
            it = c.execute("SELECT max_score, rubric_ids, revision FROM items WHERE item_id=?", (item_id,)).fetchone()
            if it is None:
                raise LabelError(f"unknown item {item_id!r}")
            if int(it["revision"]) != int(expected_item_revision):
                raise StaleWrite(f"item revision is {it['revision']}, you loaded {expected_item_revision}; reload")
            score = float(score)
            if score < 0 or score > float(it["max_score"]) + 1e-9:
                raise LabelError(f"score {score} outside 0..{it['max_score']}")
            allowed = set(json.loads(it["rubric_ids"] or "[]"))
            rubric_clean = sorted({str(x) for x in (rubric or [])})
            if any(x not in allowed for x in rubric_clean):
                raise LabelError("unknown rubric item in final label")
            labels = [self._label_dict(r) for r in c.execute(
                "SELECT * FROM labels WHERE item_id=? AND status='saved' ORDER BY grader", (item_id,))]
            contributing = sorted(l["grader"] for l in labels)
            revs = {l["grader"]: l["revision"] for l in labels}
            ts = _now()
            c.execute("INSERT OR REPLACE INTO final_labels(item_id, score, rubric, note, source, adjudicator, "
                      "contributing_graders, from_revisions, finalized_at, schema_version) VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (item_id, score, json.dumps(rubric_clean), note or "", source, adjudicator or "",
                       json.dumps(contributing), json.dumps(revs, sort_keys=True), ts, SCHEMA_VERSION))
            c.execute("UPDATE items SET revision=revision+1 WHERE item_id=?", (item_id,))
            c.execute("DELETE FROM claims WHERE item_id=?", (item_id,))
            self._event(c, adjudicator or "admin", f"final_{source}", item_id, int(it["revision"]) + 1,
                        {"score": score, "rubric": rubric_clean})
        return self.overview(item_id)["final"]

    def finalize_agreement(self, item_id: str, *, adjudicator: str = "admin",
                           expected_item_revision: int | None = None) -> dict:
        """FINAL from explicit agreement: requires >=2 identical saved labels."""
        ov = self.overview(item_id)
        if ov["agreement"] is not True:
            raise LabelError("no agreement to finalize (needs two or more identical saved labels)")
        l0 = next(l for l in ov["labels"] if l["status"] == "saved")
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
            for r in c.execute("SELECT * FROM final_labels ORDER BY item_id"):
                d = dict(r)
                d["rubric"] = json.loads(d["rubric"] or "[]")
                d["contributing_graders"] = json.loads(d["contributing_graders"] or "[]")
                d["from_revisions"] = json.loads(d["from_revisions"] or "{}")
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


__all__ = ["LabelDB", "StaleWrite", "LabelError", "LABEL_STATUSES", "FINAL_SOURCES", "CLAIM_TTL_S"]
