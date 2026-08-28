"""The blind review web app (Starlette; uvicorn-served).

Reviewer surface (BLIND for the whole campaign):
    GET  /                       reviewer page
    POST /api/session {name, token?}   set reviewer cookie (invite token optional)
    GET  /api/me
    POST /api/next               claim the next case (fewest-reviews-first, TTL,
                                 never the same case twice for one reviewer)
    GET  /api/items/{id}         blind payload + MY decision only
    POST /api/items/{id}/decision {verdict, confidence, issue, note,
                                   expected_revision, evidence_sha256}
    POST /api/items/{id}/skip | /flag
    GET  /api/images/{id}/{n}    answer crop
    GET  /api/pages/{id}         red-masked source page
    GET  /api/my-items

The pre-decision payload NEVER contains: the original instructor score or
verdict, the local model's score/verdict/evidence/justification/decision,
A/B/C/D audit decisions, other reviewers' decisions, agreement counts, split
names, derivability, or finals — not hidden in the UI: absent from the JSON.
Nothing is revealed after submit either; the reviewer stays blind for the
entire campaign. No model or provider is ever called by this app.

Admin surface (key-protected; the ONLY place comparisons exist):
    GET  /admin                  dashboard page
    GET  /api/admin/summary      dashboard numbers
    GET  /api/admin/items        per-case states
    GET  /api/admin/items/{id}   everything: blind payload + reviews +
                                 instructor reference + model proposal
    POST /api/admin/items/{id}/adjudicate   adjudicated_human_reference
    POST /api/admin/items/{id}/reopen
    GET  /api/admin/compare      three-way analysis (gated on completion)
    GET  /api/admin/export       deterministic campaign results
    POST /api/admin/backup       WAL-safe online backup
    GET  /api/admin/events
"""
from __future__ import annotations

import json
import os
import secrets
import urllib.parse
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from labeling_app.backup import make_backup
from labeling_app.bundle import Bundle
from labeling_app.db import LabelDB, LabelError, StaleEvidence, StaleWrite

from . import (CAMPAIGN, CONFIDENCES, ISSUES, VERDICTS, assert_not_live_review_db,
               decision_note, parse_note, score_to_verdict, verdict_to_score)

WEB = Path(__file__).resolve().parent / "web"
COOKIE = "review46_reviewer"
ADMIN_COOKIE = "review46_admin"


# ------------------------------------------------------------- helpers ------

def _reviewer(request: Request) -> str | None:
    raw = request.cookies.get(COOKIE)
    if not raw:
        return None
    return urllib.parse.unquote(raw).strip() or None


def _need_reviewer(request: Request):
    r = _reviewer(request)
    if not r:
        return JSONResponse({"error": "no reviewer name; enter your name first"}, status_code=401)
    return r


def _is_admin(request: Request) -> bool:
    key = request.app.state.admin_key
    if not key:
        return False                                  # review46 ALWAYS requires a key
    supplied = (request.headers.get("x-admin-key") or request.query_params.get("key")
                or request.cookies.get(ADMIN_COOKIE))
    return secrets.compare_digest(str(supplied or ""), key)


def _need_admin(request: Request):
    if not _is_admin(request):
        return JSONResponse({"error": "admin key required"}, status_code=403)
    return None


async def _json(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return {}
    return body if isinstance(body, dict) else {}


def _review_view(label: dict | None, max_score: float) -> dict | None:
    """A reviewer's OWN decision, decoded (never anyone else's)."""
    if label is None:
        return None
    note = parse_note(label.get("note"))
    return {"verdict": note.get("verdict") or (score_to_verdict(label["score"], max_score)
                                               if label.get("score") is not None else None),
            "confidence": note.get("confidence"), "issue": note.get("issue"),
            "text": note.get("text") or "", "status": label.get("status"),
            "flag_reason": label.get("flag_reason") or "", "revision": label["revision"],
            "updated_at": label.get("updated_at")}


def _blind_item_view(request: Request, item_id: str, reviewer: str) -> dict | None:
    """EXACTLY what a reviewer may see before (and after) deciding. Built from
    the bundle's blind payload + the reviewer's OWN saved decision. No final /
    agreement / other-reviewer / instructor / model / audit / split field is
    ever added here — tests scan this payload."""
    bundle: Bundle = request.app.state.bundle
    db: LabelDB = request.app.state.db
    payload = bundle.grader_payload(item_id)
    if payload is None:
        return None
    mine = db.get_label(item_id, reviewer)
    payload["my_review"] = _review_view(mine, float(payload.get("max_score") or 4.0))
    payload["label_revision"] = mine["revision"] if mine else 0
    return payload


# --------------------------------------------------------------- pages ------

async def page_reviewer(request: Request) -> Response:
    return HTMLResponse((WEB / "reviewer.html").read_text(encoding="utf-8"))


async def page_admin(request: Request) -> Response:
    if not _is_admin(request):
        return HTMLResponse("<h3>admin key required</h3><p>open /admin?key=YOUR_KEY</p>", status_code=403)
    resp = HTMLResponse((WEB / "admin.html").read_text(encoding="utf-8"))
    key = request.query_params.get("key")
    if key and request.app.state.admin_key and secrets.compare_digest(key, request.app.state.admin_key):
        resp.set_cookie(ADMIN_COOKIE, key, max_age=30 * 24 * 3600, samesite="lax", httponly=True)
    return resp


# ------------------------------------------------------------- session ------

async def api_session(request: Request) -> Response:
    body = await _json(request)
    name = str(body.get("name", "")).strip()
    token_required = request.app.state.invite_token
    if token_required and not secrets.compare_digest(str(body.get("token") or ""), token_required):
        return JSONResponse({"error": "invite token required"}, status_code=403)
    db: LabelDB = request.app.state.db
    try:
        db.touch_grader(name)
    except LabelError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    resp = JSONResponse({"reviewer": name})
    resp.set_cookie(COOKIE, urllib.parse.quote(name), max_age=180 * 24 * 3600, samesite="lax")
    return resp


async def api_me(request: Request) -> Response:
    r = _reviewer(request)
    db: LabelDB = request.app.state.db
    return JSONResponse({"reviewer": r, "progress": db.progress(r) if r else None,
                         "campaign": {"id": CAMPAIGN, "cases": len(request.app.state.bundle.items),
                                      "reviews_per_case": 2},
                         "invite_token_required": bool(request.app.state.invite_token)})


# --------------------------------------------------------------- items ------

async def api_next(request: Request) -> Response:
    r = _need_reviewer(request)
    if isinstance(r, Response):
        return r
    body = await _json(request)
    db: LabelDB = request.app.state.db
    item_id = db.claim_next(r, include_skipped=bool(body.get("include_skipped")))
    if item_id is None:
        return JSONResponse({"item": None, "done": True, "progress": db.progress(r)})
    return JSONResponse({"item": _blind_item_view(request, item_id, r), "done": False,
                         "progress": db.progress(r)})


async def api_item(request: Request) -> Response:
    r = _need_reviewer(request)
    if isinstance(r, Response):
        return r
    view = _blind_item_view(request, request.path_params["item_id"], r)
    if view is None:
        return JSONResponse({"error": "unknown item"}, status_code=404)
    return JSONResponse({"item": view, "progress": request.app.state.db.progress(r)})


async def api_decision(request: Request) -> Response:
    r = _need_reviewer(request)
    if isinstance(r, Response):
        return r
    item_id = request.path_params["item_id"]
    bundle: Bundle = request.app.state.bundle
    it = bundle.item(item_id)
    if it is None:
        return JSONResponse({"error": "unknown item"}, status_code=404)
    body = await _json(request)
    verdict = str(body.get("verdict") or "")
    confidence = str(body.get("confidence") or "")
    issue = str(body.get("issue") or "none")
    if verdict not in VERDICTS:
        return JSONResponse({"error": f"verdict must be one of {VERDICTS}"}, status_code=400)
    if confidence not in CONFIDENCES:
        return JSONResponse({"error": f"confidence must be one of {CONFIDENCES}"}, status_code=400)
    if issue not in ISSUES:
        return JSONResponse({"error": f"issue must be one of {ISSUES}"}, status_code=400)
    max_score = float(it.get("max_score") or 4.0)
    db: LabelDB = request.app.state.db
    try:
        label = db.save_label(
            item_id, r, score=verdict_to_score(verdict, max_score), rubric=[],
            note=decision_note(verdict, confidence, issue, str(body.get("note") or "")),
            status="saved",
            expected_revision=int(body.get("expected_revision") or 0),
            client_evidence_sha256=(str(body["evidence_sha256"]) if body.get("evidence_sha256") else None))
    except StaleEvidence as e:
        return JSONResponse({"error": str(e), "stale": True, "stale_evidence": True}, status_code=409)
    except StaleWrite as e:
        return JSONResponse({"error": str(e), "stale": True}, status_code=409)
    except LabelError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    # deliberately NO reveal of anything after submit — the reviewer stays blind
    return JSONResponse({"my_review": _review_view(label, max_score),
                         "label_revision": label["revision"],
                         "progress": db.progress(r)})


async def api_skip(request: Request) -> Response:
    return await _skip_or_flag(request, "skipped")


async def api_flag(request: Request) -> Response:
    return await _skip_or_flag(request, "flagged")


async def _skip_or_flag(request: Request, status: str) -> Response:
    r = _need_reviewer(request)
    if isinstance(r, Response):
        return r
    item_id = request.path_params["item_id"]
    if request.app.state.bundle.item(item_id) is None:
        return JSONResponse({"error": "unknown item"}, status_code=404)
    body = await _json(request)
    db: LabelDB = request.app.state.db
    try:
        label = db.save_label(item_id, r, score=None, rubric=[],
                              note=str(body.get("note") or ""), status=status,
                              flag_reason=str(body.get("reason") or ""),
                              expected_revision=int(body.get("expected_revision") or 0))
    except StaleWrite as e:
        return JSONResponse({"error": str(e), "stale": True}, status_code=409)
    except LabelError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"status": label["status"], "label_revision": label["revision"],
                         "progress": db.progress(r)})


async def api_image(request: Request) -> Response:
    try:
        n = int(request.path_params["n"])
    except ValueError:
        return Response(status_code=404)
    p = request.app.state.bundle.image_path(request.path_params["item_id"], n)
    if p is None:
        return Response(status_code=404)
    return FileResponse(str(p), media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


async def api_page(request: Request) -> Response:
    p = request.app.state.bundle.page_path(request.path_params["item_id"])
    if p is None:
        return JSONResponse({"error": "no source page available for this item"}, status_code=404)
    return FileResponse(str(p), media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


async def api_my_items(request: Request) -> Response:
    r = _need_reviewer(request)
    if isinstance(r, Response):
        return r
    return JSONResponse(request.app.state.db.my_items(r))


async def api_health(request: Request) -> Response:
    return JSONResponse({"ok": True, "campaign": CAMPAIGN,
                         "cases": len(request.app.state.bundle.items), "ai_calls": 0})


# ------------------------------------------------------- consensus math -----

def _fresh_saved(overview: dict) -> list[dict]:
    return [l for l in overview["labels"]
            if l["status"] == "saved" and not l.get("evidence_stale")]


def _case_consensus(request: Request, item_id: str) -> dict[str, Any]:
    """Independent-review consensus for ONE case (admin-side only)."""
    db: LabelDB = request.app.state.db
    bundle: Bundle = request.app.state.bundle
    it = bundle.item(item_id) or {}
    max_score = float(it.get("max_score") or 4.0)
    ov = db.overview(item_id)
    saved = _fresh_saved(ov)
    reviews = [{"reviewer": l["grader"], **(_review_view(l, max_score) or {})} for l in saved]
    verdicts = [r["verdict"] for r in reviews if r.get("verdict")]
    state = "PENDING"
    consensus = None
    if len(verdicts) >= 2:
        if len(set(verdicts)) == 1:
            state, consensus = "CONSENSUS", verdicts[0]
        else:
            state = "NEEDS_ADJUDICATION"
    elif len(verdicts) == 1:
        state = "ONE_REVIEW"
    flagged = [l for l in ov["labels"] if l["status"] == "flagged"]
    if flagged and state in ("PENDING", "ONE_REVIEW"):
        state = "NEEDS_ADJUDICATION"
    adjudicated = None
    if ov.get("final"):
        f = ov["final"]
        note = parse_note(f.get("note"))
        adjudicated = {"verdict": note.get("verdict") or score_to_verdict(f["score"], max_score),
                       "score": f["score"], "adjudicator": f.get("adjudicator"),
                       "note": note.get("text") or f.get("note"),
                       "contributing": note.get("contributing"),
                       "at": f.get("finalized_at") or f.get("created_at")}
        state = "ADJUDICATED"
    human_reference = (adjudicated["verdict"] if adjudicated
                       else consensus if state == "CONSENSUS" else None)
    return {"item_id": item_id, "state": state, "reviews": reviews,
            "n_fresh_saved": len(saved), "flagged": [l["grader"] for l in flagged],
            "consensus_verdict": consensus, "adjudicated": adjudicated,
            "human_reference_verdict": human_reference,
            "item_revision": ov["revision"], "max_score": max_score}


def _all_consensus(request: Request) -> list[dict]:
    return [_case_consensus(request, i["item_id"]) for i in request.app.state.bundle.items]


def _campaign_complete(rows: list[dict]) -> bool:
    """Every case has two fresh independent reviews, or an explicit
    adjudicated resolution (the 'marked unresolved' path is an adjudication)."""
    return all(r["n_fresh_saved"] >= 2 or r["state"] == "ADJUDICATED" for r in rows)


# --------------------------------------------------------------- admin ------

async def api_admin_summary(request: Request) -> Response:
    if (err := _need_admin(request)):
        return err
    db: LabelDB = request.app.state.db
    rows = _all_consensus(request)
    n = len(rows)
    by_reviews = {k: sum(1 for r in rows if min(r["n_fresh_saved"], 2) == k) for k in (0, 1, 2)}
    conf: dict[str, int] = {c: 0 for c in CONFIDENCES}
    issues: dict[str, int] = {i: 0 for i in ISSUES}
    for r in rows:
        for rv in r["reviews"]:
            if rv.get("confidence") in conf:
                conf[rv["confidence"]] += 1
            if rv.get("issue") in issues:
                issues[rv["issue"]] += 1
    two = [r for r in rows if r["n_fresh_saved"] >= 2]
    agree = sum(1 for r in two if r["consensus_verdict"])
    decisions = sum(r["n_fresh_saved"] for r in rows)
    summary = {
        "campaign": CAMPAIGN,
        "cases": n, "required_reviews": n * 2, "completed_review_decisions": decisions,
        "cases_by_review_count": by_reviews,
        "reviewer_agreement": {"cases_with_two_reviews": len(two), "agreements": agree,
                               "rate_pct": round(100 * agree / len(two), 1) if two else None},
        "needs_adjudication": sorted(r["item_id"] for r in rows if r["state"] == "NEEDS_ADJUDICATION"),
        "adjudicated": sum(1 for r in rows if r["state"] == "ADJUDICATED"),
        "confidence_counts": conf, "issue_counts": issues,
        "reviewers": [{"name": g, **db.progress(g)} for g in db.graders()],
        "claims_active": _active_claims(db),
        "mean_review_time_s": _mean_review_time(db),
        "campaign_complete": _campaign_complete(rows),
        "comparisons": ("available at /api/admin/compare once every case has two "
                        "independent reviews or an adjudication"),
    }
    if summary["campaign_complete"]:
        summary["comparison_preview"] = _comparisons(request, rows)["headline"]
    return JSONResponse(summary)


def _active_claims(db: LabelDB) -> int:
    """Unexpired claims. Read-only peek at the claims table (the DB module
    exposes no public counter; a read through its own connection helper keeps
    WAL/busy-timeout semantics)."""
    import time as _t
    with db._conn() as c:  # noqa: SLF001 — read-only, same package family
        return int(c.execute("SELECT COUNT(*) FROM claims WHERE expires_at > ?",
                             (_t.time(),)).fetchone()[0])


def _mean_review_time(db: LabelDB) -> float | None:
    """claim -> label wall time per (item, reviewer), from the append-only
    events trail; None until at least three measurements exist."""
    import datetime as dt
    events = db.events(limit=100000)
    claims: dict[tuple[str, str], str] = {}
    spans: list[float] = []
    for e in reversed(events):                    # oldest first
        key = (e.get("item_id") or "", e.get("grader") or "")
        if e["action"] == "claim":
            claims[key] = e["ts"]
        elif e["action"] == "label_saved" and key in claims:
            try:
                t0 = dt.datetime.fromisoformat(claims.pop(key))
                t1 = dt.datetime.fromisoformat(e["ts"])
                s = (t1 - t0).total_seconds()
                if 0 < s < 3600:
                    spans.append(s)
            except ValueError:
                pass
    return round(sum(spans) / len(spans), 1) if len(spans) >= 3 else None


async def api_admin_items(request: Request) -> Response:
    if (err := _need_admin(request)):
        return err
    bundle: Bundle = request.app.state.bundle
    rows = _all_consensus(request)
    ref = request.app.state.instructor_reference
    prop = request.app.state.model_proposals
    out = [{"item_id": r["item_id"], "case_id": bundle.id_map.get(r["item_id"]),
            "state": r["state"], "n_reviews": r["n_fresh_saved"],
            "reviewers": [rv["reviewer"] for rv in r["reviews"]],
            "human_reference_verdict": r["human_reference_verdict"],
            "instructor_derived_verdict": (ref.get(r["item_id"]) or {}).get("instructor_derived_verdict"),
            "model_verdict": (prop.get(r["item_id"]) or {}).get("verdict"),
            "evidence_issue_flag": (ref.get(r["item_id"]) or {}).get("evidence_issue_flag"),
            "item_revision": r["item_revision"]} for r in rows]
    state = request.query_params.get("state")
    if state:
        out = [o for o in out if o["state"] == state]
    return JSONResponse({"items": out})


async def api_admin_item(request: Request) -> Response:
    if (err := _need_admin(request)):
        return err
    item_id = request.path_params["item_id"]
    bundle: Bundle = request.app.state.bundle
    if bundle.item(item_id) is None:
        return JSONResponse({"error": "unknown item"}, status_code=404)
    view = _case_consensus(request, item_id)
    view["case_id"] = bundle.id_map.get(item_id)
    view["content"] = bundle.grader_payload(item_id)
    view["instructor_reference"] = request.app.state.instructor_reference.get(item_id)
    view["model_proposal"] = request.app.state.model_proposals.get(item_id)
    return JSONResponse(view)


async def api_admin_adjudicate(request: Request) -> Response:
    if (err := _need_admin(request)):
        return err
    item_id = request.path_params["item_id"]
    bundle: Bundle = request.app.state.bundle
    it = bundle.item(item_id)
    if it is None:
        return JSONResponse({"error": "unknown item"}, status_code=404)
    body = await _json(request)
    verdict = str(body.get("verdict") or "")
    if verdict not in VERDICTS:
        return JSONResponse({"error": f"verdict must be one of {VERDICTS}"}, status_code=400)
    max_score = float(it.get("max_score") or 4.0)
    cons = _case_consensus(request, item_id)
    note = json.dumps({"verdict": verdict, "text": str(body.get("note") or ""),
                       "contributing": [{"reviewer": r["reviewer"], "verdict": r.get("verdict"),
                                         "revision": r.get("revision")} for r in cons["reviews"]]},
                      ensure_ascii=False, sort_keys=True)
    db: LabelDB = request.app.state.db
    try:
        final = db.set_final(item_id, score=verdict_to_score(verdict, max_score), rubric=[],
                             note=note, source="adjudicated",
                             adjudicator=str(body.get("adjudicator") or "admin"),
                             expected_item_revision=int(body.get("expected_item_revision") or -1))
    except StaleWrite as e:
        return JSONResponse({"error": str(e), "stale": True}, status_code=409)
    except (LabelError, TypeError, ValueError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"final": final, "case": _case_consensus(request, item_id)})


async def api_admin_reopen(request: Request) -> Response:
    if (err := _need_admin(request)):
        return err
    item_id = request.path_params["item_id"]
    request.app.state.db.reopen(item_id)
    return JSONResponse({"case": _case_consensus(request, item_id)})


def _comparisons(request: Request, rows: list[dict]) -> dict[str, Any]:
    """Phase-15 analysis: model vs human consensus (A), instructor vs human
    consensus (B), model vs instructor (C), three-way (D). Human reference =
    two-review consensus or adjudicated verdict. Instructor-side comparisons
    use the derived verdict where derivable; cases without one are reported,
    never guessed. Nothing here declares a source universally correct."""
    ref = request.app.state.instructor_reference
    prop = request.app.state.model_proposals

    def conf_matrix(pairs):
        m: dict[str, dict[str, int]] = {}
        for a, b in pairs:
            m.setdefault(a, {}).setdefault(b, 0)
            m[a][b] += 1
        return m

    def block(pairs, name_a, name_b):
        n = len(pairs)
        agree = sum(1 for a, b in pairs if a == b)
        up = sum(1 for a, b in pairs if VERDICTS.index(b) > VERDICTS.index(a))
        down = sum(1 for a, b in pairs if VERDICTS.index(b) < VERDICTS.index(a))
        recalls = []
        for cls in VERDICTS:
            support = sum(1 for a, _ in pairs if a == cls)
            if support:
                recalls.append(sum(1 for a, b in pairs if a == cls and b == cls) / support)
        f1s = []
        for cls in VERDICTS:
            support = sum(1 for a, _ in pairs if a == cls)
            predicted = sum(1 for _, b in pairs if b == cls)
            if not support and not predicted:
                continue
            tp = sum(1 for a, b in pairs if a == b == cls)
            prec = tp / predicted if predicted else 0.0
            rec = tp / support if support else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        return {"cases": n, "agreement": agree,
                "agreement_pct": round(100 * agree / n, 1) if n else None,
                f"{name_b}_higher_than_{name_a}": up, f"{name_b}_lower_than_{name_a}": down,
                "confusion": conf_matrix(pairs),
                "balanced_accuracy": round(sum(recalls) / len(recalls), 4) if recalls else None,
                "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else None}

    human = {r["item_id"]: r["human_reference_verdict"] for r in rows if r["human_reference_verdict"]}
    model = {i: p["verdict"] for i, p in prop.items() if p.get("verdict")}
    instr = {i: v["instructor_derived_verdict"] for i, v in ref.items()
             if v.get("instructor_derived_verdict")}

    mh = [(human[i], model[i]) for i in human if i in model]
    ih = [(human[i], instr[i]) for i in human if i in instr]
    mi = [(instr[i], model[i]) for i in instr if i in model]
    three = {"all_agree": 0, "human_model_agree_only": 0, "human_instructor_agree_only": 0,
             "model_instructor_agree_only": 0, "all_disagree": 0}
    for i in human:
        if i in model and i in instr:
            h, m, s = human[i], model[i], instr[i]
            if h == m == s:
                three["all_agree"] += 1
            elif h == m:
                three["human_model_agree_only"] += 1
            elif h == s:
                three["human_instructor_agree_only"] += 1
            elif m == s:
                three["model_instructor_agree_only"] += 1
            else:
                three["all_disagree"] += 1
    out = {
        "A_model_vs_human_consensus": block(mh, "human", "model"),
        "B_instructor_vs_human_consensus": block(ih, "human", "instructor"),
        "C_model_vs_instructor": block(mi, "instructor", "model"),
        "D_three_way": three,
        "human_reference_cases": len(human),
        "cases_without_derivable_instructor_verdict": sum(
            1 for i in human if i not in instr),
        "caveats": ["no source is declared universally correct from majority",
                    "seen exams only — no generalization claim; HELD_OUT remains sealed",
                    "instructor-side comparisons cover derivable cases only"],
    }
    out["headline"] = {"model_vs_human_pct": out["A_model_vs_human_consensus"]["agreement_pct"],
                       "instructor_vs_human_pct": out["B_instructor_vs_human_consensus"]["agreement_pct"],
                       "model_vs_instructor_pct": out["C_model_vs_instructor"]["agreement_pct"]}
    return out


async def api_admin_compare(request: Request) -> Response:
    if (err := _need_admin(request)):
        return err
    rows = _all_consensus(request)
    if not _campaign_complete(rows) and not request.query_params.get("partial"):
        return JSONResponse({
            "campaign_complete": False,
            "error": "comparative metrics are withheld until every case has two independent "
                     "reviews or an adjudicated resolution (pass ?partial=1 for an "
                     "explicitly-partial preview over consensus-complete cases only)"},
            status_code=409)
    out = _comparisons(request, rows)
    out["campaign_complete"] = _campaign_complete(rows)
    out["partial"] = not out["campaign_complete"]
    return JSONResponse(out)


async def api_admin_export(request: Request) -> Response:
    if (err := _need_admin(request)):
        return err
    data = _export_payload(request)
    path = request.app.state.data_dir / "exports" / "campaign_results.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1, sort_keys=True),
                    encoding="utf-8", newline="\n")
    return JSONResponse(data, headers={"Content-Disposition": 'attachment; filename="campaign_results.json"',
                                       "X-Export-Path": str(path)})


def _export_payload(request: Request) -> dict:
    """Deterministic full-campaign export: every source separately, nothing
    merged, nothing overwritten."""
    bundle: Bundle = request.app.state.bundle
    rows = _all_consensus(request)
    ref = request.app.state.instructor_reference
    prop = request.app.state.model_proposals
    cases = []
    for r in sorted(rows, key=lambda r: bundle.id_map.get(r["item_id"], "")):
        i = r["item_id"]
        cases.append({
            "case_id": bundle.id_map.get(i), "item_id": i, "state": r["state"],
            "original_instructor_reference": ref.get(i),
            "local_model_proposal": prop.get(i),
            "independent_human_reviews": r["reviews"],
            "adjudicated_human_reference": r["adjudicated"],
            "human_reference_verdict": r["human_reference_verdict"],
        })
    return {"campaign": CAMPAIGN, "schema_version": 1,
            "campaign_complete": _campaign_complete(rows), "cases": cases}


async def api_admin_backup(request: Request) -> Response:
    if (err := _need_admin(request)):
        return err
    body = await _json(request)
    copy_to = body.get("copy_to") or request.app.state.backup_copy_to
    out = make_backup(request.app.state.db, request.app.state.bundle, request.app.state.data_dir,
                      copy_to=Path(copy_to) if copy_to else None)
    return JSONResponse(out)


async def api_admin_events(request: Request) -> Response:
    if (err := _need_admin(request)):
        return err
    return JSONResponse({"events": request.app.state.db.events(int(request.query_params.get("limit", "200")))})


# -------------------------------------------------------------- factory -----

def create_app(*, data_dir: Path, bundle_dir: Path | None = None, admin_key: str | None = None,
               invite_token: str | None = None, backup_copy_to: Path | None = None) -> Starlette:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    bundle = Bundle(bundle_dir or (data_dir / "bundle"))
    bundle.verify_evidence()
    priv = bundle.root / "private"
    campaign_meta = json.loads((priv / "campaign.json").read_text(encoding="utf-8"))
    if campaign_meta.get("campaign") != CAMPAIGN or campaign_meta.get("held_out") != 0:
        raise RuntimeError("bundle is not the SEEN-46 campaign bundle")
    db_path = data_dir / "labels.db"
    assert_not_live_review_db(db_path)
    db = LabelDB(db_path)
    db.load_items(bundle.items)
    # every case is human-reviewable (reviewers judge the EXPLANATION;
    # selection correctness composes later) and wants TWO independent reviews
    db.sync_eligibility([i["item_id"] for i in bundle.items], [], eligibility_known=True)
    db.sync_evidence(bundle.fingerprints)
    db.set_wanted_labels("all", n=2)
    key = admin_key or os.environ.get("REVIEW46_ADMIN_KEY")
    if not key:
        key_file = data_dir / "admin_key.txt"
        if key_file.exists():
            key = key_file.read_text(encoding="utf-8").strip()
        else:
            key = secrets.token_urlsafe(24)
            key_file.write_text(key, encoding="utf-8")
    routes = [
        Route("/", page_reviewer),
        Route("/admin", page_admin),
        Route("/api/health", api_health),
        Route("/api/session", api_session, methods=["POST"]),
        Route("/api/me", api_me),
        Route("/api/next", api_next, methods=["POST"]),
        Route("/api/my-items", api_my_items),
        Route("/api/items/{item_id}", api_item),
        Route("/api/items/{item_id}/decision", api_decision, methods=["POST"]),
        Route("/api/items/{item_id}/skip", api_skip, methods=["POST"]),
        Route("/api/items/{item_id}/flag", api_flag, methods=["POST"]),
        Route("/api/images/{item_id}/{n}", api_image),
        Route("/api/pages/{item_id}", api_page),
        Route("/api/admin/summary", api_admin_summary),
        Route("/api/admin/items", api_admin_items),
        Route("/api/admin/items/{item_id}", api_admin_item),
        Route("/api/admin/items/{item_id}/adjudicate", api_admin_adjudicate, methods=["POST"]),
        Route("/api/admin/items/{item_id}/reopen", api_admin_reopen, methods=["POST"]),
        Route("/api/admin/compare", api_admin_compare),
        Route("/api/admin/export", api_admin_export),
        Route("/api/admin/backup", api_admin_backup, methods=["POST"]),
        Route("/api/admin/events", api_admin_events),
    ]
    app = Starlette(routes=routes)
    app.state.db = db
    app.state.bundle = bundle
    app.state.data_dir = data_dir
    app.state.admin_key = key
    app.state.invite_token = invite_token if invite_token is not None else (os.environ.get("REVIEW46_INVITE_TOKEN") or None)
    app.state.backup_copy_to = str(backup_copy_to) if backup_copy_to else os.environ.get("REVIEW46_BACKUP_COPY_TO")
    app.state.instructor_reference = json.loads((priv / "instructor_reference.json").read_text(encoding="utf-8"))
    app.state.model_proposals = json.loads((priv / "model_proposals.json").read_text(encoding="utf-8"))
    return app


__all__ = ["create_app", "COOKIE", "ADMIN_COOKIE"]
