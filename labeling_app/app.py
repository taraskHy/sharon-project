"""The web app (Starlette ASGI; served by uvicorn).

    GET  /                      grader page
    GET  /admin                 admin page (optional LABELING_ADMIN_KEY)
    POST /api/session {name}    set the grader name cookie
    GET  /api/me                grader + progress
    POST /api/next              claim the next item for this grader
    GET  /api/items/{id}        one item (grader view: own label only)
    POST /api/items/{id}/label  save / skip / flag  (expected_revision -> 409 when stale)
    GET  /api/images/{id}/{n}   answer crop
    GET  /api/pages/{id}        full source page (red instructor ink masked; only when available)
    GET  /api/my-items
    admin: /api/admin/summary | /items | /items/{id} | /items/{id}/final | /finalize-agreement |
           /reopen | /policy | /export | /backup | /events

The grader view NEVER contains: other graders' labels, expected labels,
model outputs, OCR confidence, splits, writers, repository paths.
No model or provider is ever called.
"""
from __future__ import annotations

import json
import os
import urllib.parse
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from .backup import make_backup
from .bundle import Bundle
from .db import LabelDB, LabelError, StaleWrite
from .export import export_final, write_export

WEB = Path(__file__).resolve().parent / "web"
COOKIE = "labeling_grader"
ADMIN_COOKIE = "labeling_admin"


def _grader(request: Request) -> str | None:
    raw = request.cookies.get(COOKIE)
    if not raw:
        return None
    name = urllib.parse.unquote(raw).strip()
    return name or None


def _need_grader(request: Request) -> str | JSONResponse:
    g = _grader(request)
    if not g:
        return JSONResponse({"error": "no grader name; enter your name first"}, status_code=401)
    return g


def _is_admin(request: Request) -> bool:
    key = request.app.state.admin_key
    if not key:
        return True
    supplied = (request.headers.get("x-admin-key") or request.query_params.get("key")
                or request.cookies.get(ADMIN_COOKIE))
    return supplied == key


def _need_admin(request: Request) -> JSONResponse | None:
    if not _is_admin(request):
        return JSONResponse({"error": "admin key required"}, status_code=403)
    return None


async def _json(request: Request) -> dict:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return {}
    return body if isinstance(body, dict) else {}


# ----------------------------------------------------------------- pages --

async def page_grader(request: Request) -> Response:
    return HTMLResponse((WEB / "grader.html").read_text(encoding="utf-8"))


async def page_admin(request: Request) -> Response:
    if not _is_admin(request):
        return HTMLResponse("<h3>admin key required</h3><p>open /admin?key=YOUR_KEY</p>", status_code=403)
    resp = HTMLResponse((WEB / "admin.html").read_text(encoding="utf-8"))
    key = request.query_params.get("key")
    if key and request.app.state.admin_key and key == request.app.state.admin_key:
        resp.set_cookie(ADMIN_COOKIE, key, max_age=30 * 24 * 3600, samesite="lax")
    return resp


# --------------------------------------------------------------- session --

async def api_session(request: Request) -> Response:
    body = await _json(request)
    name = str(body.get("name", "")).strip()
    db: LabelDB = request.app.state.db
    try:
        db.touch_grader(name)
    except LabelError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    resp = JSONResponse({"grader": name})
    resp.set_cookie(COOKIE, urllib.parse.quote(name), max_age=180 * 24 * 3600, samesite="lax")
    return resp


async def api_logout(request: Request) -> Response:
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE)
    return resp


async def api_me(request: Request) -> Response:
    g = _grader(request)
    db: LabelDB = request.app.state.db
    return JSONResponse({"grader": g, "progress": db.progress(g) if g else None,
                         "bundle": {"items": request.app.state.bundle.meta.get("items")}})


# ------------------------------------------------------------------ items --

def _item_view(request: Request, item_id: str, grader: str) -> dict | None:
    bundle: Bundle = request.app.state.bundle
    db: LabelDB = request.app.state.db
    payload = bundle.grader_payload(item_id)
    if payload is None:
        return None
    mine = db.get_label(item_id, grader)
    payload["my_label"] = mine
    payload["label_revision"] = mine["revision"] if mine else 0
    payload["final"] = bool(db.overview(item_id)["final"])
    return payload


async def api_next(request: Request) -> Response:
    g = _need_grader(request)
    if isinstance(g, Response):
        return g
    body = await _json(request)
    db: LabelDB = request.app.state.db
    item_id = db.claim_next(g, include_skipped=bool(body.get("include_skipped")))
    if item_id is None:
        return JSONResponse({"item": None, "done": True, "progress": db.progress(g)})
    return JSONResponse({"item": _item_view(request, item_id, g), "done": False, "progress": db.progress(g)})


async def api_item(request: Request) -> Response:
    g = _need_grader(request)
    if isinstance(g, Response):
        return g
    item_id = request.path_params["item_id"]
    view = _item_view(request, item_id, g)
    if view is None:
        return JSONResponse({"error": "unknown item"}, status_code=404)
    return JSONResponse({"item": view, "progress": request.app.state.db.progress(g)})


async def api_label(request: Request) -> Response:
    g = _need_grader(request)
    if isinstance(g, Response):
        return g
    item_id = request.path_params["item_id"]
    body = await _json(request)
    db: LabelDB = request.app.state.db
    if request.app.state.bundle.item(item_id) is None:
        return JSONResponse({"error": "unknown item"}, status_code=404)
    try:
        label = db.save_label(item_id, g, score=body.get("score"), rubric=body.get("rubric") or [],
                              note=str(body.get("note") or ""), status=str(body.get("status") or "saved"),
                              flag_reason=str(body.get("flag_reason") or ""),
                              expected_revision=int(body.get("expected_revision") or 0))
    except StaleWrite as e:
        return JSONResponse({"error": str(e), "stale": True}, status_code=409)
    except LabelError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"label": label, "progress": db.progress(g)})


async def api_image(request: Request) -> Response:
    item_id = request.path_params["item_id"]
    try:
        n = int(request.path_params["n"])
    except ValueError:
        return Response(status_code=404)
    p = request.app.state.bundle.image_path(item_id, n)
    if p is None:
        return Response(status_code=404)
    return FileResponse(str(p), media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


async def api_page(request: Request) -> Response:
    """The full source page for an item (instructor red ink masked at build
    time; served only when the bundle marked it available)."""
    item_id = request.path_params["item_id"]
    p = request.app.state.bundle.page_path(item_id)
    if p is None:
        return JSONResponse({"error": "no source page available for this item"}, status_code=404)
    return FileResponse(str(p), media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


async def api_my_items(request: Request) -> Response:
    g = _need_grader(request)
    if isinstance(g, Response):
        return g
    return JSONResponse(request.app.state.db.my_items(g))


# ------------------------------------------------------------------ admin --

async def api_admin_summary(request: Request) -> Response:
    if (err := _need_admin(request)):
        return err
    bundle: Bundle = request.app.state.bundle
    summary = request.app.state.db.summary()
    # dataset-level eligibility accounting (source cases vs human workload)
    summary["eligibility"] = bundle.meta.get("eligibility")
    summary["ineligible_item_ids"] = bundle.ineligible_item_ids()
    return JSONResponse(summary)


async def api_admin_items(request: Request) -> Response:
    if (err := _need_admin(request)):
        return err
    state = request.query_params.get("state")
    ovs = request.app.state.db.all_overviews()
    if state:
        ovs = [o for o in ovs if o["state"] == state]
    compact = [{k: o[k] for k in ("item_id", "state", "revision", "wanted_labels", "n_saved", "n_skipped",
                                  "n_flagged", "agreement", "eligible")}
               | {"graders": [l["grader"] for l in o["labels"]],
                  "case_id": request.app.state.bundle.id_map.get(o["item_id"])} for o in ovs]
    return JSONResponse({"items": compact})


async def api_admin_item(request: Request) -> Response:
    if (err := _need_admin(request)):
        return err
    item_id = request.path_params["item_id"]
    bundle: Bundle = request.app.state.bundle
    db: LabelDB = request.app.state.db
    if bundle.item(item_id) is None:
        return JSONResponse({"error": "unknown item"}, status_code=404)
    ov = db.overview(item_id)
    ov["case_id"] = bundle.id_map.get(item_id)
    ov["content"] = bundle.grader_payload(item_id)
    ov["provenance_private"] = bundle.private_provenance.get(item_id)     # admin only (source file etc.)
    return JSONResponse(ov)


async def api_admin_final(request: Request) -> Response:
    if (err := _need_admin(request)):
        return err
    item_id = request.path_params["item_id"]
    if request.app.state.bundle.item(item_id) is None:
        return JSONResponse({"error": "unknown item (not in the served bundle)"}, status_code=404)
    body = await _json(request)
    db: LabelDB = request.app.state.db
    try:
        final = db.set_final(item_id, score=float(body.get("score")), rubric=body.get("rubric") or [],
                             note=str(body.get("note") or ""), source="adjudicated",
                             adjudicator=str(body.get("adjudicator") or "admin"),
                             expected_item_revision=int(body.get("expected_item_revision") or -1))
    except StaleWrite as e:
        return JSONResponse({"error": str(e), "stale": True}, status_code=409)
    except (LabelError, TypeError, ValueError) as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"final": final, "overview": db.overview(item_id)})


async def api_admin_finalize_agreement(request: Request) -> Response:
    if (err := _need_admin(request)):
        return err
    item_id = request.path_params["item_id"]
    if request.app.state.bundle.item(item_id) is None:
        return JSONResponse({"error": "unknown item (not in the served bundle)"}, status_code=404)
    body = await _json(request)
    db: LabelDB = request.app.state.db
    try:
        final = db.finalize_agreement(item_id, adjudicator=str(body.get("adjudicator") or "admin"),
                                      expected_item_revision=(int(body["expected_item_revision"])
                                                              if body.get("expected_item_revision") is not None else None))
    except StaleWrite as e:
        return JSONResponse({"error": str(e), "stale": True}, status_code=409)
    except LabelError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"final": final, "overview": db.overview(item_id)})


async def api_admin_reopen(request: Request) -> Response:
    if (err := _need_admin(request)):
        return err
    item_id = request.path_params["item_id"]
    db: LabelDB = request.app.state.db
    db.reopen(item_id)
    return JSONResponse({"overview": db.overview(item_id)})


async def api_admin_policy(request: Request) -> Response:
    if (err := _need_admin(request)):
        return err
    body = await _json(request)
    db: LabelDB = request.app.state.db
    try:
        n = db.set_wanted_labels(str(body.get("mode") or "none"), body.get("items") or [], int(body.get("n") or 2))
    except LabelError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"updated": n, "summary": db.summary()})


async def api_admin_export(request: Request) -> Response:
    if (err := _need_admin(request)):
        return err
    db: LabelDB = request.app.state.db
    bundle: Bundle = request.app.state.bundle
    data_dir: Path = request.app.state.data_dir
    path = data_dir / "exports" / "final_labels.json"
    data = write_export(db, bundle, path)
    return JSONResponse(data, headers={"Content-Disposition": 'attachment; filename="final_labels.json"',
                                       "X-Export-Path": str(path)})


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


async def api_health(request: Request) -> Response:
    return JSONResponse({"ok": True, "items": len(request.app.state.bundle.items), "ai_calls": 0})


# --------------------------------------------------------------- factory --

def create_app(*, data_dir: Path, bundle_dir: Path | None = None, admin_key: str | None = None,
               backup_copy_to: Path | None = None, dataset_dir: Path | None = None) -> Starlette:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    bundle = Bundle(bundle_dir or (data_dir / "bundle"))
    recompute = {"applied": False, "reason": "no dataset directory provided"}
    if dataset_dir is not None:
        # Recompute eligibility from the dataset (single source of truth) so a
        # STALE bundle built before eligibility filtering still fails safely.
        recompute = bundle.apply_dataset_eligibility(dataset_dir)
    db = LabelDB(data_dir / "labels.db")
    db.load_items(bundle.items)
    # Fail-safe reconciliation: unknown eligibility never flips a flag, and
    # items no longer in the served bundle are retired from the workload.
    db.sync_eligibility([i["item_id"] for i in bundle.items], bundle.ineligible_item_ids(),
                        eligibility_known=bundle.eligibility_known())
    routes = [
        Route("/", page_grader),
        Route("/admin", page_admin),
        Route("/api/health", api_health),
        Route("/api/session", api_session, methods=["POST"]),
        Route("/api/logout", api_logout, methods=["POST"]),
        Route("/api/me", api_me),
        Route("/api/next", api_next, methods=["POST"]),
        Route("/api/my-items", api_my_items),
        Route("/api/items/{item_id}", api_item),
        Route("/api/items/{item_id}/label", api_label, methods=["POST"]),
        Route("/api/images/{item_id}/{n}", api_image),
        Route("/api/pages/{item_id}", api_page),
        Route("/api/admin/summary", api_admin_summary),
        Route("/api/admin/items", api_admin_items),
        Route("/api/admin/items/{item_id}", api_admin_item),
        Route("/api/admin/items/{item_id}/final", api_admin_final, methods=["POST"]),
        Route("/api/admin/items/{item_id}/finalize-agreement", api_admin_finalize_agreement, methods=["POST"]),
        Route("/api/admin/items/{item_id}/reopen", api_admin_reopen, methods=["POST"]),
        Route("/api/admin/policy", api_admin_policy, methods=["POST"]),
        Route("/api/admin/export", api_admin_export),
        Route("/api/admin/backup", api_admin_backup, methods=["POST"]),
        Route("/api/admin/events", api_admin_events),
    ]
    app = Starlette(routes=routes)
    app.state.db = db
    app.state.bundle = bundle
    app.state.eligibility_recompute = recompute
    app.state.data_dir = data_dir
    app.state.admin_key = admin_key or os.environ.get("LABELING_ADMIN_KEY") or None
    app.state.backup_copy_to = str(backup_copy_to) if backup_copy_to else os.environ.get("LABELING_BACKUP_COPY_TO")
    return app


__all__ = ["create_app", "COOKIE", "ADMIN_COOKIE"]
