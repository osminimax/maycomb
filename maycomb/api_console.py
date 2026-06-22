"""Operator console API: REST under /api plus the realtime WebSocket."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, PlainTextResponse

from . import __version__, wire
from .config import RUNTIME_KEYS, coerce_runtime_value, public_dict
from .ids import new_call_id
from .state import Hub, OpError
from .tokenizer import TokenCodec

router = APIRouter(prefix="/api")


def _hub(request: Request) -> Hub:
    return request.app.state.hub


def _op(fn):
    try:
        return fn()
    except OpError as e:
        raise HTTPException(status_code=e.status, detail=e.message) from None


@router.get("/state")
async def get_state(request: Request):
    hub = _hub(request)
    pending = hub.db.list_exchanges(status="pending", limit=500)
    return {
        "version": __version__,
        "settings": public_dict(hub.settings),
        "tokenizer_active": hub.codec.encoding_name,
        "pending_count": len(pending),
    }


@router.put("/config")
async def put_config(request: Request):
    hub = _hub(request)
    body = await request.json()
    applied = {}
    for key, value in (body or {}).items():
        if key not in RUNTIME_KEYS:
            raise HTTPException(422, f"'{key}' is not runtime-configurable")
        try:
            coerced = coerce_runtime_value(key, value)
        except (TypeError, ValueError):
            raise HTTPException(422, f"invalid value for '{key}'") from None
        setattr(hub.settings, key, coerced)
        hub.db.set_setting(key, coerced)
        applied[key] = coerced
    if "tokenizer" in applied:
        hub.codec = TokenCodec(hub.settings.tokenizer)
    hub.notify("config", extra_settings=None)
    return {"applied": applied, "settings": public_dict(hub.settings),
            "tokenizer_active": hub.codec.encoding_name}


@router.get("/exchanges")
async def list_exchanges(request: Request, status: str | None = None, limit: int = 200):
    hub = _hub(request)
    rows = hub.db.list_exchanges(status=status, limit=limit)
    for row in rows:
        s = hub.summary(row["exchange_id"])
        row["runtime"] = s["runtime"] if s else None
    return {"exchanges": rows}


@router.get("/exchanges/{exchange_id}")
async def get_exchange(request: Request, exchange_id: str):
    hub = _hub(request)
    summary = hub.summary(exchange_id)
    if summary is None:
        raise HTTPException(404, "exchange not found")
    parsed_ev = hub.db.last_event(exchange_id, ["request_parsed"])
    result_ev = hub.db.last_event(
        exchange_id, ["response_submitted", "exchange_aborted", "request_rejected"])
    injected = [e for e in hub.db.events(exchange_id, ["error_injected"])]
    return {
        "summary": summary,
        "request": (parsed_ev or {}).get("data", {}).get("request"),
        "unknown_fields": (parsed_ev or {}).get("data", {}).get("_unknown_fields", []),
        "draft": hub.db.get_draft(exchange_id),
        "result": result_ev,
        "injections": injected,
    }


@router.get("/exchanges/{exchange_id}/events")
async def get_events(request: Request, exchange_id: str, limit: int = 2000):
    hub = _hub(request)
    if hub.db.get_exchange(exchange_id) is None:
        raise HTTPException(404, "exchange not found")
    return {"events": hub.db.events(exchange_id, limit=limit)}


@router.get("/exchanges/{exchange_id}/raw")
async def get_raw(request: Request, exchange_id: str):
    hub = _hub(request)
    raw = hub.db.read_raw(exchange_id)
    if raw is None:
        raise HTTPException(404, "raw body not found")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("utf-8", errors="replace")
    return PlainTextResponse(text)


@router.post("/exchanges/{exchange_id}/draft")
async def save_draft(request: Request, exchange_id: str):
    hub = _hub(request)
    body = await request.json()
    revision = _op(lambda: hub.save_draft(
        exchange_id, body.get("draft") or {}, mode=body.get("mode") or "draft"))
    return {"revision": revision}


@router.post("/exchanges/{exchange_id}/submit")
async def submit(request: Request, exchange_id: str):
    hub = _hub(request)
    body = await request.json()
    try:
        result = await hub.submit(exchange_id, body or {})
    except OpError as e:
        raise HTTPException(status_code=e.status, detail=e.message) from None
    return result


@router.post("/exchanges/{exchange_id}/validate")
async def validate(request: Request, exchange_id: str):
    """Dry-run draft validation so the console can show blockers before submit."""
    hub = _hub(request)
    body = await request.json()
    rt = hub.runtimes.get(exchange_id)
    parsed = rt.parsed if rt else None
    if parsed is None:
        ev = hub.db.last_event(exchange_id, ["request_parsed"])
        parsed = (ev or {}).get("data", {}).get("request")
    if parsed is None:
        raise HTTPException(404, "exchange not found")
    try:
        draft = hub._normalize_console_draft(body.get("draft") or {})
    except OpError as e:
        raise HTTPException(status_code=e.status, detail=e.message) from None
    blockers, warnings, auto_tags = wire.validate_draft(parsed, draft)
    return {"blockers": blockers, "warnings": warnings, "auto_tags": auto_tags}


@router.post("/exchanges/{exchange_id}/abort")
async def abort(request: Request, exchange_id: str):
    hub = _hub(request)
    body = await request.json()
    try:
        await hub.abort(exchange_id, body.get("kind") or "graceful",
                        finish_reason=body.get("finish_reason") or "stop",
                        style=body.get("style"))
    except OpError as e:
        raise HTTPException(status_code=e.status, detail=e.message) from None
    return {"ok": True}


@router.post("/exchanges/{exchange_id}/inject")
async def inject(request: Request, exchange_id: str):
    hub = _hub(request)
    body = await request.json()
    try:
        await hub.inject(exchange_id, body.get("kind") or "", body.get("params"))
    except OpError as e:
        raise HTTPException(status_code=e.status, detail=e.message) from None
    return {"ok": True}


@router.post("/tool-call-id")
async def tool_call_id():
    return {"id": new_call_id()}


@router.websocket("/ws")
async def console_ws(websocket: WebSocket):
    hub: Hub = websocket.app.state.hub
    await websocket.accept()
    hub.ws_clients.add(websocket)
    try:
        await websocket.send_text(wire.dumps({
            "type": "hello",
            "settings": public_dict(hub.settings),
            "tokenizer_active": hub.codec.encoding_name,
        }))
        while True:
            try:
                msg = json.loads(await websocket.receive_text())
            except ValueError:
                continue
            t = msg.get("type")
            if t == "ping":
                await websocket.send_text('{"type":"pong"}')
                continue
            if isinstance(t, str) and t.startswith("live_"):
                try:
                    ack = await hub.live_message(msg)
                    await websocket.send_text(wire.dumps(
                        {"type": "live_ack", "for": t,
                         "exchange_id": msg.get("exchange_id"), **ack}))
                except OpError as e:
                    await websocket.send_text(wire.dumps(
                        {"type": "live_error", "for": t,
                         "exchange_id": msg.get("exchange_id"),
                         "message": e.message}))
    except WebSocketDisconnect:
        pass
    finally:
        hub.ws_clients.discard(websocket)
