"""OpenAI-compatible surface: POST /v1/chat/completions, GET /v1/models."""
from __future__ import annotations

import asyncio
import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

from . import wire
from .sse import AbruptDropResponse, ExchangeSSEResponse
from .state import Hub, ResultPayload

router = APIRouter()


def _error_response(err: wire.ApiError) -> JSONResponse:
    return JSONResponse(status_code=err.status, content=err.body, headers=err.headers)


def _check_auth(request: Request, hub: Hub) -> None:
    """spec §1: header required; default mode accepts any value."""
    auth = request.headers.get("authorization")
    if not auth:
        raise wire.err_missing_auth()
    if hub.settings.auth_mode == "fixed":
        if auth != f"Bearer {hub.settings.api_key}":
            raise wire.err_bad_key()


def _request_headers_snapshot(request: Request) -> dict:
    keep = ("content-type", "user-agent", "x-request-id", "openai-organization")
    out = {k: request.headers.get(k) for k in keep if request.headers.get(k)}
    out["authorization"] = "present" if request.headers.get("authorization") else "missing"
    return out


@router.get("/v1/models")
async def list_models(request: Request):
    hub: Hub = request.app.state.hub
    try:
        _check_auth(request, hub)
    except wire.ApiError as e:
        return _error_response(e)
    return JSONResponse({"object": "list", "data": hub.settings.models})


@router.post("/v1/chat/completions")
async def chat_completions(request: Request):
    hub: Hub = request.app.state.hub

    # read-only mode rejects before anything is persisted (spec §9.1)
    if hub.settings.read_only:
        return _error_response(wire.err_read_only())

    body = await request.body()
    exchange_id, chatcmpl_id = hub.new_ids()
    raw_path, sha256, nbytes = hub.db.store_raw(exchange_id, body)

    parsed = None
    parse_error: wire.ApiError | None = None
    try:
        parsed = json.loads(body)
    except ValueError:
        parse_error = wire.err_bad_json()

    model = parsed.get("model") if isinstance(parsed, dict) else None
    stream = bool(parsed.get("stream")) if isinstance(parsed, dict) else False
    hub.db.insert_exchange(
        exchange_id, chatcmpl_id,
        model=model if isinstance(model, str) else None,
        stream=stream, status="pending",
        preview=wire.preview_text(parsed if isinstance(parsed, dict) else None),
        flags={}, raw_path=raw_path, raw_sha256=sha256, raw_bytes=nbytes)
    hub.db.append_event(exchange_id, "request_received", {
        "path": raw_path, "sha256": sha256, "bytes": nbytes,
        "headers": _request_headers_snapshot(request)})

    def reject(err: wire.ApiError) -> JSONResponse:
        hub.db.append_event(exchange_id, "request_rejected",
                            {"status": err.status, "error": err.body})
        hub.db.update_exchange(exchange_id, status="rejected")
        hub.notify("exchange_update", exchange_id)
        return _error_response(err)

    try:
        _check_auth(request, hub)
    except wire.ApiError as e:
        return reject(e)
    if parse_error is not None:
        return reject(parse_error)
    try:
        wire.validate_request(parsed)
    except wire.ApiError as e:
        return reject(e)

    normalized, unknown = wire.normalize_request(parsed)
    hub.db.append_event(exchange_id, "request_parsed",
                        {"request": normalized, "_unknown_fields": unknown})
    hub.db.update_exchange(exchange_id, flags=wire.request_flags(parsed, unknown))

    rt = hub.register_runtime(exchange_id, chatcmpl_id, parsed)
    hub.notify("exchange_new", exchange_id)

    if rt.stream:
        return ExchangeSSEResponse(rt, hub)
    return await _wait_nonstream(request, hub, rt)


async def _wait_nonstream(request: Request, hub: Hub, rt) -> Response:
    async def poll_disconnect() -> None:
        while True:
            if await request.is_disconnected():
                return
            await asyncio.sleep(0.5)

    poll_task = asyncio.create_task(poll_disconnect())
    try:
        done, _ = await asyncio.wait({rt.result_future, poll_task},
                                     return_when=asyncio.FIRST_COMPLETED)
        if rt.result_future in done:
            try:
                rp: ResultPayload = rt.result_future.result()
            except asyncio.CancelledError:
                return _error_response(wire.ApiError(
                    500, wire.error_body("Server shutting down.", "server_error")))
            if rp.drop:
                return AbruptDropResponse()
            return Response(content=rp.raw.encode("utf-8"), status_code=rp.status,
                            media_type="application/json", headers=rp.headers)
        hub.on_client_disconnect(rt)
        return Response(status_code=204)  # client is gone; never delivered
    finally:
        poll_task.cancel()
