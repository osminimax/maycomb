"""SSE delivery for streaming exchanges (spec §4, §8).

Header sending is deferred until the first frame or the first keep-alive tick.
This leaves a window in which HTTP-status error injection (spec §9.2) is still
possible for streaming requests; once 200 + SSE headers are out, only
stream-level injections remain.

Hard abort raises OperatorHardAbort out of the ASGI app *after* the response
has started: uvicorn then closes the TCP connection without the terminal
chunk, which clients observe as a protocol-level truncated stream (spec §8).
"""
from __future__ import annotations

import asyncio

from starlette.responses import Response

from .state import ExchangeRuntime, Hub
from .wire import SSE_DONE

SSE_HEADERS = [
    (b"content-type", b"text/event-stream; charset=utf-8"),
    (b"cache-control", b"no-cache"),
    (b"connection", b"keep-alive"),
    (b"x-accel-buffering", b"no"),
]


class OperatorHardAbort(Exception):
    """Raised to make the server drop the connection mid-stream on purpose."""


class ExchangeSSEResponse(Response):
    def __init__(self, rt: ExchangeRuntime, hub: Hub):
        super().__init__(content=b"", media_type="text/event-stream")
        self.rt = rt
        self.hub = hub

    async def __call__(self, scope, receive, send) -> None:  # noqa: C901
        rt, hub = self.rt, self.hub
        started = False

        async def start_sse() -> None:
            nonlocal started
            if not started:
                await send({"type": "http.response.start", "status": 200,
                            "headers": SSE_HEADERS})
                started = True
                rt.headers_started = True

        async def send_text(text: str, *, more: bool = True) -> None:
            await send({"type": "http.response.body",
                        "body": text.encode("utf-8"), "more_body": more})

        async def send_data_frame(raw: str, *, more: bool = True) -> None:
            await start_sse()
            await send_text(raw, more=more)
            hub.on_chunk_sent(rt, raw)
            rt.first_data_sent = True

        disconnected = asyncio.Event()

        async def watch_disconnect() -> None:
            while True:
                msg = await receive()
                if msg["type"] == "http.disconnect":
                    disconnected.set()
                    return

        watcher = asyncio.create_task(watch_disconnect())
        disc_wait: asyncio.Task | None = None
        get_task: asyncio.Task | None = None
        try:
            while True:
                if get_task is None:
                    get_task = asyncio.create_task(rt.out_queue.get())
                if disc_wait is None or disc_wait.done():
                    disc_wait = asyncio.create_task(disconnected.wait())
                # a delay injection silences keep-alives until real data flows,
                # otherwise client read-timeout tests could never trigger
                delay_armed = bool(rt.injection
                                   and rt.injection.get("kind") == "delay")
                use_keepalive = (not rt.first_data_sent
                                 and hub.settings.keepalive_interval > 0
                                 and not delay_armed)
                done, _ = await asyncio.wait(
                    {get_task, disc_wait},
                    timeout=hub.settings.keepalive_interval if use_keepalive else None,
                    return_when=asyncio.FIRST_COMPLETED)

                if disc_wait in done:
                    hub.on_client_disconnect(rt)
                    if not started:  # keep the server quiet: close an unstarted response
                        try:
                            await start_sse()
                            await send_text("", more=False)
                        except Exception:
                            pass
                    return

                if get_task not in done:
                    # re-check at tick time: a delay may have been armed while
                    # this wait was already running
                    if rt.injection and rt.injection.get("kind") == "delay":
                        continue
                    # keep-alive comment while the operator composes (spec §4.1)
                    await start_sse()
                    await send_text(": keep-alive\n\n")
                    continue

                frame = get_task.result()
                get_task = None

                if frame.kind == "data":
                    await send_data_frame(frame.payload_sse)
                    inj = rt.injection
                    if (inj and inj.get("kind") == "stream_cut"
                            and not inj.get("applied")
                            and rt.chunk_count >= inj.get("after_chunks", 0)):
                        inj["applied"] = True
                        hub.on_hard_abort(rt, "stream_cut_injected")
                        raise OperatorHardAbort(rt.exchange_id)
                    continue

                if frame.kind == "finish":
                    if rt.chunk_count == 0:
                        # a finish without any prior chunk still needs the role
                        # chunk first (spec §4.2 order)
                        await send_data_frame(hub.role_frame(rt))
                    await send_data_frame(hub.chunk_frame(
                        rt, delta={}, finish_reason=frame.finish_reason))
                    usage = hub.build_stream_usage(rt)
                    rt.usage_final = usage
                    if rt.include_usage:
                        await send_data_frame(hub.chunk_frame(rt, usage=usage))
                    await start_sse()
                    await send_text(SSE_DONE, more=False)
                    hub.on_chunk_sent(rt, SSE_DONE)
                    hub.on_stream_complete(rt, partial=frame.partial,
                                           finish_reason=frame.finish_reason)
                    return

                if frame.kind == "http_error":
                    if started:
                        continue  # guarded at inject(); never expected here
                    headers = [(b"content-type", b"application/json")]
                    for k, v in frame.headers.items():
                        headers.append((k.encode("latin-1"), str(v).encode("latin-1")))
                    await send({"type": "http.response.start", "status": frame.status,
                                "headers": headers})
                    started = True
                    rt.headers_started = True
                    await send_text(frame.body_raw, more=False)
                    hub.on_injected_delivered(rt, frame.label)
                    return

                if frame.kind == "hard_abort":
                    await start_sse()
                    hub.on_hard_abort(rt, frame.reason)
                    raise OperatorHardAbort(rt.exchange_id)

        except asyncio.CancelledError:
            hub.on_client_disconnect(rt)
            raise
        finally:
            watcher.cancel()
            if get_task is not None:
                get_task.cancel()
            if disc_wait is not None:
                disc_wait.cancel()


class AbruptDropResponse(Response):
    """Non-streaming cancel, 'drop' style (spec §8): start 200, then sever."""

    def __init__(self):
        super().__init__(content=b"", media_type="application/json")

    async def __call__(self, scope, receive, send) -> None:
        await send({"type": "http.response.start", "status": 200,
                    "headers": [(b"content-type", b"application/json")]})
        await send({"type": "http.response.body", "body": b"", "more_body": True})
        raise OperatorHardAbort("nonstream_drop")
