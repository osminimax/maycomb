"""Exchange runtime state and operator operations.

The Hub connects three worlds:
  - the OpenAI-compatible endpoint (waiting clients),
  - the operator console (REST + WebSocket),
  - the SQLite WAL event log.

Frame flow for streaming exchanges: operator actions enqueue Frames into the
runtime's out_queue; the SSE consumer (sse.py) sends them on the wire and calls
back into the Hub for WAL recording (`chunk_sent` per frame, spec §4.5).
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

from . import wire
from .config import Settings
from .db import Database, now_iso
from .ids import chatcmpl_id_for, new_call_id, new_exchange_uuid
from .tokenizer import TokenCodec, build_usage


class OpError(Exception):
    """Console-side operation error -> HTTP status + message."""

    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message
        super().__init__(message)


@dataclass
class Frame:
    kind: str                  # data | finish | http_error | hard_abort
    payload_sse: str = ""      # for data: full "data: {...}\n\n" frame
    finish_reason: str = "stop"
    partial: bool = False
    status: int = 0            # for http_error
    body_raw: str = ""
    headers: dict = field(default_factory=dict)
    label: str = ""            # injection kind label
    reason: str = ""           # for hard_abort


@dataclass
class ResultPayload:
    status: int = 200
    raw: str = ""
    headers: dict = field(default_factory=dict)
    drop: bool = False


@dataclass
class ExchangeRuntime:
    exchange_id: str
    chatcmpl_id: str
    model: str
    stream: bool
    include_usage: bool
    parsed: dict
    status: str = "pending"
    done: bool = False
    out_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    result_future: asyncio.Future | None = None
    sender_task: asyncio.Task | None = None
    headers_started: bool = False
    first_data_sent: bool = False
    sent_frames: list[str] = field(default_factory=list)
    chunk_count: int = 0
    stream_created: int | None = None
    live: dict | None = None
    injection: dict | None = None
    submit_meta: dict | None = None
    usage_final: dict | None = None


_LIVE_STAGES = {"role": 0, "reasoning": 1, "content": 2, "tool_calls": 3}


class Hub:
    def __init__(self, db: Database, settings: Settings, codec: TokenCodec):
        self.db = db
        self.settings = settings
        self.codec = codec
        self.runtimes: dict[str, ExchangeRuntime] = {}
        self.ws_clients: set = set()

    # ================================================================ identity
    def new_ids(self) -> tuple[str, str]:
        u = new_exchange_uuid()
        return str(u), chatcmpl_id_for(u)

    def runtime(self, exchange_id: str) -> ExchangeRuntime:
        rt = self.runtimes.get(exchange_id)
        if rt is None:
            raise OpError(404, "exchange not found or no longer active")
        return rt

    # ================================================================ registry
    def register_runtime(self, exchange_id: str, chatcmpl_id: str, parsed: dict) -> ExchangeRuntime:
        so = parsed.get("stream_options") or {}
        rt = ExchangeRuntime(
            exchange_id=exchange_id,
            chatcmpl_id=chatcmpl_id,
            model=str(parsed.get("model", "")),
            stream=bool(parsed.get("stream")),
            include_usage=bool(isinstance(so, dict) and so.get("include_usage")),
            parsed=parsed,
        )
        if not rt.stream:
            rt.result_future = asyncio.get_running_loop().create_future()
        self.runtimes[exchange_id] = rt
        return rt

    def _drop_runtime(self, exchange_id: str) -> None:
        self.runtimes.pop(exchange_id, None)

    # =============================================================== summaries
    def summary(self, exchange_id: str) -> dict | None:
        row = self.db.get_exchange(exchange_id)
        if row is None:
            return None
        rt = self.runtimes.get(exchange_id)
        row["runtime"] = None
        if rt is not None:
            row["runtime"] = {
                "headers_started": rt.headers_started,
                "first_data_sent": rt.first_data_sent,
                "chunks_sent": rt.chunk_count,
                "sending": rt.sender_task is not None and not rt.sender_task.done(),
                "live_stage": (rt.live or {}).get("stage"),
                "injection": rt.injection,
                "done": rt.done,
            }
        return row

    def notify(self, type_: str, exchange_id: str | None = None, **extra: Any) -> None:
        payload: dict[str, Any] = {"type": type_, **extra}
        if exchange_id:
            payload["summary"] = self.summary(exchange_id)
        try:
            asyncio.get_running_loop().create_task(self.broadcast(payload))
        except RuntimeError:
            pass

    async def broadcast(self, payload: dict) -> None:
        text = wire.dumps(payload)
        for ws in list(self.ws_clients):
            try:
                await ws.send_text(text)
            except Exception:
                self.ws_clients.discard(ws)

    # ============================================================ chunk frames
    def _created(self, rt: ExchangeRuntime) -> int:
        if rt.stream_created is None:
            rt.stream_created = int(time.time())
        return rt.stream_created

    def chunk_frame(self, rt: ExchangeRuntime, *, delta: dict | None = None,
                    finish_reason: str | None = None, usage: dict | None = None) -> str:
        obj = wire.build_chunk(
            chatcmpl_id=rt.chatcmpl_id, created=self._created(rt), model=rt.model,
            delta=delta, finish_reason=finish_reason, usage=usage)
        return wire.sse_frame(obj)

    def role_frame(self, rt: ExchangeRuntime) -> str:
        return self.chunk_frame(rt, delta={"role": "assistant"})

    # ============================================================== draft save
    def _normalize_console_draft(self, draft: dict) -> dict:
        out = {
            "reasoning_content": (draft.get("reasoning_content") or None),
            "content": (draft.get("content") or None),
            "finish_reason": draft.get("finish_reason") or "auto",
            "tool_calls": [],
        }
        for tc in draft.get("tool_calls") or []:
            if not isinstance(tc, dict):
                raise OpError(422, "tool_calls entries must be objects")
            out["tool_calls"].append({
                "id": tc.get("id") or new_call_id(),
                "name": tc.get("name") or "",
                "arguments": tc.get("arguments", ""),
            })
        return out

    def _draft_record(self, draft: dict, mode: str) -> dict:
        """spec §11 draft_saved payload: arguments stored as objects when valid."""
        tcs = []
        for tc in draft["tool_calls"]:
            entry: dict[str, Any] = {"id": tc["id"], "name": tc["name"]}
            args = tc["arguments"]
            if isinstance(args, str):
                try:
                    entry["arguments_obj"] = json.loads(args)
                except ValueError:
                    entry["arguments_raw"] = args
            else:
                entry["arguments_obj"] = args
            tcs.append(entry)
        return {
            "reasoning_content": draft["reasoning_content"],
            "content": draft["content"],
            "tool_calls": tcs,
            "finish_reason": draft["finish_reason"],
            "mode": mode,
        }

    def save_draft(self, exchange_id: str, draft: dict, mode: str = "draft") -> int:
        if self.settings.read_only:
            raise OpError(409, "read-only mode")
        rt = self.runtime(exchange_id)
        if rt.done:
            raise OpError(409, "exchange already finished")
        record = self._draft_record(self._normalize_console_draft(draft), mode)
        revision = self.db.save_draft(exchange_id, record)
        self.db.append_event(exchange_id, "draft_saved", {"revision": revision, **record})
        return revision

    # ================================================================== submit
    async def submit(self, exchange_id: str, body: dict) -> dict:
        if self.settings.read_only:
            raise OpError(409, "read-only mode")
        rt = self.runtime(exchange_id)
        if rt.done or rt.status != "pending":
            raise OpError(409, f"exchange is not pending (status={rt.status})")

        draft = self._normalize_console_draft(body.get("draft") or {})
        meta_in = body.get("meta") or {}
        mode = body.get("mode") or ("paced" if rt.stream else "json")
        if rt.stream and mode not in ("paced", "instant"):
            raise OpError(422, "mode must be paced or instant for streaming exchanges")
        if not rt.stream:
            mode = "json"
        pacing = body.get("pacing") or {}

        blockers, warnings, auto_tags = wire.validate_draft(rt.parsed, draft)
        bypass = bool(meta_in.get("validation_bypass"))
        if blockers and not bypass:
            raise OpError(422, "validation failed: " + "; ".join(blockers))

        tags = [str(t) for t in (meta_in.get("tags") or [])]
        bypassed = bool(blockers and bypass)
        if bypassed:
            for t in auto_tags + ["validation_bypassed"]:
                if t not in tags:
                    tags.append(t)

        try:
            finish_reason = wire.resolve_finish_reason(
                draft.get("finish_reason"), bool(draft["tool_calls"]))
        except ValueError as e:
            raise OpError(422, str(e)) from None

        # final draft revision is recorded at the moment of submission
        record = self._draft_record(draft, mode)
        revision = self.db.save_draft(exchange_id, record)
        self.db.append_event(exchange_id, "draft_saved", {"revision": revision, **record})

        rt.submit_meta = {
            "operator_note": str(meta_in.get("operator_note") or ""),
            "tags": tags,
            "validation_bypassed": bypassed,
            "mode": mode,
        }

        delay_ms = 0
        if rt.injection and rt.injection.get("kind") == "delay" and not rt.injection.get("applied"):
            delay_ms = int(rt.injection.get("delay_ms", 0))
            rt.injection["applied"] = True

        wire_tool_calls = wire.build_wire_tool_calls(draft["tool_calls"])
        rt.status = "active"
        self.db.update_exchange(exchange_id, status="active")

        if rt.stream:
            tps = float(pacing.get("tokens_per_second") or self.settings.tokens_per_second)
            ttft_ms = pacing.get("ttft_ms")
            ttft_ms = self.settings.ttft_ms if ttft_ms is None else int(ttft_ms)
            interleave = pacing.get("interleave")
            interleave = self.settings.stress_interleave if interleave is None else bool(interleave)
            rt.sender_task = asyncio.get_running_loop().create_task(
                self._paced_sender(rt, draft, wire_tool_calls, finish_reason,
                                   mode=mode, tps=tps,
                                   ttft_s=(ttft_ms + delay_ms) / 1000.0,
                                   interleave=interleave))
        else:
            asyncio.get_running_loop().create_task(
                self._deliver_nonstream(rt, draft, wire_tool_calls, finish_reason,
                                        delay_s=delay_ms / 1000.0))
        self.notify("exchange_update", exchange_id)
        return {"warnings": warnings, "finish_reason": finish_reason, "revision": revision}

    async def _paced_sender(self, rt: ExchangeRuntime, draft: dict,
                            wire_tool_calls: list[dict] | None, finish_reason: str,
                            *, mode: str, tps: float, ttft_s: float,
                            interleave: bool) -> None:
        try:
            self._created(rt)  # fix `created` at response generation time (spec §1)
            deltas = wire.plan_deltas(
                reasoning_content=draft["reasoning_content"],
                content=draft["content"],
                tool_calls=wire_tool_calls,
                splitter=self.codec.split_pieces,
                instant=(mode == "instant"),
                interleave=interleave,
                alias=self.settings.reasoning_alias,
            )
            if ttft_s > 0:
                await asyncio.sleep(ttft_s)
            sleep_per = (1.0 / tps) if (mode == "paced" and tps > 0) else 0.0
            # Pace against an absolute schedule instead of sleeping a fixed slice
            # per token. On Windows the asyncio timer resolution is ~15.6ms, so a
            # per-token sleep shorter than that (e.g. above ~64 tok/s) silently
            # collapses to no delay on a busy loop; targeting cumulative deadlines
            # keeps the average rate honest and the stream observably paced on
            # every platform.
            loop = asyncio.get_running_loop()
            start = loop.time()
            paced_sent = 0
            for delta, paced in deltas:
                rt.out_queue.put_nowait(Frame("data", payload_sse=self.chunk_frame(rt, delta=delta)))
                if paced and sleep_per > 0:
                    paced_sent += 1
                    gap = (start + paced_sent * sleep_per) - loop.time()
                    if gap > 0:
                        await asyncio.sleep(gap)
            rt.out_queue.put_nowait(Frame("finish", finish_reason=finish_reason, partial=False))
        except asyncio.CancelledError:
            pass  # an abort path owns the stream ending from here

    async def _deliver_nonstream(self, rt: ExchangeRuntime, draft: dict,
                                 wire_tool_calls: list[dict] | None, finish_reason: str,
                                 *, delay_s: float) -> None:
        if delay_s > 0:
            await asyncio.sleep(delay_s)
        if rt.done or rt.result_future is None or rt.result_future.done():
            return
        created = int(time.time())  # response object creation time (spec §1)
        message = wire.build_message(
            reasoning_content=draft["reasoning_content"],
            content=draft["content"],
            tool_calls=wire_tool_calls,
            alias=self.settings.reasoning_alias,
        )
        usage = build_usage(
            self.codec, rt.parsed.get("messages"),
            reasoning_content=draft["reasoning_content"],
            content=draft["content"],
            tool_calls=wire_tool_calls,
        )
        obj = wire.build_response_obj(
            chatcmpl_id=rt.chatcmpl_id, created=created, model=rt.model,
            message=message, finish_reason=finish_reason, usage=usage)
        raw = wire.dumps(obj)
        rt.usage_final = usage
        meta = rt.submit_meta or {}
        # byte-identical wire object + meta (spec §11 response_submitted)
        self.db.append_event(rt.exchange_id, "response_submitted", {
            "response": obj,
            "response_raw": raw,
            "usage": usage,
            "meta": {**meta, "partial": False},
        })
        rt.done = True
        rt.status = "completed"
        self.db.update_exchange(rt.exchange_id, status="completed",
                                finish_reason=finish_reason, partial=False,
                                tags=meta.get("tags") or [])
        rt.result_future.set_result(ResultPayload(200, raw, {}))
        self.notify("exchange_update", rt.exchange_id)
        self._drop_runtime(rt.exchange_id)

    # =================================================================== abort
    async def abort(self, exchange_id: str, kind: str, *, finish_reason: str = "stop",
                    style: str | None = None) -> None:
        rt = self.runtime(exchange_id)
        if rt.done:
            raise OpError(409, "exchange already finished")

        if not rt.stream:
            # spec §8: non-streaming cancel -> 500 error object or abrupt close
            style = style or self.settings.nonstream_cancel
            self._cancel_sender(rt)
            self.db.append_event(exchange_id, "exchange_aborted", {
                "reason": "operator_cancel", "style": style,
                "draft": self.db.get_draft(exchange_id)})
            rt.done = True
            rt.status = "aborted"
            self.db.update_exchange(exchange_id, status="aborted")
            if rt.result_future and not rt.result_future.done():
                if style == "drop":
                    rt.result_future.set_result(ResultPayload(drop=True))
                else:
                    body = wire.error_body(
                        "The operator cancelled this exchange.", "server_error")
                    rt.result_future.set_result(ResultPayload(500, wire.dumps(body), {}))
            self.notify("exchange_update", exchange_id)
            self._drop_runtime(exchange_id)
            return

        if kind == "graceful":
            if finish_reason not in ("stop", "length"):
                raise OpError(422, "graceful finish_reason must be stop or length (spec §8)")
            self._cancel_sender(rt)
            if rt.submit_meta is None:
                rt.submit_meta = {"operator_note": "", "tags": [],
                                  "validation_bypassed": False, "mode": "aborted_draft"}
            rt.out_queue.put_nowait(Frame("finish", finish_reason=finish_reason, partial=True))
        elif kind == "hard":
            self._cancel_sender(rt)
            rt.out_queue.put_nowait(Frame("hard_abort", reason="operator_hard_abort"))
        else:
            raise OpError(422, "kind must be graceful or hard for streaming exchanges")

    def _cancel_sender(self, rt: ExchangeRuntime) -> None:
        if rt.sender_task and not rt.sender_task.done():
            rt.sender_task.cancel()

    # =============================================================== injection
    async def inject(self, exchange_id: str, kind: str, params: dict | None) -> None:
        rt = self.runtime(exchange_id)
        if rt.done:
            raise OpError(409, "exchange already finished")
        params = params or {}

        if kind in ("rate_limit", "server_error", "context_length_exceeded"):
            if rt.stream and rt.headers_started:
                raise OpError(409, "SSE already started — HTTP-status injection impossible. "
                                   "Arm a delay first or use stream_cut.")
            if kind == "rate_limit":
                retry_after = int(params.get("retry_after", 30))
                status, headers = 429, {"Retry-After": str(retry_after)}
                body = wire.error_body(
                    "Rate limit reached for this mock deployment. Please retry later.",
                    "rate_limit_error", code="rate_limit_exceeded")
            elif kind == "server_error":
                status, headers = 500, {}
                body = wire.error_body(
                    "The server had an error while processing your request.",
                    "server_error")
            else:
                status, headers = 400, {}
                body = wire.error_body(
                    "This model's maximum context length is exceeded by your messages.",
                    "invalid_request_error", param="messages",
                    code="context_length_exceeded")
            self.db.append_event(exchange_id, "error_injected",
                                 {"kind": kind, "params": params, "status": status})
            raw = wire.dumps(body)
            if rt.stream:
                rt.out_queue.put_nowait(Frame("http_error", status=status, body_raw=raw,
                                              headers=headers, label=kind))
            else:
                rt.done = True
                rt.status = "injected"
                self.db.update_exchange(exchange_id, status="injected",
                                        tags=[f"injected:{kind}"])
                if rt.result_future and not rt.result_future.done():
                    rt.result_future.set_result(ResultPayload(status, raw, headers))
                self.notify("exchange_update", exchange_id)
                self._drop_runtime(exchange_id)
            return

        if kind == "delay":
            delay_ms = int(params.get("delay_ms", 10000))
            self.db.append_event(exchange_id, "error_injected",
                                 {"kind": kind, "params": {"delay_ms": delay_ms}})
            rt.injection = {"kind": "delay", "delay_ms": delay_ms, "applied": False}
            self.notify("exchange_update", exchange_id)
            return

        if kind == "stream_cut":
            if not rt.stream:
                raise OpError(409, "stream_cut applies to streaming exchanges only")
            after = params.get("after_chunks")
            self.db.append_event(exchange_id, "error_injected",
                                 {"kind": kind, "params": params})
            if after is None:
                self._cancel_sender(rt)
                rt.out_queue.put_nowait(Frame("hard_abort", reason="stream_cut_injected"))
            else:
                rt.injection = {"kind": "stream_cut", "after_chunks": int(after),
                                "applied": False}
                self.notify("exchange_update", exchange_id)
            return

        raise OpError(422, f"unknown injection kind: {kind}")

    # =============================================================== live mode
    async def live_message(self, msg: dict) -> dict:
        """Handle a console live-mode WS message. Returns an ack payload."""
        if self.settings.read_only:
            raise OpError(409, "read-only mode")
        xid = msg.get("exchange_id") or ""
        rt = self.runtime(xid)
        t = msg.get("type")

        if t == "live_start":
            if not rt.stream:
                raise OpError(409, "live mode requires a streaming request")
            if rt.done or rt.status != "pending":
                raise OpError(409, f"exchange is not pending (status={rt.status})")
            rt.live = {"stage": "role", "tool_index": 0}
            rt.status = "active"
            self._created(rt)
            self.db.update_exchange(xid, status="active")
            rt.out_queue.put_nowait(Frame("data", payload_sse=self.role_frame(rt)))
            self.notify("exchange_update", xid)
            return {"stage": "role"}

        if rt.live is None or rt.done:
            raise OpError(409, "live mode is not active for this exchange")
        stage = rt.live["stage"]

        if t == "live_text":
            field_name = msg.get("field")
            text = msg.get("text") or ""
            if field_name not in ("reasoning_content", "content"):
                raise OpError(422, "field must be reasoning_content or content")
            target = "reasoning" if field_name == "reasoning_content" else "content"
            if _LIVE_STAGES[target] < _LIVE_STAGES[stage]:
                raise OpError(409, f"{field_name} can no longer be sent (stage={stage}, spec §4.2)")
            if not text:
                return {"stage": stage}
            rt.live["stage"] = target
            delta = {field_name: text}
            if field_name == "reasoning_content" and self.settings.reasoning_alias:
                delta["reasoning"] = text
            rt.out_queue.put_nowait(Frame("data", payload_sse=self.chunk_frame(rt, delta=delta)))
            return {"stage": target}

        if t == "live_tool_call":
            idx = rt.live["tool_index"]
            rt.live["stage"] = "tool_calls"
            rt.live["tool_index"] = idx + 1
            call_id = msg.get("id") or new_call_id()
            name = msg.get("name") or ""
            args = msg.get("arguments") or ""
            meta_delta = {"tool_calls": [{"index": idx, "id": call_id, "type": "function",
                                          "function": {"name": name, "arguments": ""}}]}
            rt.out_queue.put_nowait(Frame("data", payload_sse=self.chunk_frame(rt, delta=meta_delta)))
            if args:
                frag = {"tool_calls": [{"index": idx, "function": {"arguments": args}}]}
                rt.out_queue.put_nowait(Frame("data", payload_sse=self.chunk_frame(rt, delta=frag)))
            return {"stage": "tool_calls", "tool_index": idx, "id": call_id}

        if t == "live_finish":
            has_tools = rt.live["tool_index"] > 0
            try:
                fr = wire.resolve_finish_reason(msg.get("finish_reason"), has_tools)
            except ValueError as e:
                raise OpError(422, str(e)) from None
            rt.submit_meta = {
                "operator_note": str(msg.get("operator_note") or ""),
                "tags": [str(x) for x in (msg.get("tags") or [])],
                "validation_bypassed": False,
                "mode": "live",
            }
            rt.out_queue.put_nowait(Frame("finish", finish_reason=fr, partial=False))
            return {"stage": "finish", "finish_reason": fr}

        raise OpError(422, f"unknown live message type: {t}")

    # ====================================================== SSE consumer hooks
    def on_chunk_sent(self, rt: ExchangeRuntime, raw: str) -> None:
        rt.sent_frames.append(raw)
        rt.chunk_count += 1
        self.db.append_event(rt.exchange_id, "chunk_sent", {
            "chunk_index": rt.chunk_count - 1,
            "sse_payload_raw": raw,
            "sent_at": now_iso(),
        })
        if rt.chunk_count % 8 == 0:
            self.notify("exchange_update", rt.exchange_id)

    def build_stream_usage(self, rt: ExchangeRuntime) -> dict:
        recon = wire.reconstruct_from_frames(rt.sent_frames)
        msg = (recon or {}).get("choices", [{}])[0].get("message", {}) if recon else {}
        return build_usage(
            self.codec, rt.parsed.get("messages"),
            reasoning_content=msg.get("reasoning_content"),
            content=msg.get("content"),
            tool_calls=msg.get("tool_calls"),
        )

    def on_stream_complete(self, rt: ExchangeRuntime, *, partial: bool,
                           finish_reason: str) -> None:
        if rt.done:
            return
        rt.done = True
        rt.status = "completed"
        recon = wire.reconstruct_from_frames(rt.sent_frames)
        meta = rt.submit_meta or {"operator_note": "", "tags": [],
                                  "validation_bypassed": False, "mode": "unknown"}
        # dual representation of the streamed response (spec §11)
        self.db.append_event(rt.exchange_id, "response_submitted", {
            "response": recon,
            "usage": rt.usage_final,
            "meta": {**meta, "partial": partial},
        })
        self.db.update_exchange(rt.exchange_id, status="completed",
                                finish_reason=finish_reason, partial=partial,
                                tags=meta.get("tags") or [])
        self.notify("exchange_update", rt.exchange_id)
        self._drop_runtime(rt.exchange_id)

    def on_client_disconnect(self, rt: ExchangeRuntime) -> None:
        if rt.done:
            return
        rt.done = True
        rt.status = "aborted"
        self._cancel_sender(rt)
        self.db.append_event(rt.exchange_id, "client_disconnected",
                             {"chunks_sent": rt.chunk_count})
        self.db.append_event(rt.exchange_id, "exchange_aborted", {
            "reason": "client_disconnected",
            "draft": self.db.get_draft(rt.exchange_id),
            "reconstruction": wire.reconstruct_from_frames(rt.sent_frames),
        })
        self.db.update_exchange(rt.exchange_id, status="aborted")
        self.notify("exchange_update", rt.exchange_id)
        self._drop_runtime(rt.exchange_id)

    def on_hard_abort(self, rt: ExchangeRuntime, reason: str) -> None:
        if rt.done:
            return
        rt.done = True
        rt.status = "aborted"
        self._cancel_sender(rt)
        self.db.append_event(rt.exchange_id, "exchange_aborted", {
            "reason": reason,
            "draft": self.db.get_draft(rt.exchange_id),
            "reconstruction": wire.reconstruct_from_frames(rt.sent_frames),
            "chunks_sent": rt.chunk_count,
        })
        self.db.update_exchange(rt.exchange_id, status="aborted")
        self.notify("exchange_update", rt.exchange_id)
        self._drop_runtime(rt.exchange_id)

    def on_injected_delivered(self, rt: ExchangeRuntime, label: str) -> None:
        if rt.done:
            return
        rt.done = True
        rt.status = "injected"
        self.db.update_exchange(rt.exchange_id, status="injected",
                                tags=[f"injected:{label}"])
        self.notify("exchange_update", rt.exchange_id)
        self._drop_runtime(rt.exchange_id)

    # ================================================================ shutdown
    async def shutdown(self) -> None:
        for rt in list(self.runtimes.values()):
            self._cancel_sender(rt)
            if rt.result_future and not rt.result_future.done():
                rt.result_future.cancel()
