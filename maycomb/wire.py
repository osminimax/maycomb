"""Wire format: request validation/normalization, response & chunk builders,
SSE frame reconstruction, draft submit validation. Implements spec §§1–6, 9, 11.
"""
from __future__ import annotations

import json
from typing import Any

from . import SYSTEM_FINGERPRINT

SSE_DONE = "data: [DONE]\n\n"

FINISH_REASONS = {"stop", "tool_calls", "length", "content_filter"}

# spec §2.1 + §2.2 — everything else is "unknown" (passed through, badged)
KNOWN_REQUEST_FIELDS = {
    "model", "messages", "tools", "tool_choice", "parallel_tool_calls",
    "stream", "stream_options", "temperature", "top_p", "presence_penalty",
    "frequency_penalty", "seed", "stop", "user", "metadata",
    "max_tokens", "max_completion_tokens", "response_format",
    "n", "logprobs", "top_logprobs",
    "store", "service_tier", "modalities", "audio", "prediction",
    "reasoning_effort", "reasoning",
}


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------- errors
def error_body(message: str, type_: str, param: str | None = None,
               code: str | None = None) -> dict:
    """OpenAI standard error object (spec §9)."""
    return {"error": {"message": message, "type": type_, "param": param, "code": code}}


class ApiError(Exception):
    def __init__(self, status: int, body: dict, headers: dict | None = None):
        self.status = status
        self.body = body
        self.headers = headers or {}
        super().__init__(body.get("error", {}).get("message", "error"))


def err_missing_auth() -> ApiError:
    return ApiError(401, error_body(
        "Missing Authorization header. Pass 'Authorization: Bearer <key>'.",
        "authentication_error"))


def err_bad_key() -> ApiError:
    return ApiError(401, error_body(
        "Incorrect API key provided.", "authentication_error", code="invalid_api_key"))


def err_read_only() -> ApiError:
    return ApiError(503, error_body(
        "This Maycomb server is in read-only mode and does not accept new exchanges.",
        "server_error", code="read_only_mode"))


def err_bad_json() -> ApiError:
    return ApiError(400, error_body(
        "We could not parse the JSON body of your request.", "invalid_request_error"))


# ----------------------------------------------------------------- validation
def validate_request(parsed: Any) -> None:
    """Reject-minimum policy (spec §2.1, §9.1). Raises ApiError."""
    if not isinstance(parsed, dict):
        raise err_bad_json()
    messages = parsed.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        raise ApiError(400, error_body(
            "'messages' is required and must be a non-empty array.",
            "invalid_request_error", param="messages"))
    if "model" not in parsed or not isinstance(parsed.get("model"), str) or not parsed["model"]:
        raise ApiError(400, error_body(
            "'model' is required.", "invalid_request_error", param="model"))
    n = parsed.get("n")
    if n is not None and n != 1:
        raise ApiError(400, error_body(
            f"You requested n={n} but this server only supports n=1.",
            "invalid_request_error", param="n"))
    if parsed.get("logprobs") is True:
        raise ApiError(400, error_body(
            "logprobs is not supported by this server.",
            "invalid_request_error", param="logprobs"))


def sort_keys_deep(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sort_keys_deep(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return [sort_keys_deep(v) for v in obj]
    return obj


def normalize_request(parsed: dict) -> tuple[dict, list[str]]:
    """spec §11 request_parsed: sorted keys, only-sent fields, unknown list."""
    unknown = sorted(k for k in parsed if k not in KNOWN_REQUEST_FIELDS)
    return sort_keys_deep(parsed), unknown


def request_flags(parsed: dict, unknown: list[str]) -> dict:
    """Console badge summary stored on the exchange row."""
    rf = parsed.get("response_format")
    tc = parsed.get("tool_choice")
    if isinstance(tc, dict):
        tc_label = tc.get("function", {}).get("name") or tc.get("type") or "object"
        tc_label = f"function:{tc_label}" if tc.get("type") == "function" else str(tc_label)
    else:
        tc_label = tc
    so = parsed.get("stream_options") or {}
    return {
        "stream": bool(parsed.get("stream")),
        "include_usage": bool(isinstance(so, dict) and so.get("include_usage")),
        "n_messages": len(parsed.get("messages") or []),
        "n_tools": len(parsed.get("tools") or []),
        "tool_choice": tc_label,
        "parallel_tool_calls": parsed.get("parallel_tool_calls"),
        "response_format": (rf or {}).get("type") if isinstance(rf, dict) else rf,
        "max_tokens": parsed.get("max_completion_tokens", parsed.get("max_tokens")),
        "unknown_fields": unknown,
        "reasoning_request": {
            k: parsed[k] for k in ("reasoning_effort", "reasoning") if k in parsed
        } or None,
    }


def preview_text(parsed: dict | None) -> str:
    if not parsed:
        return "(unparsable request)"
    messages = parsed.get("messages") or []
    chosen = None
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            chosen = msg
            break
    if chosen is None and messages:
        chosen = messages[-1] if isinstance(messages[-1], dict) else None
    if not chosen:
        return "(no messages)"
    content = chosen.get("content")
    if isinstance(content, list):
        texts = [p.get("text", "") for p in content
                 if isinstance(p, dict) and p.get("type") == "text"]
        content = " ".join(texts)
    if not isinstance(content, str):
        content = dumps(content) if content is not None else ""
    content = " ".join(content.split())
    label = chosen.get("role", "?")
    return f"[{label}] {content[:140]}" if content else f"[{label}] (empty)"


# ---------------------------------------------------------- response builders
def build_wire_tool_calls(draft_tool_calls: list[dict] | None) -> list[dict] | None:
    """Draft tool calls -> wire shape; arguments always a JSON string (spec §3)."""
    if not draft_tool_calls:
        return None
    out = []
    for tc in draft_tool_calls:
        args = tc.get("arguments")
        if not isinstance(args, str):
            args = dumps(args if args is not None else {})
        else:
            try:
                args = dumps(json.loads(args))  # normalize valid JSON text
            except (ValueError, TypeError):
                pass  # intentionally broken JSON goes out as-is (spec §3, bypass path)
        out.append({
            "id": tc["id"],
            "type": "function",
            "function": {"name": tc.get("name") or "", "arguments": args},
        })
    return out


def build_message(*, reasoning_content: str | None, content: str | None,
                  tool_calls: list[dict] | None, alias: bool) -> dict:
    """Assistant message per spec §3/§5: reasoning omitted when absent,
    reasoning before content, content null when empty."""
    msg: dict[str, Any] = {"role": "assistant"}
    if reasoning_content:
        msg["reasoning_content"] = reasoning_content
        if alias:
            msg["reasoning"] = reasoning_content
    msg["content"] = content if content else None
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def build_response_obj(*, chatcmpl_id: str, created: int, model: str,
                       message: dict, finish_reason: str, usage: dict) -> dict:
    return {
        "id": chatcmpl_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "system_fingerprint": SYSTEM_FINGERPRINT,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
            "logprobs": None,
        }],
        "usage": usage,
    }


def build_chunk(*, chatcmpl_id: str, created: int, model: str,
                delta: dict | None = None, finish_reason: str | None = None,
                usage: dict | None = None) -> dict:
    """Chunk skeleton per spec §4.3 / usage chunk per §4.6."""
    obj: dict[str, Any] = {
        "id": chatcmpl_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "system_fingerprint": SYSTEM_FINGERPRINT,
    }
    if usage is not None:
        obj["choices"] = []
        obj["usage"] = usage
    else:
        obj["choices"] = [{
            "index": 0,
            "delta": delta if delta is not None else {},
            "finish_reason": finish_reason,
            "logprobs": None,
        }]
    return obj


def sse_frame(chunk_obj: dict) -> str:
    return f"data: {dumps(chunk_obj)}\n\n"


# ----------------------------------------------------------- delta planning
def plan_deltas(*, reasoning_content: str | None, content: str | None,
                tool_calls: list[dict] | None, splitter, instant: bool,
                interleave: bool, alias: bool) -> list[tuple[dict, bool]]:
    """Ordered list of (delta, paced) per spec §4.2/§4.4.

    paced=True deltas carry ~one token and participate in rate limiting.
    The finish/usage chunks are not planned here — the SSE consumer emits them
    on the finish marker so that aborts always close the stream legally.
    """
    deltas: list[tuple[dict, bool]] = [({"role": "assistant"}, False)]

    def pieces(text: str) -> list[str]:
        return [text] if instant else splitter(text)

    if reasoning_content:
        for p in pieces(reasoning_content):
            d = {"reasoning_content": p}
            if alias:
                d["reasoning"] = p
            deltas.append((d, True))
    if content:
        for p in pieces(content):
            deltas.append(({"content": p}, True))

    if tool_calls:
        metas = []
        frags: list[list[str]] = []
        for i, tc in enumerate(tool_calls):
            metas.append({"tool_calls": [{
                "index": i, "id": tc["id"], "type": "function",
                "function": {"name": tc["function"]["name"], "arguments": ""},
            }]})
            frags.append(pieces(tc["function"]["arguments"]))
        if interleave and len(tool_calls) > 1:
            # stress mode: fragments of different indexes interleaved (spec §4.4)
            sent_meta = [False] * len(tool_calls)
            queues = [list(f) for f in frags]
            while any(queues) or not all(sent_meta):
                for i in range(len(tool_calls)):
                    if not sent_meta[i]:
                        deltas.append((metas[i], False))
                        sent_meta[i] = True
                    if queues[i]:
                        p = queues[i].pop(0)
                        deltas.append(({"tool_calls": [
                            {"index": i, "function": {"arguments": p}}]}, True))
        else:
            for i in range(len(tool_calls)):
                deltas.append((metas[i], False))
                for p in frags[i]:
                    deltas.append(({"tool_calls": [
                        {"index": i, "function": {"arguments": p}}]}, True))
    return deltas


# -------------------------------------------------------------- reconstruction
def reconstruct_from_frames(frames: list[str]) -> dict | None:
    """Rebuild the §3 non-streaming object from sent SSE frames (spec §11).

    Returns None if no data chunk was ever sent. The usage key is present only
    if a usage chunk was on the wire.
    """
    chunks = []
    for raw in frames:
        payload = raw.strip()
        if not payload.startswith("data:"):
            continue
        payload = payload[len("data:"):].strip()
        if payload == "[DONE]":
            continue
        try:
            chunks.append(json.loads(payload))
        except ValueError:
            continue
    if not chunks:
        return None

    first = chunks[0]
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    tool_calls: dict[int, dict] = {}
    finish_reason = None
    usage = None
    for ch in chunks:
        if not ch.get("choices"):
            if ch.get("usage") is not None:
                usage = ch["usage"]
            continue
        choice = ch["choices"][0]
        delta = choice.get("delta") or {}
        if isinstance(delta.get("reasoning_content"), str):
            reasoning_parts.append(delta["reasoning_content"])
        if isinstance(delta.get("content"), str):
            content_parts.append(delta["content"])
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            slot = tool_calls.setdefault(
                idx, {"id": None, "type": "function",
                      "function": {"name": "", "arguments": ""}})
            if tc.get("id"):
                slot["id"] = tc["id"]
            if tc.get("type"):
                slot["type"] = tc["type"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["function"]["name"] = fn["name"]
            if isinstance(fn.get("arguments"), str):
                slot["function"]["arguments"] += fn["arguments"]
        if choice.get("finish_reason"):
            finish_reason = choice["finish_reason"]

    wire_tool_calls = [tool_calls[i] for i in sorted(tool_calls)] or None
    message = build_message(
        reasoning_content="".join(reasoning_parts) or None,
        content="".join(content_parts) or None,
        tool_calls=wire_tool_calls,
        alias=False,
    )
    obj = {
        "id": first.get("id"),
        "object": "chat.completion",
        "created": first.get("created"),
        "model": first.get("model"),
        "system_fingerprint": first.get("system_fingerprint"),
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish_reason,
            "logprobs": None,
        }],
    }
    if usage is not None:
        obj["usage"] = usage
    return obj


# ----------------------------------------------------------- draft validation
def resolve_finish_reason(requested: str | None, has_tool_calls: bool) -> str:
    """Console auto rule (spec §6): tool_calls auto-set; manual picks honored."""
    if requested and requested != "auto":
        if requested not in FINISH_REASONS:
            raise ValueError(f"invalid finish_reason: {requested}")
        return requested
    return "tool_calls" if has_tool_calls else "stop"


def validate_draft(parsed_request: dict, draft: dict) -> tuple[list[str], list[str], list[str]]:
    """Returns (blockers, warnings, auto_tags).

    Blockers stop submission unless the operator bypasses (spec §2.1
    response_format, §3 broken arguments). Warnings never block (spec
    tool_choice policy). auto_tags are applied when a bypass occurs.
    """
    blockers: list[str] = []
    warnings: list[str] = []
    auto_tags: list[str] = []

    tool_calls = draft.get("tool_calls") or []
    for i, tc in enumerate(tool_calls):
        args = tc.get("arguments")
        if not isinstance(args, str):
            continue  # object form: always serializable
        try:
            json.loads(args)
        except ValueError:
            blockers.append(f"tool_calls[{i}].arguments is not valid JSON")
            if "broken_tool_arguments" not in auto_tags:
                auto_tags.append("broken_tool_arguments")

    rf = parsed_request.get("response_format")
    rf_type = rf.get("type") if isinstance(rf, dict) else None
    content = draft.get("content") or None
    if rf_type in ("json_object", "json_schema") and not tool_calls:
        if not content:
            blockers.append(f"response_format={rf_type} requires JSON content but content is empty")
            auto_tags.append("response_format_violation")
        else:
            parsed_content = None
            try:
                parsed_content = json.loads(content)
            except ValueError:
                blockers.append(f"content is not valid JSON (response_format={rf_type})")
                auto_tags.append("response_format_violation")
            if parsed_content is not None and rf_type == "json_schema":
                schema = (rf.get("json_schema") or {}).get("schema")
                if schema:
                    try:
                        import jsonschema
                        try:
                            jsonschema.validate(parsed_content, schema)
                        except jsonschema.ValidationError as e:
                            blockers.append(f"content violates json_schema: {e.message}")
                            auto_tags.append("response_format_violation")
                    except ImportError:
                        warnings.append("jsonschema not installed — schema not checked")

    tc_choice = parsed_request.get("tool_choice")
    if tc_choice == "required" and not tool_calls:
        warnings.append("tool_choice=required but draft has no tool calls")
    if tc_choice == "none" and tool_calls:
        warnings.append("tool_choice=none but draft has tool calls")
    if isinstance(tc_choice, dict) and tc_choice.get("type") == "function":
        want = (tc_choice.get("function") or {}).get("name")
        names = [tc.get("name") for tc in tool_calls]
        if want and want not in names:
            warnings.append(f"tool_choice forces function '{want}' but draft calls {names or 'none'}")
    if tool_calls:
        known = {t.get("function", {}).get("name")
                 for t in parsed_request.get("tools") or [] if isinstance(t, dict)}
        for tc in tool_calls:
            if known and tc.get("name") not in known:
                warnings.append(f"tool call '{tc.get('name')}' is not in the request tools list")

    return blockers, warnings, auto_tags
