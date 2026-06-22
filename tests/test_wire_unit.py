import json

import pytest

from maycomb import wire
from maycomb.tokenizer import TokenCodec, build_usage, serialize_message_for_count

CODEC = TokenCodec("approx")


# ------------------------------------------------------------- normalization
def test_normalize_sorts_and_flags_unknown():
    parsed = {"stream": True, "model": "m", "messages": [{"role": "user", "content": "hi"}],
              "zeta_custom": 1, "alpha_custom": {"b": 1, "a": 2}}
    normalized, unknown = wire.normalize_request(parsed)
    assert unknown == ["alpha_custom", "zeta_custom"]
    assert list(normalized.keys()) == sorted(parsed.keys())
    assert list(normalized["alpha_custom"].keys()) == ["a", "b"]
    # only-sent fields: no defaults injected
    assert "temperature" not in normalized


def test_validate_request_rules():
    base = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    wire.validate_request(base)  # ok
    with pytest.raises(wire.ApiError) as e:
        wire.validate_request({**base, "n": 3})
    assert e.value.status == 400 and e.value.body["error"]["param"] == "n"
    with pytest.raises(wire.ApiError) as e:
        wire.validate_request({**base, "logprobs": True})
    assert e.value.body["error"]["param"] == "logprobs"
    with pytest.raises(wire.ApiError) as e:
        wire.validate_request({"model": "m", "messages": []})
    assert e.value.body["error"]["param"] == "messages"
    wire.validate_request({**base, "n": 1, "logprobs": False})  # ok


# ------------------------------------------------------------------ message
def test_build_message_rules():
    m = wire.build_message(reasoning_content=None, content="", tool_calls=None, alias=False)
    assert "reasoning_content" not in m       # omitted, never empty string (§3)
    assert m["content"] is None               # empty -> null
    tc = [{"id": "call_x", "type": "function",
           "function": {"name": "f", "arguments": "{}"}}]
    m = wire.build_message(reasoning_content="생각", content=None, tool_calls=tc, alias=True)
    assert list(m.keys()) == ["role", "reasoning_content", "reasoning", "content", "tool_calls"]
    assert m["reasoning"] == "생각"


def test_build_wire_tool_calls_normalizes_and_keeps_broken():
    out = wire.build_wire_tool_calls([
        {"id": "call_a", "name": "f", "arguments": '{ "x" : 1 }'},
        {"id": "call_b", "name": "g", "arguments": '{"broken": '},
    ])
    assert out[0]["function"]["arguments"] == '{"x":1}'       # normalized string
    assert out[1]["function"]["arguments"] == '{"broken": '   # intentionally broken kept


def test_resolve_finish_reason():
    assert wire.resolve_finish_reason("auto", False) == "stop"
    assert wire.resolve_finish_reason("auto", True) == "tool_calls"
    assert wire.resolve_finish_reason(None, True) == "tool_calls"
    assert wire.resolve_finish_reason("length", True) == "length"
    assert wire.resolve_finish_reason("content_filter", False) == "content_filter"
    with pytest.raises(ValueError):
        wire.resolve_finish_reason("bogus", False)


# ------------------------------------------------------------------- chunks
def _plan(interleave=False, instant=False):
    tcs = wire.build_wire_tool_calls([
        {"id": "call_a", "name": "get_weather", "arguments": '{"city": "Seoul", "unit": "celsius"}'},
        {"id": "call_b", "name": "get_time", "arguments": '{"tz": "Asia/Seoul"}'},
    ])
    return wire.plan_deltas(
        reasoning_content="차근차근 생각해 보자.", content=None, tool_calls=tcs,
        splitter=CODEC.split_pieces, instant=instant, interleave=interleave, alias=False)


def _phase(delta: dict) -> int:
    if "role" in delta:
        return 0
    if "reasoning_content" in delta:
        return 1
    if "content" in delta:
        return 2
    if "tool_calls" in delta:
        return 3
    return 4


def test_plan_deltas_order_sequential():
    deltas = [d for d, _ in _plan()]
    assert deltas[0] == {"role": "assistant"}
    phases = [_phase(d) for d in deltas]
    assert phases == sorted(phases)  # §4.2: no regression
    # tool call delta shape: first chunk per index has full meta + empty args
    seen_meta = set()
    for d in deltas:
        for tc in d.get("tool_calls", []):
            if tc["index"] not in seen_meta:
                assert tc["id"].startswith("call_") and tc["type"] == "function"
                assert tc["function"]["arguments"] == ""
                seen_meta.add(tc["index"])
            else:
                assert set(tc.keys()) == {"index", "function"}
                assert "name" not in tc["function"]
    # sequential mode: index 0 fragments all precede index 1's meta
    idx_order = [tc["index"] for d in deltas for tc in d.get("tool_calls", [])]
    assert idx_order == sorted(idx_order)


def test_plan_deltas_interleave():
    idx_order = [tc["index"] for d, _ in _plan(interleave=True)
                 for tc in d.get("tool_calls", [])]
    switches = sum(1 for a, b in zip(idx_order, idx_order[1:]) if a != b)
    assert switches > 2  # fragments of different indexes are interleaved


def test_instant_minimal_chunks():
    deltas = [d for d, _ in _plan(instant=True)]
    # role + reasoning(1) + 2 * (meta + args)
    assert len(deltas) == 1 + 1 + 4


# ----------------------------------------------------------- reconstruction
def test_reconstruction_matches_built_response():
    tcs = wire.build_wire_tool_calls([
        {"id": "call_a", "name": "get_weather", "arguments": '{"city": "서울"}'}])
    deltas = wire.plan_deltas(
        reasoning_content="생각 중", content="결과는 다음과 같습니다",
        tool_calls=tcs, splitter=CODEC.split_pieces,
        instant=False, interleave=False, alias=False)
    frames = [wire.sse_frame(wire.build_chunk(
        chatcmpl_id="chatcmpl-X", created=123, model="m", delta=d)) for d, _ in deltas]
    frames.append(wire.sse_frame(wire.build_chunk(
        chatcmpl_id="chatcmpl-X", created=123, model="m",
        delta={}, finish_reason="tool_calls")))
    usage = {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3,
             "prompt_tokens_details": {"cached_tokens": 0},
             "completion_tokens_details": {"reasoning_tokens": 1}}
    frames.append(wire.sse_frame(wire.build_chunk(
        chatcmpl_id="chatcmpl-X", created=123, model="m", usage=usage)))
    frames.append(wire.SSE_DONE)

    recon = wire.reconstruct_from_frames(frames)
    expected_msg = wire.build_message(
        reasoning_content="생각 중", content="결과는 다음과 같습니다",
        tool_calls=tcs, alias=False)
    expected = wire.build_response_obj(
        chatcmpl_id="chatcmpl-X", created=123, model="m",
        message=expected_msg, finish_reason="tool_calls", usage=usage)
    assert recon == expected


# -------------------------------------------------------------------- usage
def test_usage_math_relational():
    messages = [{"role": "system", "content": "you are helpful"},
                {"role": "user", "content": [{"type": "text", "text": "안녕하세요"}]}]
    tcs = [{"id": "c", "type": "function",
            "function": {"name": "fn", "arguments": '{"a":1}'}}]
    u = build_usage(CODEC, messages, reasoning_content="깊은 생각",
                    content="응답", tool_calls=tcs)
    expected_prompt = sum(3 + CODEC.count(serialize_message_for_count(m))
                          for m in messages) + 3
    assert u["prompt_tokens"] == expected_prompt
    rt = CODEC.count("깊은 생각")
    assert u["completion_tokens_details"]["reasoning_tokens"] == rt
    assert u["completion_tokens"] == (rt + CODEC.count("응답")
                                      + CODEC.count("fn") + CODEC.count('{"a":1}'))
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
    assert u["prompt_tokens_details"] == {"cached_tokens": 0}


def test_split_pieces_preserves_text():
    text = "한국어와 English mixed 텍스트, with punctuation! 🎈"
    for codec in (CODEC, TokenCodec("o200k_base")):
        assert "".join(codec.split_pieces(text)) == text


# --------------------------------------------------------- draft validation
def test_validate_draft_blockers_and_warnings():
    req = {"model": "m", "messages": [{"role": "user", "content": "x"}],
           "tools": [{"type": "function", "function": {"name": "known_fn"}}],
           "tool_choice": "required",
           "response_format": {"type": "json_object"}}
    draft = {"reasoning_content": None, "content": "not-json", "finish_reason": "auto",
             "tool_calls": []}
    blockers, warnings, tags = wire.validate_draft(req, draft)
    assert any("not valid JSON" in b for b in blockers)
    assert "response_format_violation" in tags
    assert any("tool_choice=required" in w for w in warnings)

    draft2 = {"reasoning_content": None, "content": None, "finish_reason": "auto",
              "tool_calls": [{"id": "c", "name": "other_fn", "arguments": "{broken"}]}
    blockers2, warnings2, tags2 = wire.validate_draft(req, draft2)
    assert any("arguments is not valid JSON" in b for b in blockers2)
    assert "broken_tool_arguments" in tags2
    assert any("not in the request tools list" in w for w in warnings2)


def test_validate_draft_json_schema():
    schema = {"type": "object", "properties": {"x": {"type": "integer"}},
              "required": ["x"]}
    req = {"model": "m", "messages": [],
           "response_format": {"type": "json_schema",
                               "json_schema": {"name": "t", "schema": schema}}}
    ok = {"reasoning_content": None, "content": json.dumps({"x": 1}),
          "finish_reason": "auto", "tool_calls": []}
    bad = {**ok, "content": json.dumps({"x": "no"})}
    assert wire.validate_draft(req, ok)[0] == []
    assert any("violates json_schema" in b for b in wire.validate_draft(req, bad)[0])
