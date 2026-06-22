import asyncio
import json

import pytest

from maycomb import wire
from maycomb.state import OpError

from .conftest import AUTH, sse_chunks, sse_data_frames, wait_pending

pytestmark = pytest.mark.anyio

STREAM_REQ = {
    "model": "mock-large-1",
    "messages": [{"role": "user", "content": "서울 날씨와 시간 알려줘"}],
    "tools": [
        {"type": "function", "function": {"name": "get_weather",
                                          "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "get_time",
                                          "parameters": {"type": "object"}}},
    ],
    "stream": True,
}


def phase_of(chunk: dict) -> int:
    if not chunk.get("choices"):
        return 5  # usage chunk
    c = chunk["choices"][0]
    d = c.get("delta") or {}
    if c.get("finish_reason"):
        return 4
    if "role" in d:
        return 0
    if "reasoning_content" in d:
        return 1
    if "content" in d:
        return 2
    if "tool_calls" in d:
        return 3
    return 4


SUBMIT_DRAFT = {
    "reasoning_content": "도구 두 개를 호출해야 한다.",
    "content": "",
    "finish_reason": "auto",
    "tool_calls": [
        {"id": "call_AAAAAAAAAAAAAAAAAAAAAAAA", "name": "get_weather",
         "arguments": '{"city": "Seoul", "unit": "celsius"}'},
        {"id": "call_BBBBBBBBBBBBBBBBBBBBBBBB", "name": "get_time",
         "arguments": '{"tz": "Asia/Seoul"}'},
    ],
}


async def test_stream_roundtrip_instant_with_usage(client):
    req = {**STREAM_REQ, "stream_options": {"include_usage": True}}
    post = asyncio.create_task(
        client.post("/v1/chat/completions", json=req, headers=AUTH))
    [pending] = await wait_pending(client)
    xid = pending["exchange_id"]
    assert pending["flags"]["include_usage"] is True

    r = await client.post(f"/api/exchanges/{xid}/submit",
                          json={"draft": SUBMIT_DRAFT, "mode": "instant"})
    assert r.status_code == 200

    resp = await post
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    frames = sse_data_frames(resp.text)
    assert frames[-1].strip() == "data: [DONE]"
    chunks = sse_chunks(resp.text)

    # common skeleton (§4.3): same id/created/model on every chunk
    assert len({c["id"] for c in chunks}) == 1
    assert chunks[0]["id"] == pending["chatcmpl_id"]
    assert len({c["created"] for c in chunks}) == 1
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert all(c["system_fingerprint"] == "fp_mockup_v1" for c in chunks)

    # §4.2 order: role -> reasoning -> tool_calls -> finish -> usage
    phases = [phase_of(c) for c in chunks]
    assert phases == sorted(phases)
    assert phases.count(0) == 1 and phases.count(4) == 1 and phases.count(5) == 1
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}

    # §4.4 tool delta shape
    first_seen = set()
    for c in chunks:
        for tc in (c["choices"][0].get("delta") or {}).get("tool_calls", []) if c.get("choices") else []:
            if tc["index"] not in first_seen:
                first_seen.add(tc["index"])
                assert tc["id"] and tc["type"] == "function"
                assert tc["function"]["arguments"] == ""
            else:
                assert "id" not in tc
    assert first_seen == {0, 1}

    finish_chunk = chunks[phases.index(4)]
    assert finish_chunk["choices"][0]["delta"] == {}
    assert finish_chunk["choices"][0]["finish_reason"] == "tool_calls"

    usage_chunk = chunks[-1]
    assert usage_chunk["choices"] == []
    assert usage_chunk["usage"]["completion_tokens_details"]["reasoning_tokens"] > 0

    # dual representation (§11): stored response == replay of stored chunks
    events = (await client.get(f"/api/exchanges/{xid}/events")).json()["events"]
    chunk_events = [e for e in events if e["type"] == "chunk_sent"]
    assert len(chunk_events) == len(frames)
    assert [e["data"]["chunk_index"] for e in chunk_events] == list(range(len(frames)))
    stored_frames = [e["data"]["sse_payload_raw"] for e in chunk_events]
    submitted = next(e for e in events if e["type"] == "response_submitted")
    assert wire.reconstruct_from_frames(stored_frames) == submitted["data"]["response"]
    assert submitted["data"]["meta"]["partial"] is False

    # the client could rebuild the same object from what it received
    client_recon = wire.reconstruct_from_frames(frames)
    assert client_recon == submitted["data"]["response"]
    msg = client_recon["choices"][0]["message"]
    assert msg["content"] is None
    assert json.loads(msg["tool_calls"][0]["function"]["arguments"]) == {
        "city": "Seoul", "unit": "celsius"}


async def test_stream_paced_splits_and_no_usage_chunk(client):
    post = asyncio.create_task(
        client.post("/v1/chat/completions", json=STREAM_REQ, headers=AUTH))
    [pending] = await wait_pending(client)
    xid = pending["exchange_id"]
    text = "여러 단어로 이루어진 페이스드 모드 테스트 문장입니다 hello paced world"
    await client.post(f"/api/exchanges/{xid}/submit", json={
        "draft": {"reasoning_content": "", "content": text,
                  "finish_reason": "auto", "tool_calls": []},
        "mode": "paced",
        "pacing": {"tokens_per_second": 2000},
    })
    resp = await post
    chunks = sse_chunks(resp.text)
    content_chunks = [c for c in chunks
                      if c.get("choices") and "content" in (c["choices"][0]["delta"] or {})]
    assert len(content_chunks) > 3  # actually split into pieces
    assert "".join(c["choices"][0]["delta"]["content"] for c in content_chunks) == text
    # include_usage absent -> no usage chunk on the wire (§4.6)
    assert all(c.get("choices") for c in chunks)
    # ...but usage is still recorded in storage
    events = (await client.get(f"/api/exchanges/{xid}/events")).json()["events"]
    submitted = next(e for e in events if e["type"] == "response_submitted")
    assert submitted["data"]["usage"]["completion_tokens"] > 0
    assert "usage" not in submitted["data"]["response"]


async def test_keepalive_comment_before_first_chunk(client):
    post = asyncio.create_task(
        client.post("/v1/chat/completions", json=STREAM_REQ, headers=AUTH))
    [pending] = await wait_pending(client)
    await asyncio.sleep(0.5)  # keepalive_interval=0.2 in test settings
    await client.post(f"/api/exchanges/{pending['exchange_id']}/submit", json={
        "draft": {"content": "응답", "tool_calls": []}, "mode": "instant"})
    text = (await post).text
    assert ": keep-alive" in text
    assert text.index(": keep-alive") < text.index("data:")


async def test_graceful_abort_without_any_submit(client):
    """Pending stream + graceful finish: role + finish(length) + [DONE] (§8)."""
    post = asyncio.create_task(
        client.post("/v1/chat/completions", json=STREAM_REQ, headers=AUTH))
    [pending] = await wait_pending(client)
    xid = pending["exchange_id"]
    r = await client.post(f"/api/exchanges/{xid}/abort",
                          json={"kind": "graceful", "finish_reason": "length"})
    assert r.status_code == 200
    resp = await post
    assert resp.status_code == 200
    chunks = sse_chunks(resp.text)
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert sse_data_frames(resp.text)[-1].strip() == "data: [DONE]"
    detail = (await client.get(f"/api/exchanges/{xid}")).json()
    assert detail["summary"]["status"] == "completed"
    assert detail["summary"]["partial"] is True
    assert detail["result"]["data"]["meta"]["partial"] is True


async def test_graceful_abort_mid_stream_is_partial_prefix(client):
    post = asyncio.create_task(
        client.post("/v1/chat/completions", json=STREAM_REQ, headers=AUTH))
    [pending] = await wait_pending(client)
    xid = pending["exchange_id"]
    long_text = " ".join(f"단어{i}" for i in range(300))
    await client.post(f"/api/exchanges/{xid}/submit", json={
        "draft": {"content": long_text, "tool_calls": []},
        "mode": "paced", "pacing": {"tokens_per_second": 50},
    })
    await asyncio.sleep(0.3)  # let a few chunks flow
    await client.post(f"/api/exchanges/{xid}/abort",
                      json={"kind": "graceful", "finish_reason": "stop"})
    resp = await post
    chunks = sse_chunks(resp.text)
    got = "".join((c["choices"][0]["delta"] or {}).get("content", "")
                  for c in chunks if c.get("choices"))
    assert 0 < len(got) < len(long_text)
    assert long_text.startswith(got)
    detail = (await client.get(f"/api/exchanges/{xid}")).json()
    assert detail["summary"]["partial"] is True
    # stored usage reflects only what was actually sent
    assert detail["result"]["data"]["usage"]["completion_tokens"] < 300


async def test_hard_abort_truncates_stream(client):
    post = asyncio.create_task(
        client.post("/v1/chat/completions", json=STREAM_REQ, headers=AUTH))
    [pending] = await wait_pending(client)
    xid = pending["exchange_id"]
    await client.post(f"/api/exchanges/{xid}/submit", json={
        "draft": {"content": " ".join(["긴 텍스트"] * 200), "tool_calls": []},
        "mode": "paced", "pacing": {"tokens_per_second": 50},
    })
    await asyncio.sleep(0.2)
    r = await client.post(f"/api/exchanges/{xid}/abort", json={"kind": "hard"})
    assert r.status_code == 200
    with pytest.raises(Exception):
        await post  # connection severed without finish chunk / [DONE]
    detail = (await client.get(f"/api/exchanges/{xid}")).json()
    assert detail["summary"]["status"] == "aborted"
    assert detail["result"]["data"]["reason"] == "operator_hard_abort"
    assert detail["result"]["data"]["reconstruction"] is not None


async def test_http_error_injection_with_delay_armed(client):
    """delay arms first (suppresses keep-alive/headers), then a 500 injection
    can still become a real HTTP status on a streaming request (§9.2)."""
    post = asyncio.create_task(
        client.post("/v1/chat/completions", json=STREAM_REQ, headers=AUTH))
    [pending] = await wait_pending(client)
    xid = pending["exchange_id"]
    await client.post(f"/api/exchanges/{xid}/inject",
                      json={"kind": "delay", "params": {"delay_ms": 60000}})
    await asyncio.sleep(0.4)  # longer than keepalive interval: headers still held
    r = await client.post(f"/api/exchanges/{xid}/inject",
                          json={"kind": "server_error"})
    assert r.status_code == 200
    resp = await post
    assert resp.status_code == 500
    assert resp.json()["error"]["type"] == "server_error"
    detail = (await client.get(f"/api/exchanges/{xid}")).json()
    assert detail["summary"]["status"] == "injected"


async def test_stream_cut_after_n_chunks(client):
    post = asyncio.create_task(
        client.post("/v1/chat/completions", json=STREAM_REQ, headers=AUTH))
    [pending] = await wait_pending(client)
    xid = pending["exchange_id"]
    await client.post(f"/api/exchanges/{xid}/inject",
                      json={"kind": "stream_cut", "params": {"after_chunks": 3}})
    await client.post(f"/api/exchanges/{xid}/submit", json={
        "draft": {"content": " ".join(["문장입니다"] * 50), "tool_calls": []},
        "mode": "paced", "pacing": {"tokens_per_second": 2000},
    })
    with pytest.raises(Exception):
        await post
    detail = (await client.get(f"/api/exchanges/{xid}")).json()
    assert detail["summary"]["status"] == "aborted"
    assert detail["result"]["data"]["reason"] == "stream_cut_injected"
    events = (await client.get(f"/api/exchanges/{xid}/events")).json()["events"]
    sent = [e for e in events if e["type"] == "chunk_sent"]
    assert len(sent) == 3


async def test_live_mode_order_and_completion(client, app):
    post = asyncio.create_task(
        client.post("/v1/chat/completions", json=STREAM_REQ, headers=AUTH))
    [pending] = await wait_pending(client)
    xid = pending["exchange_id"]
    hub = app.state.hub

    await hub.live_message({"type": "live_start", "exchange_id": xid})
    await hub.live_message({"type": "live_text", "exchange_id": xid,
                            "field": "reasoning_content", "text": "추론 "})
    await hub.live_message({"type": "live_text", "exchange_id": xid,
                            "field": "reasoning_content", "text": "계속"})
    await hub.live_message({"type": "live_text", "exchange_id": xid,
                            "field": "content", "text": "안녕"})
    with pytest.raises(OpError):  # reasoning after content: forbidden (§4.2)
        await hub.live_message({"type": "live_text", "exchange_id": xid,
                                "field": "reasoning_content", "text": "역행"})
    await hub.live_message({"type": "live_tool_call", "exchange_id": xid,
                            "name": "get_weather", "arguments": '{"city":"서울"}'})
    with pytest.raises(OpError):  # content after tool_calls: forbidden
        await hub.live_message({"type": "live_text", "exchange_id": xid,
                                "field": "content", "text": "역행"})
    ack = await hub.live_message({"type": "live_finish", "exchange_id": xid,
                                  "finish_reason": "auto"})
    assert ack["finish_reason"] == "tool_calls"

    resp = await post
    chunks = sse_chunks(resp.text)
    phases = [phase_of(c) for c in chunks]
    assert phases == sorted(phases)
    recon = wire.reconstruct_from_frames(sse_data_frames(resp.text))
    msg = recon["choices"][0]["message"]
    assert msg["reasoning_content"] == "추론 계속"
    assert msg["content"] == "안녕"
    assert msg["tool_calls"][0]["function"]["name"] == "get_weather"
    detail = (await client.get(f"/api/exchanges/{xid}")).json()
    assert detail["result"]["data"]["meta"]["mode"] == "live"


async def test_interleaved_tool_call_stress_mode(client):
    post = asyncio.create_task(
        client.post("/v1/chat/completions", json=STREAM_REQ, headers=AUTH))
    [pending] = await wait_pending(client)
    xid = pending["exchange_id"]
    await client.post(f"/api/exchanges/{xid}/submit", json={
        "draft": SUBMIT_DRAFT, "mode": "paced",
        "pacing": {"tokens_per_second": 5000, "interleave": True},
    })
    resp = await post
    idx_order = []
    for c in sse_chunks(resp.text):
        if not c.get("choices"):
            continue
        for tc in (c["choices"][0]["delta"] or {}).get("tool_calls", []):
            idx_order.append(tc["index"])
    switches = sum(1 for a, b in zip(idx_order, idx_order[1:]) if a != b)
    assert switches > 2
    # interleaved fragments still reassemble correctly
    recon = wire.reconstruct_from_frames(sse_data_frames(resp.text))
    tcs = recon["choices"][0]["message"]["tool_calls"]
    assert json.loads(tcs[0]["function"]["arguments"]) == {"city": "Seoul",
                                                           "unit": "celsius"}
    assert json.loads(tcs[1]["function"]["arguments"]) == {"tz": "Asia/Seoul"}
