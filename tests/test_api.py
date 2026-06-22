import asyncio
import json

import pytest

from .conftest import AUTH, wait_pending

pytestmark = pytest.mark.anyio

BASE_REQ = {
    "model": "my-arbitrary-model",
    "messages": [{"role": "user", "content": "서울 날씨 알려줘"}],
}


# ----------------------------------------------------------------- rejects
async def test_models_auth_and_list(client):
    r = await client.get("/v1/models")
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "authentication_error"

    r = await client.get("/v1/models", headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert "mock-large-1" in ids and "mock-reasoning-1" in ids
    assert all(m["owned_by"] == "mockup" for m in body["data"])


async def test_chat_missing_auth(client):
    r = await client.post("/v1/chat/completions", json=BASE_REQ)
    assert r.status_code == 401
    assert r.json()["error"]["type"] == "authentication_error"


async def test_n_gt_1_rejected(client):
    r = await client.post("/v1/chat/completions", json={**BASE_REQ, "n": 3}, headers=AUTH)
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"
    assert err["param"] == "n"
    assert "n=3" in err["message"]


async def test_logprobs_rejected(client):
    r = await client.post("/v1/chat/completions",
                          json={**BASE_REQ, "logprobs": True}, headers=AUTH)
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "logprobs"


async def test_empty_messages_rejected(client):
    r = await client.post("/v1/chat/completions",
                          json={"model": "m", "messages": []}, headers=AUTH)
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "messages"


async def test_invalid_json_rejected_but_preserved(client):
    r = await client.post("/v1/chat/completions", content=b"{this is not json",
                          headers={**AUTH, "Content-Type": "application/json"})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    # raw preserved + rejected exchange visible to the console (spec §9.1)
    data = (await client.get("/api/exchanges", params={"status": "rejected"})).json()
    assert len(data["exchanges"]) == 1
    xid = data["exchanges"][0]["exchange_id"]
    raw = await client.get(f"/api/exchanges/{xid}/raw")
    assert raw.text == "{this is not json"


# ------------------------------------------------------------ happy path
async def test_nonstream_roundtrip(client):
    req = {**BASE_REQ, "temperature": 0.7, "weird_field": {"z": 1, "a": 2}}
    post = asyncio.create_task(
        client.post("/v1/chat/completions", json=req, headers=AUTH))
    [pending] = await wait_pending(client)
    xid = pending["exchange_id"]
    assert pending["model"] == "my-arbitrary-model"
    assert pending["flags"]["unknown_fields"] == ["weird_field"]

    sub = await client.post(f"/api/exchanges/{xid}/submit", json={
        "draft": {
            "reasoning_content": "사용자가 날씨를 물었다.",
            "content": "오늘 서울은 맑습니다.",
            "finish_reason": "auto",
            "tool_calls": [],
        },
        "meta": {"operator_note": "demo", "tags": ["smoke"]},
    })
    assert sub.status_code == 200
    assert sub.json()["finish_reason"] == "stop"

    resp = await post
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/json")
    body = resp.json()
    assert body["id"] == pending["chatcmpl_id"]
    assert body["id"].startswith("chatcmpl-") and len(body["id"]) == 9 + 24
    assert body["object"] == "chat.completion"
    assert body["system_fingerprint"] == "fp_mockup_v1"
    assert body["model"] == "my-arbitrary-model"   # echo, even unknown model
    choice = body["choices"][0]
    assert choice["logprobs"] is None
    assert choice["finish_reason"] == "stop"
    msg = choice["message"]
    assert list(msg.keys())[0] == "role"
    assert msg["reasoning_content"] == "사용자가 날씨를 물었다."
    assert msg["content"] == "오늘 서울은 맑습니다."
    u = body["usage"]
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
    assert u["prompt_tokens_details"] == {"cached_tokens": 0}
    assert u["completion_tokens_details"]["reasoning_tokens"] > 0

    # stored wire bytes identical to what the client received (spec §11)
    events = (await client.get(f"/api/exchanges/{xid}/events")).json()["events"]
    by_type = {}
    for e in events:
        by_type.setdefault(e["type"], []).append(e)
    assert by_type["response_submitted"][0]["data"]["response_raw"] == resp.text
    assert by_type["response_submitted"][0]["data"]["meta"]["tags"] == ["smoke"]
    parsed = by_type["request_parsed"][0]["data"]
    assert parsed["_unknown_fields"] == ["weird_field"]
    assert list(parsed["request"]["weird_field"].keys()) == ["a", "z"]
    assert [e["seq"] for e in events] == sorted(e["seq"] for e in events)
    assert events[0]["type"] == "request_received"
    assert all(e["v"] == "wire/1" and e["data"]["v"] == "wire/1" for e in events)

    # exchange now completed
    detail = (await client.get(f"/api/exchanges/{xid}")).json()
    assert detail["summary"]["status"] == "completed"
    assert detail["summary"]["finish_reason"] == "stop"


async def test_reasoning_omitted_and_content_null(client):
    """tool-call-only response: content null, no reasoning key (§3)."""
    post = asyncio.create_task(
        client.post("/v1/chat/completions", json=BASE_REQ, headers=AUTH))
    [pending] = await wait_pending(client)
    xid = pending["exchange_id"]
    r = await client.post(f"/api/exchanges/{xid}/submit", json={
        "draft": {"reasoning_content": "", "content": "",
                  "finish_reason": "auto",
                  "tool_calls": [{"name": "get_weather",
                                  "arguments": "{\"city\": \"Seoul\"}"}]},
    })
    assert r.status_code == 200
    body = (await post).json()
    msg = body["choices"][0]["message"]
    assert "reasoning_content" not in msg
    assert msg["content"] is None
    assert body["choices"][0]["finish_reason"] == "tool_calls"
    tc = msg["tool_calls"][0]
    assert tc["id"].startswith("call_") and len(tc["id"]) == 5 + 24
    assert tc["type"] == "function"
    assert json.loads(tc["function"]["arguments"]) == {"city": "Seoul"}


# --------------------------------------------------- response_format guard
async def test_response_format_blocks_then_bypass(client):
    req = {**BASE_REQ, "response_format": {"type": "json_object"}}
    post = asyncio.create_task(
        client.post("/v1/chat/completions", json=req, headers=AUTH))
    [pending] = await wait_pending(client)
    xid = pending["exchange_id"]

    bad = {"draft": {"reasoning_content": "", "content": "이건 JSON이 아님",
                     "finish_reason": "auto", "tool_calls": []}}
    r = await client.post(f"/api/exchanges/{xid}/submit", json=bad)
    assert r.status_code == 422
    assert "valid JSON" in r.json()["detail"]

    r = await client.post(f"/api/exchanges/{xid}/submit",
                          json={**bad, "meta": {"validation_bypass": True}})
    assert r.status_code == 200
    body = (await post).json()
    assert body["choices"][0]["message"]["content"] == "이건 JSON이 아님"
    detail = (await client.get(f"/api/exchanges/{xid}")).json()
    tags = detail["summary"]["tags"]
    assert "validation_bypassed" in tags and "response_format_violation" in tags
    assert detail["result"]["data"]["meta"]["validation_bypassed"] is True


# -------------------------------------------------------------- injection
async def test_inject_rate_limit_nonstream(client):
    post = asyncio.create_task(
        client.post("/v1/chat/completions", json=BASE_REQ, headers=AUTH))
    [pending] = await wait_pending(client)
    xid = pending["exchange_id"]
    r = await client.post(f"/api/exchanges/{xid}/inject",
                          json={"kind": "rate_limit", "params": {"retry_after": 7}})
    assert r.status_code == 200
    resp = await post
    assert resp.status_code == 429
    assert resp.headers["retry-after"] == "7"
    err = resp.json()["error"]
    assert err["type"] == "rate_limit_error"
    assert err["code"] == "rate_limit_exceeded"
    detail = (await client.get(f"/api/exchanges/{xid}")).json()
    assert detail["summary"]["status"] == "injected"
    assert "injected:rate_limit" in detail["summary"]["tags"]
    assert detail["injections"][0]["data"]["kind"] == "rate_limit"


async def test_inject_context_exceeded_nonstream(client):
    post = asyncio.create_task(
        client.post("/v1/chat/completions", json=BASE_REQ, headers=AUTH))
    [pending] = await wait_pending(client)
    r = await client.post(f"/api/exchanges/{pending['exchange_id']}/inject",
                          json={"kind": "context_length_exceeded"})
    assert r.status_code == 200
    resp = await post
    assert resp.status_code == 400
    err = resp.json()["error"]
    assert err["code"] == "context_length_exceeded"
    assert err["param"] == "messages"


# ------------------------------------------------------------------ abort
async def test_abort_nonstream_500(client):
    post = asyncio.create_task(
        client.post("/v1/chat/completions", json=BASE_REQ, headers=AUTH))
    [pending] = await wait_pending(client)
    xid = pending["exchange_id"]
    await client.post(f"/api/exchanges/{xid}/draft", json={
        "draft": {"content": "작성 중이던 내용", "tool_calls": []}})
    r = await client.post(f"/api/exchanges/{xid}/abort", json={"kind": "cancel"})
    assert r.status_code == 200
    resp = await post
    assert resp.status_code == 500
    assert resp.json()["error"]["type"] == "server_error"
    events = (await client.get(f"/api/exchanges/{xid}/events")).json()["events"]
    aborted = [e for e in events if e["type"] == "exchange_aborted"]
    assert aborted[0]["data"]["reason"] == "operator_cancel"
    # last draft preserved with the abort (spec §8)
    assert aborted[0]["data"]["draft"]["content"] == "작성 중이던 내용"


async def test_abort_nonstream_drop(client):
    post = asyncio.create_task(
        client.post("/v1/chat/completions", json=BASE_REQ, headers=AUTH))
    [pending] = await wait_pending(client)
    r = await client.post(f"/api/exchanges/{pending['exchange_id']}/abort",
                          json={"kind": "cancel", "style": "drop"})
    assert r.status_code == 200
    with pytest.raises(Exception):
        await post  # connection severed without a complete response


# ----------------------------------------------------------- config modes
async def test_read_only_mode(client):
    r = await client.put("/api/config", json={"read_only": True})
    assert r.status_code == 200
    resp = await client.post("/v1/chat/completions", json=BASE_REQ, headers=AUTH)
    assert resp.status_code == 503
    err = resp.json()["error"]
    assert err["type"] == "server_error" and err["code"] == "read_only_mode"
    await client.put("/api/config", json={"read_only": False})
    # back to normal: request becomes pending again
    post = asyncio.create_task(
        client.post("/v1/chat/completions", json=BASE_REQ, headers=AUTH))
    [pending] = await wait_pending(client)
    await client.post(f"/api/exchanges/{pending['exchange_id']}/abort",
                      json={"kind": "cancel"})
    await post


async def test_fixed_auth_mode(client):
    await client.put("/api/config", json={"auth_mode": "fixed", "api_key": "sek-123"})
    r = await client.post("/v1/chat/completions", json=BASE_REQ,
                          headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"
    post = asyncio.create_task(client.post(
        "/v1/chat/completions", json=BASE_REQ,
        headers={"Authorization": "Bearer sek-123"}))
    [pending] = await wait_pending(client)
    await client.post(f"/api/exchanges/{pending['exchange_id']}/abort",
                      json={"kind": "cancel"})
    await post
    await client.put("/api/config", json={"auth_mode": "any"})


# ------------------------------------------------------------------ drafts
async def test_draft_revisions_and_arguments_obj(client):
    post = asyncio.create_task(
        client.post("/v1/chat/completions", json=BASE_REQ, headers=AUTH))
    [pending] = await wait_pending(client)
    xid = pending["exchange_id"]
    r1 = await client.post(f"/api/exchanges/{xid}/draft", json={
        "draft": {"content": "v1", "tool_calls": [
            {"name": "f", "arguments": '{"a": 1}'}]}})
    r2 = await client.post(f"/api/exchanges/{xid}/draft", json={
        "draft": {"content": "v2", "tool_calls": [
            {"name": "f", "arguments": '{"broken'}]}})
    assert (r1.json()["revision"], r2.json()["revision"]) == (1, 2)
    events = (await client.get(f"/api/exchanges/{xid}/events")).json()["events"]
    drafts = [e["data"] for e in events if e["type"] == "draft_saved"]
    assert drafts[0]["tool_calls"][0]["arguments_obj"] == {"a": 1}
    assert drafts[1]["tool_calls"][0]["arguments_raw"] == '{"broken'
    assert "arguments_obj" not in drafts[1]["tool_calls"][0]
    detail = (await client.get(f"/api/exchanges/{xid}")).json()
    assert detail["draft"]["revision"] == 2
    await client.post(f"/api/exchanges/{xid}/abort", json={"kind": "cancel"})
    await post


# ----------------------------------------------------------- alias toggle
async def test_reasoning_alias_toggle(client):
    await client.put("/api/config", json={"reasoning_alias": True})
    post = asyncio.create_task(
        client.post("/v1/chat/completions", json=BASE_REQ, headers=AUTH))
    [pending] = await wait_pending(client)
    await client.post(f"/api/exchanges/{pending['exchange_id']}/submit", json={
        "draft": {"reasoning_content": "사고", "content": "응답",
                  "finish_reason": "auto", "tool_calls": []}})
    msg = (await post).json()["choices"][0]["message"]
    assert msg["reasoning_content"] == "사고"
    assert msg["reasoning"] == "사고"
    await client.put("/api/config", json={"reasoning_alias": False})
