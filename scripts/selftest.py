"""실서버 종단간 셀프테스트.

`maycomb serve`가 떠 있는 상태에서 실행하면 하네스 역할(OpenAI 클라이언트)과
오퍼레이터 역할(콘솔 API)을 동시에 수행하며 와이어 동작을 검증합니다.

사용: uv run python scripts/selftest.py [--base http://127.0.0.1:8000]
"""
from __future__ import annotations

import argparse
import json
import threading
import time

import httpx

AUTH = {"Authorization": "Bearer selftest"}


def wait_pending(op: httpx.Client, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        rows = op.get("/api/exchanges", params={"status": "pending"}).json()["exchanges"]
        if rows:
            return rows[0]
        time.sleep(0.05)
    raise TimeoutError("pending exchange did not appear")


def operator_submit(op: httpx.Client, body: dict, delay: float = 0.0) -> dict:
    pending = wait_pending(op)
    if delay:
        time.sleep(delay)
    r = op.post(f"/api/exchanges/{pending['exchange_id']}/submit", json=body)
    r.raise_for_status()
    return pending


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    args = ap.parse_args()
    harness = httpx.Client(base_url=args.base, headers=AUTH, timeout=30.0)
    op = httpx.Client(base_url=args.base, timeout=10.0)
    req = {"model": "smoke-model-x",
           "messages": [{"role": "user", "content": "셀프테스트입니다"}]}

    # 1) models + auth
    assert harness.get("/v1/models").json()["data"][0]["id"] == "mock-large-1"
    assert httpx.get(f"{args.base}/v1/models").status_code == 401
    print("[1] /v1/models + 401 ........................ OK")

    # 2) non-streaming roundtrip
    t = threading.Thread(target=operator_submit, args=(op, {
        "draft": {"reasoning_content": "스모크 사고", "content": "스모크 응답",
                  "finish_reason": "auto", "tool_calls": []}}))
    t.start()
    r = harness.post("/v1/chat/completions", json=req)
    t.join()
    body = r.json()
    assert r.status_code == 200
    assert body["system_fingerprint"] == "fp_mockup_v1"
    assert body["model"] == "smoke-model-x"
    assert body["choices"][0]["message"]["content"] == "스모크 응답"
    assert body["usage"]["completion_tokens_details"]["reasoning_tokens"] > 0
    print("[2] non-streaming roundtrip ................. OK")

    # 3) streaming paced + tool call + include_usage + keep-alive
    sreq = {**req, "stream": True, "stream_options": {"include_usage": True},
            "tools": [{"type": "function",
                       "function": {"name": "get_weather",
                                    "parameters": {"type": "object"}}}]}
    t = threading.Thread(target=operator_submit, args=(op, {
        "draft": {"reasoning_content": "도구를 부르자", "content": "",
                  "finish_reason": "auto",
                  "tool_calls": [{"name": "get_weather",
                                  "arguments": '{"city": "Seoul"}'}]},
        "mode": "paced", "pacing": {"tokens_per_second": 80}}, 1.2))
    t.start()
    saw_keepalive = saw_usage = False
    deltas = []
    timestamps = []
    with harness.stream("POST", "/v1/chat/completions", json=sreq,
                        timeout=None) as resp:
        assert resp.status_code == 200
        for line in resp.iter_lines():
            if line.startswith(":"):
                saw_keepalive = True
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            timestamps.append(time.monotonic())
            if not chunk.get("choices"):
                saw_usage = chunk.get("usage") is not None
                continue
            deltas.append(chunk["choices"][0])
    t.join()
    assert deltas[0]["delta"] == {"role": "assistant"}
    assert deltas[-1]["finish_reason"] == "tool_calls"
    args_text = "".join(
        tc.get("function", {}).get("arguments", "")
        for d in deltas for tc in (d["delta"] or {}).get("tool_calls", []))
    assert json.loads(args_text) == {"city": "Seoul"}
    assert saw_usage, "usage chunk missing"
    spread = timestamps[-1] - timestamps[0]
    assert spread > 0.05, f"pacing too fast to be real ({spread:.3f}s)"
    note = " (+keep-alive)" if saw_keepalive else ""
    print(f"[3] streaming paced/tool/usage{note} ........ OK")

    # 4) hard abort -> protocol-level truncation
    def hard_abort():
        pending = wait_pending(op)
        op.post(f"/api/exchanges/{pending['exchange_id']}/submit", json={
            "draft": {"content": "긴 내용 " * 300, "tool_calls": []},
            "mode": "paced", "pacing": {"tokens_per_second": 20}})
        time.sleep(1.0)
        op.post(f"/api/exchanges/{pending['exchange_id']}/abort",
                json={"kind": "hard"})
    t = threading.Thread(target=hard_abort)
    t.start()
    got_protocol_error = False
    try:
        with harness.stream("POST", "/v1/chat/completions", json={**req, "stream": True},
                            timeout=None) as resp:
            for line in resp.iter_lines():
                if line.startswith("data:") and "[DONE]" in line:
                    raise AssertionError("stream ended cleanly; expected truncation")
    except (httpx.RemoteProtocolError, httpx.ReadError):
        got_protocol_error = True
    t.join()
    assert got_protocol_error, "expected RemoteProtocolError on hard abort"
    print("[4] hard abort = truncated stream ........... OK")

    # 5) 429 injection (non-streaming)
    def inject_429():
        pending = wait_pending(op)
        op.post(f"/api/exchanges/{pending['exchange_id']}/inject",
                json={"kind": "rate_limit", "params": {"retry_after": 5}})
    t = threading.Thread(target=inject_429)
    t.start()
    r = harness.post("/v1/chat/completions", json=req)
    t.join()
    assert r.status_code == 429 and r.headers.get("retry-after") == "5"
    assert r.json()["error"]["type"] == "rate_limit_error"
    print("[5] 429 injection + Retry-After ............. OK")

    print("\nALL SELFTESTS PASSED")


if __name__ == "__main__":
    main()
