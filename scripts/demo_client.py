"""Maycomb 데모 하네스 클라이언트.

서버를 띄운 뒤 실행하면 (1) 스트리밍 tool-call 요청을 보내고, 오퍼레이터가
콘솔에서 응답하면 chunk를 출력합니다. (2) tool 결과를 붙여 후속 요청을 보냅니다.

사용: uv run python scripts/demo_client.py [--base http://127.0.0.1:8000]
"""
from __future__ import annotations

import argparse
import json

import httpx

TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "도시의 현재 날씨를 조회한다",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}]


def stream_once(client: httpx.Client, messages: list[dict]) -> dict:
    """스트리밍 요청 1회. 누적 파서로 §3 메시지를 재구성해 반환."""
    body = {
        "model": "mock-large-1",
        "messages": messages,
        "tools": TOOLS,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    reasoning, content = [], []
    tool_calls: dict[int, dict] = {}
    print("→ POST /v1/chat/completions (stream) — 콘솔에서 응답을 작성하세요…")
    with client.stream("POST", "/v1/chat/completions", json=body, timeout=None) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line.startswith("data:"):
                if line.startswith(":"):
                    print("  (keep-alive)")
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                print("\n← [DONE]")
                break
            chunk = json.loads(payload)
            if not chunk.get("choices"):
                print(f"\n← usage: {chunk.get('usage')}")
                continue
            delta = chunk["choices"][0].get("delta") or {}
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
                print(f"\x1b[2m{delta['reasoning_content']}\x1b[0m", end="", flush=True)
            if delta.get("content"):
                content.append(delta["content"])
                print(delta["content"], end="", flush=True)
            for tc in delta.get("tool_calls") or []:
                slot = tool_calls.setdefault(tc.get("index", 0),
                                             {"id": None, "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                    print(f"\n[tool_call {slot['name']}] ", end="", flush=True)
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
                    print(fn["arguments"], end="", flush=True)
            if chunk["choices"][0].get("finish_reason"):
                print(f"\n← finish_reason={chunk['choices'][0]['finish_reason']}")
    return {
        "reasoning_content": "".join(reasoning) or None,
        "content": "".join(content) or None,
        "tool_calls": [
            {"id": s["id"], "type": "function",
             "function": {"name": s["name"], "arguments": s["arguments"]}}
            for _, s in sorted(tool_calls.items())
        ] or None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    args = ap.parse_args()

    client = httpx.Client(base_url=args.base,
                          headers={"Authorization": "Bearer demo-key"})
    print("모델 목록:", [m["id"] for m in client.get("/v1/models").json()["data"]])

    messages: list[dict] = [
        {"role": "system", "content": "You are a helpful weather assistant."},
        {"role": "user", "content": "서울 날씨 어때?"},
    ]
    msg = stream_once(client, messages)

    if msg["tool_calls"]:
        assistant_msg: dict = {"role": "assistant",
                               "content": msg["content"],
                               "tool_calls": msg["tool_calls"]}
        messages.append(assistant_msg)
        for tc in msg["tool_calls"]:
            messages.append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "content": json.dumps({"city": "Seoul", "temp_c": 23,
                                       "condition": "맑음"}, ensure_ascii=False),
            })
        print("\n--- tool 결과를 붙여 후속 요청 ---")
        stream_once(client, messages)


if __name__ == "__main__":
    main()
