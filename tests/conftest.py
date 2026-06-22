from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from maycomb.app import create_app
from maycomb.config import Settings

AUTH = {"Authorization": "Bearer test-key"}


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
def settings(tmp_path):
    return Settings(
        data_dir=str(tmp_path / "data"),
        tokenizer="approx",          # hermetic: no tiktoken download in tests
        keepalive_interval=0.2,
        tokens_per_second=500.0,
        ttft_ms=0,
    )


@pytest.fixture
def app(settings):
    return create_app(settings)


@pytest.fixture
async def client(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://maycomb.test",
                                 timeout=10.0) as c:
        yield c


async def wait_pending(client: httpx.AsyncClient, n: int = 1, timeout: float = 5.0) -> list[dict]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        data = (await client.get("/api/exchanges", params={"status": "pending"})).json()
        if len(data["exchanges"]) >= n:
            return data["exchanges"]
        await asyncio.sleep(0.02)
    raise TimeoutError("no pending exchange appeared")


def sse_data_frames(text: str) -> list[str]:
    """Split an SSE body into data frames (without trailing blank lines)."""
    return [block for block in text.split("\n\n") if block.startswith("data:")]


def sse_chunks(text: str) -> list[dict]:
    import json
    out = []
    for block in sse_data_frames(text):
        payload = block[len("data:"):].strip()
        if payload != "[DONE]":
            out.append(json.loads(payload))
    return out
