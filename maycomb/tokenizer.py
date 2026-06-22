"""Token counting (spec §7) and token-boundary text splitting (spec §4.4/§4.5).

tiktoken is used when available; if the encoding cannot be loaded (e.g. offline,
no cached BPE file) we fall back to a word-level approximation. Counts are
estimates by design — the dataset always keeps the raw text (spec §7 면책).
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

_WORD_RE = re.compile(r"\S+\s*|\s+")

PER_MESSAGE_OVERHEAD = 3   # spec §7: +3 per message
PRIMING_OVERHEAD = 3       # spec §7: +3 reply priming


class TokenCodec:
    def __init__(self, encoding_name: str = "o200k_base"):
        self.requested = encoding_name
        self.encoding_name = encoding_name
        self._enc = None
        if encoding_name != "approx":
            try:
                import tiktoken
                self._enc = tiktoken.get_encoding(encoding_name)
            except Exception:
                self._enc = None
        if self._enc is None:
            self.encoding_name = "approx"

    # ------------------------------------------------------------------ counts
    def count(self, text: str | None) -> int:
        if not text:
            return 0
        if self._enc is not None:
            return len(self._enc.encode(text, disallowed_special=()))
        words = _WORD_RE.findall(text)
        return max(len(words), math.ceil(len(text) / 4))

    # ------------------------------------------------------------------ pieces
    def split_pieces(self, text: str) -> list[str]:
        """Split into ~token-sized, valid-unicode pieces.

        Boundaries follow BPE tokens, i.e. they are arbitrary w.r.t. JSON
        syntax tokens (spec §4.4 requirement for arguments fragments).
        """
        if not text:
            return []
        if self._enc is not None:
            ids = self._enc.encode(text, disallowed_special=())
            pieces: list[str] = []
            buf = b""
            for tid in ids:
                buf += self._enc.decode_single_token_bytes(tid)
                try:
                    pieces.append(buf.decode("utf-8"))
                except UnicodeDecodeError:
                    continue  # multi-byte char split across tokens: keep buffering
                buf = b""
            if buf:
                pieces.append(buf.decode("utf-8", errors="replace"))
            return pieces
        # fallback: word-ish pieces, long runs cut every 4 chars so that even
        # whitespace-free JSON still splits at arbitrary boundaries (spec §4.4)
        pieces = []
        for w in _WORD_RE.findall(text):
            pieces.extend(w[i:i + 4] for i in range(0, len(w), 4))
        return pieces


def _content_to_text(content: Any) -> str:
    """Flatten message content (string or multipart array) for counting."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    parts.append(part["text"])
                else:
                    parts.append(json.dumps(part, ensure_ascii=False))
            else:
                parts.append(str(part))
        return "\n".join(parts)
    return str(content)


def serialize_message_for_count(msg: Any) -> str:
    if not isinstance(msg, dict):
        return str(msg)
    parts = [str(msg.get("role", ""))]
    if msg.get("name"):
        parts.append(str(msg["name"]))
    if msg.get("tool_call_id"):
        parts.append(str(msg["tool_call_id"]))
    text = _content_to_text(msg.get("content"))
    if text:
        parts.append(text)
    if isinstance(msg.get("reasoning_content"), str):
        parts.append(msg["reasoning_content"])
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            fn = tc.get("function") or {}
            parts.append(str(fn.get("name", "")))
            parts.append(str(fn.get("arguments", "")))
    return "\n".join(p for p in parts if p)


def count_prompt_tokens(codec: TokenCodec, messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    total = 0
    for msg in messages:
        try:
            total += PER_MESSAGE_OVERHEAD + codec.count(serialize_message_for_count(msg))
        except Exception:
            total += PER_MESSAGE_OVERHEAD
    return total + PRIMING_OVERHEAD


def build_usage(codec: TokenCodec, messages: Any, *,
                reasoning_content: str | None,
                content: str | None,
                tool_calls: list[dict] | None) -> dict:
    """Usage object per spec §7."""
    prompt = count_prompt_tokens(codec, messages)
    reasoning_tokens = codec.count(reasoning_content)
    completion = reasoning_tokens + codec.count(content)
    for tc in tool_calls or []:
        fn = tc.get("function") or {}
        completion += codec.count(fn.get("name") or "")
        completion += codec.count(fn.get("arguments") or "")
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "prompt_tokens_details": {"cached_tokens": 0},
        "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
    }
