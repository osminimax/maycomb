"""Wire identifier rules (spec §1).

- response id : "chatcmpl-" + base62(24), bijective with the internal exchange UUID
- tool call id: "call_" + base62(24), random
"""
from __future__ import annotations

import secrets
import uuid

B62_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
_B62_INDEX = {c: i for i, c in enumerate(B62_ALPHABET)}

CHATCMPL_PREFIX = "chatcmpl-"
CALL_PREFIX = "call_"
B62_WIDTH = 24  # 62^24 > 2^128, so any UUID fits


def b62_encode(n: int, width: int = B62_WIDTH) -> str:
    if n < 0:
        raise ValueError("negative value")
    digits: list[str] = []
    while n:
        n, r = divmod(n, 62)
        digits.append(B62_ALPHABET[r])
    s = "".join(reversed(digits)) or "0"
    if len(s) > width:
        raise ValueError("value does not fit in width")
    return s.rjust(width, "0")


def b62_decode(s: str) -> int:
    n = 0
    for c in s:
        try:
            n = n * 62 + _B62_INDEX[c]
        except KeyError:
            raise ValueError(f"invalid base62 char: {c!r}") from None
    return n


def new_exchange_uuid() -> uuid.UUID:
    return uuid.uuid4()


def chatcmpl_id_for(exchange_uuid: uuid.UUID) -> str:
    """1:1 mapping wire id <-> exchange UUID (spec §1 [결정])."""
    return CHATCMPL_PREFIX + b62_encode(exchange_uuid.int)


def exchange_uuid_for(chatcmpl_id: str) -> uuid.UUID:
    body = chatcmpl_id.removeprefix(CHATCMPL_PREFIX)
    return uuid.UUID(int=b62_decode(body))


def new_call_id() -> str:
    return CALL_PREFIX + "".join(secrets.choice(B62_ALPHABET) for _ in range(B62_WIDTH))
