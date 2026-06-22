import uuid

import pytest

from maycomb.ids import (B62_ALPHABET, b62_decode, b62_encode, chatcmpl_id_for,
                         exchange_uuid_for, new_call_id)


def test_chatcmpl_roundtrip():
    for _ in range(200):
        u = uuid.uuid4()
        cid = chatcmpl_id_for(u)
        assert cid.startswith("chatcmpl-")
        body = cid[len("chatcmpl-"):]
        assert len(body) == 24
        assert all(c in B62_ALPHABET for c in body)
        assert exchange_uuid_for(cid) == u


def test_b62_zero_and_width():
    assert b62_encode(0, 4) == "0000"
    assert b62_decode("0000") == 0
    with pytest.raises(ValueError):
        b62_encode(62 ** 5, 5)
    with pytest.raises(ValueError):
        b62_decode("ab!c")


def test_call_id_format():
    seen = set()
    for _ in range(100):
        cid = new_call_id()
        assert cid.startswith("call_")
        body = cid[len("call_"):]
        assert len(body) == 24
        assert all(c in B62_ALPHABET for c in body)
        seen.add(cid)
    assert len(seen) == 100
