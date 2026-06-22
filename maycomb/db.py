"""SQLite storage: exchange index, WAL event log, drafts, raw bodies, settings overrides.

Every WAL event payload carries `v: "wire/1"` (spec preamble + §11).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import WIRE_VERSION

_SCHEMA = """
CREATE TABLE IF NOT EXISTS exchanges (
    exchange_id   TEXT PRIMARY KEY,
    chatcmpl_id   TEXT UNIQUE NOT NULL,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    model         TEXT,
    stream        INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL,
    finish_reason TEXT,
    partial       INTEGER NOT NULL DEFAULT 0,
    tags          TEXT NOT NULL DEFAULT '[]',
    preview       TEXT,
    flags         TEXT NOT NULL DEFAULT '{}',
    raw_path      TEXT,
    raw_sha256    TEXT,
    raw_bytes     INTEGER
);
CREATE INDEX IF NOT EXISTS idx_exchanges_status ON exchanges(status, created_at);

CREATE TABLE IF NOT EXISTS wal_events (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    exchange_id TEXT NOT NULL,
    type        TEXT NOT NULL,
    ts          TEXT NOT NULL,
    v           TEXT NOT NULL,
    data        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_wal_exchange ON wal_events(exchange_id, seq);
CREATE INDEX IF NOT EXISTS idx_wal_type ON wal_events(type, seq);

CREATE TABLE IF NOT EXISTS drafts (
    exchange_id TEXT PRIMARY KEY,
    revision    INTEGER NOT NULL,
    data        TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


class Database:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "maycomb.db"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---------------------------------------------------------------- raw body
    def store_raw(self, exchange_id: str, body: bytes) -> tuple[str, str, int]:
        path = self.raw_dir / f"{exchange_id}.json"
        path.write_bytes(body)
        sha = hashlib.sha256(body).hexdigest()
        return str(path), sha, len(body)

    def read_raw(self, exchange_id: str) -> bytes | None:
        path = self.raw_dir / f"{exchange_id}.json"
        return path.read_bytes() if path.exists() else None

    # ------------------------------------------------------------------ events
    def append_event(self, exchange_id: str, type_: str, data: dict) -> int:
        payload = {"v": WIRE_VERSION, **data}
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO wal_events (exchange_id, type, ts, v, data) VALUES (?,?,?,?,?)",
                (exchange_id, type_, now_iso(), WIRE_VERSION, dumps(payload)),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def events(self, exchange_id: str, types: list[str] | None = None,
               limit: int = 5000) -> list[dict]:
        q = "SELECT seq, exchange_id, type, ts, v, data FROM wal_events WHERE exchange_id=?"
        args: list[Any] = [exchange_id]
        if types:
            q += f" AND type IN ({','.join('?' * len(types))})"
            args += types
        q += " ORDER BY seq LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(q, args).fetchall()
        return [
            {"seq": r["seq"], "type": r["type"], "ts": r["ts"], "v": r["v"],
             "data": json.loads(r["data"])}
            for r in rows
        ]

    def last_event(self, exchange_id: str, types: list[str]) -> dict | None:
        evs = self.events(exchange_id, types)
        return evs[-1] if evs else None

    # --------------------------------------------------------------- exchanges
    def insert_exchange(self, exchange_id: str, chatcmpl_id: str, *, model: str | None,
                        stream: bool, status: str, preview: str | None, flags: dict,
                        raw_path: str, raw_sha256: str, raw_bytes: int) -> None:
        ts = now_iso()
        with self._lock:
            self._conn.execute(
                "INSERT INTO exchanges (exchange_id, chatcmpl_id, created_at, updated_at, model,"
                " stream, status, preview, flags, raw_path, raw_sha256, raw_bytes)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (exchange_id, chatcmpl_id, ts, ts, model, int(stream), status,
                 preview, dumps(flags), raw_path, raw_sha256, raw_bytes),
            )
            self._conn.commit()

    def update_exchange(self, exchange_id: str, **cols: Any) -> None:
        if not cols:
            return
        cols["updated_at"] = now_iso()
        sets, args = [], []
        for k, v in cols.items():
            if k == "tags":
                v = dumps(v)
            elif k == "flags":
                v = dumps(v)
            elif isinstance(v, bool):
                v = int(v)
            sets.append(f"{k}=?")
            args.append(v)
        args.append(exchange_id)
        with self._lock:
            self._conn.execute(f"UPDATE exchanges SET {', '.join(sets)} WHERE exchange_id=?", args)
            self._conn.commit()

    def get_exchange(self, exchange_id: str) -> dict | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT * FROM exchanges WHERE exchange_id=?", (exchange_id,)
            ).fetchone()
        return self._row_to_exchange(r) if r else None

    def list_exchanges(self, status: str | None = None, limit: int = 200) -> list[dict]:
        with self._lock:
            if status:
                rows = self._conn.execute(
                    "SELECT * FROM exchanges WHERE status=? ORDER BY created_at DESC, rowid DESC"
                    " LIMIT ?", (status, limit)).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM exchanges ORDER BY created_at DESC, rowid DESC LIMIT ?",
                    (limit,)).fetchall()
        return [self._row_to_exchange(r) for r in rows]

    @staticmethod
    def _row_to_exchange(r: sqlite3.Row) -> dict:
        d = dict(r)
        d["stream"] = bool(d["stream"])
        d["partial"] = bool(d["partial"])
        d["tags"] = json.loads(d["tags"] or "[]")
        d["flags"] = json.loads(d["flags"] or "{}")
        return d

    def mark_stale_active(self) -> int:
        """On startup: exchanges left pending/active by a previous process are dead."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT exchange_id FROM exchanges WHERE status IN ('pending','active')"
            ).fetchall()
        n = 0
        for r in rows:
            xid = r["exchange_id"]
            self.append_event(xid, "exchange_aborted",
                              {"reason": "server_restart", "draft": self.get_draft(xid)})
            self.update_exchange(xid, status="aborted")
            n += 1
        return n

    # ------------------------------------------------------------------ drafts
    def save_draft(self, exchange_id: str, data: dict) -> int:
        with self._lock:
            r = self._conn.execute(
                "SELECT revision FROM drafts WHERE exchange_id=?", (exchange_id,)
            ).fetchone()
            revision = (r["revision"] + 1) if r else 1
            self._conn.execute(
                "INSERT INTO drafts (exchange_id, revision, data, updated_at) VALUES (?,?,?,?)"
                " ON CONFLICT(exchange_id) DO UPDATE SET revision=excluded.revision,"
                " data=excluded.data, updated_at=excluded.updated_at",
                (exchange_id, revision, dumps(data), now_iso()),
            )
            self._conn.commit()
        return revision

    def get_draft(self, exchange_id: str) -> dict | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT revision, data FROM drafts WHERE exchange_id=?", (exchange_id,)
            ).fetchone()
        if not r:
            return None
        d = json.loads(r["data"])
        d["revision"] = r["revision"]
        return d

    # ---------------------------------------------------------------- settings
    def get_settings_overrides(self) -> dict:
        with self._lock:
            rows = self._conn.execute("SELECT key, value FROM settings").fetchall()
        return {r["key"]: json.loads(r["value"]) for r in rows}

    def set_setting(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO settings (key, value) VALUES (?,?)"
                " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, dumps(value)),
            )
            self._conn.commit()
