"""Settings: TOML file defaults + SQLite-persisted runtime overrides."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

DEFAULT_MODELS = [
    {"id": "mock-large-1", "object": "model", "created": 1780000000, "owned_by": "mockup"},
    {"id": "mock-reasoning-1", "object": "model", "created": 1780000000, "owned_by": "mockup"},
]


@dataclass
class Settings:
    # [server]
    host: str = "127.0.0.1"
    port: int = 8000
    cors: bool = True
    # [auth]
    auth_mode: str = "any"  # any | fixed
    api_key: str = ""
    # [storage]
    data_dir: str = "./data"
    # [models]
    models: list[dict] = field(default_factory=lambda: [dict(m) for m in DEFAULT_MODELS])
    # [tokenizer]
    tokenizer: str = "o200k_base"  # o200k_base | cl100k_base | approx
    # [wire]
    reasoning_alias: bool = False        # also emit `reasoning` next to `reasoning_content`
    keepalive_interval: float = 15.0     # SSE comment interval before first chunk (spec §4.1)
    # [pacing]
    tokens_per_second: float = 30.0
    ttft_ms: int = 0
    stress_interleave: bool = False      # interleaved parallel tool_call fragments (spec §4.4)
    # [modes]
    read_only: bool = False
    nonstream_cancel: str = "error500"   # error500 | drop  (spec §8 last row)


# keys the console may change at runtime; persisted in the settings table
RUNTIME_KEYS = {
    "auth_mode": str,
    "api_key": str,
    "tokenizer": str,
    "reasoning_alias": bool,
    "keepalive_interval": float,
    "tokens_per_second": float,
    "ttft_ms": int,
    "stress_interleave": bool,
    "read_only": bool,
    "nonstream_cancel": str,
}

_SECTION_MAP = {
    ("server", "host"): "host",
    ("server", "port"): "port",
    ("server", "cors"): "cors",
    ("auth", "mode"): "auth_mode",
    ("auth", "api_key"): "api_key",
    ("storage", "data_dir"): "data_dir",
    ("tokenizer", "encoding"): "tokenizer",
    ("wire", "reasoning_alias"): "reasoning_alias",
    ("wire", "keepalive_interval_s"): "keepalive_interval",
    ("pacing", "tokens_per_second"): "tokens_per_second",
    ("pacing", "ttft_ms"): "ttft_ms",
    ("pacing", "stress_interleave"): "stress_interleave",
    ("modes", "read_only"): "read_only",
    ("modes", "nonstream_cancel"): "nonstream_cancel",
}


def load_settings(path: str | Path | None = None) -> Settings:
    s = Settings()
    if path is None:
        candidate = Path("maycomb.toml")
        path = candidate if candidate.exists() else None
    if path is None:
        return s
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    for (section, key), attr in _SECTION_MAP.items():
        if section in raw and key in raw[section]:
            setattr(s, attr, raw[section][key])
    models = raw.get("models", {}).get("list")
    if models:
        s.models = [
            {
                "id": m["id"],
                "object": "model",
                "created": int(m.get("created", 1780000000)),
                "owned_by": m.get("owned_by", "mockup"),
            }
            for m in models
        ]
    return s


def coerce_runtime_value(key: str, value: Any) -> Any:
    typ = RUNTIME_KEYS[key]
    if typ is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)
    return typ(value)


def public_dict(s: Settings) -> dict:
    out = {f.name: getattr(s, f.name) for f in fields(s)}
    if out.get("api_key"):
        out["api_key"] = "***"  # never echo the key back to the console
    return out
