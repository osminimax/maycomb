"""CLI: serve / export / verify."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .config import load_settings


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .app import create_app

    settings = load_settings(args.config)
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    if args.data_dir:
        settings.data_dir = args.data_dir
    app = create_app(settings)
    print(f"[maycomb] v{__version__}  console: http://{settings.host}:{settings.port}/  "
          f"api: http://{settings.host}:{settings.port}/v1")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    return 0


def _open_db(args: argparse.Namespace):
    from .db import Database
    settings = load_settings(args.config)
    if args.data_dir:
        settings.data_dir = args.data_dir
    return Database(settings.data_dir)


def _cmd_export(args: argparse.Namespace) -> int:
    db = _open_db(args)
    statuses = (None if args.status == "all" else args.status)
    rows = db.list_exchanges(status=statuses, limit=args.limit)
    out_path = Path(args.out)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in reversed(rows):  # chronological
            if args.exclude_partial and row.get("partial"):
                continue
            xid = row["exchange_id"]
            parsed = db.last_event(xid, ["request_parsed"]) or {}
            result = db.last_event(
                xid, ["response_submitted", "exchange_aborted", "request_rejected"]) or {}
            response = (result.get("data") or {}).get("response")
            if args.strip_reasoning and isinstance(response, dict):
                for choice in response.get("choices") or []:
                    msg = choice.get("message")
                    if isinstance(msg, dict):
                        msg.pop("reasoning_content", None)
                        msg.pop("reasoning", None)
            line = {
                "exchange_id": xid,
                "chatcmpl_id": row["chatcmpl_id"],
                "created_at": row["created_at"],
                "model": row["model"],
                "stream": row["stream"],
                "status": row["status"],
                "finish_reason": row["finish_reason"],
                "partial": row["partial"],
                "tags": row["tags"],
                "request": (parsed.get("data") or {}).get("request"),
                "unknown_fields": (parsed.get("data") or {}).get("_unknown_fields"),
                "result_type": result.get("type"),
                "response": response,
                "usage": (result.get("data") or {}).get("usage"),
                "meta": (result.get("data") or {}).get("meta"),
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            n += 1
    print(f"[maycomb] exported {n} exchange(s) -> {out_path}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    """Dual-representation check (spec §11): chunk_sent replay == stored response."""
    from .wire import reconstruct_from_frames
    db = _open_db(args)
    rows = [r for r in db.list_exchanges(limit=args.limit)
            if r["stream"] and r["status"] == "completed"]
    ok = bad = 0
    for row in rows:
        xid = row["exchange_id"]
        frames = [e["data"]["sse_payload_raw"]
                  for e in db.events(xid, ["chunk_sent"])]
        recon = reconstruct_from_frames(frames)
        stored = ((db.last_event(xid, ["response_submitted"]) or {})
                  .get("data") or {}).get("response")
        if recon == stored:
            ok += 1
        else:
            bad += 1
            print(f"MISMATCH {row['chatcmpl_id']} ({xid})")
            if args.verbose:
                print("  reconstructed:", json.dumps(recon, ensure_ascii=False)[:500])
                print("  stored       :", json.dumps(stored, ensure_ascii=False)[:500])
    print(f"[maycomb] verify: {ok} ok, {bad} mismatch "
          f"(of {len(rows)} completed streaming exchanges)")
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="maycomb", description="Maycomb LLM mockup server")
    parser.add_argument("--version", action="version", version=f"maycomb {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("serve", help="run the mockup server + operator console")
    p.add_argument("--config", default=None, help="path to maycomb.toml")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--data-dir", default=None)
    p.set_defaults(fn=_cmd_serve)

    p = sub.add_parser("export", help="export exchanges to JSONL")
    p.add_argument("--config", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--status", default="completed",
                   help="completed|aborted|injected|rejected|pending|all")
    p.add_argument("--limit", type=int, default=100000)
    p.add_argument("--exclude-partial", action="store_true")
    p.add_argument("--strip-reasoning", action="store_true")
    p.add_argument("--out", default="maycomb_export.jsonl")
    p.set_defaults(fn=_cmd_export)

    p = sub.add_parser("verify", help="check chunk replay == stored response")
    p.add_argument("--config", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--limit", type=int, default=100000)
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=_cmd_verify)

    args = parser.parse_args(argv)
    sys.exit(args.fn(args))
