"""FastAPI application assembly."""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import __version__, api_console, api_openai
from .config import RUNTIME_KEYS, Settings, coerce_runtime_value, load_settings
from .db import Database
from .state import Hub
from .tokenizer import TokenCodec

STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or load_settings()
    db = Database(settings.data_dir)

    # runtime overrides persisted by the console win over the TOML file
    for key, value in db.get_settings_overrides().items():
        if key in RUNTIME_KEYS:
            try:
                setattr(settings, key, coerce_runtime_value(key, value))
            except (TypeError, ValueError):
                pass

    hub = Hub(db, settings, TokenCodec(settings.tokenizer))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        stale = db.mark_stale_active()
        if stale:
            print(f"[maycomb] marked {stale} stale exchange(s) as aborted (server_restart)")
        yield
        await hub.shutdown()
        db.close()

    app = FastAPI(title="Maycomb", version=__version__, lifespan=lifespan)
    app.state.hub = hub

    if settings.cors:
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
            allow_headers=["*"])

    app.include_router(api_openai.router)
    app.include_router(api_console.router)

    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def console_index():
        return FileResponse(STATIC_DIR / "console.html")

    return app
