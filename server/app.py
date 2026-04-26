"""
FastAPI application for TriageAgent environment.

Endpoints:
    POST /reset   — reset the environment
    POST /step    — execute an action
    GET  /state   — get current environment state
    GET  /schema  — action/observation schemas
    WS   /ws      — WebSocket for persistent sessions
    GET  /        — mission control UI (index.html)
    GET  /web     — mission control UI (HF Space base_path)

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

import os

try:
    from openenv.core.env_server.http_server import create_fastapi_app
except Exception as e:
    raise ImportError("openenv-core required. Run: uv sync") from e

try:
    from ..models import TriageAction, TriageObservation
    from .triage_environment import TriageAgentEnvironment
except ImportError:
    from models import TriageAction, TriageObservation
    from server.triage_environment import TriageAgentEnvironment


app = create_fastapi_app(
    TriageAgentEnvironment,
    TriageAction,
    TriageObservation,
    max_concurrent_envs=16,
)

# ── /web prefix rewrite ────────────────────────────────────────────────────────
# On HuggingFace Spaces the app is served under /web, so the frontend sends API
# calls to /web/health, /web/reset, /web/step, etc.  This middleware rewrites
# those paths to their canonical forms before routing, while leaving
# GET /web  (the HTML page) and /web/static/* (assets) untouched.
from starlette.middleware.base import BaseHTTPMiddleware

class _StripWebPrefix(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        path = request.scope["path"]
        if path.startswith("/web/") and not path.startswith("/web/static"):
            stripped = path[4:]          # /web/health → /health
            request.scope["path"] = stripped
            request.scope["raw_path"] = stripped.encode()
        return await call_next(request)

app.add_middleware(_StripWebPrefix)

# ── Static UI ─────────────────────────────────────────────────────────────────
_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
_INDEX_HTML  = os.path.join(_STATIC_DIR, "index.html")

if os.path.isdir(_STATIC_DIR):
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    # Serve static assets at /static/* and /web/static/*
    app.mount("/static",     StaticFiles(directory=_STATIC_DIR), name="static")
    app.mount("/web/static", StaticFiles(directory=_STATIC_DIR), name="web-static")

    @app.get("/")
    async def root():
        return FileResponse(_INDEX_HTML)

    @app.get("/web")
    async def web_root():
        return FileResponse(_INDEX_HTML)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
