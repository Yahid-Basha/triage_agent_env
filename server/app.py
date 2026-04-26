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
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv-core required. Run: uv sync") from e

try:
    from ..models import TriageAction, TriageObservation
    from .triage_environment import TriageAgentEnvironment
except ImportError:
    from models import TriageAction, TriageObservation
    from server.triage_environment import TriageAgentEnvironment


app = create_app(
    TriageAgentEnvironment,
    TriageAction,
    TriageObservation,
    env_name="triage_agent_env",
    max_concurrent_envs=16,
)

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
