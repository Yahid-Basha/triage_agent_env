"""
FastAPI application for TriageAgent environment.

Endpoints:
    POST /reset   — reset the environment
    POST /step    — execute an action
    GET  /state   — get current environment state
    GET  /schema  — action/observation schemas
    WS   /ws      — WebSocket for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

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


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
