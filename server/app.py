"""
FastAPI application for the RAG Judge Env Environment.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required. Install dependencies with 'uv sync'"
    ) from e

try:
    from ..models import RAGAction, RAGObservation
    from .rag_judge_env_environment import RagJudgeEnvEnvironment
except ImportError:
    from models import RAGAction, RAGObservation
    from server.rag_judge_env_environment import RagJudgeEnvEnvironment


app = create_app(
    RagJudgeEnvEnvironment,
    RAGAction,
    RAGObservation,
    env_name="rag_judge_env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
