"""TriageAgent Environment Client."""

from typing import Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import TriageAction, TriageObservation


class TriageAgentEnvClient(EnvClient[TriageAction, TriageObservation, State]):
    """
    Client for the TriageAgent environment.

    Maintains a persistent WebSocket connection to the environment server.
    Each client instance has its own dedicated environment session.

    Example:
        >>> with TriageAgentEnvClient(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.ticket_title)
        ...     action = TriageAction(tool_name="search_kb", query="password reset")
        ...     result = client.step(action)
        ...     print(result.observation.tool_result)
    """

    def _step_payload(self, action: TriageAction) -> Dict:
        return {
            "tool_name": action.tool_name,
            "query": action.query,
            "status": action.status,
            "max_results": action.max_results,
            "ticket_id": action.ticket_id,
            "article_id": action.article_id,
            "incident_id": action.incident_id,
            "resolution": action.resolution,
            "cited_artifacts": action.cited_artifacts,
            "confidence": action.confidence,
            "escalate": action.escalate,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TriageObservation]:
        obs_data = payload.get("observation", {})
        observation = TriageObservation(
            ticket_id=obs_data.get("ticket_id", ""),
            ticket_title=obs_data.get("ticket_title", ""),
            ticket_description=obs_data.get("ticket_description", ""),
            tool_name=obs_data.get("tool_name", ""),
            tool_result=obs_data.get("tool_result", {}),
            turn=obs_data.get("turn", 0),
            max_turns=obs_data.get("max_turns", 20),
            remaining_budget=obs_data.get("remaining_budget", 20),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            info=obs_data.get("info", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
