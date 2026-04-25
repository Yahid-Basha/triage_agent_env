import random
import uuid
from pathlib import Path
from typing import Optional

try:
    from openenv.core.env_server.interfaces import Environment
except ImportError as e:
    raise ImportError("openenv-core required. Run: uv sync") from e

try:
    from ..models import TriageAction, TriageObservation, TriageState
    from .corpus import Corpus
    from . import tools
    from . import rewards
except ImportError:
    from models import TriageAction, TriageObservation, TriageState
    from server.corpus import Corpus
    from server import tools
    from server import rewards

MAX_TURNS = 20
TIMEOUT_REWARD = -0.3


class _EpisodeState:
    """Mutable episode state. Plain Python — no Pydantic. Passed to tools/rewards."""

    __slots__ = (
        "episode_id", "step_count",
        "target_ticket_id", "gold_resolution", "gold_cited_ids",
        "difficulty", "is_unanswerable",
        "tools_called", "artifacts_viewed",
        "searches_made", "fetches_made",
        "submitted", "submitted_resolution", "submitted_citations",
        "submitted_confidence", "submitted_escalate",
    )

    def __init__(self):
        self.episode_id: Optional[str] = None
        self.step_count: int = 0
        self.target_ticket_id: str = ""
        self.gold_resolution: str = ""
        self.gold_cited_ids: list = []
        self.difficulty: str = "medium"
        self.is_unanswerable: bool = False
        self.tools_called: list = []
        self.artifacts_viewed: list = []
        self.searches_made: int = 0
        self.fetches_made: int = 0
        self.submitted: bool = False
        self.submitted_resolution: Optional[str] = None
        self.submitted_citations: list = []
        self.submitted_confidence: Optional[float] = None
        self.submitted_escalate: bool = False


class TriageAgentEnvironment(Environment):
    """
    OpenEnv environment for enterprise IT ticket triage.

    The agent receives a new ticket and resolves it by calling up to 7
    MCP-style tools against in-memory KB, ticket, and incident stores.
    Episode terminates on submit_resolution or when MAX_TURNS is exceeded.
    """

    # Each connection gets its own instance — all state is in self._ep and
    # self._current_ticket, so sessions are fully isolated.
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        data_dir = Path(__file__).parent.parent / "data"
        self._corpus = Corpus(data_dir)
        self._ep = _EpisodeState()
        self._current_ticket: dict = {}

    # ------------------------------------------------------------------ #
    # OpenEnv interface                                                    #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> TriageObservation:
        if seed is not None:
            random.seed(seed)

        train_tickets = self._corpus.train_tickets
        if not train_tickets:
            raise ValueError(
                "No training tickets found. "
                "Add data/train_tickets.json (run scripts/generate_corpus.py)."
            )

        ticket = random.choice(train_tickets)

        ep = _EpisodeState()
        ep.episode_id = episode_id or str(uuid.uuid4())[:8]
        ep.target_ticket_id = ticket.get("ticket_id", "")
        ep.gold_resolution = ticket.get("gold_resolution", "")
        ep.gold_cited_ids = list(ticket.get("gold_cited_ids", []))
        ep.difficulty = ticket.get("difficulty", "medium")
        ep.is_unanswerable = ticket.get("is_unanswerable", False)

        self._ep = ep
        self._current_ticket = ticket

        return TriageObservation(
            ticket_id=ep.target_ticket_id,
            ticket_title=ticket.get("title", ""),
            ticket_description=ticket.get("description", ""),
            tool_name="reset",
            tool_result={},
            turn=0,
            max_turns=MAX_TURNS,
            remaining_budget=MAX_TURNS,
            done=False,
            reward=None,
            info={},
        )

    def step(
        self,
        action: TriageAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> TriageObservation:
        ep = self._ep
        ep.step_count += 1
        turn = ep.step_count
        t = self._current_ticket

        # Hard timeout
        if turn > MAX_TURNS:
            return TriageObservation(
                ticket_id=ep.target_ticket_id,
                ticket_title=t.get("title", ""),
                ticket_description=t.get("description", ""),
                tool_name="timeout",
                tool_result={"error": "Maximum turns exceeded"},
                turn=turn,
                max_turns=MAX_TURNS,
                remaining_budget=0,
                done=True,
                reward=TIMEOUT_REWARD,
                info={"reason": "timeout"},
            )

        name = action.tool_name
        done = False
        reward = None
        info: dict = {}

        if name == "search_kb":
            result = tools.search_kb(
                ep, self._corpus,
                query=action.query or "",
                max_results=action.max_results,
            )

        elif name == "get_article":
            result = tools.get_article(ep, self._corpus, article_id=action.article_id or "")

        elif name == "search_tickets":
            result = tools.search_tickets(
                ep, self._corpus,
                query=action.query or "",
                status=action.status,
                max_results=action.max_results,
            )

        elif name == "get_ticket":
            result = tools.get_ticket(ep, self._corpus, ticket_id=action.ticket_id or "")

        elif name == "search_incidents":
            result = tools.search_incidents(
                ep, self._corpus,
                query=action.query or "",
                max_results=action.max_results,
            )

        elif name == "get_incident":
            result = tools.get_incident(ep, self._corpus, incident_id=action.incident_id or "")

        elif name == "submit_resolution":
            result = tools.submit_resolution(
                ep, self._corpus,
                resolution=action.resolution or "",
                cited_artifacts=action.cited_artifacts or [],
                confidence=action.confidence,
                escalate=action.escalate,
            )
            done = True
            reward = rewards.compute_total_reward(ep)
            info = rewards.reward_breakdown(ep)

        else:
            result = {"error": f"Unknown tool '{name}'. Valid tools: search_kb, get_article, "
                               "search_tickets, get_ticket, search_incidents, get_incident, "
                               "submit_resolution"}

        return TriageObservation(
            ticket_id=ep.target_ticket_id,
            ticket_title=t.get("title", ""),
            ticket_description=t.get("description", ""),
            tool_name=name,
            tool_result=result,
            turn=turn,
            max_turns=MAX_TURNS,
            remaining_budget=max(0, MAX_TURNS - turn),
            done=done,
            reward=reward,
            info=info,
        )

    @property
    def state(self) -> TriageState:
        ep = self._ep
        return TriageState(
            episode_id=ep.episode_id,
            step_count=ep.step_count,
            target_ticket_id=ep.target_ticket_id,
            gold_resolution=ep.gold_resolution,
            gold_cited_ids=list(ep.gold_cited_ids),
            difficulty=ep.difficulty,
            is_unanswerable=ep.is_unanswerable,
            tools_called=list(ep.tools_called),
            artifacts_viewed=list(ep.artifacts_viewed),
            searches_made=ep.searches_made,
            fetches_made=ep.fetches_made,
            submitted=ep.submitted,
            submitted_resolution=ep.submitted_resolution,
            submitted_citations=list(ep.submitted_citations),
            submitted_confidence=ep.submitted_confidence,
            submitted_escalate=ep.submitted_escalate,
        )
