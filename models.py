from typing import Optional, List, Any, Literal
from pydantic import Field, ConfigDict
from openenv.core.env_server.types import Action, Observation, State

TOOL_NAMES = Literal[
    "search_kb",
    "get_article",
    "search_tickets",
    "get_ticket",
    "search_incidents",
    "get_incident",
    "submit_resolution",
]


class TriageAction(Action):
    tool_name: TOOL_NAMES = "search_kb"
    query: Optional[str] = None
    status: Optional[str] = None
    max_results: int = 5
    ticket_id: Optional[str] = None
    article_id: Optional[str] = None
    incident_id: Optional[str] = None
    resolution: Optional[str] = None
    cited_artifacts: Optional[List[str]] = None
    confidence: Optional[float] = None
    escalate: bool = False


class TriageObservation(Observation):
    ticket_id: str = ""
    ticket_title: str = ""
    ticket_description: str = ""
    tool_name: str = "reset"
    tool_result: dict = Field(default_factory=dict)
    turn: int = 0
    max_turns: int = 20
    remaining_budget: int = 20
    info: dict = Field(default_factory=dict)


class TriageState(State):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Task
    target_ticket_id: str = ""
    gold_resolution: str = ""
    gold_cited_ids: List[str] = Field(default_factory=list)
    difficulty: str = "medium"
    is_unanswerable: bool = False

    # Tracked behavior
    tools_called: List[Any] = Field(default_factory=list)
    artifacts_viewed: List[str] = Field(default_factory=list)
    searches_made: int = 0
    fetches_made: int = 0

    # Terminal data
    submitted: bool = False
    submitted_resolution: Optional[str] = None
    submitted_citations: List[str] = Field(default_factory=list)
    submitted_confidence: Optional[float] = None
    submitted_escalate: bool = False
