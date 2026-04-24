from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

from openenv.core.env_server.types import Action, Observation


class TaskType(str, Enum):
    RELEVANCE = "relevance"
    HALLUCINATION = "hallucination"
    FULL_JUDGMENT = "full_judgment"


class RAGAction(Action):
    relevant_chunk_ids: Optional[List[int]] = None
    hallucinated_claims: Optional[List[str]] = None
    relevance_score: Optional[float] = None
    faithfulness_score: Optional[float] = None
    citation_accuracy_score: Optional[float] = None
    reasoning: Optional[str] = None


class RAGObservation(Observation):
    # Observation base already provides: done=False, reward=None, metadata={}
    query: str
    retrieved_chunks: List[str]
    chunk_ids: List[int]
    generated_answer: Optional[str] = None
    cited_sources: Optional[List[int]] = None
    task_type: TaskType
    instructions: str


class RAGReward(BaseModel):
    score: float
    feedback: str
    partial_scores: Optional[dict] = None