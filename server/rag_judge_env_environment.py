import random
from typing import Optional, List

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import RAGAction, RAGObservation, RAGReward, TaskType
except ImportError:
    from models import RAGAction, RAGObservation, RAGReward, TaskType

try:
    from .dataset import TASKS
except ImportError:
    from dataset import TASKS

# Number of evaluation questions per episode.
# Each step presents one question from a rotating task type.
QUESTIONS_PER_EPISODE = 3


class RagJudgeEnvEnvironment(Environment):
    """
    Multi-step RAG pipeline evaluation environment.

    Each episode runs QUESTIONS_PER_EPISODE questions cycling through
    relevance → hallucination → full_judgment (randomly ordered).
    The agent receives a new question after each step; done=True only
    after the final question is answered.
    """

    def __init__(self):
        super().__init__()
        self.steps = 0
        self.max_steps = 8
        self._episode_tasks: List[str] = []   # task types queued for this episode
        self._episode_data: List[dict] = []   # dataset samples queued for this episode
        self._step_rewards: List[float] = []  # reward at each step
        self.current_task_type: Optional[TaskType] = None
        self.current_data: Optional[dict] = None
        self.done: bool = False

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> RAGObservation:
        if seed is not None:
            random.seed(seed)

        self.steps = 0
        self.done = False
        self._step_rewards = []

        # Build the episode queue: one sample per task type, in random order
        task_types = ["relevance", "hallucination", "full_judgment"]
        random.shuffle(task_types)
        self._episode_tasks = task_types[:QUESTIONS_PER_EPISODE]
        self._episode_data = [self._sample_task(t) for t in self._episode_tasks]

        # Load the first question
        self._load_step(0)
        return self._build_observation()

    def step(self, action: RAGAction, timeout_s: Optional[float] = None, **kwargs) -> RAGObservation:
        reward_obj = self._grade(action)
        self._step_rewards.append(reward_obj.score)
        self.steps += 1

        # Advance to next question, or end episode
        if self.steps >= len(self._episode_tasks):
            self.done = True
        else:
            self._load_step(self.steps)

        obs = self._build_observation()
        obs.reward = reward_obj.score
        obs.done = self.done
        obs.metadata["feedback"] = reward_obj.feedback
        obs.metadata["step_rewards"] = list(self._step_rewards)
        obs.metadata["steps_remaining"] = max(0, len(self._episode_tasks) - self.steps)
        if reward_obj.partial_scores:
            obs.metadata["partial_scores"] = reward_obj.partial_scores
        return obs

    @property
    def state(self) -> State:
        return State(
            episode_id=None,
            step_count=self.steps,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sample_task(self, task_type: str) -> dict:
        pool = TASKS[task_type]
        weights = [s.get("weight", 1) for s in pool]
        return random.choices(pool, weights=weights, k=1)[0]

    def _load_step(self, idx: int) -> None:
        self.current_task_type = TaskType(self._episode_tasks[idx])
        self.current_data = self._episode_data[idx]

    def _build_observation(self) -> RAGObservation:
        d = self.current_data
        step_num = self.steps + 1  # 1-indexed for display
        total = len(self._episode_tasks)
        progress = f" [Question {step_num}/{total}]"

        if self.current_task_type == TaskType.RELEVANCE:
            return RAGObservation(
                query=d["query"],
                retrieved_chunks=d["chunks"],
                chunk_ids=list(range(len(d["chunks"]))),
                task_type=self.current_task_type,
                instructions=(
                    "Identify which chunk IDs are relevant to the query. "
                    "Set relevant_chunk_ids in your action." + progress
                )
            )
        elif self.current_task_type == TaskType.HALLUCINATION:
            return RAGObservation(
                query=d["query"],
                retrieved_chunks=[d["context"]],
                chunk_ids=[0],
                generated_answer=d["answer"],
                task_type=self.current_task_type,
                instructions=(
                    "Identify hallucinated claims in the answer not supported by context. "
                    "Set hallucinated_claims in your action." + progress
                )
            )
        else:
            return RAGObservation(
                query=d["query"],
                retrieved_chunks=d["chunks"],
                chunk_ids=list(range(len(d["chunks"]))),
                generated_answer=d["answer"],
                cited_sources=d["cited_ids"],
                task_type=self.current_task_type,
                instructions=(
                    "Score relevance, faithfulness, and citation accuracy between 0.0 and 1.0." + progress
                )
            )

    def _grade(self, action: RAGAction) -> RAGReward:
        d = self.current_data

        if self.current_task_type == TaskType.RELEVANCE:
            predicted = set(action.relevant_chunk_ids or [])
            ground_truth = set(d["relevant_ids"])
            if not ground_truth:
                score = 1.0 if not predicted else 0.0
            else:
                tp = len(predicted & ground_truth)
                fp = len(predicted - ground_truth)
                fn = len(ground_truth - predicted)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                score = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)
            return RAGReward(
                score=round(score, 2),
                feedback=f"F1 score against ground truth relevant chunks: {score:.2f}"
            )

        elif self.current_task_type == TaskType.HALLUCINATION:
            predicted = [c.lower().strip() for c in (action.hallucinated_claims or [])]
            ground_truth = [h.lower().strip() for h in d["hallucinations"]]
            matched = sum(
                1 for gt in ground_truth
                if any(gt in p or p in gt for p in predicted)
            )
            score = matched / len(ground_truth) if ground_truth else 1.0
            over_flag_penalty = max(0, len(predicted) - len(ground_truth)) * 0.1
            score = max(0.0, round(score - over_flag_penalty, 2))
            return RAGReward(
                score=score,
                feedback=f"Detected {matched}/{len(ground_truth)} hallucinations. Penalty: {over_flag_penalty}"
            )

        else:  # full_judgment
            gt = d["ground_truth"]
            scores = {
                "relevance": 1.0 - abs((action.relevance_score or 0) - gt["relevance"]),
                "faithfulness": 1.0 - abs((action.faithfulness_score or 0) - gt["faithfulness"]),
                "citation": 1.0 - abs((action.citation_accuracy_score or 0) - gt["citation_accuracy"])
            }
            scores = {k: max(0.0, round(v, 2)) for k, v in scores.items()}
            final = round((scores["relevance"] * 0.3 +
                          scores["faithfulness"] * 0.4 +
                          scores["citation"] * 0.3), 2)
            return RAGReward(
                score=final,
                feedback="Weighted score: relevance 30%, faithfulness 40%, citation 30%",
                partial_scores=scores
            )
