import os
import json
import re
from typing import List, Optional
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from server.rag_judge_env_environment import RagJudgeEnvEnvironment
except ImportError:
    from rag_judge_env.server.rag_judge_env_environment import RagJudgeEnvEnvironment

try:
    from models import RAGAction, TaskType
except ImportError:
    from rag_judge_env.models import RAGAction, TaskType

# --- mandatory variables (per submission spec) ---
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")  # optional: only used with from_docker_image()

BENCHMARK = "rag_judge_env"
MAX_STEPS = 8
SUCCESS_SCORE_THRESHOLD = 0.5

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# Each episode covers all 3 task types in a single multi-step run.
# We run one episode per entry so the hackathon evaluator sees 3 START/END blocks.
TASK_TYPES = ["relevance", "hallucination", "full_judgment"]


def extract_json(text: str) -> str:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text


def build_prompt(obs) -> str:
    base = f"""You are a RAG evaluation expert.

Query: {obs.query}

Retrieved Chunks:
{chr(10).join(f'[{i}] {c}' for i, c in zip(obs.chunk_ids, obs.retrieved_chunks))}
"""
    if obs.generated_answer:
        base += f"\nGenerated Answer: {obs.generated_answer}"
    if obs.cited_sources:
        base += f"\nCited chunk IDs: {obs.cited_sources}"

    base += f"\n\nTask Instructions: {obs.instructions}"
    base += "\n\nRespond ONLY with valid JSON matching the RAGAction schema (fields: relevant_chunk_ids, hallucinated_claims, relevance_score, faithfulness_score, citation_accuracy_score, reasoning). No markdown, no explanation."
    return base


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def run_task(task_type: str) -> float:
    env = RagJudgeEnvEnvironment()
    obs = env.reset(task_type=task_type)

    log_start(task=task_type, env=BENCHMARK, model=MODEL_NAME)

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    try:
        for step in range(1, MAX_STEPS + 1):
            error_msg: Optional[str] = None
            action_json = "{}"

            try:
                prompt = build_prompt(obs)
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=500
                )
                raw = response.choices[0].message.content.strip()
                clean = extract_json(raw)
                data = json.loads(clean)
                action = RAGAction(**data)
                action_json = json.dumps(data)
            except Exception as e:
                error_msg = str(e).replace("\n", " ")[:120]
                action = RAGAction(reasoning=f"parse error: {e}")
                action_json = json.dumps({"reasoning": str(e)[:80]})

            obs = env.step(action)
            reward = obs.reward or 0.0
            done = obs.done
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_json, reward=reward, done=done, error=error_msg)

            if done:
                break

        score = round(sum(rewards) / len(rewards), 2) if rewards else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


if __name__ == "__main__":
    for task in TASK_TYPES:
        run_task(task)
