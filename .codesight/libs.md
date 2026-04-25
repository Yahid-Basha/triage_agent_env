# Libraries

- `client.py` — class RagJudgeEnv
- `inference.py`
  - function extract_json: (text) -> str
  - function build_prompt: (obs) -> str
  - function log_start: (task, env, model) -> None
  - function log_step: (step, action, reward, done, error) -> None
  - function log_end: (success, steps, score, rewards) -> None
  - function run_task: (task_type) -> float
- `models.py`
  - class TaskType
  - class RAGAction
  - class RAGObservation
  - class RAGReward
- `server/app.py` — function main: (host, port)
- `server/rag_judge_env_environment.py` — class RagJudgeEnvEnvironment
