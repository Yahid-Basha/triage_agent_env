# RAG Pipeline Quality Evaluator

An OpenEnv RL environment where an AI agent evaluates the quality of Retrieval-Augmented Generation (RAG) pipelines. The agent receives retrieval results and must judge relevance, detect hallucinations, and score citation accuracy — tasks critical in production AI systems that are difficult to automate reliably.

## Motivation

Enterprise AI systems rely on RAG pipelines to ground LLM responses in retrieved documents. Evaluating pipeline quality — whether retrieved chunks are relevant, whether answers are faithful to sources, whether citations are accurate — is a high-value, real-world task. This environment trains and evaluates agents on exactly these judgment tasks using realistic enterprise IT support scenarios.

## Task Types

| Task | Difficulty | Description |
|---|---|---|
| `relevance` | easy → hard | Given a query and retrieved chunks, identify which chunk IDs are relevant |
| `hallucination` | medium → hard | Given context and a generated answer, identify fabricated claims not supported by the source |
| `full_judgment` | easy → hard | Score relevance, faithfulness, and citation accuracy (0.0–1.0 each) for a full RAG response |

## Action Space (`RAGAction`)

```python
class RAGAction(BaseModel):
    relevant_chunk_ids: Optional[List[int]]    # chunk IDs the agent deems relevant
    hallucinated_claims: Optional[List[str]]   # claims not supported by the context
    relevance_score: Optional[float]           # 0.0–1.0 for full_judgment
    faithfulness_score: Optional[float]        # 0.0–1.0 for full_judgment
    citation_accuracy_score: Optional[float]   # 0.0–1.0 for full_judgment
    reasoning: Optional[str]                   # optional explanation
```

## Observation Space (`RAGObservation`)

```python
class RAGObservation(BaseModel):
    query: str                          # the user query
    retrieved_chunks: List[str]         # candidate chunks from retriever
    chunk_ids: List[int]                # IDs corresponding to each chunk
    generated_answer: Optional[str]     # LLM answer (hallucination/full_judgment only)
    cited_sources: Optional[List[int]]  # chunk IDs the answer claims to cite
    task_type: TaskType                 # RELEVANCE | HALLUCINATION | FULL_JUDGMENT
    instructions: str                   # task-specific instructions for the agent
```

## Reward Function

### Relevance
F1 score between predicted and ground-truth relevant chunk IDs.

### Hallucination
Recall of ground-truth hallucinations detected, with over-flagging penalty (−0.1 per extra claim beyond ground truth count).

### Full Judgment
Weighted average: **relevance 30% + faithfulness 40% + citation accuracy 30%**. Each dimension scored as `max(0, 1 − |predicted − ground_truth|)`.

## Dataset

16 samples across enterprise IT support / RAG systems domain. Hard samples are weighted 2× in random selection so evaluations skew toward challenging cases.

- **6 relevance** (2 easy, 1 medium, 3 hard): password reset, SLA policy, DB access, vendor onboarding, Okta SCIM config with topically-related distractors, REST API v2.3 breaking changes with domain distractors
- **5 hallucination** (2 medium, 3 hard): payment outage, data retention, API autoscaling, embedding cache optimisation (subtle 1% off), DB maintenance window (1-hour shift)
- **5 full_judgment** (1 easy, 2 medium, 2 hard): breach incident, low-confidence retrieval, ticket deduplication, adversarial reranking (correct answer / wrong citations), HNSW embedding index

## Quick Start

```python
from openenv import RAGEnv, RAGAction

env = RAGEnv.from_env("yahid/rag_judge_env")
obs = await env.reset(task_type="relevance")
result = await env.step(RAGAction(relevant_chunk_ids=[0, 2]))
print(result.reward)  # 0.0 – 1.0
```

## Local Setup

```bash
git clone https://huggingface.co/spaces/yahid/rag_judge_env
cd rag_judge_env
uv sync
export HF_TOKEN=your_token
python inference.py
```

## Baseline Scores

Model: `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router

| Task | Score |
|---|---|
| relevance | 0.80 |
| hallucination | 0.67 |
| full_judgment | 1.00 |

## Environment Spec

- `openenv validate`: ✅ passes
- Max steps per episode: 8 (single-turn, done=True after step 1)
- All rewards: [0.0, 1.0]
- Runtime: < 1 minute per task on vcpu=2 / 8GB RAM
