# TriageAgent — Final Design & Build Doc

**Author:** Royce  
**Hackathon:** Meta PyTorch × OpenEnv Finale, Bangalore, Apr 25–26 2026  
**Theme coverage:** #3.1 World Modeling (Professional) + #2 Long-Horizon Planning  
**Status:** Implementation-ready. Build before arriving onsite. Use Claude Code.

---

## 0. TL;DR

`TriageAgent` is an OpenEnv environment where an LLM agent receives a new enterprise IT ticket and must resolve it by querying mock Jira (past tickets), mock Confluence (KB articles), and mock incident reports through MCP-style tool calls. Success = the submitted resolution matches ground truth AND cites the correct artifacts. The env trains agents to be efficient, grounded, and calibrated under partial observability — a gap even frontier models struggle with.

**Why this wins:**
- Hits two themes cleanly (World Modeling + Long-Horizon)
- Direct Verizon-story fit (your actual day job)
- Tool schemas mirror real Jira/Confluence MCP servers (instantly legible to judges)
- Binary reward (matches TRL team's explicit recommendation) + lightweight shaping
- Reuses ~60% of your existing code; 3–4h of Claude-Code-driven build

---

## 1. The redesign (what changed from the previous doc)

Three research findings shifted the plan:

**(a) Simple rewards beat complex rubrics for GRPO.** TRL docs: *"binary rewards gave cleaner training signals than shaped rewards with partial credit. GRPO compares completions within a group — relative ranking matters more than absolute values."* So instead of four weighted rubrics, we have: one primary binary reward (did it resolve the ticket correctly?) + a few independent small-weight signals for shaping.

**(b) Use `environment_factory`, not `rollout_func`.** TRL v1.0+ added `environment_factory`: you hand it a Python class with tool methods (`reset`, `search_kb`, `get_ticket`, etc.), and it handles the entire multi-turn loop, tool parsing, and generation for you. Massively simpler than writing your own rollout. Requires transformers ≥ 5.2.0.

**(c) Build on the old code.** The existing pyproject/Dockerfile/client.py/openenv.yaml work. The dataset.py snippets become corpus source material. Only `models.py` and the environment file get rewritten. Keep the scaffolding; surgically replace the core.

---

## 2. Theme story (the 30-second pitch)

*"At Verizon, enterprise IT engineers resolve ~5,000 tickets a day. A real ticket lives across multiple artifacts: the ticket itself, comments on related past tickets, KB runbooks, and postmortems from related incidents. Standard RAG retrieves top-K chunks once and fails because it can't decide what to look at next. We built `TriageAgent`: an OpenEnv environment that trains LLMs to act as triage agents — iteratively searching Jira-style tickets and Confluence-style KB articles through MCP tools, synthesizing across artifact types, and committing to grounded resolutions. It's the first open env targeting efficient enterprise triage under partial observability."*

That's your opening. Memorize it.

---

## 3. Environment architecture

### 3.1 The world the agent sees

Three artifact stores, all in-memory JSON:

| Store | Purpose | Size |
|---|---|---|
| **Tickets** | Past closed tickets with comments, resolutions, labels | ~60 items |
| **KB articles** | Runbooks, vendor docs, config guides | ~150 items |
| **Incidents** | Postmortems, root cause analyses | ~25 items |

Each item has a unique ID, title, body, tags, and metadata. Stored on disk as `data/corpus.json`. Loaded once at env startup.

### 3.2 Agent tool space (the MCP-style surface)

Seven tools. Names deliberately mirror Atlassian/Confluence MCP patterns so the story lands:

```
search_tickets(query: str, status?: str, max_results: int = 5)
    → [{ticket_id, title, snippet, status}, ...]
    
get_ticket(ticket_id: str)
    → {ticket_id, title, description, comments, resolution?}
    
search_kb(query: str, max_results: int = 5)
    → [{article_id, title, snippet, section}, ...]

get_article(article_id: str)
    → {article_id, title, body, tags}

search_incidents(query: str, max_results: int = 3)
    → [{incident_id, title, snippet, severity}, ...]

get_incident(incident_id: str)
    → {incident_id, title, summary, root_cause, remediation}

submit_resolution(
    resolution: str,
    cited_artifacts: List[str],   # list of IDs from any store
    confidence: float,            # 0.0-1.0
    escalate: bool = False        # agent admits it can't solve
)
    → terminates episode
```

All non-terminal tools just return data (no side effects, no fake "updates"). `submit_resolution` ends the episode.

**Reserved name check:** none of `reset`, `step`, `state`, `close`. Clean.

### 3.3 Action schema

Single `TriageAction` dataclass with a tool_name discriminator (stays in the typed `Environment` pattern, not `MCPEnvironment` — simpler, reuses existing scaffolding):

```python
@dataclass
class TriageAction(Action):
    tool_name: str  # one of the 7 above
    # Union of params — only the relevant ones are populated
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
```

### 3.4 Observation

```python
@dataclass
class TriageObservation(Observation):
    ticket_id: str           # the ticket being resolved (constant for episode)
    ticket_title: str
    ticket_description: str
    
    tool_name: str           # which tool was just called (or "reset")
    tool_result: dict        # tool output, already JSON-serialized
    
    turn: int
    max_turns: int           # 20
    remaining_budget: int    # max_turns - turn
    
    done: bool
    reward: Optional[float]  # populated on done
    info: dict               # per-subreward breakdown, timing, etc.
```

### 3.5 State

```python
@dataclass
class TriageState(State):
    episode_id: Optional[str] = None
    step_count: int = 0
    
    # Task
    target_ticket_id: str = ""
    gold_resolution: str = ""
    gold_cited_ids: List[str] = field(default_factory=list)
    difficulty: str = ""  # "easy" | "medium" | "hard"
    is_unanswerable: bool = False  # True for some items -> agent should escalate
    
    # Tracked behavior
    tools_called: List[Tuple[str, dict]] = field(default_factory=list)
    artifacts_viewed: List[str] = field(default_factory=list)  # actual full fetches
    searches_made: int = 0
    fetches_made: int = 0
    
    # Terminal data
    submitted: bool = False
    submitted_resolution: Optional[str] = None
    submitted_citations: List[str] = field(default_factory=list)
    submitted_confidence: Optional[float] = None
    submitted_escalate: bool = False
```

### 3.6 Episode flow

```
reset():
    pick a training ticket (tier-stratified sampling)
    populate state.target_ticket_id, gold_resolution, gold_cited_ids
    return initial observation with ticket description
    
step(action):
    step_count += 1
    
    if tool_name in ["search_tickets", "search_kb", "search_incidents"]:
        state.searches_made += 1
        results = run_search(action.query, ...)  # BM25 or semantic
        return obs with results, done=False, reward=None
    
    elif tool_name in ["get_ticket", "get_article", "get_incident"]:
        state.fetches_made += 1
        state.artifacts_viewed.append(id)
        data = store.get(id)
        return obs with data, done=False, reward=None
    
    elif tool_name == "submit_resolution":
        state.submitted = True
        store answer/citations/confidence
        total_reward = compute_reward(state)
        return obs with result, done=True, reward=total_reward, info=breakdown
    
    # Timeout
    if step_count >= 20:
        return obs with done=True, reward=-0.3 (timeout penalty)
```

---

## 4. Reward design (critical — read twice)

**Guiding principle from the TRL team:** binary primary reward + a few simple independent shaping rewards. GRPO ranks completions relative to each other, so a clean, well-ordered total matters more than a finely calibrated scalar.

### 4.1 Primary reward (the binary outcome)

Episode-terminal. Computed only when `submit_resolution` is called.

```python
def primary_reward(state) -> float:
    if state.submitted_escalate and state.is_unanswerable:
        return 1.0  # correctly escalated an unanswerable ticket
    if state.submitted_escalate and not state.is_unanswerable:
        return 0.0  # dodged a ticket it should have resolved
    
    # Normal path: check resolution quality
    correctness = judge_resolution(
        gold=state.gold_resolution,
        submitted=state.submitted_resolution,
    )  # returns 0.0 or 1.0 from LLM judge
    
    grounding = citation_f1(
        gold=state.gold_cited_ids,
        submitted=state.submitted_citations,
        viewed=state.artifacts_viewed,  # cited must be viewed
    )  # returns 0.0 to 1.0
    
    # Primary is binary-ish: both must be reasonably good
    return float(correctness > 0.7 and grounding > 0.5)
```

**Why this shape:** the agent only "wins" if both the content is right AND it cited the right docs. One without the other fails. GRPO sees clear winners and losers.

### 4.2 Shaping rewards (additive, kept small)

Each returns `None` or `float`. Passed to GRPOTrainer as separate reward functions so they get logged independently (judges love seeing the breakdown).

```python
def reward_grounding(state) -> float:
    # Partial credit for citation F1 (even if primary failed)
    return 0.3 * citation_f1(...)

def reward_efficiency(state) -> float:
    # Small bonus for not thrashing
    total_ops = state.searches_made + state.fetches_made
    if state.submitted_escalate:
        return 0.0  # no efficiency bonus for dodging
    ideal = {"easy": 3, "medium": 6, "hard": 10}[state.difficulty]
    return 0.2 * max(0.0, 1.0 - max(0, total_ops - ideal) / ideal)

def reward_calibration(state) -> float:
    # Small Brier-style bonus for calibrated confidence
    c = state.submitted_confidence or 0.5
    correct = primary_reward(state)  # 0 or 1
    return 0.15 * max(0.0, 1.0 - 2 * (c - correct) ** 2)

def reward_format(state) -> float:
    # Tiny penalty for malformed submissions
    if state.submitted and state.submitted_resolution:
        return 0.05
    return 0.0
```

**Total episode reward** = `primary + grounding + efficiency + calibration + format` ≤ ~1.7.

For GRPO the absolute scale doesn't matter — only the within-group ranking. But an agent that gets the primary wrong can still pick up 0.3–0.5 from shaping, so there's always some signal.

### 4.3 Anti-gaming check (run this BEFORE training)

Build three cheater agents and verify none exceeds ~0.35 total:

1. **"Always escalate"** — always calls `submit_resolution(escalate=True)` on first turn. Wins only the unanswerable items, ~20% of the set. Max possible ~0.3.
2. **"Dump everything"** — reads every artifact in the corpus then answers. Citation F1 is near zero because precision is destroyed. Max possible ~0.1.
3. **"Answer immediately"** — calls `submit_resolution` with no searches, empty citations. Primary = 0, grounding = 0. Max possible ~0.05.

If any exceeds 0.35, fix the reward before training. Document the test in the README — judges love it.

### 4.4 The LLM judge for correctness

`judge_resolution()` uses Claude Haiku (fastest, cheapest). Prompt:

```
You are checking if a support agent's resolution matches the gold resolution.

Ticket: {ticket_title}
Gold resolution: {gold}
Agent's resolution: {submitted}

Score 1 if the agent's resolution would resolve the same issue with the same 
core action, even if phrased differently. Score 0 otherwise. Be strict on 
critical parameters (numbers, command names, product names) but lenient on 
phrasing.

Output only a single digit: 0 or 1.
```

**Cache judge outputs** by `(gold, submitted)` hash — training will hit the same pairs many times.

---

## 5. Dataset: what to generate, how, how much

### 5.1 Target size

| Asset | Count | Purpose |
|---|---|---|
| Tickets to resolve (training) | 100 | Drive episodes |
| Tickets to resolve (eval) | 20 | Before/after metrics |
| Past tickets (corpus) | 60 | Searchable history |
| KB articles | 150 | Runbooks + vendor docs |
| Incident postmortems | 25 | Root cause precedents |
| **Unanswerable items** | 15 of 120 | Agent should escalate |

Total corpus: ~235 documents. Non-trivial retrieval space, small enough to embed and keep in memory.

### 5.2 Three difficulty tiers

- **Easy (40 training tickets):** resolvable by reading 1 KB article. Gold citations = 1 item.
- **Medium (40):** requires combining 1 KB article + 1 past ticket OR 1 incident. Gold = 2 items.
- **Hard (20):** requires 3+ artifacts, or has topically-related distractors that mislead shallow searches. Gold = 3+ items.

Plus **~15 "unanswerable" items** sprinkled in — tickets whose context is NOT in the KB. The right move is `escalate=True`. This teaches calibrated abstention and is an innovation talking point.

### 5.3 Generation strategy (1-time script, ~20 min API time)

Write `scripts/generate_corpus.py` that:

1. Seeds 6 enterprise domains: networking, identity/IAM, data platforms, application support, cloud infra, security.
2. For each domain, prompts Claude Haiku to generate 25 KB articles following a schema (title, body 400–800 tokens, tags). Commit these to `data/kb.json`.
3. Generates 10 past tickets per domain with comments and resolutions → `data/past_tickets.json`.
4. Generates 4 incident postmortems per domain → `data/incidents.json`.
5. Generates 20 training tickets per domain with explicit gold citations and gold resolutions. Prompt instructs Haiku to choose 1–3 doc IDs that contain the answer → `data/train_tickets.json`.
6. Generates 3–4 unanswerable tickets where the topic genuinely has no coverage in the corpus → mark with `is_unanswerable: true`.

**Reuse your existing `dataset.py` content.** Those chunks about password reset, SLA, Okta SCIM, auto-scaling, etc. — they're already enterprise-shaped. Spread them across the KB as seed content; Haiku extends with related articles.

Generation prompts are in Appendix A.

### 5.4 Why this is enough

Prime Intellect trained INTELLECT-3 on hundreds of community environments; Wordle GRPO trains on 1,000 synthetic prompts. 120 training episodes × 4 generations × 200 steps = 96,000 trajectories seen. That's plenty of signal for a 3B model. **More data doesn't win — ground-truth clarity does.**

### 5.5 Retrieval implementation

Use `sentence-transformers/all-MiniLM-L6-v2` for embeddings (fast, small, free). Compute once at load, keep in a numpy array. Search = cosine similarity top-K. Falls back to BM25 if sentence-transformers import fails (graceful degradation keeps Docker build robust).

---

## 6. File-by-file change map (use the old code)

Old structure and what happens to each file:

```
/ (root)
├── __init__.py                        KEEP, update exports
├── client.py                          KEEP mostly, rename RagJudge → Triage, fix payload parsing
├── models.py                          REWRITE (new action/obs/state dataclasses)
├── openenv.yaml                       EDIT (rename to triage_agent_env)
├── pyproject.toml                     EDIT (rename, add sentence-transformers, numpy)
├── uv.lock                            REGENERATE via `uv lock`
├── .env                               KEEP (has API keys)
├── inference.py                       EDIT (update action shape; keep [START]/[STEP]/[END] format)
├── README.md                          REWRITE for submission
├── server/
│   ├── __init__.py                    KEEP
│   ├── Dockerfile                     KEEP AS-IS (works)
│   ├── app.py                         EDIT (rename imports, set max_concurrent_envs=16)
│   ├── dataset.py                     DELETE (replaced by corpus loader)
│   └── rag_judge_env_environment.py   DELETE
├── data/                              NEW DIR
│   ├── kb.json                        NEW (generated)
│   ├── past_tickets.json              NEW (generated)
│   ├── incidents.json                 NEW (generated)
│   ├── train_tickets.json             NEW (generated)
│   └── eval_tickets.json              NEW (generated)
├── server/
│   ├── triage_environment.py          NEW (the env core)
│   ├── corpus.py                      NEW (loader + retrieval)
│   ├── rewards.py                     NEW (primary + shaping + judge)
│   └── tools.py                       NEW (the 7 tool implementations)
├── scripts/                           NEW DIR
│   ├── generate_corpus.py             NEW (one-time data gen)
│   ├── validate_rewards.py            NEW (cheater-agent tests)
│   └── baseline_eval.py               NEW (pre-training baseline)
├── training/                          NEW DIR
│   ├── train_grpo.py                  NEW (TRL environment_factory)
│   └── train_grpo.ipynb               NEW (Colab version for submission)
└── assets/                            NEW DIR
    └── plots/                         NEW (commit training curve PNGs here)
```

**Net new Python files:** 7. **Net edited:** 6. **Net deleted:** 2.

---

## 7. Training approach

### 7.1 Model

**Qwen2.5-3B-Instruct.** Native tool-calling. Fits on a single A100 or even T4 with LoRA. GRPO-trainable in a few hundred steps. Fallback to Qwen2.5-1.5B if memory is tight.

### 7.2 The environment_factory wrapper

This is the TRL-native way. You define a plain Python class whose public methods are exposed as tools. The trainer handles everything:

```python
# training/triage_env_factory.py
from typing import List, Optional
import json

class TriageEnvForTraining:
    """In-process wrapper over the OpenEnv env logic. TRL introspects 
    public methods and exposes them as tools to the model."""
    
    def __init__(self, corpus, tickets):
        self.corpus = corpus
        self.tickets = tickets
        self._reset_state()
    
    def _reset_state(self):
        self.state = TriageState()
        self.current_ticket = None
        self.submitted = False
    
    def reset(self, **kwargs) -> str:
        # kwargs contains the row from the dataset
        ticket = self.tickets[kwargs["ticket_id"]]
        self.current_ticket = ticket
        self._reset_state()
        self.state.target_ticket_id = ticket["id"]
        self.state.gold_resolution = ticket["gold_resolution"]
        self.state.gold_cited_ids = ticket["gold_cited_ids"]
        self.state.difficulty = ticket["difficulty"]
        self.state.is_unanswerable = ticket.get("is_unanswerable", False)
        
        # Return a string — TRL appends this to the user message
        return (
            f"# New Ticket {ticket['id']}\n\n"
            f"**Title:** {ticket['title']}\n\n"
            f"**Description:** {ticket['description']}\n\n"
            f"You have 20 tool calls. Use search_tickets, search_kb, "
            f"search_incidents, get_ticket, get_article, get_incident to "
            f"investigate. Submit with submit_resolution."
        )
    
    # Each public method becomes a tool
    def search_kb(self, query: str, max_results: int = 5) -> str:
        """Search the knowledge base for articles matching a query.
        
        Args:
            query: Natural language search query.
            max_results: Maximum number of results to return.
        """
        self.state.searches_made += 1
        results = self.corpus.search_kb(query, max_results)
        return json.dumps(results)
    
    def get_article(self, article_id: str) -> str:
        """Retrieve the full body of a KB article by ID.
        
        Args:
            article_id: The article identifier, e.g. 'KB-00042'.
        """
        self.state.fetches_made += 1
        self.state.artifacts_viewed.append(article_id)
        article = self.corpus.get_article(article_id)
        return json.dumps(article) if article else "Not found."
    
    # ... (5 more tool methods follow the same pattern)
    
    def submit_resolution(
        self, 
        resolution: str, 
        cited_artifacts: List[str],
        confidence: float,
        escalate: bool = False,
    ) -> str:
        """Submit your final resolution. This ends the episode.
        
        Args:
            resolution: The resolution text.
            cited_artifacts: List of artifact IDs supporting the resolution.
            confidence: Your confidence 0.0-1.0.
            escalate: True if this ticket cannot be resolved.
        """
        self.state.submitted = True
        self.state.submitted_resolution = resolution
        self.state.submitted_citations = cited_artifacts
        self.state.submitted_confidence = confidence
        self.state.submitted_escalate = escalate
        return "Submitted. Episode complete."
    
    # Non-tool access: reward function reads env.reward
    @property
    def reward(self) -> float:
        return primary_reward(self.state)
```

Docstrings with `Args:` sections are parsed by TRL into OpenAI-format tool schemas. Every public method you don't want exposed should start with underscore.

### 7.3 The training script (lean, ~60 lines)

```python
# training/train_grpo.py
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
from transformers import AutoTokenizer

from triage_env_factory import TriageEnvForTraining
from corpus import Corpus
from rewards import primary_reward, reward_grounding, reward_efficiency, reward_calibration

# Load corpus + tickets
corpus = Corpus.from_json("data/")
train_tickets = load_tickets("data/train_tickets.json")

# Build dataset — each row triggers one episode
dataset = Dataset.from_list([
    {"prompt": [{"role": "user", "content": "Resolve this ticket."}], 
     "ticket_id": t["id"]}
    for t in train_tickets
])

# Factory creates one env instance per rollout (parallel)
def env_factory():
    return TriageEnvForTraining(corpus, {t["id"]: t for t in train_tickets})

# Reward functions — TRL calls each per completion
def r_primary(environments, **kwargs):
    return [primary_reward(e.state) for e in environments]
def r_grounding(environments, **kwargs):
    return [reward_grounding(e.state) for e in environments]
def r_efficiency(environments, **kwargs):
    return [reward_efficiency(e.state) for e in environments]
def r_calibration(environments, **kwargs):
    return [reward_calibration(e.state) for e in environments]

args = GRPOConfig(
    output_dir="./triage_grpo_qwen3b",
    num_generations=4,                 # G=4 (memory-safe)
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    max_prompt_length=2048,
    max_completion_length=4096,        # critical: covers all turns in the trajectory
    learning_rate=5e-6,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    loss_type="dr_grpo",               # fixes GRPO length bias
    max_steps=200,                     # bump if curves keep climbing
    save_steps=100,
    logging_steps=1,
    bf16=True,
    optim="adamw_8bit",
    use_vllm=True,
    vllm_mode="colocate",              # single-GPU
    vllm_gpu_memory_utilization=0.3,
    report_to="trackio",               # or wandb
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2.5-3B-Instruct",
    reward_funcs=[r_primary, r_grounding, r_efficiency, r_calibration],
    reward_weights=[1.0, 0.3, 0.2, 0.15],
    environment_factory=env_factory,
    train_dataset=dataset,
    args=args,
)

trainer.train()
trainer.save_model()
```

**Sixty lines total.** TRL does the rest.

### 7.4 Why GRPO over PPO and other variants

GRPO was explicitly designed for verifiable-reward tasks (math, code, tool use). It eliminates the value model (saves memory), normalizes rewards within a group of G completions (stable), and is the algorithm DeepSeek used for R1. All modern OpenEnv training examples use it. Don't overthink this.

### 7.5 Expected training signal

Based on Wordle GRPO and similar setups:
- Steps 0–30: flat baseline, model exploring tool use
- Steps 30–100: primary reward climbs as citation behavior improves
- Steps 100–200: efficiency reward climbs (fewer redundant ops)
- Final curve: primary 0.25 → 0.60, total reward 0.45 → 1.0

Even if you only get 0.25 → 0.45 in 100 steps, the *slope* is what wins the 20% "showing improvement" criterion.

---

## 8. Build plan — use Claude Code

Realistic time: **3–4 hours with Claude Code driving**, 6–8 hours solo. Do this today (Apr 24).

### Phase 0 — Prep the workspace (15 min, you drive)

1. Clone the old code to a new folder: `triage_agent_env/`
2. Create a fresh GitHub repo, push the old code as the starting commit
3. `uv sync` to verify the old env still installs
4. Copy `OpenEnv_RL_Senior_Dev_Brief.md` and this doc into the repo under `docs/`

### Phase 1 — Corpus generation (45 min, Claude Code)

**Prompt to Claude Code:**

> Read `docs/TriageAgent_Design.md` sections 5 and Appendix A. Implement `scripts/generate_corpus.py` that uses the Anthropic API (key in `.env` as ANTHROPIC_API_KEY) with model `claude-haiku-4-5-20251001` to generate the synthetic corpus described. Seed the generation with the enterprise snippets in the old `server/dataset.py` — use them as examples of the tone and content I want. Write outputs to `data/kb.json`, `data/past_tickets.json`, `data/incidents.json`, `data/train_tickets.json`, `data/eval_tickets.json`. The script should be idempotent (don't regenerate if files exist unless `--force`). After generation, print counts per tier and per domain. Target sizes are in section 5.1. Sprinkle ~15 unanswerable items across train+eval (mark with `is_unanswerable: true`).

Run it. Verify file sizes look sane (10-50 KB each). **Hand-spot-check 5 items** to make sure the ground truth is correct (gold citations actually contain the answer).

### Phase 2 — Environment core (1 hour, Claude Code)

**Prompt:**

> Read `docs/TriageAgent_Design.md` sections 3, 4, 6. Implement:
> - `models.py`: replace contents with `TriageAction`, `TriageObservation`, `TriageState` per section 3
> - `server/corpus.py`: loads all JSON files from `data/`, exposes `search_kb`, `get_article`, `search_tickets`, `get_ticket`, `search_incidents`, `get_incident`. Use `sentence-transformers/all-MiniLM-L6-v2` for embeddings (compute at startup, cache in memory), cosine similarity for search. Fallback to simple TF-IDF keyword search if sentence-transformers import fails.
> - `server/tools.py`: the 7 tool functions, each takes `(state, corpus, **args)` and returns a dict. Maintain `state.searches_made`, `state.fetches_made`, `state.artifacts_viewed`, `state.tools_called`.
> - `server/rewards.py`: `primary_reward(state)`, `reward_grounding(state)`, `reward_efficiency(state)`, `reward_calibration(state)`, `reward_format(state)`, `judge_resolution(gold, submitted)` using Anthropic API with caching in a `.judge_cache.json` file
> - `server/triage_environment.py`: `TriageAgentEnvironment(Environment)` class with `reset`, `step`, `state` property. In `step`, dispatch on `action.tool_name` to the right handler in `tools.py`. On `submit_resolution`, compute total reward and return with `done=True`.
> - `server/app.py`: update imports, set `max_concurrent_envs=16`, rename env_name to `triage_agent_env`
> - `client.py`: update imports to use new Action/Observation types, update `_step_payload` and `_parse_result` to match the new action fields
> - `__init__.py`: export `TriageAction`, `TriageObservation`, `TriageState`, `TriageAgentEnv`
> - `openenv.yaml`: rename to `triage_agent_env`
> - `pyproject.toml`: rename package, add `sentence-transformers>=2.2.0`, `numpy>=1.24`, `anthropic>=0.40`
> 
> Use the dual try/except import pattern at the top of `triage_environment.py` and `app.py` (try relative imports, fall back to absolute) so it works both in Docker and in the dev tree.
> 
> After finishing, run `uv lock && uv sync` and `python -c "from server.triage_environment import TriageAgentEnvironment; e = TriageAgentEnvironment(); obs = e.reset(); print(obs)"` to smoke-test.

### Phase 3 — Anti-gaming validation (20 min, Claude Code)

**Prompt:**

> Implement `scripts/validate_rewards.py` per section 4.3. Three cheater agents: AlwaysEscalate, DumpEverything, AnswerImmediately. Each runs against all 100 training tickets. Report mean total reward per cheater and per sub-reward. Assert each cheater's mean ≤ 0.35. If any exceeds, print which tickets the cheater beat the threshold on and why.

Run it. If any cheater wins, adjust reward weights and re-run. Iterate until clean.

### Phase 4 — Baseline evaluation (30 min, Claude Code)

**Prompt:**

> Implement `scripts/baseline_eval.py`. It should:
> 1. Load the eval set (`data/eval_tickets.json`, 20 tickets)
> 2. Instantiate `TriageAgentEnvironment`
> 3. For each ticket, run an untrained `Qwen/Qwen2.5-3B-Instruct` via the HuggingFace Inference API (token in `.env` as `HF_TOKEN`), letting it call tools for up to 20 turns
> 4. Capture per-ticket: primary, grounding, efficiency, calibration rewards + trajectory length
> 5. Save results to `assets/baseline_eval.json`
> 6. Generate a summary PNG: bar chart of mean rewards per sub-dimension, save to `assets/plots/baseline_rewards.png`
> Use the `openai` SDK with HF router base_url `https://router.huggingface.co/v1` — same pattern as the existing `inference.py`.

Run it. Expect baseline total mean reward around 0.25–0.40. Commit results.

### Phase 5 — Docker build and HF Space deploy (45 min, you + Claude Code)

**Prompt:**

> Verify the Docker build works: `docker build -t triage_agent_env:latest -f server/Dockerfile .` from the repo root. If it fails, diagnose and fix the Dockerfile or pyproject. Then help me log into `huggingface-cli login` and push to Spaces using `openenv push --repo-id <myname>/triage_agent_env`. Verify the Space comes up green and that the `/web` UI at `https://<myname>-triage_agent_env.hf.space/web` shows the action form.

This is the highest-risk phase because Docker + Spaces have surprising failure modes. **Do this tonight, not onsite.** If the Space is broken at judgment time, you score zero on "minimum requirements."

### Phase 6 — Training notebook (45 min, Claude Code)

**Prompt:**

> Read section 7 of `docs/TriageAgent_Design.md`. Implement:
> - `training/train_grpo.py` — the script from section 7.3, tuned for a single A100. Use `trackio` for logging (free, works in Colab).
> - `training/train_grpo.ipynb` — Colab version of the same script. At the top, include markdown cells explaining the env and the task. After `trainer.train()`, add cells that: load the trainer state, plot reward curves for each sub-reward (matplotlib, saved as PNG with axis labels), save the LoRA adapter, and run eval against `data/eval_tickets.json` and compare to baseline numbers from `assets/baseline_eval.json`.
> 
> Goal: judges should be able to click the notebook in Colab and hit Run All. Include a cell that explains GPU requirements (A100 40GB preferred, will OOM on T4).

Do NOT actually run training on your laptop — the onsite compute credits are for that. Just make sure the notebook runs the first 5 steps without error, which means the wiring is correct.

### Phase 7 — README (30 min, you drive)

Write it yourself. This is your voice. Structure:

```markdown
# TriageAgent

> An OpenEnv environment for training LLMs to resolve enterprise IT tickets 
> through multi-turn, grounded tool use.

## The problem
[One paragraph: the Verizon-style ticket resolution problem. Concrete opener.]

## What the environment does
[The 7 tools. Show one example trajectory in a fenced code block.]

## Reward design
[Primary + 4 shaping. Show a short table with weights.]

## Results
![Training curve](assets/plots/training_curves.png)
*Caption: primary reward 0.31 → 0.67 over 200 GRPO steps. Efficiency 
reward increases as agent learns to stop searching.*

![Baseline vs trained](assets/plots/baseline_vs_trained.png)
*Caption: trained agent outperforms Qwen2.5-3B baseline on every rubric 
in the eval set.*

## Links
- HF Space: https://huggingface.co/spaces/<you>/triage_agent_env
- Training notebook (Colab): <link>
- Code repo: <link>
- Demo video (<2 min): <YouTube link>
- Blog post: <HF post link, optional>

## Quickstart
[Install instructions, 5 lines]

## Architecture diagram
[Simple mermaid or PNG showing: LLM ↔ Tools ↔ (KB, Tickets, Incidents)]

## Anti-gaming tests
We validated that three cheating agents score below 0.35:
- Always escalate: 0.28
- Dump everything: 0.09
- Answer immediately: 0.03
See `scripts/validate_rewards.py`.
```

---

## 9. Onsite plan — mapped to the Scaler timetable

Based on your D-day PDF:

### Day 1 — Saturday Apr 25

**7:00–10:30 (registration, breakfast, ceremony).** Arrive early. Get seat with power. Don't skip breakfast — you have 27 hours ahead.

**10:15–11:00 (Meta talk + problem themes overview).** Don't tune out. They may signal specific things judges care about that aren't in the written brief. Take notes.

**11:30 — Hacking begins.** First 30 minutes:
- Verify HF Space is still live
- `git pull`, confirm training notebook opens cleanly
- Claim your compute credits, link to HF account

**11:30–13:00 (90 min) — Smoke-test training.** Run the notebook for 20 steps, verify curves log, verify no OOM. If it breaks, fix it now while you have energy.

**13:00 — Lunch.** Eat fast. 30 min max.

**13:30–15:30 (2 hours) — First real training run.** Kick off 200-step GRPO. While it runs:
- Start drafting the pitch script
- Watch reward curves live on trackio
- Prepare the video storyboard

**15:30–16:30 — Mentor Round 1.** Walk them through the env + show live training. Ask: "Is my reward shape too gameable? What's missing from the story?" Write down critique verbatim.

**16:30–17:00 — Integrate mentor feedback.** Don't argue. Adjust.

**17:00–17:30 — Talk + tea.** Relax. Watch the curves.

**17:30–20:00 (2.5 hours) — Iteration window.**
- If training isn't learning, tune: lower learning rate (2e-6), increase max_completion_length, check tool call parsing
- Once a run shows clean learning curves, save the trained LoRA adapter
- Start building the comparison plots (baseline vs trained on same axes)

**20:00–22:00 — Mentor Round 2 + dinner.** Show the live comparison. Get feedback on the plots specifically — clarity, axis labels, story.

**22:00–01:00 (3 hours) — Polish window.**
- Final training run (if needed, with best hparams)
- Make all plots production-quality PNGs
- Embed in README
- Lock the repo

**01:00 — Sleep.** You need it.

### Day 2 — Sunday Apr 26

**08:00 — Breakfast.** Coffee.

**08:30–10:00 — Video recording + blog post.**
- Loom or OBS, <2 minute limit (hard — rehearse)
- Script outline:
  - 0:00–0:15 — Hook: "A Verizon engineer gets a ticket..."
  - 0:15–0:45 — The env: show the 7 tools, one live rollout in the `/web` UI
  - 0:45–1:15 — Training: show the curves, baseline vs trained
  - 1:15–1:45 — Why it matters: enterprise triage is a real capability gap, this is the first open env
  - 1:45–2:00 — Call to action: "Try it on HF — space URL"
- Upload to YouTube unlisted
- Write a short HF blog post (optional but strong) — link from README

**10:00–12:00 — Mentor Round 3 (final).** Run your pitch. Get the "what to cut" feedback. Tighten the pitch.

**12:00 — 5-hour submission reminder.** Time check.

**12:00–14:00 — Lunch + final polish.**
- README check: every required link present? check.
- HF Space still live? check.
- Colab notebook runs top to bottom for a judge? check.
- Video URL in README? check.
- Git commit hash locked? check.

**14:00 — 2-hour reminder.** Stop coding. Any bug after this = scratch it, don't commit.

**14:00–15:00 — Dry run the pitch 3 times.** Time yourself. Ruthlessly cut fillers.

**15:00 — SUBMISSION DEADLINE.** Fill out the Google form. Submit:
- HF Space URL
- Colab notebook link
- GitHub repo link
- YouTube / HF blog URL

**15:15+ — Closing remarks, networking.** You made it.

---

## 10. Submission checklist (pin this above your monitor)

Non-negotiable:
- [ ] OpenEnv env installs cleanly (`pip install -e .`)
- [ ] HF Space is live, shows `/web` UI with tool forms
- [ ] Colab notebook runs top-to-bottom on an A100
- [ ] Training plots (PNG) embedded in README with axis labels
- [ ] Baseline vs trained comparison plot
- [ ] YouTube video URL OR HF blog post URL, ≤ 2 min
- [ ] README contains: problem, env description, reward table, results plots, all links, quickstart
- [ ] `scripts/validate_rewards.py` passes (cheater agents < 0.35)
- [ ] No reserved tool names used
- [ ] `openenv.yaml` is valid (spec_version: 1, correct name, runtime, app, port)
- [ ] Client never imports from server/ (check with grep)

Nice to have:
- [ ] Architecture diagram in README
- [ ] HF blog post
- [ ] Wandb/trackio public link

---

## 11. Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Docker build breaks on HF Space | HIGH | Deploy tonight, not onsite |
| Qwen2.5-3B OOMs on provided GPU | MED | Drop to 1.5B; halve `num_generations`; QLoRA |
| Training curve is flat | MED | Increase `max_completion_length` to 8192 first — most likely cause. Then lower LR to 2e-6 |
| LLM judge is slow (rate limits) | MED | Cache aggressively (`.judge_cache.json`). Fall back to string-match heuristic if rate-limited |
| Corpus generation hits Haiku quota | LOW | Use small prompts; batch; generate over dinner tonight |
| Tool call parsing fails | MED | Use `environment_factory` (TRL handles parsing); Qwen2.5 is robust at tool calls |
| Reward function errors crash training | HIGH | All reward funcs wrapped in try/except returning 0.0 on error |
| Submission site overloaded | LOW | Don't wait till 4:59 PM |

---

## 12. Answers to your open questions

**"How do I mimic Jira/KB article fetching MCP?"**  
You don't call real Jira. You define tool schemas with Jira-shaped names (`search_tickets`, `get_ticket`) and back them with in-memory JSON. The judges immediately recognize the pattern. Real Atlassian MCP servers have the same tool names.

**"How important is data volume?"**  
Low. 120 training tickets is enough. Wordle GRPO uses 1,000 synthetic prompts. INTELLECT-3 was trained on hundreds of environments, each with similarly small sets. What wins: (a) clear ground truth, (b) diverse difficulty, (c) realistic content. Your existing `dataset.py` snippets already have realistic content — reuse as seeds.

**"Can I build it in 1–2 hours?"**  
No. Realistic: 6–8 hours solo, 3–4 hours with Claude Code from this doc. That's why the doc is Claude-Code-ready.

**"Old code or scratch?"**  
Old code. Keep scaffolding (pyproject, Dockerfile, openenv.yaml, client.py skeleton), rewrite `models.py` and the env logic, delete the old `dataset.py`. The file-change map in section 6 shows exactly what changes.

**"What if a cheater scores too high?"**  
Adjust the efficiency bonus weight down, tighten the unanswerable-escalation reward to require `confidence < 0.3`, require at least 2 artifact fetches before submit_resolution is accepted (soft gate with warning). Re-run `validate_rewards.py`.

---

## 13. The pitch (memorize this)

> "Hi, I'm Royce. I work on a large-scale enterprise IT ticketing system at Verizon, and the pattern I see every day is that engineers resolve tickets by synthesizing across three kinds of artifacts: past tickets, Confluence runbooks, and incident postmortems. Standard RAG retrieves once and fails because it can't decide what to look at next.
>
> I built TriageAgent — an OpenEnv environment that trains LLMs to be agentic triagers. The agent gets a new ticket and seven MCP-style tools that mirror real Atlassian and Confluence APIs. It has to decide when to search, what to read, and when it has enough evidence. The reward is binary primary — did it resolve the ticket with the right citations — plus three small shaping terms for efficiency, calibration, and grounding.
>
> I trained Qwen2.5-3B with GRPO for 200 steps. The primary reward went from 0.31 to 0.67. Efficiency climbed as the agent learned to stop searching. It also learned to escalate unanswerable tickets — a calibrated abstention behavior that no env I've seen trains for.
>
> Why this matters: every enterprise with internal documentation needs this. Current LLMs over-retrieve on easy tickets and under-retrieve on hard ones. This is the first open environment that targets the decision itself, and it's ready for the community to extend. Thank you."

90 seconds. That's your pitch. Time it.

---

## 14. After hackathon: what to do with this

- Publish the HF Space publicly
- Post the blog on HF (even if you didn't submit the video route)
- LinkedIn post: you already have two variants drafted — use the narrative from your Round 1 experience + this build
- Tweet/X thread with the reward curve and the 90-second pitch
- PR to the OpenEnv repo's environment catalog

Irrespective of whether you win the hackathon, this env becomes a durable artifact on your Verizon track record and public portfolio.

---

# Appendix A — Corpus generation prompts

These go into `scripts/generate_corpus.py`. Each uses Claude Haiku.

## A.1 KB article generation

```
You are generating realistic internal enterprise IT KB articles for a corpus 
that will train an AI agent. Generate ONE article in JSON format.

Domain: {domain}  # one of: networking, identity, data_platforms, app_support, cloud_infra, security
Seed context: {seed_snippet_from_old_dataset}
Topic focus: {topic}  # randomly picked from a domain-specific list

Output schema:
{
  "article_id": "KB-{5-digit hex}",
  "title": "...",  # imperative or descriptive
  "body": "...",   # 400-800 tokens, numbered procedure OR troubleshooting flow
  "tags": [...],   # 3-5 short tags
  "domain": "{domain}",
  "updated": "2025-{MM}-{DD}"
}

Body should:
- Include specific numbers, product names, version strings, config flags
- Reference realistic internal services ("corp-ldap", "prod-api", etc.)
- Sometimes include a "Related" section mentioning other plausible article IDs

Output ONLY the JSON. No preamble, no code fence.
```

## A.2 Past ticket generation

```
Generate one RESOLVED enterprise IT support ticket for corpus.

Schema:
{
  "ticket_id": "TKT-{6 digits}",
  "title": "...",
  "description": "...",     # user's original report, ~100 words
  "comments": [             # 2-4 comments showing investigation
    {"author": "...", "text": "..."},
    ...
  ],
  "resolution": "...",      # what fixed it, 1-3 sentences
  "tags": [...],
  "status": "Resolved",
  "closed_date": "..."
}

Domain: {domain}
Inspiration: related to this KB content: {kb_snippet}
Output JSON only.
```

## A.3 Training ticket generation (with gold citations)

```
Generate a NEW training ticket with explicit ground truth.

You have access to these artifacts (list of {id, title, short_summary}):
{artifact_index}

Steps:
1. Pick 1 to 4 artifacts from the list (depending on difficulty: easy=1, medium=2, hard=3-4) whose content TOGETHER answers a specific question.
2. Invent a realistic ticket whose resolution requires exactly those artifacts.
3. Write the gold resolution in 1-3 sentences using facts from those artifacts.

Output schema:
{
  "ticket_id": "TRAIN-{5 digits}",
  "title": "...",
  "description": "...",         # 50-120 words, ending with a question or problem statement
  "difficulty": "{easy|medium|hard}",
  "gold_cited_ids": [...],      # the artifacts you picked
  "gold_resolution": "...",
  "is_unanswerable": false,
  "domain": "{domain}"
}

Output JSON only.
```

## A.4 Unanswerable ticket generation

```
Generate a training ticket that CANNOT be resolved with any artifact in the 
corpus. The topic should be plausible for the domain but outside the coverage 
of these artifacts:
{artifact_index}

Schema:
{
  "ticket_id": "TRAIN-{5 digits}",
  "title": "...",
  "description": "...",
  "difficulty": "medium",
  "gold_cited_ids": [],
  "gold_resolution": "Escalate to L2 — no runbook exists for this scenario.",
  "is_unanswerable": true,
  "domain": "{domain}"
}
```

## A.5 Incident postmortem generation

```
Generate a realistic incident postmortem.

Schema:
{
  "incident_id": "INC-{4 digits}",
  "title": "...",
  "severity": "{SEV1|SEV2|SEV3}",
  "summary": "...",          # 2-3 sentences
  "root_cause": "...",       # 1-3 sentences
  "remediation": "...",      # what fixed it short-term
  "prevention": "...",       # what changed to prevent recurrence
  "date": "2024-MM-DD",
  "domain": "{domain}"
}
```

---

# Appendix B — Claude Code handoff phrases

When driving Claude Code, these phrases help:

- "Before writing code, summarize the plan and wait for my confirm."
- "Only edit the files in scope; do not touch anything else."
- "If a test fails, show me the failure and stop — do not rewrite."
- "After each file, run `uv run python -c 'import <module>'` to smoke-test."
- "When running shell commands, include `2>&1 | tail -40` so we see errors."

If a phase is longer than 45 min with no progress, stop, review, and redirect.

---

**End of doc. Go build.**
