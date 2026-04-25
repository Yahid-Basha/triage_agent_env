# OpenEnv & RL Environments — The Senior Dev Brief

A ground-up technical briefing so you can hold your own in any OpenEnv, rubric, or RL-training conversation. Read this before touching code. It's dense on purpose.

---

## 1. The "environments are the new data" thesis — what you're actually participating in

The industry has had a quiet paradigm shift over the last 12–18 months. During the pretraining era, progress was data-bound: more tokens, better filtering, more diverse web corpora. That curve flattened. The new bottleneck is **environments**: interactive, verifiable worlds where an LLM can take actions, receive feedback, and learn from rollouts.

Concretely:

- **Prime Intellect's Environments Hub** launched in August 2025 as the first open community registry for RL environments. Within two months they had crowdsourced 400+ environments via bounties and an RL residency, and are now committing six-figure grants to push that toward thousands. Their INTELLECT-3 model (106B MoE) was trained on community-contributed environments across 512 H200 GPUs.
- **Meta's OpenEnv** (your framework) launched in October 2025 as the framework-agnostic standard — a Gymnasium-style spec (`step`, `reset`, `state`) packaged as Docker containers deployable to HF Spaces. Supporters include HuggingFace, Unsloth, vLLM, SkyRL, Oumi, Patronus, and others.
- **NVIDIA NeMo Gym** (launched early 2026) is the third major player, focused on verifiable rollouts that feed into NeMo RL, TRL, and Unsloth.

The "1000 to 100,000 environments" number you heard is real. Prime Intellect's public goal is thousands of open environments. The bet: whoever has the richest environment ecosystem will train the best agents, because *data* alone can't teach an LLM to be a good agent — you need a world it can act in and fail in.

**Why this matters for your pitch:** your hackathon isn't an academic exercise. You're contributing to a live, contested ecosystem. Framing your env as "an environment the open community could actually use to train retrieval agents" is legitimate, not hyperbole. The judges come from this world.

---

## 2. What OpenEnv actually is (at the system-design level)

OpenEnv is a **spec + runtime + CLI** for making RL environments look like HTTP services. The core insight: if every environment speaks the same API (`/reset`, `/step`, `/state` over HTTP/WebSocket, packaged as Docker), then any training framework can consume any environment with zero glue code. This mirrors what OpenAPI did for web services.

### The architecture

```
┌──────────────────────────────────────────┐
│  Client process (your trainer)            │
│  ├─ EnvClient (async or sync)             │
│  └─ Types: Action, Observation, State     │
└───────────────┬──────────────────────────┘
                │ HTTP / WebSocket
                │ POST /reset, POST /step, GET /state
                ▼
┌──────────────────────────────────────────┐
│  Docker container                         │
│  ├─ FastAPI server                        │
│  ├─ Environment class (step/reset/state)  │
│  └─ Your domain logic (RAG, KB, rubrics)  │
└──────────────────────────────────────────┘
```

WebSocket (`/ws`) is used for persistent sessions during training. HTTP endpoints remain for debugging, stateless calls, and interop. There's also a built-in Gradio web UI at `/web` for interactive inspection — great for demo videos.

### The three primitives

Every OpenEnv environment declares three typed dataclasses:

```python
from openenv.core.env_server import Action, Observation, State
from dataclasses import dataclass

@dataclass
class MyAction(Action):
    """What the agent sends."""
    tool_name: str
    arguments: dict

@dataclass
class MyObservation(Observation):
    """What the environment returns after a step."""
    result: str
    done: bool
    reward: float | None = None

@dataclass
class MyState(State):
    """Episode-level metadata the env tracks."""
    episode_id: str | None = None
    step_count: int = 0
    custom_field: int = 0
```

Then three methods on the `Environment` subclass:

- `reset() -> Observation` — start a new episode, return initial observation
- `step(action) -> Observation` — execute action, return next observation (which includes reward and done flag)
- `state -> State` — current episode metadata (as a property, not a method)

That's the whole contract. Everything else is domain logic.

### Two environment flavors

There are two patterns OpenEnv supports, and choosing the right one matters:

**(1) Typed `Environment` (default).** Define a bespoke action class per environment. Used by most envs — textarena, chess, coding_env, wildfire. You write the schema, the agent adheres to it. Good when your action space is fixed and domain-specific.

**(2) `MCPEnvironment` + `MCPToolClient`.** Actions are generic tool calls following the Model Context Protocol pattern: `{tool_name, arguments}`. The environment advertises its available tools via `actions()` method returning `List[ToolDefinition]`. Good when your actions map cleanly onto tool-calling LLMs (Claude, GPT-4), because the action schema *is* the tool schema. Reserved names: `reset`, `step`, `state`, `close` — don't use these as tool names.

**For your EnterpriseRetrievalAgentEnv, I recommend the MCPEnvironment pattern.** Your actions (`search`, `read_doc`, `answer`) are exactly tool calls. Using MCP means a trained LLM can be deployed against the env with zero action-translation glue, which is a real-world selling point and scores on storytelling.

### The CLI and deployment path

```bash
openenv init my_env          # scaffolds the standard structure
openenv validate             # lints the manifest
openenv build                # builds Docker image
openenv serve                # local run with uvicorn
openenv push --repo-id you/my-env  # deploys to HF Space
```

The standard scaffold:

```
my_env/
├── __init__.py
├── client.py              # EnvClient subclass
├── models.py              # Action / Observation / State dataclasses
├── openenv.yaml           # manifest (name, type, runtime, port)
├── pyproject.toml
└── server/
    ├── __init__.py
    ├── app.py             # FastAPI app factory
    ├── my_environment.py  # Environment subclass
    └── Dockerfile
```

Key discipline points:

- `client.py` must NEVER import from `server/`. This is the client/server boundary judges will look for.
- `openenv.yaml` needs `spec_version: 1` and the correct `runtime`.
- Use the dual-import pattern inside `server/my_environment.py`:
  ```python
  try:
      from models import MyAction, MyObservation, MyState  # Docker path
  except ImportError:
      from ..models import MyAction, MyObservation, MyState  # dev path
  ```
- Export all public symbols in the top-level `__init__.py`.

---

## 3. Rubrics — the part most teams get wrong

A rubric is a **structured, natural-language evaluation criterion** that decomposes a fuzzy quality judgment ("is this answer good?") into measurable sub-dimensions. Rubrics are the bridge between subjective tasks (long-form QA, tool use, creative writing) and the objective scalar reward that RL algorithms like GRPO need.

### Why rubrics beat scalar reward models

Pre-2024, the dominant pattern was: train a reward model that emits a single score. This has two problems. First, it's **uninterpretable** — you can't inspect *why* something scored 0.3. Second, it's **easy to game** — the policy learns to satisfy whatever proxy features the RM latched onto, not the underlying quality. This is reward hacking.

Rubrics-as-Rewards (RaR, Gunjal et al. 2025) flipped this. Instead of a black-box RM, you express quality as a set of explicit criteria, each evaluated independently, usually by an LLM judge. The reward is an aggregation over rubrics. This is **interpretable** (you can see which criterion failed) and **harder to game** (gaming one criterion often costs you on another).

### OpenEnv's rubric primitives

OpenEnv ships a rubric system in `openenv.core.rubrics`. The key class is `LLMJudge`:

```python
from openenv.core.rubrics import LLMJudge

judge = LLMJudge(
    prompt_template="Rate this answer for factual accuracy.\n"
                    "Question: {question}\n"
                    "Answer: {answer}\n"
                    "Grounding docs: {docs}\n"
                    "Score 0–1 and explain:",
    client=anthropic_client,  # any LLM client
)
score = await judge(action=answer, observation=obs)
```

You compose multiple `LLMJudge` instances — one per quality dimension — and aggregate. OpenEnv also has `EvalHarness` for batched evaluation and supports delayed rewards (score computed at episode end over the full trajectory, not per-step).

### Designing rubrics that don't suck — the research

This is where the 2025/2026 research has moved fast. Three papers you should know the core ideas of:

**OpenRubrics (Oct 2025)** introduced **Contrastive Rubric Generation (CRG)**: derive rubrics by contrasting chosen vs. rejected responses. The insight is that you can mine rubrics from preference data — for each (preferred, rejected) pair, ask an LLM "what criteria distinguish these?" and get a rubric set that's grounded in actual quality differences rather than made up. Produces both **hard rules** (explicit constraints: "answer must cite at least one source") and **principles** (implicit qualities: "tone should match the query domain").

**RRD — Recursive Rubric Decomposition (Meta Superintelligence Labs, Feb 2026)** addresses the common failure: coarse rubrics that lack coverage, conflate dimensions, or are correlated. RRD decomposes coarse rubrics into fine-grained discriminative criteria, filters misaligned/redundant ones, and applies correlation-aware weighting so you don't double-count. Delivered +17.7 points on JudgeBench and 60–160% reward gains vs. prior rubric baselines.

**OpenRS / PAMR (2026)** — Pairwise Adaptive Meta-Rubrics. The idea: rubrics are more reliable when applied pairwise (is A better than B on criterion X?) than pointwise (score A from 0–1 on criterion X), because pointwise calibration is unstable across the judge LLM.

### Practical rubric composition principles

For your hackathon:

1. **Decompose into orthogonal dimensions.** Each rubric measures one thing. Don't have a "quality" rubric — have separate rubrics for accuracy, citation grounding, efficiency, and appropriate uncertainty.
2. **Mix verifiable and LLM-judged.** Deterministic checks (did the answer cite doc_id X? is the format valid?) are cheap, fast, and ungameable. LLM-judged rubrics catch the fuzzy stuff. A good env uses both.
3. **Weight them.** Declare weights that sum to 1. This forces you to articulate what matters. For retrieval: correctness > grounding > efficiency > uncertainty is a defensible ordering.
4. **Test for gameability.** Before training, manually construct a "cheating" agent that tries to hit one rubric while ignoring others. If it still gets high total reward, your composition is broken.
5. **Prefer binary where possible.** NVIDIA's NeMo Gym guide explicitly recommends binary signals over partial credit for GRPO stability. Use 0/1 verifiable rubrics wherever feasible and reserve graded scores for genuinely continuous dimensions.

---

## 4. Training stack — what actually runs against your env

Your env is the environment. A trainer runs the LLM-as-agent loop against it. The dominant stack for hackathons is **TRL + Unsloth + vLLM**.

### GRPO and its cousins

**GRPO (Group Relative Policy Optimization)** is the algorithm DeepSeek used to train R1 and is the default for LLM RL training on verifiable tasks. The core idea: for each prompt, sample G completions (typically 4–16), score each, compute their mean and std, and use z-score-normalized rewards as the advantage signal. This eliminates the separate critic/value model that PPO needs, saving significant memory.

Key hyperparameters you'll touch:
- `num_generations` — the G in GRPO. 4–8 is standard for hackathon compute. 16 is better but 2× memory.
- `learning_rate` — `5e-6` is the well-tested default. Don't go higher unless you know why.
- `max_prompt_length` / `max_completion_length` — budget carefully.
- `max_steps` — for a hackathon, 100–500 is the realistic range. DeepSeek showed 100 GRPO steps can produce visible behavior change.
- `reward_funcs` — a list of callable reward functions; `reward_weights` weights them.

Variants worth knowing (all supported in TRL ≥1.0 and Unsloth):
- **DAPO** — adds dynamic sampling and clipping improvements; usually more stable.
- **Dr. GRPO** — "doctored" GRPO, fixes a length-bias in standard GRPO's advantage calculation; often higher quality.
- **GSPO, BOLT, Counterfactual GRPO** — newer variants; don't touch these unless you have cycles.

In your GRPOConfig, you can switch variants:
```python
GRPOConfig(
    epsilon=0.2,
    epsilon_high=0.28,  # one-sided clipping (DAPO)
    delta=1.5,          # two-sided clipping
    loss_type='dr_grpo',  # or 'bnpo', 'grpo'
    ...
)
```

### The TRL × OpenEnv integration pattern

TRL added first-class OpenEnv support via `trl.experimental.openenv`. The pattern:

```python
from trl import GRPOTrainer, GRPOConfig
from trl.experimental.openenv import generate_rollout_completions

def custom_rollout(prompts, model, **kwargs):
    # 1. Generate completions with vLLM
    completions = generate_rollout_completions(prompts, model, ...)
    # 2. Step through your OpenEnv env for each completion
    env_rewards = []
    for prompt, completion in zip(prompts, completions):
        obs = env.step(action=parse_action(completion))
        env_rewards.append(obs.reward)
    # 3. Return with env_reward in kwargs
    return {"completions": completions, "env_reward": env_rewards}

def reward_fn(**kwargs):
    return kwargs["env_reward"]  # just pipe through

trainer = GRPOTrainer(
    model=model,
    args=GRPOConfig(...),
    reward_funcs=[reward_fn],
    rollout_func=custom_rollout,  # key hookup
    train_dataset=prompts_dataset,
)
```

The judges who wrote this hackathon *designed for exactly this pattern*. Use it.

### vLLM modes

vLLM is the inference engine TRL uses for fast generation during rollouts. Two modes:

- **Colocate** — vLLM in the same process as training. Requires 1 GPU. Simpler, good for hackathon.
- **Server** — vLLM as a separate process. Requires 2+ GPUs but enables higher throughput.

Pick colocate unless you're given multi-GPU compute.

### Async vs sync training

Prime Intellect's PRIME-RL and Axolotl's GRPO implementation both emphasize **async training**: the policy keeps generating rollouts in a background thread while the trainer updates weights on slightly stale samples. This matters at scale (multi-node, long-horizon). For a 2-day hackathon with a 3B model, you don't need async. Use sync GRPO.

### RULER — when you don't have time to design a reward

OpenPipe's **ART (Agent Reinforcement Trainer)** built on Unsloth introduces **RULER (Relative Universal LLM-Elicited Rewards)** — an LLM judge that scores trajectories relatively ("rank these 8 trajectories") without hand-designed rewards. Cuts reward engineering time 2–3×. I would NOT use this for your hackathon because the innovation score comes from having a thoughtful, decomposed rubric — but it's useful to know it exists as a fallback.

---

## 5. Multi-turn trajectories and the long-horizon problem

Most hackathon envs are single-step: prompt in, answer out, score. The interesting envs are multi-turn — the agent takes multiple actions before the episode ends. Your EnterpriseRetrievalAgentEnv is multi-turn.

### Trajectory anatomy

A trajectory is the full sequence `(obs_0, action_0, obs_1, action_1, ..., obs_T, done)` of an episode. For LLM agents, each action is typically a tool call (generated as JSON inside a completion), and each observation is the tool result fed back into the model's context.

Key challenges:

- **Credit assignment.** If the episode is 15 steps and the final reward is "answer was correct," which action deserves credit? GRPO handles this by assigning the episode-final reward to every step in the trajectory. Crude but works for short horizons.
- **Context blowup.** Each step appends to the context. At 20 steps with verbose docs you're at 20k+ tokens. Budget aggressively.
- **Partial observability.** The agent can't see all docs at once — it must choose what to search and read. This is precisely what makes the env interesting and why it fits Theme #2 / #3.1.

### The RLM paradigm (Recursive Language Models)

Prime Intellect's current research focus: the agent can spawn sub-LLM calls to process chunks of context, returning only the relevant parts. Think "recursive map-reduce on documents." For your env, this would mean the agent's `read_doc` tool might internally chunk the doc and return a summary rather than the raw text. This is optional but a nice sophistication if you have time — it's literally the 2026 frontier.

---

## 6. What makes a winning env — principles distilled

From Prime Intellect's environment-design guide, NVIDIA's NeMo Gym guide, and the hackathon judging brief:

1. **Target a real capability gap.** The best envs exist because current LLMs *can't* do something. For you: agents that know when to stop retrieving. That's a real gap; frontier models retrieve too much or too little.
2. **Profile your baseline.** Run a small base model and a large frontier model against your env. If the frontier model doesn't consistently outscore the base, your rubrics are broken. Do this *before* training.
3. **Prefer binary, verifiable signals where possible.** Save LLM judges for genuinely subjective dimensions.
4. **Mix deterministic and LLM-judged rubrics.** Deterministic: citation correctness, format, efficiency. LLM-judged: answer quality, uncertainty calibration.
5. **Test for reward hacking before training.** Construct a cheater. Does it win?
6. **Connect your training loop to your env, not a static dataset.** Judges explicitly call this out.
7. **Plot baseline vs. trained on the same axes.** PNG. In README. With labels. Not Colab-only.

---

## 7. Current frontier in numbers (as of April 2026)

Useful context for your pitch:

- INTELLECT-3 (Prime Intellect, Nov 2025): 106B MoE trained on 512 H200s with env-based RL, SOTA for its size on math/code/reasoning, outperformed several larger frontier models. Proof that community envs can train real models.
- TRL v1.0 (April 2026): unified SFT/RM/DPO/GRPO stack, first-class OpenEnv integration.
- Unsloth FP8 RL + 380K context RL (Jan–Feb 2026): now possible to train gpt-oss with massive context on consumer-ish hardware. You won't use this but the headline matters: the infra tailwinds are huge.
- RRD rubric refinement (Meta, Feb 2026): rubric quality alone can swing RFT rewards 60–160%.
- Prime Intellect Environments Hub: 400+ envs at launch, scaling to thousands via bounties ($100–$5000 per env).

You're operating inside this whole wave. Say so.

---

## 8. Glossary of terms you should be fluent in

- **Rollout** — one full agent trajectory: reset → steps → done. Rollouts are the unit of data for RL.
- **Advantage** — how much better an action was than average. In GRPO, it's the z-score-normalized reward within a group of G rollouts.
- **On-policy vs off-policy** — on-policy means you train on rollouts from the current model. GRPO is nominally on-policy, but async GRPO is "mildly off-policy" because the weights used to generate have drifted.
- **Reward hacking** — the policy exploits a proxy in the reward function without solving the real task. Your mitigation is rubric composition + adversarial testing.
- **RLVR** — RL with Verifiable Rewards. Training where the reward comes from a verifiable checker (math answer correct, code passes tests) rather than a human preference model. The regime GRPO shines in.
- **RFT** — Reinforcement Fine-Tuning. Umbrella term for GRPO-style RL on a pretrained model.
- **LLM-as-a-judge** — using an LLM to score outputs instead of a trained reward model. Core to rubrics-as-rewards.
- **Context folding** — techniques to compress long histories into fewer tokens. RLM is one approach.
- **Harness** — the scaffolding around a model-under-test: the prompt formatter, tool parser, retry logic, etc. Each env defines a harness.
- **Verifier** — the scoring function that turns a trajectory into a reward. In OpenEnv terms, this is your env's step/reward logic plus rubrics.

---

## 9. Common failure modes to avoid in the next 48 hours

1. **Env reward that's all-or-nothing.** 0 until the very end, then 1. GRPO will flail. Give per-step shaping rewards for progress (a good retrieval is +0.1 even before the final answer).
2. **Rubric that's a single LLM judge score.** Decompose. Always.
3. **Training script that uses a static dataset instead of your env.** Judges will notice. `rollout_func` must call your env.
4. **Dockerfile copies from wrong path.** You already hit this once; don't again. `docker build ... -f server/Dockerfile .` from root.
5. **Client imports from server.** Breaks the abstraction; judges might re-deploy your env and this will fail.
6. **No baseline plot.** Baseline vs trained on same axes is the single most important visual.
7. **Forgetting `num_generations` effect on memory.** 8 generations at 2K completion length on a 3B model is non-trivial. Start at 4.
8. **Over-ambitious action space.** Three actions (search, read, answer) > seven actions you don't have time to test.

---

## 10. The pitch you're building toward

When you present to judges, the narrative arc should be:

1. **Real problem** — "At Verizon, enterprise IT engineers get tickets that require synthesizing 3–5 internal docs. Standard RAG retrieves top-K chunks and calls it done. 60% of those chunks are noise."
2. **The capability gap** — "Current LLMs are bad at *stopping*. They either answer from the first chunk or keep retrieving forever. There's no RL env that trains for retrieval efficiency under partial observability."
3. **What the env teaches** — "EnterpriseRetrievalAgentEnv forces the agent to make explicit search/read/answer decisions. Reward decomposes across correctness, grounding, efficiency, and uncertainty."
4. **What we trained** — "Qwen2.5-3B-Instruct, GRPO, 250 steps, on synthetic enterprise QA from three difficulty tiers."
5. **What changed** — "Baseline 0.31, trained 0.67. Efficiency curve flattens — agent learns to stop at 3 retrievals instead of 8."
6. **Why it matters** — "Every enterprise with internal docs wants this. This env is the first open substrate for training it."

You have the Verizon story. You have the codebase. You have the thesis. The two days ahead are about turning them into exactly this pitch.

---

Read the design doc next. Then come back with questions. I'll have answers on action-space specifics, rubric implementations, dataset shape, TRL wiring, and anything else.
