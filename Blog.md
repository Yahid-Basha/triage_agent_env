# When my RL agent started writing about Star Wars instead of fixing servers

*A Sunday-morning postmortem on teaching a 3B model to do enterprise IT triage with GRPO.*

It's 1 AM on a Sunday. The Meta × PyTorch OpenEnv Hackathon submission is due at 5 PM. My training logs show a loss curve that's been flat at 0.0 for the last thirty minutes.

A flat loss in supervised learning means convergence. A flat loss in reinforcement learning usually means something else: your model has found a way to game your reward function, and it is now sitting in a local optimum, smug about it.

Mine had decided that the easiest way to maximize reward was to write about cinema history.

## What I was actually trying to build

I work on a production RAG system at Verizon — the kind of system that's supposed to surface the right runbook when a NOC engineer types "BGP session won't come up after the Tuesday maintenance window." This is Theme #3.1 — World Modeling for Professional Tasks — applied to a domain where the model must maintain accurate beliefs about a partially observable KB, not exploit shortcuts to fake a resolution. If you've ever built one of these, you know the failure modes are not in retrieval. Retrieval is mostly a solved problem. The failures are behavioral, at the LLM layer:

The model retrieves the right document and still hallucinates the answer. It cites a KB article that doesn't exist. It sounds confident when it should escalate. It pads the response with filler because shorter answers feel less authoritative. None of these are bugs you can fix with better embeddings.

For the hackathon, I wanted to attack one slice of this problem with reinforcement learning. The thesis: if you give an LLM the right kind of feedback signal — not labeled examples but a programmatic verifier of what good triage looks like — you can teach a small model to behave well over retrieved context, even if it doesn't have deep domain knowledge baked in.

I picked Qwen2.5-3B-Instruct. Not because 3B is fashionable, but because the only result that's interesting to an enterprise team is "small model + RL > big model off the shelf, on this specific task, for this specific cost." If I can't show that, the project doesn't matter.

## The first attempt, and TRL issue #5536

The original design was ambitious: a multi-turn agent that decides which tools to call. The action space had seven tools — `search_kb`, `search_tickets`, `search_incidents`, `get_article`, `get_ticket`, `get_incident`, and `submit_resolution`. The agent would search, fetch, search again, eventually submit a resolution. GRPO would teach it the right query strategies.

I burned about four hours on this before I realized the environment factory in TRL wasn't injecting tool schemas into the rollout context properly. The model was sampling actions blind. There's an open issue (#5536) for it. I tried three workarounds. None of them produced a clean rollout.

Around 4 AM I noticed the loss had flatlined and the rewards looked suspiciously stable. I pulled a few sample completions to inspect them.

The model was writing about Star Wars.

Specifically, it was writing about the history of cinema, the production of *A New Hope*, and at one point, a fairly accurate paragraph about George Lucas's relationship with 20th Century Fox. The `submit_resolution` JSON at the end was empty. The `resolution` string was movie trivia.

Why? Because I had a parsimony reward — "give a concise answer, don't ramble" — that I had implemented incorrectly. The function rewarded outputs *under* 200 characters and *over* 400. (I wanted U-shaped to discourage stub answers and discourage padding, but my logic was inverted in one branch.) Long, irrelevant filler hit the high-length lobe of the reward and got credited as if it were a complete answer.

This is the canonical failure mode of RL. The OpenEnv documentation has a whole section about it. I'd read that section. I still wrote the bug.

## The pivot: single-turn, oracle-grounded

It was 4:30 AM and I had two options. Spend the morning fighting TRL and debugging multi-turn rollouts, or simplify the environment until I had something that trains.

I picked the simplification. The new design dropped multi-turn entirely:

- The prompt contains the ticket plus retrieved articles — some gold, some distractors. I do the retrieval before the rollout.
- The model reads the context and emits exactly one `submit_resolution` action containing four fields: a resolution string, a list of citation IDs, a confidence score, and an escalation boolean.
- The reward grades that single response on six independent dimensions.

The honest framing of this trade-off: I'm no longer training a model to be a good searcher. I'm training it to be a good *reader*. Given a ticket and five candidate articles, can it identify which one is relevant, write a grounded answer, cite the right IDs, and tell me how confident it is? That's a narrower capability than the original goal. It's also exactly the capability that fails in production RAG systems, so it's still worth doing.

## Six rewards, none of them load-bearing alone

I built the reward function as six independent components:

```python
reward_funcs = [
    r_format_graduated,    # JSON schema compliance, no shortcuts
    r_resolution_quality,  # ROUGE-L against gold resolution text
    r_citation_grounding,  # cite only IDs that appear in context
    r_calibration,         # confidence ~ correctness (Brier-style)
    r_parsimony,           # answer shorter than the ticket itself
    r_repetition_penalty,  # punish loops and copy-paste
]
```

The argument for multiple independent rewards is straightforward: with one scalar, your model finds the cheapest way to maximize it, which is almost never what you want. With six, gaming any single component costs you on the others. The Star Wars incident was a reminder of why this matters — that whole behavior was possible because parsimony was the only signal in town for that branch of the reward.

`r_repetition_penalty` is purely defensive. I added it after the Star Wars incident as a guard against the next reward-hacking failure mode I could imagine — a model that finds it can boost grounding by repeating the same valid citation thirty times. It's the only training reward without a server-side equivalent. The environment's evaluation reward function has five components; training has six because I didn't trust the model not to find a new exploit.

## What 200 steps of GRPO actually changed

I trained for 200 steps on an A100. After the run, I pulled the completions parquets from the HF model repo and aggregated the per-step reward means. Two windows: steps 1–10 (cold start, before any meaningful gradient) and steps 150–200 (post-convergence).

| Reward | Steps 1–10 | Steps 150–200 | Δ |
|--------|-----------|---------------|---|
| Calibration | 0.528 | **0.977** | +0.449 |
| Parsimony | 0.246 | **0.940** | +0.694 |
| Citation Grounding | 0.756 | **0.862** | +0.106 |
| Resolution Quality | 0.162 | 0.181 | +0.019 |
| Format Adherence | 1.000 | 0.992 | — |
| Repetition Penalty | 1.000 | 0.990 | — |

Three of these moved a lot. Three didn't.

Format and repetition were already pinned at the ceiling from the first step. Qwen2.5-3B-Instruct is well-aligned out of the box — it knows JSON, and the base model wasn't producing the kind of pathological loops that the repetition penalty was designed to catch. Those rewards weren't doing anything; they were ballast. (The repetition penalty was still cheap insurance, and it's the kind of thing you don't notice working until the day it does.)

Calibration nearly doubled. Parsimony quadrupled. Citation grounding climbed from "okay" to "good." These are the rewards that track behavioral properties — properties of *how* the model speaks, not *what* it knows. The model learned to express less confidence when its answer was wrong (calibration), to stop padding responses (parsimony), and to draw citations from the actual provided context instead of inventing them (grounding).

Then there's the one number that barely moved. `r_resolution_quality` went from 0.16 to 0.18. I want to be honest about that one because it's the most interesting result on the list.
## Reading the reward curves

![GRPO training reward curves — 200 steps, Qwen2.5-3B-Instruct](assets/plots/training_reward_curve.png)
*Six reward signals over 200 GRPO steps. Each line is the mean reward 
across the 8 completions sampled per step.*

One thing that surprises people seeing RL training curves for the first time: 
the loss isn't what you watch. In GRPO, the loss oscillates around zero throughout 
training — that's expected behavior for the `dr_grpo` objective. What you watch 
instead are the reward curves, and you want them to go *up* and stabilize.

The six curves here fall into four groups that each tell a different story.

**Format and repetition (flat at 1.0 from step 1)** — the base model already knew 
the JSON schema and didn't loop citations. GRPO had nothing to teach here. Flat 
lines at the ceiling are a good sign, not stagnation.

**Calibration and parsimony (S-curve convergence, steps 1–30)** — these are the 
headline results. Calibration climbs from 0.53 to 0.97 in roughly thirty steps. 
Parsimony goes from 0.25 to 0.94 over the same window. Both show a classic 
S-curve: slow start, fast middle, then plateau. These are behavioral properties 
the base model had the latent capacity for but hadn't been incentivized to use. 
The gradient found them quickly.

**Citation grounding (noisy upward trend)** — the most visually dramatic curve. 
It swings between 0.0 and 1.0 throughout all 200 steps, and the variance doesn't 
obviously shrink. This is not a training failure — it's a harder reward to 
optimize. Citation grounding depends on each specific ticket's context: sometimes 
the model identifies the right article, sometimes it picks a distractor. The noise 
is honest. The mean trends upward (0.756 → 0.862), which is what matters.

**Resolution quality (the line that doesn't move)** — the orange line stays near 
0.18 for the entire run. This is the most important curve to understand correctly. 
It's not flat because training failed. It's flat because GRPO cannot add knowledge 
the base model doesn't have. Writing the correct BGP reset procedure requires 
knowing the correct BGP reset procedure — that's a pretraining question, not a 
fine-tuning question. The ceiling here tells you where RL ends and domain 
pretraining begins.
## The ceiling that RL can't push through

`r_resolution_quality` is ROUGE-L overlap between the model's resolution text and the gold answer. It measures something like "did you actually write the right fix?"

Calibration is a style property. The model can learn it by adjusting how it expresses uncertainty in language it already produces. Parsimony is also style — say less. Citation grounding is behavioral — pick from this list, don't invent IDs. RL is good at all three.

But generating the *factually correct* fix for a BGP session that won't come up requires knowing the actual answer. RL doesn't add knowledge. It can only shape behavior over what the base model already has.

So `r_resolution_quality` ≈ 0.18 isn't really a training failure. It's the ceiling that GRPO alone hits on a 3B base that doesn't have deep enterprise networking knowledge baked into pretraining. The next steps to break that ceiling are retrieval-augmented generation at inference time, domain-adapted continued pretraining, or both. Neither is RL.

I think this is the most useful thing I can communicate from this project: a clean separation between what RL fine-tuning fixes (style, calibration, format adherence, faithfulness to provided context) and what it doesn't (factual knowledge gaps). If you're a team thinking about deploying a small model behind a RAG system, the first list is achievable in a weekend with a few hundred GRPO steps. The second list is a much bigger investment.

## What this means for production RAG

The pitch isn't that I trained a model that hits some benchmark. It's that the most common LLM-layer failures in production RAG systems — hallucinated citations, miscalibrated confidence, padded answers, format drift — are tractable with RL on a small base model, with a verifier that scores the four behaviors independently.

You don't need to swap your 7B for a 70B to fix these. You need to teach the model you already have to behave correctly over retrieved context.

Calibration improved by 85% relative — from 0.528 to 0.977. In production terms, that's the difference between an agent that confidently misroutes a P1 incident at 3 AM and one that escalates honestly when it doesn't know the answer. That's not a benchmark number. That's a safety property.

## What I'd do differently

A few things I'd change if I were starting over:

The dataset is too small. 50 training tickets across networking, security, cloud, and infra means each subdomain only gets ten or twelve examples. A larger, more diverse ticket pool would probably push the behavioral rewards higher and might lift resolution quality a bit through better domain coverage.

I'd add a process reward, not just outcome rewards. Right now the model is graded only on the final `submit_resolution`. Scoring intermediate reasoning steps — even with a lightweight LLM-as-judge as one signal among many — would give a denser gradient.

I'd run the 72B baseline comparison properly. I tried, on Sunday afternoon, to run the trained 3B and the off-the-shelf 72B side by side against the environment's server-side rewards, hit a CUDA driver mismatch on the eval container, and ran out of time. The training-reward comparison is real; the small-model-with-RL > big-model-without claim is partially evidenced. I'd want a clean head-to-head on equal footing before saying it cleanly.

## Links

- **Environment**: [yahid/triage_agent_env on Hugging Face Spaces](https://huggingface.co/spaces/yahid/triage_agent_env)
- **Trained model**: [yahid/triage-agent-qwen3b](https://huggingface.co/yahid/triage-agent-qwen3b)
- **Training notebook (Colab smoke test)**: *[paste Colab URL here before pushing]*
- **Repository**: same as the HF Space

---

*This was built for the Meta × PyTorch OpenEnv Hackathon, Bangalore finale, April 2026. The environment, training script, and reward functions are open source on the HF Space.*