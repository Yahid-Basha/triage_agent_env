---
title: Triage Agent Env Environment Server
emoji: 🎫
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

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
_Caption: primary reward 0.31 → 0.67 over 200 GRPO steps. Efficiency
reward increases as agent learns to stop searching._

![Baseline vs trained](assets/plots/baseline_vs_trained.png)
_Caption: trained agent outperforms Qwen2.5-3B baseline on every rubric
in the eval set._

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
