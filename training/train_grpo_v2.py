#!/usr/bin/env python3
"""
GRPO training for TriageAgent — pragmatic single-call rollout approach.

Why this design:
TRL's environment_factory has a known bug where tool schemas aren't reliably
injected into the model's context (TRL issue #5366), so the model emits plain
text instead of tool calls and tools/call_frequency stays at 0.

This script bypasses that by running the entire agent loop INSIDE the reward
function. The model generates a single completion containing one tool call as
JSON. We parse it, execute it against the env, then score how grounded the
final action is. Over training, the model learns to output tool calls that
score well — which is the GRPO signal we want.

Trade-off: episodes are 1-step instead of multi-turn. We still get genuine
GRPO training of tool-using behavior on the env's actual reward function.

Usage:
    python training/train_grpo.py              # full 200-step run
    python training/train_grpo.py --smoke-test # 5-step verification
"""

import argparse
import json
import os
import re
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from server.corpus import Corpus
from server.rewards import (
    primary_reward,
    reward_calibration,
    reward_efficiency,
    reward_grounding,
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL      = "Qwen/Qwen2.5-3B-Instruct"
DATA_DIR   = ROOT / "data"
OUTPUT_DIR = ROOT / "triage_grpo_qwen3b"
PLOTS_DIR  = ROOT / "assets" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

HF_HUB_REPO_ID = os.environ.get("HF_HUB_REPO_ID", "yahid/triage-agent-qwen3b")


# ── Episode state (same as before) ────────────────────────────────────────────

class _State:
    __slots__ = (
        "target_ticket_id", "gold_resolution", "gold_cited_ids",
        "difficulty", "is_unanswerable",
        "tools_called", "artifacts_viewed",
        "searches_made", "fetches_made",
        "submitted", "submitted_resolution", "submitted_citations",
        "submitted_confidence", "submitted_escalate",
    )

    def __init__(self):
        self.target_ticket_id = ""
        self.gold_resolution = ""
        self.gold_cited_ids = []
        self.difficulty = "medium"
        self.is_unanswerable = False
        self.tools_called = []
        self.artifacts_viewed = []
        self.searches_made = 0
        self.fetches_made = 0
        self.submitted = False
        self.submitted_resolution = None
        self.submitted_citations = []
        self.submitted_confidence = None
        self.submitted_escalate = False


# ── In-memory corpus singleton (loaded once) ──────────────────────────────────

_CORPUS: Optional[Corpus] = None
_TICKET_INDEX: Dict[str, dict] = {}


def get_corpus() -> Corpus:
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = Corpus(DATA_DIR)
    return _CORPUS


def get_ticket_index() -> Dict[str, dict]:
    global _TICKET_INDEX
    if not _TICKET_INDEX:
        with open(DATA_DIR / "train_tickets.json") as f:
            tickets = json.load(f)
        _TICKET_INDEX = {t.get("ticket_id", t.get("id", "")): t for t in tickets}
    return _TICKET_INDEX


# ── Tool execution ────────────────────────────────────────────────────────────

def _execute_tool(state: _State, tool_name: str, args: dict) -> dict:
    """Execute one tool call against the corpus, mutate state in place."""
    corpus = get_corpus()
    state.tools_called.append((tool_name, args))

    if tool_name == "search_kb":
        state.searches_made += 1
        return {"results": corpus.search_kb(
            query=args.get("query", ""),
            max_results=args.get("max_results", 5)
        )}

    if tool_name == "search_tickets":
        state.searches_made += 1
        return {"results": corpus.search_tickets(
            query=args.get("query", ""),
            status=args.get("status"),
            max_results=args.get("max_results", 5)
        )}

    if tool_name == "search_incidents":
        state.searches_made += 1
        return {"results": corpus.search_incidents(
            query=args.get("query", ""),
            max_results=args.get("max_results", 3)
        )}

    if tool_name == "get_article":
        state.fetches_made += 1
        aid = args.get("article_id", "")
        state.artifacts_viewed.append(aid)
        result = corpus.get_article(article_id=aid)
        return result if result else {"error": "Not found"}

    if tool_name == "get_ticket":
        state.fetches_made += 1
        tid = args.get("ticket_id", "")
        state.artifacts_viewed.append(tid)
        result = corpus.get_ticket(ticket_id=tid)
        return result if result else {"error": "Not found"}

    if tool_name == "get_incident":
        state.fetches_made += 1
        iid = args.get("incident_id", "")
        state.artifacts_viewed.append(iid)
        result = corpus.get_incident(incident_id=iid)
        return result if result else {"error": "Not found"}

    if tool_name == "submit_resolution":
        state.submitted = True
        state.submitted_resolution = args.get("resolution", "")
        state.submitted_citations  = list(args.get("cited_artifacts", []))
        state.submitted_confidence = float(args.get("confidence", 0.5))
        state.submitted_escalate   = bool(args.get("escalate", False))
        return {"accepted": True}

    return {"error": f"Unknown tool: {tool_name}"}


# ── Tool-call extraction from model output ────────────────────────────────────

TOOL_CALL_REGEXES = [
    # ```json {...} ``` block
    re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL),
    # <tool_call>{...}</tool_call>
    re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL),
    # First standalone JSON object that mentions a known tool
    re.compile(r'(\{[^{}]*"(?:tool_name|name|function)"[^{}]*\})', re.DOTALL),
]

KNOWN_TOOLS = {
    "search_kb", "search_tickets", "search_incidents",
    "get_article", "get_ticket", "get_incident",
    "submit_resolution",
}


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract a tool call from model output. Returns None if unparseable."""
    for rx in TOOL_CALL_REGEXES:
        for match in rx.finditer(text):
            try:
                obj = json.loads(match.group(1))
            except Exception:
                continue
            # Normalize: accept tool_name, name, or function.name
            name = obj.get("tool_name") or obj.get("name")
            if not name and "function" in obj:
                name = obj["function"].get("name") if isinstance(obj["function"], dict) else None
            if name not in KNOWN_TOOLS:
                continue
            args = obj.get("arguments") or obj.get("parameters") or {k: v for k, v in obj.items()
                                                                       if k not in ("tool_name", "name", "function")}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            return {"tool_name": name, "arguments": args}
    return None


# ── Episode runner — runs in reward function ──────────────────────────────────

# Cache: maps ticket_id → cached agent runs to avoid re-running across reward fns
_EPISODE_CACHE: Dict[tuple, _State] = {}


def run_episode_from_completion(completion: str, ticket_id: str) -> _State:
    """Run the agent loop for one completion + ticket, return final state.

    The 'completion' is one model output. We extract the tool call, execute it,
    and that's the episode. Single-turn but it teaches the model to emit
    well-grounded tool calls.

    For multi-turn-like behavior, the model is taught via the prompt to issue
    a 'submit_resolution' call in the same completion that includes references
    to artifact IDs (which the model can recall from its earlier context).
    """
    cache_key = (ticket_id, hash(completion))
    if cache_key in _EPISODE_CACHE:
        return _EPISODE_CACHE[cache_key]

    ticket_index = get_ticket_index()
    ticket = ticket_index.get(ticket_id) or next(iter(ticket_index.values()))

    state = _State()
    state.target_ticket_id = ticket.get("ticket_id", ticket.get("id", ""))
    state.gold_resolution  = ticket.get("gold_resolution", "")
    state.gold_cited_ids   = list(ticket.get("gold_cited_ids", []))
    state.difficulty       = ticket.get("difficulty", "medium")
    state.is_unanswerable  = ticket.get("is_unanswerable", False)

    # Try to parse and execute the tool call in the completion
    parsed = parse_tool_call(completion)
    if parsed is not None:
        try:
            _execute_tool(state, parsed["tool_name"], parsed["arguments"])
        except Exception:
            pass  # bad args → episode fails gracefully, low reward

    # If model didn't submit a resolution, force a no-op submission so rewards
    # can be computed (otherwise primary always returns 0 trivially)
    if not state.submitted:
        state.submitted = True
        state.submitted_resolution = ""  # empty → low correctness score
        state.submitted_citations = []
        state.submitted_confidence = 0.5
        state.submitted_escalate = False

    _EPISODE_CACHE[cache_key] = state
    if len(_EPISODE_CACHE) > 4096:
        _EPISODE_CACHE.clear()
    return state


# ── Reward functions — TRL signature ──────────────────────────────────────────

def r_primary(prompts, completions, ticket_id, **kwargs):
    out = []
    for completion, tid in zip(completions, ticket_id):
        text = _completion_text(completion)
        state = run_episode_from_completion(text, tid)
        out.append(primary_reward(state))
    return out


def r_grounding(prompts, completions, ticket_id, **kwargs):
    out = []
    for completion, tid in zip(completions, ticket_id):
        text = _completion_text(completion)
        state = run_episode_from_completion(text, tid)
        out.append(reward_grounding(state))
    return out


def r_efficiency(prompts, completions, ticket_id, **kwargs):
    out = []
    for completion, tid in zip(completions, ticket_id):
        text = _completion_text(completion)
        state = run_episode_from_completion(text, tid)
        out.append(reward_efficiency(state))
    return out


def r_calibration(prompts, completions, ticket_id, **kwargs):
    out = []
    for completion, tid in zip(completions, ticket_id):
        text = _completion_text(completion)
        state = run_episode_from_completion(text, tid)
        out.append(reward_calibration(state))
    return out


def r_format(prompts, completions, ticket_id, **kwargs):
    """Shaped reward: +0.1 for valid tool call, +0.05 extra for submit_resolution."""
    out = []
    for completion in completions:
        text = _completion_text(completion)
        parsed = parse_tool_call(text)
        if parsed is None:
            out.append(0.0)
        elif parsed["tool_name"] == "submit_resolution":
            out.append(0.15)  # higher reward for actually submitting
        else:
            out.append(0.1)   # lower reward for search-only
    return out


def _completion_text(completion):
    """Extract text from completion (handles both str and chat-format)."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        # conversational format: [{"role": "assistant", "content": "..."}]
        for msg in reversed(completion):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content", "")
        return ""
    return str(completion)


# ── Build prompts that explain the tool-call format ───────────────────────────

SYSTEM_PROMPT = """You are an enterprise IT triage agent. You resolve support tickets by calling tools.

Available tools:
- search_kb(query, max_results=5): search KB articles
- search_tickets(query, max_results=5): search past resolved tickets
- search_incidents(query, max_results=3): search incident postmortems
- get_article(article_id): retrieve full KB article body
- get_ticket(ticket_id): retrieve full past ticket
- get_incident(incident_id): retrieve full incident postmortem
- submit_resolution(resolution, cited_artifacts, confidence, escalate=False): submit final answer

You MUST output your tool call as a single JSON object inside a ```json code block:
```json
{"tool_name": "submit_resolution", "arguments": {"resolution": "...", "cited_artifacts": ["KB-00001"], "confidence": 0.8, "escalate": false}}
```

Respond with ONE tool call. If you have enough information to resolve the ticket, call submit_resolution. Otherwise call a search or get tool first."""


def build_dataset(tickets: list):
    from datasets import Dataset

    rows = []
    for t in tickets:
        tid = t.get("ticket_id", t.get("id", ""))
        title = t.get("title", "")
        description = t.get("description", "")

        rows.append({
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"# New Ticket: {tid}\n\n"
                    f"**Title:** {title}\n\n"
                    f"**Description:** {description}\n\n"
                    "Resolve this ticket. Output exactly one tool call in a ```json``` block."
                )},
            ],
            "ticket_id": tid,
        })

    return Dataset.from_list(rows)


# ── Plot helpers ──────────────────────────────────────────────────────────────

def save_reward_curve(log_history: list, output_path: Path):
    reward_keys = [
        ("reward", "Total reward", "#2d6a4f", 1.8),
        ("rewards/r_primary/mean", "Primary (binary)", "#1d3557", 1.2),
        ("rewards/r_grounding/mean", "Grounding", "#457b9d", 1.2),
        ("rewards/r_efficiency/mean", "Efficiency", "#e9c46a", 1.2),
        ("rewards/r_calibration/mean", "Calibration", "#e76f51", 1.2),
        ("rewards/r_format/mean", "Format (tool-call valid)", "#9d4edd", 1.0),
    ]

    curves = {}
    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        for key, _, _, _ in reward_keys:
            if key in entry:
                curves.setdefault(key, {"steps": [], "vals": []})
                curves[key]["steps"].append(step)
                curves[key]["vals"].append(entry[key])

    if not curves:
        print("  No reward data in log_history — skipping plot.")
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    for key, label, colour, lw in reward_keys:
        if key in curves:
            ax.plot(curves[key]["steps"], curves[key]["vals"],
                    label=label, color=colour, linewidth=lw, alpha=0.9)

    ax.set_xlabel("Training step", fontsize=12)
    ax.set_ylabel("Mean reward", fontsize=12)
    ax.set_title("TriageAgent — GRPO Training Reward Curves", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Reward curve saved → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--no-vllm", action="store_true")
    args_cli = parser.parse_args()

    smoke     = args_cli.smoke_test
    use_vllm  = not args_cli.no_vllm
    max_steps = 5 if smoke else 200

    if smoke:
        use_vllm = False

    # Pre-load corpus & tickets so first reward call doesn't pay the cost
    get_corpus()
    ticket_index = get_ticket_index()
    train_tickets = list(ticket_index.values())
    dataset = build_dataset(train_tickets)

    print(f"Loaded {len(train_tickets)} training tickets.")
    print(f"max_steps={max_steps}  use_vllm={use_vllm}  smoke={smoke}")

    # ── Logging ───────────────────────────────────────────────────────────────
    report_to = "none"
    try:
        import wandb
        wandb.init(project="triage-agent-grpo", name=f"qwen3b-{'smoke' if smoke else 'full'}-v2")
        report_to = "wandb"
        print("wandb logging enabled.")
    except Exception as e:
        print(f"wandb not available ({e})")

    from trl import GRPOConfig, GRPOTrainer

    warmup_steps = max(1, int(max_steps * 0.1))

    vllm_kwargs = dict(
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.4,
    ) if use_vllm else dict(use_vllm=False)

    training_args = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        num_generations=4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_completion_length=1024,
        learning_rate=5e-6,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        loss_type="dr_grpo",
        reward_weights=[1.0, 0.3, 0.2, 0.15, 0.1],
        max_steps=max_steps,
        save_steps=25,
        logging_steps=1,
        bf16=True,
        report_to=report_to,
        push_to_hub=True,
        hub_model_id=HF_HUB_REPO_ID,
        hub_strategy="every_save",
        remove_unused_columns=False,
        **vllm_kwargs,
        temperature=1.0,
        top_p=0.9,
        repetition_penalty=1.1,
    )

    trainer = GRPOTrainer(
        model=MODEL,
        reward_funcs=[r_primary, r_grounding, r_efficiency, r_calibration, r_format],
        train_dataset=dataset,
        args=training_args,
    )

    try:
        print("\nStarting training…")
        trainer.train()
        trainer.save_model()
        print(f"\n✓ Training complete. Model saved → {OUTPUT_DIR}")
        save_reward_curve(trainer.state.log_history, PLOTS_DIR / "reward_curve.png")

        # Auto-commit plots if in a git repo
        for cmd in [
            ["git", "add", "assets/plots/reward_curve.png"],
            ["git", "commit", "-m", f"feat: GRPO reward curve ({max_steps} steps)"],
            ["git", "push"],
        ]:
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
            print(f"  $ {' '.join(cmd)}  →  rc={r.returncode}")

    except Exception as e:
        import traceback
        print(f"\n✗ Training failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        try:
            trainer.save_model()
            print(f"✓ Partial model saved → {OUTPUT_DIR}")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
