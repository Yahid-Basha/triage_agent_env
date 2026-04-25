#!/usr/bin/env python3
"""
scripts/baseline_eval.py — Pre-training baseline evaluation.

Runs Qwen/Qwen2.5-3B-Instruct on all eval tickets via the HuggingFace
Inference API (OpenAI-compatible), calling the 7 triage tools for up to
20 turns per ticket.

Outputs:
    assets/baseline_eval.json           — per-ticket results + summary
    assets/plots/baseline_rewards.png   — bar chart of mean sub-rewards

Usage:
    uv run python scripts/baseline_eval.py
    uv run python scripts/baseline_eval.py --tickets data/train_tickets.json  # override ticket file
"""

import argparse
import json
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Load .env before anything else
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI

from server.corpus import Corpus
from server import rewards as R
from server.triage_environment import _EpisodeState, MAX_TURNS
from models import TriageAction, TriageObservation

# ------------------------------------------------------------------ #
# Config                                                               #
# ------------------------------------------------------------------ #

# Provider-agnostic — works with HF, Groq, Together, OpenAI, etc.
# Set BASE_URL, MODEL, API_KEY in .env (or export them).
API_KEY = (
    os.getenv("API_KEY")
    or os.getenv("HF_TOKEN")
)
BASE_URL = (
    os.getenv("BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or os.getenv("API_BASE_URL")
    or "https://router.huggingface.co/v1"
)
MODEL = (
    os.getenv("MODEL")
    or os.getenv("MODEL_NAME")
    or "Qwen/Qwen2.5-7B-Instruct"
)

DATA_DIR = ROOT / "data"
ASSETS_DIR = ROOT / "assets"
PLOTS_DIR = ASSETS_DIR / "plots"
OUTPUT_JSON = ASSETS_DIR / "baseline_eval.json"
OUTPUT_PNG = PLOTS_DIR / "baseline_rewards.png"

SUB_KEYS = ["primary", "grounding", "efficiency", "calibration", "format"]

# ------------------------------------------------------------------ #
# Tool schemas (OpenAI function-calling format)                        #
# ------------------------------------------------------------------ #

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_kb",
            "description": "Search the knowledge base for articles matching a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query."},
                    "max_results": {"type": "integer", "description": "Max results (default 5).", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_article",
            "description": "Retrieve the full body of a KB article by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "article_id": {"type": "string", "description": "Article ID, e.g. 'KB-00042'."},
                },
                "required": ["article_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_tickets",
            "description": "Search past resolved tickets by keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "status": {"type": "string", "description": "Filter by status, e.g. 'Resolved'."},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ticket",
            "description": "Retrieve a full past ticket with comments and resolution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {"type": "string", "description": "Ticket ID, e.g. 'TKT-000042'."},
                },
                "required": ["ticket_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_incidents",
            "description": "Search incident postmortems by keyword.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 3},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_incident",
            "description": "Retrieve a full incident postmortem by ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "incident_id": {"type": "string", "description": "Incident ID, e.g. 'INC-0042'."},
                },
                "required": ["incident_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_resolution",
            "description": (
                "Submit your final resolution. This ENDS the episode. "
                "Call this once you have enough information to resolve the ticket."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "resolution": {
                        "type": "string",
                        "description": "The resolution text (1-3 sentences describing the fix).",
                    },
                    "cited_artifacts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "IDs of KB articles, tickets, or incidents you used.",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Your confidence 0.0-1.0.",
                    },
                    "escalate": {
                        "type": "boolean",
                        "description": "True if this ticket cannot be resolved with available information.",
                        "default": False,
                    },
                },
                "required": ["resolution", "cited_artifacts", "confidence"],
            },
        },
    },
]

SYSTEM_PROMPT = f"""You are an enterprise IT triage agent. You receive a support ticket and \
must resolve it by querying the knowledge base, past tickets, and incident postmortems \
using the provided tools.

Strategy:
1. Search for relevant KB articles, past tickets, and incidents related to the problem.
2. Fetch the most relevant results to read in detail (use get_article, get_ticket, get_incident).
3. Synthesize a resolution based on what you found.
4. Call submit_resolution with your answer, the artifact IDs you cited, and a confidence score.
   Set escalate=true only if the problem has no solution in the available knowledge base.

You have at most {MAX_TURNS} tool calls. Be efficient: search first, fetch the most \
relevant results, then submit. Always call submit_resolution to end the episode."""


# ------------------------------------------------------------------ #
# Environment helpers                                                  #
# ------------------------------------------------------------------ #

def reset_env_to_ticket(env, ticket: dict) -> TriageObservation:
    """Directly wire an eval ticket into the environment, bypassing reset()."""
    ep = _EpisodeState()
    ep.episode_id = str(uuid.uuid4())[:8]
    ep.target_ticket_id = ticket.get("ticket_id", "")
    ep.gold_resolution = ticket.get("gold_resolution", "")
    ep.gold_cited_ids = list(ticket.get("gold_cited_ids", []))
    ep.difficulty = ticket.get("difficulty", "medium")
    ep.is_unanswerable = ticket.get("is_unanswerable", False)
    env._ep = ep
    env._current_ticket = ticket
    return TriageObservation(
        ticket_id=ep.target_ticket_id,
        ticket_title=ticket.get("title", ""),
        ticket_description=ticket.get("description", ""),
        tool_name="reset",
        tool_result={},
        turn=0,
        max_turns=MAX_TURNS,
        remaining_budget=MAX_TURNS,
        done=False,
        reward=None,
        info={},
    )


# ------------------------------------------------------------------ #
# Model interaction                                                    #
# ------------------------------------------------------------------ #

def tool_call_to_action(tc) -> TriageAction:
    """Convert an OpenAI tool_call object to a TriageAction."""
    name = tc.function.name
    try:
        args = json.loads(tc.function.arguments or "{}")
    except json.JSONDecodeError:
        args = {}
    return TriageAction(tool_name=name, **{k: v for k, v in args.items()
                                           if k in TriageAction.model_fields})


def assistant_msg_dict(msg) -> dict:
    """Convert openai ChatCompletionMessage to a plain dict for the message list."""
    d: dict = {"role": "assistant", "content": msg.content or ""}
    if msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    return d


def call_model(client: OpenAI, messages: list, max_retries: int = 3):
    """Call the model with simple exponential-backoff retry on transient errors."""
    delay = 2.0
    last_exc = None
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=1024,
                temperature=0.2,
            )
        except Exception as exc:
            last_exc = exc
            code = getattr(getattr(exc, "response", None), "status_code", None)
            if code == 429 or "rate" in str(exc).lower():
                print(f"      [rate-limited, retry {attempt+1}/{max_retries} in {delay:.0f}s]")
                time.sleep(delay)
                delay *= 2
            else:
                raise
    raise last_exc


# ------------------------------------------------------------------ #
# Per-ticket episode runner                                            #
# ------------------------------------------------------------------ #

def run_ticket(ticket: dict, env, client: OpenAI) -> dict:
    """Run one eval episode and return a result dict."""
    tid = ticket.get("ticket_id", "?")
    obs = reset_env_to_ticket(env, ticket)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"# Ticket {obs.ticket_id}\n\n"
                f"**Title:** {obs.ticket_title}\n\n"
                f"**Description:** {obs.ticket_description}\n\n"
                f"Investigate and resolve this ticket. Call submit_resolution when done."
            ),
        },
    ]

    done = False
    done_reason = "timeout"
    final_obs = obs
    llm_calls = 0
    error_turns = 0

    for _ in range(MAX_TURNS):
        # Stop if a previous tool call already terminated the episode
        if done:
            break

        try:
            response = call_model(client, messages)
        except Exception as exc:
            print(f"      [API error on turn {env._ep.step_count+1}: {str(exc)[:80]}]")
            error_turns += 1
            if error_turns >= 3:
                break
            time.sleep(3)
            continue

        llm_calls += 1
        msg = response.choices[0].message
        messages.append(assistant_msg_dict(msg))

        if not msg.tool_calls:
            # Model responded with text, no tool call — inject a nudge once
            if llm_calls == 1:
                messages.append({
                    "role": "user",
                    "content": "Please use one of the available tools to investigate or submit your resolution.",
                })
            else:
                break  # give up
            continue

        # Execute each tool call in the response
        for tc in msg.tool_calls:
            try:
                action = tool_call_to_action(tc)
            except Exception as exc:
                # Malformed args — return an error result to the model
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps({"error": f"Invalid arguments: {exc}"}),
                })
                continue

            final_obs = env.step(action)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(final_obs.tool_result),
            })

            if final_obs.done:
                done = True
                done_reason = "submitted"
                break  # stop processing remaining tool calls in this batch

        # Small courtesy delay to avoid hammering the API
        time.sleep(0.3)

    # Compute reward breakdown from final env state
    state = env.state   # TriageState (Pydantic snapshot)
    if done_reason == "submitted" and final_obs.info:
        breakdown = final_obs.info
    else:
        breakdown = R.reward_breakdown(state)

    total_reward = breakdown.get("total", 0.0) if done_reason == "submitted" else 0.0

    return {
        "ticket_id": tid,
        "difficulty": ticket.get("difficulty", "medium"),
        "is_unanswerable": ticket.get("is_unanswerable", False),
        "done_reason": done_reason,
        "turns": state.step_count,
        "llm_calls": llm_calls,
        "rewards": {k: round(breakdown.get(k, 0.0), 4) for k in SUB_KEYS + ["total"]},
        "total_reward": round(total_reward, 4),
        "submitted": state.submitted,
        "submitted_escalate": state.submitted_escalate,
        "n_searches": state.searches_made,
        "n_fetches": state.fetches_made,
    }


# ------------------------------------------------------------------ #
# Plotting                                                             #
# ------------------------------------------------------------------ #

def plot_results(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    labels = SUB_KEYS
    values = [summary["mean_rewards"].get(k, 0.0) for k in labels]
    colors = ["#2563eb", "#16a34a", "#d97706", "#7c3aed", "#db2777"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="white", linewidth=0.8)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    ax.set_ylim(0, max(values) * 1.35 + 0.05)
    ax.set_xlabel("Reward dimension", fontsize=12)
    ax.set_ylabel("Mean reward", fontsize=12)
    ax.set_title(
        f"Baseline rewards — {MODEL}\n"
        f"n={summary['n_tickets']}  "
        f"submitted={summary['n_submitted']}/{summary['n_tickets']}  "
        f"mean_total={summary['mean_rewards'].get('total', 0):.3f}",
        fontsize=11,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Plot saved → {path}")


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _save_checkpoint(results: list, all_tickets: list, path: Path) -> None:
    """Write partial results to disk so crashes don't lose work."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(results)
    n_sub = sum(1 for r in results if r.get("submitted"))
    partial = {
        "meta": {
            "model": MODEL,
            "base_url": BASE_URL,
            "eval_date": datetime.now().strftime("%Y-%m-%d"),
            "n_tickets_total": len(all_tickets),
            "n_tickets_done": n,
            "partial": n < len(all_tickets),
        },
        "results": results,
    }
    with open(path, "w") as f:
        json.dump(partial, f, indent=2)


# ------------------------------------------------------------------ #
# Main                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline eval for TriageAgent")
    parser.add_argument(
        "--tickets",
        default=str(DATA_DIR / "eval_tickets.json"),
        help="Path to eval tickets JSON (default: data/eval_tickets.json)",
    )
    parser.add_argument(
        "--output", default=str(OUTPUT_JSON),
        help="Output JSON path",
    )
    parser.add_argument(
        "--plot", default=str(OUTPUT_PNG),
        help="Output PNG path",
    )
    parser.add_argument(
        "--start-from", type=int, default=1, metavar="N",
        help="1-based ticket index to resume from (loads previous results from --output)",
    )
    args = parser.parse_args()

    # Validate inputs
    if not API_KEY:
        print("ERROR: API_KEY (or HF_TOKEN) not set. Add it to .env or export API_KEY=...")
        sys.exit(1)

    tickets_path = Path(args.tickets)
    if not tickets_path.exists():
        print(f"ERROR: Ticket file not found: {tickets_path}")
        sys.exit(1)

    with open(tickets_path) as f:
        tickets = json.load(f)
    if not isinstance(tickets, list) or not tickets:
        print("ERROR: Ticket file must be a non-empty JSON array.")
        sys.exit(1)

    start_idx = args.start_from - 1  # convert to 0-based
    if start_idx < 0 or start_idx >= len(tickets):
        print(f"ERROR: --start-from {args.start_from} out of range (1–{len(tickets)}).")
        sys.exit(1)

    print(f"Loaded {len(tickets)} tickets from {tickets_path}")

    # Load previous results when resuming mid-run
    out_path = Path(args.output)
    prior_results: list = []
    if start_idx > 0:
        if out_path.exists():
            with open(out_path) as f:
                saved = json.load(f)
            prior_results = saved.get("results", [])[:start_idx]
            print(f"Resuming from ticket {args.start_from} — "
                  f"loaded {len(prior_results)} prior result(s) from {out_path}")
        else:
            print(f"WARNING: --start-from {args.start_from} requested but {out_path} not found; "
                  f"prior results will be empty.")

    # Init env (loads corpus + embeddings)
    print("Initialising TriageAgentEnvironment …")
    from server.triage_environment import TriageAgentEnvironment
    env = TriageAgentEnvironment()
    print(f"  Corpus: {len(env._corpus._kb)} KB articles, "
          f"{len(env._corpus._tickets)} past tickets, "
          f"{len(env._corpus._incidents)} incidents")

    # Init LLM client — provider-agnostic
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print(f"  Model   : {MODEL}")
    print(f"  Base URL: {BASE_URL}")
    key_hint = (API_KEY or "")[:8] + "..." if API_KEY else "NOT SET"
    print(f"  API key : {key_hint}\n")

    # Run eval for tickets[start_idx:]
    new_results = []
    total = len(tickets)
    for i, ticket in enumerate(tickets[start_idx:], start=start_idx):
        tid = ticket.get("ticket_id", f"#{i+1}")
        diff = ticket.get("difficulty", "?")
        unans = " [unanswerable]" if ticket.get("is_unanswerable") else ""
        print(f"[{i+1:>3}/{total}] {tid} [{diff}]{unans}")

        result = run_ticket(ticket, env, client)
        new_results.append(result)

        r = result["rewards"]
        print(f"        done={result['done_reason']:<10} "
              f"turns={result['turns']:>2}  "
              f"primary={r['primary']:.2f}  "
              f"grounding={r['grounding']:.2f}  "
              f"total={r['total']:.3f}")

        # Checkpoint after every ticket so a crash doesn't lose work
        _save_checkpoint(prior_results + new_results, tickets, out_path)

    results = prior_results + new_results

    # Compute summary
    n = len(results)
    n_submitted = sum(1 for r in results if r["submitted"])
    n_timeout = n - n_submitted
    mean_turns = sum(r["turns"] for r in results) / n if n else 0

    mean_rewards = {}
    for k in SUB_KEYS + ["total"]:
        mean_rewards[k] = round(sum(r["rewards"].get(k, 0.0) for r in results) / n, 4) if n else 0.0

    # Per-difficulty breakdown
    by_diff: Dict[str, list] = {}
    for r in results:
        by_diff.setdefault(r["difficulty"], []).append(r["rewards"].get("total", 0.0))
    diff_summary = {
        d: {"n": len(v), "mean_total": round(sum(v) / len(v), 4)}
        for d, v in by_diff.items()
    }

    summary = {
        "model": MODEL,
        "base_url": BASE_URL,
        "eval_date": datetime.now().strftime("%Y-%m-%d"),
        "n_tickets": n,
        "n_submitted": n_submitted,
        "n_timeout": n_timeout,
        "mean_turns": round(mean_turns, 2),
        "mean_rewards": mean_rewards,
        "by_difficulty": diff_summary,
        "partial": False,
    }

    output = {"meta": summary, "results": results}

    # Final save (replaces checkpoints)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # Print summary table
    print(f"\n{'─'*55}")
    print(f"  Baseline summary — {MODEL}")
    print(f"{'─'*55}")
    print(f"  Tickets evaluated : {n}  (submitted={n_submitted}, timeout={n_timeout})")
    print(f"  Mean turns        : {mean_turns:.1f}")
    for k in SUB_KEYS + ["total"]:
        print(f"  mean_{k:<14}: {mean_rewards[k]:.4f}")
    print(f"{'─'*55}")

    # Plot
    plot_results(summary, Path(args.plot))


if __name__ == "__main__":
    main()
