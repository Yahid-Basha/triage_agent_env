#!/usr/bin/env python3
"""
GRPO training for TriageAgent — single A100 (40 GB) configuration.

Uses TRL's environment_factory so the trainer handles the full multi-turn
tool-calling loop. All reward functions are logged independently so judges
can see each sub-reward curve.

Usage:
    uv run python training/train_grpo.py [--smoke-test]

Deps (not in pyproject.toml — install in the training environment):
    pip install trl>=0.12 transformers>=5.2 datasets bitsandbytes wandb
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt

# ── Project root on path ─────────────────────────────────────────────────────
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
MODEL = "Qwen/Qwen2.5-3B-Instruct"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "triage_grpo_qwen3b"


# ── Episode state ─────────────────────────────────────────────────────────────

class _State:
    """Minimal episode state — matches the attrs that rewards.py reads."""

    __slots__ = (
        "target_ticket_id", "gold_resolution", "gold_cited_ids",
        "difficulty", "is_unanswerable",
        "tools_called", "artifacts_viewed",
        "searches_made", "fetches_made",
        "submitted", "submitted_resolution", "submitted_citations",
        "submitted_confidence", "submitted_escalate",
    )

    def __init__(self):
        self.target_ticket_id: str = ""
        self.gold_resolution: str = ""
        self.gold_cited_ids: list = []
        self.difficulty: str = "medium"
        self.is_unanswerable: bool = False
        self.tools_called: list = []
        self.artifacts_viewed: list = []
        self.searches_made: int = 0
        self.fetches_made: int = 0
        self.submitted: bool = False
        self.submitted_resolution: Optional[str] = None
        self.submitted_citations: list = []
        self.submitted_confidence: Optional[float] = None
        self.submitted_escalate: bool = False


# ── TRL environment class ─────────────────────────────────────────────────────

class TriageEnvForTraining:
    """
    In-process TRL environment_factory wrapper.

    TRL introspects public methods (those without a leading underscore) and
    generates OpenAI-format tool schemas from their docstrings (Args: blocks).
    The trainer calls reset(**dataset_row_kwargs) to start each episode and
    reads the `reward` property after the episode ends.
    """

    def __init__(self, corpus: Corpus, ticket_index: dict):
        self._corpus = corpus
        self._ticket_index = ticket_index
        self._state = _State()

    # ── TRL lifecycle ─────────────────────────────────────────────────────────

    def reset(self, ticket_id: str = "", **kwargs) -> str:
        """Start a new triage episode for the given ticket."""
        ticket = self._ticket_index.get(ticket_id)
        if ticket is None:
            ticket = next(iter(self._ticket_index.values()))

        s = _State()
        s.target_ticket_id = ticket.get("ticket_id", ticket.get("id", ""))
        s.gold_resolution = ticket.get("gold_resolution", "")
        s.gold_cited_ids = list(ticket.get("gold_cited_ids", []))
        s.difficulty = ticket.get("difficulty", "medium")
        s.is_unanswerable = ticket.get("is_unanswerable", False)
        self._state = s

        return (
            f"# New Ticket: {s.target_ticket_id}\n\n"
            f"**Title:** {ticket.get('title', '')}\n\n"
            f"**Description:** {ticket.get('description', '')}\n\n"
            "You have 20 tool calls. Use search_kb, search_tickets, "
            "search_incidents, get_article, get_ticket, or get_incident to "
            "gather evidence. Then call submit_resolution to close the ticket."
        )

    @property
    def reward(self) -> float:
        """Total episode reward — read by TRL after each episode ends."""
        return (
            primary_reward(self._state)
            + reward_grounding(self._state)
            + reward_efficiency(self._state)
            + reward_calibration(self._state)
        )

    # ── Tools (public methods = exposed to model) ─────────────────────────────

    def search_kb(self, query: str, max_results: int = 5) -> str:
        """Search the knowledge base for articles matching a query.

        Args:
            query: Natural language search query.
            max_results: Maximum number of results to return (1-10).
        """
        self._state.searches_made += 1
        self._state.tools_called.append(("search_kb", {"query": query}))
        results = self._corpus.search_kb(query=query, max_results=max_results)
        return json.dumps(results, ensure_ascii=False)

    def get_article(self, article_id: str) -> str:
        """Retrieve the full body of a KB article by ID.

        Args:
            article_id: The article identifier, e.g. 'KB-00042'.
        """
        self._state.fetches_made += 1
        self._state.artifacts_viewed.append(article_id)
        self._state.tools_called.append(("get_article", {"article_id": article_id}))
        result = self._corpus.get_article(article_id=article_id)
        return json.dumps(result, ensure_ascii=False) if result else '{"error": "Not found."}'

    def search_tickets(
        self,
        query: str,
        status: Optional[str] = None,
        max_results: int = 5,
    ) -> str:
        """Search past resolved tickets for similar issues.

        Args:
            query: Natural language search query.
            status: Filter by ticket status, e.g. 'Resolved'.
            max_results: Maximum number of results to return (1-10).
        """
        self._state.searches_made += 1
        self._state.tools_called.append(("search_tickets", {"query": query}))
        results = self._corpus.search_tickets(
            query=query, status=status, max_results=max_results
        )
        return json.dumps(results, ensure_ascii=False)

    def get_ticket(self, ticket_id: str) -> str:
        """Retrieve a full past ticket including comments and resolution.

        Args:
            ticket_id: The ticket identifier, e.g. 'TKT-000123'.
        """
        self._state.fetches_made += 1
        self._state.artifacts_viewed.append(ticket_id)
        self._state.tools_called.append(("get_ticket", {"ticket_id": ticket_id}))
        result = self._corpus.get_ticket(ticket_id=ticket_id)
        return json.dumps(result, ensure_ascii=False) if result else '{"error": "Not found."}'

    def search_incidents(self, query: str, max_results: int = 3) -> str:
        """Search incident postmortems for related outages or failures.

        Args:
            query: Natural language search query.
            max_results: Maximum number of results to return (1-5).
        """
        self._state.searches_made += 1
        self._state.tools_called.append(("search_incidents", {"query": query}))
        results = self._corpus.search_incidents(query=query, max_results=max_results)
        return json.dumps(results, ensure_ascii=False)

    def get_incident(self, incident_id: str) -> str:
        """Retrieve a full incident postmortem by ID.

        Args:
            incident_id: The incident identifier, e.g. 'INC-1234'.
        """
        self._state.fetches_made += 1
        self._state.artifacts_viewed.append(incident_id)
        self._state.tools_called.append(("get_incident", {"incident_id": incident_id}))
        result = self._corpus.get_incident(incident_id=incident_id)
        return json.dumps(result, ensure_ascii=False) if result else '{"error": "Not found."}'

    def submit_resolution(
        self,
        resolution: str,
        cited_artifacts: List[str],
        confidence: float,
        escalate: bool = False,
    ) -> str:
        """Submit your final resolution. This ends the episode.

        Args:
            resolution: The resolution text to send to the requester.
            cited_artifacts: List of artifact IDs (KB, ticket, or incident) that support the resolution.
            confidence: Your confidence in the resolution, between 0.0 and 1.0.
            escalate: Set True if this ticket cannot be resolved with available information.
        """
        self._state.submitted = True
        self._state.submitted_resolution = resolution
        self._state.submitted_citations = list(cited_artifacts)
        self._state.submitted_confidence = float(confidence)
        self._state.submitted_escalate = bool(escalate)
        self._state.tools_called.append(("submit_resolution", {}))
        return "Resolution submitted. Episode complete."


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_tickets(path: Path) -> list:
    with open(path) as f:
        return json.load(f)


def build_dataset(tickets: list):
    from datasets import Dataset

    return Dataset.from_list([
        {
            "prompt": [
                {
                    "role": "user",
                    "content": "You are an enterprise IT triage agent. Resolve the ticket.",
                }
            ],
            "ticket_id": t.get("ticket_id", t.get("id", "")),
        }
        for t in tickets
    ])


# ── Per-group reward functions (TRL interface) ────────────────────────────────

def r_primary(environments, **kwargs):
    return [primary_reward(e._state) for e in environments]


def r_grounding(environments, **kwargs):
    return [reward_grounding(e._state) for e in environments]


def r_efficiency(environments, **kwargs):
    return [reward_efficiency(e._state) for e in environments]


def r_calibration(environments, **kwargs):
    return [reward_calibration(e._state) for e in environments]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run only 5 steps to verify the wiring (no GPU required).",
    )
    args_cli = parser.parse_args()

    max_steps = 5 if args_cli.smoke_test else 200

    # Load data
    corpus = Corpus(DATA_DIR)
    train_tickets = load_tickets(DATA_DIR / "train_tickets.json")
    ticket_index = {t.get("ticket_id", t.get("id", "")): t for t in train_tickets}
    dataset = build_dataset(train_tickets)

    print(f"Loaded {len(train_tickets)} training tickets. max_steps={max_steps}")

    # Environment factory — TRL calls this once per parallel rollout slot
    def env_factory():
        return TriageEnvForTraining(corpus, ticket_index)

    # wandb logging — run `wandb login` once in the environment
    try:
        import wandb
        wandb.init(project="triage-agent-grpo")
        print("wandb logging enabled.")
    except Exception as e:
        print(f"wandb not available ({e}). Logging to stdout only.")

    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        # GRPO group size — 4 completions per prompt, memory-safe on A100 40 GB
        num_generations=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        # Context length — max_completion_length covers the full multi-turn trajectory
        max_completion_length=512,
        # Optimizer
        learning_rate=5e-6,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        reward_weights=[1.0, 0.3, 0.2, 0.15],
        optim="adamw_8bit",
        # GRPO variant — dr_grpo fixes the length bias present in vanilla GRPO
        loss_type="dr_grpo",
        # Schedule
        max_steps=max_steps,
        save_steps=25,
        logging_steps=1,
        # Hardware
        bf16=True,
        # vLLM co-location: generation + training share the same A100
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.4,
        # Logging
        report_to="wandb",
        push_to_hub=True,
        hub_model_id="yahid/triage-agent-qwen3b",
        hub_strategy="every_save",
    )

    trainer = GRPOTrainer(
        model=MODEL,
        reward_funcs=[r_primary, r_grounding, r_efficiency, r_calibration],
        
        environment_factory=env_factory,
        train_dataset=dataset,
        args=training_args,
    )
 
    try:
        trainer.train()

        log_history = trainer.state.log_history
        steps = [x["step"] for x in log_history if "reward" in x]
        rewards = [x["reward"] for x in log_history if "reward" in x]

        plt.figure(figsize=(10, 5))
        plt.plot(steps, rewards)
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("GRPO Training — Reward Curve")
        plt.savefig(ROOT / "assets/plots/reward_curve.png", dpi=150)
        print("Plot saved.")

        trainer.save_model()
        print(f"\n✓ Training complete. Model saved to {OUTPUT_DIR}")
    except Exception as e:
        import traceback
        print(f"\n✗ Training failed: {type(e).__name__}: {e}")
        traceback.print_exc()
        # Still try to save whatever we have
        try:
            trainer.save_model()
            print(f"✓ Partial model saved to {OUTPUT_DIR}")
        except Exception:
            pass
        raise

if __name__ == "__main__":
    main()
