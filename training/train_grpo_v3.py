#!/usr/bin/env python3
"""
GRPO training for TriageAgent — grounded single-call rollout approach (v3).

Changes from v2:
- build_training_prompt injects retrieved context (gold + distractors) into every
  prompt so the model learns to cite from evidence, not recall.
- Five new reward functions: r_format_graduated, r_resolution_quality,
  r_citation_grounding, r_calibration, r_parsimony.
- GRPOConfig tuned: num_generations=8, max_completion_length=512, lr=1e-5,
  temperature=0.9, reward_weights=[0.3, 1.5, 1.0, 0.5, 0.3].

Usage:
    python training/train_grpo_v3.py              # full 200-step run
    python training/train_grpo_v3.py --smoke-test # 5-step verification
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
from rouge_score import rouge_scorer

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


# ── System prompt for grounded resolution ────────────────────────────────────

SYSTEM_PROMPT_GROUNDED = """You are an enterprise IT triage agent. Resolve the ticket using ONLY the retrieved context provided in the user message.

Output EXACTLY one JSON object inside a ```json code block, with this exact schema:
```json
{"tool_name": "submit_resolution", "arguments": {"resolution": "...", "cited_artifacts": ["KB-00001"], "confidence": 0.85, "escalate": false}}
```

Required fields and types:
- "resolution" (string): how to fix the issue, in your own words
- "cited_artifacts" (list of strings): artifact IDs from Retrieved Context only — never invent IDs
- "confidence" (float 0.0–1.0): your certainty that the resolution is correct
- "escalate" (bool): true ONLY if Retrieved Context lacks relevant information

Do not add any other fields. Do not output text outside the JSON block."""


# ── Prompt builder with retrieved context ────────────────────────────────────

def build_training_prompt(ticket, corpus, distractor_k=3):
    # Real signal: gold-cited articles
    gold_articles = []
    for aid in ticket.get("gold_cited_ids", []):
        art = corpus.get_article(aid) or corpus.get_ticket(aid) or corpus.get_incident(aid)
        if art:
            gold_articles.append(art)

    # Realistic noise: distractors from actual search
    distractor_hits = corpus.search_kb(ticket["title"], max_results=distractor_k + 2)
    distractors = [
        corpus.get_article(h["article_id"]) for h in distractor_hits
        if h["article_id"] not in ticket.get("gold_cited_ids", [])
    ]
    distractors = [d for d in distractors if d is not None][:distractor_k]

    # Shuffle so position doesn't leak gold
    import random
    context_items = gold_articles + distractors
    random.shuffle(context_items)

    context_block = "\n\n".join(
        f"### {a.get('article_id', a.get('id', ''))}\n{a.get('title', '')}\n{a.get('body', a.get('content', ''))[:1000]}"
        for a in context_items
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT_GROUNDED},
        {"role": "user", "content": (
            f"# Ticket: {ticket.get('ticket_id', ticket.get('id', ''))}\n"
            f"**Title:** {ticket['title']}\n"
            f"**Description:** {ticket['description']}\n\n"
            f"# Retrieved Context:\n{context_block}\n\n"
            "Resolve this ticket using ONLY the retrieved context. Output exactly one "
            "`submit_resolution` tool call as a JSON code block."
        )},
    ]


def build_dataset(tickets: list, corpus: Corpus):
    from datasets import Dataset
    rows = []
    for t in tickets:
        tid = t.get("ticket_id", t.get("id", ""))
        prompt = build_training_prompt(t, corpus)
        rows.append({
            "prompt": prompt,
            "ticket_id": tid,
        })
    return Dataset.from_list(rows)


# ── Reward functions ──────────────────────────────────────────────────────────

# ── Reward 1: Graduated format (5 partial-credit tiers) ──
def r_format_graduated(completions, **kwargs):
    """0.0 → 1.0 in five steps. Forces variance even when model fails to submit."""
    out = []
    for c in completions:
        text = _completion_text(c)
        score = 0.0
        # Tier 1: contains any tool-call markup
        if "```json" in text or "<tool_call>" in text:
            score = 0.2
        parsed = parse_tool_call(text)
        if parsed is None:
            out.append(score)
            continue
        # Tier 2: parses to JSON
        score = 0.4
        # Tier 3: valid tool name
        if parsed["tool_name"] in KNOWN_TOOLS:
            score = 0.6
        # Tier 4: actually calls submit_resolution
        if parsed["tool_name"] == "submit_resolution":
            score = 0.8
            args = parsed.get("arguments", {})
            # Tier 5: all required fields present and non-empty
            req = {"resolution", "cited_artifacts", "confidence"}
            if (req.issubset(args.keys())
                and isinstance(args.get("resolution"), str)
                and len(args["resolution"]) > 20
                and isinstance(args.get("cited_artifacts"), list)):
                score = 1.0
        out.append(score)
    return out


# ── Reward 2: Resolution quality (ROUGE-L vs gold) ──
_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def r_resolution_quality(completions, ticket_id, **kwargs):
    """Continuous text-similarity score. Drives most of the variance."""
    idx = get_ticket_index()
    out = []
    for c, tid in zip(completions, ticket_id):
        text = _completion_text(c)
        parsed = parse_tool_call(text)
        if parsed is None or parsed["tool_name"] != "submit_resolution":
            out.append(0.0); continue
        submitted = parsed.get("arguments", {}).get("resolution", "")
        gold = idx.get(tid, {}).get("gold_resolution", "")
        if not submitted or not gold:
            out.append(0.0); continue
        f1 = _ROUGE.score(gold, submitted)["rougeL"].fmeasure
        out.append(f1)
    return out


# ── Reward 3: Citation F1 vs gold ──
def r_citation_grounding(completions, ticket_id, **kwargs):
    idx = get_ticket_index()
    out = []
    for c, tid in zip(completions, ticket_id):
        text = _completion_text(c)
        parsed = parse_tool_call(text)
        ticket = idx.get(tid, {})
        gold = set(ticket.get("gold_cited_ids", []))
        is_unans = ticket.get("is_unanswerable", False)

        if parsed is None or parsed["tool_name"] != "submit_resolution":
            out.append(0.0); continue
        cited = set(parsed.get("arguments", {}).get("cited_artifacts", []))

        # Unanswerable case: reward abstention (cite nothing, escalate)
        if is_unans:
            esc = bool(parsed["arguments"].get("escalate", False))
            out.append(1.0 if (not cited and esc) else 0.3 if not cited else 0.0)
            continue

        if not gold:
            esc = bool(parsed.get("arguments", {}).get("escalate", False))
            out.append(1.0 if esc else 0.1)
            continue
        if not cited:
            out.append(0.0); continue
        tp = len(cited & gold)
        if tp == 0:
            out.append(0.0); continue
        p, r = tp / len(cited), tp / len(gold)
        out.append(2 * p * r / (p + r))
    return out


# ── Reward 4: Confidence calibration (Brier-style) ──
def r_calibration(completions, ticket_id, **kwargs):
    """Reward = 1 - (confidence - actual_quality)^2.

    Couples two output dimensions — confidence AND resolution quality —
    so it varies across completions in a way no other reward can fake.
    """
    idx = get_ticket_index()
    out = []
    for c, tid in zip(completions, ticket_id):
        text = _completion_text(c)
        parsed = parse_tool_call(text)
        if parsed is None or parsed["tool_name"] != "submit_resolution":
            out.append(0.0); continue
        args = parsed.get("arguments", {})
        try:
            conf = float(args.get("confidence", 0.5))
            conf = max(0.0, min(1.0, conf))
        except Exception:
            out.append(0.0); continue
        # Compute resolution quality on the fly (cheap; can cache)
        gold = idx.get(tid, {}).get("gold_resolution", "")
        submitted = args.get("resolution", "")
        quality = _ROUGE.score(gold, submitted)["rougeL"].fmeasure if (gold and submitted) else 0.0
        out.append(1.0 - (conf - quality) ** 2)
    return out


# ── Reward 5: Length parsimony (anti-stub, anti-rambling) ──
def r_parsimony(completions, **kwargs):
    out = []
    for c in completions:
        text = _completion_text(c)
        parsed = parse_tool_call(text)
        if parsed is None:
            out.append(0.0); continue
        n = len(text.split())
        if n < 40:        out.append(0.2)   # stub
        elif n < 80:      out.append(0.6)
        elif n <= 400:    out.append(1.0)   # sweet spot
        elif n <= 800:    out.append(0.6)
        else:             out.append(0.2)   # rambling
    return out


# ── Plot helpers ──────────────────────────────────────────────────────────────

def save_reward_curve(log_history: list, output_path: Path):
    reward_keys = [
        ("reward", "Total reward", "#2d6a4f", 1.8),
        ("rewards/r_format_graduated/mean", "Format", "#9d4edd", 1.0),
        ("rewards/r_resolution_quality/mean", "Resolution Quality", "#1d3557", 1.2),
        ("rewards/r_citation_grounding/mean", "Citation F1", "#457b9d", 1.2),
        ("rewards/r_calibration/mean", "Calibration", "#e76f51", 1.2),
        ("rewards/r_parsimony/mean", "Parsimony", "#e9c46a", 1.2),
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

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    # Pre-load corpus & tickets so first reward call doesn't pay the cost
    get_corpus()
    ticket_index = get_ticket_index()
    train_tickets = list(ticket_index.values())
    dataset = build_dataset(train_tickets, get_corpus())

    print(f"Loaded {len(train_tickets)} training tickets.")
    print(f"max_steps={max_steps}  use_vllm={use_vllm}  smoke={smoke}")

    # ── Logging ───────────────────────────────────────────────────────────────
    report_to = "none"
    try:
        import wandb
        wandb.init(project="triage-agent-grpo", name=f"qwen3b-{'smoke' if smoke else 'full'}-v3")
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

    extra_kwargs = {}
    try:
        import inspect
        from trl import GRPOConfig as _check
        sig = inspect.signature(_check)
        if "log_completions" in sig.parameters:
            extra_kwargs["log_completions"] = True
            extra_kwargs["num_completions_to_print"] = 2
    except Exception:
        pass

    training_args = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        num_generations=8,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_completion_length=512,
        learning_rate=1e-5,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        loss_type="dr_grpo",
        reward_weights=[0.3, 1.5, 1.0, 0.5, 0.3],
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
        temperature=0.9,
        top_p=0.95,
        repetition_penalty=1.0,
        **extra_kwargs,
    )

    trainer = GRPOTrainer(
        model=MODEL,
        reward_funcs=[r_format_graduated, r_resolution_quality, r_citation_grounding, r_calibration, r_parsimony],
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
