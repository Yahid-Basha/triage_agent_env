#!/usr/bin/env python3

# ── vllm_ascend stub — must run before any TRL import ────────────────────────
# TRL's Ascend-patched import_utils.py calls importlib.util.find_spec("vllm_ascend")
# at module level. find_spec raises ValueError if the module is in sys.modules
# but has __spec__=None. This stub gives it a valid spec so TRL continues.
import sys as _sys, types as _types, importlib.util as _iutil
def _stub_pkg(name):
    m = _types.ModuleType(name)
    m.__spec__ = _iutil.spec_from_loader(name, loader=None)
    m.__path__ = []
    m.__package__ = name
    _sys.modules[name] = m
    return m

_stub_pkg("vllm_ascend")
_stub_pkg("vllm_ascend.distributed")
_stub_pkg("vllm_ascend.distributed.device_communicators")
_stub_pkg("mergekit")
_mk_config = _stub_pkg("mergekit.config")
_mk_config.MergeConfiguration = type("MergeConfiguration", (), {})

_mk_merge = _stub_pkg("mergekit.merge")
_mk_merge.MergeOptions = type("MergeOptions", (), {})
_mk_merge.run_merge = lambda *args, **kwargs: None

_stub_pkg("llm_blender")

_pyhccl = _stub_pkg("vllm_ascend.distributed.device_communicators.pyhccl")
_pyhccl.PyHcclCommunicator = type("PyHcclCommunicator", (), {})

del _stub_pkg, _pyhccl  # clean up helper


# ─────────────────────────────────────────────────────────────────────────────
"""
GRPO training for TriageAgent — grounded single-call rollout approach (v4.1).

Changes from v4:
- r_citation_grounding gains a behavioral floor: completions that cite any ID
  from the Retrieved Context (even the wrong article) now score 0.3 instead of
  0.0. Hallucinated IDs (not in context) score 0.05. This bootstraps non-zero
  citation gradient so GRPO can amplify the behavior. The floor only activates
  when the prompt is available (via the new `prompts` kwarg that TRL passes).
- build_training_prompt appends an explicit citation requirement sentence so the
  base model is less likely to emit `cited_artifacts: []` before any gradient.
- _smoke_generate now returns (ticket_id, completion, prompt) triples and uses
  a fresh GenerationConfig to avoid the `do_sample=False + top_p` warning.
- print_reward_table forwards prompts to r_citation_grounding so the floor
  logic activates in smoke-test scoring too.
- Added _prompt_user_text() helper used by the citation floor.

Changes from v3:
- Replaced parse_tool_call with a balanced-brace JSON extractor that handles
  all four formats Qwen emits: code-fenced standard, dict shorthand
  ({"submit_resolution": {...}}), Python call syntax (submit_resolution({...})),
  and OpenAI function form. Verified against 8 real smoke-test completions:
  6/8 useful parses, 1/8 correctly rejected (string-value malformed),
  1/8 partial (model invented `resolution_notes` field — format reward will
  punish this and GRPO will iron it out).
- Reward shaping fixes for cold-start (the 8-completion w&b export at step 5
  showed r_resolution_quality and r_citation_grounding pinned at 0 while
  r_format_graduated sat at 0.8 — no gradient toward correct schema):
    * r_resolution_quality now falls back to alternate prose keys
      (`details`, `resolution_steps`, `description`, `solution`, …) with a
      0.6× discount so the canonical `resolution` field still wins.
    * r_citation_grounding regex-scans args + completion text for
      KB-/INC-/TKT- IDs when `cited_artifacts` is absent, also at 0.6×.
    * r_format_graduated no longer credits 0.8 when `resolution` is
      missing/empty; Python-call form is capped at 0.5 and double-wrap at
      0.35 so canonical JSON-object form is strictly preferred.

Changes from v2:
- build_training_prompt injects retrieved context (gold + distractors) into every
  prompt so the model learns to cite from evidence, not recall.
- Five new reward functions: r_format_graduated, r_resolution_quality,
  r_citation_grounding, r_calibration, r_parsimony.
- GRPOConfig tuned: num_generations=8, max_completion_length=512, lr=1e-5,
  temperature=0.9, reward_weights=[0.3, 1.5, 1.0, 0.5, 0.3].

Usage:
    python training/train_grpo_v4.py              # full 200-step run
    python training/train_grpo_v4.py --smoke-test # 5-step verification
"""

import argparse
import json
import os
import re
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from rouge_score import rouge_scorer

# Lazy import — TrainerCallback is available whenever TRL/transformers are installed.
try:
    from transformers import TrainerCallback as _TrainerCallback
except ImportError:
    _TrainerCallback = object  # graceful no-op if transformers isn't on path yet

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

# ── Completion buffer (populated by r_parsimony, flushed by CompletionDumper) ─
# Each entry: {ticket_id, completion, parsed, resolution, cited_artifacts,
#              confidence, escalate, r_parsimony}
_PENDING_COMPLETIONS: List[dict] = []


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
#
# Robust parser using balanced-brace JSON extraction. Handles four formats:
#   (A) Standard:      {"tool_name": "...", "arguments": {...}}
#   (B) OpenAI:        {"function": {"name": "...", "arguments": "..."}}
#   (C) Dict shorthand: {"submit_resolution": {...args...}}
#   (D) Python call:   submit_resolution({...})
# All four can be wrapped in ```json ... ``` fences or <tool_call> tags, or bare.
#
# Verified against 8 real smoke-test completions: see /tests/test_parser.py.

KNOWN_TOOLS = {
    "search_kb", "search_tickets", "search_incidents",
    "get_article", "get_ticket", "get_incident",
    "submit_resolution",
}


def _extract_balanced_json(s: str, start: int) -> Optional[str]:
    """From index `start` (must be '{'), return the substring through the matching '}'.

    Walks the string respecting double-quoted strings and backslash escapes, so
    nested objects and braces inside strings don't break it.
    """
    if start >= len(s) or s[start] != '{':
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        c = s[i]
        if escape:
            escape = False
            continue
        if c == '\\':
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def _normalize(name: str, args: Any) -> Optional[Dict[str, Any]]:
    """Validate tool name and coerce args to a dict. Returns None if invalid."""
    if name not in KNOWN_TOOLS:
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    if not isinstance(args, dict):
        args = {}
    return {"tool_name": name, "arguments": args}


def _normalize_object(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Given a parsed JSON object, figure out the tool name + args."""
    # Form A: standard {"tool_name": "...", "arguments": {...}}
    name = obj.get("tool_name") or obj.get("name")
    if isinstance(name, str) and name in KNOWN_TOOLS:
        args = obj.get("arguments") or obj.get("parameters") or {}
        return _normalize(name, args)

    # Form B: OpenAI-style {"function": {"name": "...", "arguments": "..."}}
    fn = obj.get("function")
    if isinstance(fn, dict):
        name = fn.get("name")
        if isinstance(name, str) and name in KNOWN_TOOLS:
            return _normalize(name, fn.get("arguments", {}))

    # Form C: shorthand {"submit_resolution": {...args...}}
    # NOTE: only accept when the value is a dict — string values are malformed.
    for tool in KNOWN_TOOLS:
        if tool in obj and isinstance(obj[tool], dict):
            return _normalize(tool, obj[tool])

    return None


def _try_parse_body(body: str) -> Optional[Dict[str, Any]]:
    """Try every supported tool-call format on a chunk of text."""
    # Form D: Python call syntax `tool_name({...})`
    call_match = re.search(r'\b(' + '|'.join(KNOWN_TOOLS) + r')\s*\(\s*\{', body)
    if call_match:
        brace_start = call_match.end() - 1  # position of the '{'
        json_part = _extract_balanced_json(body, brace_start)
        if json_part:
            try:
                args = json.loads(json_part)
                result = _normalize(call_match.group(1), args)
                if result:
                    return result
            except json.JSONDecodeError:
                pass

    # Forms A/B/C: standalone JSON objects (try every '{' until one parses)
    idx = 0
    while idx < len(body):
        brace = body.find('{', idx)
        if brace == -1:
            break
        json_part = _extract_balanced_json(body, brace)
        if json_part:
            try:
                obj = json.loads(json_part)
                if isinstance(obj, dict):
                    result = _normalize_object(obj)
                    if result:
                        return result
            except json.JSONDecodeError:
                pass
            idx = brace + 1
        else:
            break

    return None


def parse_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract a tool call from model output. Returns None if unparseable."""
    if not isinstance(text, str):
        return None

    # Strategy 1: ```json ... ``` or ``` ... ``` code fence (highest priority)
    for m in re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL):
        result = _try_parse_body(m.group(1))
        if result:
            return result

    # Strategy 2: <tool_call>...</tool_call> tags
    for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        result = _try_parse_body(m.group(1))
        if result:
            return result

    # Strategy 3: bare body (no fences, no tags)
    return _try_parse_body(text)


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


def _prompt_user_text(prompt) -> str:
    """Extract the user-message text from a chat-format prompt list."""
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
    return ""


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

SYSTEM_PROMPT_GROUNDED = """You are an enterprise IT triage agent. You resolve support tickets using ONLY the retrieved context provided in the user message.

You MUST output your answer in this EXACT format — no other format is accepted:

```json
{"tool_name": "submit_resolution", "arguments": {"resolution": "plain text answer here", "cited_artifacts": ["KB-00001"], "confidence": 0.85, "escalate": false}}
```

Critical format rules — violations are penalised:
- The outer wrapper MUST be a ```json ... ``` code fence. Do NOT use <tool_call> tags, Python syntax, or any other format.
- "resolution" MUST be a plain string (not a JSON object, not a list, not nested). Write the resolution as prose.
- "cited_artifacts" MUST be a JSON array of string IDs (e.g. ["KB-00001"]). Cite ONLY IDs that appear in the Retrieved Context section.
- "confidence" MUST be a float between 0.0 and 1.0.
- "escalate" MUST be a boolean. Set to true only when the context is insufficient to resolve the ticket; in that case cite nothing.
- Output exactly ONE tool call. Do not search or fetch — all relevant context is already provided."""


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
    ONESHOT = """Example of correct output format:
    ```json
    {"tool_name": "submit_resolution", "arguments": {"resolution": "Verify TCP/179 reachability, check BGP timers, correct any AS or MD5 mismatches.", "cited_artifacts": ["KB-00001"], "confidence": 0.85, "escalate": false}}
    ```

    Now resolve THIS ticket:
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT_GROUNDED},
        {"role": "user", "content": (
            f"{ONESHOT} # Ticket: {ticket.get('ticket_id', ticket.get('id', ''))}\n"
            f"**Title:** {ticket['title']}\n"
            f"**Description:** {ticket['description']}\n\n"
            f"# Retrieved Context:\n{context_block}\n\n"
            "Resolve this ticket using ONLY the retrieved context. Output exactly one "
            "`submit_resolution` tool call as a JSON code block. "
            "`cited_artifacts` MUST list at least one ID from the Retrieved Context above "
            "(e.g. KB-XXXXX); an empty list [] is only valid when escalate=true."
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

# Fallback extractors used by the quality/citation rewards so that early in
# training, when the model is still using wrong field names like `details` or
# `resolution_steps`, the rewards aren't pinned at 0 — GRPO needs SOME
# gradient to learn the schema. Canonical keys are still preferred via a
# discount multiplier in the reward functions below.

_ID_PATTERN = re.compile(r'\b(?:KB|INC|TKT|TRAIN|TICKET)-\d+\b')

_RESOLUTION_FALLBACK_KEYS = (
    "details",
    "description",
    "solution",
    "resolution_steps",
    "resolution_notes",
    "notes",
    "answer",
    "text",
    "action",
    "steps_to_resolve",
    "summary",
)


def _extract_resolution_text(args: dict) -> Tuple[str, bool]:
    """Best-effort extraction of resolution prose. Returns (text, used_canonical_key)."""
    if not isinstance(args, dict):
        return "", False
    canonical = args.get("resolution")
    if isinstance(canonical, str) and canonical.strip():
        return canonical, True
    if isinstance(canonical, dict):
        nested = canonical.get("resolution") or canonical.get("text") or ""
        if isinstance(nested, str) and nested.strip():
            return nested, False
    for key in _RESOLUTION_FALLBACK_KEYS:
        v = args.get(key)
        if isinstance(v, str) and v.strip():
            return v, False
    string_vals = [v for v in args.values() if isinstance(v, str) and len(v.strip()) > 20]
    if string_vals:
        return max(string_vals, key=len), False
    return "", False


def _extract_citation_ids(args: dict, full_text: str) -> Tuple[Set[str], bool]:
    """Best-effort citation extraction. Returns (ids, used_canonical_field)."""
    if isinstance(args, dict):
        canonical = args.get("cited_artifacts")
        if isinstance(canonical, list):
            ids = {str(c) for c in canonical if isinstance(c, (str, int))}
            ids = {i for i in ids if _ID_PATTERN.fullmatch(i)}
            if ids:
                return ids, True
        if isinstance(canonical, str) and _ID_PATTERN.fullmatch(canonical):
            return {canonical}, True
        for v in args.values():
            if isinstance(v, list):
                ids = {str(x) for x in v if isinstance(x, str) and _ID_PATTERN.fullmatch(x)}
                if ids:
                    return ids, False
            elif isinstance(v, str) and _ID_PATTERN.fullmatch(v):
                return {v}, False
    return set(_ID_PATTERN.findall(full_text or "")), False


# ── Reward 1: Graduated format (5 partial-credit tiers) ──
def r_format_graduated(completions, **kwargs):
    """0.0 → 1.0 in five steps. Forces variance even when model fails to submit.

    Canonical format (```json fence) is rewarded more than alternative parseable
    formats at every tier so GRPO has a clear gradient toward the one true format.
    """
    out = []
    for c in completions:
        text = _completion_text(c)
        score = 0.0
        canonical = "```json" in text
        # Tier 1: any tool-call markup present
        # Canonical format gets 0.2; alternative formats (e.g. <tool_call>) get 0.1
        # so the model sees a reward difference from the very first token.
        if canonical:
            score = 0.2
        elif "<tool_call>" in text:
            score = 0.1
        parsed = parse_tool_call(text)
        if parsed is None:
            out.append(score)
            continue
        # Tier 2: parses to JSON
        score = 0.4 if canonical else 0.3

        # Detect malformed-but-parseable structures the system prompt bans.
        # Without these caps, the model gets 0.8 for emitting
        # `submit_resolution(submit_resolution({wrong_keys}))` — a strong false
        # positive that kills the gradient toward proper JSON-object form.
        py_call = bool(re.search(
            r'\b(?:' + '|'.join(KNOWN_TOOLS) + r')\s*\(', text))
        double_wrap = bool(re.search(
            r'submit_resolution\s*\([^)]*?submit_resolution\s*\(',
            text, re.DOTALL))

        # Tier 3: valid tool name
        if parsed["tool_name"] in KNOWN_TOOLS:
            score = 0.6 if canonical else 0.5
        # Tier 4: submit_resolution with a real `resolution` string
        # (Was unconditionally 0.8 — that's what made the format reward lie
        # about quality when the model used wrong keys.)
        if parsed["tool_name"] == "submit_resolution":
            args = parsed.get("arguments", {})
            has_resolution = (
                isinstance(args.get("resolution"), str)
                and len(args["resolution"].strip()) > 20
            )
            if has_resolution:
                score = 0.8 if canonical else 0.65
            # Tier 5: all required fields present, correct types, AND canonical format.
            req = {"resolution", "cited_artifacts", "confidence"}
            fields_ok = (
                has_resolution
                and req.issubset(args.keys())
                and isinstance(args.get("cited_artifacts"), list)
            )
            if fields_ok and canonical and not py_call:
                score = 1.0
            elif fields_ok:
                score = 0.85

        # Structural penalties — applied last so they cap whatever tier was reached.
        if double_wrap:
            score = min(score, 0.35)
        elif py_call:
            score = min(score, 0.5)

        out.append(score)
    return out


# ── Reward 2: Resolution quality (ROUGE-L vs gold) ──
_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def r_resolution_quality(completions, ticket_id, **kwargs):
    """Continuous text-similarity score. Drives most of the variance.

    Uses a fallback extractor: if the model wrote prose under the wrong key
    (`details`, `resolution_steps`, etc.) we still measure ROUGE against gold,
    but multiply by 0.6 so the canonical `resolution` field is strictly preferred.
    Without this fallback, the entire reward is 0 for every completion that
    doesn't already know the schema, which gives GRPO no gradient to learn it.
    """
    idx = get_ticket_index()
    out = []
    for c, tid in zip(completions, ticket_id):
        text = _completion_text(c)
        parsed = parse_tool_call(text)
        if parsed is None or parsed["tool_name"] != "submit_resolution":
            out.append(0.0); continue
        args = parsed.get("arguments", {})
        submitted, used_canonical = _extract_resolution_text(args)
        gold = idx.get(tid, {}).get("gold_resolution", "")
        gold = gold if isinstance(gold, str) else str(gold)
        if not submitted or not gold:
            out.append(0.0); continue
        f1 = _ROUGE.score(gold, submitted)["rougeL"].fmeasure
        if not used_canonical:
            f1 *= 0.6
        out.append(f1)
    return out


# ── Reward 3: Citation F1 vs gold ──
def r_citation_grounding(completions, ticket_id, prompts=None, **kwargs):
    """F1 of cited IDs vs gold, with a behavioral floor for context-grounded citations.

    Falls back to regex-extracting KB-/INC-/TKT- IDs from the completion text
    when the model omits `cited_artifacts` (common during cold-start). Fallback
    hits are multiplied by 0.6 so canonical `cited_artifacts` is preferred.

    Behavioral floor (requires `prompts` kwarg that TRL passes during training):
      - Cited an ID that appears in the Retrieved Context but isn't gold: 0.3
        This bootstraps non-zero gradient so GRPO can teach correct citing.
      - Cited an ID NOT in context (hallucinated): 0.05 — small signal, no reward.
      - Cited a gold ID: floor(0.3) + 0.7 * F1, so perfect citation reaches 1.0.
    When prompts is None (smoke-test without prompt data), falls back to plain F1.
    """
    idx = get_ticket_index()
    _prompts = list(prompts) if prompts is not None else [None] * len(completions)
    out = []
    for c, tid, pr in zip(completions, ticket_id, _prompts):
        text = _completion_text(c)
        parsed = parse_tool_call(text)
        ticket = idx.get(tid, {})
        gold = set(ticket.get("gold_cited_ids", []))
        is_unans = ticket.get("is_unanswerable", False)

        if parsed is None or parsed["tool_name"] != "submit_resolution":
            out.append(0.0); continue
        args = parsed.get("arguments", {})
        cited, used_canonical = _extract_citation_ids(args, text)

        if is_unans:
            esc = bool(args.get("escalate", False))
            out.append(1.0 if (not cited and esc) else 0.3 if not cited else 0.0)
            continue

        if not gold:
            esc = bool(args.get("escalate", False))
            out.append(1.0 if esc else 0.1)
            continue

        if not cited:
            out.append(0.0); continue

        # Extract IDs present in the retrieved context block from the prompt.
        # "### KB-00002" headers are the canonical marker build_training_prompt uses.
        ctx_ids = set(re.findall(r'### (\S+)', _prompt_user_text(pr))) if pr is not None else set()

        tp = len(cited & gold)
        if tp == 0:
            # Cited but didn't match gold — check if at least from context.
            if ctx_ids:
                cited_in_ctx = bool(cited & ctx_ids)
                out.append(0.3 if cited_in_ctx else 0.05)
            else:
                out.append(0.0)
            continue

        # tp > 0: at least one gold ID cited correctly.
        p_val = tp / len(cited)
        r_val = tp / len(gold)
        f1 = 2 * p_val * r_val / (p_val + r_val)
        if not used_canonical:
            f1 *= 0.6
        # Floor of 0.3 for citing from context; without prompts keep plain F1.
        if ctx_ids and (cited & ctx_ids):
            out.append(0.3 + 0.7 * f1)
        else:
            out.append(f1)
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
        gold = idx.get(tid, {}).get("gold_resolution", "")
        gold = gold if isinstance(gold, str) else str(gold)
        submitted, used_canonical = _extract_resolution_text(args)
        quality = _ROUGE.score(gold, submitted)["rougeL"].fmeasure if (gold and submitted) else 0.0
        if not used_canonical:
            quality *= 0.6
        out.append(1.0 - (conf - quality) ** 2)
    return out


# ── Reward 5: Length parsimony (anti-stub, anti-rambling) ──
def r_parsimony(completions, **kwargs):
    tids = list(kwargs.get("ticket_id") or [])
    out = []
    for i, c in enumerate(completions):
        text = _completion_text(c)
        parsed = parse_tool_call(text)
        if parsed is None:
            score = 0.0
        else:
            n = len(text.split())
            if n < 40:       score = 0.2   # stub
            elif n < 80:     score = 0.6
            elif n <= 400:   score = 1.0   # sweet spot
            elif n <= 800:   score = 0.6
            else:            score = 0.2   # rambling
        out.append(score)

        # Capture to disk buffer — r_parsimony runs last so all parse data is fresh.
        if i < len(tids):
            entry: dict = {
                "ticket_id": tids[i],
                "completion": text[:3000],
                "parsed": parsed is not None,
                "r_parsimony": round(score, 4),
            }
            if parsed and parsed["tool_name"] == "submit_resolution":
                args = parsed.get("arguments", {})
                res_text, _ = _extract_resolution_text(args)
                cited, _ = _extract_citation_ids(args, text)
                entry["resolution"] = res_text[:500] if res_text else None
                entry["cited_artifacts"] = sorted(cited) if cited else []
                entry["confidence"] = args.get("confidence")
                entry["escalate"] = args.get("escalate")
            _PENDING_COMPLETIONS.append(entry)

    return out


# ── CompletionDumper callback ─────────────────────────────────────────────────

class CompletionDumper(_TrainerCallback):
    """Saves completion batches from _PENDING_COMPLETIONS to JSONL every N steps.

    Each file is named completions/step_NNNN.jsonl and contains one JSON object
    per completion with ticket_id, full completion text, parse status, extracted
    resolution/citations, and the parsimony reward score.
    Use these files post-run to read actual model outputs and inspect quality.
    """

    def __init__(self, out_dir: Path, dump_every: int = 25):
        self.out_dir = Path(out_dir)
        self.dump_every = dump_every
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        if _PENDING_COMPLETIONS and state.global_step % self.dump_every == 0:
            self._flush(state.global_step)

    def on_train_end(self, args, state, control, **kwargs):
        if _PENDING_COMPLETIONS:
            self._flush(state.global_step)

    def _flush(self, step: int):
        global _PENDING_COMPLETIONS
        entries = _PENDING_COMPLETIONS[:]
        _PENDING_COMPLETIONS.clear()
        for e in entries:
            e.setdefault("step", step)
        path = self.out_dir / f"step_{step:04d}.jsonl"
        with open(path, "w") as fh:
            for entry in entries:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"  ✓ {len(entries)} completions → {path.name}")


# ── Smoke-test reward table ───────────────────────────────────────────────────

REWARD_WEIGHTS = [0.3, 1.5, 1.0, 0.5, 0.3]   # must match GRPOConfig reward_weights
_REWARD_WEIGHT_SUM = sum(REWARD_WEIGHTS)

_REWARD_FUNCS = [
    ("format",   r_format_graduated),
    ("quality",  r_resolution_quality),
    ("citation", r_citation_grounding),
    ("calib",    r_calibration),
    ("parsim",   r_parsimony),
]


def _smoke_generate(model_or_path, tickets: list, n: int = 8) -> List[tuple]:
    """Run fast greedy inference on the first *n* tickets.

    Returns a list of (ticket_id, completion_text, prompt_messages) triples.
    The prompt is included so print_reward_table can activate the citation floor.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
    import torch

    print(f"\n  Generating {n} smoke completions for reward table…")
    tokenizer = AutoTokenizer.from_pretrained(model_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    # Use a fresh GenerationConfig to avoid inheriting model's top_p/top_k
    # defaults which trigger "do_sample=False but top_p is set" warnings.
    gen_cfg = GenerationConfig(
        max_new_tokens=256,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    corpus = get_corpus()
    results = []
    for t in tickets[:n]:
        tid = t.get("ticket_id", t.get("id", ""))
        messages = build_training_prompt(t, corpus)
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids, generation_config=gen_cfg)
        new_tokens = out[0][input_ids.shape[-1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        results.append((tid, text, messages))
    return results


def print_reward_table(completions_by_ticket: List[tuple]) -> None:
    """Print a compact reward table for a batch of (ticket_id, completion[, prompt]) tuples.

    Columns: #  ticket_id  format  quality  citation  calib  parsim  TOTAL
    Accepts both 2-tuples (ticket_id, completion) and 3-tuples
    (ticket_id, completion, prompt). When prompts are present they are forwarded
    to r_citation_grounding so the behavioral floor activates.
    """
    if not completions_by_ticket:
        print("  (no completions to score)")
        return

    tids         = [t[0] for t in completions_by_ticket]
    completions  = [t[1] for t in completions_by_ticket]
    prompts_list = [t[2] if len(t) > 2 else None for t in completions_by_ticket]

    raw: Dict[str, list] = {}
    for name, fn in _REWARD_FUNCS:
        try:
            if name == "citation":
                raw[name] = fn(completions, ticket_id=tids, prompts=prompts_list)
            elif name in ("quality", "calib"):
                raw[name] = fn(completions, ticket_id=tids)
            else:
                raw[name] = fn(completions)
        except Exception as e:
            raw[name] = [0.0] * len(completions)
            print(f"  [warn] {name} reward failed: {e}")

    n_cols = len(_REWARD_FUNCS)
    names  = [n for n, _ in _REWARD_FUNCS]

    # Weighted total per completion
    totals = []
    for i in range(len(completions)):
        total = sum(
            raw[names[j]][i] * REWARD_WEIGHTS[j]
            for j in range(n_cols)
        ) / _REWARD_WEIGHT_SUM
        totals.append(total)

    # ── Header ────────────────────────────────────────────────────────────────
    COL_W = 9
    TID_W = 14
    hdr = f"  {'#':<3}  {'ticket_id':<{TID_W}}"
    for n in names:
        hdr += f"  {n:>{COL_W}}"
    hdr += f"  {'TOTAL':>{COL_W}}"
    sep = "  " + "-" * (len(hdr) - 2)

    print()
    print("  ┌─ Smoke-test reward summary " + "─" * max(0, len(sep) - 30) + "┐")
    print(hdr)
    print(sep)

    col_sums = {n: 0.0 for n in names}
    for i, (tid, total) in enumerate(zip(tids, totals)):
        row = f"  {i+1:<3}  {tid:<{TID_W}}"
        for n in names:
            v = raw[n][i]
            col_sums[n] += v
            row += f"  {v:>{COL_W}.3f}"
        row += f"  {total:>{COL_W}.3f}"
        print(row)

    # Mean row
    print(sep)
    mean_total = sum(totals) / len(totals)
    mean_row = f"  {'avg':<3}  {'':< {TID_W}}"
    for n in names:
        mean_row += f"  {col_sums[n]/len(completions):>{COL_W}.3f}"
    mean_row += f"  {mean_total:>{COL_W}.3f}"
    print(mean_row)
    print("  └" + "─" * (len(sep) - 3) + "┘")
    print()


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


def save_per_ticket_heatmap(completions_by_ticket: List[tuple], output_path: Path):
    """Heatmap: rows = tickets, cols = reward components + TOTAL.

    Color encodes score (0=red, 1=green). Lets you instantly see which tickets
    the model handles well and which rewards are bottlenecks per ticket.
    """
    if not completions_by_ticket:
        return

    tids         = [t[0] for t in completions_by_ticket]
    completions  = [t[1] for t in completions_by_ticket]
    prompts_list = [t[2] if len(t) > 2 else None for t in completions_by_ticket]

    names = [n for n, _ in _REWARD_FUNCS]
    all_raw: Dict[str, list] = {}
    for name, fn in _REWARD_FUNCS:
        try:
            if name == "citation":
                all_raw[name] = fn(completions, ticket_id=tids, prompts=prompts_list)
            elif name in ("quality", "calib"):
                all_raw[name] = fn(completions, ticket_id=tids)
            else:
                all_raw[name] = fn(completions)
        except Exception:
            all_raw[name] = [0.0] * len(completions)

    scores_matrix = []
    for i in range(len(completions)):
        row = [all_raw[n][i] for n in names]
        total = sum(row[j] * REWARD_WEIGHTS[j] for j in range(len(names))) / _REWARD_WEIGHT_SUM
        scores_matrix.append(row + [total])

    col_labels = names + ["TOTAL"]
    data = np.array(scores_matrix)

    fig, ax = plt.subplots(figsize=(max(9, len(col_labels) * 1.6), max(4, len(tids) * 0.7)))
    im = ax.imshow(data, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(tids)))
    ax.set_yticklabels([t[:14] for t in tids], fontsize=9)
    for i in range(len(tids)):
        for j in range(len(col_labels)):
            v = data[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if 0.25 < v < 0.80 else "white")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    ax.set_title("Per-Ticket Reward Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Reward heatmap saved → {output_path}")


def save_calibration_scatter(completions_by_ticket: List[tuple], output_path: Path):
    """Scatter of model confidence vs ROUGE-L quality.

    Points on the diagonal = perfectly calibrated. Points above = overconfident,
    below = underconfident. Useful for understanding if the model knows when it's wrong.
    """
    if not completions_by_ticket:
        return

    tids        = [t[0] for t in completions_by_ticket]
    completions = [t[1] for t in completions_by_ticket]
    idx = get_ticket_index()

    xs, ys, labels = [], [], []
    for comp, tid in zip(completions, tids):
        text = _completion_text(comp)
        parsed = parse_tool_call(text)
        if parsed is None or parsed["tool_name"] != "submit_resolution":
            continue
        args = parsed.get("arguments", {})
        try:
            conf = float(args.get("confidence", 0.5))
            conf = max(0.0, min(1.0, conf))
        except Exception:
            continue
        gold = idx.get(tid, {}).get("gold_resolution", "")
        submitted, _ = _extract_resolution_text(args)
        if not submitted or not gold:
            continue
        quality = _ROUGE.score(gold, submitted)["rougeL"].fmeasure
        xs.append(conf)
        ys.append(quality)
        labels.append(tid)

    if not xs:
        print("  (no valid completions for calibration scatter — skipping)")
        return

    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(xs, ys, c=range(len(xs)), cmap="tab10", s=90, alpha=0.85, zorder=3)
    for x, y, lbl in zip(xs, ys, labels):
        ax.annotate(lbl[:12], (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=7, alpha=0.8)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.35, linewidth=1.5, label="perfect calibration")
    ax.set_xlabel("Model confidence", fontsize=12)
    ax.set_ylabel("ROUGE-L quality", fontsize=12)
    ax.set_title("Confidence vs Quality (calibration)", fontsize=13, fontweight="bold")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.04, label="ticket index")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Calibration scatter saved → {output_path}")


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
        wandb.init(project="triage-agent-grpo", name=f"qwen3b-{'smoke' if smoke else 'full'}-v4")
        report_to = "wandb"
        print("wandb logging enabled.")
    except Exception as e:
        print(f"wandb not available ({e})")

    from trl import GRPOConfig, GRPOTrainer

    warmup_steps = max(1, int(max_steps * 0.1))

    vllm_kwargs = dict(
        use_vllm=True,
        vllm_gpu_memory_utilization=0.4,
    ) if use_vllm else dict(use_vllm=False)

    extra_kwargs = {}
    try:
        import inspect
        from trl import GRPOConfig as _check
        sig = inspect.signature(_check)
        if "log_completions" in sig.parameters:
            extra_kwargs["log_completions"] = True
            extra_kwargs["num_completions_to_print"] = 0  # log to wandb only, skip console table
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

    # Save completions to disk every 25 steps for post-run inspection.
    completion_dumper = CompletionDumper(
        out_dir=OUTPUT_DIR / "completions",
        dump_every=25 if not smoke else 1,
    )
    trainer.add_callback(completion_dumper)

    try:
        print("\nStarting training…")
        trainer.train()
        trainer.save_model()
        print(f"\n✓ Training complete. Model saved → {OUTPUT_DIR}")
        save_reward_curve(trainer.state.log_history, PLOTS_DIR / "reward_curve.png")

        # Post-training eval: reward table + heatmap + calibration scatter.
        # Runs whether --smoke-test or full run so you always get fresh plots.
        try:
            n_eval = min(8, len(train_tickets))
            eval_pairs = _smoke_generate(str(OUTPUT_DIR), train_tickets, n=n_eval)
            print_reward_table(eval_pairs)
            save_per_ticket_heatmap(eval_pairs, PLOTS_DIR / "reward_heatmap.png")
            save_calibration_scatter(eval_pairs, PLOTS_DIR / "calibration_scatter.png")
        except Exception as e:
            print(f"  [warn] post-training eval failed: {e}")

        # Auto-commit plots if in a git repo
        for cmd in [
            ["git", "add", "assets/plots/"],
            ["git", "commit", "-m", f"feat: GRPO plots ({max_steps} steps)"],
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