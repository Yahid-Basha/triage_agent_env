#!/usr/bin/env python3
"""
Triage Agent inference script.

Loads yahid/triage-agent-qwen3b and runs it against the TriageAgentEnvironment
server via HTTP. Zero dependency on server.* or openenv.core.

Environment variables:
    OPENENV_BASE_URL   Server URL (default: http://localhost:8000)
    HF_MODEL_ID        Model to load (default: yahid/triage-agent-qwen3b)
    HF_TOKEN           HuggingFace token for private models

Usage:
    OPENENV_BASE_URL=http://localhost:8000 python inference.py
    python inference.py --max-episodes 5
    python inference.py --model yahid/triage-agent-qwen3b
"""

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ── Config ────────────────────────────────────────────────────────────────────
ROOT             = Path(__file__).parent
DATA_DIR         = ROOT / "data"
BASE_URL         = os.getenv("OPENENV_BASE_URL", "http://localhost:8000").rstrip("/")
MODEL_NAME       = os.getenv("HF_MODEL_ID", "yahid/triage-agent-qwen3b")
HF_TOKEN         = os.getenv("HF_TOKEN")
SUCCESS_THRESHOLD = 0.5
BENCHMARK        = "triage_agent_env"

# ── System prompt (must match training v4) ────────────────────────────────────
SYSTEM_PROMPT = """You are an enterprise IT triage agent. You resolve support tickets using ONLY the retrieved context provided in the user message.

You MUST output your answer in this EXACT format — no other format is accepted:

```json
{"tool_name": "submit_resolution", "arguments": {"resolution": "plain text answer here", "cited_artifacts": ["KB-00001"], "confidence": 0.85, "escalate": false}}
```

Critical format rules:
- The outer wrapper MUST be a ```json ... ``` code fence.
- "resolution" MUST be a plain string.
- "cited_artifacts" MUST be a JSON array of string IDs from the Retrieved Context only.
- "confidence" MUST be a float 0.0–1.0.
- "escalate" MUST be a boolean. True only when context is insufficient.
- Output exactly ONE tool call."""

ONESHOT = """Example of correct output format:
```json
{"tool_name": "submit_resolution", "arguments": {"resolution": "Verify TCP/179 reachability, check BGP timers, correct any AS or MD5 mismatches.", "cited_artifacts": ["KB-00001"], "confidence": 0.85, "escalate": false}}
```

Now resolve THIS ticket:
"""

KNOWN_TOOLS = {
    "search_kb", "search_tickets", "search_incidents",
    "get_article", "get_ticket", "get_incident", "submit_resolution",
}


# ── Lightweight corpus (no server.* import) ───────────────────────────────────

class _Corpus:
    """Minimal in-memory KB for context retrieval — reads JSON directly."""

    def __init__(self, data_dir: Path):
        self._articles: List[dict] = []
        self._article_index: Dict[str, dict] = {}

        kb_path = data_dir / "kb.json"
        if kb_path.exists():
            with open(kb_path) as f:
                raw = json.load(f)
            # support both list and {"articles": [...]} shapes
            items = raw if isinstance(raw, list) else raw.get("articles", [])
            for a in items:
                self._articles.append(a)
                aid = a.get("article_id") or a.get("id", "")
                if aid:
                    self._article_index[aid] = a

    def get_article(self, aid: str) -> Optional[dict]:
        return self._article_index.get(aid)

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        """TF-IDF cosine similarity over title + body."""
        if not self._articles:
            return []
        q_toks = _tokenize(query)
        if not q_toks:
            return self._articles[:top_k]

        q_freq = {t: q_toks.count(t) for t in set(q_toks)}
        n_docs = len(self._articles)
        scores = []
        for a in self._articles:
            text = f"{a.get('title', '')} {a.get('body', a.get('content', ''))}"
            d_toks = _tokenize(text)
            d_freq = {t: d_toks.count(t) for t in set(d_toks)}
            overlap = sum(q_freq.get(t, 0) * d_freq.get(t, 0) for t in q_freq)
            scores.append((overlap, a))
        scores.sort(key=lambda x: x[0], reverse=True)
        results = [a for sc, a in scores if sc > 0]
        return results[:top_k] if results else self._articles[:top_k]


def _tokenize(text: str) -> List[str]:
    return re.findall(r'\b[a-z0-9]+\b', text.lower())


# ── Tool-call parser (balanced-brace, handles all 4 Qwen formats) ─────────────

def _extract_balanced_json(s: str, start: int) -> Optional[str]:
    if start >= len(s) or s[start] != '{':
        return None
    depth, in_str, escape = 0, False, False
    for i in range(start, len(s)):
        c = s[i]
        if escape:
            escape = False; continue
        if c == '\\':
            escape = True; continue
        if c == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def _normalize(name: str, args: Any) -> Optional[Dict]:
    if name not in KNOWN_TOOLS:
        return None
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            args = {}
    return {"tool_name": name, "arguments": args if isinstance(args, dict) else {}}


def _normalize_object(obj: Dict) -> Optional[Dict]:
    name = obj.get("tool_name") or obj.get("name")
    if isinstance(name, str) and name in KNOWN_TOOLS:
        return _normalize(name, obj.get("arguments") or obj.get("parameters") or {})
    fn = obj.get("function")
    if isinstance(fn, dict):
        n = fn.get("name")
        if isinstance(n, str) and n in KNOWN_TOOLS:
            return _normalize(n, fn.get("arguments", {}))
    for tool in KNOWN_TOOLS:
        if tool in obj and isinstance(obj[tool], dict):
            return _normalize(tool, obj[tool])
    return None


def _try_parse_body(body: str) -> Optional[Dict]:
    m = re.search(r'\b(' + '|'.join(KNOWN_TOOLS) + r')\s*\(\s*\{', body)
    if m:
        jp = _extract_balanced_json(body, m.end() - 1)
        if jp:
            try:
                r = _normalize(m.group(1), json.loads(jp))
                if r:
                    return r
            except json.JSONDecodeError:
                pass
    idx = 0
    while idx < len(body):
        brace = body.find('{', idx)
        if brace == -1:
            break
        jp = _extract_balanced_json(body, brace)
        if jp:
            try:
                obj = json.loads(jp)
                if isinstance(obj, dict):
                    r = _normalize_object(obj)
                    if r:
                        return r
            except json.JSONDecodeError:
                pass
            idx = brace + 1
        else:
            break
    return None


def parse_tool_call(text: str) -> Optional[Dict]:
    if not isinstance(text, str):
        return None
    for m in re.finditer(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL):
        r = _try_parse_body(m.group(1))
        if r:
            return r
    for m in re.finditer(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        r = _try_parse_body(m.group(1))
        if r:
            return r
    return _try_parse_body(text)


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(ticket: dict, corpus: _Corpus) -> List[Dict]:
    # Gold articles
    gold_articles = [
        corpus.get_article(aid)
        for aid in ticket.get("gold_cited_ids", [])
        if corpus.get_article(aid)
    ]
    # Distractors via search
    hits = corpus.search(ticket.get("title", ""), top_k=6)
    gold_ids = set(ticket.get("gold_cited_ids", []))
    distractors = [a for a in hits if a.get("article_id", a.get("id", "")) not in gold_ids][:3]

    context_items = gold_articles + distractors
    random.shuffle(context_items)

    context_block = "\n\n".join(
        f"### {a.get('article_id', a.get('id', ''))}\n{a.get('title', '')}\n"
        f"{a.get('body', a.get('content', ''))[:1000]}"
        for a in context_items
    )

    tid = ticket.get("ticket_id", ticket.get("id", ""))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"{ONESHOT} # Ticket: {tid}\n"
            f"**Title:** {ticket['title']}\n"
            f"**Description:** {ticket['description']}\n\n"
            f"# Retrieved Context:\n{context_block}\n\n"
            "Resolve this ticket using ONLY the retrieved context. "
            "Output exactly one `submit_resolution` tool call as a JSON code block. "
            "`cited_artifacts` MUST list at least one ID from the Retrieved Context above; "
            "an empty list [] is only valid when escalate=true."
        )},
    ]


# ── Logging ───────────────────────────────────────────────────────────────────

def log_start(ticket_id: str, model: str) -> None:
    print(f"[START] task={ticket_id} env={BENCHMARK} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.4f} done={str(done).lower()} error={err}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    r_str = ",".join(f"{r:.4f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={r_str}", flush=True)


# ── HTTP env client ───────────────────────────────────────────────────────────

def env_reset(session) -> dict:
    resp = session.post(f"{BASE_URL}/reset", json={}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def env_step(session, action: dict) -> dict:
    resp = session.post(f"{BASE_URL}/step", json={"action": action}, timeout=30)
    resp.raise_for_status()
    return resp.json()


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name} …", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN, trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
        token=HF_TOKEN,
        trust_remote_code=True,
    )
    mdl.eval()
    print(f"Loaded on {device}.", flush=True)
    return mdl, tok


def generate(model, tokenizer, messages: List[Dict], max_new_tokens: int = 512) -> str:
    import torch
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


# ── Episode ───────────────────────────────────────────────────────────────────

def run_episode(ticket: dict, corpus: _Corpus, session, model, tokenizer) -> float:
    import requests

    tid = ticket.get("ticket_id", ticket.get("id", "UNKNOWN"))
    log_start(ticket_id=tid, model=MODEL_NAME)

    rewards: List[float] = []
    step = 0
    score = 0.0
    success = False

    try:
        env_reset(session)  # init server-side episode

        messages  = build_prompt(ticket, corpus)
        error_msg = None
        action_payload: dict

        try:
            raw    = generate(model, tokenizer, messages)
            parsed = parse_tool_call(raw)

            if parsed is None:
                parsed    = {"tool_name": "submit_resolution",
                             "arguments": {"resolution": "", "cited_artifacts": [],
                                           "confidence": 0.1, "escalate": True}}
                error_msg = "parse_failed"

            args = parsed["arguments"]
            action_payload = {
                "tool_name":       parsed["tool_name"],
                "query":           args.get("query"),
                "resolution":      args.get("resolution"),
                "cited_artifacts": args.get("cited_artifacts"),
                "confidence":      args.get("confidence"),
                "escalate":        args.get("escalate", False),
                "max_results":     args.get("max_results"),
            }
            # drop None fields to keep payload clean
            action_payload = {k: v for k, v in action_payload.items() if v is not None}

        except Exception as e:
            error_msg     = str(e).replace("\n", " ")[:120]
            action_payload = {"tool_name": "submit_resolution",
                              "resolution": "", "cited_artifacts": [],
                              "confidence": 0.1, "escalate": True}

        obs    = env_step(session, action_payload)
        step   = 1
        reward = obs.get("reward") or 0.0
        done   = obs.get("done", True)
        rewards.append(reward)

        log_step(step=step,
                 action=json.dumps({k: v for k, v in action_payload.items() if k != "resolution"}),
                 reward=reward,
                 done=done,
                 error=error_msg)

        score   = round(sum(rewards) / len(rewards), 4)
        success = score >= SUCCESS_THRESHOLD

    except Exception as outer:
        print(f"  Episode error: {outer}", flush=True)
    finally:
        log_end(success=success, steps=step, score=score, rewards=rewards)

    return score


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global MODEL_NAME

    parser = argparse.ArgumentParser()
    parser.add_argument("--model",        default=MODEL_NAME)
    parser.add_argument("--max-episodes", type=int, default=None)
    cli = parser.parse_args()
    MODEL_NAME = cli.model

    import requests

    # Verify server
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=10)
        r.raise_for_status()
        print(f"Server healthy: {BASE_URL}", flush=True)
    except Exception as e:
        print(f"Server not reachable at {BASE_URL}: {e}", flush=True)
        sys.exit(1)

    corpus = _Corpus(DATA_DIR)

    eval_path = DATA_DIR / "eval_tickets.json"
    with open(eval_path) as f:
        tickets = json.load(f)
    if cli.max_episodes:
        tickets = tickets[:cli.max_episodes]

    print(f"Eval tickets: {len(tickets)}", flush=True)
    model, tokenizer = load_model(MODEL_NAME)

    session = requests.Session()
    scores  = []
    for ticket in tickets:
        s = run_episode(ticket, corpus, session, model, tokenizer)
        scores.append(s)
        print(f"  → score={s:.4f}  avg={sum(scores)/len(scores):.4f}", flush=True)

    mean = sum(scores) / len(scores) if scores else 0.0
    print(f"\nFinal mean score: {mean:.4f} over {len(scores)} episodes", flush=True)


if __name__ == "__main__":
    main()
