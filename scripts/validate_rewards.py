#!/usr/bin/env python3
"""
scripts/validate_rewards.py  —  Anti-gaming reward validation (§4.3).

Three cheating agents, each run against every training ticket.
Asserts each cheater's mean total reward ≤ 0.35.
If a cheater exceeds the threshold, prints the offending tickets and why.

Cheaters:
  AlwaysEscalate   — submit_resolution(escalate=True) on turn 1, no searches.
  DumpEverything   — fetch every artifact in the corpus, then submit with all
                     cited (destroys citation precision).
  AnswerImmediately — submit_resolution on turn 1, no searches, empty citations.

Usage:
    uv run python scripts/validate_rewards.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from server.corpus import Corpus          # noqa: E402
from server import rewards as R           # noqa: E402

THRESHOLD = 0.35
DATA_DIR = ROOT / "data"


# ------------------------------------------------------------------ #
# Minimal mutable state (mirrors _EpisodeState, no env dependency)    #
# ------------------------------------------------------------------ #

class _State:
    __slots__ = (
        "episode_id", "step_count",
        "target_ticket_id", "gold_resolution", "gold_cited_ids",
        "difficulty", "is_unanswerable",
        "tools_called", "artifacts_viewed",
        "searches_made", "fetches_made",
        "submitted", "submitted_resolution", "submitted_citations",
        "submitted_confidence", "submitted_escalate",
    )

    def __init__(self, ticket: dict):
        self.episode_id = ticket.get("ticket_id", "")
        self.step_count = 0
        self.target_ticket_id = ticket.get("ticket_id", "")
        self.gold_resolution = ticket.get("gold_resolution", "")
        self.gold_cited_ids = list(ticket.get("gold_cited_ids", []))
        self.difficulty = ticket.get("difficulty", "medium")
        self.is_unanswerable = ticket.get("is_unanswerable", False)
        self.tools_called = []
        self.artifacts_viewed = []
        self.searches_made = 0
        self.fetches_made = 0
        self.submitted = False
        self.submitted_resolution = None
        self.submitted_citations = []
        self.submitted_confidence = 0.5
        self.submitted_escalate = False


# ------------------------------------------------------------------ #
# Cheater strategies                                                   #
# ------------------------------------------------------------------ #

def always_escalate(ticket: dict, corpus: Corpus) -> dict:
    """Turn 1: submit_resolution(escalate=True). No artifact access."""
    s = _State(ticket)
    s.step_count = 1
    s.submitted = True
    s.submitted_resolution = ""
    s.submitted_citations = []
    s.submitted_confidence = 0.5
    s.submitted_escalate = True
    return R.reward_breakdown(s)


def dump_everything(ticket: dict, corpus: Corpus) -> dict:
    """Fetch every artifact in the corpus, cite all of them, submit gold resolution.
    Maximally adversarial: resolution quality is perfect but citation precision
    is destroyed (~1-4 gold IDs out of 235 total → F1 < 0.05)."""
    s = _State(ticket)

    all_ids: list = []
    for a in corpus._kb:
        aid = a.get("article_id", "")
        if aid:
            all_ids.append(aid)
            s.fetches_made += 1
    for t in corpus._tickets:
        tid = t.get("ticket_id", "")
        if tid:
            all_ids.append(tid)
            s.fetches_made += 1
    for inc in corpus._incidents:
        iid = inc.get("incident_id", "")
        if iid:
            all_ids.append(iid)
            s.fetches_made += 1

    s.artifacts_viewed = list(all_ids)
    s.step_count = s.fetches_made

    s.submitted = True
    # Use gold resolution so correctness is maximal — the only thing that
    # can defeat this cheater is citation precision.
    s.submitted_resolution = ticket.get("gold_resolution", "resolved")
    s.submitted_citations = list(all_ids)   # cite everything → precision ≈ 0
    s.submitted_confidence = 0.5
    s.submitted_escalate = False
    return R.reward_breakdown(s)


def answer_immediately(ticket: dict, corpus: Corpus) -> dict:
    """Turn 1: submit with a generic resolution, no searches, empty citations."""
    s = _State(ticket)
    s.step_count = 1
    s.submitted = True
    s.submitted_resolution = "The issue has been resolved."
    s.submitted_citations = []
    s.submitted_confidence = 0.5
    s.submitted_escalate = False
    return R.reward_breakdown(s)


# ------------------------------------------------------------------ #
# Runner                                                               #
# ------------------------------------------------------------------ #

CHEATERS = [
    ("AlwaysEscalate",    always_escalate),
    ("DumpEverything",    dump_everything),
    ("AnswerImmediately", answer_immediately),
]

SUB_KEYS = ["primary", "grounding", "efficiency", "calibration", "format", "total"]


def run_cheater(name: str, fn, tickets: list, corpus: Corpus) -> dict:
    per_ticket = [(t, fn(t, corpus)) for t in tickets]
    n = len(per_ticket)
    means = {k: sum(r[k] for _, r in per_ticket) / n for k in SUB_KEYS} if n else {k: 0.0 for k in SUB_KEYS}
    over = [(t, r) for t, r in per_ticket if r["total"] > THRESHOLD]
    return {"name": name, "means": means, "n": n, "over_threshold": over}


def _why(ticket: dict, breakdown: dict) -> str:
    parts = []
    if breakdown["primary"] > 0:
        if ticket.get("is_unanswerable"):
            parts.append("correct escalation on unanswerable ticket")
        else:
            parts.append(f"primary={breakdown['primary']:.2f}")
    if breakdown["grounding"] > 0.05:
        parts.append(f"grounding={breakdown['grounding']:.3f}")
    if breakdown["efficiency"] > 0.05:
        parts.append(f"efficiency={breakdown['efficiency']:.3f}")
    if breakdown["calibration"] > 0.05:
        parts.append(f"calibration={breakdown['calibration']:.3f}")
    return ", ".join(parts) if parts else "sub-rewards accumulated"


def print_result(result: dict) -> bool:
    name = result["name"]
    means = result["means"]
    n = result["n"]
    over = result["over_threshold"]
    passed = means["total"] <= THRESHOLD

    print(f"\n{'─'*62}")
    print(f"  {name:<22}  n={n}  threshold={THRESHOLD}")
    print(f"{'─'*62}")
    for k in SUB_KEYS[:-1]:
        print(f"    {k:<14}: {means[k]:.4f}")
    mark = "✓ PASS" if passed else "✗ FAIL"
    print(f"    {'TOTAL':<14}: {means['total']:.4f}   {mark}")

    if not passed:
        print(f"\n  Tickets driving the mean above {THRESHOLD} ({len(over)} total):")
        for t, r in over[:30]:
            tid = t.get("ticket_id", "?")
            diff = t.get("difficulty", "?")
            tag = " [unanswerable]" if t.get("is_unanswerable") else ""
            why = _why(t, r)
            print(f"    {tid} [{diff}]{tag}  total={r['total']:.3f}  — {why}")
        if len(over) > 30:
            print(f"    ... and {len(over) - 30} more")

    return passed


def main() -> None:
    corpus = Corpus(DATA_DIR)
    tickets = corpus.train_tickets

    if not tickets:
        print("ERROR: No training tickets found. Add data/train_tickets.json first.")
        sys.exit(1)

    n_unans = sum(1 for t in tickets if t.get("is_unanswerable"))
    print(f"Corpus loaded: {len(corpus._kb)} KB articles, "
          f"{len(corpus._tickets)} past tickets, "
          f"{len(corpus._incidents)} incidents")
    print(f"Training tickets: {len(tickets)} total, {n_unans} unanswerable "
          f"({100*n_unans/len(tickets):.0f}%)")
    print(f"\nRunning anti-gaming validation — threshold: mean ≤ {THRESHOLD}")

    results = [run_cheater(name, fn, tickets, corpus) for name, fn in CHEATERS]

    all_pass = True
    for r in results:
        if not print_result(r):
            all_pass = False

    print(f"\n{'═'*62}")
    if all_pass:
        print("  ALL CHEATERS BELOW THRESHOLD — reward design is anti-gaming.")
        print(f"{'═'*62}")
    else:
        print("  SOME CHEATERS EXCEEDED THRESHOLD.")
        print("  Adjust reward weights in server/rewards.py before training.")
        print(f"{'═'*62}")
        sys.exit(1)


if __name__ == "__main__":
    main()
