"""
Reward functions for TriageAgentEnvironment.

All functions accept a mutable _EpisodeState (or any object with the same attrs).
judge_resolution uses pure-Python token overlap F1 — no API calls.
"""
import re
from typing import Any, Dict, List, Optional


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _token_set(text: str) -> set:
    return set(re.findall(r'[a-z0-9]+', text.lower()))


def judge_resolution(gold: str, submitted: str) -> float:
    """Token overlap F1 between gold and submitted resolution strings."""
    if not gold or not submitted:
        return 0.0
    gold_toks = _token_set(gold)
    sub_toks = _token_set(submitted)
    if not gold_toks or not sub_toks:
        return 0.0
    tp = len(gold_toks & sub_toks)
    precision = tp / len(sub_toks)
    recall = tp / len(gold_toks)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def citation_f1(
    gold: List[str],
    submitted: List[str],
    viewed: Optional[List[str]] = None,
) -> float:
    """Citation F1. Submitted citations must have been viewed to count."""
    if viewed is not None:
        viewed_set = set(viewed)
        submitted = [c for c in submitted if c in viewed_set]
    gold_set = set(gold)
    sub_set = set(submitted)
    if not gold_set and not sub_set:
        return 1.0
    if not gold_set or not sub_set:
        return 0.0
    tp = len(gold_set & sub_set)
    precision = tp / len(sub_set)
    recall = tp / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ------------------------------------------------------------------ #
# Reward functions                                                     #
# ------------------------------------------------------------------ #

def primary_reward(state: Any) -> float:
    """Binary: 1.0 iff resolution is correct AND well-cited."""
    if not state.submitted:
        return 0.0
    if state.submitted_escalate and state.is_unanswerable:
        return 1.0
    if state.submitted_escalate and not state.is_unanswerable:
        return 0.0
    correctness = judge_resolution(
        gold=state.gold_resolution,
        submitted=state.submitted_resolution or "",
    )
    grounding = citation_f1(
        gold=state.gold_cited_ids,
        submitted=state.submitted_citations,
        viewed=state.artifacts_viewed,
    )
    return float(correctness > 0.7 and grounding > 0.5)


def reward_grounding(state: Any) -> float:
    """Partial credit for citation F1 (even if primary failed)."""
    if not state.submitted:
        return 0.0
    return 0.3 * citation_f1(
        gold=state.gold_cited_ids,
        submitted=state.submitted_citations,
        viewed=state.artifacts_viewed,
    )


def reward_efficiency(state: Any) -> float:
    """Small bonus for not over-searching relative to ticket difficulty.
    Requires at least one search or fetch — zero ops earns no credit."""
    if not state.submitted or state.submitted_escalate:
        return 0.0
    total_ops = state.searches_made + state.fetches_made
    if total_ops == 0:
        return 0.0  # answered without looking at anything — no efficiency credit
    ideal = {"easy": 3, "medium": 6, "hard": 10}.get(state.difficulty, 6)
    return 0.2 * max(0.0, 1.0 - max(0, total_ops - ideal) / ideal)


def reward_calibration(state: Any) -> float:
    """Brier-style bonus for calibrated confidence."""
    if not state.submitted:
        return 0.0
    c = state.submitted_confidence if state.submitted_confidence is not None else 0.5
    correct = primary_reward(state)
    return 0.15 * max(0.0, 1.0 - 2 * (c - correct) ** 2)


def reward_format(state: Any) -> float:
    """Tiny bonus for a non-empty, well-formed submission."""
    if state.submitted and state.submitted_resolution:
        return 0.05
    return 0.0


def compute_total_reward(state: Any) -> float:
    return (
        primary_reward(state)
        + reward_grounding(state)
        + reward_efficiency(state)
        + reward_calibration(state)
        + reward_format(state)
    )


def reward_breakdown(state: Any) -> Dict[str, float]:
    return {
        "primary": primary_reward(state),
        "grounding": reward_grounding(state),
        "efficiency": reward_efficiency(state),
        "calibration": reward_calibration(state),
        "format": reward_format(state),
        "total": compute_total_reward(state),
    }
