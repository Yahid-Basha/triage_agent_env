"""
Tool handlers for TriageAgentEnvironment.

Each function takes (state, corpus, **kwargs) and returns a dict.
`state` is a mutable _EpisodeState instance (plain Python object).
"""
from typing import Any, Dict, List, Optional


def search_kb(state: Any, corpus: Any, query: str = "", max_results: int = 5) -> Dict:
    state.searches_made += 1
    state.tools_called.append(("search_kb", {"query": query, "max_results": max_results}))
    results = corpus.search_kb(query, max_results)
    return {"results": results}


def get_article(state: Any, corpus: Any, article_id: str = "") -> Dict:
    state.fetches_made += 1
    state.tools_called.append(("get_article", {"article_id": article_id}))
    if article_id and article_id not in state.artifacts_viewed:
        state.artifacts_viewed.append(article_id)
    article = corpus.get_article(article_id)
    if article is None:
        return {"error": f"Article '{article_id}' not found"}
    return article


def search_tickets(
    state: Any, corpus: Any, query: str = "", status: Optional[str] = None, max_results: int = 5
) -> Dict:
    state.searches_made += 1
    state.tools_called.append(("search_tickets", {"query": query, "status": status, "max_results": max_results}))
    results = corpus.search_tickets(query, status=status, max_results=max_results)
    return {"results": results}


def get_ticket(state: Any, corpus: Any, ticket_id: str = "") -> Dict:
    state.fetches_made += 1
    state.tools_called.append(("get_ticket", {"ticket_id": ticket_id}))
    if ticket_id and ticket_id not in state.artifacts_viewed:
        state.artifacts_viewed.append(ticket_id)
    ticket = corpus.get_ticket(ticket_id)
    if ticket is None:
        return {"error": f"Ticket '{ticket_id}' not found"}
    return ticket


def search_incidents(state: Any, corpus: Any, query: str = "", max_results: int = 3) -> Dict:
    state.searches_made += 1
    state.tools_called.append(("search_incidents", {"query": query, "max_results": max_results}))
    results = corpus.search_incidents(query, max_results)
    return {"results": results}


def get_incident(state: Any, corpus: Any, incident_id: str = "") -> Dict:
    state.fetches_made += 1
    state.tools_called.append(("get_incident", {"incident_id": incident_id}))
    if incident_id and incident_id not in state.artifacts_viewed:
        state.artifacts_viewed.append(incident_id)
    incident = corpus.get_incident(incident_id)
    if incident is None:
        return {"error": f"Incident '{incident_id}' not found"}
    return incident


def submit_resolution(
    state: Any,
    corpus: Any,
    resolution: str = "",
    cited_artifacts: Optional[List[str]] = None,
    confidence: Optional[float] = None,
    escalate: bool = False,
) -> Dict:
    state.submitted = True
    state.submitted_resolution = resolution
    state.submitted_citations = list(cited_artifacts or [])
    state.submitted_confidence = confidence
    state.submitted_escalate = escalate
    state.tools_called.append((
        "submit_resolution",
        {
            "resolution": resolution,
            "cited_artifacts": cited_artifacts,
            "confidence": confidence,
            "escalate": escalate,
        },
    ))
    return {"status": "submitted", "message": "Resolution submitted. Episode complete."}
