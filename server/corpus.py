import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False


def _tokenize(text: str) -> List[str]:
    return re.findall(r'\b[a-z0-9]+\b', text.lower())


def _snippet(text: str, max_chars: int = 200) -> str:
    if len(text) <= max_chars:
        return text
    trimmed = text[:max_chars]
    last_space = trimmed.rfind(' ')
    return (trimmed[:last_space] if last_space > 0 else trimmed) + '...'


class _KeywordIndex:
    """TF-IDF keyword search index."""

    def __init__(self, texts: List[str]):
        self._n = len(texts)
        if self._n == 0:
            self._idf: Dict[str, float] = {}
            self._vectors: List[Dict[str, float]] = []
            return

        tokenized = [_tokenize(t) for t in texts]
        df: Counter = Counter()
        for toks in tokenized:
            for tok in set(toks):
                df[tok] += 1

        self._idf = {t: math.log((self._n + 1) / (df[t] + 1)) for t in df}

        self._vectors = []
        for toks in tokenized:
            tf = Counter(toks)
            total = len(toks) or 1
            self._vectors.append(
                {t: (tf[t] / total) * self._idf.get(t, 0.0) for t in tf}
            )

    def search(self, query: str, k: int) -> List[int]:
        if not self._vectors:
            return []
        q_toks = _tokenize(query)
        if not q_toks:
            return list(range(min(k, self._n)))

        q_tf = Counter(q_toks)
        q_total = len(q_toks)
        q_vec = {t: (q_tf[t] / q_total) * self._idf.get(t, 0.0) for t in q_tf}

        scores = [
            (i, sum(q_vec.get(t, 0.0) * dv.get(t, 0.0) for t in q_vec))
            for i, dv in enumerate(self._vectors)
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [i for i, _ in scores[:k]]


class Corpus:
    def __init__(self, data_dir: Optional[Path] = None):
        if data_dir is None:
            data_dir = Path(__file__).parent.parent / "data"
        self._data_dir = Path(data_dir)

        self._kb: List[Dict] = self._load("kb.json")
        self._tickets: List[Dict] = self._load("past_tickets.json")
        self._incidents: List[Dict] = self._load("incidents.json")
        self._train: List[Dict] = self._load("train_tickets.json")
        self._eval: List[Dict] = self._load("eval_tickets.json")

        self._kb_map = {a["article_id"]: a for a in self._kb if "article_id" in a}
        self._ticket_map = {t["ticket_id"]: t for t in self._tickets if "ticket_id" in t}
        self._incident_map = {i["incident_id"]: i for i in self._incidents if "incident_id" in i}

        self._build_indices()

    def _load(self, filename: str) -> List[Dict]:
        path = self._data_dir / filename
        if not path.exists():
            return []
        try:
            with open(path) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _kb_texts(self) -> List[str]:
        return [a.get("title", "") + " " + a.get("body", "") for a in self._kb]

    def _ticket_texts(self) -> List[str]:
        return [t.get("title", "") + " " + t.get("description", "") for t in self._tickets]

    def _incident_texts(self) -> List[str]:
        return [
            i.get("title", "") + " " + i.get("summary", "") + " " + i.get("root_cause", "")
            for i in self._incidents
        ]

    def _build_indices(self):
        kb_texts = self._kb_texts()
        ticket_texts = self._ticket_texts()
        incident_texts = self._incident_texts()
        has_data = any([kb_texts, ticket_texts, incident_texts])

        if _ST_AVAILABLE and has_data:
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self._kb_emb = (
                self._model.encode(kb_texts, show_progress_bar=False)
                if kb_texts else np.zeros((0, 384))
            )
            self._ticket_emb = (
                self._model.encode(ticket_texts, show_progress_bar=False)
                if ticket_texts else np.zeros((0, 384))
            )
            self._incident_emb = (
                self._model.encode(incident_texts, show_progress_bar=False)
                if incident_texts else np.zeros((0, 384))
            )
            self._use_semantic = True
        else:
            self._kb_idx = _KeywordIndex(kb_texts)
            self._ticket_idx = _KeywordIndex(ticket_texts)
            self._incident_idx = _KeywordIndex(incident_texts)
            self._use_semantic = False

    def _sem_search(self, query: str, embeddings, k: int) -> List[int]:
        if embeddings.shape[0] == 0:
            return []
        q = self._model.encode([query], show_progress_bar=False)[0]
        q_norm = q / (np.linalg.norm(q) + 1e-8)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        scores = (embeddings / norms) @ q_norm
        top = np.argsort(scores)[::-1][:k]
        return top.tolist()

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def search_kb(self, query: str, max_results: int = 5) -> List[Dict]:
        if self._use_semantic:
            indices = self._sem_search(query, self._kb_emb, max_results)
        else:
            indices = self._kb_idx.search(query, max_results)
        return [
            {
                "article_id": self._kb[i].get("article_id", ""),
                "title": self._kb[i].get("title", ""),
                "snippet": _snippet(self._kb[i].get("body", "")),
                "section": self._kb[i].get("domain", ""),
            }
            for i in indices
        ]

    def get_article(self, article_id: str) -> Optional[Dict]:
        a = self._kb_map.get(article_id)
        if not a:
            return None
        return {
            "article_id": a.get("article_id", ""),
            "title": a.get("title", ""),
            "body": a.get("body", ""),
            "tags": a.get("tags", []),
        }

    def search_tickets(
        self, query: str, status: Optional[str] = None, max_results: int = 5
    ) -> List[Dict]:
        if self._use_semantic:
            # Search all, then filter by status
            k = min(len(self._tickets), max(max_results * 3, 20))
            indices = self._sem_search(query, self._ticket_emb, k)
            if status:
                indices = [
                    i for i in indices
                    if self._tickets[i].get("status", "").lower() == status.lower()
                ]
            indices = indices[:max_results]
        else:
            if status:
                pool = [
                    (i, t) for i, t in enumerate(self._tickets)
                    if t.get("status", "").lower() == status.lower()
                ]
                pool_texts = [t.get("title", "") + " " + t.get("description", "") for _, t in pool]
                local_idx = _KeywordIndex(pool_texts)
                local_hits = local_idx.search(query, max_results)
                indices = [pool[j][0] for j in local_hits]
            else:
                indices = self._ticket_idx.search(query, max_results)

        return [
            {
                "ticket_id": self._tickets[i].get("ticket_id", ""),
                "title": self._tickets[i].get("title", ""),
                "snippet": _snippet(self._tickets[i].get("description", "")),
                "status": self._tickets[i].get("status", ""),
            }
            for i in indices
        ]

    def get_ticket(self, ticket_id: str) -> Optional[Dict]:
        t = self._ticket_map.get(ticket_id)
        if not t:
            return None
        return {
            "ticket_id": t.get("ticket_id", ""),
            "title": t.get("title", ""),
            "description": t.get("description", ""),
            "comments": t.get("comments", []),
            "resolution": t.get("resolution"),
        }

    def search_incidents(self, query: str, max_results: int = 3) -> List[Dict]:
        if self._use_semantic:
            indices = self._sem_search(query, self._incident_emb, max_results)
        else:
            indices = self._incident_idx.search(query, max_results)
        return [
            {
                "incident_id": self._incidents[i].get("incident_id", ""),
                "title": self._incidents[i].get("title", ""),
                "snippet": _snippet(self._incidents[i].get("summary", "")),
                "severity": self._incidents[i].get("severity", ""),
            }
            for i in indices
        ]

    def get_incident(self, incident_id: str) -> Optional[Dict]:
        inc = self._incident_map.get(incident_id)
        if not inc:
            return None
        return {
            "incident_id": inc.get("incident_id", ""),
            "title": inc.get("title", ""),
            "summary": inc.get("summary", ""),
            "root_cause": inc.get("root_cause", ""),
            "remediation": inc.get("remediation", ""),
        }

    @property
    def train_tickets(self) -> List[Dict]:
        return self._train

    @property
    def eval_tickets(self) -> List[Dict]:
        return self._eval
