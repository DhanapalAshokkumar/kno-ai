"""
Knowledge base connector for kno.ai.
Stores documents in Firestore and provides full-text keyword search.
Simple and reliable — no external vector infrastructure required.
"""
import os
import re
import logging
from datetime import datetime, timezone
from typing import Optional

from google.cloud import firestore

logger = logging.getLogger(__name__)

_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "kno-ai-494516")
_COLLECTION = "knowledge_base"

_db: Optional[firestore.Client] = None


def _get_db() -> firestore.Client:
    global _db
    if _db is None:
        _db = firestore.Client(project=_PROJECT)
    return _db


def ingest_text(
    text: str,
    display_name: str,
    source: str,
    url: str,
    author: str = "",
    modified_date: str = "",
) -> bool:
    """Store a document in the Firestore knowledge base.

    Uses a stable document ID derived from source + title so re-ingestion
    is idempotent (overwrites stale content rather than duplicating).

    Returns True on success, False on failure.
    """
    try:
        db = _get_db()
        # Stable doc ID: source + sanitised title
        doc_id = re.sub(r"[^a-z0-9_-]", "_", f"{source}__{display_name}".lower())[:500]

        db.collection(_COLLECTION).document(doc_id).set({
            "title":         display_name,
            "source":        source,
            "url":           url,
            "author":        author,
            "modified_date": modified_date,
            "text":          text[:10_000],   # cap at 10 KB per doc
            "ingested_at":   datetime.now(timezone.utc).isoformat(),
        })
        return True
    except Exception as e:
        logger.error("Failed to ingest '%s': %s", display_name, e)
        return False


def search_knowledge_base(query: str, top_k: int = 5) -> list[dict]:
    """Search the Firestore knowledge base for documents matching a query.

    Performs a case-insensitive keyword search across title and text fields.
    Returns up to top_k results ranked by relevance (title matches first).

    Args:
        query: Natural language query or keywords
        top_k: Max results to return (default 5)

    Returns:
        List of dicts with keys: text, source, title, url, author, date, score
    """
    try:
        db = _get_db()
        terms = [t.lower() for t in query.split() if len(t) > 2]
        if not terms:
            return []

        # Fetch all docs (knowledge base is small — typically <500 pages)
        docs = list(db.collection(_COLLECTION).stream())
        if not docs:
            return []

        scored = []
        for doc in docs:
            d = doc.to_dict()
            title_lower = d.get("title", "").lower()
            text_lower  = d.get("text",  "").lower()

            # Score: 3 pts per term in title, 1 pt per term in text
            score = sum(
                3 * title_lower.count(t) + text_lower.count(t)
                for t in terms
            )
            if score > 0:
                # Extract a relevant snippet around the first matching term
                snippet = _extract_snippet(d.get("text", ""), terms)
                scored.append((score, {
                    "text":   snippet,
                    "title":  d.get("title", "Unknown"),
                    "source": d.get("source", ""),
                    "url":    d.get("url", ""),
                    "author": d.get("author", ""),
                    "date":   d.get("modified_date", ""),
                    "score":  score,
                }))

        # Sort by score descending, return top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    except Exception as e:
        logger.error("Knowledge base search failed: %s", e)
        return []


def _extract_snippet(text: str, terms: list[str], window: int = 300) -> str:
    """Return a ~300-char snippet of text centred on the first term match."""
    text_lower = text.lower()
    best_pos = len(text)
    for t in terms:
        pos = text_lower.find(t)
        if 0 <= pos < best_pos:
            best_pos = pos
    if best_pos == len(text):
        return text[:window]
    start = max(0, best_pos - 100)
    return text[start: start + window].strip()


def get_corpus_stats() -> dict:
    """Return basic stats about the knowledge base."""
    try:
        docs = list(_get_db().collection(_COLLECTION).stream())
        sources = {}
        for doc in docs:
            src = doc.to_dict().get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        return {"total": len(docs), "by_source": sources}
    except Exception as e:
        return {"error": str(e)}
