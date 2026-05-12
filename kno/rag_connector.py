"""
Knowledge base connector for kno.ai.

Semantic vector search using Vertex AI text-embedding-004 embeddings
stored alongside documents in Firestore.

Architecture:
  ingest_text()          → embed document with text-embedding-004
                           → store {text, embedding, metadata} in Firestore
  search_knowledge_base() → embed query with text-embedding-004
                           → fetch all doc embeddings from Firestore
                           → rank by cosine similarity
                           → return top-k with source citations

No external vector infrastructure (Vector Search / RAG Engine) required.
For a knowledge base <500 pages this is fast: ~50ms for the embedding
call + in-process cosine similarity over 768-dim vectors.

Docs without embeddings (ingested before this version) fall back to
keyword scoring so the KB remains searchable during a rolling re-ingest.
"""
import math
import os
import re
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from google.cloud import firestore

logger = logging.getLogger(__name__)

_PROJECT    = os.environ.get("GOOGLE_CLOUD_PROJECT", "kno-ai-494516")
_LOCATION   = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
_COLLECTION = "knowledge_base"

# text-embedding-004 produces 768-dimensional vectors
_EMBED_MODEL = "text-embedding-004"
# Cosine similarity threshold below which results are suppressed.
# 0.62 prevents unrelated pages (e.g. "Onboarding Guide") from matching
# unrelated queries (e.g. "deployment").  Genuine topic matches typically
# score 0.65+; tangential but relevant docs sit around 0.55–0.64.
_SIM_THRESHOLD = 0.62

_db:          Optional[firestore.Client]     = None
_embed_model: Optional[TextEmbeddingModel]   = None


# ── Singletons ────────────────────────────────────────────────────────────────

def _get_db() -> firestore.Client:
    global _db
    if _db is None:
        _db = firestore.Client(project=_PROJECT)
    return _db


def _get_embed_model() -> TextEmbeddingModel:
    global _embed_model
    if _embed_model is None:
        vertexai.init(project=_PROJECT, location=_LOCATION)
        _embed_model = TextEmbeddingModel.from_pretrained(_EMBED_MODEL)
    return _embed_model


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _embed(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Return a 768-dim embedding vector for *text*.

    task_type:
      "RETRIEVAL_DOCUMENT" — for text being stored
      "RETRIEVAL_QUERY"    — for a search query
    """
    # text-embedding-004 input limit ≈ 2 048 tokens (~8 000 chars).
    # Truncate conservatively to avoid token-limit errors.
    truncated = text[:4_000]
    model = _get_embed_model()
    result = model.get_embeddings([TextEmbeddingInput(truncated, task_type)])
    return result[0].values


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity in pure Python — no numpy needed."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def _keyword_score(title: str, text: str, query: str) -> float:
    """Fallback keyword score (0-1 range) for docs without embeddings."""
    terms = [t.lower() for t in query.split() if len(t) > 2]
    if not terms:
        return 0.0
    title_l, text_l = title.lower(), text.lower()
    raw = sum(3 * title_l.count(t) + text_l.count(t) for t in terms)
    # Normalise: assume a score of 10 ≈ cosine 0.6
    return min(raw / 10.0 * 0.6, 0.85)


# ── Public API ────────────────────────────────────────────────────────────────

def ingest_text(
    text: str,
    display_name: str,
    source: str,
    url: str,
    author: str = "",
    modified_date: str = "",
) -> bool:
    """Store a document with its semantic embedding in Firestore.

    Uses a stable doc ID (source + title) so re-ingestion is idempotent —
    stale content is overwritten rather than duplicated.

    Returns True on success, False on failure.
    """
    try:
        db = _get_db()
        doc_id = re.sub(r"[^a-z0-9_-]", "_",
                        f"{source}__{display_name}".lower())[:500]

        # Generate embedding for semantic retrieval
        embedding = _embed(text, task_type="RETRIEVAL_DOCUMENT")

        db.collection(_COLLECTION).document(doc_id).set({
            "title":         display_name,
            "source":        source,
            "url":           url,
            "author":        author,
            "modified_date": modified_date,
            "text":          text[:10_000],   # store up to 10 KB for snippet extraction
            "embedding":     embedding,       # 768-dim float list
            "embed_model":   _EMBED_MODEL,
            "ingested_at":   datetime.now(timezone.utc).isoformat(),
        })
        logger.info("Ingested '%s' (%d chars, 768-dim embedding)", display_name, len(text))
        return True

    except Exception as e:
        logger.error("Failed to ingest '%s': %s", display_name, e)
        return False


def _parse_date(date_str: str) -> Optional[datetime]:
    """Parse a date string in common formats into a UTC-aware datetime."""
    if not date_str:
        return None
    # Normalise: strip microseconds and trailing Z for consistent parsing
    s = date_str.strip()
    # "2026-05-11T17:54:31.123456+00:00" → "2026-05-11T17:54:31+00:00"
    import re as _re
    s = _re.sub(r"\.\d+", "", s)       # remove fractional seconds
    s = s.replace("Z", "+00:00")        # replace Z with explicit UTC offset
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(s[:len(fmt) + 6], fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            continue
    return None


def _build_query(db: firestore.Client,
                 source_type: Optional[str],
                 days_ago: Optional[int]) -> tuple:
    """Build the most selective Firestore query possible using server-side indexes.

    Strategy
    --------
    Push equality (source) and range (date) filters to Firestore so the SDK
    only streams the matching subset.  Author is a substring match — Firestore
    doesn't support that natively — so it stays as a Python post-filter.

    Composite indexes required (declared in firestore.indexes.json):
      (source, modified_date DESC)
      (source, ingested_at  DESC)
      (modified_date DESC)          ← single-field, auto-created by Firestore
      (ingested_at  DESC)           ← single-field, auto-created by Firestore

    Returns: (query_ref, ts_cutoff_or_None)
      query_ref  — a Firestore Query ready for .stream()
      ts_cutoff  — datetime used for the cutoff (None if no days_ago given);
                   kept so the caller can post-filter docs that store their date
                   in modified_date vs ingested_at without re-computing it.
    """
    coll = db.collection(_COLLECTION)
    ts_cutoff: Optional[datetime] = None

    # ── 1. Equality filter: source ────────────────────────────────────────────
    q = coll.where("source", "==", source_type.lower()) if source_type else coll

    # ── 2. Range filter: date ─────────────────────────────────────────────────
    # Prefer modified_date; fall back to ingested_at for docs that lack it.
    # Firestore requires a composite index when combining equality + range on
    # different fields — those are declared in firestore.indexes.json.
    if days_ago:
        ts_cutoff = datetime.now(timezone.utc) - timedelta(days=days_ago)
        cutoff_iso = ts_cutoff.isoformat()
        if source_type:
            # Use modified_date with the (source, modified_date) composite index.
            # Docs without modified_date won't match but are rare in practice.
            q = q.where("modified_date", ">=", cutoff_iso)
        else:
            # No source filter — rely on the single-field modified_date index.
            q = q.where("modified_date", ">=", cutoff_iso)

    return q, ts_cutoff


def search_knowledge_base(query: str, top_k: int = 5,
                           days_ago: int = None,
                           author: str = None,
                           source_type: str = None) -> list[dict]:
    """Semantic search over the knowledge base using text-embedding-004.

    Server-side Firestore indexes handle source_type (equality) and days_ago
    (range on modified_date).  Author is a Python post-filter (substring match).
    Cosine similarity re-ranks the surviving set.

    Args:
        query:       Natural language question or keywords.
        top_k:       Max results to return (default 5).
        days_ago:    Only consider docs modified/ingested within the last N days.
        author:      Case-insensitive substring match against the author field.
        source_type: Exact match against the source field, e.g. "confluence".

    Returns:
        List of dicts — keys: text, source, title, url, author, date, score
    """
    try:
        db = _get_db()

        # ── Server-side filtering (indexes) ───────────────────────────────────
        q, ts_cutoff = _build_query(db, source_type, days_ago)
        raw_docs = list(q.stream())
        if not raw_docs:
            return []

        # ── Python post-filters ───────────────────────────────────────────────
        filtered: list[tuple] = []
        for doc in raw_docs:
            d = doc.to_dict()

            # Author: case-insensitive substring (can't do this in Firestore)
            if author and author.lower() not in d.get("author", "").lower():
                continue

            # Date fallback: docs that store date only in ingested_at (no
            # modified_date) were skipped by the server-side range filter above;
            # include them here if they fall inside the window.
            if ts_cutoff and not d.get("modified_date"):
                fallback_dt = _parse_date(d.get("ingested_at", ""))
                if fallback_dt and fallback_dt < ts_cutoff:
                    continue

            filtered.append((doc, d))

        if not filtered:
            return []

        # ── Filter-only mode: no query → return docs sorted by date ──────────
        if not query or not query.strip():
            results = []
            for _, d in filtered:
                results.append({
                    "text":   d.get("text", "")[:350],
                    "title":  d.get("title", "Unknown"),
                    "source": d.get("source", ""),
                    "url":    d.get("url", ""),
                    "author": d.get("author", ""),
                    "date":   d.get("modified_date", ""),
                    "score":  1.0,
                })
            # Sort by modified_date descending (most-recent first)
            results.sort(
                key=lambda r: r["date"] or "0000",
                reverse=True,
            )
            return results[:top_k]

        # ── Embed query once ──────────────────────────────────────────────────
        try:
            query_vec = _embed(query, task_type="RETRIEVAL_QUERY")
        except Exception as e:
            logger.warning("Embedding failed (%s); falling back to keyword search", e)
            query_vec = None

        # ── Score each surviving doc ──────────────────────────────────────────
        scored: list[tuple[float, dict]] = []
        for _, d in filtered:
            title      = d.get("title", "Unknown")
            text       = d.get("text", "")
            stored_vec = d.get("embedding")

            if query_vec and stored_vec:
                score = _cosine(query_vec, stored_vec)
            else:
                score = _keyword_score(title, text, query)

            if score >= _SIM_THRESHOLD:
                scored.append((score, {
                    "text":   _extract_snippet(text, query),
                    "title":  title,
                    "source": d.get("source", ""),
                    "url":    d.get("url", ""),
                    "author": d.get("author", ""),
                    "date":   d.get("modified_date", ""),
                    "score":  round(score, 3),
                }))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:top_k]]

    except Exception as e:
        logger.error("Knowledge base search failed: %s", e)
        return []


def backfill_embeddings() -> dict:
    """Generate embeddings for any documents that were ingested without one.

    Call this once after upgrading from keyword-only to semantic search,
    or via POST /admin/kb/backfill.

    Returns: {"updated": N, "skipped": N, "failed": N}
    """
    db     = _get_db()
    docs   = list(db.collection(_COLLECTION).stream())
    counts = {"updated": 0, "skipped": 0, "failed": 0}

    for doc in docs:
        d = doc.to_dict()
        if d.get("embedding"):
            counts["skipped"] += 1
            continue
        try:
            embedding = _embed(d.get("text", ""), task_type="RETRIEVAL_DOCUMENT")
            db.collection(_COLLECTION).document(doc.id).update({
                "embedding":   embedding,
                "embed_model": _EMBED_MODEL,
            })
            counts["updated"] += 1
            logger.info("Backfilled embedding for '%s'", d.get("title"))
        except Exception as e:
            logger.error("Backfill failed for '%s': %s", d.get("title"), e)
            counts["failed"] += 1

    return counts


def get_corpus_stats() -> dict:
    """Return basic stats about the knowledge base."""
    try:
        docs            = list(_get_db().collection(_COLLECTION).stream())
        sources: dict   = {}
        has_embeddings  = 0
        for doc in docs:
            d = doc.to_dict()
            sources[d.get("source", "unknown")] = \
                sources.get(d.get("source", "unknown"), 0) + 1
            if d.get("embedding"):
                has_embeddings += 1
        return {
            "total":           len(docs),
            "with_embeddings": has_embeddings,
            "without_embeddings": len(docs) - has_embeddings,
            "by_source":       sources,
        }
    except Exception as e:
        return {"error": str(e)}


# ── Snippet extraction ────────────────────────────────────────────────────────

def _extract_snippet(text: str, query: str, window: int = 350) -> str:
    """Return a ~350-char snippet of *text* anchored near a query term."""
    text_lower = text.lower()
    best_pos   = len(text)
    for term in query.lower().split():
        if len(term) <= 2:
            continue
        pos = text_lower.find(term)
        if 0 <= pos < best_pos:
            best_pos = pos
    if best_pos == len(text):
        return text[:window]
    start = max(0, best_pos - 100)
    return text[start: start + window].strip()
