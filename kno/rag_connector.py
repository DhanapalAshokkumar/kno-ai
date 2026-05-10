"""
Vertex AI RAG Engine connector for kno.ai.
Creates a shared knowledge corpus from Confluence pages and Drive documents,
enabling semantic search with traceable citations.
"""
import os
import logging
from typing import Optional
from datetime import datetime

import vertexai
from vertexai.preview import rag

logger = logging.getLogger(__name__)

_PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT", "kno-ai-494516")
_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
_CORPUS_DISPLAY_NAME = "kno-ai-knowledge"

# Module-level corpus name cache (warm across requests on same Cloud Run instance)
_corpus_name: Optional[str] = None


def _init():
    vertexai.init(project=_PROJECT, location=_LOCATION)


def get_or_create_corpus() -> str:
    """Return the RAG corpus resource name, creating it if needed."""
    global _corpus_name
    if _corpus_name:
        return _corpus_name

    _init()
    try:
        for c in rag.list_corpora():
            if c.display_name == _CORPUS_DISPLAY_NAME:
                _corpus_name = c.name
                logger.info("Using existing RAG corpus: %s", _corpus_name)
                return _corpus_name
    except Exception as e:
        logger.warning("Could not list corpora: %s", e)

    # Create new corpus with Vertex AI managed embeddings
    try:
        corpus = rag.create_corpus(
            display_name=_CORPUS_DISPLAY_NAME,
            description="kno.ai company knowledge base — Confluence + Drive",
            embedding_model_config=rag.RagEmbeddingModelConfig(
                vertex_prediction_endpoint=rag.VertexPredictionEndpoint(
                    publisher_model="publishers/google/models/text-embedding-005"
                )
            ),
        )
        _corpus_name = corpus.name
        logger.info("Created new RAG corpus: %s", _corpus_name)
        return _corpus_name
    except Exception as e:
        logger.error("Failed to create RAG corpus: %s", e)
        raise


def ingest_confluence_page(
    title: str,
    content: str,
    url: str,
    author: str = "",
    updated_date: str = "",
    space: str = "",
) -> bool:
    """Add a Confluence page to the RAG corpus.

    Args:
        title:        Page title
        content:      Plain-text page content
        url:          Full Confluence page URL
        author:       Author display name
        updated_date: ISO date string of last update
        space:        Confluence space key

    Returns:
        True on success, False on failure.
    """
    try:
        corpus_name = get_or_create_corpus()
        text = f"Title: {title}\nSpace: {space}\nAuthor: {author}\nUpdated: {updated_date}\nURL: {url}\n\n{content}"

        rag.upload_file(
            corpus_name=corpus_name,
            path=None,
            display_name=f"confluence::{title}",
            description=f"Confluence | {space} | {author} | {updated_date} | {url}",
            # Inline text upload via RagFile
            rag_file=rag.RagFile(
                display_name=f"confluence::{title}",
                rag_file_metadata=rag.RagMetadata(
                    metadata={
                        "source": "confluence",
                        "title": title,
                        "url": url,
                        "author": author,
                        "updated_date": updated_date,
                        "space": space,
                    }
                ),
            ),
        )
        return True
    except Exception as e:
        logger.error("Failed to ingest Confluence page '%s': %s", title, e)
        return False


def ingest_text(
    text: str,
    display_name: str,
    source: str,
    url: str,
    author: str = "",
    modified_date: str = "",
) -> bool:
    """Generic text ingestion into the RAG corpus.

    Args:
        text:          Plain text content
        display_name:  Human-readable name for the document
        source:        Source system, e.g. 'confluence', 'drive', 'jira'
        url:           Canonical link back to the source
        author:        Author/owner name
        modified_date: ISO date string of last modification

    Returns:
        True on success, False on failure.
    """
    try:
        import tempfile, pathlib
        corpus_name = get_or_create_corpus()

        # Write to a temp file — Vertex AI RAG upload_file requires a file path
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(f"Source: {source}\nTitle: {display_name}\nURL: {url}\n"
                    f"Author: {author}\nModified: {modified_date}\n\n{text}")
            tmp_path = f.name

        rag.upload_file(
            corpus_name=corpus_name,
            path=tmp_path,
            display_name=display_name,
            description=f"{source} | {author} | {modified_date} | {url}",
        )
        pathlib.Path(tmp_path).unlink(missing_ok=True)
        return True
    except Exception as e:
        logger.error("Failed to ingest '%s': %s", display_name, e)
        return False


def search_knowledge_base(query: str, top_k: int = 5) -> list[dict]:
    """Search the RAG corpus and return passages with full citation metadata.

    Args:
        query: Natural language search query
        top_k: Number of passages to return (default 5)

    Returns:
        List of dicts with keys: text, source, title, url, author, date, score
    """
    try:
        corpus_name = get_or_create_corpus()
        response = rag.retrieval_query(
            rag_resources=[rag.RagResource(rag_corpus=corpus_name)],
            text=query,
            rag_retrieval_config=rag.RagRetrievalConfig(
                top_k=top_k,
                filter=rag.Filter(vector_distance_threshold=0.7),
            ),
        )

        results = []
        for ctx in response.contexts.contexts:
            # Parse metadata out of the description field (set during ingest)
            desc = ctx.source_display_name or ""
            parts = desc.split(" | ")
            results.append({
                "text": ctx.text,
                "title": ctx.source_display_name or "Unknown",
                "url": parts[3] if len(parts) > 3 else "",
                "author": parts[1] if len(parts) > 1 else "",
                "date": parts[2] if len(parts) > 2 else "",
                "source": parts[0] if parts else "unknown",
                "score": ctx.score if hasattr(ctx, "score") else 0.0,
            })
        return results
    except Exception as e:
        logger.error("RAG search failed: %s", e)
        return []
