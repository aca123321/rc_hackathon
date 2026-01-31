"""
Document retrieval module for semantic search using pgvector.

Provides methods to retrieve the most relevant document chunks
from the knowledge base using vector similarity search.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from config import DEFAULT_TOP_K, LOG_LEVEL
from db import get_connection
from embeddings import create_embedding

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved document chunk with metadata."""

    id: int
    text: str
    source_file: str
    similarity_score: float
    num_words: int
    text_length: int

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "text": self.text,
            "source_file": self.source_file,
            "similarity_score": self.similarity_score,
            "num_words": self.num_words,
            "text_length": self.text_length,
        }


def retrieve_top_k(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    similarity_threshold: Optional[float] = None,
) -> list[RetrievedChunk]:
    """
    Retrieve the top-k most relevant document chunks for a given query.

    Uses cosine similarity search with pgvector to find the most
    semantically similar chunks in the knowledge base.

    Args:
        query: The natural language query to search for.
        top_k: Number of top results to return. Default is 5.
        similarity_threshold: Optional minimum similarity score (0-1).
                            Chunks below this threshold are filtered out.

    Returns:
        List of RetrievedChunk objects sorted by similarity (highest first).

    Example:
        >>> chunks = retrieve_top_k("What is the FMLA policy?", top_k=3)
        >>> for chunk in chunks:
        ...     print(f"{chunk.source_file}: {chunk.similarity_score:.3f}")
    """
    # Create embedding for the query
    logger.debug(f"Creating embedding for query: {query[:100]}...")
    query_embedding = create_embedding(query)

    # Connect to database and perform similarity search
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # Use cosine distance operator (<=>), which is 1 - cosine_similarity
        # So we convert back: similarity = 1 - distance
        cursor.execute(
            """
            SELECT 
                id,
                text,
                source_file,
                1 - (embedding <=> %s::vector) as similarity,
                num_words,
                text_length
            FROM knowledge_base
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, query_embedding, top_k),
        )

        results = cursor.fetchall()
        logger.info(
            f"Database returned {len(results)} raw results (requested top_k={top_k})"
        )

        # Log similarity scores for raw results
        if results:
            similarities = [float(row[3]) for row in results]
            logger.info(
                f"Similarity scores: min={min(similarities):.3f}, max={max(similarities):.3f}"
            )
            logger.info(f"Top 5 similarities: {[f'{s:.3f}' for s in similarities[:5]]}")

        chunks = []
        for row in results:
            chunk = RetrievedChunk(
                id=row[0],
                text=row[1],
                source_file=row[2] or "unknown",
                similarity_score=float(row[3]),
                num_words=row[4],
                text_length=row[5],
            )

            # Log similarity scores for debugging
            if len(chunks) < 5:  # Log first 5
                logger.debug(
                    f"Chunk {chunk.id}: similarity={chunk.similarity_score:.3f}, source={chunk.source_file}"
                )

            # Apply similarity threshold if specified
            if (
                similarity_threshold is None
                or chunk.similarity_score >= similarity_threshold
            ):
                chunks.append(chunk)
            else:
                logger.debug(
                    f"Filtered out chunk {chunk.id} with similarity {chunk.similarity_score:.3f} < {similarity_threshold}"
                )

        logger.info(
            f"Retrieved {len(chunks)} chunks after filtering (top_k={top_k}, threshold={similarity_threshold})"
        )
        return chunks

    finally:
        cursor.close()
        conn.close()


def retrieve_with_deduplication(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    max_chunks_per_source: int = 2,
    similarity_threshold: Optional[float] = None,
) -> list[RetrievedChunk]:
    """
    Retrieve top-k chunks with source diversity.

    Limits the number of chunks from any single source document
    to ensure diverse retrieval across multiple sources.

    Args:
        query: The natural language query to search for.
        top_k: Target number of results to return.
        max_chunks_per_source: Maximum chunks from any single source.
        similarity_threshold: Optional minimum similarity score.

    Returns:
        List of RetrievedChunk objects with source diversity.
    """
    # Retrieve more than top_k to allow for filtering
    fetch_multiplier = 2  # Fetch 3x to have room for deduplication
    candidates = retrieve_top_k(
        query,
        top_k=top_k * fetch_multiplier,
        similarity_threshold=similarity_threshold,
    )

    # Apply source diversity constraint
    source_counts: dict[str, int] = {}
    diverse_chunks: list[RetrievedChunk] = []

    for chunk in candidates:
        source = chunk.source_file
        current_count = source_counts.get(source, 0)

        if current_count < max_chunks_per_source:
            diverse_chunks.append(chunk)
            source_counts[source] = current_count + 1

            if len(diverse_chunks) >= top_k:
                break

    logger.info(
        f"Retrieved {len(diverse_chunks)} diverse chunks from "
        f"{len(source_counts)} sources"
    )
    return diverse_chunks


def get_unique_sources(chunks: list[RetrievedChunk]) -> list[str]:
    """
    Get unique source files from a list of chunks.

    Args:
        chunks: List of retrieved chunks.

    Returns:
        List of unique source file names in order of first appearance.
    """
    seen: set[str] = set()
    sources: list[str] = []
    for chunk in chunks:
        if chunk.source_file not in seen:
            seen.add(chunk.source_file)
            sources.append(chunk.source_file)
    return sources


def format_chunks_for_context(
    chunks: list[RetrievedChunk],
    *,
    include_scores: bool = False,
) -> str:
    """
    Format retrieved chunks into a context string for the LLM.

    Args:
        chunks: List of retrieved chunks.
        include_scores: If True, include similarity scores in output.

    Returns:
        Formatted string with numbered chunks and source citations.
    """
    if not chunks:
        return "No relevant documents found."

    formatted_parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[Document {i}] Source: {chunk.source_file}"
        if include_scores:
            header += f" (relevance: {chunk.similarity_score:.2f})"

        formatted_parts.append(f"{header}\n{chunk.text}")

    return "\n\n---\n\n".join(formatted_parts)
