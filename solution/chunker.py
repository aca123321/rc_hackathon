"""
Text chunking utilities for splitting text into overlapping word-based chunks.
"""

from __future__ import annotations

from dataclasses import dataclass

from config import CHUNK_SIZE, CHUNK_OVERLAP

# Alias for backward compatibility
OVERLAP = CHUNK_OVERLAP


@dataclass
class Chunk:
    """A single text chunk with metadata."""

    text: str
    index: int  # 0-based chunk index
    start_word: int  # 0-based index of first word in original text
    end_word: int  # 0-based index of last word (exclusive)
    word_count: int


def chunk_text(
    text: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[Chunk]:
    """
    Split text into overlapping chunks based on word count.

    Args:
        text: The input string to chunk.
        chunk_size: Maximum number of words per chunk. Default 500.
        overlap: Number of overlapping words between consecutive chunks. Default 100.

    Returns:
        List of Chunk objects, each containing:
            - text: the chunk's text
            - index: 0-based chunk number
            - start_word: starting word index in original
            - end_word: ending word index (exclusive)
            - word_count: number of words in the chunk

    Raises:
        ValueError: If overlap >= chunk_size or either is non-positive.

    Example:
        >>> chunks = chunk_text("word " * 5000, chunk_size=2000, overlap=500)
        >>> len(chunks)
        3
        >>> chunks[0].word_count
        2000
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
        )

    words = text.split()
    total_words = len(words)

    if total_words == 0:
        return []

    chunks: list[Chunk] = []
    step = chunk_size - overlap  # how far to advance each iteration
    start = 0
    index = 0

    while start < total_words:
        end = min(start + chunk_size, total_words)
        chunk_words = words[start:end]
        chunks.append(
            Chunk(
                text=" ".join(chunk_words),
                index=index,
                start_word=start,
                end_word=end,
                word_count=len(chunk_words),
            )
        )
        index += 1
        start += step

        # Stop if we've already captured to the end
        if end == total_words:
            break

    return chunks


def chunk_text_simple(
    text: str,
    *,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP,
) -> list[str]:
    """
    Convenience wrapper that returns just the chunk text strings.

    Args:
        text: The input string to chunk.
        chunk_size: Maximum number of words per chunk. Default 500.
        overlap: Number of overlapping words between consecutive chunks. Default 100.

    Returns:
        List of chunk text strings.
    """
    return [c.text for c in chunk_text(text, chunk_size=chunk_size, overlap=overlap)]
