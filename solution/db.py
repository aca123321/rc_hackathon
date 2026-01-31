"""
Database connection and operations for the knowledge base.
"""

from typing import Optional
import psycopg2
from pgvector.psycopg2 import register_vector
from datetime import datetime

from config import DB_CONFIG


def get_connection() -> psycopg2.extensions.connection:
    """
    Get a connection to the PostgreSQL database.

    Returns:
        PostgreSQL connection object with pgvector support registered.
    """
    conn = psycopg2.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        dbname=DB_CONFIG["database"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
    )
    register_vector(conn)
    return conn


def insert_knowledge_base_entry(
    text: str,
    embedding: list[float],
    num_words: int,
    text_length: int,
    source_file: Optional[str] = None,
) -> int:
    """
    Insert a new entry into the knowledge_base table.

    Args:
        text: The text content to store.
        embedding: The embedding vector as a list of floats.
        num_words: Number of words in the text.
        text_length: Length of the text in characters.
        source_file: The source file path for citation (e.g., PDF path).

    Returns:
        The id of the inserted row.
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO knowledge_base (text, embedding, num_words, text_length, source_file, created_at)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (text, embedding, num_words, text_length, source_file, datetime.now()),
    )

    result = cursor.fetchone()
    if result is None:
        raise RuntimeError("INSERT did not return an id")
    row_id = result[0]
    conn.commit()
    cursor.close()
    conn.close()

    return row_id


def get_knowledge_base_count() -> int:
    """Get the total number of entries in the knowledge_base table."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM knowledge_base")
    result = cursor.fetchone()
    count = result[0] if result else 0
    cursor.close()
    conn.close()
    return count


def get_knowledge_base_stats() -> dict:
    """Get detailed stats about the knowledge_base table."""
    conn = get_connection()
    cursor = conn.cursor()

    stats = {}

    # Total entries
    cursor.execute("SELECT COUNT(*) FROM knowledge_base")
    result = cursor.fetchone()
    stats["total_entries"] = result[0] if result else 0

    # Entries with valid embeddings
    cursor.execute("SELECT COUNT(*) FROM knowledge_base WHERE embedding IS NOT NULL")
    result = cursor.fetchone()
    stats["with_embeddings"] = result[0] if result else 0

    # Check embedding dimension (sample)
    cursor.execute(
        "SELECT array_length(embedding::real[], 1) FROM knowledge_base WHERE embedding IS NOT NULL LIMIT 1"
    )
    result = cursor.fetchone()
    stats["embedding_dimension"] = result[0] if result else None

    cursor.close()
    conn.close()
    return stats
