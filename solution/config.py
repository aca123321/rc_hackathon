"""
Centralized configuration management for the NovaTech RAG Agent.

Loads configuration from environment variables and .env file.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from the same directory as this config file
_env_path = Path(__file__).parent / ".env"
load_dotenv(_env_path)


def _get_env(key: str, default: str | None = None) -> str:
    """Get environment variable with optional default."""
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


def _get_env_int(key: str, default: int) -> int:
    """Get environment variable as integer."""
    value = os.getenv(key)
    if value is None:
        return default
    return int(value)


def _get_env_float(key: str, default: float) -> float:
    """Get environment variable as float."""
    value = os.getenv(key)
    if value is None:
        return default
    return float(value)


# =============================================================================
# OpenAI Configuration
# =============================================================================
OPENAI_API_KEY = _get_env("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = _get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = _get_env(
    "OPENAI_CHAT_MODEL", "gpt-5.2"
)  # Default to gpt-4o for best balance

# =============================================================================
# PostgreSQL Database Configuration
# =============================================================================
POSTGRES_HOST = _get_env("POSTGRES_HOST", "localhost")
POSTGRES_PORT = _get_env("POSTGRES_PORT", "5432")
POSTGRES_DB = _get_env("POSTGRES_DB", "rapidclaim")
POSTGRES_USER = _get_env("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = _get_env("POSTGRES_PASSWORD")

DB_CONFIG = {
    "host": POSTGRES_HOST,
    "port": POSTGRES_PORT,
    "database": POSTGRES_DB,
    "user": POSTGRES_USER,
    "password": POSTGRES_PASSWORD,
}

# =============================================================================
# RAG Agent Configuration
# =============================================================================
DEFAULT_TOP_K = _get_env_int("DEFAULT_TOP_K", 10)
MAX_CONVERSATION_HISTORY = _get_env_int("MAX_CONVERSATION_HISTORY", 6)
MAX_CHUNKS_PER_SOURCE = _get_env_int("MAX_CHUNKS_PER_SOURCE", 2)
SIMILARITY_THRESHOLD = _get_env_float("SIMILARITY_THRESHOLD", 0.05)

# =============================================================================
# Chunker Configuration
# =============================================================================
CHUNK_SIZE = _get_env_int("CHUNK_SIZE", 500)
CHUNK_OVERLAP = _get_env_int("CHUNK_OVERLAP", 100)

# =============================================================================
# Logging Configuration
# =============================================================================
LOG_LEVEL = _get_env("LOG_LEVEL", "INFO")
