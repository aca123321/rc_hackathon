"""
Embedding generation using OpenAI's embedding models.
"""

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def create_embedding(text: str) -> list[float]:
    """
    Creates an embedding from the given text using OpenAI's embedding model.

    The model is configured via the OPENAI_EMBEDDING_MODEL environment variable.

    Args:
        text: The text to create an embedding for.

    Returns:
        A list of floats representing the embedding vector.
    """
    response = client.embeddings.create(input=text, model=OPENAI_EMBEDDING_MODEL)
    return response.data[0].embedding
