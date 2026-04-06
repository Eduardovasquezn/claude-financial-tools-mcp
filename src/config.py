"""Configuration for the sentiment agent."""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for sentiment agent."""

    # Embedding Configuration
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")

    # Qdrant Configuration
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION_NAME", "stock-market")
    EMBEDDING_DIMENSIONS: int = int(
        os.getenv("EMBEDDING_DIMENSIONS", "3072")
    )  # Gemini default is 3072
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "gemini-embedding-2-preview"
    )  # Gemini for embeddings
