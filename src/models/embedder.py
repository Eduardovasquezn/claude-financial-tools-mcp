"""Embedding model wrapper supporting Gemini and SentenceTransformers."""

import os
import time
from pathlib import Path
from typing import Any, List, Optional, Union

import google.genai as genai
from PIL import Image

from src.config import Config
from src.helpers.logs import get_logger

logger = get_logger()


class Embedder:
    """Wrapper for embedding model supporting both Gemini and SentenceTransformers."""

    def __init__(self, model_name: str | None = None):
        """
        Initialize embedder.

        Args:
            model_name: Name of the embedding model (defaults to Config.EMBEDDING_MODEL)
                - For Gemini: "gemini-embedding-2-preview" or "text-embedding-004"
                - For SentenceTransformers: "all-MiniLM-L6-v2", etc.
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")

        # Check if it's a Gemini model
        if self.model_name.startswith("gemini-") or self.model_name.startswith(
            "text-embedding-"
        ):
            self.provider = "gemini"
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY environment variable is required for Gemini embeddings"
                )
            self.client = genai.Client(api_key=api_key)

            # Get actual dimensions by doing a test embedding
            test_result = self.client.models.embed_content(
                model=self.model_name,
                contents=["test"],
            )
            if test_result.embeddings and test_result.embeddings[0].values:
                self.dimensions = len(test_result.embeddings[0].values)
            else:
                raise ValueError("Failed to get embedding dimensions from test")
            logger.info(
                f"Using Google Gemini embeddings with {self.dimensions} dimensions"
            )
        else:
            self.provider = "sentence_transformers"
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]

            self.model = SentenceTransformer(self.model_name)
            self.dimensions = self.model.get_sentence_embedding_dimension()
            logger.info(f"Using SentenceTransformers with {self.dimensions} dimensions")

    def _embed_with_retry(self, contents: List[str], max_retries: int = 3) -> Any:
        """
        Embed content with automatic retry on rate limits.

        Args:
            contents: List of texts to embed
            max_retries: Maximum number of retry attempts

        Returns:
            Embedding result from Gemini API
        """
        import re

        for attempt in range(max_retries):
            try:
                return self.client.models.embed_content(
                    model=self.model_name, contents=contents
                )
            except Exception as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str and "RESOURCE_EXHAUSTED" in error_str

                if is_rate_limit and attempt < max_retries - 1:
                    # Extract retry delay from error message
                    match = re.search(r"Please retry in ([\d.]+)s", error_str)
                    delay = float(match.group(1)) if match else 16

                    logger.warning(
                        f"Rate limit hit. Waiting {delay:.1f}s (retry {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(delay)
                else:
                    # Either not a rate limit error, or max retries exceeded
                    raise

        raise Exception(f"Failed after {max_retries} retries")

    def embed(self, text: str | List[str]) -> List[float] | List[List[float]]:
        """
        Embed text using the configured model with batch support.

        Args:
            text: Single string or list of strings to embed

        Returns:
            Single embedding vector or list of embedding vectors
        """
        if self.provider == "gemini":
            if isinstance(text, str):
                # Single text
                result = self._embed_with_retry([text])
                if result.embeddings:
                    return result.embeddings[0].values
                return []
            else:
                # Batch embed - send all texts in ONE API call
                result = self._embed_with_retry(text)
                if result.embeddings:
                    return [emb.values for emb in result.embeddings]
                return []
        else:
            # SentenceTransformers - already batched
            embeddings = self.model.encode(text)
            if isinstance(text, str):
                return embeddings.tolist()
            return [emb.tolist() for emb in embeddings]

    def embed_image(
        self, image: Union[str, Path, Image.Image]
    ) -> List[float]:
        """
        Embed an image using Gemini's multimodal embedding.

        Args:
            image: Path to image file or PIL Image object

        Returns:
            Embedding vector for the image

        Raises:
            ValueError: If provider is not Gemini (only Gemini supports image embeddings)
        """
        if self.provider != "gemini":
            raise ValueError(
                "Image embedding is only supported with Gemini models. "
                f"Current provider: {self.provider}"
            )

        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # Gemini embed_content accepts PIL Images
        result = self._embed_with_retry([image])
        if result.embeddings:
            return result.embeddings[0].values
        return []

    def embed_multimodal(
        self, text: str, image: Union[str, Path, Image.Image]
    ) -> List[float]:
        """
        Embed both text and image together for richer context.

        Args:
            text: Text description or context
            image: Path to image file or PIL Image object

        Returns:
            Combined embedding vector

        Raises:
            ValueError: If provider is not Gemini
        """
        if self.provider != "gemini":
            raise ValueError(
                "Multimodal embedding is only supported with Gemini models. "
                f"Current provider: {self.provider}"
            )

        # Load image if path is provided
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # Gemini can embed text + image together
        result = self._embed_with_retry([text, image])
        if result.embeddings:
            return result.embeddings[0].values
        return []


# Global embedder instance - will be initialized with config
_embedder: Optional[Embedder] = None


def get_embedder(model_name: Optional[str] = None) -> Embedder:
    """
    Get or create embedder instance (singleton pattern).

    Args:
        model_name: Optional model name (defaults to Config.EMBEDDING_MODEL)

    Returns:
        Embedder instance
    """
    global _embedder
    if _embedder is None:
        _embedder = Embedder(model_name)
    return _embedder
