"""Gemini embedding service built on langchain-google-genai.

This module only turns text/Documents into embedding vectors. Vector
storage and retrieval are implemented in later phases.
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config.settings import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL
from utils.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingServiceError(Exception):
    """Raised when embedding configuration or an embedding request fails."""


class GeminiEmbeddingService:
    """Generates embeddings for text and LangChain Documents using Gemini.

    The underlying Gemini client is created lazily on first use, so
    constructing this service (and importing this module) never requires
    GOOGLE_API_KEY to be set. The key is only needed once an embedding is
    actually requested.
    """

    def __init__(
        self,
        model: str = EMBEDDING_MODEL,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        if not model:
            raise EmbeddingServiceError(
                "No embedding model configured. Set EMBEDDING_MODEL in your .env file."
            )
        self._model = model
        self._batch_size = batch_size
        self._client: Optional[GoogleGenerativeAIEmbeddings] = None

    @property
    def _embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Return the underlying LangChain embeddings client, creating it on first use."""
        if self._client is None:
            try:
                self._client = GoogleGenerativeAIEmbeddings(model=self._model)
            except Exception as exc:
                logger.error(
                    "Failed to initialize Gemini embeddings client (%s).",
                    type(exc).__name__,
                )
                raise EmbeddingServiceError(
                    "Unable to initialize Gemini embeddings. Check that GOOGLE_API_KEY "
                    "is set correctly in your .env file."
                ) from exc
        return self._client

    def embed_text(self, text: str) -> List[float]:
        """Embed a single piece of text, returning its embedding vector."""
        validated_text = self._validate_text(text)
        try:
            return self._embeddings.embed_query(validated_text)
        except EmbeddingServiceError:
            raise
        except Exception as exc:
            logger.error("Gemini embedding request failed (%s).", type(exc).__name__)
            raise EmbeddingServiceError(
                "Failed to generate an embedding. Please try again."
            ) from exc

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts, preserving the exact input order."""
        if texts is None or not isinstance(texts, list):
            raise EmbeddingServiceError("texts must be a list of strings.")
        if not texts:
            raise EmbeddingServiceError("No texts were provided to embed.")

        validated_texts = [self._validate_text(text) for text in texts]

        try:
            return self._embeddings.embed_documents(
                validated_texts, batch_size=self._batch_size
            )
        except EmbeddingServiceError:
            raise
        except Exception as exc:
            logger.error("Gemini batch embedding request failed (%s).", type(exc).__name__)
            raise EmbeddingServiceError(
                "Failed to generate embeddings. Please try again."
            ) from exc

    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """Embed LangChain Documents by their page_content, preserving order.

        The original Document objects are not modified.
        """
        if documents is None or not isinstance(documents, list):
            raise EmbeddingServiceError("documents must be a list of Document objects.")
        if not documents:
            raise EmbeddingServiceError("No documents were provided to embed.")

        texts = [document.page_content for document in documents]
        return self.embed_texts(texts)

    def get_embedding_dimension(self) -> int:
        """Return the vector length produced by this model.

        There is no static way to know a Gemini embedding model's dimension
        without calling it, so this makes one embedding request — avoid
        calling it more than necessary.
        """
        return len(self.embed_text("dimension probe"))

    @staticmethod
    def _validate_text(text: str) -> str:
        if text is None:
            raise EmbeddingServiceError("text must be a string, not None.")
        if not isinstance(text, str):
            raise EmbeddingServiceError(f"text must be a string, got {type(text).__name__}.")
        if not text.strip():
            raise EmbeddingServiceError("Cannot embed empty or whitespace-only text.")
        return text


if __name__ == "__main__":
    # Developer-only smoke test against the real Gemini API.
    # Requires a valid GOOGLE_API_KEY. Never run automatically by the app or tests.
    # Usage: python -m retrieval.embeddings
    sample_text = "DocuMind Gemini embedding smoke test."
    smoke_service = GeminiEmbeddingService()
    embedding_vector = smoke_service.embed_text(sample_text)
    print(f"Embedded text: {sample_text!r}")
    print(f"Vector length: {len(embedding_vector)}")
