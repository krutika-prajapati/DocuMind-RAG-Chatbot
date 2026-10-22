"""Gemini embedding service built on langchain-google-genai.

This module only turns text/Documents into embedding vectors. Vector
storage and retrieval are implemented in later phases.
"""

import re
import time
from typing import Callable, List, Optional, TypeVar

from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config.settings import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL
from utils.logging_config import get_logger

logger = get_logger(__name__)

_T = TypeVar("_T")

# GoogleGenerativeAIEmbeddings has no retry/backoff of its own (unlike the
# chat client, which retries up to 6 times by default) -- a rate limit or
# transient server error is otherwise immediately fatal. This matters
# especially on a shared free-tier API key: a 429 can happen even when this
# process's own call volume is well-paced, simply because another
# concurrent user hit the same key. These constants bound that retry.
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_EMBEDDING_RETRIES = 4
_DEFAULT_RETRY_DELAY_SECONDS = 15.0
_MAX_RETRY_DELAY_SECONDS = 60.0
_RETRY_DELAY_PATTERN = re.compile(r"([\d.]+)")


class EmbeddingServiceError(Exception):
    """Raised when embedding configuration or an embedding request fails."""


def _underlying_api_error(exc: Exception):
    """Return the original google-genai APIError behind exc, if any.

    langchain_google_genai wraps the real error in its own
    GoogleGenerativeAIError via `raise ... from e`, so the original error
    (which carries the HTTP status code and Google's structured error
    details) is reachable as `__cause__`.
    """
    try:
        from google.genai.errors import APIError
    except ImportError:
        return None

    if isinstance(exc, APIError):
        return exc
    cause = getattr(exc, "__cause__", None)
    return cause if isinstance(cause, APIError) else None


def _extract_status_code(exc: Exception) -> Optional[int]:
    api_error = _underlying_api_error(exc)
    return getattr(api_error, "code", None) if api_error is not None else None


def _extract_retry_delay_seconds(exc: Exception) -> Optional[float]:
    """Best-effort extraction of Google's own reported retry delay (e.g.
    "18s" in a RESOURCE_EXHAUSTED response's RetryInfo), so a retry waits
    exactly as long as asked instead of guessing. Returns None if absent or
    unparseable -- callers fall back to a fixed default in that case."""
    api_error = _underlying_api_error(exc)
    if api_error is None:
        return None
    try:
        error_details = (api_error.details or {}).get("error", {}).get("details", [])
        for entry in error_details:
            retry_delay = entry.get("retryDelay")
            if retry_delay:
                match = _RETRY_DELAY_PATTERN.match(str(retry_delay))
                if match:
                    return float(match.group(1))
    except (AttributeError, TypeError, ValueError):
        return None
    return None


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
        return self._call_with_retry(
            lambda: self._embeddings.embed_query(validated_text),
            "Failed to generate an embedding. Please try again.",
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts, preserving the exact input order."""
        if texts is None or not isinstance(texts, list):
            raise EmbeddingServiceError("texts must be a list of strings.")
        if not texts:
            raise EmbeddingServiceError("No texts were provided to embed.")

        validated_texts = [self._validate_text(text) for text in texts]

        return self._call_with_retry(
            lambda: self._embeddings.embed_documents(
                validated_texts, batch_size=self._batch_size
            ),
            "Failed to generate embeddings. Please try again.",
        )

    def _call_with_retry(self, operation: Callable[[], _T], error_message: str) -> _T:
        """Run operation(), retrying on a retryable Gemini API error
        (429/500/502/503/504) with a delay that honors Google's own
        reported retryDelay when present. Any other exception, or a
        retryable one that persists past the retry budget, raises
        EmbeddingServiceError immediately -- this never retries silently
        forever and never masks a non-retryable failure.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_EMBEDDING_RETRIES + 1):
            try:
                return operation()
            except EmbeddingServiceError:
                raise
            except Exception as exc:
                last_exc = exc
                status_code = _extract_status_code(exc)
                is_last_attempt = attempt == _MAX_EMBEDDING_RETRIES
                if status_code not in _RETRYABLE_STATUS_CODES or is_last_attempt:
                    logger.error(
                        "Gemini embedding request failed (%s, status=%s).",
                        type(exc).__name__,
                        status_code,
                    )
                    raise EmbeddingServiceError(error_message) from exc

                delay = min(
                    _extract_retry_delay_seconds(exc) or _DEFAULT_RETRY_DELAY_SECONDS,
                    _MAX_RETRY_DELAY_SECONDS,
                )
                logger.warning(
                    "Gemini embedding request hit a retryable error (status=%s); "
                    "retrying in %.1fs (attempt %d/%d).",
                    status_code,
                    delay,
                    attempt + 1,
                    _MAX_EMBEDDING_RETRIES,
                )
                time.sleep(delay)

        raise EmbeddingServiceError(error_message) from last_exc  # pragma: no cover

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
