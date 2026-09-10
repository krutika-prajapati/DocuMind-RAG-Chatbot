"""Semantic retrieval over the persistent ChromaDB vector store.

    User Query -> Query Embedding -> Chroma Similarity Search -> Top-K Chunks

This module only retrieves relevant chunks. It does not call an LLM, rewrite
queries, or generate answers — those are implemented in later phases.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from config.settings import RETRIEVAL_TOP_K
from retrieval.vector_store import ChromaVectorStore, VectorStoreError
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Safety cap on top_k to prevent an accidentally huge retrieval request.
MAX_TOP_K = 50


class RetrievalError(Exception):
    """Raised when a retrieval request is invalid or the search fails."""


@dataclass
class RetrievedChunk:
    """One chunk returned by a similarity search, with its distance and metadata.

    `distance` is Chroma's raw similarity_search_with_score value for this
    collection's distance metric (L2/Euclidean by default): LOWER means MORE
    similar. It is a raw distance, not a normalized 0-1 similarity score —
    see retrieval/vector_store.py's similarity_search() for details.
    """

    page_content: str
    distance: float
    source: Optional[str]
    file_name: Optional[str]
    file_type: Optional[str]
    page: Optional[int]
    chunk_index: Optional[int]
    chunk_id: Optional[str]
    document_hash: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


def validate_top_k(top_k: int) -> None:
    """Raise RetrievalError if top_k is not a positive integer within a sane range."""
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise RetrievalError(f"top_k must be an integer, got {type(top_k).__name__}.")
    if top_k <= 0:
        raise RetrievalError(f"top_k must be greater than 0, got {top_k}.")
    if top_k > MAX_TOP_K:
        raise RetrievalError(f"top_k must not exceed {MAX_TOP_K}, got {top_k}.")


class DocumentRetriever:
    """Retrieves the most relevant stored chunks for a user query.

    Built entirely on the existing ChromaVectorStore (and, through it, the
    existing Gemini embedding path) — this class never talks to Chroma or
    Gemini directly, and it never generates an answer.
    """

    def __init__(
        self,
        vector_store: Optional[ChromaVectorStore] = None,
        default_top_k: int = RETRIEVAL_TOP_K,
    ) -> None:
        validate_top_k(default_top_k)
        self._vector_store = vector_store or ChromaVectorStore()
        self._default_top_k = default_top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_hash: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Return the top_k most relevant chunks for query, most relevant first.

        Pass document_hash to restrict the search to one uploaded document;
        omit it (default) to search across all indexed documents. Returns an
        empty list if the vector store has no indexed chunks (or none match
        an optional document_hash filter) — this never crashes and never
        calls an answer-generation LLM.
        """
        validated_query = self._validate_query(query)

        effective_top_k = self._default_top_k if top_k is None else top_k
        validate_top_k(effective_top_k)

        if document_hash is not None and (
            not isinstance(document_hash, str) or not document_hash.strip()
        ):
            raise RetrievalError("document_hash must be a non-empty string when provided.")

        try:
            results = self._vector_store.similarity_search(
                validated_query, top_k=effective_top_k, document_hash=document_hash
            )
        except VectorStoreError as exc:
            logger.error("Retrieval search failed (%s).", type(exc).__name__)
            raise RetrievalError(
                "Failed to search the knowledge base. Please try again."
            ) from exc

        return [self._to_retrieved_chunk(document, distance) for document, distance in results]

    @staticmethod
    def _to_retrieved_chunk(document: Document, distance: float) -> RetrievedChunk:
        metadata = document.metadata or {}
        return RetrievedChunk(
            page_content=document.page_content,
            distance=distance,
            source=metadata.get("source"),
            file_name=metadata.get("file_name"),
            file_type=metadata.get("file_type"),
            page=metadata.get("page"),
            chunk_index=metadata.get("chunk_index"),
            chunk_id=metadata.get("chunk_id"),
            document_hash=metadata.get("document_hash"),
            metadata=metadata,
        )

    @staticmethod
    def _validate_query(query: str) -> str:
        if query is None:
            raise RetrievalError("query must be a string, not None.")
        if not isinstance(query, str):
            raise RetrievalError(f"query must be a string, got {type(query).__name__}.")
        if not query.strip():
            raise RetrievalError("query cannot be empty or whitespace-only.")
        return query
