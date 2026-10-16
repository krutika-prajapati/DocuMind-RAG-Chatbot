"""Semantic retrieval over the persistent ChromaDB vector store.

    User Query -> Query Embedding -> Chroma Similarity Search -> Top-K Chunks

This module only retrieves relevant chunks. It does not call an LLM, rewrite
queries, or generate answers — those are implemented in later phases.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

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
        omit it (default) to search across all indexed documents — in that
        case, every indexed document is guaranteed a fair share of the
        candidate pool (see _search_across_all_documents), so a broad
        question ("tell me about both documents") isn't accidentally
        answered from just one of them. Returns an empty list if the vector
        store has no indexed chunks (or none match an optional
        document_hash filter) — this never crashes and never calls an
        answer-generation LLM.

        After the similarity search, each result's immediate same-document
        neighbor chunks (chunk_index +/- 1) are additionally fetched and
        appended as supplementary context (see _augment_with_neighbors) --
        this never reorders or removes the original results, so content
        split across two adjacent chunks (e.g. a financial table) is more
        likely to be answerable from a single retrieve() call.
        """
        validated_query = self._validate_query(query)

        effective_top_k = self._default_top_k if top_k is None else top_k
        validate_top_k(effective_top_k)

        if document_hash is not None and (
            not isinstance(document_hash, str) or not document_hash.strip()
        ):
            raise RetrievalError("document_hash must be a non-empty string when provided.")

        try:
            if document_hash is not None:
                results = self._vector_store.similarity_search(
                    validated_query, top_k=effective_top_k, document_hash=document_hash
                )
            else:
                results = self._search_across_all_documents(validated_query, effective_top_k)
        except VectorStoreError as exc:
            logger.error("Retrieval search failed (%s).", type(exc).__name__)
            raise RetrievalError(
                "Failed to search the knowledge base. Please try again."
            ) from exc

        augmented_results = self._augment_with_neighbors(results)
        return [
            self._to_retrieved_chunk(document, distance)
            for document, distance in augmented_results
        ]

    def _augment_with_neighbors(
        self, results: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """Supplement similarity results with each result's immediate
        same-document chunk neighbors (chunk_index - 1 and chunk_index + 1).

        Content such as a financial table or a multi-page note is sometimes
        split across two adjacent chunks by the character-based splitter; a
        question needing both halves may otherwise only retrieve one of
        them. This never reorders or replaces the original results -- they
        stay first, exactly as similarity search ranked them -- neighbors
        are appended afterward as supplementary context, deduplicated by
        chunk_id, and only ever looked up within the SAME document_hash as
        the result that requested them (chunk_index is only unique per
        document, never globally). If a neighbor lookup fails, the original
        results are still returned unchanged -- this never breaks retrieval.
        """
        if not results:
            return results

        seen_chunk_ids = {
            document.metadata.get("chunk_id")
            for document, _distance in results
            if document.metadata.get("chunk_id") is not None
        }

        # For every result, note which neighbor chunk_index values to fetch
        # per document_hash, keeping the best (lowest) distance among the
        # anchors that requested each index.
        wanted: Dict[str, Dict[int, float]] = {}
        for document, distance in results:
            document_hash = document.metadata.get("document_hash")
            chunk_index = document.metadata.get("chunk_index")
            if document_hash is None or not isinstance(chunk_index, int):
                continue
            for neighbor_index in (chunk_index - 1, chunk_index + 1):
                if neighbor_index < 0:
                    continue
                per_document = wanted.setdefault(document_hash, {})
                if neighbor_index not in per_document or distance < per_document[neighbor_index]:
                    per_document[neighbor_index] = distance

        supplementary: List[Tuple[Document, float]] = []
        for document_hash, indexes_to_distance in wanted.items():
            try:
                neighbor_documents = self._vector_store.get_chunks_by_index(
                    document_hash, list(indexes_to_distance.keys())
                )
            except VectorStoreError as exc:
                logger.warning(
                    "Could not fetch neighbor chunks for a result; continuing "
                    "without them (%s).",
                    type(exc).__name__,
                )
                continue

            for neighbor_document in neighbor_documents:
                chunk_id = neighbor_document.metadata.get("chunk_id")
                if chunk_id is not None and chunk_id in seen_chunk_ids:
                    continue
                if chunk_id is not None:
                    seen_chunk_ids.add(chunk_id)
                neighbor_index = neighbor_document.metadata.get("chunk_index")
                anchor_distance = indexes_to_distance.get(neighbor_index, 0.0)
                supplementary.append((neighbor_document, anchor_distance))

        return results + supplementary

    def _search_across_all_documents(
        self, query: str, top_k: int
    ) -> List[Tuple[Document, float]]:
        """Search every indexed document and return the best top_k overall,
        guaranteeing each document a fair share of the candidate pool.

        A single flat nearest-neighbor search across the whole collection
        has no guarantee that every indexed document is represented in the
        results — for a broad or generic query, all top_k matches can
        happen to come from just one document, silently excluding the
        others from the answer's context. To avoid that, this queries each
        document individually (via the existing document_hash filter) for
        its own best ceil(top_k / num_documents) chunks, merges every
        document's candidates, and returns the globally best top_k by
        distance. Whenever top_k >= the number of indexed documents, every
        document is guaranteed to contribute at least one candidate to that
        merged pool. With 0 or 1 indexed documents there's nothing to
        diversify, so this falls back to a single plain search.
        """
        try:
            indexed_documents = self._vector_store.list_documents()
        except VectorStoreError as exc:
            logger.warning(
                "Could not enumerate indexed documents for diversified retrieval; "
                "falling back to a single combined search (%s).",
                type(exc).__name__,
            )
            indexed_documents = []

        if len(indexed_documents) <= 1:
            return self._vector_store.similarity_search(query, top_k=top_k)

        per_document_quota = math.ceil(top_k / len(indexed_documents))
        candidates: List[Tuple[Document, float]] = []
        for document in indexed_documents:
            candidates.extend(
                self._vector_store.similarity_search(
                    query, top_k=per_document_quota, document_hash=document.document_hash
                )
            )

        candidates.sort(key=lambda candidate: candidate[1])
        return candidates[:top_k]

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
