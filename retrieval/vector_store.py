"""Persistent ChromaDB vector storage for embedded document chunks.

    Chunked Documents -> Gemini Embeddings -> Persistent ChromaDB

This module stores, upserts, deletes, and reports statistics on chunk
vectors + metadata, and exposes a similarity-search primitive for
retrieval/retriever.py to build on. It does not interpret results or call
an LLM.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings as LangChainEmbeddings

from config.settings import CHROMA_COLLECTION_NAME, CHROMA_DIR, VECTOR_STORE_BATCH_SIZE
from retrieval.embeddings import GeminiEmbeddingService
from utils.logging_config import get_logger

logger = get_logger(__name__)


class VectorStoreError(Exception):
    """Raised when a vector store operation fails or receives invalid input."""


@dataclass
class IndexedDocument:
    """Summary of one distinct document currently in the vector store.

    Derived entirely from stored chunk metadata — document_hash is the
    document's identity; two uploads that produce the same document_hash
    are the same content and collapse into one entry here.
    """

    document_hash: str
    file_name: str
    file_type: str
    chunk_count: int


class _GeminiEmbeddingsAdapter(LangChainEmbeddings):
    """Adapts GeminiEmbeddingService to LangChain's Embeddings interface.

    Chroma calls `embedding_function.embed_documents(texts: list[str])`, but
    GeminiEmbeddingService.embed_documents() takes LangChain Document
    objects, not raw strings. This adapter bridges that gap by delegating to
    the service's string-based methods, without duplicating any embedding
    logic.
    """

    def __init__(self, service: GeminiEmbeddingService) -> None:
        self._service = service

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._service.embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._service.embed_text(text)


class ChromaVectorStore:
    """Persistent ChromaDB storage for embedded document chunks.

    Wraps langchain_chroma.Chroma so the rest of the app never touches
    Chroma directly. Storage is idempotent: chunks are upserted by their
    deterministic chunk_id, so re-processing the same document never
    creates duplicate vectors — it safely updates the existing ones.
    """

    def __init__(
        self,
        embedding_service: Optional[GeminiEmbeddingService] = None,
        persist_directory: Path = CHROMA_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        batch_size: int = VECTOR_STORE_BATCH_SIZE,
    ) -> None:
        self._batch_size = batch_size
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        service = embedding_service or GeminiEmbeddingService()

        try:
            self._store = Chroma(
                collection_name=collection_name,
                embedding_function=_GeminiEmbeddingsAdapter(service),
                persist_directory=str(self.persist_directory),
            )
        except Exception as exc:
            logger.error(
                "Failed to initialize Chroma collection (%s).", type(exc).__name__
            )
            raise VectorStoreError(
                "Unable to initialize the vector store. Check your Chroma configuration."
            ) from exc

    def add_chunks(self, chunks: List[Document]) -> int:
        """Add document chunks to the vector store, upserting by chunk_id.

        Uses each chunk's existing deterministic chunk_id as the Chroma
        record ID. Re-adding chunks that already exist (same chunk_id)
        updates them in place instead of creating duplicates, so this is
        safe to call repeatedly for the same document. Returns the number
        of chunks stored.

        All-or-nothing: a large document is embedded in several batches
        (VECTOR_STORE_BATCH_SIZE at a time), and if a later batch fails
        (e.g. a rate limit that outlasted the embedding service's own
        retries) after earlier batches already succeeded, those earlier
        batches are rolled back before raising. Without this, the document
        would be left permanently half-indexed, and -- since indexing is
        skipped for a document_hash that already has any chunks stored --
        a retry would silently never complete it.
        """
        self._validate_chunks(chunks)

        chunk_ids = [chunk.metadata["chunk_id"] for chunk in chunks]
        added_ids: List[str] = []

        try:
            stored = 0
            for start in range(0, len(chunks), self._batch_size):
                batch_chunks = chunks[start : start + self._batch_size]
                batch_ids = chunk_ids[start : start + self._batch_size]
                self._store.add_documents(documents=batch_chunks, ids=batch_ids)
                added_ids.extend(batch_ids)
                stored += len(batch_chunks)
            return stored
        except Exception as exc:
            logger.error("Failed to add chunks to Chroma (%s).", type(exc).__name__)
            self._rollback_partial_add(added_ids)
            raise VectorStoreError(
                "Failed to store document chunks. Please try again."
            ) from exc

    def _rollback_partial_add(self, added_ids: List[str]) -> None:
        """Best-effort delete of chunk ids already added before a later
        batch failed. Never raises -- a failed rollback is logged clearly
        so the resulting partial state is visible, but it never replaces
        or hides the original error that triggered the rollback.
        """
        if not added_ids:
            return
        try:
            self._store.delete(ids=added_ids)
            logger.warning(
                "Rolled back %d partially-added chunk(s) after a failed "
                "add_chunks() call, so this document is not left half-indexed.",
                len(added_ids),
            )
        except Exception:
            logger.error(
                "Failed to roll back %d partially-added chunk(s) after a failed "
                "add_chunks() call -- the vector store may now contain an "
                "incomplete set of this document's chunks.",
                len(added_ids),
            )

    def delete_document(self, document_hash: str) -> int:
        """Delete every chunk belonging to one uploaded document.

        Documents are identified by document_hash (not filename, since two
        different uploads can share a filename). Returns the number of
        chunks deleted.
        """
        if not document_hash or not isinstance(document_hash, str):
            raise VectorStoreError("document_hash must be a non-empty string.")

        try:
            matches = self._store.get(where={"document_hash": document_hash}, include=[])
            matching_ids = matches.get("ids") or []
            if matching_ids:
                self._store.delete(ids=matching_ids)
            return len(matching_ids)
        except Exception as exc:
            logger.error(
                "Failed to delete document chunks from Chroma (%s).", type(exc).__name__
            )
            raise VectorStoreError(
                "Failed to delete document chunks. Please try again."
            ) from exc

    def similarity_search(
        self,
        query: str,
        top_k: int,
        document_hash: Optional[str] = None,
    ) -> List[Tuple[Document, float]]:
        """Return the top_k chunks most similar to query, with raw distances.

        Distance is Chroma's raw similarity_search_with_score value for this
        collection's configured distance metric (L2/Euclidean by default) —
        LOWER means MORE similar. It is not a normalized 0-1 similarity
        score. Results are already ordered most-similar-first by Chroma.
        Pass document_hash to restrict the search to one uploaded document;
        omit it (default) to search across all indexed documents.
        """
        if not query or not isinstance(query, str) or not query.strip():
            raise VectorStoreError("query must be a non-empty string.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise VectorStoreError("top_k must be a positive integer.")

        where = {"document_hash": document_hash} if document_hash else None

        try:
            return self._store.similarity_search_with_score(query, k=top_k, filter=where)
        except Exception as exc:
            logger.error("Chroma similarity search failed (%s).", type(exc).__name__)
            raise VectorStoreError(
                "Failed to search the vector store. Please try again."
            ) from exc

    def get_chunks_by_index(
        self, document_hash: str, chunk_indexes: List[int]
    ) -> List[Document]:
        """Return the stored chunks for one document at specific chunk_index values.

        A direct metadata lookup (not a similarity search) used to fetch a
        result's immediate neighbor chunks (chunk_index ± 1) so content split
        across adjacent chunks of the same document is more likely to be
        retrieved together. Never returns chunks from a different
        document_hash — chunk_index is only unique within one document.
        Returns an empty list if none of the requested indexes exist for
        this document (e.g. the result is already at the first or last
        chunk_index) or if chunk_indexes is empty.
        """
        if not document_hash or not isinstance(document_hash, str):
            raise VectorStoreError("document_hash must be a non-empty string.")
        if not chunk_indexes:
            return []

        try:
            result = self._store.get(
                where={
                    "$and": [
                        {"document_hash": document_hash},
                        {"chunk_index": {"$in": list(chunk_indexes)}},
                    ]
                },
                include=["documents", "metadatas"],
            )
        except Exception as exc:
            logger.error(
                "Failed to fetch neighbor chunks from Chroma (%s).", type(exc).__name__
            )
            raise VectorStoreError(
                "Failed to fetch neighboring chunks. Please try again."
            ) from exc

        texts = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(texts, metadatas)
        ]

    def list_documents(self) -> List[IndexedDocument]:
        """Return a summary of every distinct document currently indexed.

        Derived entirely from stored chunk metadata (document_hash,
        file_name, file_type) — no separate inventory is kept; Chroma's own
        metadata remains the single source of truth. This is a
        metadata-only scan (like get_stats()'s document_count), so its cost
        is proportional to the number of stored chunks, not embeddings.
        """
        try:
            result = self._store.get(include=["metadatas"])
        except Exception as exc:
            logger.error("Failed to list indexed documents (%s).", type(exc).__name__)
            raise VectorStoreError(
                "Failed to read the knowledge base. Please try again."
            ) from exc

        summaries: Dict[str, IndexedDocument] = {}
        for metadata in result.get("metadatas", []) or []:
            if not metadata:
                continue
            document_hash = metadata.get("document_hash")
            if not document_hash:
                continue
            if document_hash not in summaries:
                summaries[document_hash] = IndexedDocument(
                    document_hash=document_hash,
                    file_name=metadata.get("file_name") or "unknown",
                    file_type=metadata.get("file_type") or "unknown",
                    chunk_count=0,
                )
            summaries[document_hash].chunk_count += 1

        return list(summaries.values())

    def get_stats(self) -> Dict[str, int]:
        """Return basic collection statistics.

        total_chunks comes from Chroma's indexed record count (cheap).
        document_count is derived by scanning stored metadata (not
        embeddings) for distinct document_hash values, so it costs more
        than total_chunks but stays proportional to metadata size only.
        """
        try:
            total_chunks = self._store._collection.count()
        except Exception as exc:
            logger.error("Failed to read Chroma collection stats (%s).", type(exc).__name__)
            raise VectorStoreError("Failed to read vector store statistics.") from exc

        stats: Dict[str, int] = {"total_chunks": total_chunks}

        if total_chunks == 0:
            stats["document_count"] = 0
            return stats

        try:
            result = self._store.get(include=["metadatas"])
            document_hashes = {
                metadata["document_hash"]
                for metadata in result.get("metadatas", []) or []
                if metadata and metadata.get("document_hash")
            }
            stats["document_count"] = len(document_hashes)
        except Exception as exc:
            logger.warning(
                "Could not compute document_count from Chroma metadata (%s).",
                type(exc).__name__,
            )

        return stats

    @staticmethod
    def _validate_chunks(chunks: List[Document]) -> None:
        if chunks is None or not isinstance(chunks, list):
            raise VectorStoreError("chunks must be a list of Document objects.")
        if not chunks:
            raise VectorStoreError("No chunks were provided to store.")

        for index, chunk in enumerate(chunks):
            if not isinstance(chunk, Document):
                raise VectorStoreError(
                    f"chunks[{index}] is not a Document (got {type(chunk).__name__})."
                )
            if not chunk.page_content or not chunk.page_content.strip():
                raise VectorStoreError(f"chunks[{index}] has empty page_content.")
            if not isinstance(chunk.metadata, dict):
                raise VectorStoreError(f"chunks[{index}] has malformed metadata.")

            chunk_id = chunk.metadata.get("chunk_id")
            if not chunk_id or not isinstance(chunk_id, str):
                raise VectorStoreError(f"chunks[{index}] is missing a valid chunk_id.")
