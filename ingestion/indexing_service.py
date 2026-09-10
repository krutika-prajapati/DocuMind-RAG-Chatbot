"""Document indexing orchestration: chunks -> Gemini embeddings -> Chroma.

    ProcessedDocument.chunks -> GeminiEmbeddingService -> ChromaVectorStore

Connects the existing ingestion pipeline (ingestion/pipeline.py — validate,
save, parse, chunk) to persistent vector storage (retrieval/vector_store.py
— embed, upsert, delete, inventory). Reuses ChromaVectorStore as-is; no
embedding client or Chroma access is duplicated here, and this module never
talks to Chroma or Gemini directly itself.
"""

from dataclasses import dataclass
from typing import List, Optional

from ingestion.pipeline import ProcessedDocument
from retrieval.vector_store import ChromaVectorStore, IndexedDocument, VectorStoreError
from utils.logging_config import get_logger

logger = get_logger(__name__)


class IndexingError(Exception):
    """Raised when indexing, deleting, or listing documents fails."""


@dataclass
class IndexingResult:
    """Outcome of indexing one ProcessedDocument's chunks."""

    document_hash: str
    chunks_indexed: int
    already_indexed: bool


class DocumentIndexingService:
    """Embeds and stores a ProcessedDocument's chunks in the vector store.

    Idempotent by design: if a document with the same content
    (document_hash) is already indexed, re-indexing is skipped entirely —
    no redundant Gemini embedding calls are made, and no duplicate chunks
    are ever created. A file whose content actually changed gets a new
    document_hash and is indexed as a distinct entry alongside the old one;
    use delete_document() to remove an outdated version explicitly.
    """

    def __init__(self, vector_store: Optional[ChromaVectorStore] = None) -> None:
        self._vector_store = vector_store or ChromaVectorStore()

    def index_processed_document(self, processed: ProcessedDocument) -> IndexingResult:
        """Embed and store processed.chunks, skipping work if already indexed."""
        if not processed.chunks:
            raise IndexingError("This document produced no chunks to index.")

        document_hash = processed.chunks[0].metadata.get("document_hash")
        if not document_hash:
            raise IndexingError("This document is missing required indexing metadata.")

        if self._is_already_indexed(document_hash):
            return IndexingResult(
                document_hash=document_hash, chunks_indexed=0, already_indexed=True
            )

        try:
            chunks_indexed = self._vector_store.add_chunks(processed.chunks)
        except VectorStoreError as exc:
            logger.error("Failed to index document chunks (%s).", type(exc).__name__)
            raise IndexingError(
                "Failed to index this document. Please try again."
            ) from exc

        return IndexingResult(
            document_hash=document_hash, chunks_indexed=chunks_indexed, already_indexed=False
        )

    def delete_document(self, document_hash: str) -> int:
        """Remove every indexed chunk belonging to one document."""
        try:
            return self._vector_store.delete_document(document_hash)
        except VectorStoreError as exc:
            logger.error("Failed to delete document (%s).", type(exc).__name__)
            raise IndexingError(
                "Failed to delete this document. Please try again."
            ) from exc

    def list_documents(self) -> List[IndexedDocument]:
        """Return a summary of every currently indexed document."""
        try:
            return self._vector_store.list_documents()
        except VectorStoreError as exc:
            logger.error("Failed to list indexed documents (%s).", type(exc).__name__)
            raise IndexingError(
                "Failed to read the knowledge base. Please try again."
            ) from exc

    def get_stats(self) -> dict:
        """Return basic knowledge-base statistics (total_chunks, document_count)."""
        try:
            return self._vector_store.get_stats()
        except VectorStoreError as exc:
            logger.error("Failed to read vector store stats (%s).", type(exc).__name__)
            raise IndexingError(
                "Failed to read knowledge base statistics. Please try again."
            ) from exc

    def _is_already_indexed(self, document_hash: str) -> bool:
        try:
            documents = self._vector_store.list_documents()
        except VectorStoreError:
            return False
        return any(document.document_hash == document_hash for document in documents)
