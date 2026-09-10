"""Tests for document indexing orchestration (ingestion/indexing_service.py).

Gemini embeddings are always mocked (via a fake embedding service), and
Chroma runs against a real temporary directory (pytest's tmp_path) — the
same pattern used by tests/test_vector_store.py. No real Google API call
happens anywhere in this file.
"""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from ingestion.chunking import chunk_documents
from ingestion.indexing_service import DocumentIndexingService, IndexingError, IndexingResult
from ingestion.pipeline import ProcessedDocument, process_uploaded_file
from retrieval.vector_store import ChromaVectorStore, VectorStoreError
from utils.hashing import compute_document_hash


class FakeEmbeddingService:
    """Deterministic embedding — no network calls."""

    def embed_texts(self, texts):
        return [[float(len(t)), 0.0] for t in texts]

    def embed_text(self, text):
        return [float(len(text)), 0.0]


def _make_store(tmp_path, **kwargs) -> ChromaVectorStore:
    return ChromaVectorStore(
        embedding_service=FakeEmbeddingService(),
        persist_directory=tmp_path,
        **kwargs,
    )


def _make_processed_document(text: str, file_name: str, file_type: str = "txt") -> ProcessedDocument:
    """Build a ProcessedDocument the same way process_uploaded_file() would."""
    document = Document(
        page_content=text,
        metadata={"source": file_name, "file_name": file_name, "file_type": file_type},
    )
    document_hash = compute_document_hash(text.encode("utf-8"))
    chunks = chunk_documents(
        [document], document_hash=document_hash, chunk_size=200, chunk_overlap=40
    )
    return ProcessedDocument(
        file_name=file_name,
        file_type=file_type,
        saved_path=Path(file_name),
        documents=[document],
        chunks=chunks,
    )


def test_index_processed_document_success(tmp_path):
    store = _make_store(tmp_path)
    service = DocumentIndexingService(vector_store=store)
    processed = _make_processed_document(
        "The vessel has a cargo capacity of 10,000 tons. " * 20, "vessel.txt"
    )

    result = service.index_processed_document(processed)

    assert isinstance(result, IndexingResult)
    assert result.already_indexed is False
    assert result.chunks_indexed == processed.num_chunks
    assert result.chunks_indexed > 0

    stats = service.get_stats()
    assert stats["total_chunks"] == processed.num_chunks
    assert stats["document_count"] == 1


def test_duplicate_document_is_not_reindexed(tmp_path):
    store = _make_store(tmp_path)
    service = DocumentIndexingService(vector_store=store)
    processed = _make_processed_document(
        "The vessel has a cargo capacity of 10,000 tons. " * 20, "vessel.txt"
    )

    first = service.index_processed_document(processed)
    second = service.index_processed_document(processed)

    assert first.already_indexed is False
    assert first.chunks_indexed > 0
    assert second.already_indexed is True
    assert second.chunks_indexed == 0  # no redundant embedding/storage work
    assert second.document_hash == first.document_hash

    # No duplicate chunks were created.
    stats = service.get_stats()
    assert stats["total_chunks"] == processed.num_chunks
    assert stats["document_count"] == 1


def test_changed_document_content_is_indexed_as_new_version(tmp_path):
    store = _make_store(tmp_path)
    service = DocumentIndexingService(vector_store=store)

    original = _make_processed_document("Original content. " * 30, "vessel.txt")
    changed = _make_processed_document("Completely different content. " * 30, "vessel.txt")

    first = service.index_processed_document(original)
    second = service.index_processed_document(changed)

    assert first.document_hash != second.document_hash
    assert second.already_indexed is False
    assert second.chunks_indexed > 0

    # Both versions coexist until one is explicitly deleted.
    stats = service.get_stats()
    assert stats["document_count"] == 2


def test_index_processed_document_rejects_document_with_no_chunks(tmp_path):
    store = _make_store(tmp_path)
    service = DocumentIndexingService(vector_store=store)
    empty_processed = ProcessedDocument(
        file_name="empty.txt", file_type="txt", saved_path=Path("empty.txt"), chunks=[]
    )

    with pytest.raises(IndexingError):
        service.index_processed_document(empty_processed)


def test_index_processed_document_rejects_chunk_missing_document_hash(tmp_path):
    store = _make_store(tmp_path)
    service = DocumentIndexingService(vector_store=store)
    chunk = Document(page_content="content", metadata={"chunk_id": "c1"})
    processed = ProcessedDocument(
        file_name="bad.txt", file_type="txt", saved_path=Path("bad.txt"), chunks=[chunk]
    )

    with pytest.raises(IndexingError):
        service.index_processed_document(processed)


def test_indexing_error_is_raised_safely_on_vector_store_failure(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    service = DocumentIndexingService(vector_store=store)
    processed = _make_processed_document("Some content here. " * 20, "vessel.txt")

    def explode(*args, **kwargs):
        raise VectorStoreError("internal chroma failure detail")

    monkeypatch.setattr(store, "add_chunks", explode)

    with pytest.raises(IndexingError) as exc_info:
        service.index_processed_document(processed)

    assert "internal chroma failure detail" not in str(exc_info.value)


def test_delete_document_removes_all_its_chunks(tmp_path):
    store = _make_store(tmp_path)
    service = DocumentIndexingService(vector_store=store)
    processed = _make_processed_document("Some content here. " * 20, "vessel.txt")
    result = service.index_processed_document(processed)

    deleted_count = service.delete_document(result.document_hash)

    assert deleted_count == result.chunks_indexed
    assert service.get_stats() == {"total_chunks": 0, "document_count": 0}
    assert service.list_documents() == []


def test_delete_document_error_is_raised_safely(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    service = DocumentIndexingService(vector_store=store)

    def explode(*args, **kwargs):
        raise VectorStoreError("internal delete failure detail")

    monkeypatch.setattr(store, "delete_document", explode)

    with pytest.raises(IndexingError) as exc_info:
        service.delete_document("some-hash")

    assert "internal delete failure detail" not in str(exc_info.value)


def test_list_documents_returns_summary_for_each_indexed_document(tmp_path):
    store = _make_store(tmp_path)
    service = DocumentIndexingService(vector_store=store)
    vessel = _make_processed_document("Vessel content. " * 20, "vessel.txt")
    baking = _make_processed_document("Baking content. " * 20, "baking.pdf", file_type="pdf")

    service.index_processed_document(vessel)
    service.index_processed_document(baking)

    documents = service.list_documents()
    file_names = {doc.file_name for doc in documents}

    assert len(documents) == 2
    assert file_names == {"vessel.txt", "baking.pdf"}
    for doc in documents:
        assert doc.chunk_count > 0


def test_list_documents_empty_when_nothing_indexed(tmp_path):
    store = _make_store(tmp_path)
    service = DocumentIndexingService(vector_store=store)

    assert service.list_documents() == []


def test_list_documents_error_is_raised_safely(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    service = DocumentIndexingService(vector_store=store)

    def explode(*args, **kwargs):
        raise VectorStoreError("internal list failure detail")

    monkeypatch.setattr(store, "list_documents", explode)

    with pytest.raises(IndexingError) as exc_info:
        service.list_documents()

    assert "internal list failure detail" not in str(exc_info.value)


def test_get_stats_error_is_raised_safely(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    service = DocumentIndexingService(vector_store=store)

    def explode(*args, **kwargs):
        raise VectorStoreError("internal stats failure detail")

    monkeypatch.setattr(store, "get_stats", explode)

    with pytest.raises(IndexingError) as exc_info:
        service.get_stats()

    assert "internal stats failure detail" not in str(exc_info.value)


def test_end_to_end_with_real_process_uploaded_file(tmp_path):
    """Integration test: real ingestion pipeline -> indexing service -> Chroma."""
    store = _make_store(tmp_path)
    service = DocumentIndexingService(vector_store=store)
    uploads_dir = tmp_path / "uploads"

    processed = process_uploaded_file(
        "handbook.txt",
        (b"Employee handbook safety requirements. " * 30),
        upload_dir=uploads_dir,
    )

    result = service.index_processed_document(processed)

    assert result.already_indexed is False
    assert result.chunks_indexed == processed.num_chunks
    documents = service.list_documents()
    assert len(documents) == 1
    assert documents[0].file_name == "handbook.txt"
    assert documents[0].file_type == "txt"
