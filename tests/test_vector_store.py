"""Tests for persistent ChromaDB vector storage (retrieval/vector_store.py).

Gemini embeddings are always mocked (via a fake embedding service) so these
tests never require GOOGLE_API_KEY or network access. Chroma itself runs
locally against a temporary directory (pytest's tmp_path), which is the only
way to genuinely verify persistence behavior.
"""

import pytest
from langchain_core.documents import Document

from ingestion.chunking import chunk_documents
from retrieval.vector_store import ChromaVectorStore, IndexedDocument, VectorStoreError
from utils.hashing import compute_document_hash


class FakeEmbeddingService:
    """Deterministic stand-in for GeminiEmbeddingService — no network calls."""

    def embed_texts(self, texts):
        return [self._vector(text) for text in texts]

    def embed_text(self, text):
        return self._vector(text)

    @staticmethod
    def _vector(text: str):
        return [float(len(text)), float(sum(ord(c) for c in text) % 100)]


def _make_chunk(text: str, chunk_id: str, **metadata) -> Document:
    metadata = {**metadata, "chunk_id": chunk_id}
    return Document(page_content=text, metadata=metadata)


def _make_store(tmp_path, **kwargs) -> ChromaVectorStore:
    return ChromaVectorStore(
        embedding_service=FakeEmbeddingService(),
        persist_directory=tmp_path,
        **kwargs,
    )


def test_vector_store_initializes(tmp_path):
    store = _make_store(tmp_path)

    assert store is not None
    assert store.get_stats()["total_chunks"] == 0


def test_persist_directory_is_created_and_used(tmp_path):
    target_dir = tmp_path / "nested" / "chroma"
    assert not target_dir.exists()

    store = _make_store(target_dir)

    assert target_dir.exists()
    assert store.persist_directory == target_dir


def test_collection_name_is_configurable(tmp_path):
    store = _make_store(tmp_path, collection_name="custom_collection")

    assert store.collection_name == "custom_collection"
    assert store._store._collection.name == "custom_collection"


def test_add_chunks_stores_records(tmp_path):
    store = _make_store(tmp_path)
    chunks = [
        _make_chunk("first chunk text", "id-1", document_hash="doc-1"),
        _make_chunk("second chunk text", "id-2", document_hash="doc-1"),
    ]

    stored = store.add_chunks(chunks)

    assert stored == 2
    assert store.get_stats()["total_chunks"] == 2


def test_chunk_metadata_is_preserved(tmp_path):
    store = _make_store(tmp_path)
    chunk = _make_chunk(
        "policy content",
        "id-meta",
        source="company_policy.pdf",
        file_name="company_policy.pdf",
        file_type="pdf",
        page=4,
        chunk_index=0,
        chunk_size=len("policy content"),
        document_hash="doc-hash-1",
    )

    store.add_chunks([chunk])

    result = store._store.get(ids=["id-meta"], include=["metadatas", "documents"])
    assert result["documents"] == ["policy content"]
    metadata = result["metadatas"][0]
    assert metadata["source"] == "company_policy.pdf"
    assert metadata["file_name"] == "company_policy.pdf"
    assert metadata["file_type"] == "pdf"
    assert metadata["page"] == 4
    assert metadata["chunk_index"] == 0
    assert metadata["chunk_size"] == len("policy content")
    assert metadata["chunk_id"] == "id-meta"
    assert metadata["document_hash"] == "doc-hash-1"


def test_chunk_id_is_used_as_chroma_id(tmp_path):
    store = _make_store(tmp_path)
    chunk = _make_chunk("some content", "my-deterministic-id", document_hash="doc-1")

    store.add_chunks([chunk])

    result = store._store.get(ids=["my-deterministic-id"])
    assert result["ids"] == ["my-deterministic-id"]


def test_readding_same_chunks_does_not_create_duplicates(tmp_path):
    store = _make_store(tmp_path)
    chunks = [_make_chunk("same content", "id-1", document_hash="doc-1")]

    store.add_chunks(chunks)
    store.add_chunks(chunks)  # process the same document a second time

    assert store.get_stats()["total_chunks"] == 1


def test_updating_existing_chunk_works(tmp_path):
    store = _make_store(tmp_path)
    original = _make_chunk("original text", "id-1", document_hash="doc-1")
    store.add_chunks([original])

    updated = _make_chunk("updated text", "id-1", document_hash="doc-1")
    store.add_chunks([updated])

    result = store._store.get(ids=["id-1"], include=["documents"])
    assert result["documents"] == ["updated text"]
    assert store.get_stats()["total_chunks"] == 1


def test_multiple_documents_coexist(tmp_path):
    store = _make_store(tmp_path)
    doc_a_chunks = [
        _make_chunk("doc a chunk 1", "a-1", document_hash="hash-a"),
        _make_chunk("doc a chunk 2", "a-2", document_hash="hash-a"),
    ]
    doc_b_chunks = [_make_chunk("doc b chunk 1", "b-1", document_hash="hash-b")]

    store.add_chunks(doc_a_chunks)
    store.add_chunks(doc_b_chunks)

    stats = store.get_stats()
    assert stats["total_chunks"] == 3
    assert stats["document_count"] == 2


def test_delete_document_removes_only_its_chunks(tmp_path):
    store = _make_store(tmp_path)
    doc_a_chunks = [
        _make_chunk("doc a chunk 1", "a-1", document_hash="hash-a"),
        _make_chunk("doc a chunk 2", "a-2", document_hash="hash-a"),
    ]
    doc_b_chunks = [_make_chunk("doc b chunk 1", "b-1", document_hash="hash-b")]
    store.add_chunks(doc_a_chunks)
    store.add_chunks(doc_b_chunks)

    deleted_count = store.delete_document("hash-a")

    assert deleted_count == 2
    stats = store.get_stats()
    assert stats["total_chunks"] == 1
    remaining = store._store.get(include=["metadatas"])
    assert all(m["document_hash"] == "hash-b" for m in remaining["metadatas"])


def test_delete_document_rejects_invalid_hash(tmp_path):
    store = _make_store(tmp_path)
    with pytest.raises(VectorStoreError):
        store.delete_document("")


def test_delete_document_with_no_matches_returns_zero(tmp_path):
    store = _make_store(tmp_path)
    store.add_chunks([_make_chunk("content", "id-1", document_hash="hash-a")])

    deleted_count = store.delete_document("hash-does-not-exist")

    assert deleted_count == 0
    assert store.get_stats()["total_chunks"] == 1


def test_add_chunks_rejects_empty_list(tmp_path):
    store = _make_store(tmp_path)
    with pytest.raises(VectorStoreError):
        store.add_chunks([])


def test_add_chunks_rejects_none(tmp_path):
    store = _make_store(tmp_path)
    with pytest.raises(VectorStoreError):
        store.add_chunks(None)


def test_add_chunks_rejects_invalid_type(tmp_path):
    store = _make_store(tmp_path)
    with pytest.raises(VectorStoreError):
        store.add_chunks(["not a document"])


def test_add_chunks_rejects_missing_chunk_id(tmp_path):
    store = _make_store(tmp_path)
    chunk = Document(page_content="content", metadata={"document_hash": "doc-1"})
    with pytest.raises(VectorStoreError):
        store.add_chunks([chunk])


def test_add_chunks_rejects_empty_page_content(tmp_path):
    store = _make_store(tmp_path)
    chunk = _make_chunk("   ", "id-1", document_hash="doc-1")
    with pytest.raises(VectorStoreError):
        store.add_chunks([chunk])


def test_vector_store_error_hides_internal_details(tmp_path, monkeypatch):
    store = _make_store(tmp_path)
    fake_secret = "SUPER-SECRET-CONNECTION-STRING"

    def explode(*args, **kwargs):
        raise RuntimeError(f"connection failed using {fake_secret}")

    monkeypatch.setattr(store._store, "add_documents", explode)

    with pytest.raises(VectorStoreError) as exc_info:
        store.add_chunks([_make_chunk("content", "id-1", document_hash="doc-1")])

    assert fake_secret not in str(exc_info.value)


def test_get_stats_on_empty_collection(tmp_path):
    store = _make_store(tmp_path)

    assert store.get_stats() == {"total_chunks": 0, "document_count": 0}


def test_list_documents_on_empty_collection(tmp_path):
    store = _make_store(tmp_path)

    assert store.list_documents() == []


def test_list_documents_summarizes_each_distinct_document(tmp_path):
    store = _make_store(tmp_path)
    store.add_chunks(
        [
            _make_chunk(
                "chunk a1",
                "a-1",
                document_hash="hash-a",
                file_name="a.txt",
                file_type="txt",
            ),
            _make_chunk(
                "chunk a2",
                "a-2",
                document_hash="hash-a",
                file_name="a.txt",
                file_type="txt",
            ),
            _make_chunk(
                "chunk b1",
                "b-1",
                document_hash="hash-b",
                file_name="b.pdf",
                file_type="pdf",
            ),
        ]
    )

    documents = store.list_documents()

    assert len(documents) == 2
    by_hash = {doc.document_hash: doc for doc in documents}
    assert by_hash["hash-a"] == IndexedDocument(
        document_hash="hash-a", file_name="a.txt", file_type="txt", chunk_count=2
    )
    assert by_hash["hash-b"] == IndexedDocument(
        document_hash="hash-b", file_name="b.pdf", file_type="pdf", chunk_count=1
    )


def test_list_documents_reflects_deletion(tmp_path):
    store = _make_store(tmp_path)
    store.add_chunks(
        [
            _make_chunk("chunk a1", "a-1", document_hash="hash-a", file_name="a.txt"),
            _make_chunk("chunk b1", "b-1", document_hash="hash-b", file_name="b.txt"),
        ]
    )

    store.delete_document("hash-a")
    documents = store.list_documents()

    assert len(documents) == 1
    assert documents[0].document_hash == "hash-b"


def test_persistence_across_two_vector_store_instances(tmp_path):
    """Integration test: chunk -> (mocked) embed -> store -> reopen -> verify."""
    text = " ".join(f"Sentence number {i} about DocuMind." for i in range(80))
    document = Document(
        page_content=text,
        metadata={"source": "handbook.txt", "file_name": "handbook.txt", "file_type": "txt"},
    )
    document_hash = compute_document_hash(text.encode("utf-8"))
    chunks = chunk_documents(
        [document], document_hash=document_hash, chunk_size=200, chunk_overlap=40
    )

    store = ChromaVectorStore(
        embedding_service=FakeEmbeddingService(),
        persist_directory=tmp_path,
        collection_name="persistence_test",
    )
    stored_count = store.add_chunks(chunks)
    assert stored_count == len(chunks)
    assert store.get_stats()["total_chunks"] == len(chunks)

    # A brand-new instance pointing at the same directory must see the same data.
    reopened_store = ChromaVectorStore(
        embedding_service=FakeEmbeddingService(),
        persist_directory=tmp_path,
        collection_name="persistence_test",
    )

    assert reopened_store.get_stats()["total_chunks"] == len(chunks)
    first_chunk_id = chunks[0].metadata["chunk_id"]
    result = reopened_store._store.get(ids=[first_chunk_id], include=["documents", "metadatas"])
    assert result["documents"] == [chunks[0].page_content]
    assert result["metadatas"][0]["document_hash"] == document_hash
