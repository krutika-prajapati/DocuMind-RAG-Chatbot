"""Tests for document chunking and chunk metadata."""

import pytest
from langchain_core.documents import Document

from config.settings import validate_chunk_config
from ingestion.chunking import NoChunksProducedError, chunk_documents


def _make_document(text: str, **metadata) -> Document:
    return Document(page_content=text, metadata=metadata)


def test_normal_document_splits_into_multiple_chunks():
    text = " ".join(f"This is sentence number {i}." for i in range(200))
    document = _make_document(text, source="doc.txt", file_name="doc.txt", file_type="txt")

    chunks = chunk_documents([document], document_hash="hash1", chunk_size=200, chunk_overlap=40)

    assert len(chunks) > 1


def test_chunk_overlap_is_applied():
    text = "".join(str(i % 10) for i in range(1000))
    document = _make_document(
        text, source="numbers.txt", file_name="numbers.txt", file_type="txt"
    )

    chunks = chunk_documents([document], document_hash="hash2", chunk_size=50, chunk_overlap=15)

    assert len(chunks) > 1
    for current_chunk, next_chunk in zip(chunks, chunks[1:]):
        overlap_tail = current_chunk.page_content[-15:]
        overlap_head = next_chunk.page_content[:15]
        assert overlap_tail == overlap_head


def test_metadata_is_preserved_and_enriched():
    document = _make_document(
        ("A" * 50 + " ") + ("B" * 50 + " ") + ("C" * 50),
        source="company_policy.pdf",
        file_name="company_policy.pdf",
        file_type="pdf",
        page=4,
    )

    chunks = chunk_documents([document], document_hash="abc123", chunk_size=40, chunk_overlap=10)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["source"] == "company_policy.pdf"
        assert chunk.metadata["file_name"] == "company_policy.pdf"
        assert chunk.metadata["file_type"] == "pdf"
        assert chunk.metadata["page"] == 4
        assert "chunk_index" in chunk.metadata
        assert "chunk_id" in chunk.metadata
        assert chunk.metadata["chunk_size"] == len(chunk.page_content)
        assert chunk.metadata["document_hash"] == "abc123"


def test_chunk_index_starts_at_zero_per_call():
    document = _make_document(
        " ".join(f"word{i}" for i in range(300)),
        source="a.txt",
        file_name="a.txt",
        file_type="txt",
    )

    chunks = chunk_documents([document], document_hash="hash-a", chunk_size=100, chunk_overlap=20)

    assert chunks[0].metadata["chunk_index"] == 0
    indexes = [chunk.metadata["chunk_index"] for chunk in chunks]
    assert indexes == list(range(len(chunks)))


def test_chunk_ids_are_deterministic():
    document = _make_document(
        " ".join(f"item {i}" for i in range(100)),
        source="b.txt",
        file_name="b.txt",
        file_type="txt",
    )

    first_run = chunk_documents([document], document_hash="hash-b", chunk_size=80, chunk_overlap=20)
    second_run = chunk_documents([document], document_hash="hash-b", chunk_size=80, chunk_overlap=20)

    first_ids = [chunk.metadata["chunk_id"] for chunk in first_run]
    second_ids = [chunk.metadata["chunk_id"] for chunk in second_run]
    assert first_ids == second_ids


def test_chunk_ids_are_unique_across_chunks():
    document = _make_document(
        " ".join(f"unique token {i}" for i in range(150)),
        source="c.txt",
        file_name="c.txt",
        file_type="txt",
    )

    chunks = chunk_documents([document], document_hash="hash-c", chunk_size=80, chunk_overlap=20)
    ids = [chunk.metadata["chunk_id"] for chunk in chunks]

    assert len(ids) == len(set(ids))


def test_whitespace_only_document_is_ignored_alongside_valid_one():
    whitespace_document = _make_document(
        "   \n\t  ", source="mixed.txt", file_name="mixed.txt", file_type="txt"
    )
    valid_document = _make_document(
        "Real content that should be chunked.",
        source="mixed.txt",
        file_name="mixed.txt",
        file_type="txt",
    )

    chunks = chunk_documents(
        [whitespace_document, valid_document],
        document_hash="hash-d",
        chunk_size=100,
        chunk_overlap=20,
    )

    assert len(chunks) == 1
    assert chunks[0].page_content.strip() == "Real content that should be chunked."


def test_all_whitespace_documents_raise_clear_error():
    whitespace_document = _make_document(
        "   ", source="blank.txt", file_name="blank.txt", file_type="txt"
    )

    with pytest.raises(NoChunksProducedError):
        chunk_documents(
            [whitespace_document], document_hash="hash-e", chunk_size=100, chunk_overlap=20
        )


def test_different_documents_do_not_share_chunk_indexes():
    document_a = _make_document(
        " ".join(f"alpha {i}" for i in range(100)),
        source="a.txt",
        file_name="a.txt",
        file_type="txt",
    )
    document_b = _make_document(
        " ".join(f"beta {i}" for i in range(50)),
        source="b.txt",
        file_name="b.txt",
        file_type="txt",
    )

    chunks_a = chunk_documents([document_a], document_hash="hash-a2", chunk_size=80, chunk_overlap=20)
    chunks_b = chunk_documents([document_b], document_hash="hash-b2", chunk_size=80, chunk_overlap=20)

    assert chunks_a[0].metadata["chunk_index"] == 0
    assert chunks_b[0].metadata["chunk_index"] == 0


def test_invalid_chunk_configuration_is_rejected():
    with pytest.raises(ValueError):
        validate_chunk_config(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError):
        validate_chunk_config(chunk_size=100, chunk_overlap=150)

    validate_chunk_config(chunk_size=100, chunk_overlap=50)  # should not raise


def test_chunk_documents_rejects_invalid_configuration():
    document = _make_document(
        "Some content here.", source="x.txt", file_name="x.txt", file_type="txt"
    )

    with pytest.raises(ValueError):
        chunk_documents([document], document_hash="hash-f", chunk_size=50, chunk_overlap=50)
