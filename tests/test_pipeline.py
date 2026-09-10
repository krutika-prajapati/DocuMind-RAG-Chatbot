"""Tests for the document ingestion pipeline (validate -> save -> load -> chunk)."""

import pytest

from ingestion.pipeline import ProcessedDocument, process_uploaded_file, save_uploaded_file
from utils.validation import EmptyFileError, UnsupportedFileTypeError


def test_save_uploaded_file_saves_and_sanitizes(tmp_path):
    saved_path = save_uploaded_file(
        "../../My Report (v1).txt", b"Hello world", upload_dir=tmp_path
    )

    assert saved_path.exists()
    assert saved_path.parent == tmp_path
    assert saved_path.name == "My_Report__v1_.txt"


def test_save_uploaded_file_rejects_unsupported_extension(tmp_path):
    with pytest.raises(UnsupportedFileTypeError):
        save_uploaded_file("malware.exe", b"data", upload_dir=tmp_path)


def test_save_uploaded_file_rejects_empty_file(tmp_path):
    with pytest.raises(EmptyFileError):
        save_uploaded_file("empty.txt", b"", upload_dir=tmp_path)


def test_process_uploaded_file_returns_documents(tmp_path):
    result = process_uploaded_file(
        "notes.txt", b"Some knowledge base content.", upload_dir=tmp_path
    )

    assert isinstance(result, ProcessedDocument)
    assert result.file_type == "txt"
    assert result.num_documents == 1
    assert "Some knowledge base content." in result.documents[0].page_content


def test_process_uploaded_file_produces_chunks_with_metadata(tmp_path):
    text = " ".join(f"word{i}" for i in range(300))
    result = process_uploaded_file("policy.txt", text.encode("utf-8"), upload_dir=tmp_path)

    assert result.num_chunks > 0
    assert result.average_chunk_size > 0
    first_chunk = result.chunks[0]
    assert first_chunk.metadata["file_name"] == "policy.txt"
    assert first_chunk.metadata["file_type"] == "txt"
    assert first_chunk.metadata["chunk_index"] == 0
    assert "chunk_id" in first_chunk.metadata
    assert "document_hash" in first_chunk.metadata


def test_process_uploaded_file_can_skip_chunking(tmp_path):
    result = process_uploaded_file(
        "notes.txt", b"Some knowledge base content.", upload_dir=tmp_path, chunk=False
    )

    assert result.num_chunks == 0
    assert result.chunks == []
    assert result.average_chunk_size == 0.0
    assert result.num_documents == 1
