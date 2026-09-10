"""Tests for document loading of TXT, Markdown, DOCX, and PDF files."""

import pytest
from docx import Document as DocxDocument

from ingestion.loaders import (
    DocumentParsingError,
    load_document,
    load_docx_document,
    load_markdown_document,
    load_pdf_document,
    load_txt_document,
)


def test_load_txt_document(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("Hello DocuMind", encoding="utf-8")

    documents = load_txt_document(file_path)

    assert len(documents) == 1
    assert "Hello DocuMind" in documents[0].page_content
    assert documents[0].metadata["file_name"] == "notes.txt"
    assert documents[0].metadata["file_type"] == "txt"
    assert documents[0].metadata["source"] == str(file_path)


def test_load_markdown_document(tmp_path):
    file_path = tmp_path / "guide.md"
    file_path.write_text("# Title\n\nSome content.", encoding="utf-8")

    documents = load_markdown_document(file_path)

    assert len(documents) == 1
    assert "Title" in documents[0].page_content
    assert documents[0].metadata["file_type"] == "md"


def test_load_docx_document(tmp_path):
    file_path = tmp_path / "handbook.docx"
    docx_file = DocxDocument()
    docx_file.add_paragraph("Employee handbook content.")
    docx_file.save(str(file_path))

    documents = load_docx_document(file_path)

    assert len(documents) == 1
    assert "Employee handbook content." in documents[0].page_content
    assert documents[0].metadata["file_type"] == "docx"
    assert documents[0].metadata["file_name"] == "handbook.docx"


def test_load_txt_document_rejects_empty_content(tmp_path):
    file_path = tmp_path / "empty.txt"
    file_path.write_text("   ", encoding="utf-8")

    with pytest.raises(DocumentParsingError):
        load_txt_document(file_path)


def test_load_pdf_document_rejects_corrupted_file(tmp_path):
    file_path = tmp_path / "broken.pdf"
    file_path.write_bytes(b"%PDF-1.4 this is not a real pdf stream")

    with pytest.raises(DocumentParsingError):
        load_pdf_document(file_path)


def test_load_document_dispatches_by_extension(tmp_path):
    file_path = tmp_path / "notes.txt"
    file_path.write_text("dispatch test", encoding="utf-8")

    documents = load_document(file_path)

    assert documents[0].metadata["file_type"] == "txt"


def test_load_document_unsupported_extension(tmp_path):
    file_path = tmp_path / "archive.zip"
    file_path.write_bytes(b"not a real zip")

    with pytest.raises(DocumentParsingError):
        load_document(file_path)
