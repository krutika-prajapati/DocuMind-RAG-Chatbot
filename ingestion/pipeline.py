"""Document ingestion pipeline: validate, save, load, and chunk uploaded documents.

    Uploaded File -> Validation -> Save -> Load -> Chunk -> Chunked Documents

Embedding and vector storage are implemented in later phases.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE, MAX_FILE_SIZE_MB, UPLOADS_DIR
from ingestion.chunking import chunk_documents
from ingestion.loaders import DocumentParsingError, load_document
from utils.hashing import compute_document_hash
from utils.logging_config import get_logger
from utils.validation import (
    sanitize_filename,
    validate_file_extension,
    validate_file_size,
)

logger = get_logger(__name__)


@dataclass
class ProcessedDocument:
    """Result of processing one uploaded file through the ingestion pipeline."""

    file_name: str
    file_type: str
    saved_path: Path
    documents: List[Document] = field(default_factory=list)
    chunks: List[Document] = field(default_factory=list)

    @property
    def num_documents(self) -> int:
        return len(self.documents)

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def average_chunk_size(self) -> float:
        """Average character count across chunks (0 if there are none)."""
        if not self.chunks:
            return 0.0
        return sum(len(chunk.page_content) for chunk in self.chunks) / len(self.chunks)


def save_uploaded_file(
    file_name: str,
    file_bytes: bytes,
    upload_dir: Optional[Path] = None,
    max_size_mb: int = MAX_FILE_SIZE_MB,
) -> Path:
    """Validate, sanitize, and persist an uploaded file to disk.

    Raises UnsupportedFileTypeError, EmptyFileError, or FileTooLargeError
    if validation fails. Returns the path to the saved file.
    """
    validate_file_extension(file_name)
    validate_file_size(len(file_bytes), max_size_mb=max_size_mb)

    target_dir = upload_dir or UPLOADS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    saved_path = target_dir / sanitize_filename(file_name)
    saved_path.write_bytes(file_bytes)

    return saved_path


def process_uploaded_file(
    file_name: str,
    file_bytes: bytes,
    upload_dir: Optional[Path] = None,
    max_size_mb: int = MAX_FILE_SIZE_MB,
    chunk: bool = True,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> ProcessedDocument:
    """Run the ingestion pipeline for one uploaded file.

    Validation -> Save -> Load -> (optionally) Chunk.
    """
    saved_path = save_uploaded_file(
        file_name, file_bytes, upload_dir=upload_dir, max_size_mb=max_size_mb
    )

    try:
        documents = load_document(saved_path)
    except DocumentParsingError:
        raise
    except Exception as exc:
        logger.exception("Unexpected error loading document: %s", saved_path)
        raise DocumentParsingError(
            "Unable to read this document. Please check that the file is valid."
        ) from exc

    chunks: List[Document] = []
    if chunk:
        document_hash = compute_document_hash(file_bytes)
        chunks = chunk_documents(
            documents,
            document_hash=document_hash,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    return ProcessedDocument(
        file_name=saved_path.name,
        file_type=saved_path.suffix.lower().lstrip("."),
        saved_path=saved_path,
        documents=documents,
        chunks=chunks,
    )
