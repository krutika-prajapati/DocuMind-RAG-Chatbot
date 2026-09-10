"""File validation and filename sanitization utilities.

These helpers protect the ingestion pipeline from unsupported, empty,
oversized, or maliciously-named uploads before any file touches disk.
"""

import os
import re
from pathlib import Path

from config.settings import MAX_FILE_SIZE_MB, SUPPORTED_EXTENSIONS


class UnsupportedFileTypeError(Exception):
    """Raised when an uploaded file has an unsupported extension."""


class EmptyFileError(Exception):
    """Raised when an uploaded file contains no data."""


class FileTooLargeError(Exception):
    """Raised when an uploaded file exceeds the configured size limit."""


def get_file_extension(file_name: str) -> str:
    """Return the lowercased file extension (including the leading dot)."""
    return Path(file_name).suffix.lower()


def validate_file_extension(file_name: str) -> None:
    """Raise UnsupportedFileTypeError if the file's extension is not supported."""
    extension = get_file_extension(file_name)
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            "Unsupported file type. Please upload PDF, DOCX, TXT, or Markdown files."
        )


def validate_file_size(size_bytes: int, max_size_mb: int = MAX_FILE_SIZE_MB) -> None:
    """Raise EmptyFileError or FileTooLargeError if the file size is invalid."""
    if size_bytes <= 0:
        raise EmptyFileError("The uploaded document is empty and cannot be processed.")

    max_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise FileTooLargeError(
            f"The uploaded file exceeds the maximum allowed size of {max_size_mb} MB."
        )


def sanitize_filename(file_name: str) -> str:
    """Return a filesystem-safe version of an untrusted filename.

    Strips directory components (defeating path traversal such as
    "../../etc/passwd") and replaces characters other than alphanumerics,
    dots, dashes, and underscores with underscores.
    """
    base_name = os.path.basename(file_name.replace("\\", "/"))
    stem = Path(base_name).stem
    extension = Path(base_name).suffix

    safe_stem = re.sub(r"[^A-Za-z0-9._-]", "_", stem).strip(".") or "file"
    safe_extension = re.sub(r"[^A-Za-z0-9.]", "", extension)

    return f"{safe_stem}{safe_extension}"
