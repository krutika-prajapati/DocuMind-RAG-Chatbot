"""Tests for file validation and filename sanitization."""

import pytest

from utils.validation import (
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    sanitize_filename,
    validate_file_extension,
    validate_file_size,
)


@pytest.mark.parametrize("file_name", ["report.pdf", "notes.docx", "readme.txt", "guide.md"])
def test_validate_file_extension_supported(file_name):
    validate_file_extension(file_name)  # should not raise


@pytest.mark.parametrize(
    "file_name", ["archive.zip", "image.png", "script.exe", "noextension"]
)
def test_validate_file_extension_unsupported(file_name):
    with pytest.raises(UnsupportedFileTypeError):
        validate_file_extension(file_name)


def test_validate_file_size_rejects_empty_file():
    with pytest.raises(EmptyFileError):
        validate_file_size(0)


def test_validate_file_size_rejects_oversized_file():
    with pytest.raises(FileTooLargeError):
        validate_file_size(50 * 1024 * 1024, max_size_mb=20)


def test_validate_file_size_accepts_valid_size():
    validate_file_size(1024, max_size_mb=20)  # should not raise


@pytest.mark.parametrize(
    "raw_name,expected",
    [
        ("../../etc/passwd.txt", "passwd.txt"),
        ("my report (final).pdf", "my_report__final_.pdf"),
        ("..hidden.md", "hidden.md"),
        ("C:\\Users\\name\\file.docx", "file.docx"),
    ],
)
def test_sanitize_filename(raw_name, expected):
    assert sanitize_filename(raw_name) == expected


def test_sanitize_filename_falls_back_when_stem_is_empty():
    assert sanitize_filename("....pdf") == "file.pdf"
