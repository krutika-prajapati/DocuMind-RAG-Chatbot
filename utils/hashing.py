"""Deterministic hashing utilities for content and chunk identification."""

import hashlib


def sha256_hex(text: str) -> str:
    """Return the hex-encoded SHA-256 digest of the given text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compute_document_hash(file_bytes: bytes) -> str:
    """Return a stable hash identifying a document's original file content."""
    return hashlib.sha256(file_bytes).hexdigest()


def compute_chunk_id(
    document_hash: str, source_key: str, chunk_index: int, chunk_content: str
) -> str:
    """Derive a deterministic, unique ID for one chunk.

    Combines the parent document's hash, its source/page key, its position,
    and its content so re-processing the same document with the same
    chunking configuration always yields the same chunk IDs, while distinct
    chunks never collide.
    """
    payload = f"{document_hash}:{source_key}:{chunk_index}:{chunk_content}"
    return sha256_hex(payload)
