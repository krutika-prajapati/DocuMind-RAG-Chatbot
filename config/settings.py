"""Application configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "100"))

LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))
if not (0.0 <= LLM_TEMPERATURE <= 2.0):
    raise ValueError(f"LLM_TEMPERATURE must be between 0.0 and 2.0, got {LLM_TEMPERATURE}.")

MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))
if MAX_CONTEXT_CHARS <= 0:
    raise ValueError(f"MAX_CONTEXT_CHARS must be greater than 0, got {MAX_CONTEXT_CHARS}.")

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "20"))
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "documind_documents")
VECTOR_STORE_BATCH_SIZE = int(os.getenv("VECTOR_STORE_BATCH_SIZE", "100"))

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "5"))
if RETRIEVAL_TOP_K <= 0:
    raise ValueError(f"RETRIEVAL_TOP_K must be greater than 0, got {RETRIEVAL_TOP_K}.")

MAX_MEMORY_MESSAGES = int(os.getenv("MAX_MEMORY_MESSAGES", "20"))
if MAX_MEMORY_MESSAGES <= 0:
    raise ValueError(f"MAX_MEMORY_MESSAGES must be greater than 0, got {MAX_MEMORY_MESSAGES}.")


def validate_chunk_config(chunk_size: int, chunk_overlap: int) -> None:
    """Raise ValueError if the chunk size/overlap combination is invalid."""
    if chunk_size <= 0:
        raise ValueError(f"CHUNK_SIZE must be greater than 0, got {chunk_size}.")
    if chunk_overlap < 0:
        raise ValueError(f"CHUNK_OVERLAP cannot be negative, got {chunk_overlap}.")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"CHUNK_OVERLAP ({chunk_overlap}) must be smaller than CHUNK_SIZE ({chunk_size})."
        )


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
validate_chunk_config(CHUNK_SIZE, CHUNK_OVERLAP)
