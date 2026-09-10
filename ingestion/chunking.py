"""Document chunking: split LangChain Documents into smaller, embeddable chunks.

    LangChain Documents -> CHUNKING -> Chunked Documents

Parsing responsibility stays in ingestion/loaders.py; this module only splits
already-parsed Documents and attaches chunk-level metadata. Embedding and
vector storage are implemented in later phases.
"""

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE, validate_chunk_config
from utils.hashing import compute_chunk_id
from utils.logging_config import get_logger

logger = get_logger(__name__)

_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class NoChunksProducedError(Exception):
    """Raised when a set of documents yields zero usable chunks."""


def _source_key(document: Document) -> str:
    """Build a stable key identifying where a document's content came from."""
    source = document.metadata.get("source", "")
    page = document.metadata.get("page")
    return f"{source}:{page}" if page is not None else str(source)


def chunk_documents(
    documents: List[Document],
    document_hash: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[Document]:
    """Split LangChain Documents into smaller chunks with enriched metadata.

    Each chunk retains the original document's metadata and gains
    chunk_index, chunk_id, chunk_size, and document_hash. chunk_index starts
    at 0 for this call and increases across all input documents (e.g. all
    pages of one uploaded file) — a separate call for a different document
    starts at 0 again. document_hash identifies all chunks that came from
    the same uploaded file, for later bulk deletion. Whitespace-only
    documents and chunks are skipped safely.

    Raises ValueError if chunk_size/chunk_overlap is invalid, or
    NoChunksProducedError if no usable text is found across all documents.
    """
    validate_chunk_config(chunk_size, chunk_overlap)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_SEPARATORS,
    )

    chunks: List[Document] = []
    chunk_index = 0
    for document in documents:
        if not document.page_content or not document.page_content.strip():
            continue

        source_key = _source_key(document)
        for split_text in splitter.split_text(document.page_content):
            if not split_text.strip():
                continue

            metadata = dict(document.metadata)
            metadata["chunk_index"] = chunk_index
            metadata["chunk_size"] = len(split_text)
            metadata["document_hash"] = document_hash
            metadata["chunk_id"] = compute_chunk_id(
                document_hash=document_hash,
                source_key=source_key,
                chunk_index=chunk_index,
                chunk_content=split_text,
            )
            chunks.append(Document(page_content=split_text, metadata=metadata))
            chunk_index += 1

    if not chunks:
        logger.warning("No usable text found to chunk (document_hash=%s).", document_hash)
        raise NoChunksProducedError("No usable text was found to split into chunks.")

    return chunks
