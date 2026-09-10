"""Tests for semantic retrieval (retrieval/retriever.py).

Gemini embeddings are always mocked (via a fake, deterministic embedding
service), so these tests never require GOOGLE_API_KEY or network access.
Chroma itself runs locally against a temporary directory (pytest's
tmp_path) — the only way to genuinely verify search/ordering behavior.
"""

import inspect

import pytest
from langchain_core.documents import Document

import retrieval.retriever as retriever_module
from retrieval.retriever import (
    MAX_TOP_K,
    DocumentRetriever,
    RetrievalError,
    RetrievedChunk,
    validate_top_k,
)
from retrieval.vector_store import ChromaVectorStore, VectorStoreError

# Embeddings are one count per keyword, so distance is predictable and
# meaningful: a text about "python" embeds close to a query about "python"
# and far from a text about "chocolate" or "rocket".
_KEYWORDS = ["python", "chocolate", "rocket"]


class FakeEmbeddingService:
    """Deterministic, keyword-count embedding — no network calls."""

    def embed_texts(self, texts):
        return [self._vector(text) for text in texts]

    def embed_text(self, text):
        return self._vector(text)

    @staticmethod
    def _vector(text: str):
        lowered = text.lower()
        return [float(lowered.count(keyword)) for keyword in _KEYWORDS]


def _make_chunk(text: str, chunk_id: str, document_hash: str, **metadata) -> Document:
    metadata = {
        "chunk_id": chunk_id,
        "document_hash": document_hash,
        **metadata,
    }
    return Document(page_content=text, metadata=metadata)


def _make_store(tmp_path, **kwargs) -> ChromaVectorStore:
    return ChromaVectorStore(
        embedding_service=FakeEmbeddingService(),
        persist_directory=tmp_path,
        **kwargs,
    )


@pytest.fixture
def populated_store(tmp_path):
    store = _make_store(tmp_path)
    chunks = [
        _make_chunk(
            "Python python python is a popular programming language.",
            "py-1",
            "doc-python",
            source="python_guide.txt",
            file_name="python_guide.txt",
            file_type="txt",
            chunk_index=0,
            chunk_size=10,
        ),
        _make_chunk(
            "Chocolate chocolate chocolate cake recipe with rich frosting.",
            "choc-1",
            "doc-chocolate",
            source="baking.pdf",
            file_name="baking.pdf",
            file_type="pdf",
            page=2,
            chunk_index=0,
            chunk_size=10,
        ),
        _make_chunk(
            "Rocket rocket rocket engines use powerful propulsion systems.",
            "rocket-1",
            "doc-rocket",
            source="space.docx",
            file_name="space.docx",
            file_type="docx",
            chunk_index=0,
            chunk_size=10,
        ),
    ]
    store.add_chunks(chunks)
    return store


def test_valid_query_returns_relevant_chunks(populated_store):
    retriever = DocumentRetriever(vector_store=populated_store)

    results = retriever.retrieve("python python python", top_k=1)

    assert len(results) == 1
    assert results[0].chunk_id == "py-1"


def test_top_k_is_respected(populated_store):
    retriever = DocumentRetriever(vector_store=populated_store)

    results = retriever.retrieve("python python python", top_k=2)

    assert len(results) == 2


def test_results_are_ordered_by_relevance(populated_store):
    retriever = DocumentRetriever(vector_store=populated_store)

    results = retriever.retrieve("python python python", top_k=3)

    assert [r.chunk_id for r in results][0] == "py-1"
    distances = [r.distance for r in results]
    assert distances == sorted(distances)  # ascending: most similar (lowest) first


def test_query_validation_rejects_none(populated_store):
    retriever = DocumentRetriever(vector_store=populated_store)
    with pytest.raises(RetrievalError):
        retriever.retrieve(None)


def test_query_validation_rejects_non_string(populated_store):
    retriever = DocumentRetriever(vector_store=populated_store)
    with pytest.raises(RetrievalError):
        retriever.retrieve(12345)


def test_query_validation_rejects_empty_string(populated_store):
    retriever = DocumentRetriever(vector_store=populated_store)
    with pytest.raises(RetrievalError):
        retriever.retrieve("")


def test_query_validation_rejects_whitespace_only(populated_store):
    retriever = DocumentRetriever(vector_store=populated_store)
    with pytest.raises(RetrievalError):
        retriever.retrieve("   \n\t  ")


@pytest.mark.parametrize("bad_top_k", [0, -1, 3.5, "5", True, MAX_TOP_K + 1])
def test_invalid_top_k_is_rejected(populated_store, bad_top_k):
    retriever = DocumentRetriever(vector_store=populated_store)
    with pytest.raises(RetrievalError):
        retriever.retrieve("python", top_k=bad_top_k)


def test_validate_top_k_accepts_reasonable_values():
    validate_top_k(1)
    validate_top_k(5)
    validate_top_k(MAX_TOP_K)


def test_empty_database_returns_no_results_cleanly(tmp_path):
    store = _make_store(tmp_path)
    retriever = DocumentRetriever(vector_store=store)

    results = retriever.retrieve("anything at all")

    assert results == []


def test_retrieved_metadata_is_preserved(populated_store):
    retriever = DocumentRetriever(vector_store=populated_store)

    results = retriever.retrieve("chocolate chocolate chocolate", top_k=1)

    chunk = results[0]
    assert chunk.source == "baking.pdf"
    assert chunk.file_name == "baking.pdf"
    assert chunk.file_type == "pdf"
    assert chunk.page == 2
    assert chunk.chunk_index == 0
    assert chunk.metadata["file_name"] == "baking.pdf"


def test_chunk_id_is_preserved(populated_store):
    retriever = DocumentRetriever(vector_store=populated_store)

    results = retriever.retrieve("rocket rocket rocket", top_k=1)

    assert results[0].chunk_id == "rocket-1"


def test_document_hash_is_preserved(populated_store):
    retriever = DocumentRetriever(vector_store=populated_store)

    results = retriever.retrieve("rocket rocket rocket", top_k=1)

    assert results[0].document_hash == "doc-rocket"


def test_missing_optional_page_is_handled_safely(populated_store):
    retriever = DocumentRetriever(vector_store=populated_store)

    results = retriever.retrieve("python python python", top_k=1)

    assert results[0].page is None  # TXT chunk has no page metadata


def test_multiple_documents_can_be_searched(populated_store):
    retriever = DocumentRetriever(vector_store=populated_store)

    results = retriever.retrieve("python OR chocolate OR rocket", top_k=3)
    document_hashes = {chunk.document_hash for chunk in results}

    assert document_hashes == {"doc-python", "doc-chocolate", "doc-rocket"}


def test_no_document_filter_searches_all_documents(populated_store):
    retriever = DocumentRetriever(vector_store=populated_store)

    results = retriever.retrieve("python python python", top_k=3)

    assert len(results) == 3


def test_document_hash_filter_restricts_to_one_document(populated_store):
    retriever = DocumentRetriever(vector_store=populated_store)

    # Query is closest to the python chunk, but filtering to the chocolate
    # document must return only that document's chunk, not the closer one.
    results = retriever.retrieve(
        "python python python", top_k=3, document_hash="doc-chocolate"
    )

    assert len(results) == 1
    assert results[0].document_hash == "doc-chocolate"
    assert results[0].chunk_id == "choc-1"


def test_document_hash_filter_rejects_invalid_value(populated_store):
    retriever = DocumentRetriever(vector_store=populated_store)
    with pytest.raises(RetrievalError):
        retriever.retrieve("python", document_hash="   ")


def test_distance_field_reflects_raw_chroma_score(populated_store):
    retriever = DocumentRetriever(vector_store=populated_store)

    results = retriever.retrieve("python python python", top_k=1)
    direct_results = populated_store.similarity_search("python python python", top_k=1)

    assert results[0].distance == direct_results[0][1]


def test_retriever_module_does_not_import_llm_or_chat():
    source = inspect.getsource(retriever_module)
    assert "llm." not in source
    assert "import llm" not in source
    assert "chat.service" not in source
    assert "chat_model" not in source.lower()


def test_vector_store_errors_are_wrapped_safely(populated_store, monkeypatch):
    # ChromaVectorStore.similarity_search already converts low-level Chroma
    # failures into VectorStoreError; simulate that and confirm the
    # retriever wraps it again into its own RetrievalError, without leaking
    # the underlying failure's details.
    def raise_vector_store_error(*args, **kwargs):
        raise VectorStoreError("internal chroma failure detail")

    monkeypatch.setattr(populated_store, "similarity_search", raise_vector_store_error)
    retriever = DocumentRetriever(vector_store=populated_store)

    with pytest.raises(RetrievalError) as exc_info:
        retriever.retrieve("python")

    assert "internal chroma failure detail" not in str(exc_info.value)


def test_secrets_are_not_exposed_in_retrieval_errors(populated_store, monkeypatch, caplog):
    import logging

    fake_secret = "SUPER-SECRET-CONNECTION-STRING"

    def explode(*args, **kwargs):
        raise VectorStoreError(f"connection failed using {fake_secret}")

    monkeypatch.setattr(populated_store, "similarity_search", explode)
    retriever = DocumentRetriever(vector_store=populated_store)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RetrievalError) as exc_info:
            retriever.retrieve("python")

    assert fake_secret not in str(exc_info.value)
    assert fake_secret not in caplog.text


def test_end_to_end_chunk_embed_store_query_retrieve(tmp_path):
    """Integration test: chunk -> (mocked) embed -> Chroma -> query -> retrieve."""
    from ingestion.chunking import chunk_documents
    from utils.hashing import compute_document_hash

    python_text = " ".join(["Python is great for data science."] * 40)
    chocolate_text = " ".join(["Chocolate desserts are delicious."] * 40)

    python_document = Document(
        page_content=python_text,
        metadata={"source": "python.txt", "file_name": "python.txt", "file_type": "txt"},
    )
    chocolate_document = Document(
        page_content=chocolate_text,
        metadata={"source": "chocolate.txt", "file_name": "chocolate.txt", "file_type": "txt"},
    )

    python_hash = compute_document_hash(python_text.encode("utf-8"))
    chocolate_hash = compute_document_hash(chocolate_text.encode("utf-8"))

    python_chunks = chunk_documents(
        [python_document], document_hash=python_hash, chunk_size=200, chunk_overlap=40
    )
    chocolate_chunks = chunk_documents(
        [chocolate_document], document_hash=chocolate_hash, chunk_size=200, chunk_overlap=40
    )

    store = _make_store(tmp_path)
    store.add_chunks(python_chunks)
    store.add_chunks(chocolate_chunks)

    retriever = DocumentRetriever(vector_store=store)
    results = retriever.retrieve("python python python", top_k=2)

    assert len(results) == 2
    assert results[0].document_hash == python_hash
    assert results[0].chunk_id in {chunk.metadata["chunk_id"] for chunk in python_chunks}
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].distance <= results[1].distance
