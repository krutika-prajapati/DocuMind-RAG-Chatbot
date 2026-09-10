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


# --- Multi-document diversification (broad/generic queries must not be
# accidentally answered from just one indexed document) ---------------


class DominantMinorityEmbeddingService:
    """Deterministic embedding for constructing a controlled "one document
    dominates a flat top-k search" scenario.

    Text starting with "CLOSE" embeds near the origin; text starting with
    "FAR" embeds far away; anything else (a neutral/broad query) embeds at
    the origin, so it is equally "close" to every CLOSE-tagged chunk and
    equally "far" from every FAR-tagged chunk.
    """

    def embed_texts(self, texts):
        return [self._vector(text) for text in texts]

    def embed_text(self, text):
        return self._vector(text)

    @staticmethod
    def _vector(text: str):
        if text.startswith("CLOSE"):
            return [0.1, 0.0]
        if text.startswith("FAR"):
            return [10.0, 0.0]
        return [0.0, 0.0]


def _make_dominant_minority_store(tmp_path, dominant_count=5, minority_count=2):
    """A knowledge base with one "dominant" document (many CLOSE chunks)
    and one "minority" document (few FAR chunks) — a flat top-k search for
    a neutral/broad query returns only the dominant document's chunks
    unless retrieval diversifies across documents.
    """
    store = ChromaVectorStore(
        embedding_service=DominantMinorityEmbeddingService(),
        persist_directory=tmp_path,
        collection_name="dominant_minority_tests",
    )
    chunks = []
    for i in range(dominant_count):
        chunks.append(
            _make_chunk(
                f"CLOSE: dominant document chunk {i} about general operations.",
                f"dom-{i}",
                "doc-dominant",
                source="dominant.txt",
                file_name="dominant.txt",
                file_type="txt",
                chunk_index=i,
                chunk_size=10,
            )
        )
    for i in range(minority_count):
        chunks.append(
            _make_chunk(
                f"FAR: minority document chunk {i} about a completely different topic.",
                f"min-{i}",
                "doc-minority",
                source="minority.txt",
                file_name="minority.txt",
                file_type="txt",
                chunk_index=i,
                chunk_size=10,
            )
        )
    store.add_chunks(chunks)
    return store


def test_flat_search_alone_would_exclude_the_minority_document(tmp_path):
    """Sanity check proving the underlying bug is real at the raw-search
    layer (bypassing DocumentRetriever's diversification entirely) — the
    minority document's chunks are completely absent from a flat top-k
    search for a broad/neutral query.
    """
    store = _make_dominant_minority_store(tmp_path)

    raw_results = store.similarity_search("QUERY: broad neutral question", top_k=3)

    document_hashes = {doc.metadata["document_hash"] for doc, _ in raw_results}
    assert document_hashes == {"doc-dominant"}


def test_broad_query_includes_every_indexed_document(tmp_path):
    store = _make_dominant_minority_store(tmp_path)
    retriever = DocumentRetriever(vector_store=store)

    results = retriever.retrieve("QUERY: broad neutral question", top_k=3)

    document_hashes = {chunk.document_hash for chunk in results}
    assert document_hashes == {"doc-dominant", "doc-minority"}
    assert len(results) == 3


def test_diversified_results_are_still_sorted_by_distance(tmp_path):
    store = _make_dominant_minority_store(tmp_path)
    retriever = DocumentRetriever(vector_store=store)

    results = retriever.retrieve("QUERY: broad neutral question", top_k=3)

    distances = [chunk.distance for chunk in results]
    assert distances == sorted(distances)


def test_query_specific_to_one_document_still_ranks_it_first(tmp_path):
    """Diversification must not weaken precision: a query that clearly
    matches one document's content still ranks that document's chunks
    first, even though the other document is now also represented.
    """
    store = _make_dominant_minority_store(tmp_path)
    retriever = DocumentRetriever(vector_store=store)

    results = retriever.retrieve("FAR: a question specifically about the minority topic", top_k=3)

    assert results[0].document_hash == "doc-minority"


def test_single_document_knowledge_base_is_unaffected(tmp_path):
    store = ChromaVectorStore(
        embedding_service=DominantMinorityEmbeddingService(), persist_directory=tmp_path
    )
    store.add_chunks(
        [
            _make_chunk(
                "CLOSE: the only document's only chunk.",
                "only-1",
                "doc-only",
                file_name="only.txt",
                file_type="txt",
                chunk_index=0,
            )
        ]
    )
    retriever = DocumentRetriever(vector_store=store)

    results = retriever.retrieve("QUERY: anything", top_k=5)

    assert len(results) == 1
    assert results[0].document_hash == "doc-only"


def test_explicit_document_hash_filter_bypasses_diversification(tmp_path):
    """Requesting one specific document (document_hash filter) must behave
    exactly as before — only that document's chunks are ever returned.
    """
    store = _make_dominant_minority_store(tmp_path)
    retriever = DocumentRetriever(vector_store=store)

    results = retriever.retrieve(
        "QUERY: broad neutral question", top_k=3, document_hash="doc-minority"
    )

    assert len(results) > 0
    assert all(chunk.document_hash == "doc-minority" for chunk in results)


def test_diversification_falls_back_to_flat_search_if_listing_documents_fails(
    tmp_path, monkeypatch
):
    store = _make_dominant_minority_store(tmp_path)
    retriever = DocumentRetriever(vector_store=store)

    def explode(*args, **kwargs):
        raise VectorStoreError("cannot list documents")

    monkeypatch.setattr(store, "list_documents", explode)

    results = retriever.retrieve("QUERY: broad neutral question", top_k=3)

    # Degrades gracefully to a plain flat search rather than crashing.
    assert len(results) == 3


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
