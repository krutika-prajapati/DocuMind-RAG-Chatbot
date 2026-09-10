"""Tests for the Gemini embedding service (retrieval/embeddings.py).

The real GoogleGenerativeAIEmbeddings client is always mocked here — these
tests never require a GOOGLE_API_KEY or network access.
"""

import logging

import pytest
from langchain_core.documents import Document

from retrieval.embeddings import EmbeddingServiceError, GeminiEmbeddingService


class FakeGoogleEmbeddings:
    """Stand-in for GoogleGenerativeAIEmbeddings used to avoid real API calls."""

    def __init__(self, model=None, **kwargs):
        self.model = model
        self.embed_documents_calls = []

    def embed_query(self, text):
        return [float(len(text)), 0.5, 0.25]

    def embed_documents(self, texts, batch_size=None):
        self.embed_documents_calls.append((texts, batch_size))
        return [[float(len(text)), 0.1] for text in texts]


@pytest.fixture
def patched_embeddings(monkeypatch):
    """Patch the underlying GoogleGenerativeAIEmbeddings class and capture the instance."""
    created = {}

    def factory(**kwargs):
        instance = FakeGoogleEmbeddings(**kwargs)
        created["instance"] = instance
        return instance

    monkeypatch.setattr("retrieval.embeddings.GoogleGenerativeAIEmbeddings", factory)
    return created


def test_service_initializes_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    service = GeminiEmbeddingService(model="models/gemini-embedding-001")

    assert service is not None


def test_service_requires_a_configured_model():
    with pytest.raises(EmbeddingServiceError):
        GeminiEmbeddingService(model="")


def test_configured_model_is_passed_to_client(patched_embeddings):
    service = GeminiEmbeddingService(model="models/gemini-embedding-001")
    service.embed_text("hello world")

    assert patched_embeddings["instance"].model == "models/gemini-embedding-001"


def test_embed_text_returns_vector(patched_embeddings):
    service = GeminiEmbeddingService(model="models/gemini-embedding-001")

    vector = service.embed_text("hello world")

    assert vector == [11.0, 0.5, 0.25]


def test_embed_texts_preserves_order(patched_embeddings):
    service = GeminiEmbeddingService(model="models/gemini-embedding-001")
    texts = ["alpha", "bravo charlie", "d"]

    vectors = service.embed_texts(texts)

    assert vectors == [[5.0, 0.1], [13.0, 0.1], [1.0, 0.1]]


def test_embed_texts_uses_configured_batch_size(patched_embeddings):
    service = GeminiEmbeddingService(model="models/gemini-embedding-001", batch_size=7)

    service.embed_texts(["a", "b"])

    _, batch_size_used = patched_embeddings["instance"].embed_documents_calls[0]
    assert batch_size_used == 7


def test_embed_documents_uses_page_content_and_preserves_order(patched_embeddings):
    documents = [
        Document(page_content="first document text", metadata={"file_name": "a.txt"}),
        Document(page_content="second", metadata={"file_name": "b.txt"}),
    ]
    original_contents = [doc.page_content for doc in documents]
    original_metadata = [dict(doc.metadata) for doc in documents]

    service = GeminiEmbeddingService(model="models/gemini-embedding-001")
    vectors = service.embed_documents(documents)

    assert len(vectors) == 2
    texts_sent, _ = patched_embeddings["instance"].embed_documents_calls[0]
    assert texts_sent == ["first document text", "second"]
    # Original Document objects must be untouched.
    assert [doc.page_content for doc in documents] == original_contents
    assert [dict(doc.metadata) for doc in documents] == original_metadata


def test_embed_text_rejects_none(patched_embeddings):
    service = GeminiEmbeddingService(model="models/gemini-embedding-001")
    with pytest.raises(EmbeddingServiceError):
        service.embed_text(None)


def test_embed_text_rejects_invalid_type(patched_embeddings):
    service = GeminiEmbeddingService(model="models/gemini-embedding-001")
    with pytest.raises(EmbeddingServiceError):
        service.embed_text(12345)


def test_embed_text_rejects_whitespace_only(patched_embeddings):
    service = GeminiEmbeddingService(model="models/gemini-embedding-001")
    with pytest.raises(EmbeddingServiceError):
        service.embed_text("   \n\t  ")


def test_embed_texts_rejects_none(patched_embeddings):
    service = GeminiEmbeddingService(model="models/gemini-embedding-001")
    with pytest.raises(EmbeddingServiceError):
        service.embed_texts(None)


def test_embed_texts_rejects_non_list(patched_embeddings):
    service = GeminiEmbeddingService(model="models/gemini-embedding-001")
    with pytest.raises(EmbeddingServiceError):
        service.embed_texts("not a list")


def test_embed_texts_rejects_empty_list(patched_embeddings):
    service = GeminiEmbeddingService(model="models/gemini-embedding-001")
    with pytest.raises(EmbeddingServiceError):
        service.embed_texts([])


def test_embed_texts_rejects_whitespace_only_entry(patched_embeddings):
    service = GeminiEmbeddingService(model="models/gemini-embedding-001")
    with pytest.raises(EmbeddingServiceError):
        service.embed_texts(["valid text", "   "])


def test_embed_documents_rejects_none(patched_embeddings):
    service = GeminiEmbeddingService(model="models/gemini-embedding-001")
    with pytest.raises(EmbeddingServiceError):
        service.embed_documents(None)


def test_embed_documents_rejects_empty_list(patched_embeddings):
    service = GeminiEmbeddingService(model="models/gemini-embedding-001")
    with pytest.raises(EmbeddingServiceError):
        service.embed_documents([])


def test_client_initialization_error_is_wrapped(monkeypatch):
    def raise_missing_key(**kwargs):
        raise ValueError(
            "API key required for Gemini Developer API. Provide api_key parameter "
            "or set GOOGLE_API_KEY/GEMINI_API_KEY environment variable."
        )

    monkeypatch.setattr("retrieval.embeddings.GoogleGenerativeAIEmbeddings", raise_missing_key)

    service = GeminiEmbeddingService(model="models/gemini-embedding-001")

    with pytest.raises(EmbeddingServiceError):
        service.embed_text("hello")


def test_api_key_never_appears_in_error_message(monkeypatch):
    fake_secret = "SUPER-SECRET-API-KEY-VALUE"

    class ExplodingEmbeddings:
        def __init__(self, **kwargs):
            pass

        def embed_query(self, text):
            raise RuntimeError(f"401 Unauthorized using key {fake_secret}")

    monkeypatch.setattr("retrieval.embeddings.GoogleGenerativeAIEmbeddings", ExplodingEmbeddings)

    service = GeminiEmbeddingService(model="models/gemini-embedding-001")

    with pytest.raises(EmbeddingServiceError) as exc_info:
        service.embed_text("hello")

    assert fake_secret not in str(exc_info.value)


def test_api_key_never_logged(monkeypatch, caplog):
    fake_secret = "SUPER-SECRET-API-KEY-VALUE"

    class ExplodingEmbeddings:
        def __init__(self, **kwargs):
            pass

        def embed_query(self, text):
            raise RuntimeError(f"401 Unauthorized using key {fake_secret}")

    monkeypatch.setattr("retrieval.embeddings.GoogleGenerativeAIEmbeddings", ExplodingEmbeddings)

    service = GeminiEmbeddingService(model="models/gemini-embedding-001")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(EmbeddingServiceError):
            service.embed_text("hello")

    assert fake_secret not in caplog.text


def test_get_embedding_dimension_returns_vector_length(patched_embeddings):
    service = GeminiEmbeddingService(model="models/gemini-embedding-001")

    dimension = service.get_embedding_dimension()

    assert dimension == 3  # FakeGoogleEmbeddings.embed_query returns a 3-element vector
