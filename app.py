"""DocuMind - RAG Knowledge Assistant

Streamlit entry point (Phase 9): document upload/indexing + grounded chat UI.

This module is UI/orchestration only. All business logic lives in the
existing services it calls:

    ingestion.pipeline.process_uploaded_file  — validate, save, parse, chunk
    ingestion.indexing_service.DocumentIndexingService — embed, store, list, delete
    chat.service.RAGService — conversation-aware retrieval + grounded answers
    chat.memory.ConversationMemory — per-session chat history

Streamlit never calls Chroma, the Gemini SDK, or an embedding client
directly — only these service abstractions.
"""

import streamlit as st

from chat.memory import ConversationMemory
from chat.service import RAGService, RAGServiceError
from config.settings import GOOGLE_API_KEY, RETRIEVAL_TOP_K
from ingestion.chunking import NoChunksProducedError
from ingestion.indexing_service import DocumentIndexingService, IndexingError
from ingestion.loaders import DocumentParsingError
from ingestion.pipeline import process_uploaded_file
from utils.logging_config import get_logger
from utils.validation import EmptyFileError, FileTooLargeError, UnsupportedFileTypeError

logger = get_logger(__name__)

SUPPORTED_FORMATS_LABEL = "PDF · DOCX · TXT · Markdown"
SOURCE_PREVIEW_CHARS = 300

INGESTION_ERRORS = (
    UnsupportedFileTypeError,
    EmptyFileError,
    FileTooLargeError,
    DocumentParsingError,
    NoChunksProducedError,
)


# --- Shared service singletons -------------------------------------------
# These wrap stateless, shared resources (the one persistent Chroma
# collection and the Gemini clients) and are safe to share across every
# session. Conversation memory is NOT cached here — it is per-session state
# below, in st.session_state.


@st.cache_resource
def get_indexing_service() -> DocumentIndexingService:
    return DocumentIndexingService()


@st.cache_resource
def get_rag_service() -> RAGService:
    return RAGService()


# --- Session state (per browser session; never shared/global) -----------

if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = ConversationMemory()
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "top_k" not in st.session_state:
    st.session_state.top_k = RETRIEVAL_TOP_K
if "upload_status" not in st.session_state:
    st.session_state.upload_status = None


# --- Helpers ---------------------------------------------------------------


def _friendly_error(exc: Exception, fallback: str) -> str:
    """Return a safe, user-facing message — never a raw exception dump."""
    message = str(exc).strip()
    return message or fallback


def _handle_upload(uploaded_file) -> None:
    """Parse/chunk (existing pipeline), then embed/index (Phase 9 service)."""
    indexing_service = get_indexing_service()
    file_bytes = uploaded_file.getvalue()

    try:
        with st.spinner(f"Parsing and chunking {uploaded_file.name}..."):
            processed = process_uploaded_file(uploaded_file.name, file_bytes)

        with st.spinner("Generating embeddings and indexing..."):
            result = indexing_service.index_processed_document(processed)
    except INGESTION_ERRORS as exc:
        st.session_state.upload_status = {
            "status": "error",
            "message": _friendly_error(
                exc, "Unable to process this document. Please check the file and try again."
            ),
        }
        return
    except IndexingError as exc:
        st.session_state.upload_status = {
            "status": "error",
            "message": _friendly_error(
                exc, "Unable to index this document right now. Please try again."
            ),
        }
        return
    except Exception:
        logger.exception("Unexpected error indexing uploaded file: %s", uploaded_file.name)
        st.session_state.upload_status = {
            "status": "error",
            "message": "Something went wrong while processing this file. Please try again.",
        }
        return

    if result.already_indexed:
        st.session_state.upload_status = {
            "status": "info",
            "message": f"This document is already indexed: {processed.file_name}",
        }
    else:
        st.session_state.upload_status = {
            "status": "success",
            "message": f"Indexed successfully: {processed.file_name}",
            "file_name": processed.file_name,
            "file_type": processed.file_type.upper(),
            "num_documents": processed.num_documents,
            "num_chunks": processed.num_chunks,
            "average_chunk_size": processed.average_chunk_size,
        }


def _handle_delete(document_hash: str, file_name: str) -> None:
    try:
        get_indexing_service().delete_document(document_hash)
        st.session_state.upload_status = {
            "status": "success",
            "message": f"Removed from the knowledge base: {file_name}",
        }
    except IndexingError as exc:
        st.session_state.upload_status = {
            "status": "error",
            "message": _friendly_error(
                exc, "Unable to delete this document right now. Please try again."
            ),
        }
    except Exception:
        logger.exception("Unexpected error deleting document: %s", document_hash)
        st.session_state.upload_status = {
            "status": "error",
            "message": "Something went wrong while deleting this document. Please try again.",
        }


def _clear_conversation() -> None:
    st.session_state.conversation_memory.clear()
    st.session_state.chat_messages = []


def _render_sources(sources, retrieved_chunks) -> None:
    """Render deterministic, application-generated source citations.

    Chunk previews are shown with st.code() — as inert data, never
    interpreted as Markdown/HTML — since retrieved document content is
    untrusted (the same principle applied to the Gemini prompt itself).
    """
    if not sources:
        return
    with st.expander(f"📎 View sources ({len(sources)})"):
        for index, source in enumerate(sources, start=1):
            location = f"Page {source.page}" if source.page is not None else "N/A"
            st.markdown(
                f"**{index}. 📄 {source.file_name or 'unknown'}** — "
                f"{location} · Chunk {source.chunk_index}"
            )
            if index - 1 < len(retrieved_chunks):
                preview = retrieved_chunks[index - 1].page_content[:SOURCE_PREVIEW_CHARS]
                st.code(preview, language=None)
            if index < len(sources):
                st.divider()


def _render_answer_details(entry: dict) -> None:
    """A light, non-technical view of the RAG pipeline for this answer."""
    with st.expander("🔍 How this answer was generated"):
        st.markdown(
            "Question → Conversation-aware query → Semantic retrieval → "
            "Retrieved document chunks → Gemini → Grounded answer"
        )
        st.markdown(f"- **Original question:** {entry.get('original_query') or '—'}")
        retrieval_query = entry.get("retrieval_query")
        original_query = entry.get("original_query")
        if retrieval_query and retrieval_query != original_query:
            st.markdown(f"- **Rewritten search query:** {retrieval_query}")
        st.markdown(f"- **Chunks retrieved:** {entry.get('num_chunks', 0)}")


def _render_retrieval_details(entry: dict) -> None:
    """Collapsed, developer-oriented retrieval debug info. No secrets, no prompts."""
    chunks = entry.get("retrieved_chunks") or []
    with st.expander("🛠️ Retrieval details"):
        st.markdown(f"- **Retrieval query:** {entry.get('retrieval_query') or '—'}")
        st.markdown(f"- **Top K:** {entry.get('top_k')}")
        if not chunks:
            st.caption("No chunks were retrieved for this turn.")
            return
        rows = [
            {
                "Source": chunk.file_name or "unknown",
                "Page": chunk.page if chunk.page is not None else "—",
                "Chunk #": chunk.chunk_index,
                "Distance": round(chunk.distance, 4),
            }
            for chunk in chunks
        ]
        st.dataframe(rows, width="stretch", hide_index=True)


def _render_assistant_message(entry: dict) -> None:
    st.write(entry["content"])
    _render_sources(entry.get("sources") or [], entry.get("retrieved_chunks") or [])
    if entry.get("num_chunks") is not None:
        st.caption(f"Sources used: {entry['num_chunks']}")
    if entry.get("has_pipeline_details"):
        _render_answer_details(entry)
        _render_retrieval_details(entry)


# --- Page setup -------------------------------------------------------------

st.set_page_config(page_title="DocuMind — Grounded RAG Chatbot", page_icon="📄")

st.title("DocuMind — Grounded RAG Chatbot")
st.caption(
    "Ask questions about your uploaded documents and get answers grounded "
    "only in those documents."
)

if not GOOGLE_API_KEY:
    st.warning(
        "⚠️ Gemini API key is not configured. Please configure GOOGLE_API_KEY "
        "in the environment."
    )

try:
    kb_stats = get_indexing_service().get_stats()
    vector_store_ok = True
except IndexingError as exc:
    st.error(_friendly_error(exc, "Unable to read the knowledge base right now."))
    kb_stats = {"total_chunks": 0, "document_count": 0}
    vector_store_ok = False


# --- Sidebar -----------------------------------------------------------

with st.sidebar:
    st.header("📚 Knowledge Base")

    st.subheader("Upload Document")
    st.caption(f"Supported formats: {SUPPORTED_FORMATS_LABEL}")
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "docx", "txt", "md"],
        label_visibility="collapsed",
    )
    if st.button("Upload & Index", disabled=uploaded_file is None, width="stretch"):
        _handle_upload(uploaded_file)
        # Rerun only on a non-error outcome so the sidebar's "Indexed
        # Documents" list and "System Status" counts (and the main-area
        # empty-knowledge-base banner) reflect the newly indexed document
        # immediately, instead of waiting for the next unrelated
        # interaction. A failed upload/index leaves the current script run
        # to finish normally so the error message renders without an extra
        # rerun cycle.
        if st.session_state.upload_status.get("status") != "error":
            st.rerun()

    status = st.session_state.upload_status
    if status:
        if status["status"] == "success":
            st.success(status["message"])
            if "num_chunks" in status:
                st.markdown(
                    f"- **File type:** {status['file_type']}\n"
                    f"- **Parsed pages/documents:** {status['num_documents']}\n"
                    f"- **Chunks created:** {status['num_chunks']}\n"
                    f"- **Average chunk size:** ~{status['average_chunk_size']:.0f} characters"
                )
        elif status["status"] == "info":
            st.info(status["message"])
        else:
            st.error(status["message"])

    st.divider()

    st.subheader("Indexed Documents")
    try:
        indexed_documents = get_indexing_service().list_documents()
    except IndexingError as exc:
        indexed_documents = []
        st.error(_friendly_error(exc, "Unable to list indexed documents right now."))

    if not indexed_documents:
        st.caption("Your knowledge base is empty.")
        st.caption("Upload a document above to get started.")
    else:
        for document in indexed_documents:
            short_hash = document.document_hash[:10]
            with st.container(border=True):
                st.markdown(f"📄 **{document.file_name}**")
                st.caption(
                    f"Type: {document.file_type.upper()} · "
                    f"Chunks: {document.chunk_count} · Status: Indexed"
                )
                st.caption(f"Hash: `{short_hash}…`")
                if st.button(
                    "🗑️ Remove", key=f"delete_{document.document_hash}", width="stretch"
                ):
                    _handle_delete(document.document_hash, document.file_name)
                    st.rerun()

    st.divider()

    st.subheader("Retrieval Settings")
    st.session_state.top_k = st.slider(
        "Top K (chunks retrieved per question)",
        min_value=1,
        max_value=10,
        value=st.session_state.top_k,
    )

    st.divider()

    st.subheader("Chat Controls")
    if st.button("🗑️ Clear Conversation", width="stretch"):
        _clear_conversation()
        st.rerun()

    st.divider()

    st.subheader("System Status")
    st.caption(f"Documents indexed: {kb_stats.get('document_count', 0)}")
    st.caption(f"Chunks indexed: {kb_stats.get('total_chunks', 0)}")
    st.caption(f"Vector store: {'Ready' if vector_store_ok else 'Unavailable'}")


# --- Main chat area ----------------------------------------------------

if kb_stats.get("total_chunks", 0) == 0:
    st.info(
        "Your knowledge base is empty. Upload a document from the sidebar "
        "to get started.\n\n"
        f"Supported formats: {SUPPORTED_FORMATS_LABEL}"
    )

for entry in st.session_state.chat_messages:
    with st.chat_message(entry["role"]):
        if entry["role"] == "assistant":
            _render_assistant_message(entry)
        else:
            st.write(entry["content"])

question = st.chat_input("Ask a question about your documents...")

if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.chat_messages.append({"role": "user", "content": question})

    if kb_stats.get("total_chunks", 0) == 0:
        notice = "Upload and index at least one document before asking questions."
        with st.chat_message("assistant"):
            st.info(notice)
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": notice, "sources": []}
        )
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                try:
                    response = get_rag_service().answer_question(
                        question,
                        top_k=st.session_state.top_k,
                        memory=st.session_state.conversation_memory,
                    )
                    error_message = None
                except RAGServiceError as exc:
                    response = None
                    error_message = _friendly_error(
                        exc, "Unable to generate an answer right now. Please try again."
                    )
                except Exception:
                    logger.exception("Unexpected error answering question.")
                    response = None
                    error_message = "Unable to generate an answer right now. Please try again."

            if error_message:
                st.error(error_message)
                st.session_state.chat_messages.append(
                    {"role": "assistant", "content": error_message, "sources": []}
                )
            else:
                entry = {
                    "role": "assistant",
                    "content": response.answer,
                    "sources": response.sources,
                    "retrieved_chunks": response.retrieved_chunks,
                    "original_query": response.original_query,
                    "retrieval_query": response.retrieval_query,
                    "num_chunks": len(response.retrieved_chunks),
                    "top_k": st.session_state.top_k,
                    "has_pipeline_details": True,
                }
                _render_assistant_message(entry)
                st.session_state.chat_messages.append(entry)
