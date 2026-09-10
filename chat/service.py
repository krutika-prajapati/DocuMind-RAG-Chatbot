"""RAG orchestration: conversation-aware retrieval -> grounded prompt -> Gemini -> answer + sources.

    User Question -> (recent conversation history)
                   -> QueryRewriter -> Standalone Retrieval Query
                   -> DocumentRetriever -> Retrieved Chunks
                   -> Original Question + Retrieved Chunks -> Grounded RAG Prompt
                   -> Gemini -> Answer + Source References
                   -> Conversation Memory

Query rewriting only affects retrieval. Final answer generation always
receives the ORIGINAL user question plus the retrieved document context —
never the rewritten query, and never conversation history — so previous
(possibly wrong) assistant replies can never become factual evidence. This
layer never talks to Chroma or Gemini directly; it only uses
DocumentRetriever, QueryRewriter, and GeminiChatService.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from chat.memory import ConversationMemory
from llm.gemini import GeminiChatError, GeminiChatService
from llm.prompts import NO_ANSWER_FALLBACK, build_rag_prompt
from retrieval.query_rewriter import QueryRewriteError, QueryRewriter
from retrieval.retriever import DocumentRetriever, RetrievalError, RetrievedChunk
from utils.logging_config import get_logger

logger = get_logger(__name__)


class RAGServiceError(Exception):
    """Raised when a question-answering request is invalid or fails."""


@dataclass
class SourceReference:
    """A deterministic, application-generated citation for one retrieved chunk.

    Built entirely from RetrievedChunk metadata — Gemini never supplies or
    influences this data.
    """

    file_name: Optional[str]
    page: Optional[int]
    chunk_index: Optional[int]
    chunk_id: Optional[str]
    document_hash: Optional[str]


@dataclass
class RAGResponse:
    """The result of answer_question(): a grounded answer plus its sources.

    original_query and retrieval_query are optional, Phase-8 additions for
    observability (e.g. showing "searched for: ..." in a future UI) — they
    do not change how existing Phase 7 callers use answer or sources.
    """

    answer: str
    sources: List[SourceReference] = field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)
    original_query: Optional[str] = None
    retrieval_query: Optional[str] = None


class RAGService:
    """Answers a user's question strictly from retrieved document context.

    Reuses the existing DocumentRetriever (which itself reuses
    ChromaVectorStore), GeminiChatService, and QueryRewriter — no second
    retrieval implementation and no second Gemini client are created here.
    If retrieval finds nothing, Gemini is never called for an answer: the
    fixed fallback is returned immediately with no sources.

    Conversation memory is optional and per-call (pass a ConversationMemory
    instance scoped to one user session); omitting it reproduces the exact
    stateless Phase 7 behavior.
    """

    def __init__(
        self,
        retriever: Optional[DocumentRetriever] = None,
        chat_service: Optional[GeminiChatService] = None,
        query_rewriter: Optional[QueryRewriter] = None,
    ) -> None:
        self._retriever = retriever or DocumentRetriever()
        self._chat_service = chat_service or GeminiChatService()
        self._query_rewriter = query_rewriter or QueryRewriter(chat_service=self._chat_service)

    def answer_question(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_hash: Optional[str] = None,
        memory: Optional[ConversationMemory] = None,
    ) -> RAGResponse:
        """Retrieve relevant chunks and generate a grounded answer.

        If `memory` is provided, its recent history is used to rewrite `query`
        into a standalone retrieval query (skipped automatically when memory
        is empty), and the original question + generated answer are recorded
        back into `memory` after a successful call. Conversation history is
        never sent to the final answer-generation prompt and never used as a
        factual source — only retrieved document context is.

        Returns RAGResponse(answer=NO_ANSWER_FALLBACK, sources=[]) without
        calling Gemini for an answer if no chunks are retrieved.
        """
        validated_query = self._validate_query(query)

        history = memory.get_recent_messages() if memory is not None else []
        try:
            retrieval_query = self._query_rewriter.rewrite(validated_query, history)
        except QueryRewriteError as exc:
            logger.warning(
                "Query rewriting failed; falling back to the original question (%s).",
                type(exc).__name__,
            )
            retrieval_query = validated_query

        try:
            chunks = self._retriever.retrieve(
                retrieval_query, top_k=top_k, document_hash=document_hash
            )
        except RetrievalError as exc:
            logger.error(
                "Retrieval failed during answer_question (%s).", type(exc).__name__
            )
            raise RAGServiceError(
                "Failed to search the knowledge base. Please try again."
            ) from exc

        if not chunks:
            response = RAGResponse(
                answer=NO_ANSWER_FALLBACK,
                sources=[],
                retrieved_chunks=[],
                original_query=validated_query,
                retrieval_query=retrieval_query,
            )
            self._remember(memory, validated_query, response.answer)
            return response

        # Final answer generation uses the ORIGINAL question + retrieved
        # document context only — never the rewritten query, never
        # conversation history.
        system_prompt, user_prompt = build_rag_prompt(validated_query, chunks)

        try:
            answer = self._chat_service.generate(system_prompt, user_prompt)
        except GeminiChatError as exc:
            logger.error(
                "Gemini generation failed during answer_question (%s).", type(exc).__name__
            )
            raise RAGServiceError(
                "Failed to generate an answer. Please try again."
            ) from exc

        sources = [self._to_source_reference(chunk) for chunk in chunks]
        response = RAGResponse(
            answer=answer,
            sources=sources,
            retrieved_chunks=chunks,
            original_query=validated_query,
            retrieval_query=retrieval_query,
        )
        self._remember(memory, validated_query, answer)
        return response

    @staticmethod
    def _remember(
        memory: Optional[ConversationMemory], user_message: str, assistant_message: str
    ) -> None:
        """Record one completed turn in memory, if a memory was provided."""
        if memory is None:
            return
        memory.add_user_message(user_message)
        memory.add_assistant_message(assistant_message)

    @staticmethod
    def _to_source_reference(chunk: RetrievedChunk) -> SourceReference:
        return SourceReference(
            file_name=chunk.file_name,
            page=chunk.page,
            chunk_index=chunk.chunk_index,
            chunk_id=chunk.chunk_id,
            document_hash=chunk.document_hash,
        )

    @staticmethod
    def _validate_query(query: str) -> str:
        if query is None:
            raise RAGServiceError("query must be a string, not None.")
        if not isinstance(query, str):
            raise RAGServiceError(f"query must be a string, got {type(query).__name__}.")
        if not query.strip():
            raise RAGServiceError("query cannot be empty or whitespace-only.")
        return query
