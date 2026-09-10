"""Context-aware query rewriting for conversational follow-up questions.

    User Question + Conversation History -> Gemini -> Standalone Retrieval Query

This module only produces a search query for DocumentRetriever — it never
answers the question, never touches Chroma directly, and never creates a
second Gemini client (it reuses the injected/shared GeminiChatService).
"""

from typing import List, Optional

from chat.memory import ConversationMessage
from llm.gemini import GeminiChatError, GeminiChatService
from llm.prompts import build_query_rewrite_prompt
from utils.logging_config import get_logger

logger = get_logger(__name__)


class QueryRewriteError(Exception):
    """Raised when query rewriting receives invalid input or fails."""


class QueryRewriter:
    """Rewrites a conversational follow-up question into a standalone query.

    If there is no conversation history, rewriting is skipped entirely and
    the (validated) original question is returned unchanged — there is
    nothing to resolve it against, so no Gemini call is made. This is the
    only deterministic short-circuit; whether a question with history
    actually needs rewriting is left to the prompt (see
    llm/prompts.py:QUERY_REWRITE_SYSTEM_INSTRUCTIONS rule 3), not to a
    hand-written keyword/pronoun heuristic.
    """

    def __init__(self, chat_service: Optional[GeminiChatService] = None) -> None:
        self._chat_service = chat_service or GeminiChatService()

    def rewrite(self, query: str, history: List[ConversationMessage]) -> str:
        """Return a standalone search query for `query`, given `history`.

        The rewrite is used ONLY for retrieval — it must preserve the
        user's intent and resolve references using the conversation
        history, without adding facts or answering the question. The
        original `query` passed in here is unaffected and should still be
        used for final answer generation by the caller.
        """
        validated_query = self._validate_query(query)

        if not history:
            return validated_query

        system_prompt, user_prompt = build_query_rewrite_prompt(validated_query, history)

        try:
            rewritten = self._chat_service.generate(system_prompt, user_prompt)
        except GeminiChatError as exc:
            logger.error("Query rewriting failed (%s).", type(exc).__name__)
            raise QueryRewriteError(
                "Failed to process the follow-up question. Please try again."
            ) from exc

        if not isinstance(rewritten, str) or not rewritten.strip():
            logger.error("Query rewriting returned an empty or malformed result.")
            raise QueryRewriteError("Query rewriting returned an unusable result.")

        return rewritten.strip()

    @staticmethod
    def _validate_query(query: str) -> str:
        if query is None:
            raise QueryRewriteError("query must be a string, not None.")
        if not isinstance(query, str):
            raise QueryRewriteError(f"query must be a string, got {type(query).__name__}.")
        if not query.strip():
            raise QueryRewriteError("query cannot be empty or whitespace-only.")
        return query
