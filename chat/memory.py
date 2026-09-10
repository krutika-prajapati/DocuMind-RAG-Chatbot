"""In-memory, session-scoped conversation history.

ConversationMemory holds the back-and-forth of one chat session so query
rewriting (retrieval/query_rewriter.py) and, later, a Streamlit UI can see
recent turns. It is NOT a knowledge source: retrieved document context
remains the only factual basis for answers (see chat/service.py).

This is intentionally a simple in-process list scoped to one conversation
— construct one instance per user session (e.g. stored in
st.session_state), never a single shared/global instance across sessions.
The API is deliberately small so it can later be swapped for a
Redis/database-backed implementation without changing callers.
"""

from dataclasses import dataclass
from typing import List, Optional

from config.settings import MAX_MEMORY_MESSAGES

USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"


class MemoryError(Exception):
    """Raised when a conversation memory operation receives invalid input."""


@dataclass(frozen=True)
class ConversationMessage:
    """One turn in a conversation: either a user or assistant message."""

    role: str
    content: str


class ConversationMemory:
    """Holds one conversation's message history for a single session.

    Only the minimal, non-sensitive data needed to resolve conversational
    references is kept — user and assistant message text. No internal
    prompts, retrieved chunks, or other implementation details are stored.
    """

    def __init__(self, max_messages: int = MAX_MEMORY_MESSAGES) -> None:
        if isinstance(max_messages, bool) or not isinstance(max_messages, int) or max_messages <= 0:
            raise MemoryError(f"max_messages must be a positive integer, got {max_messages!r}.")
        self._max_messages = max_messages
        self._messages: List[ConversationMessage] = []

    def add_user_message(self, content: str) -> None:
        """Append a user message to the history."""
        self._messages.append(
            ConversationMessage(role=USER_ROLE, content=self._validate_content(content))
        )
        self._trim()

    def add_assistant_message(self, content: str) -> None:
        """Append an assistant message to the history."""
        self._messages.append(
            ConversationMessage(role=ASSISTANT_ROLE, content=self._validate_content(content))
        )
        self._trim()

    def get_messages(self) -> List[ConversationMessage]:
        """Return all stored messages, oldest first."""
        return list(self._messages)

    def get_recent_messages(self, limit: Optional[int] = None) -> List[ConversationMessage]:
        """Return the most recent messages, oldest first.

        With no limit, returns everything currently stored (already capped
        at max_messages by add_user_message/add_assistant_message).
        """
        if limit is None:
            return list(self._messages)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise MemoryError(f"limit must be a non-negative integer, got {limit!r}.")
        if limit == 0:
            return []
        return list(self._messages[-limit:])

    def clear(self) -> None:
        """Remove all stored messages."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)

    @staticmethod
    def _validate_content(content: str) -> str:
        if content is None:
            raise MemoryError("content must be a string, not None.")
        if not isinstance(content, str):
            raise MemoryError(f"content must be a string, got {type(content).__name__}.")
        if not content.strip():
            raise MemoryError("content cannot be empty or whitespace-only.")
        return content

    def _trim(self) -> None:
        """Keep only the most recent max_messages messages.

        Trims from the oldest end, one whole message at a time — a message
        is never partially truncated.
        """
        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages :]
