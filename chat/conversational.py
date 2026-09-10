"""Deterministic detection of simple conversational messages.

    User Message -> detect_conversational_reply() -> canned reply, or None

Greetings, thanks, farewells, "how are you", and short acknowledgments
("ok", "yes", "no", ...) carry no document-related intent, so routing them
through query rewriting, semantic retrieval, and
Gemini generation is both wasteful and produces a confusing
"I don't have that information in the document." reply. This module
recognizes exactly that narrow, well-defined set of messages — nothing
else — and returns a short canned reply for them so the caller
(chat/service.py) can skip the RAG pipeline entirely for this turn.

This is a plain, deterministic lookup — no LLM call, no fuzzy matching. A
message is only ever treated as conversational if, once whitespace/case/
trailing-punctuation differences are normalized away, it matches one of the
known phrases *exactly*. A message that also asks something substantive
(e.g. "Hi, what is the maximum file size?") will never match, since it
won't be exactly equal to just "hi" after normalization — it always falls
through to the normal document-grounded RAG flow.
"""

import re
from typing import Dict, Optional

# Normalized phrase -> canned reply. Keep this list narrow and explicit
# (exactly the categories this feature is meant to cover) rather than trying
# to guess every possible greeting variant.
_CONVERSATIONAL_RESPONSES: Dict[str, str] = {
    "hi": "Hi! How can I help you with your documents?",
    "hello": "Hello! Ask me anything about your uploaded documents.",
    "hey": "Hey! How can I help you with your documents?",
    "good morning": "Good morning! How can I help you today?",
    "good afternoon": "Good afternoon! How can I help you today?",
    "good evening": "Good evening! How can I help you today?",
    "thanks": "You're welcome! Feel free to ask me anything about your documents.",
    "thank you": "You're welcome! Feel free to ask me anything about your documents.",
    "bye": "Goodbye! Have a great day.",
    "goodbye": "Goodbye! Have a great day.",
    "how are you": "I'm doing well! How can I help you with your documents?",
    "ok": "Okay! Let me know if you have any questions about your documents.",
    "okay": "Okay! Let me know if you have any questions about your documents.",
    "yes": "Great! How can I help you with your documents?",
    "no": "Okay! Let me know if you need anything else.",
}

_TRAILING_PUNCTUATION = re.compile(r"[!?.,]+$")


def _normalize(message: str) -> str:
    """Collapse whitespace, lowercase, and strip trailing punctuation."""
    collapsed = " ".join(message.split())
    lowered = collapsed.lower()
    return _TRAILING_PUNCTUATION.sub("", lowered).strip()


def detect_conversational_reply(message: str) -> Optional[str]:
    """Return a canned reply if `message` is simple conversational small talk.

    Returns None for anything else, including a message that combines a
    greeting/thanks with an actual question — that must still go through
    the normal RAG pipeline. Never raises: non-string or empty input simply
    yields None so the caller's own validation handles it.
    """
    if not isinstance(message, str):
        return None
    return _CONVERSATIONAL_RESPONSES.get(_normalize(message))
