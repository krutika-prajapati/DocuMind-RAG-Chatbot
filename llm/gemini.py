"""Gemini chat/completion service built on langchain-google-genai.

This module only sends a system + user prompt to Gemini and returns the
text response. It has no knowledge of retrieval, RAG prompt construction,
or source citations — those live in llm/prompts.py and chat/service.py.
"""

from typing import List, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config.settings import GEMINI_MODEL, LLM_TEMPERATURE
from utils.logging_config import get_logger

logger = get_logger(__name__)


class GeminiChatError(Exception):
    """Raised when Gemini chat configuration or a generation request fails."""


class GeminiChatService:
    """Generates text responses from Gemini given a system + user prompt.

    The underlying Gemini client is created lazily on first use, so
    constructing this service (and importing this module) never requires
    GOOGLE_API_KEY to be set. The key is only needed once a response is
    actually requested.
    """

    def __init__(
        self,
        model: str = GEMINI_MODEL,
        temperature: float = LLM_TEMPERATURE,
    ) -> None:
        if not model:
            raise GeminiChatError(
                "No Gemini chat model configured. Set GEMINI_MODEL in your .env file."
            )
        self._model = model
        self._temperature = temperature
        self._client: Optional[ChatGoogleGenerativeAI] = None

    @property
    def _chat_model(self) -> ChatGoogleGenerativeAI:
        """Return the underlying LangChain chat client, creating it on first use."""
        if self._client is None:
            try:
                self._client = ChatGoogleGenerativeAI(
                    model=self._model, temperature=self._temperature
                )
            except Exception as exc:
                logger.error(
                    "Failed to initialize Gemini chat client (%s).", type(exc).__name__
                )
                raise GeminiChatError(
                    "Unable to initialize Gemini. Check that GOOGLE_API_KEY is set "
                    "correctly in your .env file."
                ) from exc
        return self._client

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Send a system + user prompt to Gemini and return the text response.

        Raises GeminiChatError for invalid input, a failed request, or an
        empty/malformed response — never returns a fabricated answer.
        """
        validated_system_prompt = self._validate_prompt(system_prompt, "system_prompt")
        validated_user_prompt = self._validate_prompt(user_prompt, "user_prompt")

        messages: List[BaseMessage] = [
            SystemMessage(content=validated_system_prompt),
            HumanMessage(content=validated_user_prompt),
        ]

        try:
            response = self._chat_model.invoke(messages)
        except GeminiChatError:
            raise
        except Exception as exc:
            logger.error("Gemini chat request failed (%s).", type(exc).__name__)
            raise GeminiChatError(
                "Failed to generate a response. Please try again."
            ) from exc

        # Use the normalized `.text` accessor rather than raw `.content`:
        # some models (e.g. gemini-3.5-flash) return `.content` as a list of
        # content-block dicts instead of a plain string, and `.text` is
        # LangChain's own property for correctly handling both shapes.
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            logger.error("Gemini returned an empty or malformed response.")
            raise GeminiChatError("Gemini returned an empty or unusable response.")

        return text

    @staticmethod
    def _validate_prompt(prompt: str, name: str) -> str:
        if prompt is None:
            raise GeminiChatError(f"{name} must be a string, not None.")
        if not isinstance(prompt, str):
            raise GeminiChatError(f"{name} must be a string, got {type(prompt).__name__}.")
        if not prompt.strip():
            raise GeminiChatError(f"{name} cannot be empty or whitespace-only.")
        return prompt


if __name__ == "__main__":
    # Developer-only smoke test against the real Gemini API.
    # Requires a valid GOOGLE_API_KEY. Never run automatically by the app or tests.
    # Usage: python -m llm.gemini
    smoke_service = GeminiChatService()
    reply = smoke_service.generate(
        system_prompt="You are a helpful assistant. Keep answers to one sentence.",
        user_prompt="Say hello and confirm you are working.",
    )
    print(f"Gemini response: {reply}")
