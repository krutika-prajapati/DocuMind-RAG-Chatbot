"""Tests for the Gemini chat service (llm/gemini.py).

The real ChatGoogleGenerativeAI client is always mocked here — these tests
never require a GOOGLE_API_KEY or network access.
"""

import logging

import pytest
from langchain_core.messages import AIMessage

from llm.gemini import GeminiChatError, GeminiChatService

# Real AIMessage is used (not a hand-rolled stand-in) so `.text` behaves
# exactly like a genuine ChatGoogleGenerativeAI response — including models
# (e.g. gemini-3.5-flash) that return `.content` as a list of content-block
# dicts rather than a plain string.
FakeResponse = AIMessage


class FakeChatModel:
    """Stand-in for ChatGoogleGenerativeAI used to avoid real API calls."""

    def __init__(self, model=None, temperature=None, **kwargs):
        self.model = model
        self.temperature = temperature
        self.invoke_calls = []

    def invoke(self, messages):
        self.invoke_calls.append(messages)
        return FakeResponse(content="This is a grounded answer.")


@pytest.fixture
def patched_chat_model(monkeypatch):
    created = {}

    def factory(**kwargs):
        instance = FakeChatModel(**kwargs)
        created["instance"] = instance
        return instance

    monkeypatch.setattr("llm.gemini.ChatGoogleGenerativeAI", factory)
    return created


def test_service_initializes_without_api_key(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    service = GeminiChatService(model="gemini-2.5-flash")

    assert service is not None


def test_service_requires_a_configured_model():
    with pytest.raises(GeminiChatError):
        GeminiChatService(model="")


def test_configured_model_and_temperature_are_passed_to_client(patched_chat_model):
    service = GeminiChatService(model="gemini-2.5-flash", temperature=0.0)
    service.generate("system prompt", "user prompt")

    assert patched_chat_model["instance"].model == "gemini-2.5-flash"
    assert patched_chat_model["instance"].temperature == 0.0


def test_generate_returns_text_response(patched_chat_model):
    service = GeminiChatService(model="gemini-2.5-flash")

    result = service.generate("system prompt", "user prompt")

    assert result == "This is a grounded answer."


def test_generate_handles_list_shaped_content(monkeypatch):
    """Some models (e.g. gemini-3.5-flash) return `.content` as a list of
    content-block dicts rather than a plain string; `.text` must still be
    extracted correctly instead of being rejected as malformed."""

    class ListContentChatModel:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            return AIMessage(
                content=[{"type": "text", "text": "Answer from a block-content model.", "extras": {}}]
            )

    monkeypatch.setattr("llm.gemini.ChatGoogleGenerativeAI", ListContentChatModel)
    service = GeminiChatService(model="gemini-3.5-flash")

    result = service.generate("system prompt", "user prompt")

    assert result == "Answer from a block-content model."


def test_generate_sends_system_and_user_messages(patched_chat_model):
    from langchain_core.messages import HumanMessage, SystemMessage

    service = GeminiChatService(model="gemini-2.5-flash")
    service.generate("be helpful", "what is the answer?")

    sent_messages = patched_chat_model["instance"].invoke_calls[0]
    assert isinstance(sent_messages[0], SystemMessage)
    assert sent_messages[0].content == "be helpful"
    assert isinstance(sent_messages[1], HumanMessage)
    assert sent_messages[1].content == "what is the answer?"


def test_generate_rejects_none_system_prompt(patched_chat_model):
    service = GeminiChatService(model="gemini-2.5-flash")
    with pytest.raises(GeminiChatError):
        service.generate(None, "question")


def test_generate_rejects_invalid_type_system_prompt(patched_chat_model):
    service = GeminiChatService(model="gemini-2.5-flash")
    with pytest.raises(GeminiChatError):
        service.generate(12345, "question")


def test_generate_rejects_whitespace_only_user_prompt(patched_chat_model):
    service = GeminiChatService(model="gemini-2.5-flash")
    with pytest.raises(GeminiChatError):
        service.generate("system", "   \n\t  ")


def test_generate_rejects_empty_user_prompt(patched_chat_model):
    service = GeminiChatService(model="gemini-2.5-flash")
    with pytest.raises(GeminiChatError):
        service.generate("system", "")


def test_client_initialization_error_is_wrapped(monkeypatch):
    def raise_missing_key(**kwargs):
        raise ValueError(
            "API key required for Gemini Developer API. Provide api_key parameter "
            "or set GOOGLE_API_KEY/GEMINI_API_KEY environment variable."
        )

    monkeypatch.setattr("llm.gemini.ChatGoogleGenerativeAI", raise_missing_key)
    service = GeminiChatService(model="gemini-2.5-flash")

    with pytest.raises(GeminiChatError):
        service.generate("system", "question")


def test_empty_gemini_response_raises_clear_error(monkeypatch):
    class EmptyResponseChatModel:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            return FakeResponse("   ")

    monkeypatch.setattr("llm.gemini.ChatGoogleGenerativeAI", EmptyResponseChatModel)
    service = GeminiChatService(model="gemini-2.5-flash")

    with pytest.raises(GeminiChatError):
        service.generate("system", "question")


def test_malformed_gemini_response_raises_clear_error(monkeypatch):
    class MalformedResponseChatModel:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            return object()  # no .content attribute at all

    monkeypatch.setattr("llm.gemini.ChatGoogleGenerativeAI", MalformedResponseChatModel)
    service = GeminiChatService(model="gemini-2.5-flash")

    with pytest.raises(GeminiChatError):
        service.generate("system", "question")


def test_api_key_never_appears_in_error_message(monkeypatch):
    fake_secret = "SUPER-SECRET-API-KEY-VALUE"

    class ExplodingChatModel:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            raise RuntimeError(f"401 Unauthorized using key {fake_secret}")

    monkeypatch.setattr("llm.gemini.ChatGoogleGenerativeAI", ExplodingChatModel)
    service = GeminiChatService(model="gemini-2.5-flash")

    with pytest.raises(GeminiChatError) as exc_info:
        service.generate("system", "question")

    assert fake_secret not in str(exc_info.value)


def test_api_key_never_logged(monkeypatch, caplog):
    fake_secret = "SUPER-SECRET-API-KEY-VALUE"

    class ExplodingChatModel:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            raise RuntimeError(f"401 Unauthorized using key {fake_secret}")

    monkeypatch.setattr("llm.gemini.ChatGoogleGenerativeAI", ExplodingChatModel)
    service = GeminiChatService(model="gemini-2.5-flash")

    with caplog.at_level(logging.ERROR):
        with pytest.raises(GeminiChatError):
            service.generate("system", "question")

    assert fake_secret not in caplog.text
