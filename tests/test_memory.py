"""Tests for session-scoped conversation memory (chat/memory.py)."""

import pytest

from chat.memory import ConversationMemory, ConversationMessage, MemoryError


def test_add_user_message_is_stored():
    memory = ConversationMemory()
    memory.add_user_message("What is the cargo capacity?")

    messages = memory.get_messages()
    assert len(messages) == 1
    assert messages[0] == ConversationMessage(role="user", content="What is the cargo capacity?")


def test_add_assistant_message_is_stored():
    memory = ConversationMemory()
    memory.add_assistant_message("It is 10,000 tons.")

    messages = memory.get_messages()
    assert len(messages) == 1
    assert messages[0] == ConversationMessage(role="assistant", content="It is 10,000 tons.")


def test_messages_preserve_order():
    memory = ConversationMemory()
    memory.add_user_message("first question")
    memory.add_assistant_message("first answer")
    memory.add_user_message("second question")
    memory.add_assistant_message("second answer")

    contents = [m.content for m in memory.get_messages()]
    assert contents == ["first question", "first answer", "second question", "second answer"]


def test_get_recent_messages_returns_last_n_oldest_first():
    memory = ConversationMemory()
    for i in range(5):
        memory.add_user_message(f"message {i}")

    recent = memory.get_recent_messages(limit=2)

    assert [m.content for m in recent] == ["message 3", "message 4"]


def test_get_recent_messages_with_no_limit_returns_everything():
    memory = ConversationMemory()
    memory.add_user_message("a")
    memory.add_assistant_message("b")

    assert len(memory.get_recent_messages()) == 2


def test_get_recent_messages_with_zero_limit_returns_empty_list():
    memory = ConversationMemory()
    memory.add_user_message("a")

    assert memory.get_recent_messages(limit=0) == []


def test_get_recent_messages_rejects_negative_limit():
    memory = ConversationMemory()
    with pytest.raises(MemoryError):
        memory.get_recent_messages(limit=-1)


def test_clear_removes_all_messages():
    memory = ConversationMemory()
    memory.add_user_message("a")
    memory.add_assistant_message("b")

    memory.clear()

    assert memory.get_messages() == []
    assert len(memory) == 0


def test_empty_memory_has_zero_length_and_no_messages():
    memory = ConversationMemory()

    assert len(memory) == 0
    assert memory.get_messages() == []
    assert memory.get_recent_messages() == []


def test_memory_enforces_max_messages_limit():
    memory = ConversationMemory(max_messages=4)
    for i in range(6):
        memory.add_user_message(f"message {i}")

    messages = memory.get_messages()
    assert len(messages) == 4
    # Oldest messages were dropped; most recent ones remain, in order.
    assert [m.content for m in messages] == ["message 2", "message 3", "message 4", "message 5"]


def test_memory_limit_never_exceeded_across_mixed_roles():
    memory = ConversationMemory(max_messages=3)
    memory.add_user_message("q1")
    memory.add_assistant_message("a1")
    memory.add_user_message("q2")
    memory.add_assistant_message("a2")

    assert len(memory) == 3
    assert [m.content for m in memory.get_messages()] == ["a1", "q2", "a2"]


def test_constructor_rejects_invalid_max_messages():
    with pytest.raises(MemoryError):
        ConversationMemory(max_messages=0)
    with pytest.raises(MemoryError):
        ConversationMemory(max_messages=-5)
    with pytest.raises(MemoryError):
        ConversationMemory(max_messages="20")


def test_add_user_message_rejects_none():
    memory = ConversationMemory()
    with pytest.raises(MemoryError):
        memory.add_user_message(None)


def test_add_assistant_message_rejects_empty_string():
    memory = ConversationMemory()
    with pytest.raises(MemoryError):
        memory.add_assistant_message("")


def test_add_message_rejects_whitespace_only():
    memory = ConversationMemory()
    with pytest.raises(MemoryError):
        memory.add_user_message("   \n\t  ")


def test_add_message_rejects_non_string_content():
    memory = ConversationMemory()
    with pytest.raises(MemoryError):
        memory.add_user_message(12345)


def test_conversation_message_is_immutable():
    message = ConversationMessage(role="user", content="hello")
    with pytest.raises(Exception):
        message.content = "changed"
