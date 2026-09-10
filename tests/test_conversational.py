"""Tests for deterministic conversational-message detection (chat/conversational.py)."""

import pytest

from chat.conversational import detect_conversational_reply


@pytest.mark.parametrize(
    "message",
    [
        "Hi",
        "Hello",
        "Hey",
        "Good morning",
        "Good afternoon",
        "Good evening",
        "Thanks",
        "Thank you",
        "Bye",
        "Goodbye",
        "How are you?",
        "ok",
        "okay",
        "yes",
        "no",
    ],
)
def test_known_conversational_messages_return_a_reply(message):
    assert detect_conversational_reply(message) is not None


def test_greeting_reply_matches_expected_wording():
    assert detect_conversational_reply("Hi") == "Hi! How can I help you with your documents?"
    assert (
        detect_conversational_reply("Hello")
        == "Hello! Ask me anything about your uploaded documents."
    )
    assert detect_conversational_reply("Hey") == "Hey! How can I help you with your documents?"


def test_good_morning_reply_matches_expected_wording():
    assert (
        detect_conversational_reply("Good morning")
        == "Good morning! How can I help you today?"
    )


def test_thanks_reply_matches_expected_wording():
    expected = "You're welcome! Feel free to ask me anything about your documents."
    assert detect_conversational_reply("Thanks") == expected
    assert detect_conversational_reply("Thank you") == expected


def test_farewell_reply_matches_expected_wording():
    expected = "Goodbye! Have a great day."
    assert detect_conversational_reply("Bye") == expected
    assert detect_conversational_reply("Goodbye") == expected


def test_how_are_you_reply_matches_expected_wording():
    assert (
        detect_conversational_reply("How are you?")
        == "I'm doing well! How can I help you with your documents?"
    )


def test_acknowledgment_reply_matches_expected_wording():
    ok_expected = "Okay! Let me know if you have any questions about your documents."
    assert detect_conversational_reply("ok") == ok_expected
    assert detect_conversational_reply("okay") == ok_expected
    assert detect_conversational_reply("yes") == "Great! How can I help you with your documents?"
    assert detect_conversational_reply("no") == "Okay! Let me know if you need anything else."


@pytest.mark.parametrize(
    "variant",
    [
        "HI", "hi", "Hi", "HELLO", "hello", "THANK YOU", "Thank You", "GOODBYE",
        "OK", "Ok", "ok", "OKAY", "Okay", "okay", "YES", "Yes", "yes", "NO", "No", "no",
    ],
)
def test_detection_is_case_insensitive(variant):
    assert detect_conversational_reply(variant) is not None


@pytest.mark.parametrize(
    "message",
    ["  Hi  ", "Hi\n", "\tHello", "  Thanks   ", "   Bye  ", "  ok  ", " okay\n", "\tyes", "  no  "],
)
def test_detection_ignores_leading_and_trailing_whitespace(message):
    assert detect_conversational_reply(message) is not None


@pytest.mark.parametrize(
    "message",
    [
        "Hi!", "Hello!!", "Hey.", "Thanks!", "Bye.", "Goodbye!", "How are you???",
        "ok!", "okay.", "yes!", "no.",
    ],
)
def test_detection_ignores_trailing_punctuation(message):
    assert detect_conversational_reply(message) is not None


@pytest.mark.parametrize(
    "message",
    [
        "Hi, what is the maximum file size for Project Orion?",
        "Thanks, what database does Orion use?",
        "Hello, who can approve a document?",
        "Hey, can you summarize the safety requirements?",
        "Good morning, what is the cargo capacity of the vessel?",
        "How are you supposed to configure the retrieval settings?",
        "Yes, what database does Project Orion use?",
        "No, what is the maximum file size?",
        "Okay, tell me about Project Aurora.",
        "Yes, who can approve a document?",
    ],
)
def test_document_questions_containing_conversational_words_are_not_classified(message):
    assert detect_conversational_reply(message) is None


@pytest.mark.parametrize(
    "message",
    [
        "What is the cargo capacity of the vessel?",
        "Summarize the safety requirements.",
        "hi there",
        "good night",
        "What's up with the retrieval settings?",
        "ok cool",
        "yes please",
        "no thanks",
        "",
        "   ",
    ],
)
def test_unknown_or_general_messages_return_none(message):
    assert detect_conversational_reply(message) is None


def test_none_input_returns_none_without_raising():
    assert detect_conversational_reply(None) is None


def test_non_string_input_returns_none_without_raising():
    assert detect_conversational_reply(12345) is None
