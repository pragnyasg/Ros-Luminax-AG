"""Tests for conversation history management."""

import json

import pytest

from lumi_voice.conversation import ConversationManager, Message


class TestMessage:
    def test_to_dict(self):
        msg = Message(role="user", content="Hello")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "Hello"
        assert "timestamp" not in d

    def test_timestamp_auto(self):
        msg = Message(role="assistant", content="Hi")
        assert msg.timestamp  # auto-generated


class TestConversationManager:
    def test_add_messages(self):
        mgr = ConversationManager(system_prompt="You are a bot.", max_history=10)
        mgr.add_user_message("Hello")
        mgr.add_assistant_message("Hi there!")
        assert mgr.message_count == 2

    def test_get_messages_for_api(self):
        mgr = ConversationManager(system_prompt="System prompt here.")
        mgr.add_user_message("Test")
        messages = mgr.get_messages_for_api()
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System prompt here."
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "Test"

    def test_history_trimming(self):
        mgr = ConversationManager(system_prompt="Bot", max_history=4)
        for i in range(10):
            mgr.add_user_message(f"msg {i}")
        assert mgr.message_count == 4
        # Should keep the latest messages
        assert mgr.messages[-1].content == "msg 9"

    def test_clear(self):
        mgr = ConversationManager(system_prompt="Bot")
        mgr.add_user_message("Hello")
        mgr.add_assistant_message("Hi")
        mgr.clear()
        assert mgr.message_count == 0

    def test_reset_with_new_prompt(self):
        mgr = ConversationManager(system_prompt="Old")
        mgr.add_user_message("data")
        mgr.reset_with_new_prompt("New")
        assert mgr.system_prompt == "New"
        assert mgr.message_count == 0

    def test_save_to_file(self, tmp_path):
        mgr = ConversationManager(
            system_prompt="Bot",
            save_history=True,
            history_path=str(tmp_path / "history"),
        )
        mgr.add_user_message("Hello")
        mgr.add_assistant_message("Hi")
        filepath = mgr.save_to_file()
        assert filepath is not None
        assert filepath.exists()

        with open(filepath) as f:
            data = json.load(f)
        assert data["system_prompt"] == "Bot"
        assert len(data["messages"]) == 2

    def test_save_disabled(self):
        mgr = ConversationManager(system_prompt="Bot", save_history=False)
        assert mgr.save_to_file() is None
