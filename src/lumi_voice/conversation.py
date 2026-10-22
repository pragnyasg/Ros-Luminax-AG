"""Conversation history management for LUMI."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Message:
    """A single conversation message."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ConversationManager:
    """Manages conversation history with configurable limits."""

    def __init__(
        self,
        system_prompt: str,
        max_history: int = 20,
        save_history: bool = False,
        history_path: Optional[str] = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.save_history = save_history
        self.history_path = Path(history_path) if history_path else None
        self._messages: list[Message] = []

    @property
    def messages(self) -> list[Message]:
        return list(self._messages)

    @property
    def message_count(self) -> int:
        return len(self._messages)

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self._messages.append(Message(role="user", content=content))
        self._trim_history()
        logger.debug("Added user message (%d total)", len(self._messages))

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation."""
        self._messages.append(Message(role="assistant", content=content))
        self._trim_history()
        logger.debug("Added assistant message (%d total)", len(self._messages))

    def get_messages_for_api(self) -> list[dict]:
        """Get conversation messages formatted for the LLM API.

        Returns system prompt + conversation history.
        """
        messages = [{"role": "system", "content": self.system_prompt}]
        for msg in self._messages:
            messages.append(msg.to_dict())
        return messages

    def clear(self) -> None:
        """Clear all conversation history."""
        self._messages.clear()
        logger.info("Conversation history cleared")

    def reset_with_new_prompt(self, system_prompt: str) -> None:
        """Clear history and set a new system prompt."""
        self.system_prompt = system_prompt
        self.clear()

    def _trim_history(self) -> None:
        """Trim conversation to max_history messages."""
        if len(self._messages) > self.max_history:
            excess = len(self._messages) - self.max_history
            self._messages = self._messages[excess:]
            logger.debug("Trimmed %d old messages", excess)

    def save_to_file(self) -> Optional[Path]:
        """Save conversation to a JSON file if saving is enabled."""
        if not self.save_history or not self.history_path:
            return None

        try:
            self.history_path.mkdir(parents=True, exist_ok=True)
            filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.history_path / filename

            data = {
                "system_prompt": self.system_prompt,
                "messages": [msg.to_dict() | {"timestamp": msg.timestamp} for msg in self._messages],
                "saved_at": datetime.now().isoformat(),
            }

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info("Conversation saved to %s", filepath)
            return filepath
        except Exception as e:
            logger.error("Failed to save conversation: %s", e)
            return None
