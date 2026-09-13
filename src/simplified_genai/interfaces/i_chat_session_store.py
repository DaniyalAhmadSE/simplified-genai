from collections.abc import Sequence
from typing import Protocol

from simplified_genai.models.gen_ai_chat_message import GenAiChatMessage


class IChatSessionStore(Protocol):
    """Protocol for storing and retrieving chat session conversation history."""

    async def get_history(self, session_id: str) -> Sequence[GenAiChatMessage]:
        """Get the full message history for a given session ID."""
        ...

    async def append_messages(
        self, session_id: str, messages: Sequence[GenAiChatMessage]
    ) -> None:
        """Append new messages to the existing session history."""
        ...

    async def delete_session(self, session_id: str) -> None:
        """Delete all messages and data for a given session ID."""
        ...
