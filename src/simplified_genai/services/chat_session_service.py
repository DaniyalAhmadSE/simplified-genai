from collections.abc import Sequence
from typing import override
import uuid

from simplified_genai.interfaces.i_chat_session_service import (
    IChatSessionService,
)
from simplified_genai.interfaces.i_chat_session_store import IChatSessionStore
from simplified_genai.models.gen_ai_chat_message import GenAiChatMessage


class ChatSessionService(IChatSessionService):
    """Service managing per-session conversation history via an IChatSessionStore."""

    def __init__(self, store: IChatSessionStore) -> None:
        self.__store = store

    @override
    def start_session(self) -> str:
        """Create a new unique session ID."""
        return uuid.uuid4().hex

    @override
    async def get_history(self, session_id: str) -> Sequence[GenAiChatMessage]:
        """Fetch the history of messages for the specified session."""
        if not session_id:
            return []
        return await self.__store.get_history(session_id)

    @override
    async def append_turn(
        self,
        session_id: str,
        user_message: GenAiChatMessage,
        assistant_message: GenAiChatMessage,
    ) -> None:
        """Append a user-assistant turn to the chat session history."""
        if not session_id:
            raise ValueError("session_id must not be empty")
        await self.__store.append_messages(
            session_id, [user_message, assistant_message]
        )

    @override
    async def clear_session(self, session_id: str) -> None:
        """Clear all messages from the session."""
        if not session_id:
            return
        await self.__store.delete_session(session_id)
