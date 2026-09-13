from collections.abc import Sequence
from typing import override

from simplified_genai.interfaces.i_chat_session_store import IChatSessionStore
from simplified_genai.models.gen_ai_chat_message import GenAiChatMessage


class InMemoryChatSessionStore(IChatSessionStore):
    """In-memory implementation of IChatSessionStore.

    Stores conversation history in an internal dictionary.
    Ideal for testing, local scripts, or ephemeral single-process usage.
    """

    def __init__(self) -> None:
        self._store: dict[str, list[GenAiChatMessage]] = {}

    @override
    async def get_history(self, session_id: str) -> Sequence[GenAiChatMessage]:
        return list(self._store.get(session_id, []))

    @override
    async def append_messages(
        self, session_id: str, messages: Sequence[GenAiChatMessage]
    ) -> None:
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].extend(messages)

    @override
    async def delete_session(self, session_id: str) -> None:
        self._store.pop(session_id, None)
