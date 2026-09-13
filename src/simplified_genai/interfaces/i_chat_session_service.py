from collections.abc import Sequence
from typing import Protocol

from simplified_genai.models.gen_ai_chat_message import GenAiChatMessage


class IChatSessionService(Protocol):
    def start_session(self) -> str: ...

    async def get_history(self, session_id: str) -> Sequence[GenAiChatMessage]: ...

    async def append_turn(
        self,
        session_id: str,
        user_message: GenAiChatMessage,
        assistant_message: GenAiChatMessage,
    ) -> None: ...

    async def clear_session(self, session_id: str) -> None: ...
