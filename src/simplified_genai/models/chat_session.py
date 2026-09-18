from datetime import datetime, timezone

from pydantic import BaseModel, Field

from simplified_genai.models.gen_ai_chat_message import GenAiChatMessage


class ChatSession(BaseModel):
    session_id: str
    messages: list[GenAiChatMessage] = Field(default_factory=list[GenAiChatMessage])
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
