from enum import Enum

from pydantic import BaseModel


class GenAiChatMessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class GenAiChatMessage(BaseModel):
    role: GenAiChatMessageRole
    content: str
