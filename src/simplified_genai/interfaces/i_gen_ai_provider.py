from typing import Any, Iterable, Protocol

from pydantic import BaseModel

from simplified_genai.models.file_upload_response import (
    FileUploadResponse,
)
from simplified_genai.models.file_payload import FilePayload
from simplified_genai.models.gen_ai_chat_message import GenAiChatMessage
from simplified_genai.models.tool_choice import ToolChoice
from simplified_genai.models.tool_definition import ToolDefinition


class IGenAiProvider(Protocol):
    async def get_raw_text_response(
        self,
        *,
        user_prompt: str | None,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        files: list[FilePayload] = [],
        existing_file_upload_results: list[FileUploadResponse] = [],
        auto_delete_files: bool = True,
        thinking_config: Any | None = None,
        tools: Iterable[ToolDefinition] = [],
        max_tool_iterations: int | None = None,
        messages: Iterable[GenAiChatMessage] = [],
        tool_choice: ToolChoice = ToolChoice.AUTO,
    ) -> str: ...

    async def get_structured_response[T: BaseModel](
        self,
        *,
        schema: type[T],
        user_prompt: str | None,
        model: str | None = None,
        system_prompt: str | None = None,
        temperature: float | None = None,
        files: list[FilePayload] = [],
        existing_file_upload_results: list[FileUploadResponse] = [],
        auto_delete_files: bool = True,
        thinking_config: Any | None = None,
        tools: Iterable[ToolDefinition] = [],
        max_tool_iterations: int | None = None,
        tool_choice: ToolChoice = ToolChoice.AUTO,
    ) -> T: ...

    async def upload_file(self, file: FilePayload) -> FileUploadResponse: ...

    async def delete_uploaded_files(
        self, file_upload_results: Iterable[FileUploadResponse]
    ) -> None: ...
