import base64
import json
import mimetypes
from typing import Any, Optional, Iterable, override
from uuid import uuid4

import grpc.aio
from loguru import logger
from pydantic import BaseModel
from xai_sdk import AsyncClient  # type: ignore
from xai_sdk.chat import file as xai_file  # type: ignore
from xai_sdk.chat import assistant, image, system, user  # type: ignore
from xai_sdk.chat import tool_result as xai_tool_result  # type: ignore
from xai_sdk.types import Content  # type: ignore
from xai_sdk.aio.chat import Chat  # type: ignore
from simplified_genai.exceptions.gen_ai_overloaded_error import GenAiOverloadedError
from simplified_genai.models.base_file_upload_result_dto import (
    BaseFileUploadResultDto,
)
from simplified_genai.models.gen_ai_chat_message import (
    GenAiChatMessage,
    GenAiChatMessageRole,
)
from simplified_genai.models.grok_upload_result_dto import GrokUploadResultDto
from simplified_genai.models.file_dto import FileDto
from simplified_genai.interfaces.i_gen_ai_provider import IGenAiProvider
from simplified_genai.models.tool_choice import ToolChoice
from simplified_genai.models.tool_definition import ToolDefinition

# MIME types that the xAI vision models accept as inline images.
# All other types are uploaded via the Files API and referenced by ID.
_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
}


class GrokGenAiProvider(IGenAiProvider):
    def __init__(
        self,
        *,
        api_key: str,
        default_model: str,
    ) -> None:
        self.__llm_client = AsyncClient(api_key=api_key)
        self.__default_model = default_model
        self.__default_temperature = 0.2
        self.__default_max_tool_iterations = 5

    @override
    async def get_raw_text_response(
        self,
        *,
        user_prompt: str | None,
        model: str | None = None,
        system_prompt: Optional[str] = None,
        temperature: float | None = None,
        files: list[FileDto] = [],
        existing_file_upload_results: list[BaseFileUploadResultDto] = [],
        auto_delete_files: bool = True,
        thinking_config: Any | None = None,
        tools: Iterable[ToolDefinition] = [],
        max_tool_iterations: int | None = None,
        messages: Iterable[GenAiChatMessage] = [],
        tool_choice: ToolChoice = ToolChoice.AUTO,
    ) -> str:
        if not user_prompt and not files and not existing_file_upload_results:
            logger.error(
                "user_prompt, files, and file_upload_results cannot be null at the same time."
            )
            raise ValueError(
                "user_prompt, files, and file_upload_results cannot be null at the same time."
            )

        file_upload_results: list[GrokUploadResultDto] = []
        try:
            file_upload_results = await self.__get_uploaded_file_results(
                files, existing_file_upload_results
            )

            if not tools:
                # Fast path: no tools, use chat.parse
                chat = self.__build_chat(
                    system_prompt, temperature, model, messages=messages
                )
                self.__append_user_message(
                    chat, user_prompt, files, file_upload_results
                )

                response = await chat.sample()

                if not response.content:
                    logger.error(f"No content in response: {response}")
                    raise Exception(f"No content in response: {response}")

                return response.content

            # --- Tool loop ---
            effective_max_iterations = (
                max_tool_iterations
                if max_tool_iterations is not None
                else self.__default_max_tool_iterations
            )
            tool_dispatch: dict[str, ToolDefinition] = (
                {tool.name: tool for tool in tools} if tools else {}
            )

            chat = self.__build_chat(
                system_prompt,
                temperature,
                model,
                tools,
                messages=messages,
                tool_choice=tool_choice,
            )
            self.__append_user_message(chat, user_prompt, files, file_upload_results)

            response = None
            for _ in range(effective_max_iterations):
                response = await chat.sample()

                tool_calls = self.__extract_tool_calls_from_response(response)
                if not tool_calls:
                    if not response.content:
                        logger.error(f"No content in response: {response}")
                        raise Exception(f"No content in response: {response}")
                    return response.content

                # Append assistant response so the model sees its own tool calls
                chat.append(response)

                for tool_call in tool_calls:
                    tool_definition = tool_dispatch[tool_call.function.name]
                    arguments = json.loads(tool_call.function.parameters)
                    function_result = await tool_definition.execute(**arguments)
                    chat.append(
                        xai_tool_result(
                            result=json.dumps(function_result, default=str),
                            tool_call_id=tool_call.id,
                        )
                    )

            if not response or not response.content:
                logger.error(f"No content in response: {response}")
                raise Exception(f"No content in response: {response}")

            return response.content

        except grpc.aio.AioRpcError as e:
            logger.error(f"xAI gRPC error: {e.code()} - {e.details()}")
            raise
        finally:
            if auto_delete_files:
                await self.delete_uploaded_files(file_upload_results)

    @override
    async def get_structured_response[T: BaseModel](
        self,
        *,
        schema: type[T],
        user_prompt: str | None,
        model: str | None = None,
        system_prompt: Optional[str] = None,
        temperature: float | None = None,
        files: list[FileDto] = [],
        existing_file_upload_results: list[BaseFileUploadResultDto] = [],
        auto_delete_files: bool = True,
        thinking_config: Any | None = None,
        tools: Iterable[ToolDefinition] = [],
        max_tool_iterations: int | None = None,
        tool_choice: ToolChoice = ToolChoice.AUTO,
    ) -> T:
        if not user_prompt and not files and not existing_file_upload_results:
            logger.error(
                "user_prompt, files, and file_upload_results cannot be null at the same time."
            )
            raise ValueError(
                "user_prompt, files, and file_upload_results cannot be null at the same time."
            )

        file_upload_results: list[GrokUploadResultDto] = []

        try:
            file_upload_results = await self.__get_uploaded_file_results(
                files, existing_file_upload_results
            )

            if not tools:
                # Fast path: no tools, use chat.parse
                chat = self.__build_chat(system_prompt, temperature, model)
                self.__append_user_message(
                    chat, user_prompt, files, file_upload_results
                )

                _, parsed = await chat.parse(schema)

                if not parsed:
                    logger.error("Got no parsed object in the LLM response")
                    raise Exception("Got no parsed object in the LLM response")

                return parsed

            # --- Tool loop (step 1) + re-format (step 2) ---
            # xAI's chat.parse(schema) does NOT disable tools — it sends both tools
            # and response_format to the API, so the model could still call tools
            # again.  To avoid this we use a two-step approach:
            #   Step 1 — run the tool loop via chat.sample(), get natural-language text
            #   Step 2 — create a fresh chat with NO tools, pass the text to
            #            chat.parse(schema) for reliable structured output.

            effective_max_iterations = (
                max_tool_iterations
                if max_tool_iterations is not None
                else self.__default_max_tool_iterations
            )
            tool_dispatch = {tool.name: tool for tool in tools}

            chat = self.__build_chat(
                system_prompt, temperature, model, tools, tool_choice=tool_choice
            )
            self.__append_user_message(chat, user_prompt, files, file_upload_results)

            response = None
            for _ in range(effective_max_iterations):
                response = await chat.sample()

                tool_calls = self.__extract_tool_calls_from_response(response)
                if not tool_calls:
                    break  # Natural language response — exit loop to re-format

                chat.append(response)

                for tool_call in tool_calls:
                    tool_definition = tool_dispatch[tool_call.function.name]
                    arguments = json.loads(tool_call.function.parameters)
                    function_result = await tool_definition.execute(**arguments)
                    chat.append(
                        xai_tool_result(
                            result=json.dumps(function_result, default=str),
                            tool_call_id=tool_call.id,
                        )
                    )

            if not response or not response.content:
                logger.error(f"No content in response: {response}")
                raise Exception(f"No content in response: {response}")

            # --- Step 2: re-format as structured output ---
            format_chat = self.__build_chat(
                system_prompt=None,
                temperature=temperature,
                model=model,
                tools=[],
            )
            format_chat.append(user(response.content))
            _, parsed = await format_chat.parse(schema)

            if not parsed:
                logger.error("Got no parsed object in the LLM response")
                raise Exception("Got no parsed object in the LLM response")

            return parsed

        except grpc.aio.AioRpcError as e:
            logger.error(f"xAI gRPC error: {e.code()} - {e.details()}")
            raise

        finally:
            if auto_delete_files:
                await self.delete_uploaded_files(file_upload_results)

    # ------------------------------------------------------------------
    # Tool / function-calling helpers
    # ------------------------------------------------------------------

    def __convert_to_xai_tools(self, tools: Iterable[ToolDefinition]) -> list[Any]:
        """Convert ToolDefinitions to xAI tool descriptors."""
        from xai_sdk.chat import chat_pb2  # type: ignore

        return [
            chat_pb2.Function(
                name=tool.name,
                description=tool.description,
                parameters=json.dumps(tool.generate_json_schema()),
            )
            for tool in tools
        ]

    def __extract_tool_calls_from_response(self, response: Any) -> list[Any]:
        """Extract tool calls from a Grok response."""
        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            return []
        return list(tool_calls)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def __build_chat(
        self,
        system_prompt: str | None,
        temperature: float | None,
        model: str | None,
        tools: Iterable[ToolDefinition] = [],
        messages: Iterable[GenAiChatMessage] = [],
        tool_choice: ToolChoice = ToolChoice.AUTO,
    ):
        """Instantiate a chat session, optionally seeded with a system message."""
        chat_messages = [system(system_prompt)] if system_prompt else []
        for msg in messages:
            if msg.role is GenAiChatMessageRole.ASSISTANT:
                chat_messages.append(assistant(msg.content))
            else:
                chat_messages.append(user(msg.content))
        settings: dict[str, Any] = {
            "model": model if model else self.__default_model,
            "messages": chat_messages,
            "temperature": (
                temperature if temperature is not None else self.__default_temperature
            ),
        }
        xai_tools: list[Any] = self.__convert_to_xai_tools(tools) if tools else []
        if xai_tools:
            settings["tools"] = xai_tools
            settings["tool_choice"] = tool_choice.value
        return self.__llm_client.chat.create(**settings)

    def __append_user_message(
        self,
        chat: Chat,
        user_prompt: str | None,
        files: list[FileDto],
        uploaded_file_ids: list[GrokUploadResultDto],
    ) -> None:
        """
        Build and append the user message to the chat session.

        Images are sent inline as base64 data URIs via image().
        All other file types were pre-uploaded and are referenced by their
        file ID via file().  The two lists (files / uploaded_file_ids) stay
        in sync: images are skipped during upload so they have no file ID,
        but their position is preserved via the parallel iteration below.
        """
        parts: list[str | Content] = []

        if user_prompt:
            parts.append(user_prompt)

        for f in files:
            if f.mime_type in _IMAGE_MIME_TYPES:
                b64 = base64.standard_b64encode(f.file_binary).decode("utf-8")
                data_uri = f"data:{f.mime_type};base64,{b64}"
                parts.append(image(data_uri))

        for result in uploaded_file_ids:
            parts.append(xai_file(result.uploaded_file_id))

        chat.append(user(*parts))

    @override
    async def upload_file(self, file: FileDto) -> GrokUploadResultDto:
        file_upload_result: GrokUploadResultDto | None = None
        try:
            # Derive a sensible filename so the API can detect the format.
            extension = mimetypes.guess_extension(file.mime_type)
            filename = f"{uuid4().hex}{extension or ''}"

            uploaded = await self.__llm_client.files.upload(
                file.file_binary,
                filename=filename,
            )
            file_upload_result = GrokUploadResultDto(uploaded_file_id=uploaded.id)
        except grpc.aio.AioRpcError as e:
            if (
                e.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
                or e.code() == grpc.StatusCode.UNAVAILABLE
            ):
                logger.warning(
                    f"Generative AI provider is overloaded.\nError:\n{e}\n\nDetails:\n{e.details()}"
                )
                raise GenAiOverloadedError(e.details())
            logger.error(f"Error uploading files: {str(e)}")
            raise Exception(f"Error uploading files: {str(e)}")
        except Exception as e:
            # Clean up any files that were successfully uploaded before the error.
            if file_upload_result:
                await self.delete_uploaded_files([file_upload_result])
            logger.error(f"Error uploading files: {str(e)}")
            raise Exception(f"Error uploading files: {str(e)}")

        return file_upload_result

    @override
    async def delete_uploaded_files(
        self, file_upload_results: Iterable[BaseFileUploadResultDto]
    ) -> None:
        for result in file_upload_results:
            if not isinstance(result, GrokUploadResultDto):
                raise TypeError("Invalid file upload result")
            try:
                await self.__llm_client.files.delete(result.uploaded_file_id)
            except Exception as e:
                # Log but do not re-raise — deletion failures are non-fatal.
                logger.warning(f"Failed to delete uploaded file {result}: {e}")

    async def __upload_files(self, files: list[FileDto]) -> list[GrokUploadResultDto]:
        return [
            await self.upload_file(file)
            for file in files
            if file.mime_type not in _IMAGE_MIME_TYPES
        ]

    async def __get_uploaded_file_results(
        self,
        files: list[FileDto],
        existing_file_upload_results: list[BaseFileUploadResultDto],
    ) -> list[GrokUploadResultDto]:
        file_upload_results = [
            upload_result for upload_result in (await self.__upload_files(files))
        ]

        for result in existing_file_upload_results:
            if not isinstance(result, GrokUploadResultDto):
                raise TypeError("Invalid file upload result")
            file_upload_results.append(result)

        return file_upload_results
