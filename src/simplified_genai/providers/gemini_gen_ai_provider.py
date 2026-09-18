import json
from io import BytesIO
from typing import Any, Iterable, override

from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from google.genai.types import (
    FunctionCallingConfigMode,
    GenerateContentConfig,
    ToolConfig,
)
from loguru import logger
from pydantic import BaseModel

from simplified_genai.exceptions.gen_ai_overloaded_error import (
    GenAiOverloadedError,
)
from simplified_genai.interfaces.i_gen_ai_provider import (
    IGenAiProvider,
    ToolChoice,
)
from simplified_genai.models.base_file_upload_result_dto import (
    BaseFileUploadResultDto,
)
from simplified_genai.models.file_dto import FileDto
from simplified_genai.models.gemini_upload_result_dto import (
    GeminiUploadResultDto,
)
from simplified_genai.models.gen_ai_chat_message import (
    GenAiChatMessage,
    GenAiChatMessageRole,
)
from simplified_genai.models.tool_definition import ToolDefinition


class GeminiGenAiProvider(IGenAiProvider):

    def __init__(
        self,
        api_key: str,
        default_model: str,
    ) -> None:
        self.__llm_client = genai.Client(api_key=api_key)
        self.__default_model = default_model
        self.__default_temperature = 0.2
        self.__default_max_tool_iterations = 10

    @override
    async def get_raw_text_response(
        self,
        *,
        user_prompt: str | None,
        model: str | None = None,
        system_prompt: str | None = None,
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
        model = model if model else self.__default_model

        if not user_prompt and not files and not existing_file_upload_results:
            logger.error(
                "user_prompt, files, and file_upload_results cannot be null at the same time."
            )
            raise ValueError(
                "user_prompt, files, and file_upload_results cannot be null at the same time."
            )

        file_upload_results = await self.__get_uploaded_files(
            files, existing_file_upload_results
        )
        history_contents: list[Any] = [
            types.Content(
                role=(
                    "model" if msg.role is GenAiChatMessageRole.ASSISTANT else "user"
                ),
                parts=[types.Part.from_text(text=msg.content)],
            )
            for msg in messages
        ]
        contents: list[Any] = [
            *history_contents,
            *self.__get_contents(user_prompt, file_upload_results),
        ]

        try:
            if not tools:
                result = await self.__llm_client.aio.models.generate_content(  # type: ignore
                    model=model,
                    contents=contents,
                    config=self.__build_config(
                        temperature, system_prompt, thinking_config
                    ),
                )

                if not result.text:
                    logger.error(
                        "No text in response:\n"
                        + json.dumps(result.model_dump(), indent=2)
                    )
                    raise Exception(
                        "No text in response:\n"
                        + json.dumps(result.model_dump(), indent=2)
                    )

                return result.text

            # --- Tool loop ---
            gemini_tools = self.__convert_to_gemini_tools(tools)
            tool_dispatch = {tool.name: tool for tool in tools}
            config = self.__build_config(
                temperature,
                system_prompt,
                thinking_config,
                tool_choice,
                gemini_tools,
            )
            max_iter = (
                max_tool_iterations
                if max_tool_iterations is not None
                else self.__default_max_tool_iterations
            )

            result = await self.__run_tool_loop(
                model, contents, config, tool_dispatch, max_iter
            )

            if not result or not result.text:
                msg = f"No text in response:\n {result.model_dump_json() if result else None}"
                logger.error(msg)
                raise Exception(msg)

            return result.text

        except (ClientError, ServerError) as e:
            if (e.code == 429 and e.status == "RESOURCE_EXHAUSTED") or (
                e.code == 503 and e.status == "UNAVAILABLE"
            ):
                raise GenAiOverloadedError(e.message)
            raise e
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
        system_prompt: str | None = None,
        temperature: float | None = None,
        files: list[FileDto] = [],
        existing_file_upload_results: list[BaseFileUploadResultDto] = [],
        auto_delete_files: bool = True,
        thinking_config: Any | None = None,
        tools: Iterable[ToolDefinition] = [],
        max_tool_iterations: int | None = None,
        tool_choice: ToolChoice = ToolChoice.AUTO,
    ) -> T:
        model = model if model else self.__default_model

        if not user_prompt and not files and not existing_file_upload_results:
            logger.error(
                "user_prompt, files, and file_upload_results cannot be null at the same time."
            )
            raise ValueError(
                "user_prompt, files, and file_upload_results cannot be null at the same time."
            )

        file_upload_results = await self.__get_uploaded_files(
            files, existing_file_upload_results
        )
        contents: list[Any] = self.__get_contents(user_prompt, file_upload_results)

        try:
            if not tools:
                result = await self.__llm_client.aio.models.generate_content(  # type: ignore
                    model=model,
                    contents=contents,
                    config=self.__build_config(
                        temperature,
                        system_prompt,
                        thinking_config,
                        response_json_schema=schema.model_json_schema(),
                    ),
                )

                if not result.text:
                    logger.error(
                        "Got no text in the LLM response: "
                        f"{result.model_dump_json()}"
                    )
                    raise Exception(
                        "Got no text in the LLM response: "
                        f"{result.model_dump_json()}"
                    )

                return schema.model_validate_json(result.text)

            # --- Tool loop (step 1) + re-format (step 2) ---
            gemini_tools = self.__convert_to_gemini_tools(tools)
            tool_dispatch = {tool.name: tool for tool in tools}
            config = self.__build_config(
                temperature,
                system_prompt,
                thinking_config,
                tool_choice,
                gemini_tools,
            )
            max_iter = (
                max_tool_iterations
                if max_tool_iterations is not None
                else self.__default_max_tool_iterations
            )

            loop_result = await self.__run_tool_loop(
                model, contents, config, tool_dispatch, max_iter
            )

            if not loop_result or not loop_result.text:
                logger.error(
                    "Got no text from the tool loop: "
                    f"{loop_result.model_dump_json() if loop_result else None}"
                )
                raise Exception(
                    "Got no text from the tool loop: "
                    f"{loop_result.model_dump_json() if loop_result else None}"
                )

            format_result = await self.__llm_client.aio.models.generate_content(  # type: ignore
                model=model,
                contents=[types.Part.from_text(text=loop_result.text)],
                config=self.__build_config(
                    temperature,
                    None,
                    thinking_config,
                    response_json_schema=schema.model_json_schema(),
                ),
            )

            if not format_result or not format_result.text:
                logger.error(
                    "Got no text from the re-format call: "
                    f"{format_result.model_dump_json() if format_result else None}"
                )
                raise Exception(
                    "Got no text from the re-format call: "
                    f"{format_result.model_dump_json() if format_result else None}"
                )

            return schema.model_validate_json(format_result.text)

        except (ClientError, ServerError) as e:
            if (e.code == 429 and e.status == "RESOURCE_EXHAUSTED") or (
                e.code == 503 and e.status == "UNAVAILABLE"
            ):
                raise GenAiOverloadedError(e.message)
            raise e
        finally:
            if auto_delete_files:
                await self.delete_uploaded_files(file_upload_results)

    @override
    async def upload_file(self, file: FileDto) -> GeminiUploadResultDto:
        uploaded_file: types.File | None = None
        try:
            uploaded_file = await self.__llm_client.aio.files.upload(
                file=BytesIO(file.file_binary),
                config={"mime_type": file.mime_type},
            )
            return GeminiUploadResultDto(uploaded_file=uploaded_file)
        except Exception as e:
            if uploaded_file is not None:
                await self.delete_uploaded_files(
                    [GeminiUploadResultDto(uploaded_file=uploaded_file)]
                )
            logger.error(f"Error uploading files: {str(e)}")
            raise Exception(f"Error uploading files: {str(e)}")

    @override
    async def delete_uploaded_files(
        self, file_upload_results: Iterable[BaseFileUploadResultDto]
    ) -> None:
        for result in file_upload_results:
            if not isinstance(result, GeminiUploadResultDto):
                raise TypeError("Invalid file upload result")
            if not result.uploaded_file.name:
                continue
            await self.__llm_client.aio.files.delete(name=result.uploaded_file.name)

    def __convert_to_gemini_tools(
        self,
        tools: Iterable[ToolDefinition],
    ) -> list[types.Tool]:
        """Convert ToolDefinition objects to Gemini SDK Tool objects."""
        declarations: list[types.FunctionDeclaration] = [
            types.FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters_json_schema=tool.generate_json_schema(),
            )
            for tool in tools
        ]
        return [types.Tool(function_declarations=declarations)]

    def __build_config(
        self,
        temperature: float | None,
        system_prompt: str | None,
        thinking_config: Any | None,
        tool_choice: ToolChoice | None = None,
        gemini_tools: list[types.Tool] | None = None,
        response_json_schema: dict[str, Any] | None = None,
    ) -> GenerateContentConfig:
        """Build a GenerateContentConfig from the common parameters."""
        return GenerateContentConfig(
            response_mime_type="application/json" if gemini_tools is None else None,
            temperature=(
                temperature if temperature is not None else self.__default_temperature
            ),
            system_instruction=system_prompt,
            thinking_config=thinking_config,
            tools=gemini_tools,
            response_json_schema=response_json_schema,
            tool_config=(
                ToolConfig(
                    function_calling_config=self.__get_function_calling_config(
                        tool_choice
                    )
                )
                if tool_choice
                else None
            ),
        )

    def __get_function_calling_config(self, tool_choice: ToolChoice):
        match tool_choice:
            case ToolChoice.AUTO:
                function_calling_config = types.FunctionCallingConfig(
                    mode=FunctionCallingConfigMode.AUTO
                )
            case ToolChoice.REQUIRED:
                function_calling_config = types.FunctionCallingConfig(
                    mode=FunctionCallingConfigMode.ANY
                )
            case ToolChoice.NONE:
                function_calling_config = types.FunctionCallingConfig(
                    mode=FunctionCallingConfigMode.NONE
                )
        return function_calling_config

    async def __run_tool_loop(
        self,
        model: str,
        contents: list[Any],
        config: GenerateContentConfig,
        tool_dispatch: dict[str, ToolDefinition],
        max_iterations: int,
    ) -> types.GenerateContentResponse:
        result = None

        for _ in range(max_iterations):
            result = await self.__llm_client.aio.models.generate_content(  # type: ignore
                model=model,
                contents=contents,
                config=config,
            )

            if not result.function_calls:
                return result

            # Append model response for conversation continuity
            if result.candidates and result.candidates[0].content:
                contents.append(result.candidates[0].content)

            # Execute each function call and append result
            for function_call in result.function_calls:
                if not function_call.name:
                    continue
                tool_def = tool_dispatch.get(function_call.name)
                if tool_def is None:
                    raise ValueError(
                        f"Unknown tool called by model: {function_call.name}"
                    )
                fn_result = await tool_def.execute(**(function_call.args or {}))
                contents.append(
                    types.Part.from_function_response(
                        name=function_call.name,
                        response={"result": fn_result},
                    )
                )

            if config.tool_config:
                config.tool_config.function_calling_config = (
                    self.__get_function_calling_config(ToolChoice.AUTO)
                )

        if not result:
            raise ValueError("Got no response from model")

        return result

    async def __get_uploaded_files(
        self,
        files: list[FileDto],
        existing_file_upload_results: list[BaseFileUploadResultDto],
    ) -> list[GeminiUploadResultDto]:
        file_upload_results = [
            upload_result for upload_result in (await self.__upload_files(files))
        ]

        for result in existing_file_upload_results:
            if not isinstance(result, GeminiUploadResultDto):
                raise TypeError("Invalid file upload result")
            file_upload_results.append(result)

        return file_upload_results

    async def __upload_files(self, files: list[FileDto]) -> list[GeminiUploadResultDto]:
        return [await self.upload_file(file) for file in files]

    def __get_contents(
        self, user_prompt: str | None, file_upload_results: list[GeminiUploadResultDto]
    ) -> list[types.Part]:
        contents: list[types.Part] = []

        if user_prompt:
            contents.append(types.Part.from_text(text=user_prompt))

        for result in file_upload_results:
            if not result.uploaded_file.uri:
                raise ValueError("File URI is required")
            contents.append(
                types.Part.from_uri(
                    file_uri=result.uploaded_file.uri,  # type: ignore
                    mime_type=result.uploaded_file.mime_type,
                )
            )

        return contents
