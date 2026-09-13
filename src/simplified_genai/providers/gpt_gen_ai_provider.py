import json
import mimetypes
import re
from typing import Any, Iterable, override

from loguru import logger
from openai import AsyncOpenAI, Omit
from pydantic import BaseModel

from simplified_genai.interfaces.i_gen_ai_provider import (
    IGenAiProvider,
    ToolChoice,
)
from simplified_genai.models.base_file_upload_result_dto import (
    BaseFileUploadResultDto,
)
from simplified_genai.models.file_dto import FileDto
from simplified_genai.models.gen_ai_chat_message import GenAiChatMessage
from simplified_genai.models.gen_ai_model import GenAiModel
from simplified_genai.models.gpt_upload_result_dto import GptUploadResultDto
from simplified_genai.models.tool_definition import ToolDefinition


class GptGenAiProvider(IGenAiProvider):
    def __init__(
        self,
        api_key: str,
        supported_models: list[GenAiModel],
        default_model: GenAiModel,
    ) -> None:
        self.__llm_client = AsyncOpenAI(api_key=api_key)
        self.__default_model = default_model
        self.__supported_models = supported_models
        self.__default_temperature = 0.2
        self.__default_max_tool_iterations = 5

    @property
    def supported_models(self) -> list[GenAiModel]:
        return self.__supported_models

    @override
    async def get_raw_text_response(
        self,
        *,
        user_prompt: str | None,
        model: GenAiModel | None = None,
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
        model = self.__get_validated_model(model)
        if not user_prompt and not files and not existing_file_upload_results:
            logger.error(
                "user_prompt, files, and file_upload_results cannot be null at the same time."
            )
            raise ValueError(
                "user_prompt, files, and file_upload_results cannot be null at the same time."
            )

        input_items = await self.__build_input(
            user_prompt, system_prompt, files, existing_file_upload_results, messages
        )

        _temperature = (
            temperature
            if temperature is not None
            else (
                self.__default_temperature
                if self.__check_if_supports_temperature(model)
                else Omit()
            )
        )

        try:
            if not tools:
                # pyrefly: ignore [no-matching-overload]
                response = await self.__llm_client.responses.create(
                    model=model,
                    input=input_items,  # type: ignore[arg-type]
                    temperature=_temperature,
                )

                text = response.output_text
                if not text:
                    raw = json.dumps(response.model_dump(), indent=2)
                    logger.error(f"No text in response:\n{raw}")
                    raise Exception(f"No text in response:\n{raw}")

                return text

            # --- Tool loop ---
            effective_max_iterations = (
                max_tool_iterations
                if max_tool_iterations is not None
                else self.__default_max_tool_iterations
            )
            openai_tools = self.__convert_to_openai_tools(tools)
            tool_dispatch = {tool.name: tool for tool in tools}

            # pyrefly: ignore [no-matching-overload]
            response = await self.__llm_client.responses.create(
                model=model,
                input=input_items,  # type: ignore[arg-type]
                tools=openai_tools,
                tool_choice=tool_choice.value,
                temperature=_temperature,
            )

            for _ in range(effective_max_iterations):
                function_calls = self.__extract_function_calls_from_response(response)
                if not function_calls:
                    text = response.output_text
                    if not text:
                        raw = json.dumps(response.model_dump(), indent=2)
                        logger.error(f"No text in response:\n{raw}")
                        raise Exception(f"No text in response:\n{raw}")
                    return text

                function_call_output_items: list[dict[str, Any]] = []
                for function_call in function_calls:
                    tool_definition = tool_dispatch[function_call.name]
                    arguments = json.loads(function_call.arguments)
                    function_result = await tool_definition.execute(**arguments)
                    function_call_output_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": function_call.call_id,
                            "output": json.dumps(function_result, default=str),
                        }
                    )

                # pyrefly: ignore [no-matching-overload]
                response = await self.__llm_client.responses.create(
                    model=model,
                    previous_response_id=response.id,
                    input=function_call_output_items,  # type: ignore[arg-type]
                    tools=openai_tools,
                    tool_choice=tool_choice.value,
                    temperature=_temperature,
                )

            # Loop exhausted
            text = response.output_text
            if not text:
                raw = json.dumps(response.model_dump(), indent=2)
                logger.error(f"No text in response:\n{raw}")
                raise Exception(f"No text in response:\n{raw}")
            return text

        except Exception as e:
            logger.error(f"OpenAI bad request error: {e}")
            raise
        finally:
            if auto_delete_files:
                await self.delete_uploaded_files(existing_file_upload_results)

    @override
    async def get_structured_response[T: BaseModel](
        self,
        *,
        schema: type[T],
        user_prompt: str | None,
        model: GenAiModel | None = None,
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
        model = self.__get_validated_model(model)
        if not user_prompt and not files and not existing_file_upload_results:
            logger.error(
                "user_prompt, files, and file_upload_results cannot be null at the same time."
            )
            raise ValueError(
                "user_prompt, files, and file_upload_results cannot be null at the same time."
            )

        input_items = await self.__build_input(
            user_prompt, system_prompt, files, existing_file_upload_results
        )

        _temperature = (
            temperature
            if temperature is not None
            else (
                self.__default_temperature
                if self.__check_if_supports_temperature(model)
                else Omit()
            )
        )

        try:
            if not tools:
                # Fast path: no tools, use responses.parse
                response = await self.__llm_client.responses.parse(
                    model=model,
                    input=input_items,  # type: ignore[arg-type]
                    temperature=_temperature,
                    text_format=self.__get_text_format_schema(schema),
                )

                parsed_output = response.output_parsed
                if not parsed_output:
                    raw = json.dumps(response.model_dump(), indent=2)
                    logger.error(f"Got no parsed output in the LLM response: {raw}")
                    raise Exception(f"Got no parsed output in the LLM response: {raw}")

                return schema.model_validate(parsed_output)

            # --- Tool loop ---
            effective_max_iterations = (
                max_tool_iterations
                if max_tool_iterations is not None
                else self.__default_max_tool_iterations
            )
            openai_tools = self.__convert_to_openai_tools(tools)
            tool_dispatch = {tool.name: tool for tool in tools}

            response = await self.__llm_client.responses.parse(
                model=model,
                input=input_items,  # type: ignore[arg-type]
                tools=openai_tools,  # type: ignore[arg-type]
                tool_choice=tool_choice.value,
                temperature=_temperature,
                text_format=self.__get_text_format_schema(schema),
            )

            for _ in range(effective_max_iterations):
                function_calls = self.__extract_function_calls_from_response(response)
                if not function_calls:
                    if response.output_parsed is None:
                        raw = json.dumps(response.model_dump(), indent=2)
                        logger.error(f"Got no parsed output in the LLM response: {raw}")
                        raise Exception(
                            f"Got no parsed output in the LLM response: {raw}"
                        )
                    return schema.model_validate(response.output_parsed)

                function_call_output_items: list[dict[str, Any]] = []
                for function_call in function_calls:
                    tool_definition = tool_dispatch[function_call.name]
                    arguments = json.loads(function_call.arguments)
                    function_result = await tool_definition.execute(**arguments)
                    function_call_output_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": function_call.call_id,
                            "output": json.dumps(function_result, default=str),
                        }
                    )

                response = await self.__llm_client.responses.parse(
                    model=model,
                    previous_response_id=response.id,
                    input=function_call_output_items,  # type: ignore[arg-type]
                    tools=openai_tools,  # type: ignore[arg-type]
                    tool_choice=tool_choice.value,
                    temperature=_temperature,
                    text_format=self.__get_text_format_schema(schema),
                )

            # Loop exhausted
            if response.output_parsed is None:
                raw = json.dumps(response.model_dump(), indent=2)
                logger.error(f"Got no parsed output in the LLM response: {raw}")
                raise Exception(f"Got no parsed output in the LLM response: {raw}")
            return schema.model_validate(response.output_parsed)

        except Exception as e:
            logger.error(f"OpenAI bad request error: {e}")
            raise
        finally:
            if auto_delete_files:
                await self.delete_uploaded_files(existing_file_upload_results)

    @override
    async def upload_file(self, file: FileDto) -> GptUploadResultDto:
        extension = mimetypes.guess_extension(file.mime_type) or ".bin"
        file_name = f"upload{extension}"

        uploaded_file = await self.__llm_client.files.create(
            file=(file_name, file.file_binary),
            purpose=self.__get_purpose(file),
        )

        if self.__is_image(file.mime_type):
            file_payload: dict[str, Any] = {
                "type": "input_image",
                "file_id": uploaded_file.id,
            }
        else:
            file_payload = {"type": "input_file", "file_id": uploaded_file.id}

        return GptUploadResultDto(file_payload=file_payload)

    @override
    async def delete_uploaded_files(
        self, file_upload_results: Iterable[BaseFileUploadResultDto]
    ) -> None:
        for result in file_upload_results:
            if isinstance(result, GptUploadResultDto):
                payload = result.file_payload
                file_id = payload.get("file_id")

                if file_id:
                    try:
                        logger.info(f"Deleting file {file_id}")
                        await self.__llm_client.files.delete(file_id)
                    except Exception as e:
                        logger.error(f"Failed to delete file {file_id}: {e}")

    def __get_purpose(self, file: FileDto):
        if self.__is_image(file.mime_type):
            logger.info("Image detected, setting purpose to image.")
            return "vision"
        return "assistants"

    def __convert_to_openai_tools(self, tools: Iterable[ToolDefinition]) -> list[Any]:
        return [
            {
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.generate_json_schema(),
            }
            for tool in tools
        ]

    def __extract_function_calls_from_response(self, response: Any) -> list[Any]:
        return [
            item
            for item in response.output
            if getattr(item, "type", None) == "function_call"
        ]

    async def __build_input(
        self,
        user_prompt: str | None,
        system_prompt: str | None,
        files: list[FileDto],
        file_upload_results: list[BaseFileUploadResultDto],
        messages: Iterable[GenAiChatMessage] = [],
    ) -> list[dict[str, Any]]:
        input_items: list[dict[str, Any]] = []

        if system_prompt:
            input_items.append({"role": "system", "content": system_prompt})

        for msg in messages:
            input_items.append(
                {
                    "role": msg.role.value,
                    "content": [{"type": "input_text", "text": msg.content}],
                }
            )

        user_content: list[dict[str, Any]] = []

        if user_prompt:
            user_content.append({"type": "input_text", "text": user_prompt})

        uploaded_file_payloads = await self.__get_uploaded_file_payloads(
            files, file_upload_results
        )

        user_content.extend(uploaded_file_payloads)

        if user_content:
            input_items.append({"role": "user", "content": user_content})

        return input_items

    async def __upload_files(self, files: list[FileDto]) -> list[GptUploadResultDto]:
        return [await self.upload_file(file) for file in files]

    async def __get_uploaded_file_payloads(
        self,
        files: list[FileDto],
        file_upload_results: list[BaseFileUploadResultDto],
    ) -> list[dict[str, Any]]:
        uploaded_file_payloads = [
            upload_result.file_payload
            for upload_result in (await self.__upload_files(files))
        ]

        for result in file_upload_results:
            if not isinstance(result, GptUploadResultDto):
                raise TypeError("Invalid file upload result")
            uploaded_file_payloads.append(result.file_payload)

        return uploaded_file_payloads

    def __is_image(self, mime_type: str) -> bool:
        return mime_type.lower().startswith("image/")

    def __get_text_format_schema[T: BaseModel](self, schema: type[T]):
        sanitized_name = re.sub(r"[^a-zA-Z0-9_-]", "_", schema.__name__)[:64]
        return type(sanitized_name, (schema,), {})

    def __check_if_supports_temperature(self, default_model: GenAiModel):
        return default_model in [GenAiModel.GPT_5_NANO, GenAiModel.GPT_5_4_NANO]

    def __get_validated_model(self, model: GenAiModel | None) -> GenAiModel:
        if not model:
            return self.__default_model

        if model in self.supported_models:
            return model

        logger.warning(
            f"Model '{model}' not found in supported models list, using default model."
        )
        return self.__default_model
