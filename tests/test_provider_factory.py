from typing import Any, Iterable

from pydantic import BaseModel
import pytest

from simplified_genai import (
    BaseFileUploadResultDto,
    FileDto,
    GenAiProviderFactory,
    IGenAiProvider,
    ToolChoice,
    ToolDefinition,
)


class MockGeminiProvider(IGenAiProvider):
    async def get_raw_text_response(self, **kwargs: Any) -> str:
        return "gemini raw"

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
        raise NotImplementedError

    async def upload_file(self, file: FileDto) -> BaseFileUploadResultDto:
        raise NotImplementedError

    async def delete_uploaded_files(
        self, file_upload_results: Iterable[BaseFileUploadResultDto]
    ) -> None:
        pass


class MockGptProvider(IGenAiProvider):
    async def get_raw_text_response(self, **kwargs: Any) -> str:
        return "gpt raw"

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
        raise NotImplementedError

    async def upload_file(self, file: FileDto) -> BaseFileUploadResultDto:
        raise NotImplementedError

    async def delete_uploaded_files(
        self, file_upload_results: Iterable[BaseFileUploadResultDto]
    ) -> None:
        pass


def test_factory_resolves_primary_and_fallback() -> None:
    gemini = MockGeminiProvider()
    gpt = MockGptProvider()

    factory = GenAiProviderFactory(
        providers=[gemini, gpt],
        default_primary_gen_ai_provider_type=MockGeminiProvider,
        default_fallback_gen_ai_provider_type=MockGptProvider,
    )

    primary = factory.get_primary_gen_ai_provider()
    fallback = factory.get_fallback_gen_ai_provider()

    assert primary is gemini
    assert fallback is gpt


def test_factory_set_primary_and_fallback_type() -> None:
    gemini = MockGeminiProvider()
    gpt = MockGptProvider()

    factory = GenAiProviderFactory(
        providers=[gemini, gpt],
        default_primary_gen_ai_provider_type=MockGeminiProvider,
        default_fallback_gen_ai_provider_type=MockGptProvider,
    )

    factory.set_primary_gen_ai_provider_type(MockGptProvider)
    factory.set_fallback_gen_ai_provider_type(MockGeminiProvider)

    assert factory.get_primary_gen_ai_provider() is gpt
    assert factory.get_fallback_gen_ai_provider() is gemini


def test_factory_unknown_provider_raises() -> None:
    class UnknownProvider(MockGeminiProvider):
        pass

    factory = GenAiProviderFactory(
        providers=[MockGeminiProvider()],
        default_primary_gen_ai_provider_type=MockGeminiProvider,
        default_fallback_gen_ai_provider_type=MockGeminiProvider,
    )

    with pytest.raises(ValueError, match="No provider registered"):
        factory.get_gen_ai_provider(UnknownProvider)
