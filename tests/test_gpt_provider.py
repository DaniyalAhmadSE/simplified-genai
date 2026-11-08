from unittest.mock import AsyncMock, MagicMock, patch

from openai import Omit
import pytest

from simplified_genai import (
    FilePayload,
    GenAiChatMessage,
    GenAiChatMessageRole,
    GptGenAiProvider,
    GptUploadResultDto,
)
from simplified_genai.providers.gemini_gen_ai_provider import GeminiGenAiProvider
from simplified_genai.providers.grok_gen_ai_provider import GrokGenAiProvider


@pytest.mark.asyncio
async def test_provider_initialization_with_custom_endpoints() -> None:
    gpt_provider = GptGenAiProvider(
        api_key="test-key",
        default_model="gpt-4o",
        base_url="https://custom-openai.proxy/v1",
    )
    assert gpt_provider is not None

    gemini_provider = GeminiGenAiProvider(
        api_key="test-key",
        default_model="gemini-2.5-flash",
        base_url="https://custom-gemini.proxy/v1",
    )
    assert gemini_provider is not None

    grok_provider = GrokGenAiProvider(
        api_key="test-key",
        default_model="grok-4-1-fast-reasoning",
        api_host="custom-grok.proxy:443",
    )
    assert grok_provider is not None


def test_gpt_build_input_roles() -> None:
    provider = GptGenAiProvider(api_key="test-key", default_model="gpt-4o")
    build_input = getattr(provider, "_GptGenAiProvider__build_input")

    messages = [
        GenAiChatMessage(role=GenAiChatMessageRole.USER, content="Hello"),
        GenAiChatMessage(
            role=GenAiChatMessageRole.ASSISTANT, content="Hi, how can I help?"
        ),
    ]

    items = build_input(
        user_prompt="Continue conversation",
        system_prompt="You are a helpful assistant",
        file_upload_results=[],
        messages=messages,
    )

    assert items[0] == {
        "role": "system",
        "content": "You are a helpful assistant",
    }
    assert items[1] == {
        "role": "user",
        "content": [{"type": "input_text", "text": "Hello"}],
    }
    assert items[2] == {
        "role": "assistant",
        "content": [{"type": "output_text", "text": "Hi, how can I help?"}],
    }
    assert items[3] == {
        "role": "user",
        "content": [{"type": "input_text", "text": "Continue conversation"}],
    }


@pytest.mark.asyncio
async def test_gpt_upload_file_image_detail_auto() -> None:
    provider = GptGenAiProvider(api_key="test-key", default_model="gpt-4o")
    mock_file = MagicMock()
    mock_file.id = "file_img_123"

    mock_files_api = MagicMock()
    mock_files_api.create = AsyncMock(return_value=mock_file)

    llm_client = getattr(provider, "_GptGenAiProvider__llm_client")
    with patch.object(llm_client, "files", mock_files_api):
        payload = FilePayload(file_binary=b"dummy-image", mime_type="image/png")
        result = await provider.upload_file(payload)

        mock_files_api.create.assert_awaited_once_with(
            file=("upload.png", b"dummy-image"),
            purpose="assistants",
        )
        assert result.file_payload == {
            "type": "input_image",
            "file_id": "file_img_123",
            "detail": "auto",
        }


@pytest.mark.asyncio
async def test_gpt_auto_delete_cleans_new_and_existing_files() -> None:
    provider = GptGenAiProvider(api_key="test-key", default_model="gpt-4o")

    mock_response = MagicMock()
    mock_response.output_text = "mocked text"

    mock_responses_api = MagicMock()
    mock_responses_api.create = AsyncMock(return_value=mock_response)

    delete_mock = AsyncMock()

    new_file = FilePayload(file_binary=b"new-file", mime_type="application/pdf")
    existing_result = GptUploadResultDto(
        file_payload={"type": "input_file", "file_id": "file_existing_456"}
    )

    with (
        patch.object(provider, "upload_file") as mock_upload,
        patch.object(provider, "delete_uploaded_files", delete_mock),
        patch.object(
            getattr(provider, "_GptGenAiProvider__llm_client"),
            "responses",
            mock_responses_api,
        ),
    ):
        mock_upload.return_value = GptUploadResultDto(
            file_payload={"type": "input_file", "file_id": "file_new_123"}
        )

        res = await provider.get_raw_text_response(
            user_prompt="test",
            files=[new_file],
            existing_file_upload_results=[existing_result],
            auto_delete_files=True,
        )

        assert res == "mocked text"
        delete_mock.assert_awaited_once()
        deleted_list = delete_mock.call_args[0][0]
        assert len(deleted_list) == 2
        assert deleted_list[0].file_payload["file_id"] == "file_new_123"
        assert deleted_list[1].file_payload["file_id"] == "file_existing_456"

        # Verify temperature defaulted to Omit() when not specified
        create_kwargs = mock_responses_api.create.call_args.kwargs
        assert isinstance(create_kwargs["temperature"], Omit)
