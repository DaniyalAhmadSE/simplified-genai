# simplified-genai

A streamlined, unified Generative AI abstraction layer for Python supporting **Google Gemini**, **OpenAI GPT**, and **xAI Grok**.

Built with modern Python (3.12+) features, strict type safety, open-ended model selection, and zero boilerplate.

## Features

- **Unified Provider Interface (`IGenAiProvider`)**: Switch seamlessly between Gemini, GPT, and Grok without rewriting application logic.
- **Open-Ended Model Selection**: Use any supported or custom model ID as a string (`"gemini-2.5-flash"`, `"gpt-4o"`, `"grok-4-1-fast-reasoning"`).
- **Structured Outputs**: Direct type-safe deserialization into Pydantic models with schema enforcement.
- **Function / Tool Calling (`ToolDefinition`, `ToolChoice`)**: Automatic JSON schema generation from standard Python functions and type annotations.
- **Multimodal Inputs & File Attachments (`FilePayload`, `FileUploadResponse`)**: Uniform file handling, uploading, and automatic remote cleanup across multimodal models.
- **Decoupled Chat Sessions**: `IChatSessionStorageProvider` protocol with a built-in `InMemoryChatSessionStorageProvider` for testing, local scripts, or production extensions (Redis, Postgres, Mongo).
- **Dynamic Provider Factory (`GenAiProviderFactory`)**: Dynamic resolution of primary and fallback providers for high availability.

---

## Requirements

- **Python 3.12+** (leverages PEP 695 type parameter syntax)

---

## Installation

```bash
pip install simplified-genai
```

Or using `uv`:

```bash
uv add simplified-genai
```

---

## Quickstart

### 1. Basic Text Generation

```python
import asyncio
from simplified_genai import GeminiGenAiProvider

async def main():
    provider = GeminiGenAiProvider(
        api_key="YOUR_GEMINI_API_KEY",
        default_model="gemini-2.5-flash",
    )

    response = await provider.get_raw_text_response(
        user_prompt="Explain quantum computing in one sentence."
    )
    print(response)

asyncio.run(main())
```

---

### 2. Structured Output with Pydantic

```python
import asyncio
from pydantic import BaseModel
from simplified_genai import GptGenAiProvider

class CapitalCity(BaseModel):
    country: str
    capital: str
    population_millions: float

async def main():
    provider = GptGenAiProvider(
        api_key="YOUR_OPENAI_API_KEY",
        default_model="gpt-5-nano",
    )

    result = await provider.get_structured_response(
        schema=CapitalCity,
        user_prompt="What is the capital of France?",
    )
    print(f"{result.capital}, {result.country} (Pop: {result.population_millions}M)")

asyncio.run(main())
```

---

### 3. Tool Calling / Function Calling

```python
import asyncio
from simplified_genai import GeminiGenAiProvider, ToolChoice, ToolDefinition

def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    return f"Weather in {location} is 22°C and sunny."

weather_tool = ToolDefinition(
    function=get_weather,
    description="Get current weather in a city",
)

async def main():
    provider = GeminiGenAiProvider(
        api_key="YOUR_GEMINI_API_KEY",
        default_model="gemini-2.5-flash",
    )

    response = await provider.get_raw_text_response(
        user_prompt="What's the weather in Tokyo?",
        tools=[weather_tool],
        tool_choice=ToolChoice.AUTO,
    )
    print(response)

asyncio.run(main())
```

---

### 4. Multimodal Inputs & File Attachments

Attach binary files (images, audio, PDFs, documents) using `FilePayload`:

```python
import asyncio
from pathlib import Path
from simplified_genai import GeminiGenAiProvider, FilePayload

async def main():
    provider = GeminiGenAiProvider(
        api_key="YOUR_GEMINI_API_KEY",
        default_model="gemini-2.5-flash",
    )

    image_bytes = Path("sample.png").read_bytes()
    image_payload = FilePayload(
        file_binary=image_bytes,
        mime_type="image/png",
    )

    response = await provider.get_raw_text_response(
        user_prompt="Describe what you see in this image.",
        files=[image_payload],
        auto_delete_files=True,  # Automatically cleans up remote uploaded files after completion
    )
    print(response)

asyncio.run(main())
```

---

### 5. Chat Session Management

Manage multi-turn conversation history with automatic turn tracking:

```python
import asyncio
from simplified_genai import (
    ChatSessionService,
    GeminiGenAiProvider,
    GenAiChatMessage,
    GenAiChatMessageRole,
    InMemoryChatSessionStorageProvider,
)

async def main():
    provider = GeminiGenAiProvider(
        api_key="YOUR_GEMINI_API_KEY",
        default_model="gemini-2.5-flash",
    )
    # Use built-in in-memory storage or implement IChatSessionStorageProvider for Redis/Postgres
    chat_service = ChatSessionService(store=InMemoryChatSessionStorageProvider())
    session_id = chat_service.start_session()

    # First turn
    user_prompt = "Hello! My name is Alice."
    history = await chat_service.get_history(session_id)

    response = await provider.get_raw_text_response(
        user_prompt=user_prompt,
        messages=history,
    )
    print(f"Assistant: {response}")

    await chat_service.append_turn(
        session_id=session_id,
        user_message=GenAiChatMessage(role=GenAiChatMessageRole.USER, content=user_prompt),
        assistant_message=GenAiChatMessage(role=GenAiChatMessageRole.ASSISTANT, content=response),
    )

    # Follow-up turn with conversation history
    follow_up_prompt = "What is my name?"
    history = await chat_service.get_history(session_id)

    response = await provider.get_raw_text_response(
        user_prompt=follow_up_prompt,
        messages=history,
    )
    print(f"Assistant: {response}")

asyncio.run(main())
```

---

### 6. Provider Factory & Fallback Routing

Configure primary and fallback providers for resilience:

```python
from simplified_genai import (
    GeminiGenAiProvider,
    GrokGenAiProvider,
    GenAiProviderFactory,
)

gemini = GeminiGenAiProvider(
    api_key="YOUR_GEMINI_API_KEY",
    default_model="gemini-2.5-flash",
)

grok = GrokGenAiProvider(
    api_key="YOUR_XAI_API_KEY",
    default_model="grok-4-1-fast-reasoning",
)

factory = GenAiProviderFactory(
    providers=[gemini, grok],
    default_primary_gen_ai_provider_type=GeminiGenAiProvider,
    default_fallback_gen_ai_provider_type=GrokGenAiProvider,
)

# Retrieve default primary provider
primary = factory.get_primary_gen_ai_provider()

# Retrieve provider by specific type
grok_provider = factory.get_gen_ai_provider(GrokGenAiProvider)
```

---

## Supported Providers & Models

Model selection is open-ended. You can pass any standard model ID string supported by the underlying provider:

| Provider Class | Underlying SDK | Common Models (`default_model` / `model`) |
| :--- | :--- | :--- |
| `GeminiGenAiProvider` | `google-genai` | `"gemini-2.5-flash"`, `"gemini-3.5-flash"`, `"gemini-3.7-flash"` |
| `GptGenAiProvider` | `openai` | `"gpt-4o"`, `"gpt-4o-mini"`, `"gpt-5-nano"`, `"gpt-5.5"` |
| `GrokGenAiProvider` | `xai-sdk` | `"grok-4-1-fast-reasoning"` |

---

## License

This project is licensed under the [MIT License](LICENSE).

