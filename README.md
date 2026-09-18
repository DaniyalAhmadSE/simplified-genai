# simplified-genai

A streamlined, unified Generative AI abstraction layer for Python supporting **Google Gemini**, **OpenAI GPT**, and **xAI Grok**.

## Features

- **Unified Provider Interface (`IGenAiProvider`)**: Switch seamlessly between Gemini, GPT, and Grok without rewriting code.
- **Structured Outputs**: Direct type-safe deserialization into Pydantic models with schema enforcement.
- **Python Function Calling / Tool Use (`ToolDefinition`)**: Automatic JSON schema generation from standard Python functions and type hints.
- **Multimodal File Uploads**: Uniform file uploading and handling across multimodal models.
- **Decoupled Chat Sessions**: `IChatSessionStore` protocol with a built-in `InMemoryChatSessionStore` for instant local usage or testing.
- **Provider Factory (`GenAiProviderFactory`)**: Dynamic primary and fallback provider resolution.

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

### 3. Tool Calling / Function Calling

```python
import asyncio
from simplified_genai import GeminiGenAiProvider, ToolDefinition

def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    return f"Weather in {location} is 22°C and sunny."

weather_tool = ToolDefinition(
    function=get_weather,
    description="Get current weather in a city"
)

async def main():
    provider = GeminiGenAiProvider(
        api_key="YOUR_GEMINI_API_KEY",
        default_model="gemini-2.5-flash",
    )

    response = await provider.get_raw_text_response(
        user_prompt="What's the weather in Tokyo?",
        tools=[weather_tool],
    )
    print(response)

asyncio.run(main())
```

### 4. Chat Session Management

```python
import asyncio
from simplified_genai import (
    ChatSessionService,
    GeminiGenAiProvider,
    GenAiChatMessage,
    GenAiChatMessageRole,
    InMemoryChatSessionStore,
)

async def main():
    provider = GeminiGenAiProvider(
        api_key="YOUR_GEMINI_API_KEY",
        default_model="gemini-2.5-flash",
    )
    chat_service = ChatSessionService(store=InMemoryChatSessionStore())
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

## License

This project is licensed under the [MIT License](LICENSE).
