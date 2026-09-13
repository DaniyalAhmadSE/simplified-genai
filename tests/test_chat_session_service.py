import pytest

from simplified_genai import (
    ChatSessionService,
    InMemoryChatSessionStore,
    GenAiChatMessage,
    GenAiChatMessageRole,
)


@pytest.fixture
def chat_service() -> ChatSessionService:
    store = InMemoryChatSessionStore()
    return ChatSessionService(store=store)


def test_start_session_returns_unique_string(chat_service: ChatSessionService) -> None:
    session_a = chat_service.start_session()
    session_b = chat_service.start_session()

    assert session_a
    assert session_b
    assert session_a != session_b


@pytest.mark.asyncio
async def test_get_history_empty_session(chat_service: ChatSessionService) -> None:
    session_id = chat_service.start_session()
    history = await chat_service.get_history(session_id)
    assert list(history) == []


@pytest.mark.asyncio
async def test_append_turn_and_get_history(chat_service: ChatSessionService) -> None:
    session_id = chat_service.start_session()

    user_msg = GenAiChatMessage(
        role=GenAiChatMessageRole.USER,
        content="What is Python?",
    )
    assistant_msg = GenAiChatMessage(
        role=GenAiChatMessageRole.ASSISTANT,
        content="Python is a programming language.",
    )

    await chat_service.append_turn(session_id, user_msg, assistant_msg)

    history = await chat_service.get_history(session_id)
    assert len(history) == 2
    assert history[0].role == GenAiChatMessageRole.USER
    assert history[0].content == "What is Python?"
    assert history[1].role == GenAiChatMessageRole.ASSISTANT
    assert history[1].content == "Python is a programming language."


@pytest.mark.asyncio
async def test_clear_session(chat_service: ChatSessionService) -> None:
    session_id = chat_service.start_session()

    await chat_service.append_turn(
        session_id,
        GenAiChatMessage(role=GenAiChatMessageRole.USER, content="Hello"),
        GenAiChatMessage(role=GenAiChatMessageRole.ASSISTANT, content="Hi"),
    )

    assert len(await chat_service.get_history(session_id)) == 2

    await chat_service.clear_session(session_id)

    assert len(await chat_service.get_history(session_id)) == 0
