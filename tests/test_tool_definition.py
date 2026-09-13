from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel
import pytest

from simplified_genai import ToolDefinition


def test_name_from_function_name() -> None:
    def foo() -> None: ...

    tool = ToolDefinition(function=foo, description="Some function")
    assert tool.name == "foo"


def test_description_stored_as_provided() -> None:
    def bar() -> None: ...

    tool = ToolDefinition(function=bar, description="My description")
    assert tool.description == "My description"


def test_generate_json_schema_infers_from_signature() -> None:
    def add(a: int, b: int) -> int: ...

    tool = ToolDefinition(function=add, description="Add two ints")
    schema = tool.generate_json_schema()

    assert schema == {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    }


def test_model_can_be_constructed() -> None:
    def greet(name: str) -> str: ...

    tool = ToolDefinition(function=greet, description="Greet someone")
    assert tool.function is greet
    assert tool.name == "greet"


def test_callable_invocation() -> None:
    def add(a: int, b: int) -> int:
        return a + b

    tool = ToolDefinition(function=add, description="Add numbers")
    result = tool.function(2, 3)
    assert result == 5


def test_json_schema_optional_parameter() -> None:
    def greet(name: str, greeting: Optional[str] = None) -> str: ...

    tool = ToolDefinition(function=greet, description="Greeting")
    schema = tool.generate_json_schema()

    assert schema["properties"]["greeting"] == {
        "type": "string",
        "nullable": True,
    }
    assert "greeting" not in schema.get("required", [])


def test_json_schema_literal_parameter() -> None:
    def choose(option: Literal["a", "b", "c"]) -> None: ...

    tool = ToolDefinition(function=choose, description="Choose option")
    schema = tool.generate_json_schema()

    assert schema["properties"]["option"] == {
        "type": "string",
        "enum": ["a", "b", "c"],
    }


def test_json_schema_enum_parameter() -> None:
    class Color(str, Enum):
        RED = "red"
        BLUE = "blue"

    def paint(color: Color) -> None: ...

    tool = ToolDefinition(function=paint, description="Paint with color")
    schema = tool.generate_json_schema()

    assert schema["properties"]["color"] == {
        "type": "string",
        "enum": ["red", "blue"],
    }


def test_json_schema_pydantic_model_parameter() -> None:
    class Address(BaseModel):
        city: str
        zip_code: str

    def ship(address: Address) -> None: ...

    tool = ToolDefinition(function=ship, description="Ship item")
    schema = tool.generate_json_schema()

    assert "properties" in schema["properties"]["address"]
    assert "city" in schema["properties"]["address"]["properties"]


@pytest.mark.asyncio
async def test_execute_sync_function() -> None:
    def multiply(x: int, y: int) -> int:
        return x * y

    tool = ToolDefinition(function=multiply, description="Multiply")
    result = await tool.execute(x=4, y=5)
    assert result == 20


@pytest.mark.asyncio
async def test_execute_async_function() -> None:
    async def fetch_data(key: str) -> str:
        return f"value_for_{key}"

    tool = ToolDefinition(function=fetch_data, description="Fetch")
    result = await tool.execute(key="test")
    assert result == "value_for_test"
