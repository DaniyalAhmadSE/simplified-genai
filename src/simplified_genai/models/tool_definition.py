import asyncio
from enum import Enum
import functools
import inspect
from collections.abc import Callable, Sequence, Iterable
import types
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict


class ToolDefinition(BaseModel):
    """Wraps a Python function for use as an LLM-callable tool.

    JSON Schema for the function's arguments is auto-generated from
    its signature via generate_json_schema(). Consumers provide
    the function and description — not the schema.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    function: Callable[..., Any]
    description: str

    @property
    def name(self) -> str:
        return self.function.__name__

    def generate_json_schema(self) -> dict[str, Any]:
        """Build JSON Schema from the function's type-annotated signature."""
        sig = inspect.signature(self.function)
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param.annotation is inspect.Parameter.empty:
                continue
            json_type = self.__map_type_to_json_schema_type(param.annotation)
            properties[param_name] = json_type
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        schema: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if required:
            schema["required"] = required
        return schema

    def __map_type_to_json_schema_type(self, annotation: Any) -> dict[str, Any]:
        """Map a Python type to its JSON Schema representation."""
        origin = get_origin(annotation)

        if origin in (list, set, tuple, Sequence, Iterable):
            args = get_args(annotation)
            item_type = self.__map_type_to_json_schema_type(
                args[0] if args and args[0] is not Ellipsis else str
            )
            return {"type": "array", "items": item_type}
        if origin is dict:
            return {"type": "object", "additionalProperties": {}}
        if origin is Literal:
            args = list(get_args(annotation))
            val_type = "string"
            if args and isinstance(args[0], int) and not isinstance(args[0], bool):
                val_type = "integer"
            elif args and isinstance(args[0], (float, int)) and not isinstance(args[0], bool):
                val_type = "number"
            elif args and isinstance(args[0], bool):
                val_type = "boolean"
            return {"type": val_type, "enum": args}
        # Handle Union types (including Optional[X] = Union[X, None])
        # Supports both typing.Union (Union[str, int]) and types.UnionType (str | int)
        if origin is Union or origin is types.UnionType:
            args = get_args(annotation)
            non_none_types = [t for t in args if t is not type(None)]
            subtypes = [self.__map_type_to_json_schema_type(t) for t in non_none_types]
            was_optional = len(non_none_types) < len(args)
            if len(subtypes) == 1:
                # Single type + None → nullable single type
                inner = subtypes[0]
                inner["nullable"] = True
                return inner
            # Multi-type union
            result: dict[str, Any] = {"anyOf": subtypes}
            if was_optional:
                result["nullable"] = True
            return result

        if annotation is Any:
            return {}  # unconstrained — any JSON value is valid

        if isinstance(annotation, type):
            if issubclass(annotation, Enum):
                values = [e.value for e in annotation]
                val_type = "string"
                if values and isinstance(values[0], int) and not isinstance(values[0], bool):
                    val_type = "integer"
                elif values and isinstance(values[0], (float, int)) and not isinstance(values[0], bool):
                    val_type = "number"
                elif values and isinstance(values[0], bool):
                    val_type = "boolean"
                return {"type": val_type, "enum": values}

            if issubclass(annotation, str):
                return {"type": "string"}
            if issubclass(annotation, bool):
                return {"type": "boolean"}
            if issubclass(annotation, int):
                return {"type": "integer"}
            if issubclass(annotation, float):
                return {"type": "number"}
            # Handle Pydantic model annotations for structured output
            if issubclass(annotation, BaseModel):
                return annotation.model_json_schema()

        raise ValueError(
            f"Unsupported type annotation for JSON Schema: {annotation!r}. "
            f"Expected a supported type (str, int, float, bool, list, dict, "
            f"Optional, Union, Any, Sequence, Enum, Literal, or a BaseModel subclass)."
        )

    async def execute(self, **kwargs: Any) -> Any:
        """Execute the tool function with the given arguments.

        Handles both sync and async functions. Sync functions are run in the
        default executor so they don't block the event loop.
        """
        if inspect.iscoroutinefunction(self.function):
            return await self.function(**kwargs)
        return await asyncio.get_running_loop().run_in_executor(
            None, functools.partial(self.function, **kwargs)
        )
