# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tool - Callable function backend for agents.

A Tool wraps a callable function with:
- Parameter validation via Pydantic schema
- Normalized response format
- Integration with the ResourceBackend/Calling pattern

Usage:
    @tool(name="calculator", description="Perform calculations")
    async def calculator(expression: str) -> float:
        return eval(expression)  # simplified example

    # Or create manually:
    tool = Tool(
        config=ToolConfig(
            provider="local",
            name="calculator",
            description="Perform calculations",
        ),
        handler=calculator,
    )
    response = await tool.call(arguments={"expression": "2 + 2"})
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from typing import Any, get_type_hints

from pydantic import BaseModel, Field

from krons.resources.backend import (
    Calling,
    NormalizedResponse,
    ResourceBackend,
    ResourceConfig,
)

logger = logging.getLogger(__name__)

__all__ = (
    "Tool",
    "ToolCalling",
    "ToolConfig",
    "tool",
)


class ToolConfig(ResourceConfig):
    """Configuration for a Tool backend.

    Extends ResourceConfig with tool-specific fields.
    """

    description: str = Field(default="", description="Human-readable tool description")
    parameters_schema: type[BaseModel] | None = Field(
        default=None,
        exclude=True,
        description="Pydantic model for parameter validation",
    )
    return_type: type | None = Field(
        default=None,
        exclude=True,
        description="Expected return type (for documentation)",
    )


class ToolCalling(Calling):
    """Calling event for Tool execution.

    Wraps a tool invocation with the standard Calling lifecycle (hooks, etc).
    """

    arguments: dict[str, Any] = Field(
        default_factory=dict,
        description="Arguments to pass to the tool handler",
    )

    @property
    def call_args(self) -> dict:
        """Get arguments for backend.call()."""
        return {"arguments": self.arguments}


class Tool(ResourceBackend):
    """Tool backend - wraps a callable function.

    Attributes:
        config: ToolConfig with name, description, schema
        handler: The callable function to execute
    """

    config: ToolConfig = Field(..., description="Tool configuration")
    handler: Callable[..., Any | Awaitable[Any]] = Field(
        ...,
        exclude=True,
        description="The callable function to execute",
    )

    @property
    def event_type(self) -> type[ToolCalling]:
        """Return ToolCalling as the event type for this backend."""
        return ToolCalling

    @property
    def description(self) -> str:
        """Tool description from config."""
        return self.config.description

    @property
    def parameters_schema(self) -> type[BaseModel] | None:
        """Parameters schema from config."""
        return self.config.parameters_schema

    def _validate_arguments(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Validate arguments against schema if defined.

        Args:
            arguments: Arguments dict to validate

        Returns:
            Validated arguments dict

        Raises:
            ValueError: If validation fails
        """
        if self.parameters_schema is None:
            return arguments

        try:
            validated = self.parameters_schema.model_validate(arguments)
            return validated.model_dump()
        except Exception as e:
            raise ValueError(f"Tool argument validation failed: {e}") from e

    async def call(self, arguments: dict[str, Any] | None = None) -> NormalizedResponse:
        """Execute the tool handler with given arguments.

        Args:
            arguments: Arguments to pass to handler

        Returns:
            NormalizedResponse with result or error
        """
        arguments = arguments or {}

        try:
            # Validate arguments
            validated_args = self._validate_arguments(arguments)

            # Execute handler (sync or async)
            if inspect.iscoroutinefunction(self.handler):
                result = await self.handler(**validated_args)
            else:
                result = self.handler(**validated_args)

            return self.normalize_response(result)

        except Exception as e:
            logger.exception(f"Tool '{self.name}' execution failed: {e}")
            return NormalizedResponse(
                status="error",
                data=None,
                error=str(e),
                raw_response={"error": str(e), "arguments": arguments},
            )

    def normalize_response(self, raw_response: Any) -> NormalizedResponse:
        """Normalize tool result to standard format.

        Args:
            raw_response: Raw result from handler

        Returns:
            NormalizedResponse with status and data
        """
        return NormalizedResponse(
            status="success",
            data=raw_response,
            raw_response={"result": raw_response},
        )

    def to_openai_schema(self) -> dict[str, Any]:
        """Generate OpenAI-compatible function schema.

        Returns:
            Dict with name, description, parameters schema
        """
        schema: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
        }

        if self.parameters_schema:
            # Get JSON schema from Pydantic model
            json_schema = self.parameters_schema.model_json_schema()
            # OpenAI expects parameters directly, not wrapped
            schema["parameters"] = {
                "type": "object",
                "properties": json_schema.get("properties", {}),
                "required": json_schema.get("required", []),
            }
        else:
            schema["parameters"] = {"type": "object", "properties": {}}

        return schema

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Generate Anthropic-compatible tool schema.

        Returns:
            Dict with name, description, input_schema
        """
        schema: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
        }

        if self.parameters_schema:
            json_schema = self.parameters_schema.model_json_schema()
            schema["input_schema"] = {
                "type": "object",
                "properties": json_schema.get("properties", {}),
                "required": json_schema.get("required", []),
            }
        else:
            schema["input_schema"] = {"type": "object", "properties": {}}

        return schema


def tool(
    name: str | None = None,
    description: str = "",
    provider: str = "local",
    parameters_schema: type[BaseModel] | None = None,
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator to create a Tool from a function.

    Args:
        name: Tool name (defaults to function name)
        description: Human-readable description
        provider: Provider name (default: "local")
        parameters_schema: Optional Pydantic model for parameter validation

    Returns:
        Decorator that creates a Tool

    Example:
        @tool(name="greet", description="Greet a person")
        async def greet(name: str) -> str:
            return f"Hello, {name}!"

        # Use directly:
        response = await greet.call(arguments={"name": "World"})
    """

    def decorator(func: Callable[..., Any]) -> Tool:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or ""

        # Try to infer parameters schema from type hints
        schema = parameters_schema
        if schema is None:
            hints = get_type_hints(func) if hasattr(func, "__annotations__") else {}
            hints.pop("return", None)
            if hints:
                # Dynamically create a Pydantic model from type hints
                schema = type(
                    f"{tool_name.title()}Params",
                    (BaseModel,),
                    {"__annotations__": hints},
                )

        config = ToolConfig(
            provider=provider,
            name=tool_name,
            description=tool_desc,
            parameters_schema=schema,
        )

        return Tool(config=config, handler=func)

    return decorator
