# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for Tool backend."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from krons.agents.tool import Tool, ToolConfig, tool


class TestToolConfig:
    """Test ToolConfig class."""

    def test_basic_config(self):
        """Test basic ToolConfig creation."""
        config = ToolConfig(
            provider="local",
            name="test_tool",
            description="A test tool",
        )

        assert config.provider == "local"
        assert config.name == "test_tool"
        assert config.description == "A test tool"

    def test_config_with_schema(self):
        """Test ToolConfig with parameters_schema."""
        from pydantic import BaseModel

        class CalcParams(BaseModel):
            expression: str

        config = ToolConfig(
            provider="local",
            name="calculator",
            description="Evaluate expressions",
            parameters_schema=CalcParams,
        )

        assert config.parameters_schema == CalcParams


class TestToolDecorator:
    """Test @tool decorator."""

    def test_decorator_creates_tool(self):
        """Test that @tool decorator creates a Tool instance."""

        @tool(name="test_func", description="Test function")
        async def test_func(x: int) -> int:
            return x * 2

        # The decorated function should be a Tool instance
        assert isinstance(test_func, Tool)
        assert test_func.name == "test_func"
        assert test_func.description == "Test function"

    def test_decorator_with_minimal_args(self):
        """Test decorator with minimal arguments."""

        @tool(name="minimal")
        async def minimal_func():
            return "done"

        assert isinstance(minimal_func, Tool)
        assert minimal_func.name == "minimal"

    def test_decorator_infers_name_from_function(self):
        """Test that decorator infers name from function if not provided."""

        @tool()
        async def auto_named():
            return "test"

        assert isinstance(auto_named, Tool)
        assert auto_named.name == "auto_named"


class TestTool:
    """Test Tool class."""

    @pytest.fixture
    def simple_handler(self):
        """Create a simple async handler."""

        async def handler(x: int, y: int) -> int:
            return x + y

        return handler

    def test_tool_creation(self, simple_handler):
        """Test Tool instantiation."""
        config = ToolConfig(
            provider="local",
            name="adder",
            description="Add two numbers",
        )

        t = Tool(config=config, handler=simple_handler)

        assert t.config.name == "adder"
        assert t.handler == simple_handler

    @pytest.mark.anyio
    async def test_tool_call(self, simple_handler):
        """Test Tool.call method."""
        config = ToolConfig(
            provider="local",
            name="adder",
            description="Add two numbers",
        )

        t = Tool(config=config, handler=simple_handler)
        response = await t.call(arguments={"x": 2, "y": 3})

        assert response.status == "success"
        assert response.data == 5

    @pytest.mark.anyio
    async def test_tool_call_with_error(self):
        """Test Tool.call with handler that raises."""

        async def failing_handler(**kwargs):
            raise ValueError("Test error")

        config = ToolConfig(
            provider="local",
            name="failer",
            description="Always fails",
        )

        t = Tool(config=config, handler=failing_handler)
        response = await t.call(arguments={})

        assert response.status == "error"
        assert "Test error" in str(response.error)

    @pytest.mark.anyio
    async def test_tool_with_schema_validation(self):
        """Test Tool with Pydantic schema validation."""
        from pydantic import BaseModel

        class AddParams(BaseModel):
            x: int
            y: int

        async def add_handler(x: int, y: int) -> int:
            return x + y

        config = ToolConfig(
            provider="local",
            name="adder",
            description="Add numbers",
            parameters_schema=AddParams,
        )

        t = Tool(config=config, handler=add_handler)
        response = await t.call(arguments={"x": 10, "y": 20})

        assert response.status == "success"
        assert response.data == 30


class TestToolSchemaGeneration:
    """Test Tool schema generation methods."""

    def test_to_openai_schema(self):
        """Test OpenAI schema generation."""
        from pydantic import BaseModel

        class WeatherParams(BaseModel):
            location: str
            units: str = "celsius"

        async def get_weather(location: str, units: str = "celsius") -> dict:
            return {"temp": 20, "units": units}

        config = ToolConfig(
            provider="local",
            name="get_weather",
            description="Get current weather",
            parameters_schema=WeatherParams,
        )

        t = Tool(config=config, handler=get_weather)
        schema = t.to_openai_schema()

        assert schema["name"] == "get_weather"
        assert schema["description"] == "Get current weather"
        assert "parameters" in schema
        assert "location" in schema["parameters"]["properties"]

    def test_to_anthropic_schema(self):
        """Test Anthropic schema generation."""
        from pydantic import BaseModel

        class CalcParams(BaseModel):
            expression: str

        async def calculator(expression: str) -> float:
            return eval(expression)

        config = ToolConfig(
            provider="local",
            name="calculator",
            description="Evaluate math expressions",
            parameters_schema=CalcParams,
        )

        t = Tool(config=config, handler=calculator)
        schema = t.to_anthropic_schema()

        assert schema["name"] == "calculator"
        assert schema["description"] == "Evaluate math expressions"
        assert "input_schema" in schema


class TestToolIntegration:
    """Integration tests for Tool with realistic scenarios."""

    @pytest.mark.anyio
    async def test_calculator_tool(self):
        """Test a realistic calculator tool."""

        async def calculator(expression: str) -> float:
            # Simple eval for testing (don't use in production!)
            return eval(expression)

        config = ToolConfig(
            provider="local",
            name="calculator",
            description="Evaluate mathematical expressions",
        )

        t = Tool(config=config, handler=calculator)

        # Test various expressions
        r1 = await t.call(arguments={"expression": "2 + 2"})
        assert r1.data == 4

        r2 = await t.call(arguments={"expression": "10 * 5"})
        assert r2.data == 50

        r3 = await t.call(arguments={"expression": "100 / 4"})
        assert r3.data == 25.0

    @pytest.mark.anyio
    async def test_string_processor_tool(self):
        """Test a string processing tool."""

        async def process_string(text: str, operation: str) -> str:
            ops = {
                "upper": text.upper,
                "lower": text.lower,
                "reverse": lambda: text[::-1],
                "title": text.title,
            }
            return ops.get(operation, lambda: text)()

        config = ToolConfig(
            provider="local",
            name="string_processor",
            description="Process strings",
        )

        t = Tool(config=config, handler=process_string)

        r1 = await t.call(arguments={"text": "hello", "operation": "upper"})
        assert r1.data == "HELLO"

        r2 = await t.call(arguments={"text": "WORLD", "operation": "lower"})
        assert r2.data == "world"

        r3 = await t.call(arguments={"text": "hello world", "operation": "title"})
        assert r3.data == "Hello World"

    @pytest.mark.anyio
    async def test_decorated_tool_call(self):
        """Test calling a decorated tool."""

        @tool(name="doubler", description="Double a number")
        async def double_it(n: int) -> int:
            return n * 2

        # Use the Tool's call method
        response = await double_it.call(arguments={"n": 21})

        assert response.status == "success"
        assert response.data == 42
