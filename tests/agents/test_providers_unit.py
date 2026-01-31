# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for agent providers with mocked dependencies."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from krons.agents.providers import (
    AnthropicMessagesEndpoint,
    OAIChatEndpoint,
    create_anthropic_config,
    create_oai_chat,
    match_endpoint,
)


class TestCreateAnthropicConfig:
    """Test create_anthropic_config factory function."""

    def test_create_anthropic_config_defaults(self):
        """Test factory with default values."""
        config = create_anthropic_config()

        assert config["provider"] == "anthropic"
        assert config["base_url"] == "https://api.anthropic.com/v1"
        assert config["endpoint"] == "messages"
        assert config["api_key"] == "ANTHROPIC_API_KEY"
        assert config["auth_type"] == "x-api-key"
        assert config["default_headers"]["anthropic-version"] == "2023-06-01"

    def test_create_anthropic_config_custom_values(self):
        """Test factory with custom values."""
        config = create_anthropic_config(
            api_key="custom_key",
            base_url="https://custom.api.com",
            endpoint="custom/endpoint",
            anthropic_version="2024-01-01",
        )

        assert config["provider"] == "anthropic"
        assert config["base_url"] == "https://custom.api.com"
        assert config["endpoint"] == "custom/endpoint"
        assert config["api_key"] == "custom_key"
        assert config["default_headers"]["anthropic-version"] == "2024-01-01"


class TestCreateOaiChat:
    """Test create_oai_chat factory function."""

    def test_create_oai_chat_defaults(self):
        """Test factory with default values."""
        config = create_oai_chat()

        assert config["provider"] == "openai"
        assert config["base_url"] == "https://api.openai.com/v1"
        assert config["endpoint"] == "chat/completions"
        assert config["api_key"] == "OPENAI_API_KEY"

    def test_create_oai_chat_custom_values(self):
        """Test factory with custom values."""
        config = create_oai_chat(
            api_key="custom_key",
            base_url="https://custom.api.com",
            endpoint="custom/endpoint",
        )

        assert config["provider"] == "openai"
        assert config["base_url"] == "https://custom.api.com"
        assert config["endpoint"] == "custom/endpoint"
        assert config["api_key"] == "custom_key"


class TestAnthropicMessagesEndpointInit:
    """Test AnthropicMessagesEndpoint initialization."""

    def test_init_with_none_config(self):
        """Test initialization with config=None uses factory defaults."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test")

        assert endpoint.config.provider == "anthropic"
        assert endpoint.config.base_url == "https://api.anthropic.com/v1"
        assert endpoint.config.endpoint == "messages"

    def test_init_with_dict_config(self):
        """Test initialization with dict config."""
        config_dict = {
            "provider": "anthropic",
            "base_url": "https://api.anthropic.com/v1",
            "endpoint": "messages",
            "api_key": "test_key",
            "name": "test",
            "auth_type": "x-api-key",
        }
        endpoint = AnthropicMessagesEndpoint(config=config_dict)

        assert endpoint.config.provider == "anthropic"
        assert endpoint.config._api_key.get_secret_value() == "test_key"


class TestOAIChatEndpointInit:
    """Test OAIChatEndpoint initialization."""

    def test_init_with_none_config(self):
        """Test initialization with config=None uses factory defaults."""
        endpoint = OAIChatEndpoint(config=None, name="test")

        assert endpoint.config.provider == "openai"
        assert endpoint.config.base_url == "https://api.openai.com/v1"
        assert endpoint.config.endpoint == "chat/completions"

    def test_init_with_dict_config(self):
        """Test initialization with dict config."""
        config_dict = {
            "provider": "openai",
            "base_url": "https://api.openai.com/v1",
            "endpoint": "chat/completions",
            "api_key": "test_key",
            "name": "test",
        }
        endpoint = OAIChatEndpoint(config=config_dict)

        assert endpoint.config.provider == "openai"
        assert endpoint.config._api_key.get_secret_value() == "test_key"


class TestMatchEndpoint:
    """Test match_endpoint function."""

    def test_match_endpoint_anthropic(self):
        """Test matching Anthropic provider."""
        endpoint = match_endpoint("anthropic", "messages")
        assert isinstance(endpoint, AnthropicMessagesEndpoint)

    def test_match_endpoint_openai(self):
        """Test matching OpenAI provider."""
        endpoint = match_endpoint("openai", "chat/completions")
        assert isinstance(endpoint, OAIChatEndpoint)

    def test_match_endpoint_unknown_raises_valueerror(self):
        """Test matching unknown provider raises ValueError (no base_url mapping)."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Unknown provider triggers warning, but OAIChatEndpoint then raises
            # ValueError because there's no base_url mapping for unknown providers
            with pytest.raises(ValueError, match="Unknown provider"):
                match_endpoint("unknown_provider", "chat/completions")
            # Warning is issued before the ValueError
            assert len(w) == 1
            assert "Unknown provider" in str(w[0].message)


class TestAnthropicNormalizeResponse:
    """Test AnthropicMessagesEndpoint.normalize_response method."""

    def test_normalize_response_text_only(self):
        """Test normalize_response with text content only."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test")
        response = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-3-sonnet",
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

        normalized = endpoint.normalize_response(response)

        assert normalized.status == "success"
        assert normalized.data == "Hello!"
        assert normalized.metadata["model"] == "claude-3-sonnet"
        assert normalized.metadata["usage"]["input_tokens"] == 10

    def test_normalize_response_with_tool_use(self):
        """Test normalize_response with tool_use content."""
        endpoint = AnthropicMessagesEndpoint(config=None, name="test")
        response = {
            "id": "msg_456",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "I'll help you with that."},
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "calculator",
                    "input": {"expression": "2+2"},
                },
            ],
            "model": "claude-3-sonnet",
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 20, "output_tokens": 15},
        }

        normalized = endpoint.normalize_response(response)

        assert normalized.status == "success"
        assert "I'll help you with that." in normalized.data
        assert normalized.metadata["stop_reason"] == "tool_use"
        # Should have tool_uses in metadata (Anthropic-specific naming)
        assert "tool_uses" in normalized.metadata


class TestOAIChatNormalizeResponse:
    """Test OAIChatEndpoint.normalize_response method."""

    def test_normalize_response_simple(self):
        """Test normalize_response with simple text response."""
        endpoint = OAIChatEndpoint(config=None, name="test")
        response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello there!",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }

        normalized = endpoint.normalize_response(response)

        assert normalized.status == "success"
        assert normalized.data == "Hello there!"
        assert normalized.metadata["model"] == "gpt-4"
        assert normalized.metadata["usage"]["total_tokens"] == 15

    def test_normalize_response_with_tool_calls(self):
        """Test normalize_response with tool calls."""
        endpoint = OAIChatEndpoint(config=None, name="test")
        response = {
            "id": "chatcmpl-456",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"location": "NYC"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 15, "completion_tokens": 10, "total_tokens": 25},
        }

        normalized = endpoint.normalize_response(response)

        assert normalized.status == "success"
        assert normalized.metadata["finish_reason"] == "tool_calls"
