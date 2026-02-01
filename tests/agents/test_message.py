# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for agent message content types."""

from __future__ import annotations

import pytest

from krons.agents.message import (
    ActionRequest,
    ActionResponse,
    Assistant,
    Instruction,
    MessageContent,
    MessageRole,
    System,
)


class TestMessageRole:
    """Test MessageRole enum."""

    def test_role_values(self):
        """Test that all expected roles exist."""
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"
        assert MessageRole.TOOL.value == "tool"


class TestSystemContent:
    """Test System message content."""

    def test_create_system(self):
        """Test creating System content."""
        system = System.create(system_message="You are a helpful assistant.")

        assert system.system_message == "You are a helpful assistant."
        assert system.role == MessageRole.SYSTEM

    def test_system_render(self):
        """Test System render method."""
        system = System.create(system_message="Be concise.")
        rendered = system.render()

        assert "Be concise." in rendered

    def test_system_to_chat(self):
        """Test System to_chat method."""
        system = System.create(system_message="Be helpful.")
        chat = system.to_chat()

        assert chat is not None
        assert chat["role"] == "system"
        assert "Be helpful." in chat["content"]


class TestInstructionContent:
    """Test Instruction message content."""

    def test_create_instruction_simple(self):
        """Test creating simple Instruction using create factory."""
        instr = Instruction.create(instruction="What is 2+2?")

        assert instr.primary == "What is 2+2?"
        assert instr.role == MessageRole.USER

    def test_create_instruction_with_context(self):
        """Test creating Instruction with context."""
        instr = Instruction.create(
            instruction="Summarize this",
            context=["The quick brown fox jumps over the lazy dog."],
        )

        assert instr.primary == "Summarize this"
        assert instr.context == ["The quick brown fox jumps over the lazy dog."]

    def test_instruction_render(self):
        """Test Instruction render method."""
        instr = Instruction.create(instruction="Test instruction")
        rendered = instr.render()

        assert "Test instruction" in rendered

    def test_instruction_to_chat(self):
        """Test Instruction to_chat method."""
        instr = Instruction.create(instruction="Hello world")
        chat = instr.to_chat()

        assert chat is not None
        assert chat["role"] == "user"
        assert "Hello world" in str(chat["content"])


class TestAssistantContent:
    """Test Assistant message content."""

    def test_create_assistant(self):
        """Test creating Assistant content."""
        assistant = Assistant.create(assistant_response="Here is my response.")

        assert assistant.assistant_response == "Here is my response."
        assert assistant.role == MessageRole.ASSISTANT

    def test_assistant_render(self):
        """Test Assistant render method."""
        assistant = Assistant.create(assistant_response="The answer is 42.")
        rendered = assistant.render()

        assert rendered == "The answer is 42."

    def test_assistant_to_chat(self):
        """Test Assistant to_chat method."""
        assistant = Assistant.create(assistant_response="I can help with that.")
        chat = assistant.to_chat()

        assert chat is not None
        assert chat["role"] == "assistant"
        assert "I can help with that." in str(chat["content"])


class TestActionRequest:
    """Test ActionRequest message content."""

    def test_create_action_request(self):
        """Test creating ActionRequest."""
        action = ActionRequest.create(
            function="calculator",
            arguments={"expression": "2+2"},
        )

        assert action.function == "calculator"
        assert action.arguments == {"expression": "2+2"}

    def test_action_request_render(self):
        """Test ActionRequest render."""
        action = ActionRequest.create(
            function="get_weather",
            arguments={"location": "NYC"},
        )
        rendered = action.render()

        assert "get_weather" in rendered
        assert "NYC" in rendered


class TestActionResponse:
    """Test ActionResponse message content."""

    def test_create_action_response_success(self):
        """Test creating successful ActionResponse."""
        response = ActionResponse.create(
            request_id="req_123",
            result="4",
        )

        assert response.request_id == "req_123"
        assert response.result == "4"
        assert response.success is True

    def test_create_action_response_error(self):
        """Test creating ActionResponse with error."""
        response = ActionResponse.create(
            request_id="req_456",
            error="Division by zero",
        )

        assert response.error == "Division by zero"
        assert response.success is False

    def test_action_response_render_success(self):
        """Test ActionResponse render for success case."""
        response = ActionResponse.create(
            request_id="req_789",
            result={"answer": 42},
        )
        rendered = response.render()

        assert "success" in rendered.lower() or "true" in rendered.lower()

    def test_action_response_render_error(self):
        """Test ActionResponse render for error case."""
        response = ActionResponse.create(
            request_id="req_000",
            error="Something went wrong",
        )
        rendered = response.render()

        assert "error" in rendered.lower() or "wrong" in rendered.lower()


class TestMessageContentProtocol:
    """Test MessageContent protocol methods."""

    def test_system_is_message_content(self):
        """Test System is a MessageContent."""
        system = System.create(system_message="Test")
        assert isinstance(system, MessageContent)

    def test_instruction_is_message_content(self):
        """Test Instruction is a MessageContent."""
        instr = Instruction.create(instruction="Test")
        assert isinstance(instr, MessageContent)

    def test_assistant_is_message_content(self):
        """Test Assistant is a MessageContent."""
        assistant = Assistant.create(assistant_response="Test")
        assert isinstance(assistant, MessageContent)

    def test_action_request_is_message_content(self):
        """Test ActionRequest is a MessageContent."""
        action = ActionRequest.create(function="test", arguments={})
        assert isinstance(action, MessageContent)

    def test_action_response_is_message_content(self):
        """Test ActionResponse is a MessageContent."""
        response = ActionResponse.create(result="test")
        assert isinstance(response, MessageContent)
