# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for agent message content types."""

from __future__ import annotations

from krons.agent.message import (
    ActionRequest,
    ActionResponse,
    Assistant,
    Instruction,
    Role,
    RoledContent,
    System,
)


class TestRole:
    """Test Role enum."""

    def test_role_values(self):
        """Test that all expected roles exist."""
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.ACTION.value == "action"
        assert Role.UNSET.value == "unset"


class TestSystemContent:
    """Test System message content."""

    def test_create_system(self):
        """Test creating System content."""
        system = System.create(system_message="You are a helpful assistant.")

        assert system.system_message == "You are a helpful assistant."
        assert system.role == Role.SYSTEM

    def test_system_render(self):
        """Test System render method."""
        system = System.create(system_message="Be concise.")
        rendered = system.render()

        assert "Be concise." in rendered

    def test_system_with_datetime(self):
        """Test System with datetime."""
        system = System.create(
            system_message="Hello",
            system_datetime="2025-01-01T00:00:00Z",
        )
        rendered = system.render()

        assert "2025-01-01T00:00:00Z" in rendered
        assert "Hello" in rendered


class TestInstructionContent:
    """Test Instruction message content."""

    def test_create_instruction_simple(self):
        """Test creating simple Instruction using create factory."""
        instr = Instruction.create(primary="What is 2+2?")

        assert instr.primary == "What is 2+2?"
        assert instr.role == Role.USER

    def test_create_instruction_with_context(self):
        """Test creating Instruction with context."""
        instr = Instruction.create(
            primary="Summarize this",
            context=["The quick brown fox jumps over the lazy dog."],
        )

        assert instr.primary == "Summarize this"
        assert instr.context == ["The quick brown fox jumps over the lazy dog."]

    def test_instruction_render(self):
        """Test Instruction render method."""
        instr = Instruction.create(primary="Test instruction")
        rendered = instr.render()

        assert "Test instruction" in rendered


class TestAssistantContent:
    """Test Assistant message content."""

    def test_assistant_role(self):
        """Test Assistant has correct role."""
        # Direct instantiation for testing
        assistant = Assistant(response="Test response")
        assert assistant.role == Role.ASSISTANT

    def test_assistant_render(self):
        """Test Assistant render method."""
        assistant = Assistant(response="The answer is 42.")
        rendered = assistant.render()

        assert rendered == "The answer is 42."

    def test_assistant_render_unset(self):
        """Test Assistant render with unset response."""
        assistant = Assistant()
        rendered = assistant.render()
        assert rendered == ""


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

    def test_action_request_from_dict(self):
        """Test ActionRequest from_dict."""
        action = ActionRequest.from_dict(
            {
                "function": "test_func",
                "arguments": {"arg1": "val1"},
            }
        )
        assert action.function == "test_func"
        assert action.arguments == {"arg1": "val1"}


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


class TestRoledContentProtocol:
    """Test RoledContent base class methods."""

    def test_system_is_roled_content(self):
        """Test System is a RoledContent."""
        system = System.create(system_message="Test")
        assert isinstance(system, RoledContent)

    def test_instruction_is_roled_content(self):
        """Test Instruction is a RoledContent."""
        instr = Instruction.create(primary="Test")
        assert isinstance(instr, RoledContent)

    def test_assistant_is_roled_content(self):
        """Test Assistant is a RoledContent."""
        assistant = Assistant(response="Test")
        assert isinstance(assistant, RoledContent)

    def test_action_request_is_roled_content(self):
        """Test ActionRequest is a RoledContent."""
        action = ActionRequest.create(function="test", arguments={})
        assert isinstance(action, RoledContent)

    def test_action_response_is_roled_content(self):
        """Test ActionResponse is a RoledContent."""
        response = ActionResponse.create(result="test")
        assert isinstance(response, RoledContent)
