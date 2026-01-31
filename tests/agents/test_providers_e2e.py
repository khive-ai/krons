# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for agent providers with real API calls.

These tests require:
- Claude CLI installed (`npm i -g @anthropic-ai/claude-code`)
- Gemini CLI installed (if testing Gemini)

Run with: pytest tests/agents/test_providers_e2e.py -v -s
"""

from __future__ import annotations

import shutil

import pytest

# Skip all tests if providers not available
pytest.importorskip("krons.agents.providers")


def _has_claude_cli() -> bool:
    """Check if Claude CLI is installed."""
    return shutil.which("claude") is not None


def _has_gemini_cli() -> bool:
    """Check if Gemini CLI is installed."""
    return shutil.which("gemini") is not None


class TestClaudeCodeEndpoint:
    """End-to-end tests for Claude Code CLI endpoint."""

    @pytest.fixture
    def endpoint(self):
        """Create Claude Code endpoint."""
        from krons.agents.providers.claude_code import ClaudeCodeEndpoint

        return ClaudeCodeEndpoint()

    @pytest.mark.anyio
    @pytest.mark.skipif(
        not _has_claude_cli(),
        reason="Claude CLI not installed",
    )
    async def test_simple_query(self, endpoint):
        """Test a simple query to Claude Code CLI."""
        response = await endpoint.call(
            request={
                "messages": [{"role": "user", "content": "What is 2 + 2? Reply with just the number."}],
                "max_turns": 1,
                "model": "haiku",  # Use haiku for cost efficiency
            }
        )

        assert response.status == "success"
        assert response.data is not None
        # Should contain "4" somewhere in the response
        assert "4" in str(response.data)

    @pytest.mark.anyio
    @pytest.mark.skipif(
        not _has_claude_cli(),
        reason="Claude CLI not installed",
    )
    async def test_json_response(self, endpoint):
        """Test getting JSON response from Claude Code CLI."""
        response = await endpoint.call(
            request={
                "messages": [
                    {
                        "role": "user",
                        "content": 'Return a JSON object with keys "name" and "value". '
                        'Use name="test" and value=42. Reply with ONLY the JSON, no markdown.',
                    }
                ],
                "max_turns": 1,
                "model": "haiku",
            }
        )

        assert response.status == "success"
        # Should be able to parse as JSON (fuzzy)
        from krons.utils.fuzzy import extract_json

        data = extract_json(response.data)
        assert data is not None
        assert "name" in data or "value" in data

    @pytest.mark.anyio
    @pytest.mark.skipif(
        not _has_claude_cli(),
        reason="Claude CLI not installed",
    )
    async def test_metadata_populated(self, endpoint):
        """Test that response metadata is populated."""
        response = await endpoint.call(
            request={
                "messages": [{"role": "user", "content": "Say hello"}],
                "max_turns": 1,
                "model": "haiku",
            }
        )

        assert response.status == "success"
        assert response.metadata is not None
        # Should have usage info
        assert "usage" in response.metadata or response.metadata.get("usage") is not None


class TestGeminiCodeEndpoint:
    """End-to-end tests for Gemini Code CLI endpoint."""

    @pytest.fixture
    def endpoint(self):
        """Create Gemini Code endpoint."""
        from krons.agents.providers.gemini import GeminiCodeEndpoint

        return GeminiCodeEndpoint()

    @pytest.mark.anyio
    @pytest.mark.skipif(
        not _has_gemini_cli(),
        reason="Gemini CLI not installed",
    )
    async def test_simple_query(self, endpoint):
        """Test a simple query to Gemini Code CLI."""
        response = await endpoint.call(
            request={
                "messages": [{"role": "user", "content": "What is 3 + 3? Reply with just the number."}],
                "max_turns": 1,
            }
        )

        assert response.status == "success"
        assert response.data is not None
        assert "6" in str(response.data)


# Simple standalone test for quick verification
if __name__ == "__main__":
    import asyncio

    async def main():
        if not _has_claude_cli():
            print("Claude CLI not installed, skipping test")
            return

        from krons.agents.providers.claude_code import ClaudeCodeEndpoint

        endpoint = ClaudeCodeEndpoint()
        print("Testing Claude Code endpoint with haiku model...")

        response = await endpoint.call(
            request={
                "messages": [{"role": "user", "content": "What is 2+2? Reply with just the number."}],
                "max_turns": 1,
                "model": "haiku",
            }
        )

        print(f"Status: {response.status}")
        print(f"Data: {response.data}")
        print(f"Metadata: {response.metadata}")

        if response.status == "success":
            print("\nTest PASSED!")
        else:
            print(f"\nTest FAILED: {response.error}")

    asyncio.run(main())
