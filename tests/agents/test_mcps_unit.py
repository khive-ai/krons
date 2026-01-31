# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for MCP wrapper with mocked dependencies."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from krons.agents.mcps import (
    DEFAULT_ALLOWED_COMMANDS,
    CommandNotAllowedError,
    MCPConnectionPool,
)
from krons.agents.mcps.wrapper import MCP_ENV_ALLOWLIST, filter_mcp_environment


class TestFilterMCPEnvironment:
    """Test filter_mcp_environment function."""

    def test_filter_allows_system_essentials(self):
        """Test that system essentials are allowed."""
        env = {
            "PATH": "/usr/bin",
            "HOME": "/home/user",
            "USER": "testuser",
            "SHELL": "/bin/bash",
            "TERM": "xterm-256color",
        }

        filtered = filter_mcp_environment(env)

        assert "PATH" in filtered
        assert "HOME" in filtered
        assert "USER" in filtered
        assert "SHELL" in filtered
        assert "TERM" in filtered

    def test_filter_allows_python_env(self):
        """Test that Python environment variables are allowed."""
        env = {
            "PYTHONPATH": "/usr/lib/python",
            "VIRTUAL_ENV": "/home/user/.venv",
            "CONDA_PREFIX": "/opt/conda",
        }

        filtered = filter_mcp_environment(env)

        assert "PYTHONPATH" in filtered
        assert "VIRTUAL_ENV" in filtered
        assert "CONDA_PREFIX" in filtered

    def test_filter_blocks_api_keys(self):
        """Test that API keys are blocked."""
        env = {
            "PATH": "/usr/bin",
            "OPENAI_API_KEY": "sk-secret-key",
            "ANTHROPIC_API_KEY": "ant-secret",
            "AWS_SECRET_ACCESS_KEY": "aws-secret",
        }

        filtered = filter_mcp_environment(env)

        assert "PATH" in filtered
        assert "OPENAI_API_KEY" not in filtered
        assert "ANTHROPIC_API_KEY" not in filtered
        assert "AWS_SECRET_ACCESS_KEY" not in filtered

    def test_filter_allows_lc_patterns(self):
        """Test that LC_* locale variables are allowed."""
        env = {
            "LC_ALL": "en_US.UTF-8",
            "LC_CTYPE": "en_US.UTF-8",
            "LC_MESSAGES": "en_US.UTF-8",
        }

        filtered = filter_mcp_environment(env)

        assert "LC_ALL" in filtered
        assert "LC_CTYPE" in filtered
        assert "LC_MESSAGES" in filtered

    def test_filter_allows_mcp_patterns(self):
        """Test that MCP_* and FASTMCP_* variables are allowed."""
        env = {
            "MCP_DEBUG": "true",
            "MCP_QUIET": "false",
            "FASTMCP_QUIET": "true",
        }

        filtered = filter_mcp_environment(env)

        assert "MCP_DEBUG" in filtered
        assert "MCP_QUIET" in filtered
        assert "FASTMCP_QUIET" in filtered

    def test_filter_with_custom_allowlist(self):
        """Test filter with custom allowlist."""
        env = {
            "PATH": "/usr/bin",
            "MY_CUSTOM_VAR": "value",
            "BLOCKED_VAR": "blocked",
        }

        filtered = filter_mcp_environment(
            env,
            allowlist={"PATH", "MY_CUSTOM_VAR"},
            patterns=(),  # No patterns
        )

        assert "PATH" in filtered
        assert "MY_CUSTOM_VAR" in filtered
        assert "BLOCKED_VAR" not in filtered

    def test_filter_uses_os_environ_by_default(self):
        """Test that filter uses os.environ when env=None."""
        with patch.dict(os.environ, {"TEST_VAR": "test_value"}, clear=False):
            # Since TEST_VAR is not in allowlist, it should be filtered
            filtered = filter_mcp_environment(env=None)
            assert "TEST_VAR" not in filtered


class TestDefaultAllowedCommands:
    """Test DEFAULT_ALLOWED_COMMANDS constant."""

    def test_contains_common_interpreters(self):
        """Test that common interpreters are in allowlist."""
        assert "python" in DEFAULT_ALLOWED_COMMANDS
        assert "python3" in DEFAULT_ALLOWED_COMMANDS
        assert "node" in DEFAULT_ALLOWED_COMMANDS
        assert "npx" in DEFAULT_ALLOWED_COMMANDS

    def test_is_frozenset(self):
        """Test that DEFAULT_ALLOWED_COMMANDS is immutable."""
        assert isinstance(DEFAULT_ALLOWED_COMMANDS, frozenset)


class TestCommandNotAllowedError:
    """Test CommandNotAllowedError exception."""

    def test_inherits_from_krons_error(self):
        """Test that CommandNotAllowedError inherits from KronsError."""
        from krons.errors import KronsError

        assert issubclass(CommandNotAllowedError, KronsError)

    def test_has_correct_defaults(self):
        """Test default message and retryable flag."""
        error = CommandNotAllowedError()

        assert error.default_message == "Command not allowed"
        assert error.default_retryable is False

    def test_custom_message(self):
        """Test custom error message."""
        error = CommandNotAllowedError("Command 'rm' is not allowed")

        assert "rm" in str(error)

    def test_serializable(self):
        """Test error serialization."""
        error = CommandNotAllowedError(
            "Blocked command",
            details={"command": "rm -rf"},
        )

        data = error.to_dict()

        assert data["error"] == "CommandNotAllowedError"
        assert data["message"] == "Blocked command"
        assert data["retryable"] is False
        assert data["details"]["command"] == "rm -rf"


class TestMCPConnectionPool:
    """Test MCPConnectionPool class structure."""

    def test_class_attributes_exist(self):
        """Test that class-level caches exist."""
        assert hasattr(MCPConnectionPool, "_clients")
        assert hasattr(MCPConnectionPool, "_configs")

    def test_clients_cache_is_dict(self):
        """Test that _clients cache is a dict."""
        assert isinstance(MCPConnectionPool._clients, dict)

    def test_configs_cache_is_dict(self):
        """Test that _configs cache is a dict."""
        assert isinstance(MCPConnectionPool._configs, dict)


class TestMCPEnvAllowlist:
    """Test MCP_ENV_ALLOWLIST constant."""

    def test_is_frozenset(self):
        """Test that MCP_ENV_ALLOWLIST is immutable."""
        assert isinstance(MCP_ENV_ALLOWLIST, frozenset)

    def test_contains_essentials(self):
        """Test that essential variables are included."""
        assert "PATH" in MCP_ENV_ALLOWLIST
        assert "HOME" in MCP_ENV_ALLOWLIST
        assert "USER" in MCP_ENV_ALLOWLIST

    def test_contains_python_vars(self):
        """Test that Python-related variables are included."""
        assert "PYTHONPATH" in MCP_ENV_ALLOWLIST
        assert "VIRTUAL_ENV" in MCP_ENV_ALLOWLIST

    def test_contains_node_vars(self):
        """Test that Node.js-related variables are included."""
        assert "NODE_PATH" in MCP_ENV_ALLOWLIST
        assert "NODE_ENV" in MCP_ENV_ALLOWLIST
