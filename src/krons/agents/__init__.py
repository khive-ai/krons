# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Agents module - AI provider integrations, message handling, and operations.

Submodules:
    providers: API endpoint implementations (OpenAI, Anthropic, Gemini, Claude Code)
    message: Message content types and preparation
    mcps: MCP (Model Context Protocol) connection management
    operations: Agent operation primitives (generate, parse, act, react)
    third_party: Provider-specific request/response models
    tool: Callable function backends
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # tool.py
    "Tool": ("krons.agents.tool", "Tool"),
    "ToolCalling": ("krons.agents.tool", "ToolCalling"),
    "ToolConfig": ("krons.agents.tool", "ToolConfig"),
    "tool": ("krons.agents.tool", "tool"),
    # providers
    "AnthropicMessagesEndpoint": (
        "krons.agents.providers",
        "AnthropicMessagesEndpoint",
    ),
    "GeminiCodeEndpoint": ("krons.agents.providers", "GeminiCodeEndpoint"),
    "OAIChatEndpoint": ("krons.agents.providers", "OAIChatEndpoint"),
    "create_anthropic_config": ("krons.agents.providers", "create_anthropic_config"),
    "create_gemini_code_config": (
        "krons.agents.providers",
        "create_gemini_code_config",
    ),
    "create_oai_chat": ("krons.agents.providers", "create_oai_chat"),
    "match_endpoint": ("krons.agents.providers", "match_endpoint"),
    # message
    "ActionRequest": ("krons.agents.message", "ActionRequest"),
    "ActionResponse": ("krons.agents.message", "ActionResponse"),
    "Assistant": ("krons.agents.message", "Assistant"),
    "Instruction": ("krons.agents.message", "Instruction"),
    "MessageContent": ("krons.agents.message", "MessageContent"),
    "MessageRole": ("krons.agents.message", "MessageRole"),
    "System": ("krons.agents.message", "System"),
    "prepare_messages_for_chat": ("krons.agents.message", "prepare_messages_for_chat"),
    # mcps
    "CommandNotAllowedError": ("krons.agents.mcps", "CommandNotAllowedError"),
    "DEFAULT_ALLOWED_COMMANDS": ("krons.agents.mcps", "DEFAULT_ALLOWED_COMMANDS"),
    "MCPConnectionPool": ("krons.agents.mcps", "MCPConnectionPool"),
    "create_mcp_callable": ("krons.agents.mcps", "create_mcp_callable"),
    "load_mcp_config": ("krons.agents.mcps", "load_mcp_config"),
    "load_mcp_tools": ("krons.agents.mcps", "load_mcp_tools"),
}

_LOADED: dict[str, object] = {}


def __getattr__(name: str) -> object:
    """Lazy import attributes on first access."""
    if name in _LOADED:
        return _LOADED[name]

    if name in _LAZY_IMPORTS:
        from importlib import import_module

        module_name, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        _LOADED[name] = value
        return value

    raise AttributeError(f"module 'krons.agents' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return all available attributes for autocomplete."""
    return list(__all__)


# TYPE_CHECKING block for static analysis
if TYPE_CHECKING:
    from krons.agents.mcps import (
        DEFAULT_ALLOWED_COMMANDS,
        CommandNotAllowedError,
        MCPConnectionPool,
        create_mcp_callable,
        load_mcp_config,
        load_mcp_tools,
    )
    from krons.agents.message import (
        ActionRequest,
        ActionResponse,
        Assistant,
        Instruction,
        MessageContent,
        MessageRole,
        System,
        prepare_messages_for_chat,
    )
    from krons.agents.providers import (
        AnthropicMessagesEndpoint,
        GeminiCodeEndpoint,
        OAIChatEndpoint,
        create_anthropic_config,
        create_gemini_code_config,
        create_oai_chat,
        match_endpoint,
    )
    from krons.agents.tool import Tool, ToolCalling, ToolConfig, tool

__all__ = [
    # tool
    "Tool",
    "ToolCalling",
    "ToolConfig",
    "tool",
    # providers
    "AnthropicMessagesEndpoint",
    "GeminiCodeEndpoint",
    "OAIChatEndpoint",
    "create_anthropic_config",
    "create_gemini_code_config",
    "create_oai_chat",
    "match_endpoint",
    # message
    "ActionRequest",
    "ActionResponse",
    "Assistant",
    "Instruction",
    "MessageContent",
    "MessageRole",
    "System",
    "prepare_messages_for_chat",
    # mcps
    "CommandNotAllowedError",
    "DEFAULT_ALLOWED_COMMANDS",
    "MCPConnectionPool",
    "create_mcp_callable",
    "load_mcp_config",
    "load_mcp_tools",
]
