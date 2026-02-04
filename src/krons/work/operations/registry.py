# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Per-session operation handler registry.

Maps operation names to async handlers. Instantiated per-Session
for isolation, testability, and per-session customization.

Handler signature: async handler(params, ctx: RequestContext) -> result
Operation._invoke() creates the RequestContext from bound session/branch.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

__all__ = ("OperationHandler", "OperationRegistry")

OperationHandler = Callable[..., Awaitable[Any]]
"""Handler signature: async (params, ctx: RequestContext) -> result"""


class OperationRegistry:
    """Map operation names to async handler functions.

    Per-session registry (not global) for isolation and testability.

    Example:
        from krons.agent.operations import generate, structure, operate

        registry = OperationRegistry()
        registry.register("generate", generate)
        registry.register("structure", structure)
        registry.register("operate", operate)

        # Called by Operation._invoke() — users call session.conduct()
        handler = registry.get("generate")
        result = await handler(params, ctx)
    """

    def __init__(self):
        """Initialize empty registry."""
        self._handlers: dict[str, OperationHandler] = {}

    def register(
        self,
        operation_name: str,
        handler: OperationHandler,
        *,
        override: bool = False,
    ) -> None:
        """Register handler for operation name.

        Args:
            operation_name: Lookup key (e.g. "generate", "operate").
            handler: Async (params, ctx) -> result.
            override: Allow replacing existing. Default False.

        Raises:
            ValueError: If name exists and override=False.
        """
        if operation_name in self._handlers and not override:
            raise ValueError(
                f"Operation '{operation_name}' already registered. "
                "Use override=True to replace."
            )
        self._handlers[operation_name] = handler

    def get(self, operation_name: str) -> OperationHandler:
        """Get handler by name. Raises KeyError with available names if not found."""
        if operation_name not in self._handlers:
            raise KeyError(
                f"Operation '{operation_name}' not registered. "
                f"Available: {self.list_names()}"
            )
        return self._handlers[operation_name]

    def has(self, operation_name: str) -> bool:
        """Check if name is registered."""
        return operation_name in self._handlers

    def unregister(self, operation_name: str) -> bool:
        """Remove registration. Returns True if existed."""
        if operation_name in self._handlers:
            del self._handlers[operation_name]
            return True
        return False

    def list_names(self) -> list[str]:
        """Return all registered operation names."""
        return list(self._handlers.keys())

    def __contains__(self, operation_name: str) -> bool:
        """Support 'name in registry' syntax."""
        return operation_name in self._handlers

    def __len__(self) -> int:
        """Count of registered operations."""
        return len(self._handlers)

    def __repr__(self) -> str:
        return f"OperationRegistry(operations={self.list_names()})"
