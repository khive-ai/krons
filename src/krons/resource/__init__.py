# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Resources module: iModel, ResourceBackend, hooks, and registry.

Core exports:
- iModel: Unified resource interface with rate limiting and hooks
- ResourceBackend/Endpoint: Backend abstractions for API calls
- HookRegistry/HookEvent/HookPhase: Lifecycle hook system
- ResourceRegistry: O(1) name-based resource lookup

Uses lazy loading for fast import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "Calling": ("krons.resource.backend", "Calling"),
    "NormalizedResponse": ("krons.resource.backend", "NormalizedResponse"),
    "NormalizedResponseModel": ("krons.resource.backend", "NormalizedResponseModel"),
    "ResourceBackend": ("krons.resource.backend", "ResourceBackend"),
    "ResourceConfig": ("krons.resource.backend", "ResourceConfig"),
    "ResourceRegistry": ("krons.resource.registry", "ResourceRegistry"),
    "iModel": ("krons.resource.imodel", "iModel"),
    "Endpoint": ("krons.resource.endpoint", "Endpoint"),
    "EndpointConfig": ("krons.resource.endpoint", "EndpointConfig"),
    "APICalling": ("krons.resource.endpoint", "APICalling"),
    "HookRegistry": ("krons.resource.hook", "HookRegistry"),
    "HookEvent": ("krons.resource.hook", "HookEvent"),
    "HookPhase": ("krons.resource.hook", "HookPhase"),
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

    raise AttributeError(f"module 'krons.resource' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return all available attributes for autocomplete."""
    return list(__all__)


# TYPE_CHECKING block for static analysis
if TYPE_CHECKING:
    from .backend import Calling, NormalizedResponse, NormalizedResponseModel, ResourceBackend, ResourceConfig
    from .endpoint import APICalling, Endpoint, EndpointConfig
    from .hook import HookEvent, HookPhase, HookRegistry
    from .imodel import iModel
    from .registry import ResourceRegistry

__all__ = (
    "APICalling",
    "Calling",
    "Endpoint",
    "EndpointConfig",
    "HookEvent",
    "HookPhase",
    "HookRegistry",
    "NormalizedResponse",
    "NormalizedResponseModel",
    "ResourceBackend",
    "ResourceConfig",
    "ResourceRegistry",
    "iModel",
)
