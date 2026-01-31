# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Core module - Foundation primitives and submodules.

Re-exports base classes from core/base/ and exposes submodules:
- types: Sentinels, base types, DB types
- specs: Spec definitions, Operable, adapters

Note: Session (Message, Branch, Session, Exchange) is now at krons.session
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping - delegates to core.base
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Registries
    "NODE_REGISTRY": ("krons.core.base", "NODE_REGISTRY"),
    "PERSISTABLE_NODE_REGISTRY": ("krons.core.base", "PERSISTABLE_NODE_REGISTRY"),
    # Classes
    "Broadcaster": ("krons.core.base", "Broadcaster"),
    "Edge": ("krons.core.base", "Edge"),
    "EdgeCondition": ("krons.core.base", "EdgeCondition"),
    "Element": ("krons.core.base", "Element"),
    "Event": ("krons.core.base", "Event"),
    "EventBus": ("krons.core.base", "EventBus"),
    "EventStatus": ("krons.core.base", "EventStatus"),
    "Execution": ("krons.core.base", "Execution"),
    "Executor": ("krons.core.base", "Executor"),
    "Flow": ("krons.core.base", "Flow"),
    "Graph": ("krons.core.base", "Graph"),
    "Handler": ("krons.core.base", "Handler"),
    "Node": ("krons.core.base", "Node"),
    "NodeConfig": ("krons.core.base", "NodeConfig"),
    "Pile": ("krons.core.base", "Pile"),
    "Processor": ("krons.core.base", "Processor"),
    "Progression": ("krons.core.base", "Progression"),
    # Functions
    "create_node": ("krons.core.base", "create_node"),
    "generate_all_ddl": ("krons.core.base", "generate_all_ddl"),
    "generate_ddl": ("krons.core.base", "generate_ddl"),
    "get_fk_dependencies": ("krons.core.base", "get_fk_dependencies"),
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

    raise AttributeError(f"module 'krons.core' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return all available attributes for autocomplete."""
    return list(__all__)


# TYPE_CHECKING block for static analysis
if TYPE_CHECKING:
    from krons.core.base import (
        NODE_REGISTRY,
        PERSISTABLE_NODE_REGISTRY,
        Broadcaster,
        Edge,
        EdgeCondition,
        Element,
        Event,
        EventBus,
        EventStatus,
        Execution,
        Executor,
        Flow,
        Graph,
        Handler,
        Node,
        NodeConfig,
        Pile,
        Processor,
        Progression,
        create_node,
        generate_all_ddl,
        generate_ddl,
        get_fk_dependencies,
    )

__all__ = [
    # constants/registries
    "NODE_REGISTRY",
    "PERSISTABLE_NODE_REGISTRY",
    # classes
    "Broadcaster",
    "Edge",
    "EdgeCondition",
    "Element",
    "Event",
    "EventBus",
    "EventStatus",
    "Execution",
    "Executor",
    "Flow",
    "Graph",
    "Handler",
    "Node",
    "NodeConfig",
    "Pile",
    "Processor",
    "Progression",
    # functions
    "create_node",
    "generate_all_ddl",
    "generate_ddl",
    "get_fk_dependencies",
]
