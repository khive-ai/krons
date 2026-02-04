# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Core primitives with lazy loading for fast import."""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping - all modules are in krons.core.base.*
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # broadcaster
    "Broadcaster": ("krons.core.base.broadcaster", "Broadcaster"),
    # element
    "Element": ("krons.core.base.element", "Element"),
    # event
    "Event": ("krons.core.base.event", "Event"),
    "EventStatus": ("krons.core.base.event", "EventStatus"),
    "Execution": ("krons.core.base.event", "Execution"),
    # eventbus
    "EventBus": ("krons.core.base.eventbus", "EventBus"),
    "Handler": ("krons.core.base.eventbus", "Handler"),
    # flow
    "Flow": ("krons.core.base.flow", "Flow"),
    # graph
    "Edge": ("krons.core.base.graph", "Edge"),
    "EdgeCondition": ("krons.core.base.graph", "EdgeCondition"),
    "Graph": ("krons.core.base.graph", "Graph"),
    # node
    "NODE_REGISTRY": ("krons.core.base.node", "NODE_REGISTRY"),
    "PERSISTABLE_NODE_REGISTRY": ("krons.core.base.node", "PERSISTABLE_NODE_REGISTRY"),
    "Node": ("krons.core.base.node", "Node"),
    "NodeConfig": ("krons.core.base.node", "NodeConfig"),
    "create_node": ("krons.core.base.node", "create_node"),
    "generate_ddl": ("krons.core.base.node", "generate_ddl"),
    "generate_all_ddl": ("krons.core.base.node", "generate_all_ddl"),
    "get_fk_dependencies": ("krons.core.base.node", "get_fk_dependencies"),
    # pile
    "Pile": ("krons.core.base.pile", "Pile"),
    # processor
    "Executor": ("krons.core.base.processor", "Executor"),
    "Processor": ("krons.core.base.processor", "Processor"),
    # progression
    "Progression": ("krons.core.base.progression", "Progression"),
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
    from .broadcaster import Broadcaster
    from .element import Element
    from .event import Event, EventStatus, Execution
    from .eventbus import EventBus, Handler
    from .flow import Flow
    from .graph import Edge, EdgeCondition, Graph
    from .node import (
        NODE_REGISTRY,
        PERSISTABLE_NODE_REGISTRY,
        Node,
        NodeConfig,
        create_node,
        generate_all_ddl,
        generate_ddl,
        get_fk_dependencies,
    )
    from .pile import Pile
    from .processor import Executor, Processor
    from .progression import Progression

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
