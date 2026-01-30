# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Core primitives with lazy loading for fast import."""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy import mapping
_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # broadcaster
    "Broadcaster": ("krons.core.broadcaster", "Broadcaster"),
    # element
    "Element": ("krons.core.element", "Element"),
    # event
    "Event": ("krons.core.event", "Event"),
    "EventStatus": ("krons.core.event", "EventStatus"),
    "Execution": ("krons.core.event", "Execution"),
    # eventbus
    "EventBus": ("krons.core.eventbus", "EventBus"),
    "Handler": ("krons.core.eventbus", "Handler"),
    # flow
    "Flow": ("krons.core.flow", "Flow"),
    # graph
    "Edge": ("krons.core.graph", "Edge"),
    "EdgeCondition": ("krons.core.graph", "EdgeCondition"),
    "Graph": ("krons.core.graph", "Graph"),
    # node
    "DEFAULT_NODE_CONFIG": ("krons.core.node", "DEFAULT_NODE_CONFIG"),
    "NODE_REGISTRY": ("krons.core.node", "NODE_REGISTRY"),
    "PERSISTABLE_NODE_REGISTRY": ("krons.core.node", "PERSISTABLE_NODE_REGISTRY"),
    "Node": ("krons.core.node", "Node"),
    "NodeConfig": ("krons.core.node", "NodeConfig"),
    "create_node": ("krons.core.node", "create_node"),
    "generate_ddl": ("krons.core.node", "generate_ddl"),
    # phrase
    "PHRASE_REGISTRY": ("krons.core.phrase", "PHRASE_REGISTRY"),
    "Phrase": ("krons.core.phrase", "Phrase"),
    "PhraseConfig": ("krons.core.phrase", "PhraseConfig"),
    "PhraseError": ("krons.core.phrase", "PhraseError"),
    "RequirementNotMet": ("krons.core.phrase", "RequirementNotMet"),
    "create_phrase": ("krons.core.phrase", "create_phrase"),
    "get_phrase": ("krons.core.phrase", "get_phrase"),
    "list_phrases": ("krons.core.phrase", "list_phrases"),
    # pile
    "Pile": ("krons.core.pile", "Pile"),
    # processor
    "Executor": ("krons.core.processor", "Executor"),
    "Processor": ("krons.core.processor", "Processor"),
    # progression
    "Progression": ("krons.core.progression", "Progression"),
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
        DEFAULT_NODE_CONFIG,
        NODE_REGISTRY,
        PERSISTABLE_NODE_REGISTRY,
        Node,
        NodeConfig,
        create_node,
        generate_ddl,
    )
    from .phrase import (
        PHRASE_REGISTRY,
        Phrase,
        PhraseConfig,
        PhraseError,
        RequirementNotMet,
        create_phrase,
        get_phrase,
        list_phrases,
    )
    from .pile import Pile
    from .processor import Executor, Processor
    from .progression import Progression

__all__ = [
    # constants
    "DEFAULT_NODE_CONFIG",
    "NODE_REGISTRY",
    "PERSISTABLE_NODE_REGISTRY",
    "PHRASE_REGISTRY",
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
    "Phrase",
    "PhraseConfig",
    "PhraseError",
    "Pile",
    "Processor",
    "Progression",
    "RequirementNotMet",
    # functions
    "create_node",
    "create_phrase",
    "generate_ddl",
    "get_phrase",
    "list_phrases",
]
