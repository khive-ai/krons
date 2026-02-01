# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""krons - Spec-Based Composable Framework.

Top-level re-exports for convenient imports:
- krons.types -> krons.core.types
- krons.specs -> krons.core.specs
- krons.session -> krons.session (top-level module)
- krons.operations -> krons.work.operations
- krons.agents -> krons.agents
- krons.resources -> krons.resources
- krons.work -> krons.work
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Lazy module re-exports via __getattr__
_MODULE_ALIASES: dict[str, str] = {
    "types": "krons.core.types",
    "specs": "krons.core.specs",
    "session": "krons.session",
    "operations": "krons.work.operations",
    "agents": "krons.agents",
    "resources": "krons.resources",
    "work": "krons.work",
}

_LOADED_MODULES: dict[str, object] = {}


def __getattr__(name: str) -> object:
    """Lazy load aliased modules."""
    if name in _LOADED_MODULES:
        return _LOADED_MODULES[name]

    if name in _MODULE_ALIASES:
        from importlib import import_module

        module = import_module(_MODULE_ALIASES[name])
        _LOADED_MODULES[name] = module
        return module

    raise AttributeError(f"module 'krons' has no attribute {name!r}")


def __dir__() -> list[str]:
    """List available attributes."""
    return list(_MODULE_ALIASES.keys())


if TYPE_CHECKING:
    from krons import agents as agents
    from krons import resources as resources
    from krons import session as session
    from krons import work as work
    from krons.core import specs as specs
    from krons.core import types as types
    from krons.work import operations as operations
