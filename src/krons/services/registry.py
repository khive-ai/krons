# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from uuid import UUID

from krons.core import Pile
from krons.types import Undefined, UndefinedType, is_sentinel

from .imodel import iModel

__all__ = ("ServiceRegistry",)


class ServiceRegistry:
    """Service registry managing iModel instances with O(1) name-based lookup.

    Provides type-safe storage via Pile[iModel] with name-based indexing.
    Services must have unique names; duplicates raise ValueError unless update=True.

    Example:
        >>> registry = ServiceRegistry()
        >>> registry.register(iModel(backend=my_endpoint))
        >>> model = registry.get("my_service")
        >>> tagged = registry.list_by_tag("api")
    """

    def __init__(self):
        """Initialize empty registry with Pile storage and name index."""
        from .imodel import iModel

        self._pile: Pile[iModel] = Pile(item_type=iModel)
        self._name_index: dict[str, UUID] = {}

    def register(self, model: iModel, update: bool = False) -> UUID:
        """Register iModel instance by name.

        Args:
            model: iModel instance to register.
            update: If True, replaces existing service with same name.

        Returns:
            UUID of registered service.

        Raises:
            ValueError: If service name exists and update=False.
        """
        if model.name in self._name_index:
            if not update:
                raise ValueError(f"Service '{model.name}' already registered")
            # Update: remove old, add new
            old_uid = self._name_index[model.name]
            self._pile.remove(old_uid)

        self._pile.add(model)
        self._name_index[model.name] = model.id

        return model.id

    def unregister(self, name: str) -> iModel:
        """Remove and return service by name. Raises KeyError if not found."""
        if name not in self._name_index:
            raise KeyError(f"Service '{name}' not found")

        uid = self._name_index.pop(name)
        return self._pile.remove(uid)

    def get(self, name: str | UUID | iModel, default: Any | UndefinedType = Undefined) -> iModel:
        """Get service by name, UUID, or return iModel passthrough. Raises KeyError if not found."""
        if isinstance(name, UUID):
            return self._pile[name]
        if isinstance(name, iModel):
            return name
        if name not in self._name_index:
            if not is_sentinel(default):
                return default
            raise KeyError(f"Service '{name}' not found")

        uid = self._name_index[name]
        return self._pile[uid]

    def has(self, name: str) -> bool:
        """Check if service exists."""
        return name in self._name_index

    def list_names(self) -> list[str]:
        """List all registered service names."""
        return list(self._name_index.keys())

    def list_by_tag(self, tag: str) -> Pile[iModel]:
        """Filter services by tag, returns Pile of matching iModels."""
        return self._pile[lambda m: tag in m.tags]

    def count(self) -> int:
        """Count registered services."""
        return len(self._pile)

    def clear(self) -> None:
        """Remove all registered services."""
        self._pile.clear()
        self._name_index.clear()

    def __len__(self) -> int:
        """Return number of registered services."""
        return len(self._pile)

    def __contains__(self, name: str) -> bool:
        """Check if service exists (supports `name in registry`)."""
        return name in self._name_index

    def __repr__(self) -> str:
        """String representation."""
        return f"ServiceRegistry(count={len(self)})"
