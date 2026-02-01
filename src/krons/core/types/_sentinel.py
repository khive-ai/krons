# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Sentinel types for distinguishing missing vs unset values.

Provides two distinct sentinel states:
    - Undefined: Field/key entirely absent from namespace (never existed)
    - Unset: Key present but value not provided (explicit "no value")

This distinction enables precise handling in serialization, validation,
and API parameter processing where None has semantic meaning.

Example:
    >>> def fetch(timeout: int | UnsetType = Unset) -> Response:
    ...     if is_unset(timeout):
    ...         timeout = DEFAULT_TIMEOUT  # user didn't specify
    ...     # vs timeout=None which could mean "no timeout"

Usage patterns:
    - Field defaults: `field: str = Unset` (user can provide or omit)
    - Dict access: `d.get(key, Undefined)` (distinguish missing from None)
    - Type hints: `MaybeSentinel[T]` for T | Undefined | Unset
"""

from __future__ import annotations

from collections.abc import Callable
from typing import (
    Any,
    ClassVar,
    Final,
    Literal,
    Self,
    TypeAlias,
    TypeGuard,
    TypeVar,
    Union,
)

__all__ = (
    "MaybeSentinel",
    "MaybeUndefined",
    "MaybeUnset",
    "SingletonType",
    "T",
    "Undefined",
    "UndefinedType",
    "Unset",
    "UnsetType",
    "is_sentinel",
    "is_undefined",
    "is_unset",
    "not_sentinel",
)

T = TypeVar("T")


class _SingletonMeta(type):
    """Metaclass ensuring single instance per subclass for identity checks."""

    _cache: ClassVar[dict[type, SingletonType]] = {}

    def __call__(cls, *a, **kw):
        if cls not in cls._cache:
            cls._cache[cls] = super().__call__(*a, **kw)
        return cls._cache[cls]


class SingletonType(metaclass=_SingletonMeta):
    """Base for singleton sentinels.

    Guarantees:
        - Single instance per subclass (safe `is` checks)
        - Falsy evaluation (bool returns False)
        - Identity preserved across copy/deepcopy/pickle

    Subclasses must implement __bool__ and __repr__.
    """

    __slots__: tuple[str, ...] = ()

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """Return self; singleton identity survives deepcopy."""
        return self

    def __copy__(self) -> Self:
        """Return self; singleton identity survives copy."""
        return self

    def __bool__(self) -> bool:
        """Subclasses must return False."""
        raise NotImplementedError

    def __repr__(self) -> str:
        """Subclasses must return sentinel name."""
        raise NotImplementedError


class UndefinedType(SingletonType):
    """Sentinel for field/key entirely absent from namespace.

    Semantics: The key was never present; the field never existed.

    Use cases:
        - dict.get(key, Undefined) to distinguish missing from None
        - Dataclass fields that may not exist in source data
        - API responses with optional fields

    Example:
        >>> config = {"timeout": None}
        >>> config.get("timeout", Undefined)  # None (explicitly set)
        >>> config.get("retries", Undefined)  # Undefined (missing)
    """

    __slots__ = ()

    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> Literal["Undefined"]:
        return "Undefined"

    def __str__(self) -> Literal["Undefined"]:
        return "Undefined"

    def __reduce__(self) -> tuple[type[UndefinedType], tuple[()]]:
        """Preserve singleton across pickle."""
        return (UndefinedType, ())

    def __or__(self, other: type) -> Any:
        """Enable union syntax: str | Undefined."""
        other_type = type(other) if isinstance(other, SingletonType) else other
        return Union[type(self), other_type]

    def __ror__(self, other: type) -> Any:
        """Enable reverse union: Undefined | str."""
        other_type = type(other) if isinstance(other, SingletonType) else other
        return Union[other_type, type(self)]


class UnsetType(SingletonType):
    """Sentinel for key present but value explicitly not provided.

    Semantics: The slot exists but user chose not to fill it.

    Use cases:
        - Function params: distinguish "not passed" from "passed None"
        - Form fields: distinguish "left blank" from "cleared"
        - Config: distinguish "use default" from "disabled"

    Example:
        >>> def request(timeout: int | None | UnsetType = Unset):
        ...     if is_unset(timeout):
        ...         timeout = 30  # default
        ...     elif timeout is None:
        ...         timeout = float('inf')  # no timeout
        ...     return make_request(timeout=timeout)
    """

    __slots__ = ()

    def __bool__(self) -> Literal[False]:
        return False

    def __repr__(self) -> Literal["Unset"]:
        return "Unset"

    def __str__(self) -> Literal["Unset"]:
        return "Unset"

    def __reduce__(self) -> tuple[type[UnsetType], tuple[()]]:
        """Preserve singleton across pickle."""
        return (UnsetType, ())

    def __or__(self, other: type) -> Any:
        """Enable union syntax: str | Unset."""
        other_type = type(other) if isinstance(other, SingletonType) else other
        return Union[type(self), other_type]

    def __ror__(self, other: type) -> Any:
        """Enable reverse union: Unset | str."""
        other_type = type(other) if isinstance(other, SingletonType) else other
        return Union[other_type, type(self)]


Undefined: Final[UndefinedType] = UndefinedType()
"""Singleton: key/field entirely absent from namespace."""

Unset: Final[UnsetType] = UnsetType()
"""Singleton: key present but value not provided."""

MaybeUndefined: TypeAlias = T | UndefinedType
"""Type alias: T or Undefined (for optional fields that may not exist)."""

MaybeUnset: TypeAlias = T | UnsetType
"""Type alias: T or Unset (for params with explicit 'not provided' state)."""

MaybeSentinel: TypeAlias = T | UndefinedType | UnsetType
"""Type alias: T or either sentinel (full optionality)."""

_EMPTY_TUPLE: tuple[Any, ...] = (tuple(), set(), frozenset(), dict(), list(), "")

AdditionalSentinels = Literal["none", "empty", "pydantic", "dataclass"]


def _is_builtin_sentinel(value: Any) -> bool:
    return isinstance(value, (UndefinedType, UnsetType))


def _is_pydantic_sentinel(value: Any) -> bool:
    from pydantic_core import PydanticUndefinedType

    return isinstance(value, PydanticUndefinedType)


def _is_none(value: Any) -> bool:
    return value is None


def _is_empty(value: Any) -> bool:
    return value in _EMPTY_TUPLE


def _is_dataclass_missing(value: Any) -> bool:
    from dataclasses import MISSING

    return value is MISSING


SENTINEL_HANDLERS: dict[str, Callable[[Any], bool]] = {
    "none": _is_none,
    "empty": _is_empty,
    "pydantic": _is_pydantic_sentinel,
    "dataclass": _is_dataclass_missing,
}

HANDLE_SEQUENCE: tuple[str, ...] = ("none", "empty", "pydantic", "dataclass")


def is_undefined(value: Any) -> bool:
    """Check if value is Undefined sentinel.

    Args:
        value: Any value to check.

    Returns:
        True if value is Undefined type instance.

    Note:
        Uses isinstance (not `is`) for robustness across module reloads.
    """
    return isinstance(value, UndefinedType)


def is_unset(value: Any) -> bool:
    """Check if value is Unset sentinel.

    Args:
        value: Any value to check.

    Returns:
        True if value is Unset type instance.

    Note:
        Uses isinstance (not `is`) for robustness across module reloads.
    """
    return isinstance(value, UnsetType)


def is_sentinel(
    value: Any,
    additions: set[AdditionalSentinels] = frozenset(),
) -> bool:
    """Check if value is any sentinel type.

    Always checks Undefined and Unset. Additional sentinel categories
    can be opted into via the additions set.

    Args:
        value: Any value to check.
        additions: Extra categories to treat as sentinel:
            "none" - treat None as sentinel
            "empty" - treat empty containers/strings as sentinel
            "pydantic" - treat PydanticUndefined as sentinel

    Returns:
        True if value matches sentinel criteria.
    """
    if _is_builtin_sentinel(value):
        return True
    for key in HANDLE_SEQUENCE:
        if key in additions and SENTINEL_HANDLERS[key](value):
            return True
    return False


def not_sentinel(
    value: T | UndefinedType | UnsetType,
    additions: set[AdditionalSentinels] = frozenset(),
) -> TypeGuard[T]:
    """Type-narrowing guard: value is NOT a sentinel.

    Args:
        value: Value to check, typically MaybeSentinel[T].
        additions: Extra categories to treat as sentinel (see is_sentinel).

    Returns:
        TypeGuard narrowing MaybeSentinel[T] to T.
    """
    return not is_sentinel(value, additions)
