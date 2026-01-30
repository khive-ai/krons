# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Base types for kron: dataclasses with sentinel-aware serialization.

Provides:
    - Enum: String-backed enum for JSON-friendly enumerations
    - ModelConfig: Configuration for sentinel handling and validation
    - Params: Immutable parameter container (frozen dataclass, custom __init__)
    - DataClass: Mutable dataclass with validation hooks
    - Meta: Hashable key-value metadata container

Key concepts:
    - Sentinel handling: Undefined/Unset fields excluded from to_dict()
    - Configurable validation: strict mode, prefill behavior
    - Hash-based equality: enables caching and deduplication
"""

from __future__ import annotations

from collections.abc import MutableMapping, MutableSequence, MutableSet, Sequence
from dataclasses import MISSING as DATACLASS_MISSING
from dataclasses import dataclass, field, fields
from enum import Enum as _Enum
from enum import StrEnum
from typing import Any, ClassVar, Literal, Self, TypedDict

from pydantic import BaseModel as _PydanticBaseModel
from typing_extensions import override

from krons.protocols import Allowable, Hashable, Serializable, implements
from krons.utils._hash import hash_obj

from ._sentinel import Undefined, Unset, is_sentinel, is_undefined

__all__ = (
    "DataClass",
    "Enum",
    "HashableModel",
    "KeysDict",
    "KeysLike",
    "Meta",
    "ModelConfig",
    "Params",
)


@implements(Allowable)
class Enum(StrEnum):
    """String-backed enum with Allowable protocol.

    Members serialize directly to their string values. Python 3.11+.
    """

    @classmethod
    def allowed(cls) -> tuple[str, ...]:
        """Return tuple of all valid member values."""
        return tuple(e.value for e in cls)


class KeysDict(TypedDict, total=False):
    """TypedDict for flexible key-type mappings."""

    key: Any


@dataclass(slots=True, frozen=True)
class ModelConfig:
    """Configuration for Params/DataClass behavior.

    Attributes:
        sentinel_additions: Additional sentinel categories beyond Undefined/Unset.
            Valid values: "none", "empty", "pydantic", "dataclass".
        strict: Require all fields have values (raise if sentinel).
        prefill_unset: Convert Undefined fields to Unset on validation.
        use_enum_values: Serialize enums as their values (not names).
    """

    sentinel_additions: frozenset[str] = field(default_factory=frozenset)
    strict: bool = False
    prefill_unset: bool = True
    use_enum_values: bool = False

    def is_sentinel(self, value: Any) -> bool:
        """Check if value is sentinel per this config's additions."""
        return is_sentinel(value, self.sentinel_additions)

    def is_sentinel_field(self, allowable: Allowable, field_name: str, /) -> bool:
        """Check if a field holds a sentinel value in the allowable namespace."""
        if field_name not in allowable.allowed():
            raise ValueError(f"Invalid field name: {field_name}")
        value = getattr(allowable, field_name, Undefined)
        return self.is_sentinel(value)


class _SentinelMixin:
    """Shared sentinel-aware serialization logic for Params and DataClass.

    Provides: allowed(), _is_sentinel(), _normalize_value(), _validate(),
    to_dict(), with_updates(), __hash__().

    Subclasses must define:
        _config: ClassVar[ModelConfig]
        _allowed_keys: ClassVar[set[str]]
    """

    __slots__ = ()

    @classmethod
    def allowed(cls) -> set[str]:
        """Return set of valid field names (excludes private/ClassVar)."""
        if cls._allowed_keys:
            return cls._allowed_keys
        cls._allowed_keys = set(f.name for f in fields(cls) if not f.name.startswith("_"))
        return cls._allowed_keys

    @classmethod
    def _is_sentinel(cls, value: Any) -> bool:
        """Check if value is sentinel per _config settings."""
        return is_sentinel(value, cls._config.sentinel_additions)

    @classmethod
    def _normalize_value(cls, value: Any) -> Any:
        """Normalize value for serialization (enum to value if configured)."""
        if cls._config.use_enum_values and isinstance(value, _Enum):
            return value.value
        return value

    def _validate(self) -> None:
        """Validate fields per _config. Raises ExceptionGroup if strict violations."""
        missing: list[Exception] = []
        for k in self.allowed():
            if self._config.strict and self._is_sentinel(getattr(self, k, Unset)):
                missing.append(ValueError(f"Missing required parameter: {k}"))
            if self._config.prefill_unset and is_undefined(getattr(self, k, Undefined)):
                object.__setattr__(self, k, Unset)
        if missing:
            raise ExceptionGroup("Missing required parameters", missing)

    def to_dict(
        self,
        mode: Literal["python", "json"] = "python",
        exclude: set[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Serialize to dict, excluding sentinel values."""
        data = {}
        exclude = exclude or set()
        for k in self.allowed():
            if k not in exclude:
                v = getattr(self, k, Undefined)
                if not self._is_sentinel(v):
                    data[k] = self._normalize_value(v)
        if mode == "json":
            from krons.utils._json_dump import json_dump

            return json_dump(data, decode=True, as_loaded=True, **kwargs)

        return data

    def with_updates(
        self, copy_containers: Literal["shallow", "deep"] | None = None, **kwargs: Any
    ) -> Self:
        """Return new instance with updated fields.

        Args:
            copy_containers: "shallow", "deep", or None (share references).
            **kwargs: Field values to update.
        """
        dict_ = self.to_dict()

        def _out(d: dict):
            d.update(kwargs)
            return type(self)(**d)

        if copy_containers is None:
            return _out(dict_)

        match copy_containers:
            case "shallow":
                for k, v in dict_.items():
                    if k not in kwargs and isinstance(
                        v, (MutableSequence, MutableMapping, MutableSet)
                    ):
                        dict_[k] = v.copy() if hasattr(v, "copy") else list(v)
                return _out(dict_)

            case "deep":
                import copy

                for k, v in dict_.items():
                    if k not in kwargs and isinstance(
                        v, (MutableSequence, MutableMapping, MutableSet)
                    ):
                        dict_[k] = copy.deepcopy(v)
                return _out(dict_)

        raise ValueError(
            f"Invalid copy_containers: {copy_containers!r}. Must be 'shallow', 'deep', or None."
        )

    def is_sentinel_field(self, field_name: str) -> bool:
        """Check if field holds a sentinel value.

        Raises:
            ValueError: If field_name not in allowed().
        """
        if field_name not in self.allowed():
            raise ValueError(f"Invalid field name: {field_name}")
        value = getattr(self, field_name, Undefined)
        return self._is_sentinel(value)

    def __hash__(self) -> int:
        """Hash based on serialized dict contents."""
        return hash_obj(self)


@implements(Serializable, Allowable, Hashable, allow_inherited=True)
@dataclass(slots=True, frozen=True, init=False)
class Params(_SentinelMixin):
    """Immutable parameter container with sentinel-aware serialization.

    Frozen dataclass with custom __init__ for sentinel support.
    Subclass and override _config for custom behavior.

    Example:
        >>> @dataclass(slots=True, frozen=True, init=False)
        ... class RequestParams(Params):
        ...     timeout: int = Unset
        ...     retries: int = 3
    """

    _config: ClassVar[ModelConfig] = ModelConfig()
    _allowed_keys: ClassVar[set[str]] = set()

    def __init__(self, **kwargs: Any):
        """Initialize from kwargs with validation.

        Raises:
            ValueError: If kwargs contains invalid field names.
            ExceptionGroup: If strict mode and required fields missing.
        """
        for f in fields(self):
            if f.name.startswith("_"):
                continue
            if f.name not in kwargs:
                if f.default is not DATACLASS_MISSING:
                    object.__setattr__(self, f.name, f.default)
                elif f.default_factory is not DATACLASS_MISSING:
                    object.__setattr__(self, f.name, f.default_factory())

        for k, v in kwargs.items():
            if k in self.allowed():
                object.__setattr__(self, k, v)
            else:
                raise ValueError(f"Invalid parameter: {k}")

        self._validate()

    def default_kw(self) -> Any:
        """Return dict with kwargs/kw fields merged into top level."""
        dict_ = self.to_dict()
        kw_ = {}
        kw_.update(dict_.pop("kwargs", {}))
        kw_.update(dict_.pop("kw", {}))
        dict_.update(kw_)
        return dict_

    def __eq__(self, other: object) -> bool:
        """Equality via hash. Returns NotImplemented for incompatible types."""
        if not isinstance(other, Params):
            return NotImplemented
        return hash(self) == hash(other)


@implements(Serializable, Allowable, Hashable, allow_inherited=True)
@dataclass(slots=True)
class DataClass(_SentinelMixin):
    """Mutable dataclass with sentinel-aware serialization.

    Like Params but mutable (not frozen). Validates on __post_init__.
    Subclass and override _config for custom behavior.
    """

    _config: ClassVar[ModelConfig] = ModelConfig()
    _allowed_keys: ClassVar[set[str]] = set()

    def __post_init__(self):
        """Validate fields after initialization."""
        self._validate()

    def __hash__(self) -> int:
        """Hash based on serialized dict contents."""
        return hash_obj(self)

    def __eq__(self, other: object) -> bool:
        """Equality via hash. Returns NotImplemented for incompatible types."""
        if not isinstance(other, DataClass):
            return NotImplemented
        return hash(self) == hash(other)


KeysLike = Sequence[str] | KeysDict
"""Type alias for key specifications: sequence of names or KeysDict."""


@implements(Hashable)
@dataclass(slots=True, frozen=True)
class Meta:
    """Immutable key-value metadata container.

    Hashable for use in sets/dicts. Special handling for callables
    (hashed by id for identity semantics).

    Attributes:
        key: Metadata key identifier.
        value: Associated value (any type).
    """

    key: str
    value: Any

    @override
    def __hash__(self) -> int:
        """Hash by (key, value). Callables use id(), unhashables use str()."""
        if callable(self.value):
            return hash((self.key, id(self.value)))
        try:
            return hash((self.key, self.value))
        except TypeError:
            return hash((self.key, str(self.value)))

    @override
    def __eq__(self, other: object) -> bool:
        """Equality by key then value. Callables compared by id."""
        if not isinstance(other, Meta):
            return NotImplemented
        if self.key != other.key:
            return False
        if callable(self.value) and callable(other.value):
            return id(self.value) == id(other.value)
        return bool(self.value == other.value)


# --- Pydantic-based hashable model ---


@implements(Hashable)
class HashableModel(_PydanticBaseModel):
    """Pydantic BaseModel with hash and equality support.

    Provides content-based hashing for use in sets/dicts. Same semantics
    as DataClass but for Pydantic models.

    Usage:
        class ServiceConfig(HashableModel):
            provider: str
            name: str
    """

    def __hash__(self) -> int:
        """Hash based on model's dict representation."""
        return hash_obj(self.model_dump())

    def __eq__(self, other: object) -> bool:
        """Equality via hash comparison."""
        if not isinstance(other, HashableModel):
            return NotImplemented
        return hash(self) == hash(other)
