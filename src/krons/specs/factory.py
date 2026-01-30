# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Content field Specs - structured content, metadata, embeddings."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID, uuid4

from krons.specs.spec import Spec, not_sentinel
from krons.types import UnsetType
from krons.types._sentinel import Unset
from krons.types.base import is_sentinel


def create_datetime_spec(name: str, *, use_default: bool) -> Spec:
    from krons.utils._utils import coerce_created_at, now_utc

    return Spec(
        datetime,
        name=name,
        default_factory=now_utc if use_default else Unset,
        validator=lambda cls, v: coerce_created_at(v),
    )


def create_uuid_spec(name: str, *, use_default: bool) -> Spec:
    from krons.utils._utils import to_uuid

    return Spec(
        UUID,
        name=name,
        default_factory=uuid4 if use_default else Unset,
        validator=lambda cls, v: to_uuid(v) if v is not None else None,
    )


def create_content_spec(
    name: str = "content",
    *,
    content_type: type = Unset,
    use_default: bool = False,
    default_factory=Unset,
) -> Spec:
    content_type = dict if is_sentinel(content_type) else content_type
    if use_default:
        _df = default_factory if not_sentinel(default_factory) else content_type
        return Spec(content_type, name=name, default_factory=_df)
    return Spec(content_type, name=name)


def create_embedding_spec(
    name: str = "embedding",
    *,
    use_default: bool = False,
    dim: int | UnsetType = Unset,
) -> Spec:
    """Create dimensioned embedding Spec

    Args:
        dim: Vector dimension (1536=OpenAI, 768=BERT, 384=MiniLM).
        name: DB column name.

    Returns:
        Spec[Vector[dim]] for DDL generation with correct pgvector type.

    Example:
        create_embedding_spec(1536)  # -> Vector(1536) in DDL
    """

    if is_sentinel(dim):
        if use_default:
            return Spec(list[float], name=name, default_factory=list)
        return Spec(list[float], name=name)

    from krons.specs.adapters.sql_ddl import Vector

    return Spec(Vector[dim], name=name)


def create_change_by_spec(name: str, *, use_uuid: bool = True):
    """Create 'created_by'/'updated_by' Spec with UUID or str type.

    Args:
        name: Field name
        use_uuid: True=UUID type, False=str type

    Returns:
        Spec for 'created_by'/'updated_by' field
    """
    if use_uuid:
        return create_uuid_spec(name, use_default=False)
    return Spec(str, name=name)


def create_enumed_str_spec(name: str, *, default=None) -> Spec:
    """Create a Spec that stores enum values as strings.

    Args:
        name: Field name.
        enum_cls: Enum class (for documentation).
        default: Default enum member or string value.

    Returns:
        Spec[str] with validator that extracts .value from enum members.
    """
    if default:
        return Spec(
            str,
            name=name,
            default=_extract_enum_value(None, default),
            validator=_extract_enum_value,
        )
    return Spec(str, name=name, validator=_extract_enum_value)


def _extract_enum_value(_cls, v, /):
    """Extract .value from enum members."""
    return v.value if hasattr(v, "value") else v
