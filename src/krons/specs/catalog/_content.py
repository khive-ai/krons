# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Content field Specs - identity, timestamps, content, metadata, embeddings."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

from krons.specs.operable import Operable
from krons.specs.spec import Spec
from krons.types._sentinel import Unset, UnsetType
from krons.types.db_types import VectorMeta
from krons.utils import now_utc


class ContentSpecs(BaseModel):
    """Core content fields for elements/nodes."""

    id: UUID = Field(default_factory=uuid4)
    created_at: datetime = Field(default_factory=now_utc)
    content: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    embedding: list[float] | None = None

    @classmethod
    def get_specs(
        cls,
        *,
        content_type: type | UnsetType = Unset,
        dim: int | UnsetType = Unset,
    ) -> list[Spec]:
        """Get list of content Specs.

        Args:
            content_type: Type for content/metadata fields (default: dict).
            dim: Embedding dimension. Unset = list[float], int = Vector[dim].
        """
        operable = Operable.from_structure(cls)
        specs = {spec.name: spec for spec in operable.get_specs()}

        # Override content/metadata type if specified
        if content_type is not Unset:
            specs["content"] = Spec(content_type, name="content").as_nullable()
            specs["metadata"] = Spec(content_type, name="metadata").as_nullable()

        # Override embedding with vector dimension if specified
        if dim is not Unset and isinstance(dim, int):
            specs["embedding"] = Spec(
                list[float],
                name="embedding",
                embedding=VectorMeta(dim),
            ).as_nullable()

        return list(specs.values())
