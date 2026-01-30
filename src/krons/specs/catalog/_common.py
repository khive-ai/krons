# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Common field Specs - reusable patterns across domain entities."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel

from krons.specs.operable import Operable
from krons.specs.spec import Spec


class CommonSpecs(BaseModel):
    """Common fields for domain entities."""

    name: str
    slug: str
    status: str = "active"
    email: str | None = None
    phone: str | None = None
    tenant_id: UUID
    settings: dict[str, Any] | None = None
    data: dict[str, Any] | None = None

    @classmethod
    def get_specs(cls, *, status_default: str = "active") -> list[Spec]:
        """Get list of common Specs.

        Args:
            status_default: Default value for status field.
        """
        operable = Operable.from_structure(cls)
        specs = {spec.name: spec for spec in operable.get_specs()}

        # Override status default if different
        if status_default != "active":
            specs["status"] = Spec(str, name="status", default=status_default)

        return list(specs.values())
