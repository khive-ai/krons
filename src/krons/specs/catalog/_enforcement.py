# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Enforcement levels and specs for policy evaluation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from krons.specs.operable import Operable
from krons.specs.spec import Spec
from krons.types.base import Enum
from krons.utils import now_utc

__all__ = (
    "EnforcementLevel",
    "EnforcementSpecs",
)


class EnforcementLevel(Enum):
    """How strictly to enforce policy violations.

    HARD_MANDATORY: Blocks action, no override possible
    SOFT_MANDATORY: Blocks action, but can be overridden with justification
    ADVISORY: Warns but allows action to proceed
    """

    HARD_MANDATORY = "hard_mandatory"
    SOFT_MANDATORY = "soft_mandatory"
    ADVISORY = "advisory"

    @classmethod
    def is_blocking(cls, result: Any) -> bool:
        """Check if policy result blocks the action."""
        enforcement = getattr(result, "enforcement", "")
        return enforcement in (
            cls.HARD_MANDATORY.value,
            cls.SOFT_MANDATORY.value,
        )

    @classmethod
    def is_advisory(cls, result: Any) -> bool:
        """Check if policy result is advisory (not blocking)."""
        return getattr(result, "enforcement", "") == cls.ADVISORY.value


class EnforcementSpecs(BaseModel):
    """Fields for policy enforcement results."""

    enforcement: str = EnforcementLevel.HARD_MANDATORY.value
    policy_id: str
    violation_code: str | None = None
    evaluated_at: datetime = Field(default_factory=now_utc)
    evaluation_ms: float = Field(default=0.0, ge=0.0)

    @field_validator("enforcement", mode="before")
    @classmethod
    def _extract_enum_value(cls, v):
        """Extract .value from enum members."""
        return v.value if hasattr(v, "value") else v

    @classmethod
    def get_specs(cls) -> list[Spec]:
        """Get list of enforcement Specs."""
        operable = Operable.from_structure(cls)
        return list(operable.get_specs())
