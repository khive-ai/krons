# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Policy protocols and types.

Defines contracts for policy resolution and evaluation.
Implementations provided by domain libs (e.g., canon-core).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from krons.enforcement.context import RequestContext
from krons.specs.catalog._enforcement import EnforcementLevel
from krons.types.base import DataClass

__all__ = (
    "EnforcementLevel",
    "PolicyEngine",
    "PolicyResolver",
    "ResolvedPolicy",
)


@dataclass(slots=True)
class ResolvedPolicy(DataClass):
    """A policy resolved for evaluation.

    Returned by PolicyResolver.resolve(). Contains policy ID and
    any resolution metadata needed by the engine.
    """

    policy_id: str
    enforcement: str = EnforcementLevel.HARD_MANDATORY.value
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PolicyEngine(Protocol):
    """Abstract policy evaluation engine.

    kron defines the contract. Implementations:
    - canon-core: OPAEngine (Rego/Regorus evaluation)
    - Testing: MockPolicyEngine
    """

    async def evaluate(
        self,
        policy_id: str,
        input_data: dict[str, Any],
        **options: Any,
    ) -> Any:
        """Evaluate a single policy against input."""
        ...

    async def evaluate_batch(
        self,
        policy_ids: Sequence[str],
        input_data: dict[str, Any],
        **options: Any,
    ) -> list[Any]:
        """Evaluate multiple policies."""
        ...


@runtime_checkable
class PolicyResolver(Protocol):
    """Resolves which policies apply to a given context.

    kron defines the contract. Implementations:
    - canon-core: CharteredResolver (charter-based resolution)
    - Testing: MockPolicyResolver, StaticPolicyResolver
    """

    def resolve(self, ctx: RequestContext) -> Sequence[ResolvedPolicy]:
        """Determine applicable policies for context."""
        ...
