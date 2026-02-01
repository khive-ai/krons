# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Session and Branch: central orchestration for messages, services, and operations.

Session owns branches, messages, services registry, and operations registry.
Branch is a named message progression with capability/resource access control.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from pydantic import Field, model_validator

from krons.core import Element, Flow, Progression
from krons.core.base.pile import Pile
from krons.core.types import HashableModel, Unset, UnsetType, not_sentinel
from krons.errors import NotFoundError
from krons.resources import ResourceRegistry, iModel
from krons.resources.backend import Calling
from krons.work.operations.node import Operation
from krons.work.operations.registry import OperationRegistry

from .message import Message

if TYPE_CHECKING:
    from krons.resources.backend import Calling

__all__ = (
    "Branch",
    "Session",
    "SessionConfig",
)


class Branch(Progression):
    """Message progression with capability and resource access control.

    Branch extends Progression with session binding and access control.
    Messages are referenced by UUID in the order list.

    Attributes:
        session_id: Owning session (immutable after creation).
        capabilities: Allowed structured output schema names.
        resources: Allowed service names for access control.
    """

    session_id: UUID = Field(..., frozen=True)
    capabilities: set[str] = Field(default_factory=set, frozen=True)
    resources: set[str] = Field(default_factory=set, frozen=True)

    def __repr__(self) -> str:
        name_str = f" name='{self.name}'" if self.name else ""
        return f"Branch(messages={len(self)}, session={str(self.session_id)[:8]}{name_str})"


class SessionConfig(HashableModel):
    default_branch_name: str | None = None
    shared_capabilities: set[str] = Field(default_factory=set)
    shared_resources: set[str] = Field(default_factory=set)
    default_gen_model: str | None = None
    default_parse_model: str | None = None
    auto_create_default_branch: bool = True


class Session(Element):
    user: str | None = None
    communications: Flow[Message, Branch] = Field(
        default_factory=lambda: Flow(item_type=Message)
    )
    resources: ResourceRegistry = Field(default_factory=ResourceRegistry)
    operations: OperationRegistry = Field(default_factory=OperationRegistry)
    config: SessionConfig = Field(default_factory=SessionConfig)
    default_branch_id: UUID | None = None

    @model_validator(mode="after")
    def _validate_default_branch(self) -> Session:
        """Auto-create default branch if configured and not present."""
        if self.config.auto_create_default_branch and self.default_branch is None:
            default_branch_name = self.config.default_branch_name or "main"
            self.create_branch(
                name=default_branch_name,
                capabilities=self.config.shared_capabilities,
                resources=self.config.shared_resources,
            )
            self.set_default_branch(default_branch_name)
        return self

    @property
    def default_gen_model(self) -> iModel | None:
        if self.config.default_gen_model is None:
            return None
        return self.resources.get(self.config.default_gen_model)

    @property
    def default_parse_model(self) -> iModel | None:
        if self.config.default_parse_model is None:
            return None
        return self.resources.get(self.config.default_parse_model)

    @property
    def messages(self) -> Pile[Message]:
        """All messages in session (Pile[Message])."""
        return self.communications.items

    @property
    def branches(self) -> Pile[Branch]:
        """All branches in session (Pile[Branch])."""
        return self.communications.progressions

    @property
    def default_branch(self) -> Branch | None:
        """Default branch, or None if unset or deleted."""
        if self.default_branch_id is None:
            return None
        return self.communications.get_progression(self.default_branch_id)

    def create_branch(
        self,
        *,
        name: str | None = None,
        capabilities: set[str] | None = None,
        resources: set[str] | None = None,
        messages: Iterable[UUID | Message] | None = None,
    ) -> Branch:
        """Create and register a new branch.

        Args:
            name: Branch name (auto: "branch_N").
            capabilities: Allowed schema names.
            resources: Allowed service names.
            messages: Initial message UUIDs or objects.

        Returns:
            Created Branch added to session.
        """
        if name:
            from .constraints import branch_name_must_be_unique

            branch_name_must_be_unique(self, name)

        order: list[UUID] = []
        if messages:
            order.extend([self._coerce_id(msg) for msg in messages])

        branch = Branch(
            session_id=self.id,
            name=name or f"branch_{len(self.branches)}",
            capabilities=capabilities or set(),
            resources=resources or set(),
            order=order,
        )

        self.communications.add_progression(branch)
        return branch

    def get_branch(
        self, branch: UUID | str | Branch, default: Branch | UnsetType = Unset, /
    ) -> Branch:
        """Get branch by UUID, name, or instance.

        Args:
            branch: Branch identifier.
            default: Return this if not found (else raise).

        Returns:
            Branch instance.

        Raises:
            NotFoundError: If not found and no default.
        """
        if isinstance(branch, Branch) and branch in self.branches:
            return branch
        with contextlib.suppress(KeyError):
            return self.communications.get_progression(branch)
        if not_sentinel(default):
            return default
        raise NotFoundError("Branch not found")

    def set_default_branch(self, branch: Branch | UUID | str) -> None:
        """Set the default branch for operations.

        Args:
            branch: Branch to set as default (must exist).

        Raises:
            NotFoundError: If branch not in session.
        """
        resolved = self.get_branch(branch)
        self.default_branch_id = resolved.id

    def fork(
        self,
        branch: Branch | UUID | str,
        *,
        name: str | None = None,
        capabilities: set[str] | Literal[True] | None = None,
        resources: set[str] | Literal[True] | None = None,
    ) -> Branch:
        """Fork branch for divergent exploration.

        Creates new branch with same messages. Use True to copy access control.

        Args:
            branch: Source branch (Branch|UUID|str).
            name: Fork name (auto: "{source}_fork").
            capabilities: True=copy, None=empty, or explicit set.
            resources: True=copy, None=empty, or explicit set.

        Returns:
            New Branch with forked_from metadata.
        """
        source = self.get_branch(branch)

        forked = self.create_branch(
            name=name or f"{source.name}_fork",
            messages=source.order,
            capabilities=(
                {*source.capabilities}
                if capabilities is True
                else (capabilities or set())
            ),
            resources=(
                {*source.resources} if resources is True else (resources or set())
            ),
        )

        forked.metadata["forked_from"] = {
            "branch_id": str(source.id),
            "branch_name": source.name,
            "created_at": source.created_at.isoformat(),
            "message_count": len(source),
        }
        return forked

    def add_message(
        self,
        message: Message,
        branches: list[Branch | UUID | str] | Branch | UUID | str | None = None,
    ) -> None:
        """Add message to session, optionally appending to branch(es)."""
        self.communications.add_item(message, progressions=branches)

    async def request(
        self,
        name: str,
        /,
        branch: Branch | UUID | str | None = None,
        poll_timeout: float | None = None,
        poll_interval: float | None = None,
        **options,
    ) -> Calling:
        if branch is not None:
            resolved_branch = self.get_branch(branch)

            from .constraints import resource_must_be_accessible

            resource_must_be_accessible(resolved_branch, name)

        resource = self.resources.get(name)
        return await resource.invoke(
            poll_timeout=poll_timeout,
            poll_interval=poll_interval,
            **options,
        )

    async def conduct(
        self,
        operation_type: str,
        branch: Branch | UUID | str | None = None,
        params: Any | None = None,
    ) -> Operation:
        """Execute operation via registry.

        Args:
            operation_type: Registry key.
            branch: Target branch (default if None).
            params: Operation parameters.

        Returns:
            Invoked Operation (result in op.execution.response).

        Raises:
            RuntimeError: No branch and no default.
            KeyError: Operation not registered.
        """
        resolved = self._resolve_branch(branch)
        op = Operation(
            operation_type=operation_type,
            parameters=params,
            timeout=None,
            streaming=False,
        )
        op.bind(self, resolved)
        await op.invoke()
        return op

    def _resolve_branch(self, branch: Branch | UUID | str | None) -> Branch:
        """Resolve to Branch, falling back to default. Raises if neither available."""
        if branch is not None:
            return self.get_branch(branch)
        if self.default_branch is not None:
            return self.default_branch
        raise RuntimeError("No branch provided and no default branch set")

    def __repr__(self) -> str:
        return (
            f"Session(messages={len(self.messages)}, "
            f"branches={len(self.branches)}, "
            f"services={len(self.resources)})"
        )
