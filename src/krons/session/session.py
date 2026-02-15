# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Session and Branch: central orchestration for messages, services, and operations.

Session owns branches, messages, services registry, and operations registry.
Branch is a named message progression with capability/resource access control.
"""

from __future__ import annotations

import atexit
import contextlib
from collections.abc import AsyncGenerator, Iterable
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from pydantic import Field, PrivateAttr, field_serializer, model_validator

from krons.core import Element, Flow, Pile, Progression
from krons.core.types import HashableModel, Unset, UnsetType, not_sentinel
from krons.errors import NotFoundError
from krons.resource import Calling, ResourceRegistry, iModel
from krons.work.operations import Operation, OperationRegistry, RequestContext

from .message import Message

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

    # Logging configuration
    log_persist_dir: str | Path | None = Field(
        default=None,
        description="Directory for session dumps. None disables logging.",
    )
    log_auto_save_on_exit: bool = Field(
        default=True,
        description="Register atexit handler on Session creation.",
    )

    @property
    def logging_enabled(self) -> bool:
        """True if logging is configured (log_persist_dir is set)."""
        return self.log_persist_dir is not None


class Session(Element):
    user: str | None = None
    communications: Flow[Message, Branch] = Field(default_factory=lambda: Flow(item_type=Message))
    resources: ResourceRegistry = Field(default_factory=ResourceRegistry, exclude=True)
    operations: OperationRegistry = Field(default_factory=OperationRegistry, exclude=True)
    config: SessionConfig = Field(default_factory=SessionConfig)
    default_branch_id: UUID | None = None

    _registered_atexit: bool = PrivateAttr(default=False)
    _dump_count: int = PrivateAttr(default=0)

    @field_serializer("communications")
    def _serialize_communications(self, flow: Flow) -> dict:
        """Use Flow's custom to_dict for proper nested serialization."""
        return flow.to_dict(mode="json")

    @model_validator(mode="after")
    def _validate_default_branch(self) -> Session:
        """Auto-create default branch and register built-in operations."""
        if self.config.auto_create_default_branch and self.default_branch is None:
            default_branch_name = self.config.default_branch_name or "main"
            self.create_branch(
                name=default_branch_name,
                capabilities=self.config.shared_capabilities,
                resources=self.config.shared_resources,
            )
            self.set_default_branch(default_branch_name)

        # Register atexit handler if configured
        if (
            self.config.logging_enabled
            and self.config.log_auto_save_on_exit
            and not self._registered_atexit
        ):
            atexit.register(self._save_at_exit)
            self._registered_atexit = True

        # Register built-in operations (lazy import avoids circular)
        from krons.agent.operations import (
            generate,
            operate,
            react,
            react_stream,
            structure,
        )

        for name, handler in (
            ("generate", generate),
            ("structure", structure),
            ("operate", operate),
            ("react", react),
            ("react_stream", react_stream),
        ):
            if not self.operations.has(name):
                self.operations.register(name, handler)

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
        with contextlib.suppress(KeyError, NotFoundError):
            return self.communications.get_progression(self.default_branch_id)
        return None

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
                {*source.capabilities} if capabilities is True else (capabilities or set())
            ),
            resources=({*source.resources} if resources is True else (resources or set())),
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
        verbose: bool = False,
    ) -> Operation:
        """Execute operation via registry.

        Args:
            operation_type: Registry key.
            branch: Target branch (default if None).
            params: Operation parameters.
            verbose: Print real-time status updates.

        Returns:
            Invoked Operation (result in op.execution.response).

        Raises:
            RuntimeError: No branch and no default.
            KeyError: Operation not registered.
        """
        resolved = self._resolve_branch(branch)

        if verbose:
            from krons.utils.display import Timer, status

            branch_name = resolved.name or str(resolved.id)[:8]
            status(
                f"conduct({operation_type}) on branch={branch_name}",
                style="info",
            )

        op = Operation(
            operation_type=operation_type,
            parameters=params,
        )
        op._verbose = verbose
        op.bind(self, resolved)

        if verbose:
            with Timer(f"{operation_type} completed"):
                await op.invoke()

            resp = op.execution.response
            if op.execution.error:
                status(f"ERROR: {op.execution.error}", style="error")
            elif isinstance(resp, str):
                status(f"response: {len(resp)} chars", style="success")
            else:
                status(f"response: {type(resp).__name__}", style="success")
        else:
            await op.invoke()

        return op

    async def stream_conduct(
        self,
        operation_type: str,
        branch: Branch | UUID | str | None = None,
        params: Any | None = None,
        verbose: bool = False,
    ) -> AsyncGenerator[Any, None]:
        """Stream operation results via async generator.

        For streaming handlers like react_stream that yield intermediate
        results per round. Bypasses the Operation wrapper since streaming
        handlers produce multiple values, not a single response.

        Args:
            operation_type: Registry key (e.g. "react_stream").
            branch: Target branch (default if None).
            params: Operation parameters.
            verbose: Enable real-time streaming output.

        Yields:
            Handler results (e.g., ReActAnalysis per round).
        """
        resolved = self._resolve_branch(branch)
        handler = self.operations.get(operation_type)

        ctx = RequestContext(
            name=operation_type,
            session_id=self.id,
            branch=resolved.name or str(resolved.id),
            _bound_session=self,
            _bound_branch=resolved,
            _verbose=verbose,
        )

        if verbose:
            from krons.utils.display import status

            branch_name = resolved.name or str(resolved.id)[:8]
            status(
                f"stream_conduct({operation_type}) on branch={branch_name}",
                style="info",
            )

        async for result in handler(params, ctx):
            yield result

    def dump(self, clear: bool = False) -> Path | None:
        """Sync dump entire session state for replay.

        Serializes session (messages, branches, config) to JSON.
        Resources and operations are excluded (re-register on restore).
        To restore: Session.from_dict(data), then re-register resources.

        Args:
            clear: Clear communications after dump (default False).

        Returns:
            Path to session file, or None if logging disabled or empty.
        """
        from krons.utils import create_path, json_dumpb
        from krons.utils.concurrency import run_async

        if not self.config.logging_enabled or len(self.messages) == 0:
            return None

        self._dump_count += 1

        filepath = run_async(
            create_path(
                directory=self.config.log_persist_dir,
                filename=f"session_{str(self.id)[:8]}_{self._dump_count}",
                extension=".json",
                timestamp=True,
                file_exist_ok=True,
            )
        )

        data = json_dumpb(self.to_dict(mode="json"), safe_fallback=True)
        std_path = Path(filepath)
        std_path.write_bytes(data)

        if clear:
            self.communications.clear()

        return std_path

    async def adump(self, clear: bool = False) -> Path | None:
        """Async dump entire session state for replay.

        Serializes the full session (messages, branches, config) to JSON.
        To restore: Session.from_dict(data), then re-register resources.

        Args:
            clear: Clear communications after dump (default False).

        Returns:
            Path to session file, or None if logging disabled or empty.
        """
        from krons.utils import create_path, json_dumpb

        if not self.config.logging_enabled or len(self.messages) == 0:
            return None

        async with self.messages:
            self._dump_count += 1

            filepath = await create_path(
                directory=self.config.log_persist_dir,
                filename=f"session_{str(self.id)[:8]}_{self._dump_count}",
                extension=".json",
                timestamp=True,
                file_exist_ok=True,
            )

            data = json_dumpb(self.to_dict(mode="json"), safe_fallback=True)
            await filepath.write_bytes(data)

            if clear:
                self.communications.clear()

        return Path(filepath)

    def _save_at_exit(self) -> None:
        """atexit callback. Dumps session synchronously. Errors are suppressed."""
        if len(self.messages) > 0:
            try:
                self.dump(clear=False)
            except Exception:
                pass  # Silent failure during interpreter shutdown

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
