# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Operation: executable graph node bridging session to handler.

Operation.invoke() creates a RequestContext from the bound session/branch
and calls the registered handler with (params, ctx). Handlers never need
to know about the factory pattern — they receive a clean RequestContext.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field, PrivateAttr

from krons.core import Event, Node

from .context import RequestContext

if TYPE_CHECKING:
    from krons.session import Branch, Session

__all__ = ("Operation",)


class Operation(Node, Event):
    """Executable operation node.

    Bridges session.conduct() to handler(params, ctx) by:
    1. Storing bound session/branch references
    2. Creating RequestContext with those references
    3. Looking up the handler from session.operations registry
    4. Calling handler(params, ctx)

    The result is stored in execution.response (via Event.invoke).
    """

    operation_type: str
    parameters: dict[str, Any] | Any = Field(
        default_factory=dict,
        description="Operation parameters (Params dataclass, dict, or model)",
    )

    _session: Any = PrivateAttr(default=None)
    _branch: Any = PrivateAttr(default=None)

    def bind(self, session: Session, branch: Branch) -> Operation:
        """Bind session and branch for execution.

        Must be called before invoke() if not using Session.conduct().

        Args:
            session: Session with operations registry and services.
            branch: Branch for message context.

        Returns:
            Self for chaining.
        """
        self._session = session
        self._branch = branch
        return self

    def _require_binding(self) -> tuple[Session, Branch]:
        """Return bound (session, branch) tuple or raise RuntimeError if unbound."""
        if self._session is None or self._branch is None:
            raise RuntimeError(
                "Operation not bound to session/branch. "
                "Use operation.bind(session, branch) or session.conduct(...)"
            )
        return self._session, self._branch

    async def _invoke(self) -> Any:
        """Execute handler via session's operation registry.

        Creates a RequestContext with bound session/branch references
        and calls handler(params, ctx). Called by Event.invoke().

        Returns:
            Handler result (stored in execution.response).

        Raises:
            RuntimeError: If not bound.
            KeyError: If operation_type not registered.
        """
        session, branch = self._require_binding()
        handler = session.operations.get(self.operation_type)

        ctx = RequestContext(
            name=self.operation_type,
            session_id=session.id,
            branch=branch.name or str(branch.id),
            _bound_session=session,
            _bound_branch=branch,
        )

        return await handler(self.parameters, ctx)

    def __repr__(self) -> str:
        bound = "bound" if self._session is not None else "unbound"
        return f"Operation(type={self.operation_type}, status={self.execution.status.value}, {bound})"
