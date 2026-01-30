# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import weakref
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar

from krons.utils import is_coro_func

logger = logging.getLogger(__name__)

__all__ = ["Broadcaster"]


class Broadcaster:
    """Singleton pub/sub with weakref-based automatic subscriber cleanup.

    Subclass and set `_event_type` to define typed broadcasters. Subscribers
    stored as weakrefs (WeakMethod for bound methods) for automatic cleanup.

    Example:
        class UserBroadcaster(Broadcaster):
            _event_type = UserEvent

        UserBroadcaster.subscribe(my_handler)
        await UserBroadcaster.broadcast(UserEvent(...))
    """

    _instance: ClassVar[Broadcaster | None] = None
    _subscribers: ClassVar[
        list[weakref.ref[Callable[[Any], None] | Callable[[Any], Awaitable[None]]]]
    ] = []
    _event_type: ClassVar[type]  #: Override in subclass to restrict event types.

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def subscribe(cls, callback: Callable[[Any], None] | Callable[[Any], Awaitable[None]]) -> None:
        """Add subscriber callback (idempotent, stored as weakref).

        Args:
            callback: Sync or async callable receiving the event.
                      Bound methods use WeakMethod; functions use weakref.
        """
        for weak_ref in cls._subscribers:
            if weak_ref() is callback:
                return
        weak_callback: weakref.ref[Callable[[Any], None] | Callable[[Any], Awaitable[None]]]
        if hasattr(callback, "__self__"):
            weak_callback = weakref.WeakMethod(callback)  # type: ignore[assignment]
        else:
            weak_callback = weakref.ref(callback)
        cls._subscribers.append(weak_callback)

    @classmethod
    def unsubscribe(
        cls, callback: Callable[[Any], None] | Callable[[Any], Awaitable[None]]
    ) -> None:
        """Remove subscriber callback.

        Args:
            callback: Previously subscribed callback to remove.
        """
        for weak_ref in list(cls._subscribers):
            if weak_ref() is callback:
                cls._subscribers.remove(weak_ref)
                return

    @classmethod
    def _cleanup_dead_refs(
        cls,
    ) -> list[Callable[[Any], None] | Callable[[Any], Awaitable[None]]]:
        """Prune dead weakrefs, return live callbacks (in-place update)."""
        callbacks, alive_refs = [], []
        for weak_ref in cls._subscribers:
            if (cb := weak_ref()) is not None:
                callbacks.append(cb)
                alive_refs.append(weak_ref)
        cls._subscribers[:] = alive_refs
        return callbacks

    @classmethod
    async def broadcast(cls, event: Any) -> None:
        """Broadcast event to all subscribers sequentially.

        Args:
            event: Event instance (must match _event_type).

        Raises:
            ValueError: If event type doesn't match _event_type.

        Note:
            Callback exceptions are logged and suppressed.
        """
        if not isinstance(event, cls._event_type):
            raise ValueError(f"Event must be of type {cls._event_type.__name__}")
        for callback in cls._cleanup_dead_refs():
            try:
                if is_coro_func(callback):
                    if (result := callback(event)) is not None:
                        await result
                else:
                    callback(event)
            except Exception as e:
                logger.error(f"Error in subscriber callback: {e}", exc_info=True)

    @classmethod
    def get_subscriber_count(cls) -> int:
        """Count live subscribers (triggers dead ref cleanup)."""
        return len(cls._cleanup_dead_refs())
