# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import weakref
from collections import defaultdict
from collections.abc import Awaitable, Callable
from typing import Any

from krons.utils.concurrency import gather

__all__ = ("EventBus", "Handler")

#: Async event handler signature: any args/kwargs, returns None.
Handler = Callable[..., Awaitable[None]]


class EventBus:
    """In-process pub/sub with weakref-based automatic handler cleanup.

    Provides topic-based event routing where handlers are stored as weak
    references, enabling automatic cleanup when handler objects are
    garbage collected.

    Example:
        bus = EventBus()
        async def on_update(data): print(data)
        bus.subscribe("updates", on_update)
        await bus.emit("updates", {"key": "value"})
    """

    def __init__(self) -> None:
        """Initialize with empty subscription registry."""
        self._subs: dict[str, list[weakref.ref[Handler]]] = defaultdict(list)

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Subscribe async handler to topic.

        Args:
            topic: Event topic name.
            handler: Async callable to invoke on emit. Stored as weakref.
        """
        self._subs[topic].append(weakref.ref(handler))

    def unsubscribe(self, topic: str, handler: Handler) -> bool:
        """Remove handler from topic subscription.

        Args:
            topic: Event topic name.
            handler: Previously subscribed handler.

        Returns:
            True if handler was found and removed, False otherwise.
        """
        if topic not in self._subs:
            return False
        for weak_ref in list(self._subs[topic]):
            if weak_ref() is handler:
                self._subs[topic].remove(weak_ref)
                return True
        return False

    def _cleanup_dead_refs(self, topic: str) -> list[Handler]:
        """Prune dead weakrefs, return live handlers."""
        handlers, alive_refs = [], []
        for weak_ref in self._subs[topic]:
            if (handler := weak_ref()) is not None:
                handlers.append(handler)
                alive_refs.append(weak_ref)
        self._subs[topic] = alive_refs
        return handlers

    async def emit(self, topic: str, *args: Any, **kwargs: Any) -> None:
        """Emit event to all topic subscribers concurrently.

        Args:
            topic: Event topic name.
            *args: Positional args passed to handlers.
            **kwargs: Keyword args passed to handlers.

        Note:
            Handler exceptions are suppressed (logged via gather).
        """
        if topic not in self._subs:
            return
        if handlers := self._cleanup_dead_refs(topic):
            await gather(*(h(*args, **kwargs) for h in handlers), return_exceptions=True)

    def clear(self, topic: str | None = None) -> None:
        """Clear subscriptions.

        Args:
            topic: Specific topic to clear, or None for all topics.
        """
        if topic is None:
            self._subs.clear()
        else:
            self._subs.pop(topic, None)

    def topics(self) -> list[str]:
        """Return list of all registered topic names."""
        return list(self._subs.keys())

    def handler_count(self, topic: str) -> int:
        """Count live handlers for topic (triggers dead ref cleanup).

        Args:
            topic: Event topic name.

        Returns:
            Number of active handlers (excludes GC'd handlers).
        """
        if topic not in self._subs:
            return 0
        return len(self._cleanup_dead_refs(topic))
