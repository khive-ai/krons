# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Async priority queue with condition-based synchronization.

Provides asyncio.PriorityQueue-like interface using anyio primitives.
Uses heapq internally with sequence numbers for stable ordering.
"""

from __future__ import annotations

import heapq
from typing import Any, Generic, TypeVar

from ._primitives import Condition

T = TypeVar("T")

__all__ = ("PriorityQueue", "QueueEmpty", "QueueFull")


class QueueEmpty(Exception):  # noqa: N818
    """Exception raised when queue.get_nowait() is called on empty queue."""


class QueueFull(Exception):  # noqa: N818
    """Exception raised when queue.put_nowait() is called on full queue."""


class PriorityQueue(Generic[T]):
    """Async priority queue using heapq + anyio.Condition.

    Unlike asyncio.PriorityQueue, nowait methods are async (require lock).
    Items stored as (priority, seq, item) for stable ordering when priorities equal.

    Attributes:
        maxsize: Maximum queue size. 0 means unlimited.
    """

    def __init__(self, maxsize: int = 0):
        """Initialize priority queue.

        Args:
            maxsize: Max items allowed. 0 = unlimited.

        Raises:
            ValueError: If maxsize < 0.
        """
        if maxsize < 0:
            raise ValueError("maxsize must be >= 0")
        self.maxsize = maxsize
        self._queue: list[Any] = []
        self._seq = 0
        self._condition = Condition()

    @staticmethod
    def _get_priority(item: Any) -> Any:
        """Extract priority: first element if tuple/list, else item itself."""
        if isinstance(item, (tuple, list)) and item:
            return item[0]
        return item

    async def put(self, item: T) -> None:
        """Put item into queue, blocking if full.

        Args:
            item: Item to enqueue. If tuple/list, first element is priority.
        """
        async with self._condition:
            while self.maxsize > 0 and len(self._queue) >= self.maxsize:
                await self._condition.wait()
            priority = self._get_priority(item)
            entry = (priority, self._seq, item)
            self._seq += 1
            heapq.heappush(self._queue, entry)
            self._condition.notify()

    async def put_nowait(self, item: T) -> None:
        """Put item, raising QueueFull if at capacity. Async (requires lock).

        Args:
            item: Item to enqueue. If tuple/list, first element is priority.

        Raises:
            QueueFull: If queue is at maxsize.
        """
        async with self._condition:
            if self.maxsize > 0 and len(self._queue) >= self.maxsize:
                raise QueueFull("Queue is full")
            priority = self._get_priority(item)
            entry = (priority, self._seq, item)
            self._seq += 1
            heapq.heappush(self._queue, entry)
            self._condition.notify()

    async def get(self) -> T:
        """Get highest priority item (lowest value), blocking if empty.

        Returns:
            Item with lowest priority value.
        """
        async with self._condition:
            while not self._queue:
                await self._condition.wait()
            _priority, _seq, item = heapq.heappop(self._queue)
            self._condition.notify()
            return item

    async def get_nowait(self) -> T:
        """Get item, raising QueueEmpty if none available. Async (requires lock).

        Returns:
            Item with lowest priority value.

        Raises:
            QueueEmpty: If queue is empty.
        """
        async with self._condition:
            if not self._queue:
                raise QueueEmpty("Queue is empty")
            _priority, _seq, item = heapq.heappop(self._queue)
            self._condition.notify()
            return item

    def qsize(self) -> int:
        """Approximate queue size. Unlocked, may be stale. Use for monitoring."""
        return len(self._queue)

    def empty(self) -> bool:
        """Check if empty. Unlocked, may be stale. Use for monitoring."""
        return len(self._queue) == 0

    def full(self) -> bool:
        """Check if full. Unlocked, may be stale. Use for monitoring."""
        return self.maxsize > 0 and len(self._queue) >= self.maxsize
