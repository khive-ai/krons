# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Resource leak detection via weakref-based tracking.

Tracks object lifetimes to detect resources that outlive their expected scope.
Uses weakref finalizers for automatic cleanup when objects are garbage collected.
"""

from __future__ import annotations

import threading
import time
import weakref
from dataclasses import dataclass

__all__ = (
    "LeakInfo",
    "LeakTracker",
    "track_resource",
    "untrack_resource",
)


@dataclass(frozen=True, slots=True)
class LeakInfo:
    """Metadata for a tracked resource.

    Attributes:
        name: Identifier for the resource (auto-generated if not provided).
        kind: Optional category/type label for grouping.
        created_at: Unix timestamp when tracking began.
    """

    name: str
    kind: str | None
    created_at: float


class LeakTracker:
    """Thread-safe tracker for detecting resource leaks.

    Uses weakref finalizers to automatically remove entries when objects
    are garbage collected. Call `live()` to inspect currently tracked objects.

    Example:
        >>> tracker = LeakTracker()
        >>> tracker.track(my_connection, name="db_conn", kind="database")
        >>> # ... later, check for leaks ...
        >>> leaks = tracker.live()
        >>> if leaks:
        ...     print(f"Potential leaks: {leaks}")
    """

    def __init__(self) -> None:
        self._live: dict[int, LeakInfo] = {}
        self._lock = threading.Lock()

    def track(self, obj: object, *, name: str | None, kind: str | None) -> None:
        """Begin tracking an object for leak detection.

        Args:
            obj: Object to track (must be weak-referenceable).
            name: Identifier (defaults to "obj-{id}").
            kind: Optional category label.
        """
        info = LeakInfo(name=name or f"obj-{id(obj)}", kind=kind, created_at=time.time())
        key = id(obj)

        def _finalizer(_key: int = key) -> None:
            with self._lock:
                self._live.pop(_key, None)

        with self._lock:
            self._live[key] = info
        weakref.finalize(obj, _finalizer)

    def untrack(self, obj: object) -> None:
        """Manually stop tracking an object."""
        with self._lock:
            self._live.pop(id(obj), None)

    def live(self) -> list[LeakInfo]:
        """Return list of currently tracked (potentially leaked) resources."""
        with self._lock:
            return list(self._live.values())

    def clear(self) -> None:
        """Remove all tracked entries."""
        with self._lock:
            self._live.clear()


_TRACKER = LeakTracker()


def track_resource(obj: object, name: str | None = None, kind: str | None = None) -> None:
    """Track an object using the global leak tracker.

    Args:
        obj: Object to track.
        name: Optional identifier.
        kind: Optional category label.
    """
    _TRACKER.track(obj, name=name, kind=kind)


def untrack_resource(obj: object) -> None:
    """Stop tracking an object in the global tracker."""
    _TRACKER.untrack(obj)
