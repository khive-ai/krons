# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Async concurrency utilities with lazy loading for fast import.

Core Patterns:
    gather: Run awaitables concurrently, collect results in order.
    race: Return first completion, cancel the rest.
    bounded_map: Apply async function with concurrency limit.
    retry: Exponential backoff with deadline awareness.
    CompletionStream: Iterate results as they complete.

Batch Processing:
    alcall: Apply function to list with retry, timeout, concurrency control.
    bcall: Batch wrapper yielding results per batch.

Primitives (anyio wrappers):
    Lock, Semaphore, Event, Condition, Queue, CapacityLimiter
    TaskGroup, create_task_group

Cancellation:
    CancelScope, move_on_after, move_on_at, fail_after, fail_at
    effective_deadline, get_cancelled_exc_class, is_cancelled

Utilities:
    run_async: Execute coroutine from sync context.
    run_sync: Run sync function in thread pool.
    sleep, current_time, is_coro_func

Resource Tracking:
    LeakTracker, track_resource, untrack_resource
"""

from __future__ import annotations

from typing import TYPE_CHECKING

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # _cancel
    "CancelScope": ("krons.utils.concurrency._cancel", "CancelScope"),
    "effective_deadline": ("krons.utils.concurrency._cancel", "effective_deadline"),
    "fail_after": ("krons.utils.concurrency._cancel", "fail_after"),
    "fail_at": ("krons.utils.concurrency._cancel", "fail_at"),
    "move_on_after": ("krons.utils.concurrency._cancel", "move_on_after"),
    "move_on_at": ("krons.utils.concurrency._cancel", "move_on_at"),
    # _errors
    "get_cancelled_exc_class": (
        "krons.utils.concurrency._errors",
        "get_cancelled_exc_class",
    ),
    "is_cancelled": ("krons.utils.concurrency._errors", "is_cancelled"),
    "non_cancel_subgroup": (
        "krons.utils.concurrency._errors",
        "non_cancel_subgroup",
    ),
    "shield": ("krons.utils.concurrency._errors", "shield"),
    # _patterns
    "CompletionStream": ("krons.utils.concurrency._patterns", "CompletionStream"),
    "bounded_map": ("krons.utils.concurrency._patterns", "bounded_map"),
    "gather": ("krons.utils.concurrency._patterns", "gather"),
    "race": ("krons.utils.concurrency._patterns", "race"),
    "retry": ("krons.utils.concurrency._patterns", "retry"),
    # _primitives
    "CapacityLimiter": ("krons.utils.concurrency._primitives", "CapacityLimiter"),
    "Condition": ("krons.utils.concurrency._primitives", "Condition"),
    "Event": ("krons.utils.concurrency._primitives", "Event"),
    "Lock": ("krons.utils.concurrency._primitives", "Lock"),
    "Queue": ("krons.utils.concurrency._primitives", "Queue"),
    "Semaphore": ("krons.utils.concurrency._primitives", "Semaphore"),
    # _priority_queue
    "PriorityQueue": ("krons.utils.concurrency._priority_queue", "PriorityQueue"),
    "QueueEmpty": ("krons.utils.concurrency._priority_queue", "QueueEmpty"),
    "QueueFull": ("krons.utils.concurrency._priority_queue", "QueueFull"),
    # _resource_tracker
    "LeakInfo": ("krons.utils.concurrency._resource_tracker", "LeakInfo"),
    "LeakTracker": ("krons.utils.concurrency._resource_tracker", "LeakTracker"),
    "track_resource": (
        "krons.utils.concurrency._resource_tracker",
        "track_resource",
    ),
    "untrack_resource": (
        "krons.utils.concurrency._resource_tracker",
        "untrack_resource",
    ),
    # _run_async
    "run_async": ("krons.utils.concurrency._run_async", "run_async"),
    # _task
    "TaskGroup": ("krons.utils.concurrency._task", "TaskGroup"),
    "create_task_group": ("krons.utils.concurrency._task", "create_task_group"),
    # _utils
    "current_time": ("krons.utils.concurrency._utils", "current_time"),
    "is_coro_func": ("krons.utils.concurrency._utils", "is_coro_func"),
    "run_sync": ("krons.utils.concurrency._utils", "run_sync"),
    "sleep": ("krons.utils.concurrency._utils", "sleep"),
    "alcall": ("krons.utils.concurrency._async_call", "alcall"),
    "bcall": ("krons.utils.concurrency._async_call", "bcall"),
}

_LOADED: dict[str, object] = {}

# Re-export built-in ExceptionGroup
ExceptionGroup = ExceptionGroup


def __getattr__(name: str) -> object:
    """Lazy import attributes on first access."""
    if name in _LOADED:
        return _LOADED[name]

    if name in _LAZY_IMPORTS:
        from importlib import import_module

        module_name, attr_name = _LAZY_IMPORTS[name]
        module = import_module(module_name)
        value = getattr(module, attr_name)
        _LOADED[name] = value
        return value

    # Special case: ConcurrencyEvent is alias for Event
    if name == "ConcurrencyEvent":
        value = __getattr__("Event")
        _LOADED[name] = value
        return value

    raise AttributeError(f"module 'krons.utils.concurrency' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return all available attributes for autocomplete."""
    return list(__all__)


# TYPE_CHECKING block for static analysis
if TYPE_CHECKING:
    from ._async_call import alcall, bcall
    from ._cancel import (
        CancelScope,
        effective_deadline,
        fail_after,
        fail_at,
        move_on_after,
        move_on_at,
    )
    from ._errors import (
        get_cancelled_exc_class,
        is_cancelled,
        non_cancel_subgroup,
        shield,
    )
    from ._patterns import CompletionStream, bounded_map, gather, race, retry
    from ._primitives import CapacityLimiter, Condition, Event, Lock, Queue, Semaphore
    from ._priority_queue import PriorityQueue, QueueEmpty, QueueFull
    from ._resource_tracker import (
        LeakInfo,
        LeakTracker,
        track_resource,
        untrack_resource,
    )
    from ._run_async import run_async
    from ._task import TaskGroup, create_task_group
    from ._utils import current_time, is_coro_func, run_sync, sleep

    ConcurrencyEvent = Event

__all__ = (
    "CancelScope",
    "CapacityLimiter",
    "CompletionStream",
    "ConcurrencyEvent",
    "Condition",
    "Event",
    "ExceptionGroup",
    "LeakInfo",
    "LeakTracker",
    "Lock",
    "PriorityQueue",
    "Queue",
    "QueueEmpty",
    "QueueFull",
    "Semaphore",
    "TaskGroup",
    "alcall",
    "bcall",
    "bounded_map",
    "create_task_group",
    "current_time",
    "effective_deadline",
    "fail_after",
    "fail_at",
    "gather",
    "get_cancelled_exc_class",
    "is_cancelled",
    "is_coro_func",
    "move_on_after",
    "move_on_at",
    "non_cancel_subgroup",
    "race",
    "retry",
    "run_async",
    "run_sync",
    "shield",
    "sleep",
    "track_resource",
    "untrack_resource",
)
