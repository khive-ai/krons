# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Concurrency utility functions.

Thin wrappers around anyio for time, sleep, thread offloading, and coroutine detection.
"""

import inspect
from collections.abc import Callable
from functools import cache, partial
from typing import Any, ParamSpec, TypeVar

import anyio
import anyio.to_thread

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

__all__ = ("current_time", "is_coro_func", "run_sync", "sleep")


@cache
def _is_coro_func_cached(func: Callable[..., Any]) -> bool:
    """Cached coroutine check. Internal: expects already-unwrapped func."""
    return inspect.iscoroutinefunction(func)


def is_coro_func(func: Callable[..., Any]) -> bool:
    """Check if func is a coroutine function, unwrapping partials first.

    Unwraps partials before caching to prevent unbounded cache growth
    (each partial instance would otherwise be a separate cache key).

    Args:
        func: Callable to check (may be wrapped in partial).

    Returns:
        True if underlying function is async def.
    """
    while isinstance(func, partial):
        func = func.func
    return _is_coro_func_cached(func)


def current_time() -> float:
    """Get current monotonic time in seconds.

    Returns:
        Monotonic clock value from anyio.
    """
    return anyio.current_time()


async def run_sync(func: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Run synchronous function in thread pool without blocking event loop.

    Args:
        func: Synchronous callable.
        *args: Positional arguments for func.
        **kwargs: Keyword arguments for func.

    Returns:
        Result of func(*args, **kwargs).
    """
    if kwargs:
        func_with_kwargs = partial(func, **kwargs)
        return await anyio.to_thread.run_sync(func_with_kwargs, *args)
    return await anyio.to_thread.run_sync(func, *args)


async def sleep(seconds: float) -> None:
    """Async sleep without blocking the event loop.

    Args:
        seconds: Duration to sleep.
    """
    await anyio.sleep(seconds)
