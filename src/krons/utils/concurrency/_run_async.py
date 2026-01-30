# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Sync-to-async bridge for running coroutines from synchronous contexts."""

import threading
from collections.abc import Awaitable
from typing import Any, TypeVar

import anyio

T = TypeVar("T")

__all__ = ("run_async",)


def run_async(coro: Awaitable[T]) -> T:
    """Execute an async coroutine from a synchronous context.

    Creates an isolated thread with its own event loop to run the coroutine,
    avoiding conflicts with any existing event loop in the current thread.
    Thread-safe and blocks until completion.

    Args:
        coro: Awaitable to execute (coroutine, Task, or Future).

    Returns:
        The result of the awaited coroutine.

    Raises:
        BaseException: Any exception raised by the coroutine is re-raised.
        RuntimeError: If the coroutine completes without producing a result.

    Example:
        >>> async def fetch_data():
        ...     return {"status": "ok"}
        >>> result = run_async(fetch_data())
        >>> result
        {'status': 'ok'}

    Note:
        Use sparingly. Prefer native async patterns when possible.
        Each call creates a new thread and event loop.
    """
    result_container: list[Any] = []
    exception_container: list[BaseException] = []

    def run_in_thread() -> None:
        try:

            async def _runner() -> T:
                return await coro

            result = anyio.run(_runner)
            result_container.append(result)
        except BaseException as e:
            exception_container.append(e)

    thread = threading.Thread(target=run_in_thread, daemon=False)
    thread.start()
    thread.join()

    if exception_container:
        raise exception_container[0]
    if not result_container:  # pragma: no cover
        raise RuntimeError("Coroutine did not produce a result")
    return result_container[0]
