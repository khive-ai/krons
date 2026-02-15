# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.utils.concurrency._async_call - Async call utilities."""

import pytest

from krons.utils.concurrency import alcall


class TestAlcall:
    """Test alcall (async list call)."""

    @pytest.mark.anyio
    async def test_alcall_basic(self):
        """alcall should execute async function on list items."""

        async def double(x):
            return x * 2

        result = await alcall([1, 2, 3], double)
        assert result == [2, 4, 6]

    @pytest.mark.anyio
    async def test_alcall_sync_function(self):
        """alcall should handle sync functions."""

        def triple(x):
            return x * 3

        result = await alcall([1, 2, 3], triple)
        assert result == [3, 6, 9]

    @pytest.mark.anyio
    async def test_alcall_with_kwargs(self):
        """alcall should pass kwargs to function."""

        async def multiply(x, factor=1):
            return x * factor

        result = await alcall([1, 2, 3], multiply, factor=5)
        assert result == [5, 10, 15]

    @pytest.mark.anyio
    async def test_alcall_max_concurrent(self):
        """alcall should respect max_concurrent."""
        import asyncio

        running = 0
        max_seen = 0

        async def track_concurrent(x):
            nonlocal running, max_seen
            running += 1
            max_seen = max(max_seen, running)
            await asyncio.sleep(0.01)
            running -= 1
            return x

        await alcall([1, 2, 3, 4, 5], track_concurrent, max_concurrent=2)
        assert max_seen <= 2

    @pytest.mark.anyio
    async def test_alcall_return_exceptions(self):
        """alcall should return exceptions when return_exceptions=True."""

        async def maybe_fail(x):
            if x == 2:
                raise ValueError("fail")
            return x

        result = await alcall([1, 2, 3], maybe_fail, return_exceptions=True)
        assert result[0] == 1
        assert isinstance(result[1], ValueError)
        assert result[2] == 3

    @pytest.mark.anyio
    async def test_alcall_raises_without_return_exceptions(self):
        """alcall should raise ExceptionGroup when return_exceptions=False."""

        async def fail(x):
            raise ValueError(f"fail {x}")

        with pytest.raises(ExceptionGroup):
            await alcall([1, 2], fail, return_exceptions=False)

    @pytest.mark.anyio
    async def test_alcall_retry(self):
        """alcall should retry on failure."""
        attempts = {}

        async def flaky(x):
            attempts[x] = attempts.get(x, 0) + 1
            if attempts[x] < 2:
                raise ValueError("transient")
            return x

        result = await alcall([1, 2], flaky, retry_attempts=2)
        assert result == [1, 2]
        assert all(a >= 2 for a in attempts.values())

    @pytest.mark.anyio
    async def test_alcall_retry_default(self):
        """alcall should use default on retry exhaustion."""

        async def always_fail(x):
            raise ValueError("permanent")

        result = await alcall([1, 2], always_fail, retry_attempts=1, retry_default="failed")
        assert result == ["failed", "failed"]

    @pytest.mark.anyio
    async def test_alcall_empty_input(self):
        """alcall should handle empty input."""

        async def identity(x):
            return x

        result = await alcall([], identity)
        assert result == []

    @pytest.mark.anyio
    async def test_alcall_preserves_order(self):
        """alcall should preserve input order in output."""
        import asyncio
        import random

        async def delay_random(x):
            await asyncio.sleep(random.uniform(0, 0.02))
            return x

        result = await alcall([1, 2, 3, 4, 5], delay_random)
        assert result == [1, 2, 3, 4, 5]
