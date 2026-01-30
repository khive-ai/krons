# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Rate-limited execution infrastructure with dual token bucket support.

Provides permission-based rate limiting for API calls with separate
request count and token usage limits, plus atomic rollback on partial acquire.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Self

from typing_extensions import override

from krons.core import Event, Executor, Processor
from krons.services.endpoint import APICalling
from krons.utils.concurrency import get_cancelled_exc_class, sleep

from .rate_limiter import TokenBucket

if TYPE_CHECKING:
    import asyncio

    from krons.core import Pile

__all__ = ("RateLimitedExecutor", "RateLimitedProcessor")

logger = logging.getLogger(__name__)


class RateLimitedProcessor(Processor):
    """Processor with dual token bucket rate limiting (requests + tokens).

    Enforces both request count and token usage limits atomically.
    Automatically rolls back request bucket if token bucket acquire fails.

    Example:
        >>> req_bucket = TokenBucket(RateLimitConfig(capacity=100, refill_rate=1.67))
        >>> tok_bucket = TokenBucket(RateLimitConfig(capacity=100000, refill_rate=1667))
        >>> processor = await RateLimitedProcessor.create(
        ...     queue_capacity=50, capacity_refresh_time=60.0,
        ...     request_bucket=req_bucket, token_bucket=tok_bucket
        ... )
    """

    event_type = APICalling

    def __init__(
        self,
        queue_capacity: int,
        capacity_refresh_time: float,
        pile: Pile[Event] | None = None,
        executor: Executor | None = None,
        request_bucket: TokenBucket | None = None,
        token_bucket: TokenBucket | None = None,
        replenishment_interval: float = 60.0,
        concurrency_limit: int = 100,
        max_queue_size: int = 1000,
        max_denial_tracking: int = 10000,
    ) -> None:
        """Initialize rate-limited processor.

        Args:
            queue_capacity: Max events per batch.
            capacity_refresh_time: Batch refresh interval (seconds).
            pile: Reference to executor's Flow.items (set by executor).
            executor: Reference to executor for progression updates.
            request_bucket: TokenBucket for request rate limiting.
            token_bucket: TokenBucket for token rate limiting.
            replenishment_interval: Rate limit reset interval.
            concurrency_limit: Max concurrent executions.
            max_queue_size: Max queue size.
            max_denial_tracking: Max denial entries to track.
        """
        super().__init__(  # type: ignore[arg-type]
            queue_capacity=queue_capacity,
            capacity_refresh_time=capacity_refresh_time,
            pile=pile,  # type: ignore[arg-type]
            executor=executor,
            concurrency_limit=concurrency_limit,
            max_queue_size=max_queue_size,
            max_denial_tracking=max_denial_tracking,
        )

        self.request_bucket = request_bucket
        self.token_bucket = token_bucket
        self.replenishment_interval = replenishment_interval
        self.concurrency_limit = concurrency_limit
        self._replenisher_task: asyncio.Task[None] | None = None

    async def start_replenishing(self) -> None:
        """Background task: periodically reset rate limit buckets to full capacity."""
        await self.start()

        try:
            while not self.is_stopped():
                await sleep(self.replenishment_interval)

                if self.request_bucket:
                    await self.request_bucket.reset()
                    logger.debug(
                        "Request bucket replenished: %d requests",
                        self.request_bucket.capacity,
                    )

                if self.token_bucket:
                    await self.token_bucket.reset()
                    logger.debug(
                        "Token bucket replenished: %d tokens",
                        self.token_bucket.capacity,
                    )

        except get_cancelled_exc_class():
            logger.info("Rate limit replenisher task cancelled.")

    @override
    @classmethod
    async def create(  # type: ignore[override]
        cls,
        queue_capacity: int,
        capacity_refresh_time: float,
        pile: Pile[Event] | None = None,
        executor: Executor | None = None,
        request_bucket: TokenBucket | None = None,
        token_bucket: TokenBucket | None = None,
        replenishment_interval: float = 60.0,
        concurrency_limit: int = 100,
        max_queue_size: int = 1000,
        max_denial_tracking: int = 10000,
    ) -> Self:
        """Factory: create processor and start background replenishment task."""
        self = cls(
            queue_capacity=queue_capacity,
            capacity_refresh_time=capacity_refresh_time,
            pile=pile,
            executor=executor,
            request_bucket=request_bucket,
            token_bucket=token_bucket,
            replenishment_interval=replenishment_interval,
            concurrency_limit=concurrency_limit,
            max_queue_size=max_queue_size,
            max_denial_tracking=max_denial_tracking,
        )

        import asyncio

        self._replenisher_task = asyncio.create_task(self.start_replenishing())

        return self

    @override
    async def stop(self) -> None:
        """Stop processor and cancel background replenishment task."""
        if self._replenisher_task:
            self._replenisher_task.cancel()
            try:
                await self._replenisher_task
            except get_cancelled_exc_class():
                pass

        await super().stop()

    @override
    async def request_permission(
        self,
        required_tokens: int | None = None,
        **kwargs: Any,
    ) -> bool:
        """Check rate limits and acquire tokens atomically.

        Acquires from request bucket first, then token bucket. If token bucket
        fails, rolls back request bucket automatically.

        Args:
            required_tokens: Token count for this request (None = skip token check).
            **kwargs: Ignored (for interface compatibility).

        Returns:
            True if permitted, False if rate limited.
        """
        if self.request_bucket is None and self.token_bucket is None:
            return True

        request_acquired = False
        if self.request_bucket:
            request_acquired = await self.request_bucket.try_acquire(tokens=1)
            if not request_acquired:
                logger.debug("Request rate limit exceeded")
                return False

        if self.token_bucket and required_tokens:
            token_acquired = await self.token_bucket.try_acquire(tokens=required_tokens)
            if not token_acquired:
                if request_acquired and self.request_bucket:
                    await self.request_bucket.release(tokens=1)

                logger.debug(
                    f"Token rate limit exceeded (required: {required_tokens}, "
                    f"available: {self.token_bucket.tokens:.0f})"
                )
                return False

        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize processor config to dict (excludes runtime state)."""
        return {
            "queue_capacity": self.queue_capacity,
            "capacity_refresh_time": self.capacity_refresh_time,
            "replenishment_interval": self.replenishment_interval,
            "concurrency_limit": self.concurrency_limit,
            "max_queue_size": self.max_queue_size,
            "max_denial_tracking": self.max_denial_tracking,
            "request_bucket": (self.request_bucket.to_dict() if self.request_bucket else None),
            "token_bucket": (self.token_bucket.to_dict() if self.token_bucket else None),
        }


class RateLimitedExecutor(Executor):
    """Executor with integrated rate limiting via RateLimitedProcessor.

    Manages processor lifecycle and forwards events for permission checking.

    Example:
        >>> executor = RateLimitedExecutor(processor_config={
        ...     "queue_capacity": 50,
        ...     "capacity_refresh_time": 60.0,
        ...     "request_bucket": req_bucket,
        ...     "token_bucket": tok_bucket,
        ... })
        >>> await executor.start()
    """

    processor_type = RateLimitedProcessor

    def __init__(
        self,
        processor_config: dict[str, Any] | None = None,
        strict_event_type: bool = False,
        name: str | None = None,
    ) -> None:
        """Initialize rate-limited executor.

        Args:
            processor_config: Config dict for RateLimitedProcessor.create().
            strict_event_type: If True, Flow enforces exact type matching.
            name: Optional name for the executor Flow.
        """
        super().__init__(
            processor_config=processor_config,
            strict_event_type=strict_event_type,
            name=name or "rate_limited_executor",
        )

    @override
    async def start(self) -> None:
        """Start executor and spawn replenishment task if not running."""
        await super().start()

        if (
            self.processor
            and isinstance(self.processor, RateLimitedProcessor)
            and not self.processor._replenisher_task
        ):
            import asyncio

            self.processor._replenisher_task = asyncio.create_task(
                self.processor.start_replenishing()
            )
