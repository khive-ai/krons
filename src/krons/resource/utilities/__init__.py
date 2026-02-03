# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Service utilities: rate limiting and resilience patterns.

Exports:
    Rate limiting:
        - RateLimitConfig: Token bucket configuration
        - TokenBucket: Rate limiter with continuous refill

    Resilience:
        - CircuitBreaker: Fail-fast with state machine (CLOSED/OPEN/HALF_OPEN)
        - CircuitBreakerOpenError: Raised when circuit is open
        - CircuitState: Circuit state enum
        - RetryConfig: Retry policy configuration
        - retry_with_backoff: Async retry with exponential backoff + jitter
"""

from .rate_limiter import RateLimitConfig, TokenBucket
from .resilience import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    RetryConfig,
    retry_with_backoff,
)

__all__ = (
    "CircuitBreaker",
    "CircuitBreakerOpenError",
    "CircuitState",
    "RateLimitConfig",
    "RetryConfig",
    "TokenBucket",
    "retry_with_backoff",
)
