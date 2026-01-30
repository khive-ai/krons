# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field, field_serializer, field_validator

from krons.core import Element, Executor
from krons.protocols import Invocable, implements
from krons.utils.concurrency import sleep

from .backend import ServiceBackend
from .endpoint import Endpoint
from .hook import HookRegistry
from .utilities.rate_limited_executor import RateLimitedExecutor
from .utilities.rate_limiter import RateLimitConfig, TokenBucket

if TYPE_CHECKING:
    from .backend import Calling


__all__ = ("iModel",)


@implements(Invocable)
class iModel(Element):  # noqa: N801
    """Unified service interface wrapping ServiceBackend with rate limiting and hooks.

    Combines ServiceBackend (API abstraction) with optional:
    - Rate limiting: TokenBucket (simple) or Executor (event-driven)
    - Hook registry: Lifecycle callbacks at PreEventCreate/PreInvocation/PostInvocation

    Attributes:
        backend: ServiceBackend instance (e.g., Endpoint for HTTP APIs).
        rate_limiter: Optional TokenBucket for simple blocking rate limits.
        executor: Optional Executor for event-driven processing with rate limiting.
        hook_registry: Optional HookRegistry for invocation lifecycle callbacks.
        provider_metadata: Provider-specific state (e.g., Claude Code session_id).
    """

    _EXECUTOR_POLL_TIMEOUT_ITERATIONS = 100
    _EXECUTOR_POLL_SLEEP_INTERVAL = 0.1

    backend: ServiceBackend | None = Field(
        None,
        description="ServiceBackend instance (e.g., Endpoint)",
    )

    rate_limiter: TokenBucket | None = Field(
        None,
        description="Optional TokenBucket rate limiter (simple blocking)",
    )

    executor: Executor | None = Field(
        None,
        description="Optional Executor for event-driven processing with rate limiting",
    )

    hook_registry: HookRegistry | None = Field(
        None,
        description="Optional HookRegistry for invocation lifecycle hooks",
    )

    provider_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific metadata (e.g., Claude Code session_id for context continuation)",
    )

    def __init__(
        self,
        backend: ServiceBackend,
        rate_limiter: TokenBucket | None = None,
        executor: Executor | None = None,
        hook_registry: HookRegistry | None = None,
        queue_capacity: int = 100,
        capacity_refresh_time: float = 60,
        limit_requests: int | None = None,
    ):
        """Initialize iModel with ServiceBackend.

        Args:
            backend: ServiceBackend instance (required).
            rate_limiter: TokenBucket for simple blocking rate limits.
            executor: Executor for event-driven processing.
            hook_registry: HookRegistry for lifecycle callbacks.
            queue_capacity: Event queue size for auto-constructed executor.
            capacity_refresh_time: Seconds for rate limit bucket refill.
            limit_requests: If set without executor, auto-constructs RateLimitedExecutor.
        """
        if executor is None and limit_requests:
            request_bucket = TokenBucket(
                RateLimitConfig(
                    capacity=limit_requests,
                    refill_rate=limit_requests / capacity_refresh_time,
                )
            )

            executor = RateLimitedExecutor(
                processor_config={
                    "queue_capacity": queue_capacity,
                    "capacity_refresh_time": capacity_refresh_time,
                    "request_bucket": request_bucket,
                }
            )

        super().__init__(
            backend=backend,
            rate_limiter=rate_limiter,
            executor=executor,
            hook_registry=hook_registry,
        )

    @property
    def name(self) -> str:
        """Service name from backend."""
        if self.backend is None:
            raise RuntimeError("Backend not configured")
        return self.backend.name

    @property
    def version(self) -> str:
        """Service version from backend."""
        if self.backend is None:
            raise RuntimeError("Backend not configured")
        return self.backend.version or ""

    @property
    def tags(self) -> set[str]:
        """Service tags from backend."""
        if self.backend is None:
            raise RuntimeError("Backend not configured")
        return self.backend.tags

    async def create_calling(
        self,
        timeout: float | None = None,
        streaming: bool = False,
        create_event_exit_hook: bool | None = None,
        create_event_hook_timeout: float = 10.0,
        create_event_hook_params: dict | None = None,
        pre_invoke_exit_hook: bool | None = None,
        pre_invoke_hook_timeout: float = 30.0,
        pre_invoke_hook_params: dict | None = None,
        post_invoke_exit_hook: bool | None = None,
        post_invoke_hook_timeout: float = 30.0,
        post_invoke_hook_params: dict | None = None,
        **arguments: Any,
    ) -> Calling:
        """Create Calling instance via backend.

        Calls create_payload on backend to get validated payload.
        Attaches hook_registry to Calling if configured.

        Args:
            timeout: Event timeout in seconds (enforced in Event.invoke via fail_after)
            streaming: Whether this is a streaming request (Event.streaming attr)
            create_event_exit_hook: Whether pre-event-create hook should trigger exit on failure (None = use default)
            create_event_hook_timeout: Timeout for pre-event-create hook execution in seconds
            create_event_hook_params: Optional parameters to pass to pre-event-create hook
            pre_invoke_exit_hook: Whether pre-invoke hook should trigger exit on failure (None = use default)
            pre_invoke_hook_timeout: Timeout for pre-invoke hook execution in seconds
            pre_invoke_hook_params: Optional parameters to pass to pre-invoke hook
            post_invoke_exit_hook: Whether post-invoke hook should trigger exit on failure (None = use default)
            post_invoke_hook_timeout: Timeout for post-invoke hook execution in seconds
            post_invoke_hook_params: Optional parameters to pass to post-invoke hook
            **arguments: Request arguments to pass to backend
        """
        from .hook import HookEvent, HookPhase

        if self.backend is None:
            raise RuntimeError("Backend not configured")

        calling_type = self.backend.event_type

        if self.hook_registry is not None and self.hook_registry._can_handle(
            hp_=HookPhase.PreEventCreate
        ):
            h_ev = HookEvent(
                hook_phase=HookPhase.PreEventCreate,
                event_like=calling_type,
                registry=self.hook_registry,
                exit=(create_event_exit_hook if create_event_exit_hook is not None else False),
                timeout=create_event_hook_timeout,
                streaming=False,
                params=create_event_hook_params or {},
            )
            await h_ev.invoke()

            if h_ev._should_exit:
                raise h_ev._exit_cause or RuntimeError(
                    "PreEventCreate hook requested exit without a cause"
                )

        payload = self.backend.create_payload(request=arguments)
        calling: Calling = calling_type(
            backend=self.backend,
            payload=payload,
            timeout=timeout,
            streaming=streaming,
        )

        if self.hook_registry is not None and self.hook_registry._can_handle(
            hp_=HookPhase.PreInvocation
        ):
            calling.create_pre_invoke_hook(
                hook_registry=self.hook_registry,
                exit_hook=(pre_invoke_exit_hook if pre_invoke_exit_hook is not None else False),
                hook_timeout=pre_invoke_hook_timeout,
                hook_params=pre_invoke_hook_params or {},
            )

        if self.hook_registry is not None and self.hook_registry._can_handle(
            hp_=HookPhase.PostInvocation
        ):
            calling.create_post_invoke_hook(
                hook_registry=self.hook_registry,
                exit_hook=(post_invoke_exit_hook if post_invoke_exit_hook is not None else False),
                hook_timeout=post_invoke_hook_timeout,
                hook_params=post_invoke_hook_params or {},
            )

        if timeout is not None:
            calling.timeout = timeout
        if streaming:
            calling.streaming = streaming

        return calling

    async def invoke(
        self,
        calling: Calling | None = None,
        poll_timeout: float | None = None,
        poll_interval: float | None = None,
        **arguments: Any,
    ) -> Calling:
        """Invoke calling with optional event-driven processing.

        Routes invocation based on executor presence:
        - If executor configured: event-driven processing with rate limiting (lionagi v0 pattern)
        - Otherwise: direct invocation with optional simple rate limiting

        Hooks are handled by Calling itself during invocation.

        Args:
            calling: Pre-created Calling instance. If provided, **arguments are IGNORED
                and the calling is invoked directly. Use this when you need to configure
                the Calling beforehand (e.g., set timeout on the Event).
            poll_timeout: Max seconds to wait for executor completion (default: 10s).
                For long-running LLM calls, increase this (e.g., 120s for large models).
            poll_interval: Seconds between status checks (default: 0.1s).
            **arguments: Request arguments passed to create_calling. IGNORED if calling provided.

        Returns:
            Calling instance with execution results populated

        Raises:
            TimeoutError: If rate limit acquisition or polling times out
            RuntimeError: If event aborted after 3 permission denials (executor path)

        Example:
            # Standard usage - create and invoke in one call
            calling = await imodel.invoke(model="gpt-4", messages=[...])

            # Pre-created calling with custom timeout
            calling = await imodel.create_calling(model="gpt-4", messages=[...])
            calling.timeout = 120.0  # 2 minute timeout
            calling = await imodel.invoke(calling=calling)
        """
        if calling is None:
            calling = await self.create_calling(**arguments)

        if self.executor:
            if self.executor.processor is None or self.executor.processor.is_stopped():
                await self.executor.start()

            await self.executor.append(calling)
            await self.executor.forward()

            # Poll for completion (fast backends see ~100-200% overhead, slow backends <10%)
            interval = poll_interval or self._EXECUTOR_POLL_SLEEP_INTERVAL
            timeout_seconds = poll_timeout or (
                self._EXECUTOR_POLL_TIMEOUT_ITERATIONS * self._EXECUTOR_POLL_SLEEP_INTERVAL
            )
            max_iterations = int(timeout_seconds / interval)
            ctr = 0

            while calling.execution.status.value in ["pending", "processing"]:
                if ctr > max_iterations:
                    raise TimeoutError(
                        f"Event processing timeout after {timeout_seconds:.1f}s: {calling.id}"
                    )
                await self.executor.forward()
                ctr += 1
                await sleep(interval)

            if calling.execution.status.value == "aborted":
                raise RuntimeError(
                    f"Event aborted after 3 permission denials (rate limited): {calling.id}"
                )
            elif calling.execution.status.value == "failed":
                raise calling.execution.error or RuntimeError(f"Event failed: {calling.id}")

            self._store_claude_code_session_id(calling)
            return calling

        else:
            if self.rate_limiter:
                acquired = await self.rate_limiter.acquire(timeout=30.0)
                if not acquired:
                    raise TimeoutError("Rate limit acquisition timeout (30s)")

            await calling.invoke()
            self._store_claude_code_session_id(calling)
            return calling

    def _store_claude_code_session_id(self, calling: Calling) -> None:
        """Extract and store Claude Code session_id for context continuation."""
        from krons.types import is_sentinel

        from .backend import NormalizedResponse

        if (
            isinstance(self.backend, Endpoint)
            and self.backend.config.provider == "claude_code"
            and not is_sentinel(calling.execution.response)
        ):
            response = calling.execution.response
            # session_id is in response metadata
            if isinstance(response, NormalizedResponse) and response.metadata:
                session_id = response.metadata.get("session_id")
                if session_id:
                    self.provider_metadata["session_id"] = session_id

    @field_serializer("backend")
    def _serialize_backend(self, backend: ServiceBackend) -> dict[str, Any] | None:
        """Serialize backend to dict with kron_class for polymorphic restoration."""
        if backend is None:
            return None
        backend_dict = backend.model_dump()
        if "metadata" not in backend_dict:
            backend_dict["metadata"] = {}
        backend_dict["metadata"]["kron_class"] = backend.__class__.class_name(full=True)
        return backend_dict

    @field_serializer("rate_limiter")
    def _serialize_rate_limiter(self, v: TokenBucket | None) -> dict[str, Any] | None:
        if v is None:
            return None
        return v.to_dict()

    @field_serializer("executor")
    def _serialize_executor(self, executor: Executor | None) -> dict[str, Any] | None:
        """Serialize executor config (ephemeral state lost, fresh capacity on restore)."""
        if executor is None:
            return None

        if isinstance(executor, RateLimitedExecutor):
            config = {**executor.processor_config}
            if "request_bucket" in config and config["request_bucket"] is not None:
                bucket = config["request_bucket"]
                if isinstance(bucket, TokenBucket):
                    config["request_bucket"] = bucket.to_dict()
            return config

        return None

    @field_validator("rate_limiter", mode="before")
    @classmethod
    def _deserialize_rate_limiter(cls, v: Any) -> TokenBucket | None:
        """Reconstruct TokenBucket from RateLimitConfig dict."""
        if v is None:
            return None

        if isinstance(v, TokenBucket):
            return v

        if not isinstance(v, dict):
            raise ValueError("rate_limiter must be a dict or TokenBucket instance")

        config = RateLimitConfig(**v)
        return TokenBucket(config)

    @field_validator("backend", mode="before")
    @classmethod
    def _deserialize_backend(cls, v: Any) -> ServiceBackend:
        """Reconstruct backend via Element polymorphic deserialization."""
        if v is None:
            raise ValueError("backend is required")

        if isinstance(v, ServiceBackend):
            return v

        if not isinstance(v, dict):
            raise ValueError("backend must be a dict or ServiceBackend instance")

        from krons.core import Element

        backend = Element.from_dict(v)

        if not isinstance(backend, ServiceBackend):
            raise ValueError(
                f"Deserialized backend must be ServiceBackend subclass, got: {type(backend).__name__}"
            )
        return backend

    @field_validator("executor", mode="before")
    @classmethod
    def _deserialize_executor(cls, v: Any) -> Executor | None:
        """Reconstruct executor from config dict (TokenBuckets get fresh capacity)."""
        if v is None:
            return None

        if isinstance(v, Executor):
            return v

        if not isinstance(v, dict):
            raise ValueError("executor must be a dict or Executor instance")

        config = {**v}
        if "request_bucket" in config and isinstance(config["request_bucket"], dict):
            config["request_bucket"] = TokenBucket(RateLimitConfig(**config["request_bucket"]))
        if "token_bucket" in config and isinstance(config["token_bucket"], dict):
            config["token_bucket"] = TokenBucket(RateLimitConfig(**config["token_bucket"]))

        return RateLimitedExecutor(processor_config=config)

    def _to_dict(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize to dict, excluding id/created_at for fresh identity on reconstruction."""
        kwargs.setdefault("exclude", set()).update({"id", "created_at"})
        return super()._to_dict(**kwargs)

    def __repr__(self) -> str:
        """String representation."""
        if self.backend is None:
            return "iModel(backend=None)"
        return f"iModel(backend={self.backend.name}, version={self.backend.version})"

    async def __aenter__(self) -> iModel:
        """Enter async context, starting executor if configured."""
        if self.executor is not None and (
            self.executor.processor is None or self.executor.processor.is_stopped()
        ):
            await self.executor.start()
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> bool:
        """Exit async context, stopping executor if running.

        Returns:
            False to propagate any exceptions (never suppresses)
        """
        if (
            self.executor is not None
            and self.executor.processor is not None
            and not self.executor.processor.is_stopped()
        ):
            await self.executor.stop()
        return False
