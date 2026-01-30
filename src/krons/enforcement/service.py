# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""KronService - typed action handlers with policy evaluation."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from pydantic import Field, PrivateAttr

from krons.services import ServiceBackend, ServiceConfig

from .context import RequestContext
from .policy import EnforcementLevel, PolicyEngine, PolicyResolver

logger = logging.getLogger(__name__)

__all__ = (
    "ActionMeta",
    "KronConfig",
    "KronService",
    "action",
    "get_action_meta",
)


class KronConfig(ServiceConfig):
    """Configuration for KronService.

    Attributes:
        operable: Canonical Operable containing all field specs for this service.
        action_timeout: Timeout for action execution (None = no timeout).
        use_policies: Enable policy evaluation.
        policy_timeout: Timeout for policy evaluation.
        fail_open_on_engine_error: Allow action if engine fails (DANGEROUS).
        hooks: Available hooks {name: callable}.
    """

    operable: Any = None
    """Canonical Operable for the service's field namespace."""

    action_timeout: float | None = None
    """Timeout for action execution in seconds. None means no timeout."""

    use_policies: bool = True
    """Enable policy evaluation."""

    policy_timeout: float = 10.0
    """Timeout for policy evaluation in seconds."""

    fail_open_on_engine_error: bool = False
    """If True, allow action when engine fails. DANGEROUS for production."""

    hooks: dict[str, Callable[..., Awaitable[Any]]] = Field(default_factory=dict)
    """Available hooks {name: hook_function}."""


_ACTION_ATTR = "_kron_action"


@dataclass(frozen=True, slots=True)
class ActionMeta:
    """Metadata for an action handler.

    Attributes:
        name: Action identifier (e.g., "consent.grant").
        inputs: Field names from service operable used as inputs.
        outputs: Field names from service operable used as outputs.
        pre_hooks: Hook names to run before action.
        post_hooks: Hook names to run after action.
    """

    name: str
    inputs: frozenset[str] = frozenset()
    outputs: frozenset[str] = frozenset()
    pre_hooks: tuple[str, ...] = ()
    post_hooks: tuple[str, ...] = ()

    # Lazily computed types (set by service at registration)
    _options_type: Any = None
    _result_type: Any = None


def action(
    name: str,
    inputs: set[str] | None = None,
    outputs: set[str] | None = None,
    pre_hooks: list[str] | None = None,
    post_hooks: list[str] | None = None,
) -> Callable[[Callable], Callable]:
    """Decorator to declare action handler metadata.

    Args:
        name: Action identifier (e.g., "consent.grant").
        inputs: Field names from service operable used as inputs.
        outputs: Field names from service operable used as outputs.
        pre_hooks: Hook names to run before action.
        post_hooks: Hook names to run after action.

    Usage:
        @action(
            name="consent.grant",
            inputs={"permissions", "subject_id"},
            outputs={"consent_id", "granted_at"},
        )
        async def _handle_grant(self, options, ctx):
            ...
    """

    def decorator(func: Callable) -> Callable:
        meta = ActionMeta(
            name=name,
            inputs=frozenset(inputs or set()),
            outputs=frozenset(outputs or set()),
            pre_hooks=tuple(pre_hooks or []),
            post_hooks=tuple(post_hooks or []),
        )
        setattr(func, _ACTION_ATTR, meta)
        return func

    return decorator


def get_action_meta(handler: Callable) -> ActionMeta | None:
    """Get action metadata from a handler method."""
    return getattr(handler, _ACTION_ATTR, None)


# =============================================================================
# KronService
# =============================================================================


class KronService(ServiceBackend):
    """Service backend with typed actions.

    Subclasses implement action handlers with @action decorator.
    Actions derive typed I/O from service's canonical operable.

    Example:
        class ConsentService(KronService):
            config = KronConfig(
                name="consent",
                operable=Operable([
                    Spec("permissions", list[str]),
                    Spec("consent_id", UUID),
                    Spec("granted_at", datetime),
                    Spec("subject_id", FK[Subject]),
                ]),
            )

            @action(
                name="consent.grant",
                inputs={"permissions", "subject_id"},
                outputs={"consent_id", "granted_at"},
            )
            async def _handle_grant(self, options, ctx):
                ...

        service = ConsentService()
        result = await service.call("consent.grant", options, ctx)
    """

    config: KronConfig = Field(default_factory=KronConfig)
    _policy_engine: PolicyEngine | None = PrivateAttr(default=None)
    _policy_resolver: PolicyResolver | None = PrivateAttr(default=None)
    _action_registry: dict[str, tuple[Callable, ActionMeta]] = PrivateAttr(default_factory=dict)

    def __init__(
        self,
        config: KronConfig | None = None,
        policy_engine: PolicyEngine | None = None,
        policy_resolver: PolicyResolver | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize service with optional policy engine and resolver.

        Args:
            config: Service configuration.
            policy_engine: PolicyEngine for policy evaluation.
            policy_resolver: PolicyResolver for determining applicable policies.
        """
        super().__init__(config=config, **kwargs)
        self._policy_engine = policy_engine
        self._policy_resolver = policy_resolver
        self._action_registry = {}
        self._register_actions()

    def _register_actions(self) -> None:
        """Scan for @action decorated methods and register them."""
        for name in dir(self):
            if name.startswith("_"):
                method = getattr(self, name, None)
                if method and callable(method):
                    meta = get_action_meta(method)
                    if meta:
                        self._action_registry[meta.name] = (method, meta)
                        self._build_action_types(meta)

    def _build_action_types(self, meta: ActionMeta) -> None:
        """Build options_type and result_type for an action from service operable."""
        if not self.config.operable:
            return

        operable = self.config.operable

        # Validate inputs/outputs exist in operable
        allowed = operable.allowed()
        invalid_inputs = meta.inputs - allowed
        invalid_outputs = meta.outputs - allowed

        if invalid_inputs:
            logger.warning(
                "Action '%s' has inputs not in operable: %s",
                meta.name,
                invalid_inputs,
            )
        if invalid_outputs:
            logger.warning(
                "Action '%s' has outputs not in operable: %s",
                meta.name,
                invalid_outputs,
            )

        # Build typed structures (frozen dataclasses)
        if meta.inputs:
            options_type = operable.compose_structure(
                _to_pascal(meta.name) + "Options",
                include=set(meta.inputs),
                frozen=True,
            )
            object.__setattr__(meta, "_options_type", options_type)

        if meta.outputs:
            result_type = operable.compose_structure(
                _to_pascal(meta.name) + "Result",
                include=set(meta.outputs),
                frozen=True,
            )
            object.__setattr__(meta, "_result_type", result_type)

    @property
    def has_engine(self) -> bool:
        """True if policy engine is configured."""
        return self._policy_engine is not None

    @property
    def has_resolver(self) -> bool:
        """True if policy resolver is configured."""
        return self._policy_resolver is not None

    async def call(
        self,
        name: str,
        options: Any,
        ctx: RequestContext,
    ) -> Any:
        """Call an action by name.

        Args:
            name: Action name (e.g., "consent.grant").
            options: Input data (dict or typed dataclass).
            ctx: Request context.

        Returns:
            Action result.

        Raises:
            ValueError: If action not found.
            PermissionError: If policy blocks action.
        """
        handler, meta = self._fetch_handler(name)

        # Update context
        ctx.name = name

        # Run pre-hooks
        await self._run_hooks(meta.pre_hooks, options, ctx)

        # Evaluate policies
        if self.config.use_policies and self._policy_engine:
            await self._evaluate_policies(ctx)

        # Validate options if we have typed options_type
        if meta._options_type and self.config.operable:
            options = self.config.operable.validate_instance(meta._options_type, options)

        # Execute handler
        result = await handler(options, ctx)

        # Run post-hooks
        await self._run_hooks(meta.post_hooks, options, ctx, result=result)

        return result

    def _fetch_handler(self, name: str) -> tuple[Callable, ActionMeta]:
        """Fetch handler and metadata by action name.

        Args:
            name: Action name.

        Returns:
            Tuple of (handler, ActionMeta).

        Raises:
            ValueError: If action not found.
        """
        if name not in self._action_registry:
            raise ValueError(f"Unknown action: {name}")
        return self._action_registry[name]

    async def _run_hooks(
        self,
        hook_names: tuple[str, ...],
        options: Any,
        ctx: RequestContext,
        result: Any = None,
    ) -> None:
        """Run named hooks from config.hooks."""
        for hook_name in hook_names:
            hook_fn = self.config.hooks.get(hook_name)
            if hook_fn:
                try:
                    await hook_fn(self, options, ctx, result)
                except Exception as e:
                    logger.error("Hook '%s' failed: %s", hook_name, e)
            else:
                logger.warning("Hook '%s' not found in config.hooks", hook_name)

    async def _evaluate_policies(self, ctx: RequestContext) -> None:
        """Evaluate policies via engine."""
        if not self._policy_engine or not self._policy_resolver:
            return

        try:
            resolved = self._policy_resolver.resolve(ctx)

            if not resolved:
                return

            policy_ids = [p.policy_id for p in resolved]
            input_data = ctx.to_dict()

            results = await self._policy_engine.evaluate_batch(policy_ids, input_data)

            for result in results:
                if EnforcementLevel.is_blocking(result):
                    raise PermissionError(f"Policy {result.policy_id} blocked: {result.message}")

        except PermissionError:
            raise
        except Exception as e:
            logger.error("Policy evaluation failed: %s", e)
            if not self.config.fail_open_on_engine_error:
                raise PermissionError(f"Policy engine error: {e}")


def _to_pascal(name: str) -> str:
    """Convert action name to PascalCase.

    consent.grant -> ConsentGrant
    consent_grant -> ConsentGrant
    """
    # Replace dots and underscores, capitalize each part
    parts = name.replace(".", "_").split("_")
    return "".join(part.capitalize() for part in parts)
