# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Worker - Declarative workflow definition via decorated methods.

A Worker defines workflows through:
- @work: Typed operations with assignment DSL (inputs -> outputs)
- @worklink: Conditional edges between work methods

Example:
    class FileCoder(Worker):
        @work(assignment="instruction, context -> code", capacity=2)
        async def write_code(self, form_id, **kwargs):
            result = await llm.chat(**kwargs)
            return form_id, result.code

        @worklink(from_="write_code", to_="execute_code")
        async def write_to_execute(self, from_result):
            form_id, code = from_result
            return {"form_id": form_id, "code": code}

    engine = WorkerEngine(worker=FileCoder())
    await engine.add_task(form=my_form, task_function="write_code")
    await engine.execute()
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable
from uuid import UUID

if TYPE_CHECKING:
    from .form import Form

__all__ = (
    "Worker",
    "WorkConfig",
    "WorkLink",
    "work",
    "worklink",
)


@dataclass
class WorkConfig:
    """Configuration for a @work decorated method.

    Attributes:
        assignment: DSL string 'inputs -> outputs' for typed I/O
        form_param_key: Parameter name that receives form ID
        capacity: Max concurrent executions (rate limiting)
        refresh_time: Seconds between capacity resets
        timeout: Max execution time in seconds
    """

    assignment: str = ""
    form_param_key: str = ""
    capacity: int = 1
    refresh_time: float = 0.1
    timeout: float | None = None


@dataclass
class WorkLink:
    """Edge definition between work methods.

    Attributes:
        from_: Source method name
        to_: Target method name
        handler_name: Name of the handler method on the Worker
    """

    from_: str
    to_: str
    handler_name: str


class Worker:
    """Base class for declarative workflow definition.

    Subclass and decorate methods with @work and @worklink to define workflows.
    Worker maintains form storage and tracks work method metadata.

    Attributes:
        name: Worker name (default: class name)
        forms: Dict mapping form IDs to Form instances
        _work_methods: Registry of @work decorated methods
        _work_links: Registry of @worklink edges
        _stopped: Stop flag for execution

    Example:
        class MyCoder(Worker):
            name = "coder"

            @work(assignment="task -> plan", capacity=2)
            async def plan(self, task_name, **kwargs):
                ...
                return task_name

            @work(assignment="plan -> code", form_param_key="task_name")
            async def implement(self, task_name, **kwargs):
                ...

            @worklink(from_="plan", to_="implement")
            async def plan_to_implement(self, from_result):
                return {"task_name": from_result}
    """

    name: str = "worker"

    def __init__(self) -> None:
        """Initialize worker state."""
        # Form storage: form_id -> Form
        self.forms: dict[str | UUID, Form] = {}

        # Collect @work methods from class
        self._work_methods: dict[str, tuple[Callable, WorkConfig]] = {}
        self._work_links: list[WorkLink] = []
        self._stopped = False

        self._collect_work_metadata()

    def _collect_work_metadata(self) -> None:
        """Scan class for @work and @worklink decorated methods."""
        for name in dir(self):
            if name.startswith("_"):
                continue

            attr = getattr(self, name, None)
            if attr is None:
                continue

            # Check for @work decorator
            if hasattr(attr, "_work_config"):
                config: WorkConfig = attr._work_config
                self._work_methods[name] = (attr, config)

            # Check for @worklink decorator
            if hasattr(attr, "_worklink_from") and hasattr(attr, "_worklink_to"):
                link = WorkLink(
                    from_=attr._worklink_from,
                    to_=attr._worklink_to,
                    handler_name=name,
                )
                self._work_links.append(link)

    def get_links_from(self, method_name: str) -> list[WorkLink]:
        """Get all outgoing links from a method."""
        return [link for link in self._work_links if link.from_ == method_name]

    def get_links_to(self, method_name: str) -> list[WorkLink]:
        """Get all incoming links to a method."""
        return [link for link in self._work_links if link.to_ == method_name]

    async def stop(self) -> None:
        """Signal worker to stop processing."""
        self._stopped = True

    async def start(self) -> None:
        """Clear stop flag to allow processing."""
        self._stopped = False

    def is_stopped(self) -> bool:
        """Check if worker is stopped."""
        return self._stopped

    def __repr__(self) -> str:
        methods = list(self._work_methods.keys())
        links = len(self._work_links)
        forms = len(self.forms)
        return f"{self.__class__.__name__}(methods={methods}, links={links}, forms={forms})"


def work(
    assignment: str = "",
    *,
    form_param_key: str = "",
    capacity: int = 1,
    refresh_time: float = 0.1,
    timeout: float | None = None,
) -> Callable[[Callable[..., Awaitable]], Callable[..., Awaitable]]:
    """Decorator for typed work methods.

    Args:
        assignment: DSL string 'inputs -> outputs' defining typed I/O.
            Used for form field binding when form_param_key is set.
        form_param_key: Parameter name that receives form ID.
            If set, the engine will bind form fields to kwargs.
        capacity: Max concurrent executions (for rate limiting).
        refresh_time: Seconds between capacity resets.
        timeout: Max execution time in seconds.

    Returns:
        Decorator that attaches WorkConfig to the method.

    Example:
        @work(assignment="context, instruction -> code", form_param_key="form_id")
        async def write_code(self, form_id, **kwargs):
            # kwargs contains context, instruction from form
            result = await llm.chat(**kwargs)
            return form_id, result.code
    """

    def decorator(func: Callable[..., Awaitable]) -> Callable[..., Awaitable]:
        config = WorkConfig(
            assignment=assignment,
            form_param_key=form_param_key,
            capacity=capacity,
            refresh_time=refresh_time,
            timeout=timeout,
        )

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        wrapper._work_config = config  # type: ignore[attr-defined]
        return wrapper

    return decorator


def worklink(
    from_: str,
    to_: str,
) -> Callable[[Callable[..., Awaitable]], Callable[..., Awaitable]]:
    """Decorator for conditional edges between work methods.

    The decorated function receives the result from the 'from_' method
    and returns kwargs dict for the 'to_' method. Return None to skip
    the edge (conditional routing).

    Args:
        from_: Source method name
        to_: Target method name

    Returns:
        Decorator that attaches WorkLink info to the method.

    Example:
        @worklink(from_="write_code", to_="execute_code")
        async def write_to_execute(self, from_result):
            form_id, code = from_result
            return {"form_id": form_id, "code": code}

        @worklink(from_="execute_code", to_="debug_code")
        async def execute_to_debug(self, from_result):
            form_id, error = from_result
            if error is not None:  # Conditional edge
                return {"form_id": form_id, "error": error}
            # Return None = edge not taken
    """

    def decorator(func: Callable[..., Awaitable]) -> Callable[..., Awaitable]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        # Store link info - handler_name will be set during _collect_work_metadata
        wrapper._worklink_from = from_  # type: ignore[attr-defined]
        wrapper._worklink_to = to_  # type: ignore[attr-defined]
        return wrapper

    return decorator
