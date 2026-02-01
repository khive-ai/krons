# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""WorkerEngine - Execution driver for Worker workflows.

The engine manages task execution, following worklinks to traverse the
workflow graph defined by @work and @worklink decorators.

Example:
    worker = FileCoder()
    engine = WorkerEngine(worker=worker, refresh_time=0.3)

    # Add a task starting at a specific function
    task = await engine.add_task(
        form=my_form,
        task_function="start_task",
        task_max_steps=20,
    )

    # Run until all tasks complete
    await engine.execute()

    # Or run indefinitely
    await engine.execute_lasting()
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from krons.utils import concurrency

if TYPE_CHECKING:
    from .form import Form
    from .worker import WorkConfig, Worker, WorkLink

__all__ = ("WorkerEngine", "WorkerTask")


@dataclass
class WorkerTask:
    """A task being executed by the engine.

    Attributes:
        id: Unique task identifier
        function: Current method name to execute
        kwargs: Arguments for the method
        status: PENDING, PROCESSING, COMPLETED, FAILED
        result: Final result when completed
        error: Exception if failed
        max_steps: Max workflow steps before stopping
        current_step: Current step count
        history: List of (function, result) tuples for debugging
    """

    id: UUID = field(default_factory=uuid4)
    function: str = ""
    kwargs: dict[str, Any] = field(default_factory=dict)
    status: str = "PENDING"
    result: Any = None
    error: Exception | None = None
    max_steps: int = 100
    current_step: int = 0
    history: list[tuple[str, Any]] = field(default_factory=list)


class WorkerEngine:
    """Execution driver for Worker workflows.

    Manages a queue of tasks, executing them through the workflow graph
    defined by the worker's @work and @worklink decorators.

    Attributes:
        worker: The Worker instance to execute
        refresh_time: Seconds between processing cycles
        tasks: Dict of active tasks by ID
        _task_queue: Async queue for pending work
        _stopped: Stop flag

    Example:
        engine = WorkerEngine(worker=my_worker)
        task = await engine.add_task(
            form=my_form,
            task_function="entry_point",
        )
        await engine.execute()
    """

    def __init__(
        self,
        worker: Worker,
        refresh_time: float = 0.1,
        max_concurrent: int = 10,
    ) -> None:
        """Initialize the engine.

        Args:
            worker: Worker instance with @work/@worklink methods
            refresh_time: Seconds between processing cycles
            max_concurrent: Max concurrent task executions
        """
        self.worker = worker
        self.refresh_time = refresh_time
        self.max_concurrent = max_concurrent

        self.tasks: dict[UUID, WorkerTask] = {}
        self._task_queue: asyncio.Queue[UUID] = asyncio.Queue()
        self._stopped = False
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def add_task(
        self,
        task_function: str,
        task_max_steps: int = 100,
        **kwargs: Any,
    ) -> WorkerTask:
        """Add a new task to the execution queue.

        Args:
            task_function: Entry method name to start execution
            task_max_steps: Max workflow steps before stopping
            **kwargs: Arguments for the entry method

        Returns:
            WorkerTask instance (can be monitored for status)

        Raises:
            ValueError: If task_function not found in worker
        """
        if task_function not in self.worker._work_methods:
            raise ValueError(
                f"Method '{task_function}' not found. "
                f"Available: {list(self.worker._work_methods.keys())}"
            )

        task = WorkerTask(
            function=task_function,
            kwargs=kwargs,
            max_steps=task_max_steps,
        )
        self.tasks[task.id] = task
        await self._task_queue.put(task.id)

        return task

    async def execute(self) -> None:
        """Execute all queued tasks until queue is empty.

        Processes tasks through their workflow graphs, following worklinks.
        Returns when all tasks are completed or failed.
        """
        self._stopped = False
        await self.worker.start()

        while not self._stopped and not self._task_queue.empty():
            await self._process_cycle()
            await concurrency.sleep(self.refresh_time)

    async def execute_lasting(self) -> None:
        """Execute indefinitely until stop() is called.

        Useful for long-running worker services that continuously
        process incoming tasks.
        """
        self._stopped = False
        await self.worker.start()

        while not self._stopped:
            await self._process_cycle()
            await concurrency.sleep(self.refresh_time)

    async def stop(self) -> None:
        """Stop the execution loop."""
        self._stopped = True
        await self.worker.stop()

    async def _process_cycle(self) -> None:
        """Process one cycle of tasks."""
        # Collect tasks to process this cycle
        tasks_to_process: list[UUID] = []

        while (
            not self._task_queue.empty() and len(tasks_to_process) < self.max_concurrent
        ):
            try:
                task_id = self._task_queue.get_nowait()
                tasks_to_process.append(task_id)
            except asyncio.QueueEmpty:
                break

        if not tasks_to_process:
            return

        # Process tasks concurrently
        async with concurrency.create_task_group() as tg:
            for task_id in tasks_to_process:
                tg.start_soon(self._process_task, task_id)

    async def _process_task(self, task_id: UUID) -> None:
        """Process a single task through one workflow step."""
        async with self._semaphore:
            task = self.tasks.get(task_id)
            if task is None or task.status in ("COMPLETED", "FAILED"):
                return

            # Check step limit
            if task.current_step >= task.max_steps:
                task.status = "COMPLETED"
                return

            task.status = "PROCESSING"
            task.current_step += 1

            try:
                # Get the work method and config
                method, config = self.worker._work_methods[task.function]

                # Prepare kwargs with form binding
                call_kwargs = dict(task.kwargs)
                if config.form_param_key and config.assignment:
                    form_id = call_kwargs.get(config.form_param_key)
                    if form_id and form_id in self.worker.forms:
                        form = self.worker.forms[form_id]
                        # Bind input fields from form to kwargs
                        for input_field in form.input_fields:
                            if input_field in form.available_data:
                                call_kwargs[input_field] = form.available_data[
                                    input_field
                                ]

                # Execute with optional timeout
                if config.timeout:
                    result = await asyncio.wait_for(
                        method(**call_kwargs),
                        timeout=config.timeout,
                    )
                else:
                    result = await method(**call_kwargs)

                # Record history
                task.history.append((task.function, result))
                task.result = result

                # Follow worklinks
                next_tasks = await self._follow_links(task, result)

                if next_tasks:
                    # Continue with next step(s)
                    for next_func, next_kwargs in next_tasks:
                        task.function = next_func
                        task.kwargs = next_kwargs
                        task.status = "PENDING"
                        await self._task_queue.put(task_id)
                        break  # Only follow first matching link for now
                else:
                    # No more links - task complete
                    task.status = "COMPLETED"

            except Exception as e:
                task.status = "FAILED"
                task.error = e

    async def _follow_links(
        self, task: WorkerTask, result: Any
    ) -> list[tuple[str, dict[str, Any]]]:
        """Follow worklinks from current method.

        Args:
            task: Current task
            result: Result from current method

        Returns:
            List of (next_function, kwargs) tuples for matching links
        """
        next_tasks: list[tuple[str, dict[str, Any]]] = []

        for link in self.worker.get_links_from(task.function):
            try:
                # Get the handler method from worker and call it
                handler = getattr(self.worker, link.handler_name)
                next_kwargs = await handler(result)

                # None means skip this edge
                if next_kwargs is not None:
                    next_tasks.append((link.to_, next_kwargs))

            except Exception:
                # Link handler failed - skip this edge
                continue

        return next_tasks

    def get_task(self, task_id: UUID) -> WorkerTask | None:
        """Get task by ID."""
        return self.tasks.get(task_id)

    def get_tasks_by_status(self, status: str) -> list[WorkerTask]:
        """Get all tasks with given status."""
        return [t for t in self.tasks.values() if t.status == status]

    @property
    def pending_tasks(self) -> list[WorkerTask]:
        """Tasks waiting to be processed."""
        return self.get_tasks_by_status("PENDING")

    @property
    def processing_tasks(self) -> list[WorkerTask]:
        """Tasks currently being processed."""
        return self.get_tasks_by_status("PROCESSING")

    @property
    def completed_tasks(self) -> list[WorkerTask]:
        """Tasks that completed successfully."""
        return self.get_tasks_by_status("COMPLETED")

    @property
    def failed_tasks(self) -> list[WorkerTask]:
        """Tasks that failed with errors."""
        return self.get_tasks_by_status("FAILED")

    def status_counts(self) -> dict[str, int]:
        """Count tasks by status."""
        counts: dict[str, int] = {}
        for task in self.tasks.values():
            counts[task.status] = counts.get(task.status, 0) + 1
        return counts

    def __repr__(self) -> str:
        counts = self.status_counts()
        total = len(self.tasks)
        return f"WorkerEngine(worker={self.worker.name}, tasks={total}, {counts})"
