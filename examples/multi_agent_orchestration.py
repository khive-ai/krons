# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Multi-Agent Orchestration: Supervisor-Worker pattern with Exchange and Claude Code.

Demonstrates combining Exchange (message passing) with Worker (task execution)
for a multi-agent system using Claude Code for actual task execution:

Architecture:
    Supervisor (Element)
      - sends tasks via Exchange (channel="task")
      - receives results via Exchange (channel="result")

    Workers (Element + Worker internals)
      - receive tasks from inbox
      - execute using Claude Code via @work methods
      - send results back to supervisor

Usage:
    uv run python examples/multi_agent_orchestration.py
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

import anyio
from pydantic import PrivateAttr

from krons.agent.third_party.claude_code import (
    ClaudeCodeRequest,
    ClaudeSession,
    stream_claude_code_cli,
)
from krons.core import Element
from krons.session import Exchange
from krons.work import Worker, work

MODEL = "sonnet"


async def call_claude(
    prompt: str,
    ws: str,
    system_prompt: str | None = None,
    max_turns: int = 1,
) -> ClaudeSession:
    """Call Claude Code and return the session."""
    request = ClaudeCodeRequest(
        prompt=prompt,
        system_prompt=system_prompt,
        model=MODEL,
        ws=ws,
        max_turns=max_turns,
        permission_mode="bypassPermissions",
        verbose=False,
    )

    session = ClaudeSession()
    async for chunk in stream_claude_code_cli(request, session):
        if isinstance(chunk, ClaudeSession):
            return chunk

    return session


@dataclass
class TaskSpec:
    """Specification for a task to be executed by a worker."""

    task_id: str
    task_type: str  # "explain", "analyze", "summarize"
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    """Result from a worker's task execution."""

    task_id: str
    worker_id: str
    success: bool
    output: str = ""
    error: str | None = None
    duration_ms: float = 0.0
    cost: float = 0.0


class AIWorker(Worker):
    """Worker that executes AI tasks via Claude Code."""

    name = "ai_worker"

    def __init__(self, worker_id: str) -> None:
        super().__init__()
        self.worker_id = worker_id
        self.workspace = f".khive/examples/multi_agent/{worker_id}"

    @work(assignment="topic -> explanation", capacity=1, timeout=120.0)
    async def explain(self, topic: str, **kwargs) -> dict:
        """Explain a topic via Claude Code."""
        prompt = f"""Provide a clear, concise explanation of: {topic}

Requirements:
- Be clear and educational
- Use examples where helpful
- Keep it under 200 words"""

        session = await call_claude(
            prompt=prompt,
            ws=self.workspace,
            system_prompt="You are an expert educator. Explain concepts clearly.",
        )

        return {
            "explanation": session.result or "Unable to explain",
            "cost": session.total_cost_usd or 0,
        }

    @work(assignment="text -> analysis", capacity=1, timeout=120.0)
    async def analyze(self, text: str, **kwargs) -> dict:
        """Analyze text via Claude Code."""
        prompt = f"""Analyze the following text:

{text[:1000]}

Provide:
1. Main themes
2. Key points
3. Tone/style assessment

Keep analysis concise."""

        session = await call_claude(
            prompt=prompt,
            ws=self.workspace,
            system_prompt="You are an analytical expert. Provide insightful analysis.",
        )

        return {
            "analysis": session.result or "Unable to analyze",
            "cost": session.total_cost_usd or 0,
        }

    @work(assignment="content -> summary", capacity=1, timeout=120.0)
    async def summarize(self, content: str, **kwargs) -> dict:
        """Summarize content via Claude Code."""
        prompt = f"""Summarize the following content in 2-3 sentences:

{content[:1500]}

Be concise and capture the key points."""

        session = await call_claude(
            prompt=prompt,
            ws=self.workspace,
            system_prompt="You are an expert summarizer. Be concise and accurate.",
        )

        return {
            "summary": session.result or "Unable to summarize",
            "cost": session.total_cost_usd or 0,
        }


class WorkerAgent(Element):
    """Agent that combines Exchange messaging with Worker task execution."""

    worker: AIWorker = None  # type: ignore
    _exchange: Exchange | None = PrivateAttr(default=None)
    _supervisor_id: UUID | None = PrivateAttr(default=None)
    _stopped: bool = PrivateAttr(default=False)

    def __init__(self, name: str, **kwargs):
        super().__init__(**kwargs)
        self.worker = AIWorker(worker_id=name)

    def bind(self, exchange: Exchange, supervisor_id: UUID) -> None:
        """Bind agent to exchange and supervisor."""
        self._exchange = exchange
        self._supervisor_id = supervisor_id
        exchange.register(self.id)

    async def run(self) -> None:
        """Run the worker agent loop."""
        if not self._exchange or not self._supervisor_id:
            raise RuntimeError("Agent not bound to exchange")

        print(f"[{self.worker.worker_id}] Started")
        self._stopped = False

        while not self._stopped:
            # Check for tasks
            msg = self._exchange.pop_message(self.id, sender=self._supervisor_id)
            if msg and msg.channel == "task":
                spec = TaskSpec(**msg.content)
                await self._execute_task(spec)

            await asyncio.sleep(0.1)

        print(f"[{self.worker.worker_id}] Stopped")

    async def _execute_task(self, spec: TaskSpec) -> None:
        """Execute a task and send result back to supervisor."""
        print(f"[{self.worker.worker_id}] Executing: {spec.task_id}")
        start = time.time()

        try:
            # Route to appropriate worker method
            if spec.task_type == "explain":
                output = await self.worker.explain(spec.payload.get("topic", ""))
            elif spec.task_type == "analyze":
                output = await self.worker.analyze(spec.payload.get("text", ""))
            elif spec.task_type == "summarize":
                output = await self.worker.summarize(spec.payload.get("content", ""))
            else:
                output = {"error": f"Unknown task type: {spec.task_type}"}

            result = TaskResult(
                task_id=spec.task_id,
                worker_id=self.worker.worker_id,
                success=True,
                output=str(
                    output.get("explanation")
                    or output.get("analysis")
                    or output.get("summary", "")
                ),
                duration_ms=(time.time() - start) * 1000,
                cost=output.get("cost", 0),
            )

        except Exception as e:
            result = TaskResult(
                task_id=spec.task_id,
                worker_id=self.worker.worker_id,
                success=False,
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

        # Send result back
        self._exchange.send(
            self.id,
            recipient=self._supervisor_id,
            content=result.__dict__,
            channel="result",
        )
        await self._exchange.sync()

        status = "SUCCESS" if result.success else "FAILED"
        print(
            f"[{self.worker.worker_id}] {spec.task_id} {status} ({result.duration_ms:.0f}ms)"
        )

    def stop(self) -> None:
        """Stop the worker."""
        self._stopped = True


class Supervisor(Element):
    """Supervisor that dispatches tasks and collects results."""

    _exchange: Exchange | None = PrivateAttr(default=None)
    _workers: list[WorkerAgent] = PrivateAttr(default_factory=list)
    _results: dict[str, TaskResult] = PrivateAttr(default_factory=dict)
    _pending: set[str] = PrivateAttr(default_factory=set)
    _stopped: bool = PrivateAttr(default=False)

    def bind(self, exchange: Exchange, workers: list[WorkerAgent]) -> None:
        """Bind supervisor to exchange and workers."""
        self._exchange = exchange
        self._workers = workers
        exchange.register(self.id)

    def dispatch(self, spec: TaskSpec, worker_idx: int | None = None) -> None:
        """Dispatch task to a worker."""
        if not self._exchange:
            raise RuntimeError("Supervisor not bound")

        # Round-robin if no specific worker
        if worker_idx is None:
            worker_idx = len(self._pending) % len(self._workers)

        worker = self._workers[worker_idx]
        self._exchange.send(
            self.id,
            recipient=worker.id,
            content=spec.__dict__,
            channel="task",
        )
        self._pending.add(spec.task_id)
        print(f"[supervisor] Dispatched {spec.task_id} to {worker.worker.worker_id}")

    async def collect(self) -> None:
        """Collect results from workers."""
        if not self._exchange:
            return

        print("[supervisor] Collecting results...")
        self._stopped = False

        while not self._stopped and self._pending:
            await self._exchange.sync()

            for worker in self._workers:
                msg = self._exchange.pop_message(self.id, sender=worker.id)
                if msg and msg.channel == "result":
                    result = TaskResult(**msg.content)
                    self._results[result.task_id] = result
                    self._pending.discard(result.task_id)
                    print(f"[supervisor] Collected {result.task_id}")

            await asyncio.sleep(0.1)

        print("[supervisor] Collection complete")

    def stop(self) -> None:
        """Stop collection."""
        self._stopped = True

    @property
    def results(self) -> dict[str, TaskResult]:
        """Get all results."""
        return dict(self._results)

    def summary(self) -> dict:
        """Get execution summary."""
        total_cost = sum(r.cost for r in self._results.values())
        return {
            "tasks_completed": len(self._results),
            "tasks_pending": len(self._pending),
            "successes": sum(1 for r in self._results.values() if r.success),
            "failures": sum(1 for r in self._results.values() if not r.success),
            "total_cost_usd": total_cost,
        }


async def main():
    """Run the multi-agent orchestration with Claude Code."""
    print("=" * 60)
    print("Multi-Agent Orchestration - Claude Code Integration")
    print("=" * 60)
    print()

    # Create exchange
    exchange = Exchange()

    # Create workers
    workers = [
        WorkerAgent(name="worker-1"),
        WorkerAgent(name="worker-2"),
    ]

    # Create supervisor
    supervisor = Supervisor()
    supervisor.bind(exchange, workers)

    # Bind workers
    for worker in workers:
        worker.bind(exchange, supervisor.id)

    # Define tasks
    tasks = [
        TaskSpec("task-1", "explain", {"topic": "async/await in Python"}),
        TaskSpec(
            "task-2", "explain", {"topic": "the actor model in distributed systems"}
        ),
        TaskSpec(
            "task-3",
            "summarize",
            {
                "content": "Python's asyncio library provides infrastructure for writing single-threaded concurrent code using coroutines, multiplexing I/O access over sockets and other resources, running network clients and servers, and other related primitives."
            },
        ),
    ]

    print(f"Tasks: {len(tasks)}")
    print(f"Workers: {len(workers)}")
    print()

    # Dispatch tasks
    print("Dispatching tasks...")
    for i, task in enumerate(tasks):
        supervisor.dispatch(task, worker_idx=i % len(workers))
    await exchange.sync()
    print()

    # Start workers and collector
    async with anyio.create_task_group() as tg:
        for worker in workers:
            tg.start_soon(worker.run)

        tg.start_soon(supervisor.collect)

        # Wait for completion
        while supervisor._pending:
            await asyncio.sleep(0.5)

        # Stop all
        for worker in workers:
            worker.stop()
        supervisor.stop()

    # Results
    print()
    print("=" * 60)
    print("Results")
    print("=" * 60)

    summary = supervisor.summary()
    print(f"Completed: {summary['successes']}/{summary['tasks_completed']}")
    print(f"Total cost: ${summary['total_cost_usd']:.4f}")
    print()

    for task_id, result in supervisor.results.items():
        print(f"{task_id} ({result.worker_id}):")
        if result.success:
            print(f"  {result.output[:100]}...")
        else:
            print(f"  ERROR: {result.error}")
        print()


if __name__ == "__main__":
    anyio.run(main)
