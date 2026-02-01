# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for krons.work.engine - WorkerEngine execution driver."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from krons.work.engine import WorkerEngine, WorkerTask
from krons.work.form import Form
from krons.work.worker import Worker, work, worklink

# =============================================================================
# Tests: WorkerTask
# =============================================================================


class TestWorkerTask:
    """Tests for WorkerTask dataclass."""

    def test_default_values(self):
        """WorkerTask should have sensible defaults."""
        task = WorkerTask()

        assert isinstance(task.id, UUID)
        assert task.function == ""
        assert task.kwargs == {}
        assert task.status == "PENDING"
        assert task.result is None
        assert task.error is None
        assert task.max_steps == 100
        assert task.current_step == 0
        assert task.history == []

    def test_custom_values(self):
        """WorkerTask should accept custom values."""
        custom_id = uuid4()
        task = WorkerTask(
            id=custom_id,
            function="process",
            kwargs={"input": "data"},
            status="PROCESSING",
            max_steps=50,
        )

        assert task.id == custom_id
        assert task.function == "process"
        assert task.kwargs == {"input": "data"}
        assert task.status == "PROCESSING"
        assert task.max_steps == 50


# =============================================================================
# Test Helpers: Sample Workers
# =============================================================================


class SimpleWorker(Worker):
    """Simple worker for testing."""

    name = "simple_worker"

    @work(assignment="input -> output")
    async def process(self, input_value, **kwargs):
        return input_value * 2


class ChainWorker(Worker):
    """Worker with chained methods for testing."""

    name = "chain_worker"

    @work(assignment="start -> middle")
    async def step1(self, value, **kwargs):
        return value + 1

    @work(assignment="middle -> end")
    async def step2(self, value, **kwargs):
        return value * 2

    @worklink(from_="step1", to_="step2")
    async def step1_to_step2(self, from_result):
        return {"value": from_result}


class BranchingWorker(Worker):
    """Worker with conditional branching for testing."""

    name = "branching_worker"

    @work()
    async def check(self, value, **kwargs):
        return value, value > 10

    @work()
    async def handle_high(self, value, **kwargs):
        return f"high: {value}"

    @work()
    async def handle_low(self, value, **kwargs):
        return f"low: {value}"

    @worklink(from_="check", to_="handle_high")
    async def to_high(self, from_result):
        value, is_high = from_result
        if is_high:
            return {"value": value}
        return None  # Skip

    @worklink(from_="check", to_="handle_low")
    async def to_low(self, from_result):
        value, is_high = from_result
        if not is_high:
            return {"value": value}
        return None  # Skip


class FailingWorker(Worker):
    """Worker that can fail for testing."""

    name = "failing_worker"

    @work()
    async def will_fail(self, **kwargs):
        raise ValueError("Intentional test failure")

    @work()
    async def will_succeed(self, **kwargs):
        return "success"


class TimeoutWorker(Worker):
    """Worker with timeout behavior for testing."""

    name = "timeout_worker"

    @work(timeout=0.01)  # Very short timeout
    async def slow_method(self, **kwargs):
        import anyio

        await anyio.sleep(1.0)  # Much longer than timeout
        return "should not reach"


# =============================================================================
# Tests: WorkerEngine Creation
# =============================================================================


class TestWorkerEngineCreation:
    """Tests for WorkerEngine instantiation."""

    def test_basic_initialization(self):
        """WorkerEngine should initialize with worker."""
        worker = SimpleWorker()
        engine = WorkerEngine(worker=worker)

        assert engine.worker is worker
        assert engine.refresh_time == 0.1
        assert engine.max_concurrent == 10
        assert engine.tasks == {}

    def test_custom_refresh_time(self):
        """WorkerEngine should accept custom refresh_time."""
        worker = SimpleWorker()
        engine = WorkerEngine(worker=worker, refresh_time=0.5)

        assert engine.refresh_time == 0.5

    def test_custom_max_concurrent(self):
        """WorkerEngine should accept custom max_concurrent."""
        worker = SimpleWorker()
        engine = WorkerEngine(worker=worker, max_concurrent=5)

        assert engine.max_concurrent == 5


# =============================================================================
# Tests: WorkerEngine.add_task
# =============================================================================


class TestWorkerEngineAddTask:
    """Tests for WorkerEngine.add_task."""

    @pytest.mark.anyio
    async def test_add_task_creates_task(self):
        """add_task should create and store task."""
        worker = SimpleWorker()
        engine = WorkerEngine(worker=worker)

        task = await engine.add_task(
            task_function="process",
            input_value=5,
        )

        assert isinstance(task, WorkerTask)
        assert task.function == "process"
        assert task.kwargs == {"input_value": 5}
        assert task.id in engine.tasks

    @pytest.mark.anyio
    async def test_add_task_custom_max_steps(self):
        """add_task should accept custom max_steps."""
        worker = SimpleWorker()
        engine = WorkerEngine(worker=worker)

        task = await engine.add_task(
            task_function="process",
            task_max_steps=20,
            input_value=5,
        )

        assert task.max_steps == 20

    @pytest.mark.anyio
    async def test_add_task_invalid_function_raises(self):
        """add_task should raise for unknown function."""
        worker = SimpleWorker()
        engine = WorkerEngine(worker=worker)

        with pytest.raises(ValueError, match="not found"):
            await engine.add_task(task_function="unknown_method")

    @pytest.mark.anyio
    async def test_add_task_queues_task(self):
        """add_task should queue task for execution."""
        worker = SimpleWorker()
        engine = WorkerEngine(worker=worker)

        task = await engine.add_task(
            task_function="process",
            input_value=5,
        )

        assert not engine._task_queue.empty()


# =============================================================================
# Tests: WorkerEngine.execute
# =============================================================================


class TestWorkerEngineExecute:
    """Tests for WorkerEngine.execute."""

    @pytest.mark.anyio
    async def test_execute_simple_task(self):
        """execute should process simple task to completion."""
        worker = SimpleWorker()
        engine = WorkerEngine(worker=worker, refresh_time=0.01)

        task = await engine.add_task(
            task_function="process",
            input_value=5,
        )

        await engine.execute()

        assert task.status == "COMPLETED"
        assert task.result == 10  # 5 * 2

    @pytest.mark.anyio
    async def test_execute_chained_tasks(self):
        """execute should follow worklinks in chain."""
        worker = ChainWorker()
        engine = WorkerEngine(worker=worker, refresh_time=0.01)

        task = await engine.add_task(
            task_function="step1",
            value=5,
        )

        await engine.execute()

        assert task.status == "COMPLETED"
        # step1: 5 + 1 = 6, step2: 6 * 2 = 12
        assert task.result == 12

    @pytest.mark.anyio
    async def test_execute_records_history(self):
        """execute should record execution history."""
        worker = ChainWorker()
        engine = WorkerEngine(worker=worker, refresh_time=0.01)

        task = await engine.add_task(
            task_function="step1",
            value=5,
        )

        await engine.execute()

        assert len(task.history) == 2
        assert task.history[0][0] == "step1"
        assert task.history[1][0] == "step2"

    @pytest.mark.anyio
    async def test_execute_branching_high(self):
        """execute should follow correct branch (high value)."""
        worker = BranchingWorker()
        engine = WorkerEngine(worker=worker, refresh_time=0.01)

        task = await engine.add_task(
            task_function="check",
            value=15,  # > 10, should go to handle_high
        )

        await engine.execute()

        assert task.status == "COMPLETED"
        assert task.result == "high: 15"

    @pytest.mark.anyio
    async def test_execute_branching_low(self):
        """execute should follow correct branch (low value)."""
        worker = BranchingWorker()
        engine = WorkerEngine(worker=worker, refresh_time=0.01)

        task = await engine.add_task(
            task_function="check",
            value=5,  # <= 10, should go to handle_low
        )

        await engine.execute()

        assert task.status == "COMPLETED"
        assert task.result == "low: 5"

    @pytest.mark.anyio
    async def test_execute_handles_failure(self):
        """execute should handle task failure gracefully."""
        worker = FailingWorker()
        engine = WorkerEngine(worker=worker, refresh_time=0.01)

        task = await engine.add_task(task_function="will_fail")

        await engine.execute()

        assert task.status == "FAILED"
        assert task.error is not None
        assert "Intentional test failure" in str(task.error)

    @pytest.mark.anyio
    async def test_execute_success_after_failure(self):
        """execute should process multiple tasks independently."""
        worker = FailingWorker()
        engine = WorkerEngine(worker=worker, refresh_time=0.01)

        fail_task = await engine.add_task(task_function="will_fail")
        success_task = await engine.add_task(task_function="will_succeed")

        await engine.execute()

        assert fail_task.status == "FAILED"
        assert success_task.status == "COMPLETED"
        assert success_task.result == "success"

    @pytest.mark.anyio
    async def test_execute_max_steps_limit(self):
        """execute should stop at max_steps limit."""

        class LoopingWorker(Worker):
            @work()
            async def loop(self, count=0, **kwargs):
                return count + 1

            @worklink(from_="loop", to_="loop")
            async def loop_to_loop(self, from_result):
                return {"count": from_result}  # Always continue

        worker = LoopingWorker()
        engine = WorkerEngine(worker=worker, refresh_time=0.001)

        task = await engine.add_task(
            task_function="loop",
            task_max_steps=5,
            count=0,
        )

        await engine.execute()

        assert task.status == "COMPLETED"
        assert task.current_step == 5  # Stopped at max

    @pytest.mark.anyio
    async def test_execute_empty_queue(self):
        """execute should return immediately with empty queue."""
        worker = SimpleWorker()
        engine = WorkerEngine(worker=worker, refresh_time=0.01)

        # Don't add any tasks
        await engine.execute()

        # Should not hang or error


# =============================================================================
# Tests: WorkerEngine.stop
# =============================================================================


class TestWorkerEngineStop:
    """Tests for WorkerEngine.stop."""

    @pytest.mark.anyio
    async def test_stop_sets_flag(self):
        """stop should set stopped flag."""
        worker = SimpleWorker()
        engine = WorkerEngine(worker=worker)

        assert engine._stopped is False

        await engine.stop()

        assert engine._stopped is True

    @pytest.mark.anyio
    async def test_stop_stops_worker(self):
        """stop should stop the worker too."""
        worker = SimpleWorker()
        engine = WorkerEngine(worker=worker)

        await engine.stop()

        assert worker.is_stopped() is True


# =============================================================================
# Tests: WorkerEngine Task Queries
# =============================================================================


class TestWorkerEngineTaskQueries:
    """Tests for WorkerEngine task query methods."""

    @pytest.mark.anyio
    async def test_get_task(self):
        """get_task should return task by ID."""
        worker = SimpleWorker()
        engine = WorkerEngine(worker=worker)

        task = await engine.add_task(
            task_function="process",
            input_value=5,
        )

        retrieved = engine.get_task(task.id)

        assert retrieved is task

    @pytest.mark.anyio
    async def test_get_task_not_found(self):
        """get_task should return None for unknown ID."""
        worker = SimpleWorker()
        engine = WorkerEngine(worker=worker)

        result = engine.get_task(uuid4())

        assert result is None

    @pytest.mark.anyio
    async def test_get_tasks_by_status(self):
        """get_tasks_by_status should filter by status."""
        worker = SimpleWorker()
        engine = WorkerEngine(worker=worker, refresh_time=0.01)

        task1 = await engine.add_task(task_function="process", input_value=1)
        task2 = await engine.add_task(task_function="process", input_value=2)

        await engine.execute()

        completed = engine.get_tasks_by_status("COMPLETED")

        assert len(completed) == 2
        assert task1 in completed
        assert task2 in completed

    @pytest.mark.anyio
    async def test_pending_tasks_property(self):
        """pending_tasks should return PENDING tasks."""
        worker = SimpleWorker()
        engine = WorkerEngine(worker=worker)

        task = await engine.add_task(task_function="process", input_value=5)

        pending = engine.pending_tasks

        assert len(pending) == 1
        assert pending[0] is task

    @pytest.mark.anyio
    async def test_completed_tasks_property(self):
        """completed_tasks should return COMPLETED tasks."""
        worker = SimpleWorker()
        engine = WorkerEngine(worker=worker, refresh_time=0.01)

        task = await engine.add_task(task_function="process", input_value=5)
        await engine.execute()

        completed = engine.completed_tasks

        assert len(completed) == 1
        assert completed[0] is task

    @pytest.mark.anyio
    async def test_failed_tasks_property(self):
        """failed_tasks should return FAILED tasks."""
        worker = FailingWorker()
        engine = WorkerEngine(worker=worker, refresh_time=0.01)

        task = await engine.add_task(task_function="will_fail")
        await engine.execute()

        failed = engine.failed_tasks

        assert len(failed) == 1
        assert failed[0] is task

    @pytest.mark.anyio
    async def test_status_counts(self):
        """status_counts should return count by status."""
        worker = FailingWorker()
        engine = WorkerEngine(worker=worker, refresh_time=0.01)

        await engine.add_task(task_function="will_fail")
        await engine.add_task(task_function="will_succeed")
        await engine.add_task(task_function="will_succeed")

        await engine.execute()

        counts = engine.status_counts()

        assert counts.get("COMPLETED", 0) == 2
        assert counts.get("FAILED", 0) == 1


# =============================================================================
# Tests: WorkerEngine.__repr__
# =============================================================================


class TestWorkerEngineRepr:
    """Tests for WorkerEngine string representation."""

    @pytest.mark.anyio
    async def test_repr_includes_worker_name(self):
        """repr should include worker name."""
        worker = SimpleWorker()
        engine = WorkerEngine(worker=worker)

        repr_str = repr(engine)

        assert "simple_worker" in repr_str

    @pytest.mark.anyio
    async def test_repr_includes_task_count(self):
        """repr should include task count."""
        worker = SimpleWorker()
        engine = WorkerEngine(worker=worker)

        await engine.add_task(task_function="process", input_value=1)
        await engine.add_task(task_function="process", input_value=2)

        repr_str = repr(engine)

        assert "tasks=2" in repr_str


# =============================================================================
# Tests: WorkerEngine with Form Binding
# =============================================================================


class TestWorkerEngineFormBinding:
    """Tests for WorkerEngine with form parameter binding."""

    @pytest.mark.anyio
    async def test_form_binding_injects_fields(self):
        """Engine should inject form fields into kwargs."""

        class FormWorker(Worker):
            @work(assignment="context, instruction -> code", form_param_key="form_id")
            async def write_code(self, form_id, **kwargs):
                # Should receive context and instruction from form
                return {
                    "form_id": form_id,
                    "context": kwargs.get("context"),
                    "instruction": kwargs.get("instruction"),
                }

        worker = FormWorker()
        engine = WorkerEngine(worker=worker, refresh_time=0.01)

        # Create and store a form
        form_id = "test_form"
        form = Form(
            assignment="context, instruction -> code",
            available_data={
                "context": "existing code",
                "instruction": "add logging",
            },
        )
        worker.forms[form_id] = form

        task = await engine.add_task(
            task_function="write_code",
            form_id=form_id,
        )

        await engine.execute()

        assert task.status == "COMPLETED"
        assert task.result["context"] == "existing code"
        assert task.result["instruction"] == "add logging"


# =============================================================================
# Tests: WorkerEngine execute_lasting
# =============================================================================


class TestWorkerEngineExecuteLasting:
    """Tests for WorkerEngine.execute_lasting."""

    @pytest.mark.anyio
    async def test_execute_lasting_runs_until_stopped(self):
        """execute_lasting should run until stop() called."""
        import anyio

        worker = SimpleWorker()
        engine = WorkerEngine(worker=worker, refresh_time=0.01)

        task = await engine.add_task(task_function="process", input_value=5)

        # Run execute_lasting in background
        async with anyio.create_task_group() as tg:
            tg.start_soon(engine.execute_lasting)

            # Wait for task to complete
            while task.status != "COMPLETED":
                await anyio.sleep(0.01)

            # Stop the engine
            await engine.stop()

        assert task.status == "COMPLETED"
        assert task.result == 10


# =============================================================================
# Tests: WorkerEngine Concurrent Execution
# =============================================================================


class TestWorkerEngineConcurrent:
    """Tests for WorkerEngine concurrent execution."""

    @pytest.mark.anyio
    async def test_concurrent_task_execution(self):
        """Engine should execute multiple tasks concurrently."""
        import anyio

        class SlowWorker(Worker):
            @work()
            async def slow_process(self, value, **kwargs):
                await anyio.sleep(0.05)
                return value * 2

        worker = SlowWorker()
        engine = WorkerEngine(worker=worker, refresh_time=0.01, max_concurrent=5)

        # Add multiple tasks
        tasks = []
        for i in range(5):
            task = await engine.add_task(
                task_function="slow_process",
                value=i,
            )
            tasks.append(task)

        # Execute all
        await engine.execute()

        # All should be completed
        for task in tasks:
            assert task.status == "COMPLETED"

    @pytest.mark.anyio
    async def test_max_concurrent_limit(self):
        """Engine should respect max_concurrent limit."""

        class TrackingWorker(Worker):
            concurrent_count = 0
            max_seen = 0

            @work()
            async def tracked_process(self, value, **kwargs):
                import anyio

                TrackingWorker.concurrent_count += 1
                if TrackingWorker.concurrent_count > TrackingWorker.max_seen:
                    TrackingWorker.max_seen = TrackingWorker.concurrent_count

                await anyio.sleep(0.02)

                TrackingWorker.concurrent_count -= 1
                return value

        worker = TrackingWorker()
        engine = WorkerEngine(worker=worker, refresh_time=0.005, max_concurrent=3)

        # Add more tasks than max_concurrent
        for i in range(10):
            await engine.add_task(task_function="tracked_process", value=i)

        await engine.execute()

        # Should never exceed max_concurrent
        assert TrackingWorker.max_seen <= 3


# =============================================================================
# Tests: WorkerEngine Edge Cases
# =============================================================================


class TestWorkerEngineEdgeCases:
    """Tests for WorkerEngine edge cases."""

    @pytest.mark.anyio
    async def test_link_handler_failure_skips_edge(self):
        """Engine should skip edge if link handler fails."""

        class BrokenLinkWorker(Worker):
            @work()
            async def step1(self, **kwargs):
                return "step1_result"

            @work()
            async def step2(self, **kwargs):
                return "step2_result"

            @worklink(from_="step1", to_="step2")
            async def broken_link(self, from_result):
                raise ValueError("Link handler error")

        worker = BrokenLinkWorker()
        engine = WorkerEngine(worker=worker, refresh_time=0.01)

        task = await engine.add_task(task_function="step1")

        await engine.execute()

        # Should complete (skipping broken link)
        assert task.status == "COMPLETED"
        assert task.result == "step1_result"

    @pytest.mark.anyio
    async def test_no_links_task_completes(self):
        """Task with no outgoing links should complete after method."""
        worker = SimpleWorker()
        engine = WorkerEngine(worker=worker, refresh_time=0.01)

        task = await engine.add_task(
            task_function="process",
            input_value=5,
        )

        await engine.execute()

        assert task.status == "COMPLETED"
        assert len(task.history) == 1

    @pytest.mark.anyio
    async def test_all_links_return_none(self):
        """Task should complete when all links return None."""

        class ConditionalWorker(Worker):
            @work()
            async def check(self, value, **kwargs):
                return value

            @work()
            async def branch_a(self, **kwargs):
                return "a"

            @work()
            async def branch_b(self, **kwargs):
                return "b"

            @worklink(from_="check", to_="branch_a")
            async def to_a(self, from_result):
                if from_result > 100:
                    return {}
                return None

            @worklink(from_="check", to_="branch_b")
            async def to_b(self, from_result):
                if from_result < -100:
                    return {}
                return None

        worker = ConditionalWorker()
        engine = WorkerEngine(worker=worker, refresh_time=0.01)

        # Value 50 triggers neither branch
        task = await engine.add_task(task_function="check", value=50)

        await engine.execute()

        # Should complete without following any link
        assert task.status == "COMPLETED"
        assert task.result == 50
        assert len(task.history) == 1
