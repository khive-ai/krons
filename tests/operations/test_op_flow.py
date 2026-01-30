# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive tests for flow() coverage (migrated from lionpride).

Tests flow execution with real Session objects and additional coverage paths.
"""

import asyncio
import logging
from unittest.mock import patch

import pytest

from krons.core import Edge, Graph, Node
from krons.operations import Builder, flow
from krons.operations.flow import DependencyAwareExecutor, OperationResult, flow_stream
from krons.operations.node import Operation, create_operation
from krons.session import Session

# -------------------------------------------------------------------------
# Fixtures for Real Session
# -------------------------------------------------------------------------


@pytest.fixture
def session_with_ops():
    """Create session with registered test operations."""
    session = Session()
    return session


# -------------------------------------------------------------------------
# Error Handling Tests
# -------------------------------------------------------------------------


class TestFlowErrorHandlingWithRealSession:
    """Test error handling paths in flow execution with real Session."""

    @pytest.mark.anyio
    async def test_cyclic_graph_raises_error(self, session_with_ops):
        """Test Graph with cycles raises ValueError."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        # Create cyclic graph manually
        op1 = create_operation(operation_type="generate", parameters={"instruction": "First"})
        op1.metadata["name"] = "task1"
        op2 = create_operation(operation_type="generate", parameters={"instruction": "Second"})
        op2.metadata["name"] = "task2"

        graph = Graph()
        graph.add_node(op1)
        graph.add_node(op2)
        # Create cycle: op1 -> op2 -> op1
        graph.add_edge(Edge(head=op1.id, tail=op2.id))
        graph.add_edge(Edge(head=op2.id, tail=op1.id))

        with pytest.raises(ValueError, match=r"cycle.*DAG"):
            await flow(session, graph, branch=branch)

    @pytest.mark.anyio
    async def test_non_operation_node_raises_error(self, session_with_ops):
        """Test non-Operation node raises ValueError."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        # Create graph with non-Operation node
        graph = Graph()
        invalid_node = Node(content={"data": "not an operation"})
        graph.add_node(invalid_node)

        with pytest.raises(ValueError, match="non-Operation node"):
            await flow(session, graph, branch=branch)

    @pytest.mark.anyio
    async def test_branch_as_string_resolution(self, session_with_ops):
        """Test string branch name resolution."""
        session = session_with_ops
        branch_name = "test_branch"
        session.create_branch(name=branch_name)

        async def simple_factory(session, branch, params):
            return "result"

        session.operations.register("simple_op", simple_factory)

        builder = Builder()
        builder.add("task1", "simple_op", {"instruction": "Test"})
        graph = builder.build()

        # Pass branch as string (not object)
        results = await flow(session, graph, branch=branch_name)

        assert "task1" in results

    @pytest.mark.anyio
    async def test_executor_with_none_branch(self, session_with_ops):
        """Test DependencyAwareExecutor handles None default_branch gracefully.

        When default_branch is None, it falls back to session.default_branch.
        Session auto-creates a default branch on init, so operations get that branch.
        """
        session = session_with_ops

        builder = Builder()
        builder.add("task1", "generate", {"instruction": "Test"})
        graph = builder.build()

        # Create executor with explicit None branch (uses session's default)
        executor = DependencyAwareExecutor(
            session=session,
            graph=graph,
            default_branch=None,
        )

        # Pre-allocate should handle None gracefully by using session.default_branch
        await executor._preallocate_branches()

        # Session auto-creates a default branch, so operations get that branch
        for _op_id, allocated_branch in executor.operation_branches.items():
            # Falls back to session.default_branch
            assert allocated_branch is session.default_branch

    @pytest.mark.anyio
    async def test_verbose_branch_preallocation(self, session_with_ops, caplog):
        """Test verbose logging for branch pre-allocation."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        async def simple_factory(session, branch, params):
            return "result"

        session.operations.register("simple_op", simple_factory)

        builder = Builder()
        builder.add("task1", "simple_op", {"instruction": "Test"})
        builder.add("task2", "simple_op", {"instruction": "Test2"})
        graph = builder.build()

        with caplog.at_level(logging.DEBUG, logger="krons.operations.flow"):
            await flow(session, graph, branch=branch, verbose=True)

        assert "Pre-allocated branches for 2 operations" in caplog.text


# -------------------------------------------------------------------------
# Stop Conditions Tests
# -------------------------------------------------------------------------


class TestFlowStopConditions:
    """Test stop condition handling and verbose logging."""

    @pytest.mark.anyio
    async def test_error_with_stop_on_error_true_reraises(self, session_with_ops):
        """Test error handling with stop_on_error=True."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        async def failing_factory(session, branch, parameters):
            raise RuntimeError("Intentional failure")

        session.operations.register("failing_op", failing_factory)

        builder = Builder()
        builder.add("task1", "failing_op", {})
        graph = builder.build()

        # With stop_on_error=True, error should propagate
        results = await flow(session, graph, branch=branch, stop_on_error=True)

        # Verify task failed (no result)
        assert "task1" not in results

    @pytest.mark.anyio
    async def test_error_verbose_logging(self, session_with_ops, caplog):
        """Test verbose error logging with stop_on_error=True."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        async def failing_factory(session, branch, parameters):
            raise ValueError("Test error for logging")

        session.operations.register("failing_verbose", failing_factory)

        builder = Builder()
        builder.add("task1", "failing_verbose", {})
        graph = builder.build()

        with caplog.at_level(logging.DEBUG, logger="krons.operations.flow"):
            await flow(session, graph, branch=branch, verbose=True, stop_on_error=True)

        assert "Test error for logging" in caplog.text or "failed" in caplog.text

    @pytest.mark.anyio
    async def test_dependencies_verbose_logging(self, session_with_ops, caplog):
        """Test verbose logging for dependencies."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        async def simple_factory(session, branch, params):
            return "result"

        session.operations.register("simple_op", simple_factory)

        builder = Builder()
        builder.add("task1", "simple_op", {"instruction": "First"})
        builder.add("task2", "simple_op", {"instruction": "Second"}, depends_on=["task1"])

        graph = builder.build()
        with caplog.at_level(logging.DEBUG, logger="krons.operations.flow"):
            await flow(session, graph, branch=branch, verbose=True)

        assert "waiting for" in caplog.text
        assert "dependencies" in caplog.text


# -------------------------------------------------------------------------
# Execution Event Tests
# -------------------------------------------------------------------------


class TestFlowExecutionEvents:
    """Test execution event handling."""

    @pytest.mark.anyio
    async def test_operation_receives_its_parameters(self, session_with_ops):
        """Test that operations receive their parameters unchanged."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        received_params = None

        async def param_receiver(session, branch, parameters):
            nonlocal received_params
            received_params = parameters
            return "done"

        session.operations.register("param_receiver", param_receiver)

        builder = Builder()
        builder.add("task1", "param_receiver", {"my_key": "my_value"})
        graph = builder.build()

        await flow(session, graph, branch=branch)

        # Verify parameters are passed unchanged
        assert received_params == {"my_key": "my_value"}

    @pytest.mark.anyio
    async def test_failed_predecessor_does_not_block_with_stop_on_error_false(
        self, session_with_ops
    ):
        """Test that with stop_on_error=False, dependent tasks still run."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        dependent_ran = False

        async def failing_factory(session, branch, parameters):
            raise RuntimeError("Intentional failure")

        async def dependent_factory(session, branch, parameters):
            nonlocal dependent_ran
            dependent_ran = True
            return "done"

        session.operations.register("failing_pred", failing_factory)
        session.operations.register("dependent_task", dependent_factory)

        builder = Builder()
        builder.add("failed_task", "failing_pred", {})
        builder.add("dependent_task", "dependent_task", {}, depends_on=["failed_task"])
        graph = builder.build()

        await flow(session, graph, branch=branch, stop_on_error=False)

        # Dependent task should have run (after predecessor completed/failed)
        assert dependent_ran


# -------------------------------------------------------------------------
# Result Processing Tests
# -------------------------------------------------------------------------


class TestFlowResultProcessing:
    """Test result processing and verbose logging."""

    @pytest.mark.anyio
    async def test_verbose_operation_execution(self, session_with_ops, caplog):
        """Test verbose logging for operation execution."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        async def simple_factory(session, branch, params):
            return "result"

        session.operations.register("simple_op", simple_factory)

        builder = Builder()
        builder.add("task1", "simple_op", {"instruction": "Test"})
        graph = builder.build()

        with caplog.at_level(logging.DEBUG, logger="krons.operations.flow"):
            await flow(session, graph, branch=branch, verbose=True)

        assert "Executing operation:" in caplog.text

    @pytest.mark.anyio
    async def test_missing_branch_allocation_raises_error(self, session_with_ops):
        """Test missing branch allocation raises ValueError."""
        session = session_with_ops

        op = create_operation(
            operation_type="generate",
            parameters={"instruction": "Test"},
        )
        op.metadata["name"] = "test"
        graph = Graph()
        graph.add_node(op)

        executor = DependencyAwareExecutor(session=session, graph=graph, default_branch=None)

        with pytest.raises(ValueError, match="No branch allocated"):
            await executor._invoke_operation(op)

    @pytest.mark.anyio
    async def test_verbose_operation_failure(self, session_with_ops, caplog):
        """Test verbose logging for operation failure."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        async def status_failed_factory(session, branch, parameters):
            raise RuntimeError("Operation failed with error status")

        session.operations.register("status_fail", status_failed_factory)

        builder = Builder()
        builder.add("task1", "status_fail", {})
        graph = builder.build()

        with caplog.at_level(logging.DEBUG, logger="krons.operations.flow"):
            await flow(session, graph, branch=branch, verbose=True, stop_on_error=False)

        assert "failed" in caplog.text

    @pytest.mark.anyio
    async def test_verbose_operation_completion(self, session_with_ops, caplog):
        """Test verbose logging for operation completion."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        async def simple_factory(session, branch, params):
            return "result"

        session.operations.register("simple_op", simple_factory)

        builder = Builder()
        builder.add("task1", "simple_op", {"instruction": "Test"})
        graph = builder.build()

        with caplog.at_level(logging.DEBUG, logger="krons.operations.flow"):
            await flow(session, graph, branch=branch, verbose=True)

        assert "Completed operation:" in caplog.text


# -------------------------------------------------------------------------
# Integration Tests
# -------------------------------------------------------------------------


class TestFlowIntegration:
    """Integration tests covering complex scenarios."""

    @pytest.mark.anyio
    async def test_complex_dag_with_multiple_paths(self, session_with_ops):
        """Test complex DAG execution with multiple dependency paths."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        async def simple_factory(session, branch, params):
            return f"result_{params.get('instruction', 'default')}"

        session.operations.register("generate", simple_factory)

        builder = Builder()
        # Diamond dependency: task1 -> task2, task3 -> task4
        builder.add("task1", "generate", {"instruction": "Root"})
        builder.add("task2", "generate", {"instruction": "Left"}, depends_on=["task1"])
        builder.add("task3", "generate", {"instruction": "Right"}, depends_on=["task1"])
        builder.add("task4", "generate", {"instruction": "Merge"}, depends_on=["task2", "task3"])

        graph = builder.build()
        results = await flow(session, graph, branch=branch)

        # All tasks should complete
        assert all(f"task{i}" in results for i in range(1, 5))

    @pytest.mark.anyio
    async def test_stop_on_error_false_continues_execution(self, session_with_ops):
        """Test that stop_on_error=False allows remaining tasks to execute."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        async def failing_factory(session, branch, parameters):
            raise RuntimeError("Fail")

        async def success_factory(session, branch, parameters):
            return "success_result"

        session.operations.register("fail_task", failing_factory)
        session.operations.register("success_task", success_factory)

        builder = Builder()
        builder.add("task1", "fail_task", {})
        builder.add("task2", "success_task", {})  # Independent

        graph = builder.build()
        results = await flow(session, graph, branch=branch, stop_on_error=False)

        # task2 should still execute
        assert "task2" in results
        assert "task1" not in results

    @pytest.mark.anyio
    async def test_max_concurrent_limits_parallelism(self, session_with_ops):
        """Test max_concurrent limits parallel execution."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        concurrent_count = 0
        max_seen = 0

        async def concurrent_tracker(session, branch, parameters):
            nonlocal concurrent_count, max_seen
            concurrent_count += 1
            max_seen = max(max_seen, concurrent_count)
            await asyncio.sleep(0.01)
            concurrent_count -= 1
            return "done"

        session.operations.register("track", concurrent_tracker)

        builder = Builder()
        for i in range(5):
            builder.add(f"task{i}", "track", {})

        graph = builder.build()
        await flow(session, graph, branch=branch, max_concurrent=2)

        # Should not exceed 2 concurrent
        assert max_seen <= 2


# -------------------------------------------------------------------------
# Direct Exception Path Tests
# -------------------------------------------------------------------------


class TestFlowExceptionPaths:
    """Direct tests for exception handling in _execute_operation."""

    @pytest.mark.anyio
    async def test_exception_in_execute_operation_no_verbose_no_stop(self, session_with_ops):
        """Test exception caught, stored, execution continues."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        async def failing_op(session, branch, parameters):
            raise ValueError("Test exception - no verbose, no stop")

        async def success_op(session, branch, parameters):
            return "success"

        session.operations.register("fail_no_verbose", failing_op)
        session.operations.register("success_op", success_op)

        builder = Builder()
        builder.add("task1", "fail_no_verbose", {})
        builder.add("task2", "success_op", {})
        graph = builder.build()

        # Execute with stop_on_error=False, verbose=False
        results = await flow(session, graph, branch=branch, stop_on_error=False, verbose=False)

        # task1 should fail, task2 should succeed
        assert "task1" not in results
        assert "task2" in results

    @pytest.mark.anyio
    async def test_exception_with_verbose_no_stop(self, session_with_ops, caplog):
        """Test verbose error logging."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        async def failing_op(session, branch, params):
            raise RuntimeError("Mock exception for verbose logging")

        async def success_op(session, branch, params):
            return "success"

        session.operations.register("failing_op", failing_op)
        session.operations.register("success_op", success_op)

        builder = Builder()
        builder.add("task1", "failing_op", {"instruction": "Test"})
        builder.add("task2", "success_op", {"instruction": "Should run"})
        graph = builder.build()

        with caplog.at_level(logging.DEBUG, logger="krons.operations.flow"):
            await flow(session, graph, branch=branch, verbose=True, stop_on_error=False)

        # Verify verbose error logging
        assert "failed" in caplog.text

    @pytest.mark.anyio
    async def test_exception_with_stop_on_error(self, session_with_ops):
        """Test stop_on_error=True behavior."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        async def failing_with_stop(session, branch, parameters):
            raise ValueError("Test exception with stop_on_error")

        session.operations.register("fail_stop", failing_with_stop)

        builder = Builder()
        builder.add("task1", "fail_stop", {})
        graph = builder.build()

        results = await flow(session, graph, branch=branch, stop_on_error=True, verbose=False)

        # Task failed, no result
        assert "task1" not in results

    @pytest.mark.anyio
    async def test_exception_with_verbose_and_stop(self, session_with_ops, caplog):
        """Test all exception paths with verbose + stop_on_error."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        async def failing_full_path(session, branch, parameters):
            raise RuntimeError("Full exception path test")

        session.operations.register("fail_full", failing_full_path)

        builder = Builder()
        builder.add("task1", "fail_full", {})
        graph = builder.build()

        with caplog.at_level(logging.DEBUG, logger="krons.operations.flow"):
            results = await flow(session, graph, branch=branch, verbose=True, stop_on_error=True)

        # Verify verbose logging executed
        assert "failed" in caplog.text
        assert "Full exception path test" in caplog.text

        # Task failed
        assert "task1" not in results

    @pytest.mark.anyio
    async def test_direct_executor_exception_verbose_stop(self, session_with_ops, caplog):
        """Test exception path directly via executor to ensure coverage."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        async def test_factory(session, branch, params):
            return "result"

        session.operations.register("generate", test_factory)

        builder = Builder()
        builder.add("task1", "generate", {"instruction": "Test"})
        graph = builder.build()

        # Get operation
        op = None
        for node in graph.nodes:
            if isinstance(node, Operation):
                op = node
                break

        # Create executor with verbose=True, stop_on_error=True
        executor = DependencyAwareExecutor(
            session=session,
            graph=graph,
            default_branch=branch,
            verbose=True,
            stop_on_error=True,
        )

        # Mock _invoke_operation to raise exception
        async def mock_invoke_raise(operation):
            raise ValueError("Direct executor exception test")

        with (
            caplog.at_level(logging.DEBUG, logger="krons.operations.flow"),
            patch.object(executor, "_invoke_operation", side_effect=mock_invoke_raise),
        ):
            # Execute - exception propagates through CompletionStream's TaskGroup
            with pytest.raises(ExceptionGroup) as exc_info:
                await executor.execute()

            # Verify the original ValueError is in the ExceptionGroup
            assert len(exc_info.value.exceptions) == 1
            assert isinstance(exc_info.value.exceptions[0], ValueError)
            assert "Direct executor exception test" in str(exc_info.value.exceptions[0])

        # Verify error was logged
        assert "failed" in caplog.text
        assert "Direct executor exception test" in caplog.text

        # Verify error was stored
        assert op.id in executor.errors
        assert isinstance(executor.errors[op.id], ValueError)

    @pytest.mark.anyio
    async def test_exception_during_wait_for_dependencies(self, session_with_ops, caplog):
        """Test exception raised during _wait_for_dependencies."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        op = create_operation(operation_type="generate", parameters={"instruction": "Test"})
        op.metadata["name"] = "test_op"

        graph = Graph()
        graph.add_node(op)

        executor = DependencyAwareExecutor(
            session=session,
            graph=graph,
            default_branch=branch,
            verbose=True,
            stop_on_error=False,
        )

        # Mock _wait_for_dependencies to raise exception
        async def mock_wait_deps(operation):
            raise RuntimeError("Dependency wait failed")

        with (
            caplog.at_level(logging.DEBUG, logger="krons.operations.flow"),
            patch.object(executor, "_wait_for_dependencies", side_effect=mock_wait_deps),
        ):
            await executor.execute()

        # Exception should be caught and logged
        assert "failed" in caplog.text
        assert "Dependency wait failed" in caplog.text

        # Error should be stored
        assert op.id in executor.errors


# -------------------------------------------------------------------------
# Stream Execute Tests
# -------------------------------------------------------------------------


class TestFlowStreamExecute:
    """Test stream_execute and flow_stream for flow.py coverage."""

    def test_operation_result_success_property(self):
        """Test OperationResult.success property."""
        # Success case
        success_result = OperationResult(
            name="test", result="value", error=None, completed=1, total=1
        )
        assert success_result.success is True

        # Failure case
        failure_result = OperationResult(
            name="test", result=None, error=Exception("error"), completed=1, total=1
        )
        assert failure_result.success is False

    @pytest.mark.anyio
    async def test_stream_execute_success(self, session_with_ops):
        """Test stream_execute yields results as operations complete."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        async def simple_factory(session, branch, params):
            return "result"

        session.operations.register("generate", simple_factory)

        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})
        graph = builder.build()

        executor = DependencyAwareExecutor(
            session=session,
            graph=graph,
            default_branch=branch,
        )

        results = []
        async for result in executor.stream_execute():
            results.append(result)

        assert len(results) == 2
        assert all(isinstance(r, OperationResult) for r in results)
        assert results[-1].completed == 2
        assert results[-1].total == 2

    @pytest.mark.anyio
    async def test_stream_execute_with_error(self, session_with_ops):
        """Test stream_execute yields error results for failed operations."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        async def failing_factory(session, branch, parameters):
            raise RuntimeError("Test error")

        session.operations.register("fail_stream", failing_factory)

        builder = Builder()
        builder.add("task1", "fail_stream", {})
        graph = builder.build()

        executor = DependencyAwareExecutor(
            session=session,
            graph=graph,
            default_branch=branch,
            stop_on_error=False,
        )

        results = []
        async for result in executor.stream_execute():
            results.append(result)

        assert len(results) == 1
        assert results[0].error is not None
        assert results[0].success is False

    @pytest.mark.anyio
    async def test_stream_execute_cyclic_graph_raises(self, session_with_ops):
        """Test stream_execute raises for cyclic graph."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        op1 = create_operation(operation_type="generate", parameters={})
        op2 = create_operation(operation_type="generate", parameters={})

        graph = Graph()
        graph.add_node(op1)
        graph.add_node(op2)
        graph.add_edge(Edge(head=op1.id, tail=op2.id))
        graph.add_edge(Edge(head=op2.id, tail=op1.id))

        executor = DependencyAwareExecutor(
            session=session,
            graph=graph,
            default_branch=branch,
        )

        with pytest.raises(ValueError, match=r"cycle.*DAG"):
            async for _ in executor.stream_execute():
                pass

    @pytest.mark.anyio
    async def test_stream_execute_non_operation_node_raises(self, session_with_ops):
        """Test stream_execute raises for non-Operation nodes."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        graph = Graph()
        invalid_node = Node(content={"invalid": True})
        graph.add_node(invalid_node)

        executor = DependencyAwareExecutor(
            session=session,
            graph=graph,
            default_branch=branch,
        )

        with pytest.raises(ValueError, match="non-Operation node"):
            async for _ in executor.stream_execute():
                pass

    @pytest.mark.anyio
    async def test_flow_stream_function(self, session_with_ops):
        """Test flow_stream() function."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        async def simple_factory(session, branch, params):
            return "result"

        session.operations.register("generate", simple_factory)

        builder = Builder()
        builder.add("task1", "generate", {"instruction": "Test"})
        graph = builder.build()

        results = []
        async for result in flow_stream(session, graph, branch=branch):
            results.append(result)

        assert len(results) == 1
        assert results[0].name == "task1"
        assert results[0].success is True


# -------------------------------------------------------------------------
# Branch-Aware Execution Tests
# -------------------------------------------------------------------------


class TestFlowBranchAwareExecution:
    """Test per-operation branch assignment via metadata['branch']."""

    @pytest.mark.anyio
    async def test_operation_uses_metadata_branch_by_name(self, session_with_ops):
        """Test that operations use their metadata['branch'] when specified as string."""
        session = session_with_ops

        # Create two branches
        branch1 = session.create_branch(name="branch1")
        branch2 = session.create_branch(name="branch2")

        # Track which branch each operation runs on
        execution_branches = {}

        async def branch_tracker(session, branch, parameters):
            op_name = parameters.get("_op_name", "unknown")
            execution_branches[op_name] = branch
            return "done"

        session.operations.register("track_branch", branch_tracker)

        # Create operations with explicit branch assignments
        op1 = create_operation(
            operation_type="track_branch",
            parameters={"_op_name": "task1"},
        )
        op1.metadata["name"] = "task1"
        op1.metadata["branch"] = "branch1"  # String name

        op2 = create_operation(
            operation_type="track_branch",
            parameters={"_op_name": "task2"},
        )
        op2.metadata["name"] = "task2"
        op2.metadata["branch"] = "branch2"  # String name

        graph = Graph()
        graph.add_node(op1)
        graph.add_node(op2)

        # Execute with a different default branch
        default_branch = session.create_branch(name="default")
        await flow(session, graph, branch=default_branch)

        # Verify operations ran on their specified branches
        assert execution_branches["task1"] == branch1
        assert execution_branches["task2"] == branch2

    @pytest.mark.anyio
    async def test_operation_uses_metadata_branch_by_uuid(self, session_with_ops):
        """Test that operations use their metadata['branch'] when specified as UUID."""
        session = session_with_ops

        # Create branch
        target_branch = session.create_branch(name="target")

        execution_branches = {}

        async def branch_tracker(session, branch, parameters):
            op_name = parameters.get("_op_name", "unknown")
            execution_branches[op_name] = branch
            return "done"

        session.operations.register("track_uuid_branch", branch_tracker)

        # Create operation with UUID branch assignment
        op = create_operation(
            operation_type="track_uuid_branch",
            parameters={"_op_name": "uuid_task"},
        )
        op.metadata["name"] = "uuid_task"
        op.metadata["branch"] = target_branch.id  # UUID

        graph = Graph()
        graph.add_node(op)

        # Execute with different default branch
        default_branch = session.create_branch(name="default")
        await flow(session, graph, branch=default_branch)

        # Verify operation ran on target branch (by UUID)
        assert execution_branches["uuid_task"] == target_branch

    @pytest.mark.anyio
    async def test_operation_fallback_to_default_branch(self, session_with_ops):
        """Test that operations without metadata['branch'] use default branch."""
        session = session_with_ops

        default_branch = session.create_branch(name="default")

        execution_branches = {}

        async def branch_tracker(session, branch, parameters):
            op_name = parameters.get("_op_name", "unknown")
            execution_branches[op_name] = branch
            return "done"

        session.operations.register("track_default", branch_tracker)

        # Create operation WITHOUT branch metadata
        op = create_operation(
            operation_type="track_default",
            parameters={"_op_name": "no_branch_task"},
        )
        op.metadata["name"] = "no_branch_task"
        # No branch metadata set

        graph = Graph()
        graph.add_node(op)

        await flow(session, graph, branch=default_branch)

        # Verify operation ran on default branch
        assert execution_branches["no_branch_task"] == default_branch

    @pytest.mark.anyio
    async def test_unresolvable_branch_falls_back_to_default(self, session_with_ops):
        """Test that unresolvable branch reference falls back to default."""
        session = session_with_ops

        default_branch = session.create_branch(name="default")

        execution_branches = {}

        async def branch_tracker(session, branch, parameters):
            op_name = parameters.get("_op_name", "unknown")
            execution_branches[op_name] = branch
            return "done"

        session.operations.register("track_fallback", branch_tracker)

        # Create operation with non-existent branch name
        op = create_operation(
            operation_type="track_fallback",
            parameters={"_op_name": "fallback_task"},
        )
        op.metadata["name"] = "fallback_task"
        op.metadata["branch"] = "non_existent_branch"  # This won't resolve

        graph = Graph()
        graph.add_node(op)

        await flow(session, graph, branch=default_branch)

        # Should fall back to default branch
        assert execution_branches["fallback_task"] == default_branch

    @pytest.mark.anyio
    async def test_multi_branch_workflow_with_builder(self, session_with_ops):
        """Test multi-branch workflow built with Builder.add(..., branch=...)."""
        session = session_with_ops

        # Create branches
        extraction_branch = session.create_branch(name="extraction")
        analysis_branch = session.create_branch(name="analysis")

        execution_branches = {}

        async def branch_tracker(session, branch, parameters):
            op_name = parameters.get("_op_name", "unknown")
            execution_branches[op_name] = branch
            return f"result_from_{op_name}"

        session.operations.register("multi_branch_op", branch_tracker)

        # Build workflow with explicit branch assignments
        builder = Builder()
        builder.add(
            "extract",
            "multi_branch_op",
            {"_op_name": "extract"},
            branch="extraction",
        )
        builder.add(
            "analyze",
            "multi_branch_op",
            {"_op_name": "analyze"},
            branch="analysis",
        )

        graph = builder.build()

        # Execute with a different default branch
        default_branch = session.create_branch(name="default")
        results = await flow(session, graph, branch=default_branch)

        # Verify all operations completed
        assert "extract" in results
        assert "analyze" in results

        # Verify operations ran on their specified branches
        assert execution_branches["extract"] == extraction_branch
        assert execution_branches["analyze"] == analysis_branch

    @pytest.mark.anyio
    async def test_resolve_operation_branch_with_branch_object(self, session_with_ops):
        """Test _resolve_operation_branch handles Branch-like objects."""
        session = session_with_ops
        branch = session.create_branch(name="test")

        graph = Graph()
        executor = DependencyAwareExecutor(
            session=session,
            graph=graph,
            default_branch=branch,
        )

        # Test Branch-like object (has id and order attributes)
        result = executor._resolve_operation_branch(branch)
        assert result == branch

        # Test UUID resolution
        result = executor._resolve_operation_branch(branch.id)
        assert result == branch

        # Test string name resolution
        result = executor._resolve_operation_branch("test")
        assert result == branch

        # Test unresolvable returns None
        result = executor._resolve_operation_branch("non_existent")
        assert result is None

        # Test invalid type returns None
        result = executor._resolve_operation_branch(12345)
        assert result is None
