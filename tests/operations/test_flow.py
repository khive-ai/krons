# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.operations.flow - OperationFlow execution."""

import pytest

from krons.core import Edge, Graph, Node
from krons.operations.flow import (
    DependencyAwareExecutor,
    OperationResult,
    flow,
    flow_stream,
)
from krons.operations.node import Operation


class TestOperationResult:
    """Test OperationResult container."""

    def test_success_result(self):
        """OperationResult.ok should be True on success."""
        result = OperationResult(
            name="test",
            result="value",
            error=None,
            completed=1,
            total=1,
        )
        assert result.success is True
        assert result.result == "value"
        assert result.error is None

    def test_error_result(self):
        """OperationResult.ok should be False on error."""
        error = RuntimeError("Test error")
        result = OperationResult(
            name="test",
            result=None,
            error=error,
            completed=1,
            total=1,
        )
        assert result.success is False
        assert result.result is None
        assert result.error is error

    def test_result_counts(self):
        """OperationResult should track completed/total counts."""
        result = OperationResult(
            name="test",
            result="value",
            error=None,
            completed=3,
            total=5,
        )
        assert result.completed == 3
        assert result.total == 5


class MockSession:
    """Mock session for testing flow execution."""

    def __init__(self, factories=None):
        self._factories = factories or {}
        self.operations = self
        self._branches = {}
        self.default_branch = None

    def get(self, op_type):
        if op_type not in self._factories:
            raise KeyError(f"Operation '{op_type}' not registered")
        return self._factories[op_type]

    def register(self, op_type, factory):
        self._factories[op_type] = factory

    def get_branch(self, ref):
        if isinstance(ref, str):
            if ref not in self._branches:
                raise ValueError(f"Branch '{ref}' not found")
            return self._branches[ref]
        return None

    def create_branch(self, name):
        class MockBranch:
            def __init__(self, name):
                from uuid import uuid4

                self.name = name
                self.id = uuid4()
                self.order = []

        branch = MockBranch(name)
        self._branches[name] = branch
        if self.default_branch is None:
            self.default_branch = branch
        return branch


class TestOperationFlow:
    """Test OperationFlow execution."""

    @pytest.mark.anyio
    async def test_run_sequential(self):
        """OperationFlow should run operations in order."""
        execution_order = []

        async def factory_a(session, branch, params):
            execution_order.append("A")
            return "result_A"

        async def factory_b(session, branch, params):
            execution_order.append("B")
            return "result_B"

        session = MockSession(
            {
                "op_a": factory_a,
                "op_b": factory_b,
            }
        )
        branch = session.create_branch("test")

        # Build graph: A -> B
        op_a = Operation(operation_type="op_a", parameters={})
        op_a.metadata["name"] = "task_a"
        op_b = Operation(operation_type="op_b", parameters={})
        op_b.metadata["name"] = "task_b"

        graph = Graph()
        graph.add_node(op_a)
        graph.add_node(op_b)
        graph.add_edge(Edge(head=op_a.id, tail=op_b.id))

        results = await flow(session, graph, branch=branch)

        # Both tasks should complete
        assert "task_a" in results
        assert "task_b" in results
        # B depends on A, so A should execute first
        assert execution_order.index("A") < execution_order.index("B")

    @pytest.mark.anyio
    async def test_run_parallel(self):
        """OperationFlow should run independent ops in parallel."""
        import asyncio

        start_times = {}
        end_times = {}

        async def factory_a(session, branch, params):
            start_times["A"] = asyncio.get_event_loop().time()
            await asyncio.sleep(0.05)
            end_times["A"] = asyncio.get_event_loop().time()
            return "result_A"

        async def factory_b(session, branch, params):
            start_times["B"] = asyncio.get_event_loop().time()
            await asyncio.sleep(0.05)
            end_times["B"] = asyncio.get_event_loop().time()
            return "result_B"

        session = MockSession(
            {
                "op_a": factory_a,
                "op_b": factory_b,
            }
        )
        branch = session.create_branch("test")

        # Build graph with independent operations (no edges)
        op_a = Operation(operation_type="op_a", parameters={})
        op_a.metadata["name"] = "task_a"
        op_b = Operation(operation_type="op_b", parameters={})
        op_b.metadata["name"] = "task_b"

        graph = Graph()
        graph.add_node(op_a)
        graph.add_node(op_b)

        results = await flow(session, graph, branch=branch)

        # Both should complete
        assert "task_a" in results
        assert "task_b" in results

        # They should overlap (start of B should be before end of A or vice versa)
        # This indicates parallel execution
        a_start, b_start = start_times["A"], start_times["B"]
        a_end, b_end = end_times["A"], end_times["B"]
        # Check overlap: (a_start < b_end) and (b_start < a_end)
        overlaps = (a_start < b_end) and (b_start < a_end)
        assert overlaps, "Operations should execute in parallel"


class TestFlowErrorHandling:
    """Test error handling in flow execution."""

    @pytest.mark.anyio
    async def test_cyclic_graph_raises(self):
        """Flow should raise for cyclic graphs."""
        session = MockSession()
        branch = session.create_branch("test")

        op_a = Operation(operation_type="op_a", parameters={})
        op_b = Operation(operation_type="op_b", parameters={})

        graph = Graph()
        graph.add_node(op_a)
        graph.add_node(op_b)
        graph.add_edge(Edge(head=op_a.id, tail=op_b.id))
        graph.add_edge(Edge(head=op_b.id, tail=op_a.id))  # Create cycle

        with pytest.raises(ValueError, match="cycle.*DAG"):
            await flow(session, graph, branch=branch)

    @pytest.mark.anyio
    async def test_non_operation_node_raises(self):
        """Flow should raise for non-Operation nodes."""
        session = MockSession()
        branch = session.create_branch("test")

        graph = Graph()
        invalid_node = Node(content={"data": "not an operation"})
        graph.add_node(invalid_node)

        with pytest.raises(ValueError, match="non-Operation node"):
            await flow(session, graph, branch=branch)

    @pytest.mark.anyio
    async def test_operation_failure_with_stop_on_error(self):
        """Failed operation with stop_on_error=True should stop execution."""

        async def failing_factory(session, branch, params):
            raise RuntimeError("Intentional failure")

        session = MockSession({"failing_op": failing_factory})
        branch = session.create_branch("test")

        op = Operation(operation_type="failing_op", parameters={})
        op.metadata["name"] = "failing_task"

        graph = Graph()
        graph.add_node(op)

        # With stop_on_error=True (default), error propagates
        results = await flow(session, graph, branch=branch, stop_on_error=True)

        # Task failed - no result
        assert "failing_task" not in results

    @pytest.mark.anyio
    async def test_operation_failure_without_stop_on_error(self):
        """Failed operation with stop_on_error=False should continue others."""
        executed = []

        async def failing_factory(session, branch, params):
            raise RuntimeError("Intentional failure")

        async def success_factory(session, branch, params):
            executed.append("success")
            return "success_result"

        session = MockSession(
            {
                "failing_op": failing_factory,
                "success_op": success_factory,
            }
        )
        branch = session.create_branch("test")

        # Two independent operations
        op_fail = Operation(operation_type="failing_op", parameters={})
        op_fail.metadata["name"] = "failing_task"
        op_success = Operation(operation_type="success_op", parameters={})
        op_success.metadata["name"] = "success_task"

        graph = Graph()
        graph.add_node(op_fail)
        graph.add_node(op_success)

        results = await flow(session, graph, branch=branch, stop_on_error=False)

        # Success task should complete
        assert "success_task" in results
        assert "failing_task" not in results
        assert "success" in executed


class TestDependencyAwareExecutor:
    """Test DependencyAwareExecutor directly."""

    @pytest.mark.anyio
    async def test_executor_preallocates_branches(self):
        """Executor should pre-allocate branches for operations."""
        session = MockSession()
        branch = session.create_branch("test")

        op = Operation(operation_type="test_op", parameters={})
        op.metadata["name"] = "task"

        graph = Graph()
        graph.add_node(op)

        executor = DependencyAwareExecutor(
            session=session,
            graph=graph,
            default_branch=branch,
        )

        await executor._preallocate_branches()

        # Operation should have branch allocated
        assert op.id in executor.operation_branches
        assert executor.operation_branches[op.id] is branch

    @pytest.mark.anyio
    async def test_executor_uses_operation_branch_metadata(self):
        """Executor should use branch from operation metadata."""
        session = MockSession()
        default_branch = session.create_branch("default")
        target_branch = session.create_branch("target")

        op = Operation(operation_type="test_op", parameters={})
        op.metadata["name"] = "task"
        op.metadata["branch"] = "target"  # Override to target branch

        graph = Graph()
        graph.add_node(op)

        executor = DependencyAwareExecutor(
            session=session,
            graph=graph,
            default_branch=default_branch,
        )

        await executor._preallocate_branches()

        # Operation should use target branch, not default
        assert executor.operation_branches[op.id] is target_branch

    @pytest.mark.anyio
    async def test_executor_missing_branch_raises(self):
        """Executor should raise if no branch allocated for operation."""
        session = MockSession()

        async def test_factory(session, branch, params):
            return "result"

        session.register("test_op", test_factory)

        op = Operation(operation_type="test_op", parameters={})
        op.metadata["name"] = "task"

        graph = Graph()
        graph.add_node(op)

        executor = DependencyAwareExecutor(
            session=session,
            graph=graph,
            default_branch=None,  # No default branch
        )

        # Force operation_branches to be empty (no pre-allocation)
        executor.operation_branches = {}

        with pytest.raises(ValueError, match="No branch allocated"):
            await executor._invoke_operation(op)


class TestFlowStream:
    """Test flow_stream for streaming execution results."""

    @pytest.mark.anyio
    async def test_stream_yields_results(self):
        """flow_stream should yield results as operations complete."""

        async def factory_a(session, branch, params):
            return "result_A"

        async def factory_b(session, branch, params):
            return "result_B"

        session = MockSession(
            {
                "op_a": factory_a,
                "op_b": factory_b,
            }
        )
        branch = session.create_branch("test")

        op_a = Operation(operation_type="op_a", parameters={})
        op_a.metadata["name"] = "task_a"
        op_b = Operation(operation_type="op_b", parameters={})
        op_b.metadata["name"] = "task_b"

        graph = Graph()
        graph.add_node(op_a)
        graph.add_node(op_b)

        results = []
        async for result in flow_stream(session, graph, branch=branch):
            results.append(result)

        assert len(results) == 2
        assert all(isinstance(r, OperationResult) for r in results)
        assert results[-1].completed == 2
        assert results[-1].total == 2

    @pytest.mark.anyio
    async def test_stream_yields_errors(self):
        """flow_stream should yield error results for failed operations."""

        async def failing_factory(session, branch, params):
            raise RuntimeError("Test error")

        session = MockSession({"failing_op": failing_factory})
        branch = session.create_branch("test")

        op = Operation(operation_type="failing_op", parameters={})
        op.metadata["name"] = "failing_task"

        graph = Graph()
        graph.add_node(op)

        results = []
        async for result in flow_stream(session, graph, branch=branch, stop_on_error=False):
            results.append(result)

        assert len(results) == 1
        assert results[0].error is not None
        assert results[0].success is False

    @pytest.mark.anyio
    async def test_stream_cyclic_raises(self):
        """flow_stream should raise for cyclic graphs."""
        session = MockSession()
        branch = session.create_branch("test")

        op_a = Operation(operation_type="op_a", parameters={})
        op_b = Operation(operation_type="op_b", parameters={})

        graph = Graph()
        graph.add_node(op_a)
        graph.add_node(op_b)
        graph.add_edge(Edge(head=op_a.id, tail=op_b.id))
        graph.add_edge(Edge(head=op_b.id, tail=op_a.id))

        with pytest.raises(ValueError, match="cycle.*DAG"):
            async for _ in flow_stream(session, graph, branch=branch):
                pass


class TestFlowMaxConcurrent:
    """Test max_concurrent parameter."""

    @pytest.mark.anyio
    async def test_max_concurrent_limits_parallelism(self):
        """max_concurrent should limit parallel execution."""
        import asyncio

        concurrent_count = 0
        max_seen = 0

        async def tracking_factory(session, branch, params):
            nonlocal concurrent_count, max_seen
            concurrent_count += 1
            max_seen = max(max_seen, concurrent_count)
            await asyncio.sleep(0.02)
            concurrent_count -= 1
            return "done"

        session = MockSession({"track": tracking_factory})
        branch = session.create_branch("test")

        # Create 5 independent operations
        ops = []
        graph = Graph()
        for i in range(5):
            op = Operation(operation_type="track", parameters={})
            op.metadata["name"] = f"task_{i}"
            ops.append(op)
            graph.add_node(op)

        await flow(session, graph, branch=branch, max_concurrent=2)

        # Should not exceed 2 concurrent
        assert max_seen <= 2


class TestFlowVerbose:
    """Test verbose logging."""

    @pytest.mark.anyio
    async def test_verbose_logging(self, caplog):
        """flow with verbose=True should log execution details."""
        import logging

        async def test_factory(session, branch, params):
            return "result"

        session = MockSession({"test_op": test_factory})
        branch = session.create_branch("test")

        op = Operation(operation_type="test_op", parameters={})
        op.metadata["name"] = "test_task"

        graph = Graph()
        graph.add_node(op)

        with caplog.at_level(logging.DEBUG, logger="krons.operations.flow"):
            await flow(session, graph, branch=branch, verbose=True)

        # Should have logged execution messages
        assert any(
            "Executing" in record.message or "Completed" in record.message
            for record in caplog.records
        )
