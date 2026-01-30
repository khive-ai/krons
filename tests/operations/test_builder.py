# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.operations.builder - OperationBuilder."""

from uuid import uuid4

import pytest

from krons.core import Edge
from krons.operations.builder import Builder, OperationGraphBuilder, _resolve_branch_ref


class TestResolveBranchRef:
    """Test _resolve_branch_ref() function."""

    def test_uuid_input_returns_uuid(self):
        """Test UUID input is returned as-is."""
        test_uuid = uuid4()
        result = _resolve_branch_ref(test_uuid)
        assert result == test_uuid

    def test_object_with_id_attribute(self):
        """Test object with id attribute returns its UUID."""

        class MockBranch:
            def __init__(self):
                self.id = uuid4()

        branch = MockBranch()
        result = _resolve_branch_ref(branch)
        assert result == branch.id

    def test_uuid_string_converts_to_uuid(self):
        """Test valid UUID string is converted."""
        test_uuid = uuid4()
        uuid_str = str(test_uuid)
        result = _resolve_branch_ref(uuid_str)
        assert result == test_uuid

    def test_branch_name_string_returned_stripped(self):
        """Test non-UUID string is returned as stripped name."""
        result = _resolve_branch_ref("  main  ")
        assert result == "main"
        assert isinstance(result, str)

    def test_branch_name_without_whitespace(self):
        """Test branch name without extra whitespace."""
        result = _resolve_branch_ref("feature/test-branch")
        assert result == "feature/test-branch"
        assert isinstance(result, str)

    def test_empty_string_raises_value_error(self):
        """Test empty string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid branch reference"):
            _resolve_branch_ref("")

    def test_whitespace_only_string_raises_value_error(self):
        """Test whitespace-only string raises ValueError."""
        with pytest.raises(ValueError, match="Invalid branch reference"):
            _resolve_branch_ref("   ")

    def test_none_raises_value_error(self):
        """Test None raises ValueError."""
        with pytest.raises(ValueError, match="Invalid branch reference"):
            _resolve_branch_ref(None)

    def test_integer_raises_value_error(self):
        """Test integer raises ValueError."""
        with pytest.raises(ValueError, match="Invalid branch reference"):
            _resolve_branch_ref(123)


class TestOperationBuilder:
    """Test OperationBuilder fluent API."""

    def test_builder_add_operation(self):
        """Builder.add() should add operation to graph."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})

        assert "task1" in builder._nodes
        assert len(builder.graph.nodes) == 1

        task1 = builder.get("task1")
        assert task1.operation_type == "generate"
        assert task1.parameters == {"instruction": "First"}
        assert task1.metadata["name"] == "task1"

    def test_builder_add_duplicate_raises(self):
        """Builder.add() should raise for duplicate name."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})

        with pytest.raises(ValueError, match="already exists"):
            builder.add("task1", "generate", {"instruction": "Second"})

    def test_builder_dependencies(self):
        """Builder should handle operation dependencies."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"}, depends_on=["task1"])

        task1 = builder.get("task1")
        task2 = builder.get("task2")

        # task2 should depend on task1
        predecessors = builder.graph.get_predecessors(task2)
        assert task1 in predecessors

    def test_builder_auto_link_heads(self):
        """Builder should auto-link from current heads when depends_on is Undefined."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})  # Auto-links to task1

        task1 = builder.get("task1")
        task2 = builder.get("task2")

        # task2 should be connected to task1
        successors = builder.graph.get_successors(task1)
        assert task2 in successors

    def test_builder_empty_depends_on_no_auto_link(self):
        """Builder should NOT auto-link when depends_on is empty list."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add(
            "task2", "generate", {"instruction": "Second"}, depends_on=[]
        )  # Explicit no deps

        task1 = builder.get("task1")
        task2 = builder.get("task2")

        # task2 should NOT depend on task1
        predecessors = builder.graph.get_predecessors(task2)
        assert task1 not in predecessors

    def test_builder_context_inheritance(self):
        """Builder should pass context between operations."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add(
            "task2",
            "generate",
            {"instruction": "Second"},
            depends_on=["task1"],
            inherit_context=True,
        )

        task2 = builder.get("task2")
        assert task2.metadata.get("inherit_context") is True
        assert "primary_dependency" in task2.metadata
        assert task2.metadata["primary_dependency"] == builder._nodes["task1"].id

    def test_builder_with_branch(self):
        """Builder should handle branch parameter."""
        builder = Builder()
        branch_uuid = uuid4()
        builder.add("task1", "generate", {"instruction": "First"}, branch=branch_uuid)

        task1 = builder.get("task1")
        assert task1.metadata.get("branch") == branch_uuid

    def test_builder_with_string_branch(self):
        """Builder should handle string branch name."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"}, branch="main")

        task1 = builder.get("task1")
        assert task1.metadata.get("branch") == "main"


class TestBuilderDependsOnMethod:
    """Test Builder.depends_on() method."""

    def test_depends_on_creates_edge(self):
        """depends_on() should create dependency edge."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"}, depends_on=[])
        builder.add("task2", "generate", {"instruction": "Second"}, depends_on=[])
        builder.depends_on("task2", "task1")

        task1 = builder.get("task1")
        task2 = builder.get("task2")

        predecessors = builder.graph.get_predecessors(task2)
        assert task1 in predecessors

    def test_depends_on_target_not_found_raises(self):
        """depends_on() raises ValueError when target not found."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})

        with pytest.raises(ValueError, match="Target operation 'nonexistent' not found"):
            builder.depends_on("nonexistent", "task1")

    def test_depends_on_dependency_not_found_raises(self):
        """depends_on() raises ValueError when dependency not found."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})

        with pytest.raises(ValueError, match="Dependency operation 'nonexistent' not found"):
            builder.depends_on("task1", "nonexistent")

    def test_depends_on_with_label(self):
        """depends_on() should accept custom edge labels."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"}, depends_on=[])
        builder.add("task2", "generate", {"instruction": "Second"}, depends_on=[])
        builder.depends_on("task2", "task1", label=["custom_label"])

        task1 = builder.get("task1")
        task2 = builder.get("task2")

        # Find edge and verify label
        matching_edges = [
            e for e in builder.graph.edges if e.head == task1.id and e.tail == task2.id
        ]
        assert len(matching_edges) >= 1
        assert "custom_label" in matching_edges[0].label


class TestBuilderGetMethods:
    """Test Builder.get() and get_by_id() methods."""

    def test_get_operation(self):
        """get() should retrieve operation by name."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})

        task1 = builder.get("task1")
        assert task1.operation_type == "generate"

    def test_get_operation_not_found_raises(self):
        """get() raises ValueError when operation not found."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})

        with pytest.raises(ValueError, match="Operation 'nonexistent' not found"):
            builder.get("nonexistent")

    def test_get_by_id(self):
        """get_by_id() should retrieve operation by UUID."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})

        task1 = builder.get("task1")
        retrieved = builder.get_by_id(task1.id)
        assert retrieved is task1

    def test_get_by_id_not_found_returns_none(self):
        """get_by_id() returns None when operation not found."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})

        random_uuid = uuid4()
        result = builder.get_by_id(random_uuid)
        assert result is None


class TestBuilderExecutionTracking:
    """Test mark_executed() and get_unexecuted_nodes()."""

    def test_mark_executed(self):
        """mark_executed() should add operations to _executed set."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})

        assert len(builder._executed) == 0

        builder.mark_executed("task1")
        assert len(builder._executed) == 1
        assert builder._nodes["task1"].id in builder._executed

    def test_mark_executed_multiple(self):
        """mark_executed() should handle multiple operations."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})
        builder.add("task3", "generate", {"instruction": "Third"})

        builder.mark_executed("task1", "task2")
        assert len(builder._executed) == 2
        assert builder._nodes["task1"].id in builder._executed
        assert builder._nodes["task2"].id in builder._executed
        assert builder._nodes["task3"].id not in builder._executed

    def test_mark_executed_nonexistent_ignored(self):
        """mark_executed() should silently ignore nonexistent operations."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})

        builder.mark_executed("task1", "nonexistent")
        assert len(builder._executed) == 1

    def test_get_unexecuted_nodes(self):
        """get_unexecuted_nodes() should return only unexecuted operations."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})
        builder.add("task3", "generate", {"instruction": "Third"})

        builder.mark_executed("task1")
        unexecuted = builder.get_unexecuted_nodes()

        assert len(unexecuted) == 2
        names = [op.metadata["name"] for op in unexecuted]
        assert "task1" not in names
        assert "task2" in names
        assert "task3" in names

    def test_get_unexecuted_nodes_empty(self):
        """get_unexecuted_nodes() should return empty when all executed."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})

        builder.mark_executed("task1", "task2")
        unexecuted = builder.get_unexecuted_nodes()
        assert len(unexecuted) == 0


class TestBuilderBuildAndClear:
    """Test build() and clear() methods."""

    def test_build_returns_graph(self):
        """build() should return the operation graph."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"}, depends_on=["task1"])

        graph = builder.build()
        assert graph is builder.graph
        assert len(graph.nodes) == 2

    def test_build_validates_dag(self):
        """build() should raise if graph has cycles."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"}, depends_on=[])
        builder.add("task2", "generate", {"instruction": "Second"}, depends_on=[])

        # Create cycle manually
        task1 = builder.get("task1")
        task2 = builder.get("task2")
        builder.graph.add_edge(Edge(head=task1.id, tail=task2.id))
        builder.graph.add_edge(Edge(head=task2.id, tail=task1.id))

        with pytest.raises(ValueError, match="DAG"):
            builder.build()

    def test_clear_resets_all_state(self):
        """clear() should reset all builder state."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"})
        builder.mark_executed("task1")

        # Verify state before clear
        assert len(builder._nodes) == 2
        assert len(builder.graph.nodes) == 2
        assert len(builder._executed) == 1
        assert len(builder._current_heads) == 1

        builder.clear()

        # Verify all state reset
        assert len(builder._nodes) == 0
        assert len(builder.graph.nodes) == 0
        assert len(builder._executed) == 0
        assert len(builder._current_heads) == 0

    def test_clear_returns_self(self):
        """clear() should return self for chaining."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})

        result = builder.clear()
        assert result is builder

    def test_clear_allows_reuse(self):
        """Builder should be reusable after clear()."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.build()

        builder.clear()
        builder.add("task2", "generate", {"instruction": "Second"})
        graph = builder.build()

        assert len(graph.nodes) == 1
        assert "task2" in builder._nodes


class TestBuilderRepr:
    """Test Builder __repr__."""

    def test_repr_output(self):
        """__repr__ should show operations, edges, executed counts."""
        builder = Builder()
        builder.add("task1", "generate", {"instruction": "First"})
        builder.add("task2", "generate", {"instruction": "Second"}, depends_on=["task1"])
        builder.mark_executed("task1")

        repr_str = repr(builder)
        assert "operations=2" in repr_str
        assert "edges=1" in repr_str
        assert "executed=1" in repr_str


class TestBuilderAlias:
    """Test Builder is alias for OperationGraphBuilder."""

    def test_alias(self):
        """Builder should be OperationGraphBuilder."""
        assert Builder is OperationGraphBuilder
