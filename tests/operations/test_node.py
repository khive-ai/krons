# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.operations.node - Operation event."""

import pytest
from pydantic import ValidationError

from krons.operations.node import Operation, create_operation
from krons.types import Undefined


class TestOperationCreation:
    """Test Operation instantiation."""

    def test_operation_type(self):
        """Operation should have operation_type."""
        op = Operation(operation_type="generate", parameters={"instruction": "Test"})
        assert op.operation_type == "generate"
        assert op.parameters == {"instruction": "Test"}

    def test_operation_parameters(self):
        """Operation should accept parameters dict."""
        params = {"instruction": "Hello", "model": "test_model", "extra": 42}
        op = Operation(operation_type="communicate", parameters=params)
        assert op.parameters == params
        assert op.parameters["instruction"] == "Hello"

    def test_operation_default_parameters(self):
        """Operation should have empty dict as default parameters."""
        op = Operation(operation_type="operate")
        assert op.parameters == {}

    def test_operation_metadata(self):
        """Operation should accept metadata."""
        op = Operation(
            operation_type="generate",
            parameters={},
            metadata={"name": "test_op", "priority": "high"},
        )
        assert op.metadata["name"] == "test_op"
        assert op.metadata["priority"] == "high"


class TestOperationBinding:
    """Test Operation session/branch binding."""

    def test_bind(self):
        """Operation.bind() should set session and branch."""

        class MockSession:
            pass

        class MockBranch:
            pass

        session = MockSession()
        branch = MockBranch()

        op = Operation(operation_type="generate", parameters={})
        result = op.bind(session, branch)

        # bind() returns self for chaining
        assert result is op
        assert op._session is session
        assert op._branch is branch

    def test_unbound_raises(self):
        """Unbound Operation should raise on invoke via _require_binding."""
        op = Operation(operation_type="generate", parameters={})

        with pytest.raises(RuntimeError, match="not bound"):
            op._require_binding()

    def test_partial_binding_raises(self):
        """Partially bound Operation should raise."""

        class MockSession:
            pass

        op = Operation(operation_type="generate", parameters={})
        op._session = MockSession()
        # _branch is still None

        with pytest.raises(RuntimeError, match="not bound"):
            op._require_binding()


class TestOperationInvoke:
    """Test Operation execution."""

    @pytest.mark.anyio
    async def test_invoke(self):
        """Operation.invoke() should execute handler."""

        class MockRegistry:
            def get(self, op_type):
                async def factory(session, branch, params):
                    return f"result_{params.get('value', 'default')}"

                return factory

        class MockSession:
            operations = MockRegistry()

        class MockBranch:
            pass

        op = Operation(operation_type="test_op", parameters={"value": "42"})
        op.bind(MockSession(), MockBranch())

        result = await op._invoke()
        assert result == "result_42"

    @pytest.mark.anyio
    async def test_invoke_unbound_raises(self):
        """Unbound Operation._invoke() should raise."""
        op = Operation(operation_type="generate", parameters={})

        with pytest.raises(RuntimeError, match="not bound"):
            await op._invoke()


class TestOperationRepr:
    """Test Operation string representation."""

    def test_repr_unbound(self):
        """Operation repr shows unbound state."""
        op = Operation(operation_type="operate", parameters={})
        repr_str = repr(op)

        assert "operate" in repr_str
        assert "unbound" in repr_str
        assert "pending" in repr_str  # Default status

    def test_repr_bound(self):
        """Operation repr shows bound state."""

        class MockSession:
            pass

        class MockBranch:
            pass

        op = Operation(operation_type="generate", parameters={})
        op.bind(MockSession(), MockBranch())
        repr_str = repr(op)

        assert "generate" in repr_str
        assert "bound" in repr_str
        assert "pending" in repr_str  # Default status


class TestCreateOperation:
    """Test create_operation helper function."""

    def test_create_operation_basic(self):
        """create_operation should create Operation with required fields."""
        op = create_operation(operation_type="generate", parameters={"instruction": "Test"})

        assert isinstance(op, Operation)
        assert op.operation_type == "generate"
        assert op.parameters == {"instruction": "Test"}

    def test_create_operation_no_type_raises(self):
        """create_operation with no type raises error."""
        # None triggers Pydantic validation error (not a valid string)
        with pytest.raises(ValidationError, match="Input should be a valid string"):
            create_operation(operation_type=None, parameters={})

    def test_create_operation_sentinel_type_raises(self):
        """create_operation with sentinel type raises ValueError."""
        with pytest.raises(ValueError, match="operation_type is required"):
            create_operation(operation_type=Undefined, parameters={})

    def test_create_operation_default_parameters(self):
        """create_operation with Undefined parameters uses empty dict."""
        op = create_operation(operation_type="operate", parameters=Undefined)
        assert op.parameters == {}

    def test_create_operation_with_metadata(self):
        """create_operation with metadata kwargs."""
        op = create_operation(
            operation_type="communicate",
            parameters={"instruction": "Hello"},
            metadata={"name": "test_op"},
        )

        assert op.operation_type == "communicate"
        assert op.metadata.get("name") == "test_op"
