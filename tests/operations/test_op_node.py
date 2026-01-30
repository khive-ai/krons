# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Coverage tests for operations/node.py module (migrated from lionpride).

Tests Operation event with real Session objects.
"""

import pytest

from krons.operations.node import Operation, create_operation
from krons.session import Session
from krons.types import Undefined


class TestOperationWithRealSession:
    """Test Operation with real Session objects."""

    def test_operation_repr_bound_with_real_session(self):
        """Test __repr__ when operation is bound to real session/branch."""
        session = Session()
        branch = session.create_branch(name="test")

        op = Operation(operation_type="generate", parameters={"instruction": "Test"})
        op.bind(session, branch)

        repr_str = repr(op)

        # Verify repr shows 'bound' state
        assert "generate" in repr_str
        assert "bound" in repr_str
        assert "pending" in repr_str.lower()

    def test_operation_repr_unbound(self):
        """Test __repr__ when operation is unbound."""
        op = Operation(operation_type="operate", parameters={"instruction": "Test"})

        repr_str = repr(op)

        # Verify repr shows 'unbound' state
        assert "operate" in repr_str
        assert "unbound" in repr_str

    def test_create_operation_no_type_raises_error(self):
        """Test create_operation with sentinel type raises ValueError."""
        with pytest.raises(ValueError, match=r"operation_type.*required"):
            create_operation(operation_type=Undefined, parameters={})

    def test_create_operation_with_metadata(self):
        """Test create_operation with metadata kwargs."""
        op = create_operation(
            operation_type="communicate",
            parameters={"instruction": "Hello"},
            metadata={"name": "test_op"},
        )

        assert op.operation_type == "communicate"
        assert op.metadata.get("name") == "test_op"


class TestOperationInvokeWithRealSession:
    """Test Operation execution with real Session."""

    @pytest.mark.anyio
    async def test_invoke_with_real_session(self):
        """Test Operation.invoke() with registered operation factory."""
        session = Session()
        branch = session.create_branch(name="test")

        # Register a simple factory
        async def test_factory(session, branch, params):
            return f"result_{params.get('value', 'default')}"

        session.operations.register("test_op", test_factory)

        op = Operation(operation_type="test_op", parameters={"value": "42"})
        op.bind(session, branch)

        result = await op._invoke()
        assert result == "result_42"

    @pytest.mark.anyio
    async def test_invoke_with_manual_binding(self):
        """Test Operation execution via manual binding and invoke.

        This tests the same flow as Session.conduct() without using it directly.
        """
        session = Session()
        branch = session.create_branch(name="test")

        # Register factory
        async def echo_factory(session, branch, params):
            return {"echo": params}

        session.operations.register("echo", echo_factory)

        # Create operation and invoke manually (same as what conduct() does)
        op = Operation(operation_type="echo", parameters={"message": "hello"})
        op.bind(session, branch)
        await op.invoke()

        # Operation should be completed with response
        assert op.execution.response == {"echo": {"message": "hello"}}

    @pytest.mark.anyio
    async def test_invoke_unregistered_operation_raises(self):
        """Test that invoking unregistered operation raises KeyError."""
        session = Session()
        branch = session.create_branch(name="test")

        op = Operation(operation_type="nonexistent_op", parameters={})
        op.bind(session, branch)

        with pytest.raises(KeyError, match="nonexistent_op"):
            await op._invoke()


class TestOperationBindingWithRealSession:
    """Test Operation binding scenarios with real Session."""

    def test_bind_returns_self_for_chaining(self):
        """Test that bind() returns self for method chaining."""
        session = Session()
        branch = session.create_branch(name="test")

        op = Operation(operation_type="test", parameters={})
        result = op.bind(session, branch)

        assert result is op

    def test_require_binding_returns_session_and_branch(self):
        """Test _require_binding returns session and branch when bound."""
        session = Session()
        branch = session.create_branch(name="test")

        op = Operation(operation_type="test", parameters={})
        op.bind(session, branch)

        s, b = op._require_binding()
        assert s is session
        assert b is branch

    def test_require_binding_raises_when_unbound(self):
        """Test _require_binding raises RuntimeError when unbound."""
        op = Operation(operation_type="test", parameters={})

        with pytest.raises(RuntimeError, match="not bound"):
            op._require_binding()

    def test_require_binding_raises_when_partially_bound(self):
        """Test _require_binding raises when only session is set."""
        session = Session()

        op = Operation(operation_type="test", parameters={})
        op._session = session
        # branch is still None

        with pytest.raises(RuntimeError, match="not bound"):
            op._require_binding()
