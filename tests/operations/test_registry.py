# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.operations.registry - OperationRegistry."""

import pytest

from krons.operations.registry import OperationRegistry


class TestOperationRegistry:
    """Test OperationRegistry."""

    def test_register_handler(self):
        """Registry.register() should add handler."""
        registry = OperationRegistry()

        async def my_factory(session, branch, params):
            return "result"

        registry.register("my_op", my_factory)

        assert "my_op" in registry
        assert registry.has("my_op")

    def test_get_handler(self):
        """Registry.get() should retrieve handler."""
        registry = OperationRegistry()

        async def my_factory(session, branch, params):
            return "result"

        registry.register("my_op", my_factory)

        factory = registry.get("my_op")
        assert factory is my_factory

    def test_get_handler_not_found_raises(self):
        """Registry.get() should raise KeyError for unknown operation."""
        registry = OperationRegistry()

        with pytest.raises(KeyError, match="not registered"):
            registry.get("nonexistent")

    def test_has_handler(self):
        """Registry.has() should check existence."""
        registry = OperationRegistry()

        async def my_factory(session, branch, params):
            return "result"

        assert registry.has("my_op") is False

        registry.register("my_op", my_factory)

        assert registry.has("my_op") is True
        assert registry.has("other_op") is False


class TestOperationRegistryDuplicate:
    """Test duplicate registration handling."""

    def test_register_duplicate_raises_by_default(self):
        """Duplicate registration should raise ValueError."""
        registry = OperationRegistry()

        async def factory1(session, branch, params):
            return "result1"

        async def factory2(session, branch, params):
            return "result2"

        registry.register("my_op", factory1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register("my_op", factory2)

    def test_register_duplicate_with_override(self):
        """Duplicate registration with override=True should succeed."""
        registry = OperationRegistry()

        async def factory1(session, branch, params):
            return "result1"

        async def factory2(session, branch, params):
            return "result2"

        registry.register("my_op", factory1)
        registry.register("my_op", factory2, override=True)

        factory = registry.get("my_op")
        assert factory is factory2


class TestOperationRegistryUnregister:
    """Test unregister() method."""

    def test_unregister_existing(self):
        """unregister() should remove existing handler."""
        registry = OperationRegistry()

        async def my_factory(session, branch, params):
            return "result"

        registry.register("my_op", my_factory)
        assert registry.has("my_op")

        result = registry.unregister("my_op")
        assert result is True
        assert not registry.has("my_op")

    def test_unregister_nonexistent(self):
        """unregister() should return False for nonexistent handler."""
        registry = OperationRegistry()

        result = registry.unregister("nonexistent")
        assert result is False


class TestOperationRegistryListNames:
    """Test list_names() method."""

    def test_list_names_empty(self):
        """list_names() should return empty list for empty registry."""
        registry = OperationRegistry()
        assert registry.list_names() == []

    def test_list_names_with_registrations(self):
        """list_names() should return all registered operation names."""
        registry = OperationRegistry()

        async def factory(session, branch, params):
            return "result"

        registry.register("op1", factory)
        registry.register("op2", factory)
        registry.register("op3", factory)

        names = registry.list_names()
        assert len(names) == 3
        assert set(names) == {"op1", "op2", "op3"}


class TestOperationRegistryDunderMethods:
    """Test __contains__, __len__, __repr__."""

    def test_contains(self):
        """'in' operator should check existence."""
        registry = OperationRegistry()

        async def factory(session, branch, params):
            return "result"

        registry.register("my_op", factory)

        assert "my_op" in registry
        assert "other_op" not in registry

    def test_len(self):
        """len() should return number of registered operations."""
        registry = OperationRegistry()

        async def factory(session, branch, params):
            return "result"

        assert len(registry) == 0

        registry.register("op1", factory)
        assert len(registry) == 1

        registry.register("op2", factory)
        assert len(registry) == 2

    def test_repr(self):
        """__repr__ should show registered operations."""
        registry = OperationRegistry()

        async def factory(session, branch, params):
            return "result"

        registry.register("my_op", factory)
        registry.register("other_op", factory)

        repr_str = repr(registry)
        assert "OperationRegistry" in repr_str
        assert "my_op" in repr_str
        assert "other_op" in repr_str


class TestOperationRegistryInvocation:
    """Test that registered factories can be invoked correctly."""

    @pytest.mark.anyio
    async def test_factory_invocation(self):
        """Registered factory should be callable."""
        registry = OperationRegistry()

        async def my_factory(session, branch, params):
            return f"result_{params['value']}"

        registry.register("my_op", my_factory)

        factory = registry.get("my_op")
        result = await factory(None, None, {"value": "42"})
        assert result == "result_42"

    @pytest.mark.anyio
    async def test_factory_receives_all_args(self):
        """Factory should receive session, branch, and params."""
        registry = OperationRegistry()

        received_args = {}

        async def tracking_factory(session, branch, params):
            received_args["session"] = session
            received_args["branch"] = branch
            received_args["params"] = params
            return "tracked"

        registry.register("tracking", tracking_factory)

        class MockSession:
            name = "test_session"

        class MockBranch:
            name = "test_branch"

        session = MockSession()
        branch = MockBranch()
        params = {"key": "value"}

        factory = registry.get("tracking")
        await factory(session, branch, params)

        assert received_args["session"] is session
        assert received_args["branch"] is branch
        assert received_args["params"] is params
