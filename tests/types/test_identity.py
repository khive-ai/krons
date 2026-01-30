# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for semantic ID[T] type annotation."""

from typing import Annotated, get_args, get_origin
from uuid import UUID, uuid4

import pytest

from krons.types import ID


class MockUser:
    """Mock user model for testing."""

    pass


class MockTenant:
    """Mock tenant model for testing."""

    pass


class TestIDType:
    """Tests for ID[T] semantic UUID typing."""

    def test_id_class_exists(self):
        """ID class should exist and inherit from UUID."""
        assert ID is not None
        assert issubclass(ID, UUID)

    def test_id_subscript_returns_annotated(self):
        """ID[Model] should return Annotated[UUID, ("ID", Model)]."""
        result = ID[MockUser]

        # Should be Annotated
        assert get_origin(result) is Annotated

        # Get type args
        args = get_args(result)
        assert len(args) == 2

        # First arg is UUID
        assert args[0] is UUID

        # Second arg is the metadata tuple
        assert args[1] == ("ID", MockUser)

    def test_id_different_types_are_distinct(self):
        """ID[User] and ID[Tenant] should produce different types."""
        user_id_type = ID[MockUser]
        tenant_id_type = ID[MockTenant]

        # Both are Annotated[UUID, ...]
        assert get_origin(user_id_type) is Annotated
        assert get_origin(tenant_id_type) is Annotated

        # But metadata differs
        user_args = get_args(user_id_type)
        tenant_args = get_args(tenant_id_type)

        assert user_args[1] == ("ID", MockUser)
        assert tenant_args[1] == ("ID", MockTenant)
        assert user_args[1] != tenant_args[1]

    def test_id_runtime_is_uuid(self):
        """At runtime, ID values should be valid UUIDs."""
        # Create a UUID value
        value: ID[MockUser] = uuid4()  # type: ignore

        # It's a UUID instance
        assert isinstance(value, UUID)

        # Can convert to/from string
        uuid_str = str(value)
        restored = UUID(uuid_str)
        assert restored == value

    def test_id_can_be_used_in_type_hints(self):
        """ID[T] can be used as a type hint."""

        def get_user_by_id(user_id: ID[MockUser]) -> MockUser | None:
            """Example function using ID[T] type hint."""
            _ = user_id
            return None

        # Function exists and can be called
        result = get_user_by_id(uuid4())  # type: ignore
        assert result is None

    def test_id_with_string_forward_ref(self):
        """ID should work with string forward references."""
        # String forward refs still produce Annotated
        result = ID["ForwardRef"]

        assert get_origin(result) is Annotated
        args = get_args(result)
        assert args[0] is UUID
        assert args[1] == ("ID", "ForwardRef")
