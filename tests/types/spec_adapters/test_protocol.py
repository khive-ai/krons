# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
SpecAdapter Protocol Tests: Framework-Agnostic Adapter Interface

**Core Abstractions**:
- **SpecAdapter[F]**: Generic protocol for framework-specific structure generation
- **Abstract Methods**: create_field, compose_structure, extract_specs
- **Optional Methods**: create_field_validator, validate_instance, dump_instance
- **Plugin Architecture**: Add new frameworks without modifying Spec/Operable code

**Design Philosophy**:
- **Framework Isolation**: Framework-specific imports only in concrete adapters
- **Generic Type Safety**: `SpecAdapter[F]` where F = field representation type
- **Hook Pattern**: `create_field_validator` as optional hook for framework-specific validator creation

**Testing Strategy**:
This test suite validates:
1. Protocol contract (abstract methods must be implemented by concrete adapters)
2. Base class behavior (default implementations, hooks)
3. Hook pattern (`create_field_validator` returns None by default)
4. Minimal adapter pattern (testing base class without framework dependencies)
"""

from typing import Any

import pytest

from krons.specs import Operable, Spec, SpecAdapter


class MinimalAdapter(SpecAdapter):
    """
    Minimal Concrete Adapter: Test fixture for SpecAdapter base class behavior

    **Purpose**: Test base protocol functionality without framework dependencies

    **Implementation**:
    - All abstract methods implemented as no-ops (minimal logic)
    - No framework imports (pure Python)
    - Returns sensible defaults (None, empty tuple, dynamic type)
    - Used to test hook pattern and base class methods
    """

    @classmethod
    def create_field(cls, spec: Spec) -> Any:
        return None

    @classmethod
    def compose_structure(
        cls,
        op,
        name: str,
        /,
        *,
        include=None,
        exclude=None,
        **kwargs,
    ):
        return type(name, (), {})

    @classmethod
    def extract_specs(cls, structure: Any) -> tuple:
        return ()


def test_create_field_validator_returns_none():
    """
    Base create_field_validator hook returns None (no validator created by default)

    **Pattern**: Hook pattern for optional framework-specific features

    **Scenario**: Adapter that doesn't support validators calls base implementation
    ```python
    # Minimal adapter with no validator support
    validator = MinimalAdapter.create_field_validator(spec)
    # Returns None (no validator created)
    ```

    **Expected Behavior**:
    - Base implementation returns None (not an error, just "no validator")
    - Concrete adapters override to return framework-specific validators
    - Enables optional feature without breaking protocol contract
    """
    # Create a simple Spec
    spec = Spec(int, name="test_field", description="Test field")

    # Call create_field_validator on the minimal adapter (uses base implementation)
    result = MinimalAdapter.create_field_validator(spec)

    # Should return None (base implementation)
    assert result is None


def test_create_field_validator_with_metadata():
    """
    Base create_field_validator ignores Spec validator metadata (concrete adapters extract it)

    **Pattern**: Hook pattern with metadata extraction left to concrete adapters

    **Scenario**: Spec contains validator metadata, but base implementation doesn't use it
    ```python
    spec = Spec(int, name="age", validator=nonneg_validator)
    # Base implementation ignores validator metadata
    result = MinimalAdapter.create_field_validator(spec)
    # Returns None (metadata not extracted)
    ```

    **Expected Behavior**:
    - Base implementation returns None (even with validator metadata present)
    - Validator metadata exists in Spec but is not extracted by base class
    - Concrete adapters extract and use validator metadata appropriately
    - Base class establishes contract, not implementation
    """

    def validator_func(value):
        return value

    # Create a Spec with validator
    spec = Spec(
        int,
        name="age",
        description="User age",
        validator=validator_func,
    )

    # Call create_field_validator on minimal adapter (base implementation)
    result = MinimalAdapter.create_field_validator(spec)

    # Base implementation returns None
    # Even with validator metadata present
    assert result is None


def test_minimal_adapter_implements_protocol():
    """MinimalAdapter should be recognized as implementing SpecAdapter protocol."""
    assert issubclass(MinimalAdapter, SpecAdapter)


def test_minimal_adapter_create_field():
    """MinimalAdapter.create_field returns None."""
    spec = Spec(str, name="test")
    result = MinimalAdapter.create_field(spec)
    assert result is None


def test_minimal_adapter_compose_structure():
    """MinimalAdapter.compose_structure returns a new type with the given name."""
    spec = Spec(str, name="field")
    op = Operable((spec,), name="TestOp")

    Model = MinimalAdapter.compose_structure(op, "TestModel")

    assert isinstance(Model, type)
    assert Model.__name__ == "TestModel"


def test_minimal_adapter_validate_instance_raises():
    """MinimalAdapter.validate_instance raises NotImplementedError (base behavior)."""

    class TestModel:
        pass

    with pytest.raises(NotImplementedError) as exc_info:
        MinimalAdapter.validate_instance(TestModel, {"data": "test"})

    assert "MinimalAdapter does not support instance validation" in str(exc_info.value)


def test_minimal_adapter_dump_instance_raises():
    """MinimalAdapter.dump_instance raises NotImplementedError (base behavior)."""

    class TestModel:
        pass

    with pytest.raises(NotImplementedError) as exc_info:
        MinimalAdapter.dump_instance(TestModel())

    assert "MinimalAdapter does not support instance dumping" in str(exc_info.value)


def test_minimal_adapter_extract_specs():
    """MinimalAdapter.extract_specs returns empty tuple."""

    class TestModel:
        pass

    result = MinimalAdapter.extract_specs(TestModel)

    assert result == ()
    assert isinstance(result, tuple)
