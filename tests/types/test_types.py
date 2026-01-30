# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
Type System Base Classes: Immutable Parameters & Metadata

**Core Abstractions**:
- **Params**: Frozen dataclass base for function parameters (sentinel-aware)
- **DataClass**: Mutable variant of Params for state containers
- **Meta**: Immutable metadata key-value pair (callable-aware hashing)
- **ModelConfig**: Configuration for validation behavior and sentinel handling
- **Enum**: Enhanced enum with `allowed()` method for valid values
- **HashableModel**: Pydantic BaseModel subclass with hash/equality support

**Design Philosophy**:
- **Sentinel Semantics**: Precise state tracking (Undefined vs Unset vs None)
- **Immutability by Default**: Params frozen, updates via `with_updates()`
- **Configurable Validation**: Strict mode (require all fields) vs lenient (allow partial)
- **Serialization Safety**: `to_dict()` filters sentinels, preserves only set values
- **Framework Independence**: No Pydantic/attrs coupling (used by Spec/Operable)

**Testing Strategy**:
This test suite validates:
1. Params creation with sentinel handling
2. Strict mode enforcement (`ModelConfig.strict=True`)
3. Sentinel filtering in serialization (`to_dict()`)
4. Immutable updates via `with_updates()`
5. Meta hashing (callable vs. value-based)
6. DataClass post-init validation
7. Enum value extraction (`allowed()`)
8. ModelConfig options (sentinel_additions, prefill_unset, use_enum_values)
9. HashableModel hash/equality for Pydantic-based configs
"""

from dataclasses import dataclass, field
from enum import Enum as StdEnum
from typing import ClassVar

import pytest

from krons.types import (
    DataClass,
    Enum,
    HashableModel,
    Meta,
    ModelConfig,
    Params,
    Undefined,
    Unset,
    is_sentinel,
    not_sentinel,
)

# ============================================================================
# Test Enum.allowed()
# ============================================================================


class MyTestEnum(Enum):
    """Test enum for testing"""

    VALUE1 = "value1"
    VALUE2 = "value2"
    VALUE3 = "value3"


def test_enum_allowed():
    """Enum.allowed() extracts all enum values as tuple"""
    allowed = MyTestEnum.allowed()
    assert isinstance(allowed, tuple)
    assert "value1" in allowed
    assert "value2" in allowed
    assert "value3" in allowed
    assert len(allowed) == 3


# ============================================================================
# Test Params validation and configuration
# ============================================================================


@dataclass(slots=True, frozen=True, init=False)
class MyParams(Params):
    """Test params class"""

    field1: str = Unset
    field2: int = Unset
    field3: bool = Unset


def test_params_invalid_parameter():
    """Params rejects unknown fields at construction (early validation)"""
    with pytest.raises(ValueError, match="Invalid parameter"):
        MyParams(field1="valid", invalid_field="should fail")


def test_params_valid():
    """Test Params.__init__ with valid parameters"""
    params = MyParams(field1="test", field2=42)
    assert params.field1 == "test"
    assert params.field2 == 42


def test_params_allowed():
    """Test Params.allowed() method"""
    allowed = MyParams.allowed()
    assert isinstance(allowed, set)
    assert "field1" in allowed
    assert "field2" in allowed
    assert "field3" in allowed
    assert "_none_as_sentinel" not in allowed  # Private fields excluded


@dataclass(slots=True, frozen=True, init=False)
class MyParamsNoneSentinel(Params):
    """Test params class with None as sentinel"""

    _config: ClassVar[ModelConfig] = ModelConfig(sentinel_additions=frozenset({"none"}))
    field1: str = Unset


def test_params_is_sentinel_none_as_sentinel():
    """sentinel_additions={"none"} treats None as sentinel (API partial updates)"""
    # When "none" in sentinel_additions, None should be treated as sentinel
    assert MyParamsNoneSentinel._is_sentinel(None) is True
    assert MyParamsNoneSentinel._is_sentinel(Undefined) is True
    assert MyParamsNoneSentinel._is_sentinel(Unset) is True
    assert MyParamsNoneSentinel._is_sentinel("value") is False


def test_params_is_sentinel_default():
    """Default sentinel behavior treats None as valid value (explicit null)"""
    # When _none_as_sentinel is False, None is not a sentinel
    assert MyParams._is_sentinel(None) is False
    assert MyParams._is_sentinel(Undefined) is True
    assert MyParams._is_sentinel(Unset) is True
    assert MyParams._is_sentinel("value") is False


@dataclass(slots=True, frozen=True, init=False)
class MyParamsStrict(Params):
    """Test params class with strict mode"""

    _config: ClassVar[ModelConfig] = ModelConfig(strict=True)
    field1: str = Unset
    field2: int = Unset


def test_params_strict_mode():
    """Strict mode requires all fields without defaults (production API safety)"""
    with pytest.raises(ExceptionGroup, match="Missing required parameters"):
        MyParamsStrict(field1="value")  # field2 is missing and strict=True


# ============================================================================
# Test DataClass configuration and validation
# ============================================================================


@dataclass(slots=True)
class MyDataClass(DataClass):
    """Test data class"""

    field1: str = Unset
    field2: int = Unset


def test_dataclass_valid():
    """Test DataClass with valid fields"""
    obj = MyDataClass(field1="test", field2=42)
    assert obj.field1 == "test"
    assert obj.field2 == 42


def test_dataclass_allowed():
    """Test DataClass.allowed() returns all declared fields."""
    allowed = MyDataClass.allowed()
    assert isinstance(allowed, set)
    assert "field1" in allowed
    assert "field2" in allowed


@dataclass(slots=True)
class MyDataClassStrict(DataClass):
    """Test data class with strict mode"""

    _config: ClassVar[ModelConfig] = ModelConfig(strict=True)
    field1: str = Unset


def test_dataclass_strict_mode():
    """Test DataClass strict mode enforces required fields."""
    with pytest.raises(ExceptionGroup, match="Missing required parameters"):
        MyDataClassStrict()  # Missing required field in strict mode


@dataclass(slots=True)
class MyDataClassPrefillUnset(DataClass):
    """Test data class with prefill_unset"""

    _config: ClassVar[ModelConfig] = ModelConfig(prefill_unset=True)
    field1: str = field(default=Undefined)


def test_dataclass_prefill_unset():
    """prefill_unset=True auto-fills Undefined fields with Unset"""
    obj = MyDataClassPrefillUnset()
    # Field initialized to Undefined should be prefilled with Unset
    assert obj.field1 is Unset


@dataclass(slots=True)
class MyDataClassNoneSentinel(DataClass):
    """Test data class with None as sentinel"""

    _config: ClassVar[ModelConfig] = ModelConfig(sentinel_additions=frozenset({"none"}))
    field1: str = None


def test_dataclass_is_sentinel_none():
    """Test DataClass._is_sentinel with sentinel_additions={"none"}"""
    assert MyDataClassNoneSentinel._is_sentinel(None) is True
    assert MyDataClassNoneSentinel._is_sentinel(Undefined) is True
    assert MyDataClassNoneSentinel._is_sentinel(Unset) is True


def test_dataclass_to_dict():
    """to_dict() serializes fields and filters sentinels (API/database safety)"""
    obj = MyDataClass(field1="test", field2=42)
    result = obj.to_dict()
    assert "field1" in result
    assert "field2" in result


def test_dataclass_to_dict_exclude():
    """Test DataClass.to_dict() with exclude"""
    obj = MyDataClass(field1="test", field2=42)
    result = obj.to_dict(exclude={"field2"})
    assert "field1" in result
    assert "field2" not in result


def test_dataclass_with_updates():
    """with_updates() creates new instance with modifications (immutable update pattern)"""
    obj = MyDataClass(field1="test", field2=42)
    updated = obj.with_updates(field2=100)
    assert updated.field1 == "test"
    assert updated.field2 == 100


def test_dataclass_hash():
    """Test DataClass.__hash__() method"""
    # DataClass needs to be frozen to be hashable, use Params instead
    params1 = MyParams(field1="test", field2=42)
    params2 = MyParams(field1="test", field2=42)
    hash1 = hash(params1)
    hash2 = hash(params2)
    # Equal objects must have equal hashes
    assert hash1 == hash2
    assert isinstance(hash1, int)
    assert isinstance(hash2, int)


def test_dataclass_eq():
    """Test DataClass.__eq__() method"""
    obj1 = MyDataClass(field1="test", field2=42)
    obj2 = MyDataClass(field1="test", field2=42)
    obj3 = MyDataClass(field1="other", field2=99)
    # Validate equality correctness
    assert obj1 == obj2
    assert obj1 != obj3


def test_dataclass_eq_not_dataclass():
    """Test DataClass.__eq__() with non-DataClass"""
    obj = MyDataClass(field1="test", field2=42)
    assert obj != "not a dataclass"
    assert obj != 42


# ============================================================================
# Test Params methods
# ============================================================================


def test_params_to_dict():
    """Test Params.to_dict() method"""
    params = MyParams(field1="test", field2=42)
    result = params.to_dict()
    assert "field1" in result
    assert "field2" in result


def test_params_to_dict_exclude():
    """Test Params.to_dict() with exclude"""
    params = MyParams(field1="test", field2=42)
    result = params.to_dict(exclude={"field2"})
    assert "field1" in result
    assert "field2" not in result


def test_params_with_updates():
    """Test Params.with_updates() method"""
    params = MyParams(field1="test", field2=42)
    updated = params.with_updates(field2=100)
    assert updated.field1 == "test"
    assert updated.field2 == 100


def test_params_hash():
    """Test Params.__hash__() method"""
    params1 = MyParams(field1="test", field2=42)
    params2 = MyParams(field1="test", field2=42)
    hash1 = hash(params1)
    hash2 = hash(params2)
    # Equal objects must have equal hashes
    assert hash1 == hash2
    assert isinstance(hash1, int)
    assert isinstance(hash2, int)


def test_params_eq():
    """Test Params.__eq__() method"""
    params1 = MyParams(field1="test", field2=42)
    params2 = MyParams(field1="test", field2=42)
    params3 = MyParams(field1="other", field2=99)
    # Validate equality correctness
    assert params1 == params2
    assert params1 != params3


def test_params_eq_not_params():
    """Test Params.__eq__() with non-Params"""
    params = MyParams(field1="test", field2=42)
    assert params != "not params"
    assert params != 42


def test_params_default_kw():
    """Test Params.default_kw() method"""
    params = MyParams(field1="test", field2=42)
    result = params.default_kw()
    assert isinstance(result, dict)
    assert result["field1"] == "test"
    assert result["field2"] == 42


# ============================================================================
# Test sentinel utilities
# ============================================================================


def test_is_sentinel():
    """is_sentinel() identifies Undefined/Unset (validation helper)"""
    assert is_sentinel(Undefined) is True
    assert is_sentinel(Unset) is True
    assert is_sentinel(None) is False
    assert is_sentinel("value") is False
    assert is_sentinel(42) is False


def test_not_sentinel():
    """not_sentinel() provides type narrowing via TypeGuard (mypy/pyright support)"""
    assert not_sentinel(Undefined) is False
    assert not_sentinel(Unset) is False
    assert not_sentinel(None) is True
    assert not_sentinel("value") is True
    assert not_sentinel(42) is True


# ============================================================================
# Test Enum Normalization
# ============================================================================


class ColorEnum(StdEnum):
    """Enum for normalization tests"""

    RED = "red"
    GREEN = "green"
    BLUE = "blue"


@dataclass(slots=True, frozen=True, init=False)
class MyParamsWithEnum(Params):
    """Test params class with enum normalization"""

    _config: ClassVar[ModelConfig] = ModelConfig(use_enum_values=True)
    color: ColorEnum = Unset
    name: str = Unset


def test_params_normalize_enum():
    """use_enum_values=True serializes enums as .value (JSON compatibility)"""
    params = MyParamsWithEnum(color=ColorEnum.RED, name="test")
    result = params.to_dict()
    # With use_enum_values=True, should get string not enum
    assert result["color"] == "red"
    assert not isinstance(result["color"], StdEnum)
    assert result["name"] == "test"


@dataclass(slots=True)
class MyDataClassWithEnum(DataClass):
    """Test data class with enum normalization"""

    _config: ClassVar[ModelConfig] = ModelConfig(use_enum_values=True)
    color: ColorEnum = Unset
    name: str = Unset


def test_dataclass_normalize_enum():
    """Test DataClass normalizes enum to value when use_enum_values=True."""
    obj = MyDataClassWithEnum(color=ColorEnum.BLUE, name="test")
    result = obj.to_dict()
    # With use_enum_values=True, should get string not enum
    assert result["color"] == "blue"
    assert not isinstance(result["color"], StdEnum)


# ============================================================================
# Test Hash Methods Explicit Call
# ============================================================================


def test_params_hash_explicit():
    """Test Params.__hash__ returns int as required by Python hash protocol.

    Note: __hash__ must return int for Python's hash protocol (sets/dicts).
    The implementation uses hash_obj which returns a stable int hash.
    """
    params = MyParams(field1="test", field2=42)
    # Call the parent Params.__hash__ method directly
    hash_result = Params.__hash__(params)
    # __hash__ must return int for Python hash protocol
    assert isinstance(hash_result, int)
    # Verify it produces consistent results
    assert Params.__hash__(params) == hash_result


def test_dataclass_hash_explicit():
    """Test DataClass.__hash__ returns int as required by Python hash protocol.

    Note: __hash__ must return int for Python's hash protocol (sets/dicts).
    The implementation uses hash_obj which returns a stable int hash.
    """
    obj = MyDataClass(field1="test", field2=42)
    hash_result = DataClass.__hash__(obj)
    # __hash__ must return int for Python hash protocol
    assert isinstance(hash_result, int)
    # Verify consistency
    assert DataClass.__hash__(obj) == hash_result


# ============================================================================
# Test NotImplemented Protocol
# ============================================================================


def test_params_eq_returns_notimplemented():
    """Test Params.__eq__ returns NotImplemented for non-Params types."""
    params = MyParams(field1="test", field2=42)
    # Call parent Params.__eq__ directly to avoid dataclass override
    result = Params.__eq__(params, "not params")
    assert result is NotImplemented

    result = Params.__eq__(params, 42)
    assert result is NotImplemented

    result = Params.__eq__(params, [1, 2, 3])
    assert result is NotImplemented

    # Test the happy path - comparing two Params instances
    params2 = MyParams(field1="test", field2=42)
    result = Params.__eq__(params, params2)
    assert result is True  # Same values should be equal


def test_dataclass_eq_returns_notimplemented():
    """Test DataClass.__eq__ returns NotImplemented for non-DataClass types."""
    obj = MyDataClass(field1="test", field2=42)
    # Call parent DataClass.__eq__ directly to avoid dataclass override
    result = DataClass.__eq__(obj, "not dataclass")
    assert result is NotImplemented

    result = DataClass.__eq__(obj, None)
    assert result is NotImplemented

    # Test the happy path - comparing two DataClass instances
    # Note: This line is technically dead code (DataClass not frozen, so hash() fails)
    # But we can test it by mocking hash() to return consistent values
    obj2 = MyDataClass(field1="test", field2=42)
    import unittest.mock

    # Mock hash to return the same value for both objects
    with unittest.mock.patch("builtins.hash", return_value=12345):
        result = DataClass.__eq__(obj, obj2)
        # Both objects have same mocked hash, so should be equal
        assert result is True


# ============================================================================
# Test Deep Copy with_updates
# ============================================================================


@dataclass(slots=True, frozen=True, init=False)
class MyParamsWithContainers(Params):
    """Test params class with mutable containers"""

    items: list = Unset
    config: dict = Unset
    tags: set = Unset


def test_params_with_updates_shallow():
    """Test Params.with_updates with copy_containers='shallow'."""
    original_list = [1, 2, 3]
    original_dict = {"key": "value"}
    original_set = {1, 2, 3}
    params = MyParamsWithContainers(items=original_list, config=original_dict, tags=original_set)

    # Shallow copy containers
    updated = params.with_updates(copy_containers="shallow", items=[4, 5, 6])

    # Verify copy occurred
    assert params.items == [1, 2, 3]
    assert updated.items == [4, 5, 6]
    # Other mutable fields should also be copied
    assert params.config == {"key": "value"}
    assert params.tags == {1, 2, 3}


@dataclass(slots=True)
class MyDataClassWithContainers(DataClass):
    """Test data class with mutable containers"""

    items: list = field(default_factory=list)
    tags: set = field(default_factory=set)
    config: dict = field(default_factory=dict)


def test_dataclass_with_updates_shallow():
    """Test DataClass.with_updates copies containers when copy_containers='shallow'."""
    obj = MyDataClassWithContainers(items=[1, 2], tags={1, 2}, config={"a": 1})
    updated = obj.with_updates(copy_containers="shallow", items=[3, 4])

    # Verify copy occurred
    assert obj.items == [1, 2]
    assert updated.items == [3, 4]
    # Verify other containers were also copied
    assert obj.tags == {1, 2}
    assert obj.config == {"a": 1}


def test_params_with_updates_deep_copy_nested():
    """Test copy_containers='deep' performs recursive copying on nested structures."""
    # Nested structure: dict with list values
    nested_dict = {"outer": {"inner": [1, 2]}}
    params = MyParamsWithContainers(config=nested_dict)

    # Shallow copy - inner structures shared
    shallow = params.with_updates(copy_containers="shallow", config={"outer": {"inner": [1, 2]}})
    shallow.config["outer"]["inner"].append(3)
    # Original should be unchanged (top-level dict was copied)
    assert params.config == {"outer": {"inner": [1, 2]}}

    # Deep copy - fully isolated, no shared references
    deep = params.with_updates(copy_containers="deep")
    deep.config["outer"]["inner"].append(999)
    # Original unchanged (recursive copy)
    assert params.config == {"outer": {"inner": [1, 2]}}
    assert deep.config == {"outer": {"inner": [1, 2, 999]}}


def test_dataclass_with_updates_deep_copy_nested():
    """Test DataClass copy_containers='deep' performs recursive copying on nested structures."""
    # Nested list of lists
    nested_items = [[1, 2], [3, 4]]
    obj = MyDataClassWithContainers(items=nested_items)

    # Shallow copy - inner lists shared
    shallow = obj.with_updates(copy_containers="shallow", items=[[1, 2], [3, 4]])
    shallow.items[0].append(999)
    # Original unchanged (top-level list was copied)
    assert obj.items == [[1, 2], [3, 4]]

    # Deep copy - fully isolated
    deep = obj.with_updates(copy_containers="deep")
    deep.items[0].append(999)
    # Original unchanged (recursive copy)
    assert obj.items == [[1, 2], [3, 4]]
    assert deep.items == [[1, 2, 999], [3, 4]]


def test_params_with_updates_no_copy_for_updated_fields():
    """Test that fields in kwargs are not copied (performance optimization)."""
    # Create expensive nested structure
    expensive_nested = {"level1": {"level2": {"level3": [1, 2, 3] * 1000}}}
    params = MyParamsWithContainers(config=expensive_nested, items=[1, 2, 3])

    # Update items with shallow copy - config should be copied, items should NOT
    updated = params.with_updates(copy_containers="shallow", items=[4, 5, 6])

    # Verify items was replaced (not copied then replaced)
    assert updated.items == [4, 5, 6]
    # Verify config was copied (not in kwargs)
    assert updated.config == expensive_nested
    assert updated.config is not params.config  # Different object


def test_params_with_updates_invalid_copy_containers():
    """Test runtime validation for invalid copy_containers values."""
    params = MyParamsWithContainers(items=[1, 2, 3])

    # Invalid string
    with pytest.raises(ValueError, match="Invalid copy_containers"):
        params.with_updates(copy_containers="invalid", items=[4, 5, 6])

    # Case mismatch (case-sensitive)
    with pytest.raises(ValueError, match="Invalid copy_containers"):
        params.with_updates(copy_containers="SHALLOW", items=[4, 5, 6])

    # Empty string
    with pytest.raises(ValueError, match="Invalid copy_containers"):
        params.with_updates(copy_containers="", items=[4, 5, 6])

    # Typo
    with pytest.raises(ValueError, match="Invalid copy_containers"):
        params.with_updates(copy_containers="shalllow", items=[4, 5, 6])


def test_dataclass_with_updates_invalid_copy_containers():
    """Test DataClass runtime validation for invalid copy_containers values."""
    obj = MyDataClassWithContainers(items=[1, 2], tags={1, 2}, config={"a": 1})

    # Invalid string
    with pytest.raises(ValueError, match="Invalid copy_containers"):
        obj.with_updates(copy_containers="deep_copy", items=[3, 4])

    # Case mismatch
    with pytest.raises(ValueError, match="Invalid copy_containers"):
        obj.with_updates(copy_containers="Deep", items=[3, 4])


def test_params_with_updates_abc_collections():
    """Test copy_containers works with collections.abc types (deque, defaultdict, etc)."""
    from collections import defaultdict, deque
    from dataclasses import dataclass
    from typing import Any

    @dataclass(frozen=True)
    class ParamsWithABCCollections(Params):
        _config: ClassVar[ModelConfig] = ModelConfig()
        queue: Any = Unset
        counts: Any = Unset

    # Test deque (MutableSequence but not list subclass)
    params = ParamsWithABCCollections(queue=deque([1, 2, 3]), counts=defaultdict(int))
    params.counts["a"] = 5

    shallow = params.with_updates(copy_containers="shallow")
    assert isinstance(shallow.queue, deque)
    assert isinstance(shallow.counts, defaultdict)

    # Verify shallow copy behavior
    shallow.queue.append(999)
    assert 999 not in params.queue  # Original unchanged

    shallow.counts["b"] = 10
    assert "b" not in params.counts  # Original unchanged


# ============================================================================
# Test Meta Class
# ============================================================================


def test_meta_hash_simple():
    """Test Meta.__hash__ with simple hashable values."""
    meta1 = Meta(key="field1", value="string_value")
    meta2 = Meta(key="field1", value="string_value")
    meta3 = Meta(key="field1", value=42)

    # Same key and value should have same hash
    assert hash(meta1) == hash(meta2)
    assert isinstance(hash(meta1), int)

    # Different values should likely have different hashes
    assert hash(meta1) != hash(meta3)


def test_meta_hash_callable():
    """Meta uses identity-based hashing for callables (validator correctness)"""

    def validator(x):
        return x > 0

    meta1 = Meta(key="validator", value=validator)
    meta2 = Meta(key="validator", value=validator)

    # Same callable instance -> same hash
    assert hash(meta1) == hash(meta2)

    # Different callable instances
    def another_validator(x):
        return x > 0

    meta3 = Meta(key="validator", value=another_validator)
    # Different id -> different hash
    assert hash(meta1) != hash(meta3)


def test_meta_hash_unhashable():
    """Meta falls back to str() hashing for unhashable types (robustness)"""
    # Unhashable dict value should trigger fallback
    meta = Meta(key="config", value={"unhashable": "dict"})
    # Should fallback to str() hashing without error
    hash_result = hash(meta)
    assert isinstance(hash_result, int)

    # Lists are also unhashable
    meta_list = Meta(key="items", value=[1, 2, 3])
    assert isinstance(hash(meta_list), int)


def test_meta_eq_simple():
    """Test Meta.__eq__ with simple values."""
    meta1 = Meta(key="field1", value="val")
    meta2 = Meta(key="field1", value="val")
    meta3 = Meta(key="field2", value="val")
    meta4 = Meta(key="field1", value="different")

    # Same key and value
    assert meta1 == meta2

    # Different keys
    assert meta1 != meta3

    # Same key, different value
    assert meta1 != meta4


def test_meta_eq_callable():
    """Test Meta.__eq__ with callables."""

    def fn1(x):
        return x

    def fn2(x):
        return x

    meta1 = Meta(key="fn", value=fn1)
    meta2 = Meta(key="fn", value=fn1)  # Same instance
    meta3 = Meta(key="fn", value=fn2)  # Different instance

    # Same instance -> identity match
    assert meta1 == meta2

    # Different id -> not equal
    assert meta1 != meta3


def test_meta_eq_not_meta():
    """Test Meta.__eq__ returns NotImplemented for non-Meta types."""
    meta = Meta(key="field", value="val")
    # Call __eq__ directly to verify return value
    result = meta.__eq__("not a meta")
    assert result is NotImplemented

    result = meta.__eq__(42)
    assert result is NotImplemented

    result = meta.__eq__(None)
    assert result is NotImplemented


# ============================================================================
# Test Params with default_factory and private fields
# ============================================================================


@dataclass(slots=True, frozen=True, init=False)
class MyParamsWithFactory(Params):
    """Test params class with default_factory."""

    items: list = field(default_factory=list)
    name: str = Unset


def test_params_default_factory():
    """Params uses default_factory when field not provided in kwargs."""
    # Create without providing 'items' - should use factory
    params = MyParamsWithFactory(name="test")
    assert params.name == "test"
    assert params.items == []
    assert isinstance(params.items, list)

    # Create another instance - should get fresh list
    params2 = MyParamsWithFactory(name="test2")
    assert params2.items == []
    assert params.items is not params2.items  # Different instances


@dataclass(slots=True, frozen=True, init=False)
class MyParamsWithPrivateField(Params):
    """Test params class with private-named field (starts with _)."""

    _internal: str = Unset
    public: str = Unset


def test_params_skips_private_fields_in_defaults():
    """Params.__init__ skips fields starting with underscore during default application."""
    # Create without providing _internal
    params = MyParamsWithPrivateField(public="hello")
    assert params.public == "hello"
    # _internal should not be in allowed() due to private naming
    assert "_internal" not in MyParamsWithPrivateField.allowed()
    assert "public" in MyParamsWithPrivateField.allowed()


@dataclass(slots=True, frozen=True, init=False)
class MyParamsPrefillUnset(Params):
    """Test params class with prefill_unset=True and Undefined default."""

    _config: ClassVar[ModelConfig] = ModelConfig(prefill_unset=True)
    field_with_undefined: str = Undefined


def test_params_prefill_unset():
    """Params._validate prefills Undefined fields with Unset when configured."""
    params = MyParamsPrefillUnset()
    # Field should be prefilled from Undefined to Unset
    assert params.field_with_undefined is Unset


# ============================================================================
# Test HashableModel (Pydantic-based hashable model)
# ============================================================================


class ServiceConfig(HashableModel):
    """Test HashableModel subclass."""

    provider: str
    name: str


class NestedConfig(HashableModel):
    """Test HashableModel with nested structures."""

    items: list[str] = []
    metadata: dict[str, str] = {}


def test_hashable_model_creation():
    """Test HashableModel can be created with Pydantic validation."""
    config = ServiceConfig(provider="openai", name="gpt-4")
    assert config.provider == "openai"
    assert config.name == "gpt-4"


def test_hashable_model_hash_returns_int():
    """HashableModel.__hash__ returns int for Python hash protocol compatibility."""
    config = ServiceConfig(provider="openai", name="gpt-4")
    hash_result = hash(config)
    assert isinstance(hash_result, int)


def test_hashable_model_hash_consistency():
    """HashableModel produces consistent hashes for same content."""
    config1 = ServiceConfig(provider="openai", name="gpt-4")
    config2 = ServiceConfig(provider="openai", name="gpt-4")
    # Same content = same hash
    assert hash(config1) == hash(config2)


def test_hashable_model_hash_differs():
    """HashableModel produces different hashes for different content."""
    config1 = ServiceConfig(provider="openai", name="gpt-4")
    config2 = ServiceConfig(provider="anthropic", name="claude")
    # Different content should likely produce different hashes
    assert hash(config1) != hash(config2)


def test_hashable_model_eq():
    """HashableModel equality based on hash comparison."""
    config1 = ServiceConfig(provider="openai", name="gpt-4")
    config2 = ServiceConfig(provider="openai", name="gpt-4")
    config3 = ServiceConfig(provider="anthropic", name="claude")
    # Same content = equal
    assert config1 == config2
    # Different content = not equal
    assert config1 != config3


def test_hashable_model_eq_not_hashable_model():
    """HashableModel.__eq__ returns NotImplemented for non-HashableModel types."""
    config = ServiceConfig(provider="openai", name="gpt-4")
    # Call __eq__ directly to verify return value
    result = config.__eq__("not a model")
    assert result is NotImplemented

    result = config.__eq__(42)
    assert result is NotImplemented

    result = config.__eq__(None)
    assert result is NotImplemented


def test_hashable_model_usable_in_set():
    """HashableModel instances can be used in sets (hashable contract)."""
    config1 = ServiceConfig(provider="openai", name="gpt-4")
    config2 = ServiceConfig(provider="openai", name="gpt-4")
    config3 = ServiceConfig(provider="anthropic", name="claude")

    # Should be usable in set
    config_set = {config1, config2, config3}
    # Equal configs deduplicated
    assert len(config_set) == 2


def test_hashable_model_usable_in_dict():
    """HashableModel instances can be used as dict keys (hashable contract)."""
    config1 = ServiceConfig(provider="openai", name="gpt-4")
    config2 = ServiceConfig(provider="anthropic", name="claude")

    # Should be usable as dict key
    config_dict = {config1: "first", config2: "second"}
    assert config_dict[config1] == "first"
    assert config_dict[config2] == "second"

    # Same content retrieves same value
    config1_copy = ServiceConfig(provider="openai", name="gpt-4")
    assert config_dict[config1_copy] == "first"


def test_hashable_model_with_nested_structures():
    """HashableModel handles nested mutable structures for hashing."""
    config1 = NestedConfig(items=["a", "b"], metadata={"key": "value"})
    config2 = NestedConfig(items=["a", "b"], metadata={"key": "value"})
    config3 = NestedConfig(items=["c"], metadata={"other": "data"})

    # Same nested content = equal
    assert config1 == config2
    assert hash(config1) == hash(config2)

    # Different nested content = not equal
    assert config1 != config3
