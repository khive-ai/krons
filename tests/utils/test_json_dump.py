# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime as dt
import decimal
from enum import Enum
from pathlib import Path
from uuid import UUID

import orjson
import pytest

from krons.utils._json_dump import (
    get_orjson_default,
    json_dump,
    json_dumpb,
    json_lines_iter,
    make_options,
)


# Test fixtures and helpers
class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3


class DummyModel:
    """Test model with model_dump method."""

    def model_dump(self):
        return {"field": "value"}


class DummyDict:
    """Test model with dict method."""

    def dict(self):
        return {"data": "test"}


class FailingModel:
    """Model that raises exception on model_dump."""

    def model_dump(self):
        raise ValueError("Intentional failure")


class ComplexObject:
    """Non-serializable object for testing safe fallback."""

    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"ComplexObject(value={self.value})"


# ============================================================================
# Test Basic Serialization
# ============================================================================


def test_json_dumpb_basic():
    """Test basic JSON serialization to bytes."""
    result = json_dumpb({"key": "value"})
    assert isinstance(result, bytes)
    assert orjson.loads(result) == {"key": "value"}


def test_json_dump_basic():
    """Test JSON serialization to string."""
    result = json_dump({"key": "value"}, decode=True)
    assert isinstance(result, str)
    assert result == '{"key":"value"}'


def test_json_dump_bytes_mode():
    """Test json_dump without decode returns bytes."""
    result = json_dump({"key": "value"}, decode=False)
    assert isinstance(result, bytes)
    assert orjson.loads(result) == {"key": "value"}


# ============================================================================
# Test Special Types Serialization
# ============================================================================


def test_path_serialization():
    """Test Path serialization."""
    path = Path("/tmp/test.txt")
    result = json_dump(path, decode=True)
    assert result == '"/tmp/test.txt"'


def test_decimal_as_string():
    """Test Decimal serialization as string (default)."""
    value = decimal.Decimal("123.456")
    result = json_dump(value, decode=True)
    assert result == '"123.456"'


def test_decimal_as_float():
    """Test Decimal serialization as float."""
    value = decimal.Decimal("123.456")
    result = json_dump(value, decimal_as_float=True, decode=True)
    data = orjson.loads(result)
    assert isinstance(data, float)
    assert abs(data - 123.456) < 0.001


def test_uuid_serialization():
    """Test UUID serialization."""
    uuid_val = UUID("12345678-1234-5678-1234-567812345678")
    result = json_dump(uuid_val, decode=True)
    assert result == '"12345678-1234-5678-1234-567812345678"'


def test_datetime_serialization():
    """Test datetime serialization."""
    dt_val = dt.datetime(2024, 1, 1, 12, 0, 0)
    result = json_dump(dt_val, decode=True)
    # orjson handles datetime natively
    assert "2024-01-01" in result


def test_date_serialization():
    """Test date serialization."""
    date_val = dt.date(2024, 1, 1)
    result = json_dump(date_val, decode=True)
    assert "2024-01-01" in result


def test_time_serialization():
    """Test time serialization."""
    time_val = dt.time(12, 30, 45)
    result = json_dump(time_val, decode=True)
    assert "12:30:45" in result


# ============================================================================
# Test Enum Serialization
# ============================================================================


def test_enum_default_value():
    """Test Enum serialization with default (value)."""
    result = json_dump(Color.RED, decode=True)
    assert result == "1"  # orjson uses .value by default


def test_enum_as_name():
    """Test Enum serialization as name.

    Note: orjson handles Enum natively using .value, so enum_as_name
    doesn't override the native behavior. This test verifies current behavior.
    """
    result = json_dump(Color.RED, enum_as_name=True, decode=True)
    # Currently returns value (1) due to orjson native handling
    assert result == "1"


# ============================================================================
# Test Set Serialization
# ============================================================================


def test_set_basic():
    """Test basic set serialization."""
    result = json_dump({"numbers"}, decode=True)
    data = orjson.loads(result)
    assert isinstance(data, list)


def test_set_deterministic():
    """Test deterministic set serialization."""
    # Create set with mixed types to trigger _normalize_for_sorting
    test_set = {3, 1, 2, "a", "b"}
    result1 = json_dump(test_set, deterministic_sets=True, decode=True)
    result2 = json_dump(test_set, deterministic_sets=True, decode=True)

    # Results should be identical (deterministic)
    assert result1 == result2

    # Verify sorting works
    data = orjson.loads(result1)
    assert isinstance(data, list)


def test_frozenset_deterministic():
    """Test deterministic frozenset serialization."""
    test_frozenset = frozenset([3, 1, 2])
    result = json_dump(test_frozenset, deterministic_sets=True, decode=True)
    data = orjson.loads(result)
    assert isinstance(data, list)
    assert sorted(data) == [1, 2, 3]


def test_set_with_objects_deterministic():
    """Test deterministic set with complex objects to trigger normalization."""
    # Objects with memory addresses will trigger _ADDR_PAT.sub
    obj1 = ComplexObject(1)
    obj2 = ComplexObject(2)
    test_set = {obj1, obj2}

    result = json_dump(test_set, deterministic_sets=True, safe_fallback=True, decode=True)
    data = orjson.loads(result)
    assert isinstance(data, list)
    assert len(data) == 2


# ============================================================================
# Test Safe Fallback
# ============================================================================


def test_safe_fallback_exception():
    """Test safe fallback with Exception."""
    exception = ValueError("test error")
    result = json_dump(exception, safe_fallback=True, decode=True)
    data = orjson.loads(result)

    assert data["type"] == "ValueError"
    assert data["message"] == "test error"


def test_safe_fallback_complex_object():
    """Test safe fallback with non-serializable object."""
    obj = ComplexObject("test")
    result = json_dump(obj, safe_fallback=True, decode=True)
    data = orjson.loads(result)

    # Should contain repr of object
    assert "ComplexObject" in data
    assert "test" in data


def test_safe_fallback_long_string():
    """Test safe fallback with long repr."""
    # Create object with very long repr (>2048 chars)
    long_value = "x" * 3000
    obj = ComplexObject(long_value)

    result = json_dump(obj, safe_fallback=True, fallback_clip=2048, decode=True)
    data = orjson.loads(result)

    # Should be clipped with placeholder
    assert "..." in data
    assert len(data) <= 2048 + 100  # Some margin for placeholder


def test_safe_fallback_custom_clip():
    """Test safe fallback with custom clip length."""
    long_value = "y" * 1000
    obj = ComplexObject(long_value)

    result = json_dump(obj, safe_fallback=True, fallback_clip=100, decode=True)
    data = orjson.loads(result)

    # Should be clipped at custom length
    assert "..." in data
    assert len(data) <= 200  # Custom clip + margin


def test_safe_fallback_without_error():
    """Test that safe_fallback prevents raising TypeError."""

    class UnserializableObject:
        pass

    obj = UnserializableObject()

    # With safe_fallback, should not raise
    result = json_dump(obj, safe_fallback=True, decode=True)
    assert isinstance(result, str)

    # Without safe_fallback, should raise
    with pytest.raises(TypeError, match="not JSON serializable"):
        json_dump(obj, safe_fallback=False, decode=True)


# ============================================================================
# Test Duck-Typed Objects
# ============================================================================


def test_model_dump_method():
    """Test object with model_dump method."""
    obj = DummyModel()
    result = json_dump(obj, decode=True)
    data = orjson.loads(result)
    assert data == {"field": "value"}


def test_dict_method():
    """Test object with dict method."""
    obj = DummyDict()
    result = json_dump(obj, decode=True)
    data = orjson.loads(result)
    assert data == {"data": "test"}


def test_failing_model_dump():
    """Test object with failing model_dump method falls back to dict."""

    class ObjectWithBoth:
        def model_dump(self):
            raise RuntimeError("model_dump failed")

        def dict(self):
            return {"fallback": "dict"}

    obj = ObjectWithBoth()
    result = json_dump(obj, decode=True)
    data = orjson.loads(result)
    assert data == {"fallback": "dict"}


def test_failing_both_methods():
    """Test object with both methods failing."""

    class FailingBoth:
        def model_dump(self):
            raise RuntimeError("model_dump failed")

        def dict(self):
            raise RuntimeError("dict failed")

    obj = FailingBoth()

    # Without safe_fallback, should raise
    with pytest.raises(TypeError):
        json_dump(obj, decode=True)

    # With safe_fallback, should not raise
    result = json_dump(obj, safe_fallback=True, decode=True)
    assert isinstance(result, str)


# ============================================================================
# Test Options
# ============================================================================


def test_make_options_default():
    """Test make_options with defaults."""
    opt = make_options()
    assert opt == 0


def test_make_options_pretty():
    """Test make_options with pretty printing."""
    opt = make_options(pretty=True)
    assert opt & orjson.OPT_INDENT_2


def test_make_options_sort_keys():
    """Test make_options with sorted keys."""
    opt = make_options(sort_keys=True)
    assert opt & orjson.OPT_SORT_KEYS


def test_make_options_append_newline():
    """Test make_options with append newline."""
    opt = make_options(append_newline=True)
    assert opt & orjson.OPT_APPEND_NEWLINE


def test_make_options_naive_utc():
    """Test make_options with naive UTC."""
    opt = make_options(naive_utc=True)
    assert opt & orjson.OPT_NAIVE_UTC


def test_make_options_utc_z():
    """Test make_options with UTC Z."""
    opt = make_options(utc_z=True)
    assert opt & orjson.OPT_UTC_Z


def test_make_options_passthrough_datetime():
    """Test make_options with passthrough datetime."""
    opt = make_options(passthrough_datetime=True)
    assert opt & orjson.OPT_PASSTHROUGH_DATETIME


def test_make_options_allow_non_str_keys():
    """Test make_options with non-string keys."""
    opt = make_options(allow_non_str_keys=True)
    assert opt & orjson.OPT_NON_STR_KEYS


def test_make_options_combined():
    """Test make_options with multiple flags."""
    opt = make_options(pretty=True, sort_keys=True, append_newline=True)
    assert opt & orjson.OPT_INDENT_2
    assert opt & orjson.OPT_SORT_KEYS
    assert opt & orjson.OPT_APPEND_NEWLINE


# ============================================================================
# Test Advanced Features
# ============================================================================


def test_custom_default_function():
    """Test providing custom default function."""

    def custom_default(obj):
        if isinstance(obj, ComplexObject):
            return {"custom": obj.value}
        raise TypeError("Not serializable")

    obj = ComplexObject("test")
    result = json_dump(obj, default=custom_default, decode=True)
    data = orjson.loads(result)
    assert data == {"custom": "test"}


def test_custom_options():
    """Test providing custom options."""
    opt = orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2
    result = json_dumpb({"b": 2, "a": 1}, options=opt)

    # Should be sorted and pretty-printed
    result_str = result.decode("utf-8")
    assert result_str.index('"a"') < result_str.index('"b"')
    assert "\n" in result_str


def test_get_orjson_default_with_order():
    """Test get_orjson_default with custom type order."""

    class CustomType:
        pass

    default = get_orjson_default(
        order=[CustomType],
        additional={CustomType: lambda x: "custom"},
    )

    obj = CustomType()
    result = default(obj)
    assert result == "custom"


def test_get_orjson_default_extend_default():
    """Test get_orjson_default with extend_default."""

    class CustomType:
        pass

    # With extend_default=True (default), should include both
    default = get_orjson_default(
        order=[CustomType],
        additional={CustomType: lambda x: "custom"},
        extend_default=True,
    )

    # Path should still work
    path = Path("/tmp/test")
    result = default(path)
    assert result == "/tmp/test"


def test_get_orjson_default_no_extend():
    """Test get_orjson_default without extend_default."""

    class CustomType:
        pass

    # With extend_default=False, should only use custom order
    default = get_orjson_default(
        order=[CustomType],
        additional={CustomType: lambda x: "custom"},
        extend_default=False,
    )

    obj = CustomType()
    result = default(obj)
    assert result == "custom"


def test_passthrough_datetime_option():
    """Test passthrough_datetime option."""
    dt_val = dt.datetime(2024, 1, 1, 12, 0, 0)

    # With passthrough_datetime, datetime should use custom handler
    result = json_dump(dt_val, passthrough_datetime=True, decode=True)
    data = orjson.loads(result)
    assert isinstance(data, str)
    assert "2024-01-01" in data


# ============================================================================
# Test json_lines_iter
# ============================================================================


def test_json_lines_iter_basic():
    """Test json_lines_iter with basic data."""
    data = [{"a": 1}, {"b": 2}, {"c": 3}]
    lines = list(json_lines_iter(data))

    assert len(lines) == 3
    for line in lines:
        assert isinstance(line, bytes)
        assert line.endswith(b"\n")


def test_json_lines_iter_with_sets():
    """Test json_lines_iter with sets."""
    data = [{"values": {1, 2, 3}}, {"values": {4, 5}}]
    lines = list(json_lines_iter(data, deterministic_sets=True))

    assert len(lines) == 2
    for line in lines:
        obj = orjson.loads(line)
        assert isinstance(obj["values"], list)


def test_json_lines_iter_with_custom_default():
    """Test json_lines_iter with custom default function."""

    def custom_default(obj):
        if isinstance(obj, ComplexObject):
            return obj.value
        raise TypeError("Not serializable")

    data = [ComplexObject(1), ComplexObject(2)]
    lines = list(json_lines_iter(data, default=custom_default))

    assert len(lines) == 2
    assert orjson.loads(lines[0]) == 1
    assert orjson.loads(lines[1]) == 2


def test_json_lines_iter_with_options():
    """Test json_lines_iter with custom options."""
    opt = orjson.OPT_SORT_KEYS
    data = [{"b": 2, "a": 1}, {"d": 4, "c": 3}]
    lines = list(json_lines_iter(data, options=opt))

    assert len(lines) == 2
    for line in lines:
        assert line.endswith(b"\n")


def test_json_lines_iter_empty():
    """Test json_lines_iter with empty iterable."""
    lines = list(json_lines_iter([]))
    assert len(lines) == 0


# ============================================================================
# Test Edge Cases
# ============================================================================


def test_nested_structures():
    """Test serialization of deeply nested structures."""
    nested = {
        "level1": {
            "level2": {
                "level3": [1, 2, 3],
                "path": Path("/tmp/test"),
                "decimal": decimal.Decimal("123.45"),
            }
        }
    }

    result = json_dump(nested, decode=True)
    data = orjson.loads(result)

    assert data["level1"]["level2"]["level3"] == [1, 2, 3]
    assert data["level1"]["level2"]["path"] == "/tmp/test"


def test_list_with_special_types():
    """Test list containing various special types."""
    data = [
        Path("/tmp"),
        decimal.Decimal("1.23"),
        Color.RED,
        {1, 2, 3},
    ]

    result = json_dump(data, enum_as_name=True, decode=True)
    parsed = orjson.loads(result)

    assert parsed[0] == "/tmp"
    assert parsed[1] == "1.23"
    # Enum returns value (1) due to orjson native handling
    assert parsed[2] == 1
    assert isinstance(parsed[3], list)


def test_empty_structures():
    """Test serialization of empty structures."""
    assert json_dump({}, decode=True) == "{}"
    assert json_dump([], decode=True) == "[]"
    assert json_dump("", decode=True) == '""'
    assert json_dump(set(), decode=True) == "[]"


def test_none_serialization():
    """Test None serialization."""
    assert json_dump(None, decode=True) == "null"
    assert json_dump({"key": None}, decode=True) == '{"key":null}'


def test_boolean_serialization():
    """Test boolean serialization."""
    assert json_dump(True, decode=True) == "true"
    assert json_dump(False, decode=True) == "false"


def test_numeric_types():
    """Test various numeric types."""
    data = {
        "int": 42,
        "float": 3.14,
        "negative": -100,
        "zero": 0,
    }
    result = json_dump(data, decode=True)
    parsed = orjson.loads(result)
    assert parsed == data


# ============================================================================
# Test Caching Behavior
# ============================================================================


def test_cached_default_reuse():
    """Test that cached default is reused for same parameters."""
    # First call
    result1 = json_dump(Path("/tmp/test1"), decode=True)
    # Second call with same parameters should use cached default
    result2 = json_dump(Path("/tmp/test2"), decode=True)

    assert result1 == '"/tmp/test1"'
    assert result2 == '"/tmp/test2"'


def test_type_cache_in_default():
    """Test that type cache works correctly in default function."""
    # Serialize multiple Path objects to test caching
    paths = [Path(f"/tmp/test{i}") for i in range(10)]

    for path in paths:
        result = json_dump(path, decode=True)
        assert result == f'"/tmp/test{paths.index(path)}"'


# ============================================================================
# Test Error Handling
# ============================================================================


def test_non_serializable_without_safe_fallback():
    """Test that non-serializable objects raise TypeError."""

    class NotSerializable:
        pass

    obj = NotSerializable()

    with pytest.raises(TypeError, match="not JSON serializable"):
        json_dump(obj, decode=True)


def test_non_serializable_with_safe_fallback():
    """Test that safe_fallback handles non-serializable objects."""

    class NotSerializable:
        pass

    obj = NotSerializable()
    result = json_dump(obj, safe_fallback=True, decode=True)

    # Should contain class name
    assert "NotSerializable" in result


def test_allow_non_str_keys():
    """Test serialization with non-string keys."""
    data = {1: "one", 2: "two", 3: "three"}

    result = json_dump(data, allow_non_str_keys=True, decode=True)
    parsed = orjson.loads(result)

    # Keys will be converted to strings by orjson
    assert parsed == {"1": "one", "2": "two", "3": "three"}


def test_as_loaded():
    """Test as_loaded option returns parsed object."""
    data = {"key": "value", "nested": {"a": 1}}
    result = json_dump(data, decode=True, as_loaded=True)

    # Should return the parsed dict, not a string
    assert isinstance(result, dict)
    assert result == data


def test_as_loaded_requires_decode():
    """Test that as_loaded=True requires decode=True."""
    with pytest.raises(ValueError, match="as_loaded=True requires decode=True"):
        json_dump({"key": "value"}, decode=False, as_loaded=True)
