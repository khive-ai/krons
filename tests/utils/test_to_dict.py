# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import dataclasses
from collections import OrderedDict
from enum import Enum

import pytest

from krons.utils.fuzzy._to_dict import to_dict

# ============================================================================
# Mock Classes for Testing
# ============================================================================


class Color(Enum):
    """Test enum with values"""

    RED = 1
    GREEN = 2
    BLUE = 3


class Status(Enum):
    """Test enum with string values"""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"


@dataclasses.dataclass
class Person:
    """Test dataclass"""

    name: str
    age: int
    email: str = "default@example.com"


@dataclasses.dataclass
class NestedData:
    """Nested dataclass for recursion testing"""

    person: Person
    tags: list


class PydanticLike:
    """Mock Pydantic model"""

    def model_dump(self, **kwargs):
        return {"name": "pydantic", "value": 42}


class ObjectWithToDict:
    """Object with to_dict method"""

    def to_dict(self, **kwargs):
        return {"method": "to_dict", "data": "value"}


class ObjectWithDict:
    """Object with dict method"""

    def dict(self, **kwargs):
        return {"method": "dict", "data": "value"}


class ObjectWithJson:
    """Object with json method returning string"""

    def json(self, **kwargs):
        return '{"method": "json", "data": "value"}'


class ObjectWithToJson:
    """Object with to_json method"""

    def to_json(self, **kwargs):
        return '{"method": "to_json", "data": "value"}'


class ObjectWithDunderDict:
    """Object with __dict__"""

    def __init__(self):
        self.a = 1
        self.b = 2


# ============================================================================
# Test to_dict (Main Function)
# ============================================================================


def test_to_dict_basic_dict():
    """Test basic dict input"""
    assert to_dict({"a": 1, "b": 2}) == {"a": 1, "b": 2}


def test_to_dict_none():
    """Test None input"""
    assert to_dict(None) == {}


def test_to_dict_empty_string():
    """Test empty string"""
    assert to_dict("") == {}


def test_to_dict_json_string():
    """Test JSON string"""
    assert to_dict('{"a": 1}') == {"a": 1}


def test_to_dict_fuzzy_parse():
    """Test fuzzy JSON parsing"""
    assert to_dict("{'a': 1, 'b': 2}", fuzzy_parse=True) == {"a": 1, "b": 2}


def test_to_dict_custom_parser():
    """Test custom parser"""

    def parser(s):
        return {"custom": s}

    result = to_dict("test", parser=parser)
    assert result == {"custom": "test"}


def test_to_dict_set():
    """Test set conversion"""
    result = to_dict({1, 2, 3})
    assert result == {1: 1, 2: 2, 3: 3}


def test_to_dict_list():
    """Test list conversion"""
    assert to_dict([1, 2, 3]) == {0: 1, 1: 2, 2: 3}


def test_to_dict_tuple():
    """Test tuple conversion"""
    assert to_dict((1, 2, 3)) == {0: 1, 1: 2, 2: 3}


def test_to_dict_pydantic_model():
    """Test Pydantic-like model"""
    obj = PydanticLike()
    result = to_dict(obj)
    assert result == {"name": "pydantic", "value": 42}


def test_to_dict_dataclass():
    """Test dataclass"""
    person = Person(name="Bob", age=35)
    result = to_dict(person)
    assert result["name"] == "Bob"
    assert result["age"] == 35


def test_to_dict_enum_class():
    """Test enum class"""
    result = to_dict(Color, use_enum_values=True)
    assert result == {"RED": 1, "GREEN": 2, "BLUE": 3}


def test_to_dict_enum_without_values():
    """Test enum class without values"""
    result = to_dict(Color, use_enum_values=False)
    assert "RED" in result


def test_to_dict_with_suppress():
    """Test suppress mode"""
    assert to_dict("{invalid json}", suppress=True) == {}


def test_to_dict_recursive_basic():
    """Test recursive processing"""
    data = {"a": '{"nested": true}', "b": [1, 2, 3]}
    result = to_dict(data, recursive=True)
    # orjson.loads() now properly parses nested JSON strings
    assert isinstance(result, dict)
    assert result["a"] == {"nested": True}  # String IS parsed in recursive mode with orjson
    assert result["b"] == [1, 2, 3]


def test_to_dict_recursive_nested_structures():
    """Test deeply nested recursive processing"""
    data = {"level1": {"level2": '{"level3": "value"}'}}
    result = to_dict(data, recursive=True)
    # orjson.loads() properly parses nested JSON strings recursively
    assert isinstance(result["level1"], dict)
    assert result["level1"]["level2"] == {"level3": "value"}


def test_to_dict_recursive_custom_objects():
    """Test recursive with custom objects"""
    obj = ObjectWithToDict()
    data = {"obj": obj}
    result = to_dict(data, recursive=True, recursive_python_only=False)
    assert isinstance(result["obj"], dict)


def test_to_dict_max_recursive_depth_default():
    """Test default max recursive depth"""
    nested = {"a": {"b": {"c": {"d": {"e": {"f": "deep"}}}}}}
    result = to_dict(nested, recursive=True)
    assert isinstance(result, dict)


def test_to_dict_max_recursive_depth_custom():
    """Test custom max recursive depth"""
    nested = {"a": {"b": {"c": "value"}}}
    result = to_dict(nested, recursive=True, max_recursive_depth=2)
    assert isinstance(result, dict)


def test_to_dict_max_recursive_depth_negative():
    """Test negative max_recursive_depth raises error"""
    with pytest.raises(ValueError, match="must be a non-negative integer"):
        to_dict({"a": 1}, recursive=True, max_recursive_depth=-1)


def test_to_dict_max_recursive_depth_too_large():
    """Test max_recursive_depth > 10 raises error"""
    with pytest.raises(ValueError, match="must be less than or equal to 10"):
        to_dict({"a": 1}, recursive=True, max_recursive_depth=11)


def test_to_dict_max_recursive_depth_boundary():
    """Test max_recursive_depth at boundaries"""
    # 0 should work
    result = to_dict({"a": 1}, recursive=True, max_recursive_depth=0)
    assert isinstance(result, dict)

    # 10 should work
    result = to_dict({"a": 1}, recursive=True, max_recursive_depth=10)
    assert isinstance(result, dict)


def test_to_dict_prioritize_model_dump_false():
    """Test prioritize_model_dump=False"""
    obj = ObjectWithToDict()
    result = to_dict(obj, prioritize_model_dump=False)
    assert result == {"method": "to_dict", "data": "value"}


def test_to_dict_complex_nested_scenario():
    """Test complex nested scenario with multiple types"""
    data = {
        "list": [1, 2, {"nested": "value"}],
        "tuple": (4, 5, 6),
        "set": {7, 8, 9},
        "json_str": '{"parsed": true}',
        "regular": "string",
    }
    result = to_dict(data, recursive=True)
    assert isinstance(result["list"], list)
    assert isinstance(result["tuple"], tuple)
    assert isinstance(result["set"], set)
    # With orjson, nested JSON strings ARE parsed properly
    assert result["json_str"] == {"parsed": True}
    assert result["regular"] == "string"


def test_to_dict_with_object_dict_attr():
    """Test object with __dict__"""
    obj = ObjectWithDunderDict()
    result = to_dict(obj)
    assert result == {"a": 1, "b": 2}


def test_to_dict_nested_dataclasses():
    """Test nested dataclasses"""
    person = Person(name="Charlie", age=40)
    nested = NestedData(person=person, tags=["tag1", "tag2"])
    result = to_dict(nested)
    assert result["person"]["name"] == "Charlie"
    assert result["tags"] == ["tag1", "tag2"]


def test_to_dict_error_without_suppress():
    """Test error propagation without suppress"""
    with pytest.raises((ValueError, TypeError)):  # JSON parsing errors
        to_dict("{invalid json}", suppress=False)


def test_to_dict_mapping_preservation():
    """Test that mapping types are converted properly"""
    ordered = OrderedDict([("z", 26), ("a", 1)])
    result = to_dict(ordered)
    assert result == {"z": 26, "a": 1}


def test_to_dict_frozenset_in_top_level():
    """Test frozenset conversion"""
    result = to_dict(frozenset([1, 2, 3]))
    assert result == {0: 1, 1: 2, 2: 3}


def test_to_dict_recursive_sequences():
    """Test recursive processing of sequences"""
    data = [1, "2", '{"three": 3}', (4, 5)]
    result = to_dict(data, recursive=True)
    # Should enumerate top-level list
    assert isinstance(result, dict)
    assert 0 in result


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


def test_to_dict_with_none_max_depth():
    """Test None as max_recursive_depth"""
    result = to_dict({"a": 1}, recursive=True, max_recursive_depth=None)
    assert result == {"a": 1}


def test_to_dict_recursive_with_enum():
    """Test recursive processing with enum values"""
    data = {"status": Status, "nested": {"color": Color}}
    result = to_dict(data, recursive=True, use_enum_values=True)
    assert isinstance(result["status"], dict)


def test_to_dict_kwargs_passthrough():
    """Test basic JSON parsing (orjson doesn't support parse_float kwargs)"""
    # orjson.loads() doesn't accept kwargs like parse_float, object_hook, etc.
    result = to_dict('{"num": 1.5}')
    assert result["num"] == 1.5


def test_to_dict_object_with_to_json():
    """Test object with to_json method"""
    obj = ObjectWithToJson()
    result = to_dict(obj, prioritize_model_dump=False)
    assert result == {"method": "to_json", "data": "value"}


def test_to_dict_object_with_dict():
    """Test object with dict method"""
    obj = ObjectWithDict()
    result = to_dict(obj, prioritize_model_dump=False)
    assert result == {"method": "dict", "data": "value"}
