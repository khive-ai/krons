# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""HashableModel test suite: content-based hashing, serialization, protocol implementation.

Design Philosophy:
    HashableModel provides content-based hashing as alternative to Element's ID-based
    hashing. Two instances with identical field values have the same hash, enabling
    use as cache keys, set deduplication, and value-based equality.

Test Architecture:
    - Construction: Basic instantiation, nested models, optional fields
    - Hashing: Content-based equality, nested structures, determinism
    - Serialization: Uses Pydantic's model_dump/model_dump_json
    - Deserialization: Uses Pydantic's model_validate/model_validate_json
    - Edge Cases: Nested HashableModels, mixed BaseModel nesting
"""

from pydantic import BaseModel

from krons.types import HashableModel


class SimpleConfig(HashableModel):
    """Test model with basic fields."""

    name: str
    value: int


class NestedConfig(HashableModel):
    """Test model with nested HashableModel."""

    config: SimpleConfig
    enabled: bool = True


class ConfigWithOptional(HashableModel):
    """Test model with optional fields."""

    required: str
    optional: str | None = None


class TestHashableModelConstruction:
    """HashableModel instantiation and field handling."""

    def test_basic_construction(self):
        """Basic instantiation with required fields."""
        config = SimpleConfig(name="test", value=42)
        assert config.name == "test"
        assert config.value == 42

    def test_nested_construction(self):
        """Nested HashableModel construction."""
        inner = SimpleConfig(name="inner", value=1)
        outer = NestedConfig(config=inner, enabled=False)
        assert outer.config.name == "inner"
        assert outer.enabled is False

    def test_optional_fields(self):
        """Optional fields and None values."""
        config = ConfigWithOptional(required="test")
        assert config.required == "test"
        assert config.optional is None


class TestHashableModelHashing:
    """Content-based hashing behavior."""

    def test_identical_content_same_hash(self):
        """Identical field values produce same hash."""
        c1 = SimpleConfig(name="test", value=42)
        c2 = SimpleConfig(name="test", value=42)
        assert hash(c1) == hash(c2)

    def test_different_content_different_hash(self):
        """Different field values produce different hashes."""
        c1 = SimpleConfig(name="test", value=42)
        c2 = SimpleConfig(name="test", value=99)
        assert hash(c1) != hash(c2)

    def test_nested_content_hash(self):
        """Nested models contribute to hash."""
        inner1 = SimpleConfig(name="test", value=1)
        inner2 = SimpleConfig(name="test", value=1)
        inner3 = SimpleConfig(name="test", value=2)

        outer1 = NestedConfig(config=inner1)
        outer2 = NestedConfig(config=inner2)
        outer3 = NestedConfig(config=inner3)

        assert hash(outer1) == hash(outer2)
        assert hash(outer1) != hash(outer3)

    def test_hash_determinism(self):
        """Hash is deterministic across multiple calls."""
        config = SimpleConfig(name="test", value=42)
        hash1 = hash(config)
        hash2 = hash(config)
        hash3 = hash(config)
        assert hash1 == hash2 == hash3

    def test_use_in_set(self):
        """Content-based deduplication in sets."""
        c1 = SimpleConfig(name="test", value=42)
        c2 = SimpleConfig(name="test", value=42)
        c3 = SimpleConfig(name="different", value=42)

        s = {c1, c2, c3}
        assert len(s) == 2  # c1 and c2 deduplicated

    def test_use_as_dict_key(self):
        """Can use as dict key (content-based)."""
        c1 = SimpleConfig(name="test", value=42)
        c2 = SimpleConfig(name="test", value=42)

        d = {c1: "value1"}
        d[c2] = "value2"  # Overwrites c1 (same content)

        assert len(d) == 1
        assert d[c1] == "value2"


class TestHashableModelSerialization:
    """Serialization using Pydantic's model_dump and model_dump_json."""

    def test_model_dump_python_mode(self):
        """model_dump with mode='python' preserves native types."""
        config = SimpleConfig(name="test", value=42)
        data = config.model_dump(mode="python")

        assert data == {"name": "test", "value": 42}
        assert isinstance(data["value"], int)

    def test_model_dump_json_mode(self):
        """model_dump with mode='json' returns JSON-safe types."""
        config = SimpleConfig(name="test", value=42)
        data = config.model_dump(mode="json")

        assert data == {"name": "test", "value": 42}
        # All values JSON-serializable
        import orjson

        orjson.dumps(data)  # Should not raise

    def test_model_dump_json_deterministic(self):
        """JSON output is deterministic."""
        config = SimpleConfig(name="test", value=42)
        json1 = config.model_dump_json()
        json2 = config.model_dump_json()

        assert json1 == json2
        assert "name" in json1
        assert "value" in json1

    def test_model_dump_json_returns_string(self):
        """model_dump_json returns string."""
        config = SimpleConfig(name="test", value=42)
        json_str = config.model_dump_json()

        assert isinstance(json_str, str)
        import orjson

        parsed = orjson.loads(json_str)
        assert parsed["name"] == "test"


class TestHashableModelDeserialization:
    """Deserialization using Pydantic's model_validate and model_validate_json."""

    def test_model_validate_from_dict(self):
        """model_validate deserialization from dict."""
        data = {"name": "test", "value": 42}
        config = SimpleConfig.model_validate(data)

        assert config.name == "test"
        assert config.value == 42

    def test_model_validate_json_string(self):
        """model_validate_json with string input."""
        json_str = '{"name": "test", "value": 42}'
        config = SimpleConfig.model_validate_json(json_str)

        assert config.name == "test"
        assert config.value == 42

    def test_model_validate_json_bytes(self):
        """model_validate_json with bytes input."""
        json_bytes = b'{"name": "test", "value": 42}'
        config = SimpleConfig.model_validate_json(json_bytes)

        assert config.name == "test"
        assert config.value == 42

    def test_roundtrip_python_mode(self):
        """Python mode roundtrip preserves content."""
        original = SimpleConfig(name="test", value=42)
        data = original.model_dump(mode="python")
        restored = SimpleConfig.model_validate(data)

        assert hash(original) == hash(restored)
        assert original.name == restored.name
        assert original.value == restored.value

    def test_roundtrip_json_mode(self):
        """JSON mode roundtrip preserves content."""
        original = SimpleConfig(name="test", value=42)
        data = original.model_dump(mode="json")
        restored = SimpleConfig.model_validate(data)

        assert hash(original) == hash(restored)

    def test_roundtrip_json_string(self):
        """JSON string roundtrip."""
        original = SimpleConfig(name="test", value=42)
        json_str = original.model_dump_json()
        restored = SimpleConfig.model_validate_json(json_str)

        assert hash(original) == hash(restored)

    def test_nested_model_roundtrip(self):
        """Nested HashableModel roundtrip."""
        inner = SimpleConfig(name="inner", value=1)
        original = NestedConfig(config=inner, enabled=True)

        json_str = original.model_dump_json()
        restored = NestedConfig.model_validate_json(json_str)

        assert hash(original) == hash(restored)
        assert restored.config.name == "inner"
        assert restored.config.value == 1


class TestHashableModelEdgeCases:
    """Edge cases and boundary conditions."""

    def test_empty_model(self):
        """Model with no required fields."""

        class EmptyModel(HashableModel):
            pass

        m1 = EmptyModel()
        m2 = EmptyModel()

        assert hash(m1) == hash(m2)

    def test_deeply_nested_models(self):
        """Multiple levels of nesting."""

        class Level3(HashableModel):
            value: int

        class Level2(HashableModel):
            inner: Level3

        class Level1(HashableModel):
            inner: Level2

        deep1 = Level1(inner=Level2(inner=Level3(value=42)))
        deep2 = Level1(inner=Level2(inner=Level3(value=42)))

        assert hash(deep1) == hash(deep2)

    def test_mutable_by_default(self):
        """HashableModel is mutable by default (Pydantic default behavior)."""
        config = SimpleConfig(name="test", value=42)
        # HashableModel doesn't configure frozen=True, so mutation is allowed
        config.value = 99
        assert config.value == 99

    def test_mutation_changes_hash(self):
        """Mutating fields changes hash."""
        config = SimpleConfig(name="test", value=42)
        hash1 = hash(config)

        config.value = 99
        hash2 = hash(config)

        assert hash1 != hash2

    def test_pydantic_equality(self):
        """HashableModel equality via hash comparison."""
        c1 = SimpleConfig(name="test", value=42)
        c2 = SimpleConfig(name="test", value=42)

        # HashableModel implements __eq__ via hash
        assert c1 == c2
        # And have same hash (content-based)
        assert hash(c1) == hash(c2)

    def test_mixed_basemodel_nesting(self):
        """HashableModel with regular BaseModel nested."""

        class RegularModel(BaseModel):
            data: str

        class MixedModel(HashableModel):
            regular: RegularModel
            value: int

        m1 = MixedModel(regular=RegularModel(data="test"), value=1)
        m2 = MixedModel(regular=RegularModel(data="test"), value=1)

        # Should be hashable and equal
        assert hash(m1) == hash(m2)

    def test_hashable_model_not_equal_to_non_hashable_model(self):
        """HashableModel.__eq__ returns NotImplemented for non-HashableModel."""
        config = SimpleConfig(name="test", value=42)

        # Comparing with non-HashableModel returns NotImplemented
        result = config.__eq__("not a model")
        assert result is NotImplemented


class TestHashableModelInheritance:
    """Test HashableModel inheritance from Pydantic BaseModel."""

    def test_inherits_from_base_model(self):
        """HashableModel inherits from Pydantic BaseModel."""
        assert issubclass(HashableModel, BaseModel)

    def test_has_hash_method(self):
        """HashableModel has __hash__ method."""
        assert hasattr(HashableModel, "__hash__")
        config = SimpleConfig(name="test", value=42)
        assert isinstance(hash(config), int)

    def test_has_eq_method(self):
        """HashableModel has __eq__ method."""
        assert hasattr(HashableModel, "__eq__")


class TestHashableModelComplexTypes:
    """Test HashableModel with complex field types."""

    def test_list_fields(self):
        """HashableModel with list fields."""

        class ListModel(HashableModel):
            items: list[int]
            tags: list[str] = []

        m1 = ListModel(items=[1, 2, 3], tags=["a", "b"])
        m2 = ListModel(items=[1, 2, 3], tags=["a", "b"])
        m3 = ListModel(items=[1, 2, 3], tags=["a", "c"])

        assert hash(m1) == hash(m2)
        assert hash(m1) != hash(m3)

    def test_dict_fields(self):
        """HashableModel with dict fields."""

        class DictModel(HashableModel):
            data: dict[str, int]

        m1 = DictModel(data={"a": 1, "b": 2})
        m2 = DictModel(data={"a": 1, "b": 2})
        m3 = DictModel(data={"a": 1, "b": 3})

        assert hash(m1) == hash(m2)
        assert hash(m1) != hash(m3)

    def test_set_fields(self):
        """HashableModel with set fields (converted from frozenset)."""

        class SetModel(HashableModel):
            items: frozenset[int]

        m1 = SetModel(items=frozenset({1, 2, 3}))
        m2 = SetModel(items=frozenset({1, 2, 3}))
        m3 = SetModel(items=frozenset({1, 2, 4}))

        assert hash(m1) == hash(m2)
        assert hash(m1) != hash(m3)

    def test_tuple_fields(self):
        """HashableModel with tuple fields."""

        class TupleModel(HashableModel):
            point: tuple[int, int]

        m1 = TupleModel(point=(1, 2))
        m2 = TupleModel(point=(1, 2))
        m3 = TupleModel(point=(1, 3))

        assert hash(m1) == hash(m2)
        assert hash(m1) != hash(m3)
