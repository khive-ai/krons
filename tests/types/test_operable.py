# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.types.operable - Operable spec container.

Operable is the validated Spec collection that enables model generation.
Key behaviors:
- Collect specs with uniqueness validation
- Provide filtering (include/exclude) for partial models
- Delegate to adapters for framework-specific generation
"""

import pytest
from pydantic import BaseModel

from krons.specs import Operable, Spec, get_adapter
from krons.specs.adapters.pydantic_adapter import PydanticSpecAdapter
from krons.types import Unset


class TestOperable:
    """Test Operable class."""

    def test_operable_from_model(self):
        """Operable.from_structure() should extract specs from Pydantic model."""

        class SimpleModel(BaseModel):
            name: str
            age: int

        op = Operable.from_structure(SimpleModel, adapter="pydantic")

        assert op.name == "SimpleModel"
        assert op.allowed() == {"name", "age"}
        assert len(op.__op_fields__) == 2

    def test_operable_get(self):
        """Operable.get() should retrieve spec by name."""
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        operable = Operable((spec1, spec2))

        result = operable.get("field1")
        assert result is spec1

        result2 = operable.get("field2")
        assert result2 is spec2

    def test_operable_get_missing(self):
        """Operable.get() returns default for missing field."""
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))

        result = operable.get("missing")
        assert result is Unset

        result_with_default = operable.get("missing", default="custom")
        assert result_with_default == "custom"

    def test_operable_iter(self):
        """get_specs() should yield all specs."""
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        operable = Operable((spec1, spec2))

        specs = operable.get_specs()
        assert len(specs) == 2
        assert spec1 in specs
        assert spec2 in specs


class TestOperableCreation:
    """Test Operable creation and validation."""

    def test_basic_creation(self):
        """Basic Operable creation with named Specs."""
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        operable = Operable((spec1, spec2), name="TestModel")

        assert len(operable.__op_fields__) == 2
        assert operable.name == "TestModel"

    def test_empty_operable(self):
        """Empty Operable is valid (zero specs)."""
        operable = Operable()
        assert len(operable.__op_fields__) == 0
        assert operable.allowed() == set()

    def test_creation_with_list(self):
        """Operable accepts list and converts to tuple."""
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        operable = Operable([spec1, spec2], name="TestModel")

        assert len(operable.__op_fields__) == 2
        assert isinstance(operable.__op_fields__, tuple)
        assert operable.name == "TestModel"

    def test_type_validation(self):
        """Operable rejects non-Spec objects."""
        with pytest.raises(TypeError, match="All specs must be Spec objects"):
            Operable(("not_a_spec",))

    def test_duplicate_name_detection(self):
        """Operable rejects duplicate field names."""
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field1")  # Duplicate name
        with pytest.raises(ValueError, match="Duplicate field names found"):
            Operable((spec1, spec2))


class TestOperableAllowed:
    """Test Operable.allowed() and check_allowed()."""

    def test_allowed(self):
        """allowed() returns set of all field names."""
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        operable = Operable((spec1, spec2))

        allowed = operable.allowed()
        assert allowed == {"field1", "field2"}

    def test_check_allowed_valid(self):
        """check_allowed() returns True for valid field names."""
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))

        assert operable.check_allowed("field1") is True

    def test_check_allowed_invalid_raises(self):
        """check_allowed() raises ValueError for invalid field names."""
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))

        with pytest.raises(ValueError, match="not allowed"):
            operable.check_allowed("field2")

    def test_check_allowed_as_boolean(self):
        """check_allowed() with as_boolean=True returns bool."""
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))

        assert operable.check_allowed("field1", as_boolean=True) is True
        assert operable.check_allowed("field2", as_boolean=True) is False


class TestOperableGetSpecs:
    """Test Operable.get_specs() filtering."""

    def test_get_specs_no_filter(self):
        """get_specs() without filters returns all specs."""
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        operable = Operable((spec1, spec2))

        specs = operable.get_specs()
        assert specs == (spec1, spec2)

    def test_get_specs_include(self):
        """get_specs() with include filters to specified fields."""
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        spec3 = Spec(bool, name="field3")
        operable = Operable((spec1, spec2, spec3))

        specs = operable.get_specs(include={"field1", "field3"})
        assert len(specs) == 2
        assert spec1 in specs
        assert spec3 in specs
        assert spec2 not in specs

    def test_get_specs_exclude(self):
        """get_specs() with exclude filters out specified fields."""
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        spec3 = Spec(bool, name="field3")
        operable = Operable((spec1, spec2, spec3))

        specs = operable.get_specs(exclude={"field2"})
        assert len(specs) == 2
        assert spec1 in specs
        assert spec3 in specs
        assert spec2 not in specs

    def test_get_specs_include_empty_set_returns_nothing(self):
        """get_specs(include=set()) returns zero specs (include nothing)."""
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        operable = Operable((spec1, spec2))

        specs = operable.get_specs(include=set())
        assert specs == ()

    def test_get_specs_exclude_empty_set_returns_all(self):
        """get_specs(exclude=set()) returns all specs (exclude nothing)."""
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        operable = Operable((spec1, spec2))

        specs = operable.get_specs(exclude=set())
        assert specs == (spec1, spec2)

    def test_get_specs_both_include_exclude_raises(self):
        """get_specs() raises ValueError when both include and exclude specified."""
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))

        with pytest.raises(ValueError, match="Cannot specify both"):
            operable.get_specs(include={"field1"}, exclude={"field2"})

    def test_field_ordering_preserved(self):
        """Field ordering is preserved from construction."""
        specs = [
            Spec(str, name="field_a"),
            Spec(int, name="field_b"),
            Spec(bool, name="field_c"),
        ]
        operable = Operable(tuple(specs))

        field_names = [s.name for s in operable.__op_fields__]
        assert field_names == ["field_a", "field_b", "field_c"]
        assert isinstance(operable.__op_fields__, tuple)


class TestOperableImmutability:
    """Test Operable immutability."""

    def test_immutability(self):
        """Operable is immutable (frozen dataclass)."""
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,))

        with pytest.raises((AttributeError, TypeError)):
            operable.name = "new_name"


class TestOperableFromStructure:
    """Test Operable.from_structure() for extracting specs from Pydantic models."""

    def test_from_structure_basic(self):
        """from_structure() creates Operable from simple Pydantic model."""

        class SimpleModel(BaseModel):
            name: str
            age: int

        op = Operable.from_structure(SimpleModel, adapter="pydantic")

        assert op.name == "SimpleModel"
        assert op.allowed() == {"name", "age"}
        assert len(op.__op_fields__) == 2

    def test_from_structure_with_defaults(self):
        """from_structure() preserves default values."""

        class ModelWithDefaults(BaseModel):
            name: str = "default_name"
            count: int = 0

        op = Operable.from_structure(ModelWithDefaults, adapter="pydantic")

        name_spec = op.get("name")
        count_spec = op.get("count")

        assert name_spec.default == "default_name"
        assert count_spec.default == 0

    def test_from_structure_nullable_fields(self):
        """from_structure() detects Optional/nullable fields."""

        class ModelWithOptional(BaseModel):
            required: str
            optional: str | None = None

        op = Operable.from_structure(ModelWithOptional, adapter="pydantic")

        required_spec = op.get("required")
        optional_spec = op.get("optional")

        assert required_spec.is_nullable is False
        assert optional_spec.is_nullable is True
        assert optional_spec.default is None

    def test_from_structure_list_fields(self):
        """from_structure() detects list/listable fields."""

        class ModelWithList(BaseModel):
            tags: list[str]
            scores: list[int] = []

        op = Operable.from_structure(ModelWithList, adapter="pydantic")

        tags_spec = op.get("tags")
        scores_spec = op.get("scores")

        assert tags_spec.is_listable is True
        assert tags_spec.base_type is str
        assert scores_spec.is_listable is True
        assert scores_spec.base_type is int

    def test_from_structure_optional_list(self):
        """from_structure() handles Optional[list[T]] correctly."""

        class ModelWithOptionalList(BaseModel):
            items: list[str] | None = None

        op = Operable.from_structure(ModelWithOptionalList, adapter="pydantic")
        items_spec = op.get("items")

        assert items_spec.is_listable is True
        assert items_spec.is_nullable is True
        assert items_spec.base_type is str
        assert items_spec.default is None

    def test_from_structure_custom_name(self):
        """from_structure() accepts custom operable name."""

        class OriginalName(BaseModel):
            field: str

        op = Operable.from_structure(OriginalName, adapter="pydantic", name="CustomName")
        assert op.name == "CustomName"

    def test_from_structure_type_error_on_non_basemodel(self):
        """from_structure() raises TypeError for non-BaseModel classes."""

        class NotAModel:
            name: str

        with pytest.raises(TypeError, match="must be a Pydantic BaseModel subclass"):
            Operable.from_structure(NotAModel, adapter="pydantic")

    def test_from_structure_roundtrip(self):
        """from_structure() -> compose_structure() produces equivalent model."""

        class OriginalModel(BaseModel):
            name: str
            count: int = 0
            tags: list[str] = []

        op = Operable.from_structure(OriginalModel, adapter="pydantic")
        ReconstructedModel = op.compose_structure()

        # Test that both models accept same data
        original = OriginalModel(name="test", count=5, tags=["a", "b"])
        reconstructed = ReconstructedModel(name="test", count=5, tags=["a", "b"])

        assert original.name == reconstructed.name
        assert original.count == reconstructed.count
        assert original.tags == reconstructed.tags


class TestOperableComposeStructure:
    """Test Operable.compose_structure() for generating Pydantic models."""

    def test_compose_structure_success(self):
        """compose_structure() generates Pydantic model from Operable."""
        spec1 = Spec(str, name="field1", default="default_value")
        spec2 = Spec(int, name="field2")
        operable = Operable((spec1, spec2), name="TestModel")

        model = operable.compose_structure("GeneratedModel")

        assert model is not None
        assert model.__name__ == "GeneratedModel"

    def test_compose_structure_with_filters(self):
        """compose_structure() respects include/exclude filters."""
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        spec3 = Spec(bool, name="field3")
        operable = Operable((spec1, spec2, spec3), name="TestModel")

        # Test with include
        model_include = operable.compose_structure("IncludeModel", include={"field1", "field3"})
        assert model_include is not None

        # Test with exclude
        model_exclude = operable.compose_structure("ExcludeModel", exclude={"field2"})
        assert model_exclude is not None

    def test_compose_structure_default_name(self):
        """compose_structure() uses Operable name as default model name."""
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,), name="OperableName")

        model = operable.compose_structure()
        assert model is not None
        assert model.__name__ == "OperableName"

        # Test without operable.name - should use "DynamicStructure"
        operable_no_name = Operable((spec1,))
        model_dynamic = operable_no_name.compose_structure()
        assert model_dynamic is not None
        assert model_dynamic.__name__ == "DynamicStructure"


class TestOperableAdapter:
    """Test Operable adapter access."""

    def test_adapter_resolution(self):
        """get_adapter resolves adapter class from __adapter_name__."""
        spec1 = Spec(str, name="field1")
        operable = Operable((spec1,), adapter="pydantic")

        assert get_adapter(operable.__adapter_name__) is PydanticSpecAdapter

    def test_unsupported_adapter(self):
        """Unsupported adapter raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported adapter"):
            get_adapter("unsupported")
