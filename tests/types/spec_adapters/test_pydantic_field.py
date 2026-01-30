# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.types.spec_adapters.pydantic_field - Pydantic adapter.

PydanticSpecAdapter converts Spec to Pydantic FieldInfo and generates BaseModel classes.
Key behaviors:
- create_field: Spec -> FieldInfo with annotations and metadata
- compose_structure: Operable -> dynamic BaseModel subclass
- extract_specs: BaseModel -> tuple[Spec, ...] (disassembly)
- validate_instance: dict -> model instance
- fuzzy_match_fields: flexible key matching
"""

import pytest
from pydantic import BaseModel, Field, ValidationError

from krons.specs import Operable, Spec
from krons.specs.adapters.pydantic_adapter import PydanticSpecAdapter


class TestPydanticSpecAdapterCreateField:
    """Test PydanticSpecAdapter.create_field()."""

    def test_create_field_basic(self):
        """create_field creates FieldInfo from simple Spec."""
        spec = Spec(str, name="username")
        field_info = PydanticSpecAdapter.create_field(spec)

        assert field_info is not None
        assert field_info.annotation == str

    def test_create_field_with_default(self):
        """create_field preserves default value."""
        spec = Spec(str, name="username", default="anonymous")
        field_info = PydanticSpecAdapter.create_field(spec)

        assert field_info.default == "anonymous"

    def test_create_field_nullable(self):
        """create_field handles nullable type annotation."""
        spec = Spec(str, name="nickname", nullable=True)
        field_info = PydanticSpecAdapter.create_field(spec)

        # Nullable spec should have default=None auto-injected
        assert field_info.default is None

    def test_create_field_nullable_required(self):
        """create_field preserves requiredness for nullable fields with required=True."""
        spec = Spec(str, name="nickname", nullable=True, required=True)
        field_info = PydanticSpecAdapter.create_field(spec)

        # With required=True, should NOT auto-inject default=None
        from pydantic_core import PydanticUndefined

        assert field_info.default is PydanticUndefined

    def test_create_field_with_description(self):
        """create_field includes description in FieldInfo."""
        spec = Spec(str, name="username", description="User's display name")
        field_info = PydanticSpecAdapter.create_field(spec)

        assert field_info.description == "User's display name"

    def test_create_field_with_constraints(self):
        """create_field passes through Pydantic field constraints."""
        spec = Spec(int, name="age", ge=0, le=150)
        field_info = PydanticSpecAdapter.create_field(spec)

        assert field_info is not None
        # Constraints are in json_schema_extra since they're custom metadata
        # The actual validation constraints are applied through metadata

    def test_create_field_listable(self):
        """create_field creates list annotation for listable Spec."""
        spec = Spec(int, name="scores", listable=True)
        field_info = PydanticSpecAdapter.create_field(spec)

        assert field_info.annotation == list[int]


class TestPydanticSpecAdapterComposeStructure:
    """Test PydanticSpecAdapter.compose_structure()."""

    def test_compose_structure_basic(self):
        """compose_structure generates BaseModel from Operable."""
        spec1 = Spec(str, name="name")
        spec2 = Spec(int, name="age", default=0)
        operable = Operable((spec1, spec2), name="Person")

        Model = PydanticSpecAdapter.compose_structure(operable, "PersonModel")

        assert Model is not None
        assert issubclass(Model, BaseModel)
        assert Model.__name__ == "PersonModel"
        assert "name" in Model.model_fields
        assert "age" in Model.model_fields

    def test_compose_structure_can_instantiate(self):
        """Generated model can be instantiated with valid data."""
        spec1 = Spec(str, name="name")
        spec2 = Spec(int, name="age", default=0)
        operable = Operable((spec1, spec2))

        Model = PydanticSpecAdapter.compose_structure(operable, "PersonModel")

        instance = Model(name="Alice")
        assert instance.name == "Alice"
        assert instance.age == 0

    def test_compose_structure_with_include(self):
        """compose_structure respects include filter."""
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        spec3 = Spec(bool, name="field3")
        operable = Operable((spec1, spec2, spec3))

        Model = PydanticSpecAdapter.compose_structure(
            operable, "FilteredModel", include={"field1", "field3"}
        )

        assert "field1" in Model.model_fields
        assert "field2" not in Model.model_fields
        assert "field3" in Model.model_fields

    def test_compose_structure_with_exclude(self):
        """compose_structure respects exclude filter."""
        spec1 = Spec(str, name="field1")
        spec2 = Spec(int, name="field2")
        spec3 = Spec(bool, name="field3")
        operable = Operable((spec1, spec2, spec3))

        Model = PydanticSpecAdapter.compose_structure(operable, "FilteredModel", exclude={"field2"})

        assert "field1" in Model.model_fields
        assert "field2" not in Model.model_fields
        assert "field3" in Model.model_fields

    def test_compose_structure_with_validator(self):
        """compose_structure includes validators from Spec metadata."""

        def must_be_positive(value):
            if value <= 0:
                raise ValueError("Must be positive")
            return value

        spec = Spec(int, name="count", validator=must_be_positive)
        operable = Operable((spec,))

        Model = PydanticSpecAdapter.compose_structure(operable, "ValidatedModel")

        # Valid value should work
        instance = Model(count=5)
        assert instance.count == 5

        # Invalid value should raise
        with pytest.raises(ValidationError):
            Model(count=-1)


class TestPydanticSpecAdapterValidateInstance:
    """Test PydanticSpecAdapter.validate_instance()."""

    def test_validate_instance_basic(self):
        """validate_instance creates instance from valid dict."""
        spec = Spec(str, name="name")
        operable = Operable((spec,))
        Model = PydanticSpecAdapter.compose_structure(operable, "TestModel")

        instance = PydanticSpecAdapter.validate_instance(Model, {"name": "Alice"})

        assert instance.name == "Alice"

    def test_validate_instance_invalid_raises(self):
        """validate_instance raises ValidationError for invalid data."""
        spec = Spec(str, name="name")
        operable = Operable((spec,))
        Model = PydanticSpecAdapter.compose_structure(operable, "TestModel")

        with pytest.raises(ValidationError):
            PydanticSpecAdapter.validate_instance(Model, {"name": 123})

    def test_validate_instance_missing_required_raises(self):
        """validate_instance raises ValidationError for missing required fields."""
        spec = Spec(str, name="name")
        operable = Operable((spec,))
        Model = PydanticSpecAdapter.compose_structure(operable, "TestModel")

        with pytest.raises(ValidationError):
            PydanticSpecAdapter.validate_instance(Model, {})


class TestPydanticSpecAdapterDumpInstance:
    """Test PydanticSpecAdapter.dump_instance()."""

    def test_dump_instance_basic(self):
        """dump_instance converts instance to dict."""
        spec = Spec(str, name="name")
        operable = Operable((spec,))
        Model = PydanticSpecAdapter.compose_structure(operable, "TestModel")
        instance = Model(name="Alice")

        data = PydanticSpecAdapter.dump_instance(instance)

        assert data == {"name": "Alice"}

    def test_dump_instance_preserves_all_fields(self):
        """dump_instance includes all field values."""
        spec1 = Spec(str, name="name")
        spec2 = Spec(int, name="age", default=0)
        operable = Operable((spec1, spec2))
        Model = PydanticSpecAdapter.compose_structure(operable, "TestModel")
        instance = Model(name="Alice", age=30)

        data = PydanticSpecAdapter.dump_instance(instance)

        assert data == {"name": "Alice", "age": 30}


class TestPydanticSpecAdapterExtractSpecs:
    """Test PydanticSpecAdapter.extract_specs()."""

    def test_extract_specs_basic(self):
        """extract_specs extracts Specs from model fields."""

        class SimpleModel(BaseModel):
            name: str
            age: int

        specs = PydanticSpecAdapter.extract_specs(SimpleModel)

        assert len(specs) == 2
        spec_names = {s.name for s in specs}
        assert spec_names == {"name", "age"}

    def test_extract_specs_with_defaults(self):
        """extract_specs preserves default values."""

        class ModelWithDefaults(BaseModel):
            name: str = "anonymous"
            count: int = 0

        specs = PydanticSpecAdapter.extract_specs(ModelWithDefaults)

        name_spec = next(s for s in specs if s.name == "name")
        count_spec = next(s for s in specs if s.name == "count")

        assert name_spec.default == "anonymous"
        assert count_spec.default == 0

    def test_extract_specs_nullable(self):
        """extract_specs detects nullable/Optional fields."""

        class ModelWithOptional(BaseModel):
            required: str
            optional: str | None = None

        specs = PydanticSpecAdapter.extract_specs(ModelWithOptional)

        required_spec = next(s for s in specs if s.name == "required")
        optional_spec = next(s for s in specs if s.name == "optional")

        assert required_spec.is_nullable is False
        assert optional_spec.is_nullable is True

    def test_extract_specs_listable(self):
        """extract_specs detects list types."""

        class ModelWithList(BaseModel):
            tags: list[str]
            scores: list[int] = []

        specs = PydanticSpecAdapter.extract_specs(ModelWithList)

        tags_spec = next(s for s in specs if s.name == "tags")
        scores_spec = next(s for s in specs if s.name == "scores")

        assert tags_spec.is_listable is True
        assert tags_spec.base_type is str
        assert scores_spec.is_listable is True
        assert scores_spec.base_type is int

    def test_extract_specs_non_basemodel_raises(self):
        """extract_specs raises TypeError for non-BaseModel."""

        class NotAModel:
            name: str

        with pytest.raises(TypeError, match="must be a Pydantic BaseModel"):
            PydanticSpecAdapter.extract_specs(NotAModel)


class TestPydanticSpecAdapterFieldValidatorPrefix:
    """Test PydanticSpecAdapter.create_field_validator() key naming."""

    def test_validator_key_has_underscore_prefix(self):
        """create_field_validator returns key with _ prefix."""
        spec = Spec(str, name="name", validator=lambda cls, v: v.strip())
        result = PydanticSpecAdapter.create_field_validator(spec)

        assert result is not None
        assert "_name_validator" in result


class TestPydanticSpecAdapterCreateFieldValidator:
    """Test PydanticSpecAdapter.create_field_validator()."""

    def test_create_field_validator_none_without_metadata(self):
        """create_field_validator returns None when no validator in Spec."""
        spec = Spec(str, name="name")
        result = PydanticSpecAdapter.create_field_validator(spec)
        assert result is None

    def test_create_field_validator_with_validator_metadata(self):
        """create_field_validator creates field_validator from Spec metadata."""

        def validate_positive(value):
            if value <= 0:
                raise ValueError("Must be positive")
            return value

        spec = Spec(int, name="count", validator=validate_positive)
        result = PydanticSpecAdapter.create_field_validator(spec)

        assert result is not None
        assert "_count_validator" in result


class TestPydanticSpecAdapterIntegration:
    """Integration tests for full adapter workflow."""

    def test_roundtrip_spec_to_model_to_spec(self):
        """Spec -> Model -> Spec preserves semantics."""

        class OriginalModel(BaseModel):
            name: str
            age: int = 0
            tags: list[str] = []
            nickname: str | None = None

        # Disassemble to Specs
        specs = PydanticSpecAdapter.extract_specs(OriginalModel)
        operable = Operable(specs, name="ReconstructedModel")

        # Reassemble to Model
        ReconstructedModel = PydanticSpecAdapter.compose_structure(operable, "ReconstructedModel")

        # Both should accept same data
        original = OriginalModel(name="Alice", age=30, tags=["dev"], nickname="Al")
        reconstructed = ReconstructedModel(name="Alice", age=30, tags=["dev"], nickname="Al")

        assert original.name == reconstructed.name
        assert original.age == reconstructed.age
        assert original.tags == reconstructed.tags
        assert original.nickname == reconstructed.nickname

    def test_validate_instance_integration(self):
        """validate_instance validates dict into model instance."""
        spec1 = Spec(str, name="name")
        spec2 = Spec(int, name="age", default=0)
        operable = Operable((spec1, spec2))
        Model = PydanticSpecAdapter.compose_structure(operable, "TestModel")

        result = PydanticSpecAdapter.validate_instance(Model, {"name": "Alice", "age": 25})

        assert result is not None
        assert result.name == "Alice"
        assert result.age == 25

    def test_dump_instance(self):
        """dump_instance converts model instance to dict."""
        spec = Spec(str, name="message")
        operable = Operable((spec,))
        Model = PydanticSpecAdapter.compose_structure(operable, "TestModel")

        instance = Model(message="Hello")
        result = PydanticSpecAdapter.dump_instance(instance)

        assert result == {"message": "Hello"}
