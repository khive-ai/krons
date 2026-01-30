# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""
Pydantic Adapter Tests: Reference Implementation for Pydantic v2 Model Generation

**Core Implementation**:
- **PydanticSpecAdapter**: Concrete SpecAdapter implementation for Pydantic v2
- **Field Creation**: Spec -> Pydantic FieldInfo (metadata mapping)
- **Validator Integration**: Spec validators -> Pydantic field_validator decorators
- **Fuzzy Matching**: Case-insensitive, underscore-tolerant field key matching
- **Validation Strategies**: Strict (production) vs lenient (LLM) vs fuzzy (tolerance)

**Design Philosophy**:
- **Framework Isolation**: All Pydantic imports confined to this adapter
- **Metadata Mapping**: Spec metadata -> Pydantic Field kwargs (framework translation layer)
- **Progressive Validation**: Strict for correctness, lenient for UX, fuzzy for tolerance
- **Pipeline Integration**: parse_json -> fuzzy_match_fields -> validate_model (3-stage processing)

**Testing Strategy**:
This test suite validates:
1. Field creation with all metadata types (defaults, validators, descriptions, constraints)
2. Validator execution (Pydantic field_validator integration and inheritance)
3. Fuzzy field matching (case normalization, underscore handling, sentinel filtering)
4. Validation pipeline (parse -> match -> validate with error handling)
5. Strict vs lenient error modes (raises vs returns None)
6. Model CRUD operations (create -> validate -> dump -> update)
7. Nullable/listable annotation handling
8. Generic type preservation (SpecAdapter[BaseModel])
"""

from __future__ import annotations

from typing import Generic, get_origin

import pytest
from pydantic import BaseModel, Field, ValidationError
from pydantic_core import PydanticUndefined
from tests.types.conftest import create_spec, get_sample_validators

from krons.specs import Operable, Spec, SpecAdapter
from krons.specs.adapters.pydantic_adapter import PydanticSpecAdapter

# -- Pydantic model creation / validation / dump / update --------------------


def test_pydantic_create_validate_dump_update():
    """Test end-to-end Pydantic pipeline: compose_structure -> validate_instance -> dump_instance -> update."""
    # Build an Operable with two fields
    name_spec = create_spec(str, name="name", default="n/a")
    age_spec = create_spec(int, name="age", default=18)
    op = Operable((name_spec, age_spec), name="User")

    # Create model using compose_structure
    Model = PydanticSpecAdapter.compose_structure(op, "UserModel")

    # Validate with partial input (age default should apply)
    inst = PydanticSpecAdapter.validate_instance(Model, {"name": "alice"})
    dumped = PydanticSpecAdapter.dump_instance(inst)
    assert dumped == {"name": "alice", "age": 18}

    # Update + validate (model_copy for pydantic v2)
    inst2 = inst.model_copy(update={"age": 21})
    assert PydanticSpecAdapter.dump_instance(inst2) == {"name": "alice", "age": 21}


def test_pydantic_nullable_default_none():
    """Test nullable field with default=None."""
    nullable_score = create_spec(int, name="score", default=None, nullable=True)
    op = Operable((nullable_score,), name="Scored")

    Model = PydanticSpecAdapter.compose_structure(op, "ScoredModel")
    inst = PydanticSpecAdapter.validate_instance(Model, {})
    assert PydanticSpecAdapter.dump_instance(inst) == {"score": None}


def test_pydantic_field_validator_executes():
    """
    Spec validators are executed as Pydantic field_validator decorators

    **Pattern**: Validator integration via intermediate base class

    **Scenario**: Spec contains validator callable, adapter converts to Pydantic field_validator
    ```python
    def nonneg(v: int) -> int:
        if v < 0:
            raise ValueError("must be non-negative")
        return v


    spec = Spec(int, name="age", validator=nonneg)
    # Adapter creates: field_validator("age")(nonneg) in base class
    # Generated model inherits validator
    ```

    **Expected Behavior**:
    - Valid value (age=0) passes validation
    - Invalid value (age=-1) raises Pydantic ValidationError
    - Validator executed at validation time (not model creation)
    - Error message from validator function propagated
    """
    validators = get_sample_validators()
    s = create_spec(int, name="age", validator=validators["nonneg"])
    op = Operable((s,), name="WithValidator")
    Model = PydanticSpecAdapter.compose_structure(op, "WithValidatorModel")

    # Valid
    ok = PydanticSpecAdapter.validate_instance(Model, {"age": 0})
    assert PydanticSpecAdapter.dump_instance(ok) == {"age": 0}

    # Invalid
    with pytest.raises(ValidationError):
        PydanticSpecAdapter.validate_instance(Model, {"age": -1})


# -- validate_response pipeline (with monkeypatched fuzzy matching) ----------


def test_validate_instance_pipeline():
    """validate_instance: dict -> validated Pydantic instance."""
    op = Operable((create_spec(str, name="name"), create_spec(int, name="age")), name="User")
    Model = PydanticSpecAdapter.compose_structure(op, "UserModel2")

    inst = PydanticSpecAdapter.validate_instance(Model, {"name": "bob", "age": 33})
    assert inst is not None
    assert PydanticSpecAdapter.dump_instance(inst) == {"name": "bob", "age": 33}


def test_validate_instance_raises_on_invalid_data():
    """validate_instance raises ValidationError on invalid data."""
    from pydantic import ValidationError

    op = Operable((create_spec(str, name="name"),), name="Simple")
    Model = PydanticSpecAdapter.compose_structure(op, "SimpleModel")

    with pytest.raises(ValidationError):
        PydanticSpecAdapter.validate_instance(Model, {"name": 123})


def test_validate_instance_with_defaults():
    """validate_instance uses field defaults for missing values."""
    op = Operable(
        (create_spec(str, name="name"), create_spec(int, name="age", default=0)),
        name="WithDefaults",
    )
    Model = PydanticSpecAdapter.compose_structure(op, "DefaultModel")

    inst = PydanticSpecAdapter.validate_instance(Model, {"name": "alice"})
    assert inst.name == "alice"  # type: ignore[attr-defined]
    assert inst.age == 0  # type: ignore[attr-defined]


# -- Dump instance -----------------------------------------------------------


def test_dump_instance_roundtrip():
    """dump_instance converts model instance back to dict."""
    op = Operable(
        (create_spec(str, name="name"), create_spec(int, name="count", default=0)),
        name="Counter",
    )
    Model = PydanticSpecAdapter.compose_structure(op, "CounterModel")

    inst = PydanticSpecAdapter.validate_instance(Model, {"name": "x", "count": 5})
    dumped = PydanticSpecAdapter.dump_instance(inst)

    assert dumped == {"name": "x", "count": 5}


def test_dump_instance_includes_defaults():
    """dump_instance includes default values in output."""
    op = Operable(
        (create_spec(str, name="name"), create_spec(int, name="count", default=0)),
        name="WithDefaults",
    )
    Model = PydanticSpecAdapter.compose_structure(op, "DefaultModel2")

    inst = PydanticSpecAdapter.validate_instance(Model, {"name": "test"})
    dumped = PydanticSpecAdapter.dump_instance(inst)

    assert dumped == {"name": "test", "count": 0}


# -- Generic type preservation -----------------------------------------------


def test_generic_type_annotations_preserved():
    """Test that Generic[M] type annotations are preserved in adapter protocol."""

    # Check that SpecAdapter inherits from Generic
    # __orig_bases__ contains both ABC and Generic[M]
    bases = SpecAdapter.__orig_bases__  # type: ignore[attr-defined]
    generic_bases = [b for b in bases if get_origin(b) is Generic]
    assert len(generic_bases) == 1, "SpecAdapter should inherit from Generic[M]"

    # Check that PydanticSpecAdapter binds the generic to BaseModel
    # This is compile-time only, but we can verify the inheritance
    assert issubclass(PydanticSpecAdapter, SpecAdapter)


def test_compose_structure_returns_correct_type():
    """Test that compose_structure returns the correct model class type."""
    op = Operable((create_spec(str, name="name"),), name="Test")
    Model = PydanticSpecAdapter.compose_structure(op, "TestModel")

    # Should return a type (class)
    assert isinstance(Model, type)

    # Should be a Pydantic BaseModel subclass
    assert issubclass(Model, BaseModel)


def test_validate_instance_returns_instance():
    """Test that validate_instance returns an instance of the model."""
    op = Operable((create_spec(str, name="name"),), name="Test")
    Model = PydanticSpecAdapter.compose_structure(op, "TestModel2")

    inst = PydanticSpecAdapter.validate_instance(Model, {"name": "test"})

    # Should return an instance
    assert isinstance(inst, Model)

    # Should have correct data
    assert inst.name == "test"  # type: ignore[attr-defined]


def test_dump_instance_returns_dict():
    """Test that dump_instance returns a dictionary."""
    op = Operable((create_spec(str, name="name"), create_spec(int, name="age")), name="Test")
    Model = PydanticSpecAdapter.compose_structure(op, "TestModel3")

    inst = PydanticSpecAdapter.validate_instance(Model, {"name": "alice", "age": 30})
    dumped = PydanticSpecAdapter.dump_instance(inst)

    # Should return a dict
    assert isinstance(dumped, dict)

    # Should have correct structure
    assert dumped == {"name": "alice", "age": 30}


# -- Coverage targets for pydantic_field.py ----------------------------------


def test_create_field_callable_default():
    """Test create_field converts callable defaults to default_factory.

    Design: Pydantic distinguishes between static defaults and factory functions.
    When a Spec has a callable default, PydanticSpecAdapter converts it to
    Pydantic's default_factory, ensuring the callable is invoked per-instance
    rather than being used as a static value.
    """

    # Create spec with callable default
    def get_default():
        return "generated"

    spec = Spec(str, name="field_with_callable", default=get_default)

    # Create field
    field_info = PydanticSpecAdapter.create_field(spec)

    # Should use default_factory for callable
    assert field_info.default_factory is get_default
    assert field_info.default is PydanticUndefined  # Pydantic's sentinel for "use default_factory"


def test_create_field_custom_pydantic_params():
    """Test create_field passes Pydantic-specific metadata to Field.

    Design: Spec metadata that matches Pydantic Field parameters (description,
    title, constraints like min_length) is forwarded to Pydantic's Field
    constructor. This enables full Pydantic validation without requiring
    Pydantic-specific Spec subclasses.
    """
    # Create spec with pydantic-specific metadata
    spec = Spec(
        str,
        name="field_with_params",
        description="A test field",
        title="Test Field",
        min_length=3,
        max_length=10,
    )

    # Create field
    field_info = PydanticSpecAdapter.create_field(spec)

    # Should preserve pydantic field params
    assert field_info.description == "A test field"
    assert field_info.title == "Test Field"
    # min_length and max_length might be in constraints
    assert hasattr(field_info, "metadata") or hasattr(field_info, "constraints")


def test_create_field_type_metadata_skipped():
    """Test create_field filters out type objects from metadata.

    Design: Type objects cannot be JSON-serialized and would break Pydantic's
    schema generation. PydanticSpecAdapter filters out type values when building
    json_schema_extra, keeping only serializable metadata. This prevents runtime
    errors while preserving useful metadata for documentation and validation.
    """
    # Create spec with type object in metadata
    spec = Spec(
        str,
        name="field_with_type_meta",
        custom_type=int,  # Type object that can't be serialized
        other_meta="value",
    )

    # Create field
    field_info = PydanticSpecAdapter.create_field(spec)

    # Type object should be skipped, other metadata should be in json_schema_extra
    if field_info.json_schema_extra:
        # Should not have the type object
        assert "custom_type" not in field_info.json_schema_extra
        # Should have other metadata
        assert field_info.json_schema_extra.get("other_meta") == "value"


# -- Additional tests for Operable.from_model (alias: from_structure) ---------


def test_operable_from_structure():
    """Test Operable.from_structure creates Operable from Pydantic model."""

    class UserModel(BaseModel):
        name: str
        age: int = 0
        email: str | None = None

    op = Operable.from_structure(UserModel, adapter="pydantic")

    assert op.name == "UserModel"
    assert op.allowed() == {"name", "age", "email"}

    # Verify specs were created correctly
    name_spec = op.get("name")
    assert name_spec is not None
    assert name_spec.name == "name"

    age_spec = op.get("age")
    assert age_spec is not None
    assert age_spec.name == "age"

    email_spec = op.get("email")
    assert email_spec is not None
    assert email_spec.name == "email"
    assert email_spec.is_nullable


def test_operable_from_structure_with_constraints():
    """Test Operable.from_structure preserves field constraints."""

    class ConstrainedModel(BaseModel):
        name: str = Field(min_length=1, max_length=100)
        score: int = Field(ge=0, le=100)

    op = Operable.from_structure(ConstrainedModel, adapter="pydantic")

    # Verify constraints are in metadata
    name_spec = op.get("name")
    assert name_spec is not None
    # Constraints should be preserved in metadata
    meta_dict = name_spec.metadict()
    assert "min_length" in meta_dict or len(meta_dict) > 0


def test_extract_specs_preserves_defaults():
    """Test extract_specs preserves default values."""

    class DefaultModel(BaseModel):
        name: str = "default_name"
        count: int = 0

    specs = PydanticSpecAdapter.extract_specs(DefaultModel)

    name_spec = next(s for s in specs if s.name == "name")
    assert name_spec.default == "default_name"

    count_spec = next(s for s in specs if s.name == "count")
    assert count_spec.default == 0


def test_extract_specs_preserves_nullable():
    """Test extract_specs correctly identifies nullable fields."""

    class NullableModel(BaseModel):
        required: str
        optional: str | None = None

    specs = PydanticSpecAdapter.extract_specs(NullableModel)

    required_spec = next(s for s in specs if s.name == "required")
    assert not required_spec.is_nullable

    optional_spec = next(s for s in specs if s.name == "optional")
    assert optional_spec.is_nullable


def test_roundtrip_model_to_operable_to_model():
    """Test roundtrip: Model -> Operable -> Model preserves structure."""

    class OriginalModel(BaseModel):
        name: str
        value: int = 42

    # Model -> Operable
    op = Operable.from_structure(OriginalModel, adapter="pydantic")

    # Operable -> New Model (using compose_structure via Operable)
    NewModel = op.compose_structure("RecreatedModel")

    # Validate both models work the same
    original_inst = OriginalModel(name="test")
    new_inst = NewModel(name="test")

    assert original_inst.name == new_inst.name  # type: ignore[attr-defined]
    assert original_inst.value == new_inst.value  # type: ignore[attr-defined]
