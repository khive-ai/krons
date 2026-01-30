# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.types.spec - Spec field specification.

Spec is the framework-agnostic field schema that stores type and metadata.
Key behaviors:
- Create from type annotations
- Create from Pydantic field info
- Support nullable, listable, default values
- Generate Annotated types for framework adapters
"""

import pytest
from pydantic import BaseModel, Field

from krons.specs import Spec
from krons.specs.spec import CommonMeta
from krons.types import Undefined


# Async fixtures for testing
async def async_default_factory():
    """Async factory for testing."""
    return "async_value"


class TestCommonMeta:
    """Test CommonMeta validation utilities."""

    def test_allowed_returns_all_values(self):
        """CommonMeta.allowed() returns all standard metadata keys."""
        allowed = CommonMeta.allowed()
        assert "name" in allowed
        assert "nullable" in allowed
        assert "listable" in allowed
        assert "validator" in allowed
        assert "default" in allowed
        assert "default_factory" in allowed
        assert "frozen" in allowed
        assert "as_fk" in allowed
        assert len(allowed) == 8

    def test_validate_rejects_both_default_and_factory(self):
        """Validation rejects conflicting default + default_factory."""
        with pytest.raises(ExceptionGroup, match="Metadata validation failed"):
            CommonMeta._validate_common_metas(default="value", default_factory=lambda: "value")

    def test_validate_rejects_non_callable_factory(self):
        """Validation rejects non-callable default_factory."""
        with pytest.raises(ExceptionGroup, match="Metadata validation failed"):
            CommonMeta._validate_common_metas(default_factory="not_a_function")

    def test_validate_rejects_non_callable_validator(self):
        """Validation rejects non-callable validators."""
        with pytest.raises(ExceptionGroup, match="Metadata validation failed"):
            CommonMeta._validate_common_metas(validator="not_callable")

    def test_prepare_success(self):
        """prepare() successfully creates metadata tuple from kwargs."""
        result = CommonMeta.prepare(name="field", nullable=True)
        assert len(result) == 2
        meta_dict = {m.key: m.value for m in result}
        assert meta_dict["name"] == "field"
        assert meta_dict["nullable"] is True


class TestSpecCreation:
    """Test Spec instantiation."""

    def test_spec_from_annotation(self):
        """Spec.from_annotation() should extract type info."""
        spec = Spec(str, name="username")
        assert spec.base_type == str
        assert spec.name == "username"

    def test_spec_from_pydantic_field(self):
        """Spec.from_pydantic_field() should extract field info."""
        # Create a spec with pydantic-compatible metadata
        spec = Spec(str, name="field", default="default_value", nullable=True)
        assert spec.name == "field"
        assert spec.default == "default_value"
        assert spec.is_nullable is True

    def test_basic_creation(self):
        """Basic Spec creation with minimal metadata."""
        spec = Spec(str, name="username")
        assert spec.base_type == str
        assert spec.name == "username"

    def test_invalid_base_type_raises(self):
        """Spec creation requires valid type for base_type."""
        with pytest.raises(ValueError, match="must be a type"):
            Spec(42, name="field")

    def test_string_base_type_rejected(self):
        """String base_type is no longer valid - must be a type or annotation."""
        with pytest.raises(ValueError, match="must be a type"):
            Spec("Person", name="subject_id")

    def test_name_cannot_be_none(self):
        """Spec name must be a valid string, not None."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            Spec(str, name=None)

    def test_name_cannot_be_empty_string(self):
        """Spec name must be non-empty string."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            Spec(str, name="")

    def test_name_must_be_string_type(self):
        """Spec name must be string type, not other types."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            Spec(str, name=123)


class TestSpecProperties:
    """Test Spec properties."""

    def test_spec_name(self):
        """Spec should have name property."""
        spec = Spec(str, name="username")
        assert spec.name == "username"

    def test_spec_base_type(self):
        """Spec should have base_type property."""
        spec = Spec(int, name="age")
        assert spec.base_type == int

    def test_spec_nullable(self):
        """Spec should track nullable status."""
        spec = Spec(int, name="age", nullable=True)
        assert spec.is_nullable is True

        spec_not_nullable = Spec(int, name="age")
        assert spec_not_nullable.is_nullable is False

    def test_spec_default(self):
        """Spec should track default value."""
        spec = Spec(str, name="field", default="default_value")
        assert spec.default == "default_value"

    def test_spec_default_undefined(self):
        """Spec without default should return Undefined."""
        spec = Spec(str, name="field")
        assert spec.default is Undefined

    def test_spec_is_listable(self):
        """Spec should track listable status."""
        spec = Spec(int, name="scores", listable=True)
        assert spec.is_listable is True

        spec_not_listable = Spec(int, name="scores")
        assert spec_not_listable.is_listable is False


class TestSpecDefaultFactory:
    """Test Spec default_factory behavior."""

    def test_default_factory(self):
        """Default factory generates fresh values on each call."""
        spec = Spec(list, name="field", default_factory=list)
        assert spec.has_default_factory is True
        result = spec.create_default_value()
        assert isinstance(result, list)

    def test_async_default_factory_warning(self):
        """Async default factory emits warning."""

        async def async_factory():
            return "value"

        with pytest.warns(UserWarning, match="Async default factories"):
            spec = Spec(str, name="field", default_factory=async_factory)
        assert spec.has_async_default_factory is True

    def test_create_default_without_default_raises(self):
        """create_default_value() requires default or factory."""
        spec = Spec(str, name="field")
        with pytest.raises(ValueError, match="No default value"):
            spec.create_default_value()

    def test_create_default_with_async_factory_raises(self):
        """Sync create_default_value() rejects async factory."""

        async def async_factory():
            return "value"

        with pytest.warns(UserWarning):
            spec = Spec(str, name="field", default_factory=async_factory)

        with pytest.raises(ValueError, match="asynchronous"):
            spec.create_default_value()


class TestSpecFluentAPI:
    """Test Spec fluent API methods."""

    def test_as_nullable(self):
        """as_nullable() creates new Spec with nullable=True."""
        spec = Spec(str, name="field")
        nullable_spec = spec.as_nullable()
        assert nullable_spec.is_nullable is True
        assert nullable_spec.name == "field"

    def test_as_listable(self):
        """as_listable() creates new Spec with listable=True."""
        spec = Spec(int, name="field")
        listable_spec = spec.as_listable()
        assert listable_spec.is_listable is True
        assert listable_spec.name == "field"

    def test_with_default(self):
        """with_default() adds static default to Spec."""
        spec = Spec(str, name="field")
        spec_with_default = spec.with_default("value")
        assert spec_with_default.default == "value"

    def test_with_default_factory(self):
        """with_default() detects callables and treats as factory."""
        spec = Spec(list, name="field")
        spec_with_factory = spec.with_default(list)
        assert spec_with_factory.has_default_factory is True

    def test_with_validator(self):
        """with_validator() adds validation function to Spec."""

        def validator(v):
            return len(v) > 0

        spec = Spec(str, name="field")
        spec_with_validator = spec.with_validator(validator)
        assert spec_with_validator.get("validator") == validator

    def test_with_updates(self):
        """with_updates() creates new Spec with modified metadata."""
        spec = Spec(str, name="field", nullable=False)
        updated = spec.with_updates(nullable=True, custom="value")
        assert updated.is_nullable is True
        assert updated.get("custom") == "value"
        assert updated.name == "field"


class TestSpecAnnotation:
    """Test Spec annotation generation."""

    def test_annotation_basic(self):
        """annotation property returns base_type for simple Spec."""
        spec = Spec(str, name="field")
        assert spec.annotation == str

    def test_annotation_nullable(self):
        """annotation generates Union[T, None] for nullable Spec."""
        spec = Spec(str, name="field", nullable=True)
        assert spec.annotation == str | None

    def test_annotation_listable(self):
        """annotation generates list[T] for listable Spec."""
        spec = Spec(int, name="field", listable=True)
        assert spec.annotation == list[int]

    def test_annotation_nullable_listable(self):
        """annotation combines nullable and listable correctly."""
        spec = Spec(int, name="field", nullable=True, listable=True)
        assert spec.annotation == list[int] | None


class TestSpecMetadataAccess:
    """Test Spec metadata access methods."""

    def test_getitem(self):
        """__getitem__ provides dict-like metadata access."""
        spec = Spec(str, name="field", custom="value")
        assert spec["name"] == "field"
        assert spec["custom"] == "value"

    def test_getitem_missing_raises(self):
        """__getitem__ raises KeyError on missing metadata key."""
        spec = Spec(str, name="field")
        with pytest.raises(KeyError, match="Metadata key 'missing'"):
            _ = spec["missing"]

    def test_get_with_default(self):
        """get() provides lenient metadata access with default fallback."""
        spec = Spec(str, name="field")
        assert spec.get("missing", "default") == "default"
        assert spec.get("name") == "field"

    def test_metadict(self):
        """metadict() converts metadata tuple to dict."""
        spec = Spec(str, name="field", nullable=True, custom="value")
        metadict = spec.metadict()
        assert metadict["name"] == "field"
        assert metadict["nullable"] is True
        assert metadict["custom"] == "value"

    def test_metadict_exclude(self):
        """metadict() supports selective key exclusion."""
        spec = Spec(str, name="field", nullable=True, custom="value")
        metadict = spec.metadict(exclude={"name"})
        assert "name" not in metadict
        assert metadict["nullable"] is True

    def test_metadict_exclude_common(self):
        """metadict() can exclude all CommonMeta keys."""
        spec = Spec(str, name="field", nullable=True, custom="value")
        metadict = spec.metadict(exclude_common=True)
        assert "name" not in metadict
        assert "nullable" not in metadict
        assert metadict["custom"] == "value"


class TestSpecImmutability:
    """Test Spec immutability (frozen dataclass)."""

    def test_immutability(self):
        """Spec is immutable (frozen dataclass)."""
        spec = Spec(str, name="field")
        with pytest.raises((AttributeError, TypeError)):  # Frozen dataclass errors
            spec.base_type = int


class TestSpecAnnotatedCaching:
    """Test Spec annotated() caching behavior."""

    def test_annotated_caching(self):
        """annotated() returns cached result for same Spec."""
        spec = Spec(str, name="field")
        annotated1 = spec.annotated()
        annotated2 = spec.annotated()
        # Should return same object from cache
        assert annotated1 is annotated2


class TestSpecFromModel:
    """Test Spec.from_model() class method."""

    def test_from_model_basic(self):
        """Spec.from_model() creates Spec from model class."""

        class MyModel(BaseModel):
            value: int

        spec = Spec.from_model(MyModel)
        assert spec.base_type is MyModel
        assert spec.name == "mymodel"

    def test_from_model_with_name(self):
        """Spec.from_model() accepts custom name."""

        class MyModel(BaseModel):
            value: int

        spec = Spec.from_model(MyModel, name="custom")
        assert spec.name == "custom"

    def test_from_model_listable(self):
        """Spec.from_model() can create listable spec."""

        class MyModel(BaseModel):
            value: int

        spec = Spec.from_model(MyModel, listable=True)
        assert spec.is_listable is True

    def test_from_model_nullable(self):
        """Spec.from_model() can create nullable spec."""

        class MyModel(BaseModel):
            value: int

        spec = Spec.from_model(MyModel, nullable=True)
        assert spec.is_nullable is True

    def test_from_model_with_default(self):
        """Spec.from_model() can add default value."""

        class MyModel(BaseModel):
            value: int

        default_instance = MyModel(value=42)
        spec = Spec.from_model(MyModel, default=default_instance)
        assert spec.default == default_instance


class TestSpecAsyncDefaultFactory:
    """Test Spec async default factory behavior."""

    @pytest.mark.anyio
    async def test_acreate_default_value_async_factory(self):
        """acreate_default_value() executes async factory."""
        with pytest.warns(UserWarning):
            spec = Spec(str, name="field", default_factory=async_default_factory)

        result = await spec.acreate_default_value()
        assert result == "async_value"

    @pytest.mark.anyio
    async def test_acreate_default_value_sync_fallback(self):
        """acreate_default_value() handles sync defaults in async context."""
        # Test with static default
        spec_static = Spec(str, name="field", default="static_value")
        result_static = await spec_static.acreate_default_value()
        assert result_static == "static_value"

        # Test with sync factory
        spec_factory = Spec(list, name="field", default_factory=list)
        result_factory = await spec_factory.acreate_default_value()
        assert isinstance(result_factory, list)


class TestSpecForeignKey:
    """Test Spec as_fk / is_fk / fk_target functionality."""

    def test_as_fk_with_string_target(self):
        """as_fk(str) stores target model name."""
        from uuid import UUID

        spec = Spec(UUID, name="user_id").as_fk("User")
        assert spec.is_fk is True
        assert spec.fk_target == "User"

    def test_as_fk_with_type_target(self):
        """as_fk(type) stores target model class."""
        from uuid import UUID

        class Tenant:
            pass

        spec = Spec(UUID, name="tenant_id").as_fk(Tenant)
        assert spec.is_fk is True
        assert spec.fk_target is Tenant

    def test_as_fk_without_target_non_observable(self):
        """as_fk() on non-Observable base_type marks FK but fk_target is Undefined."""
        from uuid import UUID

        from krons.types._sentinel import is_undefined

        spec = Spec(UUID, name="ref_id").as_fk()
        assert spec.is_fk is True
        assert is_undefined(spec.fk_target)

    def test_as_fk_without_target_observable(self):
        """as_fk() on Observable base_type resolves target to base_type."""
        from uuid import UUID

        class MyEntity:
            @property
            def id(self) -> UUID:
                return UUID(int=0)

        spec = Spec(MyEntity, name="entity").as_fk()
        assert spec.is_fk is True
        assert spec.fk_target is MyEntity

    def test_with_updates_as_fk_observable(self):
        """with_updates(as_fk=True) resolves target from Observable base_type."""
        from uuid import UUID

        class Person:
            @property
            def id(self) -> UUID:
                return UUID(int=0)

        person = Spec(Person, name="person")
        person_id = person.with_updates(name="person_id", as_fk=True)
        assert person_id.is_fk is True
        assert person_id.fk_target is Person

    def test_is_fk_false_by_default(self):
        """Spec without as_fk is not FK."""
        from krons.types._sentinel import is_undefined

        spec = Spec(str, name="field")
        assert spec.is_fk is False
        assert is_undefined(spec.fk_target)

    def test_annotated_includes_fkmeta(self):
        """annotated() includes FKMeta and uses UUID when FK target resolves."""
        from typing import get_args
        from uuid import UUID

        from krons.types.db_types import FKMeta

        spec = Spec(str, name="user_id").as_fk("User")
        annotated = spec.annotated()

        # Base type should be UUID (FK replaces base_type)
        args = get_args(annotated)
        assert args[0] is UUID

        fk_metas = [a for a in args if isinstance(a, FKMeta)]
        assert len(fk_metas) == 1
        assert fk_metas[0].model == "User"
        assert fk_metas[0].table_name == "users"

    def test_annotated_no_fkmeta_without_target(self):
        """annotated() does not include FKMeta when target cannot resolve."""
        from typing import get_args
        from uuid import UUID

        from krons.types.db_types import FKMeta

        # UUID is not Observable, so as_fk=True without target → no FKMeta
        spec = Spec(UUID, name="ref_id").as_fk()
        annotated = spec.annotated()

        args = get_args(annotated)
        fk_metas = [a for a in args if isinstance(a, FKMeta)]
        assert len(fk_metas) == 0

    def test_annotated_fk_observable(self):
        """annotated() resolves FK target from Observable base_type."""
        from typing import get_args
        from uuid import UUID

        from krons.types.db_types import FKMeta

        class Person:
            @property
            def id(self) -> UUID:
                return UUID(int=0)

        spec = Spec(Person, name="person_id").as_fk()
        annotated = spec.annotated()

        args = get_args(annotated)
        # Base type becomes UUID
        assert args[0] is UUID
        fk_metas = [a for a in args if isinstance(a, FKMeta)]
        assert len(fk_metas) == 1
        assert fk_metas[0].model is Person

    def test_as_fk_nullable(self):
        """as_fk combined with nullable produces UUID | None with FKMeta."""
        from typing import get_args, get_origin
        from uuid import UUID

        from krons.types.db_types import FKMeta

        spec = Spec(UUID, name="org_id").as_fk("Organization").as_nullable()
        annotated = spec.annotated()

        args = get_args(annotated)
        # The base type should be UUID | None
        base = args[0]
        assert type(None) in get_args(base)

        fk_metas = [a for a in args if isinstance(a, FKMeta)]
        assert len(fk_metas) == 1
        assert fk_metas[0].model == "Organization"

    def test_as_fk_preserves_other_metadata(self):
        """as_fk preserves existing metadata (name, nullable, etc.)."""
        from uuid import UUID

        spec = Spec(UUID, name="user_id", nullable=True).as_fk("User")
        assert spec.name == "user_id"
        assert spec.is_nullable is True
        assert spec.is_fk is True
        assert spec.fk_target == "User"

    def test_as_fk_immutable(self):
        """as_fk returns a new Spec (original unchanged)."""
        from uuid import UUID

        original = Spec(UUID, name="user_id")
        fk_spec = original.as_fk("User")

        assert original.is_fk is False
        assert fk_spec.is_fk is True

    def test_annotation_includes_fk(self):
        """annotation property produces FK[target] = Annotated[UUID, FKMeta]."""
        from typing import get_args
        from uuid import UUID

        from krons.types.db_types import FKMeta

        # Even with str base_type, FK resolves to UUID
        spec = Spec(str, name="user_id").as_fk("User")
        ann = spec.annotation

        args = get_args(ann)
        assert args[0] is UUID
        fk_metas = [a for a in args if isinstance(a, FKMeta)]
        assert len(fk_metas) == 1
        assert fk_metas[0].model == "User"

    def test_annotation_fk_observable(self):
        """annotation resolves FK from Observable base_type."""
        from typing import get_args
        from uuid import UUID

        from krons.types.db_types import FKMeta

        class Person:
            @property
            def id(self) -> UUID:
                return UUID(int=0)

        spec = Spec(Person, name="person_id").as_fk()
        ann = spec.annotation

        args = get_args(ann)
        assert args[0] is UUID
        fk_metas = [a for a in args if isinstance(a, FKMeta)]
        assert len(fk_metas) == 1
        assert fk_metas[0].model is Person

    def test_annotation_fk_nullable_listable(self):
        """annotation property applies FK, then listable, then nullable."""
        from typing import get_args, get_origin
        from uuid import UUID

        from krons.types.db_types import FKMeta

        spec = Spec(UUID, name="ids").as_fk("User").as_listable().as_nullable()
        ann = spec.annotation

        # Outermost: T | None (UnionType)
        union_args = get_args(ann)
        assert type(None) in union_args
        list_type = [a for a in union_args if a is not type(None)][0]

        # Next: list[Annotated[UUID, FKMeta]]
        assert get_origin(list_type) is list
        inner = get_args(list_type)[0]

        # Innermost: Annotated[UUID, FKMeta("User")]
        inner_args = get_args(inner)
        assert inner_args[0] is UUID
        fk_metas = [a for a in inner_args if isinstance(a, FKMeta)]
        assert len(fk_metas) == 1
        assert fk_metas[0].model == "User"

    def test_annotation_no_fk_without_target(self):
        """annotation does not wrap with FKMeta when target cannot resolve."""
        from uuid import UUID

        spec = Spec(UUID, name="ref_id").as_fk()
        ann = spec.annotation
        # UUID is not Observable, so no FK wrapping
        assert ann is UUID
