# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.specs.factory and kron.specs.adapters.factory."""

from datetime import datetime
from typing import Annotated, get_origin
from uuid import UUID

import pytest

from krons.specs import Spec, get_adapter
from krons.specs.adapters.factory import AdapterType
from krons.specs.factory import (
    create_change_by_spec,
    create_content_spec,
    create_datetime_spec,
    create_embedding_spec,
    create_uuid_spec,
)
from krons.types import Unset


class TestCreateDatetimeSpec:
    """Tests for create_datetime_spec factory."""

    def test_with_default(self):
        """Should create datetime Spec with default_factory when use_default=True."""
        spec = create_datetime_spec("created_at", use_default=True)

        assert isinstance(spec, Spec)
        assert spec.name == "created_at"
        assert spec.base_type is datetime
        assert spec.has_default_factory

    def test_without_default(self):
        """Should create datetime Spec without default when use_default=False."""
        spec = create_datetime_spec("updated_at", use_default=False)

        assert spec.name == "updated_at"
        assert spec.base_type is datetime
        assert not spec.has_default_factory

    def test_has_validator(self):
        """Should have coerce_created_at validator."""
        spec = create_datetime_spec("ts", use_default=True)

        assert spec.get("validator") is not None

    def test_custom_name(self):
        """Should accept arbitrary name."""
        spec = create_datetime_spec("my_timestamp", use_default=True)
        assert spec.name == "my_timestamp"


class TestCreateUuidSpec:
    """Tests for create_uuid_spec factory."""

    def test_with_default(self):
        """Should create UUID Spec with uuid4 factory when use_default=True."""
        spec = create_uuid_spec("id", use_default=True)

        assert isinstance(spec, Spec)
        assert spec.name == "id"
        assert spec.base_type is UUID
        assert spec.has_default_factory

    def test_without_default(self):
        """Should create UUID Spec without default when use_default=False."""
        spec = create_uuid_spec("ref_id", use_default=False)

        assert spec.name == "ref_id"
        assert spec.base_type is UUID
        assert not spec.has_default_factory

    def test_has_validator(self):
        """Should have to_uuid validator."""
        spec = create_uuid_spec("id", use_default=True)
        assert spec.get("validator") is not None

    def test_default_factory_produces_uuid(self):
        """Default factory should produce a UUID value."""
        spec = create_uuid_spec("id", use_default=True)
        value = spec.create_default_value()
        assert isinstance(value, UUID)


class TestCreateContentSpec:
    """Tests for create_content_spec factory."""

    def test_default_dict_type(self):
        """Should default to dict when content_type is Unset."""
        spec = create_content_spec("content")

        assert spec.name == "content"
        assert spec.base_type is dict

    def test_custom_content_type(self):
        """Should use provided content_type."""

        class MyModel:
            pass

        spec = create_content_spec("payload", content_type=MyModel)
        assert spec.base_type is MyModel

    def test_with_default_factory(self):
        """Should add default_factory when use_default=True."""
        spec = create_content_spec("data", use_default=True)

        assert spec.has_default_factory

    def test_without_default(self):
        """Should have no default when use_default=False."""
        spec = create_content_spec("content", use_default=False)

        assert not spec.has_default_factory

    def test_custom_default_factory(self):
        """Should use custom default_factory when provided."""
        spec = create_content_spec("data", use_default=True, default_factory=list)

        value = spec.create_default_value()
        assert value == []


class TestCreateEmbeddingSpec:
    """Tests for create_embedding_spec factory."""

    def test_no_dim_produces_list_float(self):
        """Without dim, should produce list[float] base_type."""
        spec = create_embedding_spec("embedding")

        assert spec.name == "embedding"
        assert spec.base_type == list[float]

    def test_with_dim_produces_vector_annotation(self):
        """With dim, should produce Vector[dim] annotated type."""
        spec = create_embedding_spec("embedding", dim=1536)

        assert spec.name == "embedding"
        assert get_origin(spec.base_type) is Annotated

    def test_common_dimensions(self):
        """Should work with common embedding dimensions."""
        for dim in (384, 768, 1024, 1536, 3072):
            spec = create_embedding_spec("emb", dim=dim)
            assert get_origin(spec.base_type) is Annotated

    def test_no_dim_with_default(self):
        """No dim + use_default should produce list with default_factory=list."""
        spec = create_embedding_spec("emb", use_default=True)

        assert spec.has_default_factory
        value = spec.create_default_value()
        assert value == []

    def test_no_dim_without_default(self):
        """No dim + no use_default should have no default."""
        spec = create_embedding_spec("emb", use_default=False)
        assert not spec.has_default_factory


class TestCreateChangeBySpec:
    """Tests for create_change_by_spec factory."""

    def test_uuid_mode(self):
        """use_uuid=True should produce UUID spec."""
        spec = create_change_by_spec("created_by", use_uuid=True)

        assert spec.name == "created_by"
        assert spec.base_type is UUID

    def test_string_mode(self):
        """use_uuid=False should produce str spec."""
        spec = create_change_by_spec("updated_by", use_uuid=False)

        assert spec.name == "updated_by"
        assert spec.base_type is str

    def test_uuid_has_validator(self):
        """UUID mode should have to_uuid validator."""
        spec = create_change_by_spec("actor", use_uuid=True)
        assert spec.get("validator") is not None

    def test_string_has_no_validator(self):
        """String mode should have no validator."""
        from krons.types._sentinel import is_undefined

        spec = create_change_by_spec("actor", use_uuid=False)
        assert is_undefined(spec.get("validator"))


class TestGetAdapter:
    """Tests for get_adapter factory function."""

    def test_pydantic_adapter(self):
        """get_adapter('pydantic') should return PydanticSpecAdapter."""
        from krons.specs.adapters.pydantic_adapter import PydanticSpecAdapter

        adapter = get_adapter("pydantic")
        assert adapter is PydanticSpecAdapter

    def test_sql_adapter(self):
        """get_adapter('sql') should return SQLSpecAdapter."""
        from krons.specs.adapters.sql_ddl import SQLSpecAdapter

        adapter = get_adapter("sql")
        assert adapter is SQLSpecAdapter

    def test_dataclass_adapter(self):
        """get_adapter('dataclass') should return DataClassSpecAdapter."""
        from krons.specs.adapters.dataclass_field import DataClassSpecAdapter

        adapter = get_adapter("dataclass")
        assert adapter is DataClassSpecAdapter

    def test_unsupported_raises(self):
        """get_adapter with unknown name should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported adapter"):
            get_adapter("unknown")

    def test_cached(self):
        """Repeated calls should return same object (cached)."""
        a1 = get_adapter("pydantic")
        a2 = get_adapter("pydantic")
        assert a1 is a2
