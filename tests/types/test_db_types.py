# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for database type annotations (FK, Vector)."""

from typing import Annotated, get_args, get_origin
from uuid import UUID

import pytest
from pydantic import BaseModel, Field

from krons.types import FK, FKMeta, Unset, Vector, VectorMeta, extract_kron_db_meta


class MockTenant:
    """Mock tenant model for testing."""

    _table_name = "tenants"


class MockUser:
    """Mock user model for testing."""

    _table_name = "users"


class TestFKMeta:
    """Tests for FKMeta metadata class."""

    def test_fkmeta_init_defaults(self):
        """FKMeta should have sensible defaults."""
        meta = FKMeta(MockUser)

        assert meta.model is MockUser
        assert meta.column == "id"
        assert meta.on_delete == "CASCADE"
        assert meta.on_update == "CASCADE"

    def test_fkmeta_custom_column(self):
        """FKMeta should accept custom column."""
        meta = FKMeta(MockUser, column="user_id")
        assert meta.column == "user_id"

    def test_fkmeta_custom_referential_actions(self):
        """FKMeta should accept custom referential actions."""
        meta = FKMeta(MockUser, on_delete="SET NULL", on_update="NO ACTION")
        assert meta.on_delete == "SET NULL"
        assert meta.on_update == "NO ACTION"

    def test_fkmeta_table_name_from_class(self):
        """FKMeta should derive table name from class _table_name."""
        meta = FKMeta(MockUser)
        assert meta.table_name == "users"

    def test_fkmeta_table_name_from_string(self):
        """FKMeta should derive table name from string reference."""
        meta = FKMeta("User")
        assert meta.table_name == "users"

    def test_fkmeta_is_resolved(self):
        """FKMeta should track resolution status."""
        string_meta = FKMeta("User")
        class_meta = FKMeta(MockUser)

        assert not string_meta.is_resolved
        assert class_meta.is_resolved

    def test_fkmeta_resolve(self):
        """FKMeta should allow resolution of string references."""
        meta = FKMeta("User")
        assert not meta.is_resolved

        meta.resolve(MockUser)
        assert meta.is_resolved
        assert meta.model is MockUser

    def test_fkmeta_repr(self):
        """FKMeta should have readable repr."""
        assert repr(FKMeta(MockUser)) == "FK[MockUser]"
        assert repr(FKMeta("User")) == "FK[User]"


class TestFK:
    """Tests for FK[T] type annotation."""

    def test_fk_subscript_returns_annotated(self):
        """FK[Model] should return Annotated[UUID, FKMeta(Model)]."""
        result = FK[MockUser]

        assert get_origin(result) is Annotated
        args = get_args(result)
        assert len(args) == 2
        assert args[0] is UUID
        assert isinstance(args[1], FKMeta)
        assert args[1].model is MockUser

    def test_fk_different_types(self):
        """FK[User] and FK[Tenant] should be distinct."""
        user_fk = FK[MockUser]
        tenant_fk = FK[MockTenant]

        user_meta = get_args(user_fk)[1]
        tenant_meta = get_args(tenant_fk)[1]

        assert user_meta.model is MockUser
        assert tenant_meta.model is MockTenant

    def test_fk_string_forward_ref(self):
        """FK should work with string forward references."""
        result = FK["ForwardRef"]

        args = get_args(result)
        assert isinstance(args[1], FKMeta)
        assert args[1].model == "ForwardRef"


class TestFKMetaExtraction:
    """Tests for extract_kron_db_meta(field_info, metas='FK') extraction."""

    def test_fk_meta_extraction_direct(self):
        """fk_meta should extract from direct FK[Model] field."""

        class TestModel(BaseModel):
            tenant_id: FK[MockTenant]

        field_info = TestModel.model_fields["tenant_id"]
        meta = extract_kron_db_meta(field_info, metas="FK")

        assert meta is not Unset
        assert isinstance(meta, FKMeta)
        assert meta.model is MockTenant

    def test_fk_meta_extraction_optional(self):
        """fk_meta should extract from FK[Model] | None field."""

        class TestModel(BaseModel):
            tenant_id: FK[MockTenant] | None = None

        field_info = TestModel.model_fields["tenant_id"]
        meta = extract_kron_db_meta(field_info, metas="FK")

        assert meta is not Unset
        assert meta.model is MockTenant

    def test_fk_meta_extraction_non_fk(self):
        """fk_meta should return Unset for non-FK fields."""

        class TestModel(BaseModel):
            name: str

        field_info = TestModel.model_fields["name"]
        meta = extract_kron_db_meta(field_info, metas="FK")

        assert meta is Unset


class TestVectorMeta:
    """Tests for VectorMeta metadata class."""

    def test_vectormeta_init(self):
        """VectorMeta should store dimension."""
        meta = VectorMeta(1536)
        assert meta.dim == 1536

    def test_vectormeta_positive_dim_required(self):
        """VectorMeta should reject non-positive dimensions."""
        with pytest.raises(ValueError, match="positive"):
            VectorMeta(0)
        with pytest.raises(ValueError, match="positive"):
            VectorMeta(-1)

    def test_vectormeta_repr(self):
        """VectorMeta should have readable repr."""
        assert repr(VectorMeta(768)) == "Vector[768]"


class TestVector:
    """Tests for Vector[dim] type annotation."""

    def test_vector_subscript_returns_annotated(self):
        """Vector[dim] should return Annotated[list[float], VectorMeta(dim)]."""
        result = Vector[1536]

        assert get_origin(result) is Annotated
        args = get_args(result)
        assert len(args) == 2
        assert args[0] == list[float]
        assert isinstance(args[1], VectorMeta)
        assert args[1].dim == 1536

    def test_vector_common_dimensions(self):
        """Vector should work with common embedding dimensions."""
        dims = [384, 768, 1024, 1536, 3072]

        for dim in dims:
            result = Vector[dim]
            args = get_args(result)
            assert args[1].dim == dim


class TestVectorMetaExtraction:
    """Tests for extract_kron_db_meta(field_info, metas='Vector') extraction."""

    def test_vector_meta_extraction_direct(self):
        """vector_meta should extract from direct Vector[dim] field."""

        class TestModel(BaseModel):
            embedding: Vector[1536]

        field_info = TestModel.model_fields["embedding"]
        meta = extract_kron_db_meta(field_info, metas="Vector")

        assert meta is not Unset
        assert isinstance(meta, VectorMeta)
        assert meta.dim == 1536

    def test_vector_meta_extraction_optional(self):
        """vector_meta should extract from Vector[dim] | None field."""

        class TestModel(BaseModel):
            embedding: Vector[768] | None = None

        field_info = TestModel.model_fields["embedding"]
        meta = extract_kron_db_meta(field_info, metas="Vector")

        assert meta is not Unset
        assert meta.dim == 768

    def test_vector_meta_extraction_non_vector(self):
        """vector_meta should return Unset for non-Vector fields."""

        class TestModel(BaseModel):
            values: list[float] = Field(default_factory=list)

        field_info = TestModel.model_fields["values"]
        meta = extract_kron_db_meta(field_info, metas="Vector")

        assert meta is Unset
