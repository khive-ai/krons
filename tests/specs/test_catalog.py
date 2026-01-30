# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.specs.catalog - BaseModel-based spec catalogs."""

from datetime import datetime
from uuid import UUID

from krons.specs.catalog import (
    AuditSpecs,
    CommonSpecs,
    ContentSpecs,
    EnforcementLevel,
    EnforcementSpecs,
)


class TestAuditSpecs:
    """Tests for AuditSpecs.get_specs()."""

    def test_get_specs_uuid_mode(self):
        """AuditSpecs with use_uuid=True should produce UUID actor fields."""
        specs = AuditSpecs.get_specs(use_uuid=True)

        assert isinstance(specs, list)
        assert len(specs) == 9

        names = [s.name for s in specs]
        assert "updated_at" in names
        assert "updated_by" in names
        assert "is_active" in names
        assert "is_deleted" in names
        assert "deleted_at" in names
        assert "deleted_by" in names
        assert "version" in names
        assert "content_hash" in names
        assert "integrity_hash" in names

    def test_get_specs_string_mode(self):
        """AuditSpecs with use_uuid=False should produce str actor fields."""
        specs = AuditSpecs.get_specs(use_uuid=False)

        assert len(specs) == 9
        by_name = {s.name: s for s in specs}

        # updated_by should be str type when use_uuid=False
        assert by_name["updated_by"].base_type is str

    def test_get_specs_uuid_actor_types(self):
        """Actor fields should be UUID when use_uuid=True."""
        specs = AuditSpecs.get_specs(use_uuid=True)
        by_name = {s.name: s for s in specs}

        assert by_name["updated_by"].base_type is UUID
        assert by_name["deleted_by"].base_type is UUID

    def test_get_specs_defaults(self):
        """Check default values on audit specs."""
        specs = AuditSpecs.get_specs(use_uuid=True)
        by_name = {s.name: s for s in specs}

        assert by_name["is_active"].default is True
        assert by_name["is_deleted"].default is False
        assert by_name["version"].default == 1

    def test_get_specs_datetime_fields(self):
        """Datetime fields should have correct base_type."""
        specs = AuditSpecs.get_specs(use_uuid=True)
        by_name = {s.name: s for s in specs}

        assert by_name["updated_at"].base_type is datetime
        assert by_name["deleted_at"].base_type is datetime

    def test_get_specs_nullable_fields(self):
        """Nullable fields should be marked."""
        specs = AuditSpecs.get_specs(use_uuid=True)
        by_name = {s.name: s for s in specs}

        assert by_name["deleted_at"].is_nullable is True
        assert by_name["updated_by"].is_nullable is True
        assert by_name["deleted_by"].is_nullable is True
        assert by_name["content_hash"].is_nullable is True
        assert by_name["integrity_hash"].is_nullable is True

    def test_basemodel_has_fields(self):
        """AuditSpecs should be a BaseModel with expected fields."""
        assert hasattr(AuditSpecs, "model_fields")
        fields = AuditSpecs.model_fields
        assert "updated_at" in fields
        assert "version" in fields
        assert "is_deleted" in fields


class TestContentSpecs:
    """Tests for ContentSpecs.get_specs()."""

    def test_get_specs_defaults(self):
        """ContentSpecs with defaults should produce 5 specs."""
        specs = ContentSpecs.get_specs()

        assert isinstance(specs, list)
        assert len(specs) == 5

        names = [s.name for s in specs]
        assert "id" in names
        assert "created_at" in names
        assert "content" in names
        assert "metadata" in names
        assert "embedding" in names

    def test_get_specs_id_is_uuid(self):
        """ID spec should be UUID type."""
        specs = ContentSpecs.get_specs()
        by_name = {s.name: s for s in specs}

        assert by_name["id"].base_type is UUID

    def test_get_specs_created_at_is_datetime(self):
        """created_at spec should be datetime type."""
        specs = ContentSpecs.get_specs()
        by_name = {s.name: s for s in specs}

        assert by_name["created_at"].base_type is datetime

    def test_get_specs_content_type_default_dict(self):
        """Content/metadata should default to dict type."""
        from typing import Any, get_origin

        specs = ContentSpecs.get_specs()
        by_name = {s.name: s for s in specs}

        # dict[str, Any] has origin of dict
        assert get_origin(by_name["content"].base_type) is dict
        assert get_origin(by_name["metadata"].base_type) is dict

    def test_get_specs_custom_content_type(self):
        """Custom content_type should propagate."""

        class MyContent:
            pass

        specs = ContentSpecs.get_specs(content_type=MyContent)
        by_name = {s.name: s for s in specs}

        assert by_name["content"].base_type is MyContent
        assert by_name["metadata"].base_type is MyContent

    def test_get_specs_embedding_no_dim(self):
        """Embedding without dim should be float with listable=True."""
        specs = ContentSpecs.get_specs()
        by_name = {s.name: s for s in specs}

        # list[float] is decomposed to float with listable=True
        assert by_name["embedding"].base_type is float
        assert by_name["embedding"].is_listable is True

    def test_get_specs_embedding_with_dim(self):
        """Embedding with dim should have VectorMeta in metadata."""
        specs = ContentSpecs.get_specs(dim=1536)
        by_name = {s.name: s for s in specs}

        from krons.types.db_types import VectorMeta, extract_kron_db_meta

        embedding_spec = by_name["embedding"]
        vec_meta = extract_kron_db_meta(embedding_spec, metas="Vector")
        assert vec_meta is not None
        assert isinstance(vec_meta, VectorMeta)
        assert vec_meta.dim == 1536

    def test_basemodel_has_fields(self):
        """ContentSpecs should be a BaseModel with expected fields."""
        assert hasattr(ContentSpecs, "model_fields")
        fields = ContentSpecs.model_fields
        assert "id" in fields
        assert "created_at" in fields
        assert "embedding" in fields


class TestCommonSpecs:
    """Tests for CommonSpecs.get_specs()."""

    def test_get_specs_defaults(self):
        """CommonSpecs should produce 8 specs."""
        specs = CommonSpecs.get_specs()

        assert isinstance(specs, list)
        assert len(specs) == 8

        names = [s.name for s in specs]
        assert "name" in names
        assert "slug" in names
        assert "status" in names
        assert "email" in names
        assert "phone" in names
        assert "tenant_id" in names
        assert "settings" in names
        assert "data" in names

    def test_get_specs_status_default(self):
        """Status field should have configurable default."""
        specs = CommonSpecs.get_specs(status_default="active")
        by_name = {s.name: s for s in specs}

        assert by_name["status"].default == "active"

    def test_get_specs_custom_status_default(self):
        """Custom status_default should propagate."""
        specs = CommonSpecs.get_specs(status_default="pending")
        by_name = {s.name: s for s in specs}

        assert by_name["status"].default == "pending"

    def test_get_specs_nullable_fields(self):
        """Email, phone, settings, data should be nullable."""
        specs = CommonSpecs.get_specs()
        by_name = {s.name: s for s in specs}

        assert by_name["email"].is_nullable is True
        assert by_name["phone"].is_nullable is True
        assert by_name["settings"].is_nullable is True
        assert by_name["data"].is_nullable is True

    def test_get_specs_tenant_id_is_uuid(self):
        """tenant_id should be UUID type."""
        specs = CommonSpecs.get_specs()
        by_name = {s.name: s for s in specs}

        assert by_name["tenant_id"].base_type is UUID

    def test_get_specs_string_fields(self):
        """name, slug, status, email, phone should be str type."""
        specs = CommonSpecs.get_specs()
        by_name = {s.name: s for s in specs}

        for field in ("name", "slug", "status", "email", "phone"):
            assert by_name[field].base_type is str

    def test_basemodel_has_fields(self):
        """CommonSpecs should be a BaseModel with expected fields."""
        assert hasattr(CommonSpecs, "model_fields")
        fields = CommonSpecs.model_fields
        assert "name" in fields
        assert "tenant_id" in fields
        assert "settings" in fields
        assert "data" in fields


class TestEnforcementLevel:
    """Tests for EnforcementLevel enum."""

    def test_enforcement_values(self):
        """EnforcementLevel should have three levels."""
        assert EnforcementLevel.HARD_MANDATORY.value == "hard_mandatory"
        assert EnforcementLevel.SOFT_MANDATORY.value == "soft_mandatory"
        assert EnforcementLevel.ADVISORY.value == "advisory"

    def test_is_blocking_hard_mandatory(self):
        """is_blocking should return True for hard_mandatory."""

        class MockResult:
            enforcement = "hard_mandatory"

        assert EnforcementLevel.is_blocking(MockResult()) is True

    def test_is_blocking_soft_mandatory(self):
        """is_blocking should return True for soft_mandatory."""

        class MockResult:
            enforcement = "soft_mandatory"

        assert EnforcementLevel.is_blocking(MockResult()) is True

    def test_is_blocking_advisory(self):
        """is_blocking should return False for advisory."""

        class MockResult:
            enforcement = "advisory"

        assert EnforcementLevel.is_blocking(MockResult()) is False

    def test_is_advisory(self):
        """is_advisory should return True only for advisory."""

        class AdvisoryResult:
            enforcement = "advisory"

        class HardResult:
            enforcement = "hard_mandatory"

        assert EnforcementLevel.is_advisory(AdvisoryResult()) is True
        assert EnforcementLevel.is_advisory(HardResult()) is False


class TestEnforcementSpecs:
    """Tests for EnforcementSpecs.get_specs()."""

    def test_get_specs_returns_five_specs(self):
        """EnforcementSpecs should produce 5 specs."""
        specs = EnforcementSpecs.get_specs()

        assert isinstance(specs, list)
        assert len(specs) == 5

        names = [s.name for s in specs]
        assert "enforcement" in names
        assert "policy_id" in names
        assert "violation_code" in names
        assert "evaluated_at" in names
        assert "evaluation_ms" in names

    def test_enforcement_default(self):
        """Enforcement spec should default to hard_mandatory."""
        specs = EnforcementSpecs.get_specs()
        by_name = {s.name: s for s in specs}

        assert by_name["enforcement"].default == "hard_mandatory"

    def test_evaluation_ms_constraints(self):
        """evaluation_ms should have ge=0.0 constraint."""
        specs = EnforcementSpecs.get_specs()
        by_name = {s.name: s for s in specs}

        assert by_name["evaluation_ms"].default == 0.0
        assert by_name["evaluation_ms"].base_type is float

    def test_violation_code_nullable(self):
        """violation_code should be nullable."""
        specs = EnforcementSpecs.get_specs()
        by_name = {s.name: s for s in specs}

        assert by_name["violation_code"].is_nullable is True

    def test_evaluated_at_is_datetime(self):
        """evaluated_at should be datetime type."""
        specs = EnforcementSpecs.get_specs()
        by_name = {s.name: s for s in specs}

        assert by_name["evaluated_at"].base_type is datetime

    def test_basemodel_has_fields(self):
        """EnforcementSpecs should be a BaseModel with expected fields."""
        assert hasattr(EnforcementSpecs, "model_fields")
        fields = EnforcementSpecs.model_fields
        assert "enforcement" in fields
        assert "policy_id" in fields
        assert "violation_code" in fields
