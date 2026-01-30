# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Spec Catalog - Reusable field Specs for Node composition and DDL generation.

Pre-defined Specs for common database patterns:
- **ContentSpecs**: id, created_at, content, metadata, embedding
- **AuditSpecs**: updated_at/by, deleted_at/by, is_deleted, version, hashes
- **CommonSpecs**: name, slug, status, email, phone, tenant_id, settings

Usage:
    from krons.specs.catalog import ContentSpecs, AuditSpecs

    content_specs = ContentSpecs.get_specs(dim=1536)
    audit_specs = AuditSpecs.get_specs(use_uuid=True)
    all_specs = content_specs + audit_specs

For custom Specs, use the factories directly:
    from krons.specs.factory import create_embedding_spec, create_content_spec

    my_embedding = create_embedding_spec("embedding", dim=1536)
    my_content = create_content_spec("payload", content_type=MyModel)
"""

from ._audit import AuditSpecs
from ._common import CommonSpecs
from ._content import ContentSpecs
from ._enforcement import EnforcementLevel, EnforcementSpecs

__all__ = (
    "AuditSpecs",
    "CommonSpecs",
    "ContentSpecs",
    "EnforcementLevel",
    "EnforcementSpecs",
)
