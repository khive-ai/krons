# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for SQL DDL schema specification dataclasses.

Tests the extracted schema spec dataclasses:
    - ColumnSpec, IndexSpec, TriggerSpec, CheckConstraintSpec, UniqueConstraintSpec
    - ForeignKeySpec with deferrable support
    - TableSpec.from_operable() integration
    - SchemaSpec.from_operables() multi-table support
"""

import pytest

from krons.errors import ValidationError
from krons.specs import Operable, Spec
from krons.specs.adapters.sql_ddl import (
    FK,
    CheckConstraintSpec,
    ColumnSpec,
    ForeignKeySpec,
    IndexMethod,
    IndexSpec,
    OnAction,
    SchemaSpec,
    SQLSpecAdapter,
    TableSpec,
    TriggerSpec,
    UniqueConstraintSpec,
)


class TestOnActionEnum:
    """Test OnAction enum for FK actions."""

    def test_all_values_exist(self):
        """All standard FK actions should be defined."""
        assert OnAction.CASCADE == "CASCADE"
        assert OnAction.SET_NULL == "SET NULL"
        assert OnAction.SET_DEFAULT == "SET DEFAULT"
        assert OnAction.RESTRICT == "RESTRICT"
        assert OnAction.NO_ACTION == "NO ACTION"

    def test_enum_string_values(self):
        """Enum values should work as strings in DDL."""
        assert f"ON DELETE {OnAction.CASCADE.value}" == "ON DELETE CASCADE"
        assert f"ON DELETE {OnAction.SET_NULL.value}" == "ON DELETE SET NULL"


class TestIndexMethodEnum:
    """Test IndexMethod enum for index types."""

    def test_standard_methods(self):
        """Standard PostgreSQL index methods should be defined."""
        assert IndexMethod.BTREE == "btree"
        assert IndexMethod.HASH == "hash"
        assert IndexMethod.GIN == "gin"
        assert IndexMethod.GIST == "gist"

    def test_pgvector_methods(self):
        """pgvector index methods should be defined."""
        assert IndexMethod.IVFFLAT == "ivfflat"
        assert IndexMethod.HNSW == "hnsw"


class TestColumnSpec:
    """Test ColumnSpec dataclass."""

    def test_basic_column_ddl(self):
        """Basic column should generate simple DDL."""
        col = ColumnSpec(name="title", type="TEXT")
        assert col.to_ddl() == '"title" TEXT'

    def test_not_null_column(self):
        """Non-nullable column should include NOT NULL."""
        col = ColumnSpec(name="title", type="TEXT", nullable=False)
        assert col.to_ddl() == '"title" TEXT NOT NULL'

    def test_column_with_default(self):
        """Column with default should include DEFAULT clause."""
        col = ColumnSpec(name="status", type="TEXT", default="'pending'")
        assert col.to_ddl() == "\"status\" TEXT DEFAULT 'pending'"

    def test_primary_key_column(self):
        """Primary key column should include PRIMARY KEY."""
        col = ColumnSpec(name="id", type="UUID", is_primary_key=True)
        assert col.to_ddl() == '"id" UUID PRIMARY KEY'

    def test_unique_column(self):
        """Unique column should include UNIQUE."""
        col = ColumnSpec(name="email", type="TEXT", nullable=False, is_unique=True)
        assert col.to_ddl() == '"email" TEXT NOT NULL UNIQUE'

    def test_frozen_hashable(self):
        """ColumnSpec should be frozen and hashable."""
        col1 = ColumnSpec(name="x", type="TEXT")
        col2 = ColumnSpec(name="x", type="TEXT")
        assert col1 == col2
        assert hash(col1) == hash(col2)
        assert {col1, col2} == {col1}  # Set deduplication works

    def test_invalid_name_raises(self):
        """Invalid column name should raise ValidationError."""
        col = ColumnSpec(name="bad;name", type="TEXT")
        with pytest.raises(ValidationError):
            col.to_ddl()


class TestForeignKeySpec:
    """Test ForeignKeySpec dataclass."""

    def test_basic_fk_ddl(self):
        """Basic FK should generate ALTER TABLE DDL."""
        fk = ForeignKeySpec(
            name="fk_posts_author",
            columns=("author_id",),
            ref_table="users",
        )
        ddl = fk.to_ddl("posts")
        assert 'ALTER TABLE "posts" ADD CONSTRAINT "fk_posts_author"' in ddl
        assert 'FOREIGN KEY ("author_id") REFERENCES "users" ("id")' in ddl
        assert "ON DELETE CASCADE ON UPDATE CASCADE" in ddl

    def test_custom_on_delete(self):
        """FK with custom on_delete should use that action."""
        fk = ForeignKeySpec(
            name="fk_comments_user",
            columns=("user_id",),
            ref_table="users",
            on_delete=OnAction.SET_NULL,
        )
        ddl = fk.to_ddl("comments")
        assert "ON DELETE SET NULL" in ddl

    def test_deferrable_fk(self):
        """Deferrable FK should include DEFERRABLE clause."""
        fk = ForeignKeySpec(
            name="fk_tree_parent",
            columns=("parent_id",),
            ref_table="tree_nodes",
            deferrable=True,
        )
        ddl = fk.to_ddl("tree_nodes")
        assert "DEFERRABLE" in ddl
        assert "INITIALLY DEFERRED" not in ddl

    def test_initially_deferred_fk(self):
        """Initially deferred FK should include both clauses."""
        fk = ForeignKeySpec(
            name="fk_tree_parent",
            columns=("parent_id",),
            ref_table="tree_nodes",
            deferrable=True,
            initially_deferred=True,
        )
        ddl = fk.to_ddl("tree_nodes")
        assert "DEFERRABLE INITIALLY DEFERRED" in ddl

    def test_composite_fk(self):
        """Composite FK with multiple columns should work."""
        fk = ForeignKeySpec(
            name="fk_order_items_product",
            columns=("tenant_id", "product_id"),
            ref_table="products",
            ref_columns=("tenant_id", "id"),
        )
        ddl = fk.to_ddl("order_items")
        assert 'FOREIGN KEY ("tenant_id", "product_id")' in ddl
        assert 'REFERENCES "products" ("tenant_id", "id")' in ddl


class TestIndexSpec:
    """Test IndexSpec dataclass."""

    def test_basic_index_ddl(self):
        """Basic index should generate CREATE INDEX DDL."""
        idx = IndexSpec(name="idx_users_email", columns=("email",))
        ddl = idx.to_ddl("users")
        assert 'CREATE INDEX IF NOT EXISTS "idx_users_email"' in ddl
        assert 'ON "public"."users"' in ddl
        assert '("email")' in ddl

    def test_unique_index(self):
        """Unique index should include UNIQUE keyword."""
        idx = IndexSpec(name="idx_users_email", columns=("email",), unique=True)
        ddl = idx.to_ddl("users")
        assert "CREATE UNIQUE INDEX" in ddl

    def test_gin_index(self):
        """GIN index should include USING gin."""
        idx = IndexSpec(name="idx_posts_tags", columns=("tags",), method=IndexMethod.GIN)
        ddl = idx.to_ddl("posts")
        assert "USING gin" in ddl

    def test_partial_index(self):
        """Partial index should include WHERE clause."""
        idx = IndexSpec(
            name="idx_users_active",
            columns=("email",),
            where="deleted_at IS NULL",
        )
        ddl = idx.to_ddl("users")
        assert "WHERE deleted_at IS NULL" in ddl

    def test_covering_index(self):
        """Covering index should include INCLUDE clause."""
        idx = IndexSpec(
            name="idx_posts_lookup",
            columns=("user_id",),
            include=("title", "created_at"),
        )
        ddl = idx.to_ddl("posts")
        assert 'INCLUDE ("title", "created_at")' in ddl

    def test_concurrent_index(self):
        """Concurrent index should include CONCURRENTLY keyword."""
        idx = IndexSpec(name="idx_users_email", columns=("email",), concurrently=True)
        ddl = idx.to_ddl("users")
        assert "CREATE INDEX CONCURRENTLY" in ddl


class TestTriggerSpec:
    """Test TriggerSpec dataclass."""

    def test_basic_trigger_ddl(self):
        """Basic trigger should generate CREATE TRIGGER DDL."""
        trigger = TriggerSpec(
            name="tr_audit_insert",
            timing="AFTER",
            events=("INSERT",),
            function="public.audit_log",
        )
        ddl = trigger.to_ddl("users")
        assert 'CREATE TRIGGER "tr_audit_insert"' in ddl
        assert "AFTER INSERT" in ddl
        assert 'ON "public"."users"' in ddl
        assert "EXECUTE FUNCTION public.audit_log();" in ddl

    def test_multiple_events(self):
        """Trigger with multiple events should use OR."""
        trigger = TriggerSpec(
            name="tr_audit",
            timing="AFTER",
            events=("INSERT", "UPDATE", "DELETE"),
            function="audit_log",
        )
        ddl = trigger.to_ddl("users")
        assert "AFTER INSERT OR UPDATE OR DELETE" in ddl

    def test_before_trigger(self):
        """BEFORE trigger should use BEFORE timing."""
        trigger = TriggerSpec(
            name="tr_validate",
            timing="BEFORE",
            events=("INSERT",),
            function="validate_data",
        )
        ddl = trigger.to_ddl("data")
        assert "BEFORE INSERT" in ddl

    def test_trigger_with_when(self):
        """Trigger with WHEN condition should include it."""
        trigger = TriggerSpec(
            name="tr_notify_change",
            timing="AFTER",
            events=("UPDATE",),
            function="notify_changes",
            when="OLD.status IS DISTINCT FROM NEW.status",
        )
        ddl = trigger.to_ddl("orders")
        assert "WHEN (OLD.status IS DISTINCT FROM NEW.status)" in ddl


class TestCheckConstraintSpec:
    """Test CheckConstraintSpec dataclass."""

    def test_check_constraint_ddl(self):
        """CHECK constraint should generate ALTER TABLE DDL."""
        check = CheckConstraintSpec(name="chk_positive_amount", expression="amount > 0")
        ddl = check.to_ddl("orders")
        assert 'ALTER TABLE "public"."orders"' in ddl
        assert 'ADD CONSTRAINT "chk_positive_amount" CHECK (amount > 0)' in ddl


class TestUniqueConstraintSpec:
    """Test UniqueConstraintSpec dataclass."""

    def test_unique_constraint_ddl(self):
        """UNIQUE constraint should generate ALTER TABLE DDL."""
        uc = UniqueConstraintSpec(name="uq_email_tenant", columns=("tenant_id", "email"))
        ddl = uc.to_ddl("users")
        assert 'ALTER TABLE "public"."users"' in ddl
        assert 'ADD CONSTRAINT "uq_email_tenant" UNIQUE ("tenant_id", "email")' in ddl


class TestTableSpec:
    """Test TableSpec dataclass."""

    def test_create_table_ddl(self):
        """TableSpec should generate CREATE TABLE DDL."""
        table = TableSpec(
            name="users",
            columns=(
                ColumnSpec(name="id", type="UUID", is_primary_key=True),
                ColumnSpec(name="email", type="TEXT", nullable=False),
                ColumnSpec(name="name", type="TEXT"),
            ),
        )
        ddl = table.to_create_table_ddl()
        assert 'CREATE TABLE IF NOT EXISTS "public"."users"' in ddl
        assert '"id" UUID PRIMARY KEY' in ddl
        assert '"email" TEXT NOT NULL' in ddl
        assert '"name" TEXT' in ddl

    def test_full_ddl_with_constraints(self):
        """to_full_ddl should include all constraints."""
        table = TableSpec(
            name="orders",
            columns=(
                ColumnSpec(name="id", type="UUID", is_primary_key=True),
                ColumnSpec(name="user_id", type="UUID", nullable=False),
                ColumnSpec(name="amount", type="INTEGER"),
            ),
            foreign_keys=(
                ForeignKeySpec(
                    name="fk_orders_user",
                    columns=("user_id",),
                    ref_table="users",
                ),
            ),
            indexes=(IndexSpec(name="idx_orders_user", columns=("user_id",)),),
            check_constraints=(CheckConstraintSpec(name="chk_positive", expression="amount > 0"),),
        )
        statements = table.to_full_ddl()
        assert len(statements) == 4  # CREATE TABLE + FK + INDEX + CHECK
        assert any("CREATE TABLE" in s for s in statements)
        assert any("FOREIGN KEY" in s for s in statements)
        assert any("CREATE INDEX" in s for s in statements)
        assert any("CHECK" in s for s in statements)

    def test_from_operable(self):
        """TableSpec.from_operable should convert Operable to TableSpec."""
        spec1 = Spec(str, name="name")
        spec2 = Spec(int, name="age", default=0)
        op = Operable((spec1, spec2))

        table = TableSpec.from_operable(op, "persons", primary_key="name")

        assert table.name == "persons"
        assert len(table.columns) == 2
        name_col = table.get_column("name")
        assert name_col is not None
        assert name_col.is_primary_key is True
        age_col = table.get_column("age")
        assert age_col is not None
        assert age_col.default == "0"

    def test_get_column(self):
        """get_column should find column by name."""
        table = TableSpec(
            name="test",
            columns=(
                ColumnSpec(name="a", type="TEXT"),
                ColumnSpec(name="b", type="INTEGER"),
            ),
        )
        assert table.get_column("a") is not None
        assert table.get_column("b") is not None
        assert table.get_column("c") is None


class TestSchemaSpec:
    """Test SchemaSpec dataclass."""

    def test_from_operables(self):
        """SchemaSpec.from_operables should create multi-table schema."""
        users_op = Operable((Spec(str, name="email"),))
        posts_op = Operable((Spec(str, name="title"),))

        schema = SchemaSpec.from_operables(
            {"users": users_op, "posts": posts_op},
            schema="myapp",
        )

        assert len(schema.tables) == 2
        assert schema.version is not None  # Hash computed
        assert schema.get_table("users") is not None
        assert schema.get_table("posts") is not None
        assert schema.get_table("comments") is None

    def test_version_changes_with_schema(self):
        """Version hash should change when schema changes."""
        op1 = Operable((Spec(str, name="name"),))
        op2 = Operable((Spec(str, name="name"), Spec(int, name="age")))

        schema1 = SchemaSpec.from_operables({"test": op1})
        schema2 = SchemaSpec.from_operables({"test": op2})

        assert schema1.version != schema2.version


class TestSQLSpecAdapterIntegration:
    """Test SQLSpecAdapter integration with new features."""

    def test_create_table_spec(self):
        """create_table_spec should create TableSpec from Operable."""
        op = Operable((Spec(str, name="title"), Spec(int, name="views", default=0)))

        table_spec = SQLSpecAdapter.create_table_spec(
            op,
            "articles",
            indexes=[{"columns": ["title"], "unique": True}],
        )

        assert table_spec.name == "articles"
        assert len(table_spec.columns) == 2
        assert len(table_spec.indexes) == 1
        assert table_spec.indexes[0].unique is True

    def test_create_index_with_method_enum(self):
        """create_index should accept IndexMethod enum."""
        ddl = SQLSpecAdapter.create_index(
            "posts",
            "tags",
            method=IndexMethod.GIN,
        )
        assert "USING gin" in ddl

    def test_fk_with_deferrable(self):
        """FK annotation with deferrable should generate correct DDL."""
        from typing import Annotated
        from uuid import UUID

        from krons.specs.adapters.sql_ddl import FKMeta

        # Create spec with deferrable FK
        deferrable_fk = Annotated[UUID, FKMeta("Parent", deferrable=True)]
        spec = Spec(deferrable_fk, name="parent_id")
        op = Operable((spec,))

        ddl = SQLSpecAdapter.compose_structure(op, "children")
        assert "DEFERRABLE" in ddl
