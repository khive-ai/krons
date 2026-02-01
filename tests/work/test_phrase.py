# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for krons.work.phrase - typed operation templates with auto-generated types."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from krons.core.specs import Operable, Spec
from krons.errors import ValidationError
from krons.work.phrase import CrudOperation, CrudPattern, Phrase, _to_pascal, phrase

# =============================================================================
# Helper: Mock Context
# =============================================================================


class MockContext:
    """Mock execution context for testing."""

    def __init__(
        self,
        *,
        query_fn=None,
        now=None,
        metadata: dict | None = None,
    ):
        self.query_fn = query_fn
        self.now = now
        self.metadata = metadata or {}

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self.metadata:
            return self.metadata[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


# =============================================================================
# Tests: _to_pascal helper
# =============================================================================


class TestToPascal:
    """Tests for _to_pascal snake_case to PascalCase conversion."""

    def test_single_word(self):
        """Single word should be capitalized."""
        assert _to_pascal("check") == "Check"

    def test_snake_case_two_words(self):
        """Two word snake_case should become PascalCase."""
        assert _to_pascal("verify_user") == "VerifyUser"

    def test_snake_case_multiple_words(self):
        """Multiple words in snake_case should become PascalCase."""
        assert _to_pascal("verify_consent_token") == "VerifyConsentToken"
        assert _to_pascal("require_monitoring_active") == "RequireMonitoringActive"

    def test_already_capitalized(self):
        """Already capitalized words should stay capitalized."""
        assert _to_pascal("Check") == "Check"

    def test_empty_string(self):
        """Empty string should return empty string."""
        assert _to_pascal("") == ""

    def test_single_underscore(self):
        """Single underscore should produce empty parts correctly."""
        assert _to_pascal("_") == ""

    def test_leading_underscore(self):
        """Leading underscore should produce empty first part."""
        assert _to_pascal("_check") == "Check"

    def test_trailing_underscore(self):
        """Trailing underscore should produce empty last part."""
        assert _to_pascal("check_") == "Check"

    def test_multiple_underscores(self):
        """Multiple consecutive underscores should produce empty parts."""
        assert _to_pascal("check__auth") == "CheckAuth"


# =============================================================================
# Tests: CrudOperation
# =============================================================================


class TestCrudOperation:
    """Tests for CrudOperation enum."""

    def test_read_value(self):
        """READ should have value 'read'."""
        assert CrudOperation.READ.value == "read"

    def test_insert_value(self):
        """INSERT should have value 'insert'."""
        assert CrudOperation.INSERT.value == "insert"

    def test_update_value(self):
        """UPDATE should have value 'update'."""
        assert CrudOperation.UPDATE.value == "update"

    def test_soft_delete_value(self):
        """SOFT_DELETE should have value 'soft_delete'."""
        assert CrudOperation.SOFT_DELETE.value == "soft_delete"

    def test_string_conversion(self):
        """CrudOperation should be convertible from string."""
        assert CrudOperation("read") == CrudOperation.READ
        assert CrudOperation("insert") == CrudOperation.INSERT


# =============================================================================
# Tests: CrudPattern
# =============================================================================


class TestCrudPattern:
    """Tests for CrudPattern declarative CRUD configuration."""

    def test_basic_initialization(self):
        """CrudPattern should initialize with table name."""
        crud = CrudPattern(table="users")
        assert crud.table == "users"
        assert crud.operation == CrudOperation.READ
        assert crud.lookup == frozenset()

    def test_operation_as_string(self):
        """CrudPattern should accept operation as string."""
        crud = CrudPattern(table="users", operation="insert")
        assert crud.operation == CrudOperation.INSERT

    def test_operation_as_enum(self):
        """CrudPattern should accept operation as enum."""
        crud = CrudPattern(table="users", operation=CrudOperation.UPDATE)
        assert crud.operation == CrudOperation.UPDATE

    def test_lookup_normalized_to_frozenset(self):
        """CrudPattern should normalize lookup to frozenset."""
        crud = CrudPattern(table="users", lookup={"user_id", "scope"})
        assert crud.lookup == frozenset({"user_id", "scope"})
        assert isinstance(crud.lookup, frozenset)

    def test_lookup_from_list(self):
        """CrudPattern should accept lookup as list."""
        crud = CrudPattern(table="users", lookup=["user_id", "scope"])
        assert crud.lookup == frozenset({"user_id", "scope"})

    def test_filters_default_empty(self):
        """CrudPattern filters should default to empty mapping."""
        crud = CrudPattern(table="users")
        assert crud.filters == {}

    def test_filters_provided(self):
        """CrudPattern should accept filters dict."""
        crud = CrudPattern(table="users", filters={"status": "active"})
        assert crud.filters["status"] == "active"

    def test_filters_immutable(self):
        """CrudPattern filters should be immutable after construction."""
        crud = CrudPattern(table="users", filters={"status": "active"})
        with pytest.raises(TypeError):
            crud.filters["injected"] = "bad"

    def test_set_fields_default_empty(self):
        """CrudPattern set_fields should default to empty mapping."""
        crud = CrudPattern(table="users")
        assert crud.set_fields == {}

    def test_set_fields_provided(self):
        """CrudPattern should accept set_fields dict."""
        crud = CrudPattern(
            table="users",
            operation="update",
            set_fields={"status": "revoked"},
        )
        assert crud.set_fields["status"] == "revoked"

    def test_set_fields_immutable(self):
        """CrudPattern set_fields should be immutable after construction."""
        crud = CrudPattern(
            table="users",
            operation="update",
            set_fields={"status": "active"},
        )
        with pytest.raises(TypeError):
            crud.set_fields["injected"] = "bad"

    def test_defaults_default_empty(self):
        """CrudPattern defaults should default to empty mapping."""
        crud = CrudPattern(table="users")
        assert crud.defaults == {}

    def test_defaults_provided(self):
        """CrudPattern should accept defaults dict."""
        crud = CrudPattern(
            table="users",
            operation="insert",
            defaults={"role": "user", "status": "pending"},
        )
        assert crud.defaults["role"] == "user"
        assert crud.defaults["status"] == "pending"

    def test_defaults_immutable(self):
        """CrudPattern defaults should be immutable after construction."""
        crud = CrudPattern(
            table="users",
            operation="insert",
            defaults={"role": "user"},
        )
        with pytest.raises(TypeError):
            crud.defaults["injected"] = "bad"

    def test_frozen_dataclass(self):
        """CrudPattern should be immutable (frozen dataclass)."""
        crud = CrudPattern(table="users")
        with pytest.raises(AttributeError):
            crud.table = "other"

    def test_invalid_table_name_sql_injection(self):
        """CrudPattern should reject SQL-unsafe table names."""
        with pytest.raises(ValidationError):
            CrudPattern(table="users; DROP TABLE --")

    def test_empty_table_name(self):
        """CrudPattern should reject empty table name."""
        with pytest.raises(ValidationError):
            CrudPattern(table="")

    def test_table_with_underscores(self):
        """CrudPattern should accept table names with underscores."""
        crud = CrudPattern(table="consent_tokens")
        assert crud.table == "consent_tokens"

    def test_table_alphanumeric(self):
        """CrudPattern should accept alphanumeric table names."""
        crud = CrudPattern(table="users2024")
        assert crud.table == "users2024"


# =============================================================================
# Tests: Phrase
# =============================================================================


class TestPhrase:
    """Tests for Phrase class - typed operation templates."""

    def test_initialization_with_handler(self):
        """Phrase should initialize with name, operable, inputs, outputs, handler."""
        spec1 = Spec(UUID, name="user_id")
        spec2 = Spec(bool, name="verified")
        op = Operable([spec1, spec2])

        async def handler(options, ctx):
            return {"verified": True}

        p = Phrase(
            name="verify_user",
            operable=op,
            inputs={"user_id"},
            outputs={"verified"},
            handler=handler,
        )

        assert p.name == "verify_user"
        assert p.inputs == ("user_id",)
        assert p.outputs == ("verified",)
        assert p.handler is handler
        assert p.crud is None

    def test_initialization_with_crud(self):
        """Phrase should initialize with crud pattern (auto-generates handler)."""
        spec1 = Spec(str, name="scope")
        spec2 = Spec(bool, name="found")
        op = Operable([spec1, spec2])

        crud = CrudPattern(table="tokens", lookup={"scope"})

        p = Phrase(
            name="find_token",
            operable=op,
            inputs={"scope"},
            outputs={"found"},
            crud=crud,
            result_parser=lambda row: {"found": row is not None},
        )

        assert p.name == "find_token"
        assert p.crud is crud
        assert p.handler is not None  # Auto-generated

    def test_requires_handler_or_crud(self):
        """Phrase should require either handler or crud."""
        spec = Spec(str, name="field")
        op = Operable([spec])

        with pytest.raises(ValueError, match="Either handler or crud"):
            Phrase(
                name="test",
                operable=op,
                inputs={"field"},
                outputs={"field"},
            )

    def test_options_type_generated(self):
        """Phrase.options_type should generate typed options dataclass from inputs."""
        spec1 = Spec(UUID, name="user_id")
        spec2 = Spec(int, name="max_age", default=300)
        spec3 = Spec(bool, name="verified")
        op = Operable([spec1, spec2, spec3])

        async def handler(options, ctx):
            return {"verified": True}

        p = Phrase(
            name="check_auth",
            operable=op,
            inputs={"user_id", "max_age"},
            outputs={"verified"},
            handler=handler,
        )

        Options = p.options_type
        assert Options.__name__ == "CheckAuthOptions"

        # Should be usable
        uid = uuid4()
        opts = Options(user_id=uid)
        assert opts.user_id == uid
        assert opts.max_age == 300  # default

    def test_result_type_generated(self):
        """Phrase.result_type should generate typed result dataclass from outputs."""
        spec1 = Spec(UUID, name="user_id")
        spec2 = Spec(bool, name="verified")
        spec3 = Spec(str, name="reason", nullable=True)
        op = Operable([spec1, spec2, spec3])

        async def handler(options, ctx):
            return {"verified": True, "reason": None}

        p = Phrase(
            name="verify_status",
            operable=op,
            inputs={"user_id"},
            outputs={"verified", "reason"},
            handler=handler,
        )

        Result = p.result_type
        assert Result.__name__ == "VerifyStatusResult"

    def test_types_cached(self):
        """Phrase types should be lazily computed and cached."""
        spec1 = Spec(str, name="query")
        spec2 = Spec(bool, name="found")
        op = Operable([spec1, spec2])

        async def handler(options, ctx):
            return {"found": True}

        p = Phrase(
            name="search",
            operable=op,
            inputs={"query"},
            outputs={"found"},
            handler=handler,
        )

        # Access twice - should be same object
        t1 = p.options_type
        t2 = p.options_type
        assert t1 is t2

        r1 = p.result_type
        r2 = p.result_type
        assert r1 is r2

    def test_input_fields_property(self):
        """Phrase.input_fields should return list of input field names."""
        spec1 = Spec(str, name="a")
        spec2 = Spec(int, name="b")
        spec3 = Spec(bool, name="c")
        op = Operable([spec1, spec2, spec3])

        async def handler(options, ctx):
            return {"c": True}

        p = Phrase(
            name="test",
            operable=op,
            inputs={"a", "b"},
            outputs={"c"},
            handler=handler,
        )

        assert set(p.input_fields) == {"a", "b"}

    def test_output_fields_property(self):
        """Phrase.output_fields should return list of output field names."""
        spec1 = Spec(str, name="a")
        spec2 = Spec(int, name="b")
        spec3 = Spec(bool, name="c")
        op = Operable([spec1, spec2, spec3])

        async def handler(options, ctx):
            return {"b": 1, "c": True}

        p = Phrase(
            name="test",
            operable=op,
            inputs={"a"},
            outputs={"b", "c"},
            handler=handler,
        )

        assert set(p.output_fields) == {"b", "c"}

    def test_is_workable_all_inputs_available(self):
        """Phrase.is_workable should return True when all inputs available."""
        spec1 = Spec(str, name="a")
        spec2 = Spec(int, name="b")
        op = Operable([spec1, spec2])

        async def handler(options, ctx):
            return {"b": 1}

        p = Phrase(
            name="test",
            operable=op,
            inputs={"a"},
            outputs={"b"},
            handler=handler,
        )

        assert p.is_workable({"a": "value"}) is True
        assert p.is_workable({"a": "value", "b": 2}) is True

    def test_is_workable_missing_inputs(self):
        """Phrase.is_workable should return False when inputs missing."""
        spec1 = Spec(str, name="a")
        spec2 = Spec(int, name="b")
        op = Operable([spec1, spec2])

        async def handler(options, ctx):
            return {"b": 1}

        p = Phrase(
            name="test",
            operable=op,
            inputs={"a"},
            outputs={"b"},
            handler=handler,
        )

        assert p.is_workable({}) is False
        assert p.is_workable({"b": 2}) is False

    def test_extract_inputs(self):
        """Phrase.extract_inputs should return dict of input field values."""
        spec1 = Spec(str, name="a")
        spec2 = Spec(int, name="b")
        spec3 = Spec(bool, name="c")
        op = Operable([spec1, spec2, spec3])

        async def handler(options, ctx):
            return {"c": True}

        p = Phrase(
            name="test",
            operable=op,
            inputs={"a", "b"},
            outputs={"c"},
            handler=handler,
        )

        data = {"a": "hello", "b": 42, "c": True, "extra": "ignored"}
        inputs = p.extract_inputs(data)

        assert inputs == {"a": "hello", "b": 42}
        assert "c" not in inputs
        assert "extra" not in inputs

    def test_extract_inputs_missing_raises(self):
        """Phrase.extract_inputs should raise KeyError for missing inputs."""
        spec1 = Spec(str, name="a")
        spec2 = Spec(int, name="b")
        op = Operable([spec1, spec2])

        async def handler(options, ctx):
            return {"b": 1}

        p = Phrase(
            name="test",
            operable=op,
            inputs={"a"},
            outputs={"b"},
            handler=handler,
        )

        with pytest.raises(KeyError):
            p.extract_inputs({})

    @pytest.mark.anyio
    async def test_call_validates_and_invokes(self):
        """Phrase.__call__ should validate options and invoke handler."""
        spec1 = Spec(str, name="name")
        spec2 = Spec(int, name="count")
        op = Operable([spec1, spec2])

        async def handler(options, ctx):
            return {"count": len(options.name)}

        p = Phrase(
            name="count_chars",
            operable=op,
            inputs={"name"},
            outputs={"count"},
            handler=handler,
        )

        result = await p({"name": "hello"}, None)
        assert result.count == 5

    @pytest.mark.anyio
    async def test_call_with_typed_options(self):
        """Phrase.__call__ should accept pre-typed options."""
        spec1 = Spec(str, name="name")
        spec2 = Spec(int, name="count")
        op = Operable([spec1, spec2])

        async def handler(options, ctx):
            return {"count": len(options.name)}

        p = Phrase(
            name="count_chars",
            operable=op,
            inputs={"name"},
            outputs={"count"},
            handler=handler,
        )

        Options = p.options_type
        result = await p(Options(name="world"), None)
        assert result.count == 5


# =============================================================================
# Tests: @phrase decorator
# =============================================================================


class TestPhraseDecorator:
    """Tests for @phrase decorator."""

    def test_decorator_creates_phrase(self):
        """@phrase decorator should create a Phrase instance."""
        spec1 = Spec(str, name="user_id")
        spec2 = Spec(bool, name="verified")
        op = Operable([spec1, spec2])

        @phrase(op, inputs={"user_id"}, outputs={"verified"})
        async def verify_user(options, ctx):
            return {"verified": True}

        assert isinstance(verify_user, Phrase)
        assert verify_user.name == "verify_user"
        assert "user_id" in verify_user.inputs
        assert "verified" in verify_user.outputs

    def test_decorator_uses_function_name(self):
        """@phrase decorator should use function name as phrase name."""
        spec1 = Spec(str, name="a")
        spec2 = Spec(int, name="b")
        op = Operable([spec1, spec2])

        @phrase(op, inputs={"a"}, outputs={"b"})
        async def my_custom_operation(options, ctx):
            return {"b": 1}

        assert my_custom_operation.name == "my_custom_operation"
        assert my_custom_operation.options_type.__name__ == "MyCustomOperationOptions"

    def test_decorator_with_custom_name(self):
        """@phrase decorator should accept custom name."""
        spec1 = Spec(str, name="a")
        spec2 = Spec(int, name="b")
        op = Operable([spec1, spec2])

        @phrase(op, inputs={"a"}, outputs={"b"}, name="custom_name")
        async def some_function(options, ctx):
            return {"b": 1}

        assert some_function.name == "custom_name"
        assert some_function.options_type.__name__ == "CustomNameOptions"

    @pytest.mark.anyio
    async def test_decorated_phrase_callable(self):
        """Decorated phrase should be callable."""
        spec1 = Spec(str, name="name")
        spec2 = Spec(int, name="length")
        op = Operable([spec1, spec2])

        @phrase(op, inputs={"name"}, outputs={"length"})
        async def get_length(options, ctx):
            return {"length": len(options.name)}

        result = await get_length({"name": "hello"}, None)
        assert result.length == 5

    def test_direct_mode_with_crud(self):
        """phrase() with crud should return Phrase directly (not decorator)."""
        spec1 = Spec(str, name="scope")
        spec2 = Spec(bool, name="verified")
        op = Operable([spec1, spec2])

        p = phrase(
            op,
            inputs={"scope"},
            outputs={"verified"},
            crud=CrudPattern(table="tokens", lookup={"scope"}),
            result_parser=lambda row: {"verified": row is not None},
            name="check_token",
        )

        assert isinstance(p, Phrase)
        assert p.name == "check_token"
        assert p.crud is not None
        assert p.crud.table == "tokens"

    def test_direct_mode_requires_name(self):
        """phrase() with crud requires explicit name."""
        spec = Spec(str, name="field")
        op = Operable([spec])

        with pytest.raises(ValueError, match="name is required"):
            phrase(
                op,
                inputs={"field"},
                outputs={"field"},
                crud=CrudPattern(table="test"),
            )


# =============================================================================
# Tests: CRUD Handler Generation
# =============================================================================


class TestCrudHandlerGeneration:
    """Tests for auto-generated CRUD handlers."""

    @pytest.mark.anyio
    async def test_crud_read_handler(self):
        """Test auto-generated READ handler."""
        spec1 = Spec(UUID, name="subject_id")
        spec2 = Spec(str, name="scope")
        spec3 = Spec(bool, name="has_consent")
        op = Operable([spec1, spec2, spec3])

        def result_parser(row):
            if row is None:
                return {"has_consent": False}
            return {"has_consent": row.get("status") == "active"}

        p = phrase(
            op,
            inputs={"subject_id", "scope"},
            outputs={"has_consent", "subject_id"},
            crud=CrudPattern(table="consent_tokens", lookup={"subject_id", "scope"}),
            result_parser=result_parser,
            name="verify_consent",
        )

        subject_id = uuid4()
        tenant_id = uuid4()

        async def mock_query_fn(table, operation, where, data, ctx):
            assert table == "consent_tokens"
            assert operation == "select_one"
            assert "subject_id" in where
            assert "tenant_id" in where
            return {"status": "active"}

        ctx = MockContext(
            query_fn=mock_query_fn,
            metadata={"tenant_id": tenant_id},
        )
        result = await p({"subject_id": subject_id, "scope": "background"}, ctx)

        assert result.has_consent is True
        assert result.subject_id == subject_id  # pass-through

    @pytest.mark.anyio
    async def test_crud_read_not_found(self):
        """Test READ handler when row not found."""
        spec1 = Spec(str, name="scope")
        spec2 = Spec(bool, name="found")
        op = Operable([spec1, spec2])

        p = phrase(
            op,
            inputs={"scope"},
            outputs={"found", "scope"},
            crud=CrudPattern(table="tokens", lookup={"scope"}),
            result_parser=lambda row: {"found": row is not None},
            name="find_token",
        )

        async def mock_query_fn(table, operation, where, data, ctx):
            return None  # Not found

        ctx = MockContext(
            query_fn=mock_query_fn,
            metadata={"tenant_id": uuid4()},
        )
        result = await p({"scope": "test"}, ctx)

        assert result.found is False
        assert result.scope == "test"  # pass-through

    @pytest.mark.anyio
    async def test_crud_insert_handler(self):
        """Test auto-generated INSERT handler."""
        spec1 = Spec(str, name="scope")
        spec2 = Spec(str, name="status")
        spec3 = Spec(UUID, name="token_id", nullable=True)
        op = Operable([spec1, spec2, spec3])

        token_id = uuid4()
        tenant_id = uuid4()

        p = phrase(
            op,
            inputs={"scope"},
            outputs={"token_id", "scope"},
            crud=CrudPattern(
                table="tokens",
                operation="insert",
                defaults={"status": "active"},
            ),
            name="create_token",
        )

        async def mock_query_fn(table, operation, where, data, ctx):
            assert table == "tokens"
            assert operation == "insert"
            assert data["scope"] == "background"
            assert data["status"] == "active"  # default
            assert data["tenant_id"] == tenant_id
            return {"token_id": token_id, "scope": data["scope"]}

        ctx = MockContext(
            query_fn=mock_query_fn,
            metadata={"tenant_id": tenant_id},
        )
        result = await p({"scope": "background"}, ctx)

        assert result.token_id == token_id
        assert result.scope == "background"

    @pytest.mark.anyio
    async def test_crud_update_handler(self):
        """Test auto-generated UPDATE handler."""
        spec1 = Spec(UUID, name="token_id")
        spec2 = Spec(str, name="status")
        spec3 = Spec(bool, name="updated")
        op = Operable([spec1, spec2, spec3])

        token_id = uuid4()
        tenant_id = uuid4()

        p = phrase(
            op,
            inputs={"token_id"},
            outputs={"updated"},
            crud=CrudPattern(
                table="tokens",
                operation="update",
                lookup={"token_id"},
                set_fields={"status": "revoked"},
            ),
            result_parser=lambda row: {"updated": row is not None},
            name="revoke_token",
        )

        async def mock_query_fn(table, operation, where, data, ctx):
            assert table == "tokens"
            assert operation == "update"
            assert where["token_id"] == token_id
            assert where["tenant_id"] == tenant_id
            assert data["status"] == "revoked"
            return {"id": token_id}

        ctx = MockContext(
            query_fn=mock_query_fn,
            metadata={"tenant_id": tenant_id},
        )
        result = await p({"token_id": token_id}, ctx)

        assert result.updated is True

    @pytest.mark.anyio
    async def test_crud_update_with_ctx_values(self):
        """Test UPDATE handler with ctx.{attr} set_fields."""
        spec1 = Spec(UUID, name="token_id")
        spec2 = Spec(str, name="updated_at")
        spec3 = Spec(bool, name="updated")
        op = Operable([spec1, spec2, spec3])

        token_id = uuid4()

        p = phrase(
            op,
            inputs={"token_id"},
            outputs={"updated"},
            crud=CrudPattern(
                table="tokens",
                operation="update",
                lookup={"token_id"},
                set_fields={"updated_at": "ctx.now"},
            ),
            result_parser=lambda row: {"updated": row is not None},
            name="touch_token",
        )

        async def mock_query_fn(table, operation, where, data, ctx):
            assert data["updated_at"] == "2025-01-01T00:00:00Z"
            return {"id": token_id}

        ctx = MockContext(
            query_fn=mock_query_fn,
            now="2025-01-01T00:00:00Z",
            metadata={},
        )
        result = await p({"token_id": token_id}, ctx)

        assert result.updated is True

    @pytest.mark.anyio
    async def test_crud_soft_delete_handler(self):
        """Test auto-generated SOFT_DELETE handler."""
        spec1 = Spec(UUID, name="token_id")
        spec2 = Spec(bool, name="deleted")
        op = Operable([spec1, spec2])

        token_id = uuid4()
        tenant_id = uuid4()

        p = phrase(
            op,
            inputs={"token_id"},
            outputs={"deleted"},
            crud=CrudPattern(
                table="tokens",
                operation="soft_delete",
                lookup={"token_id"},
            ),
            result_parser=lambda row: {"deleted": row is not None},
            name="delete_token",
        )

        async def mock_query_fn(table, operation, where, data, ctx):
            assert operation == "update"  # soft delete uses update
            assert data["is_deleted"] is True
            assert "deleted_at" in data
            return {"id": token_id}

        ctx = MockContext(
            query_fn=mock_query_fn,
            now="2025-01-01T00:00:00Z",
            metadata={"tenant_id": tenant_id},
        )
        result = await p({"token_id": token_id}, ctx)

        assert result.deleted is True

    @pytest.mark.anyio
    async def test_crud_soft_delete_no_now(self):
        """Test soft delete without ctx.now should not set deleted_at."""
        spec1 = Spec(UUID, name="token_id")
        spec2 = Spec(bool, name="deleted")
        op = Operable([spec1, spec2])

        token_id = uuid4()

        p = phrase(
            op,
            inputs={"token_id"},
            outputs={"deleted"},
            crud=CrudPattern(
                table="tokens",
                operation="soft_delete",
                lookup={"token_id"},
            ),
            result_parser=lambda row: {"deleted": row is not None},
            name="delete_token",
        )

        async def mock_query_fn(table, operation, where, data, ctx):
            assert data["is_deleted"] is True
            assert "deleted_at" not in data  # now is None
            return {"id": token_id}

        ctx = MockContext(query_fn=mock_query_fn)
        result = await p({"token_id": token_id}, ctx)

        assert result.deleted is True

    @pytest.mark.anyio
    async def test_crud_read_no_tenant(self):
        """Test READ handler without tenant_id should not inject it."""
        spec1 = Spec(str, name="scope")
        spec2 = Spec(bool, name="found")
        op = Operable([spec1, spec2])

        p = phrase(
            op,
            inputs={"scope"},
            outputs={"found"},
            crud=CrudPattern(table="tokens", lookup={"scope"}),
            result_parser=lambda row: {"found": row is not None},
            name="find_token",
        )

        async def mock_query_fn(table, operation, where, data, ctx):
            assert "tenant_id" not in where  # Should NOT inject
            return {"scope": "test"}

        ctx = MockContext(query_fn=mock_query_fn)
        result = await p({"scope": "test"}, ctx)

        assert result.found is True

    @pytest.mark.anyio
    async def test_crud_filters_applied(self):
        """Test that static filters are applied to WHERE clause."""
        spec1 = Spec(str, name="scope")
        spec2 = Spec(bool, name="found")
        op = Operable([spec1, spec2])

        p = phrase(
            op,
            inputs={"scope"},
            outputs={"found"},
            crud=CrudPattern(
                table="tokens",
                lookup={"scope"},
                filters={"status": "active", "is_deleted": False},
            ),
            result_parser=lambda row: {"found": row is not None},
            name="find_active_token",
        )

        async def mock_query_fn(table, operation, where, data, ctx):
            assert where["status"] == "active"
            assert where["is_deleted"] is False
            return {"scope": "test"}

        ctx = MockContext(query_fn=mock_query_fn)
        result = await p({"scope": "test"}, ctx)

        assert result.found is True

    @pytest.mark.anyio
    async def test_crud_missing_query_fn_raises(self):
        """Test that missing query_fn raises RuntimeError."""
        spec1 = Spec(str, name="scope")
        spec2 = Spec(bool, name="found")
        op = Operable([spec1, spec2])

        p = phrase(
            op,
            inputs={"scope"},
            outputs={"found"},
            crud=CrudPattern(table="tokens", lookup={"scope"}),
            result_parser=lambda row: {"found": row is not None},
            name="find_token",
        )

        ctx = MockContext()  # No query_fn

        with pytest.raises(RuntimeError, match="query_fn"):
            await p({"scope": "test"}, ctx)

    @pytest.mark.anyio
    async def test_result_resolution_priorities(self):
        """Test result field priority: ctx metadata > options > row > parser."""
        spec1 = Spec(str, name="scope")
        spec2 = Spec(str, name="from_ctx")
        spec3 = Spec(str, name="from_row")
        spec4 = Spec(str, name="from_parser")
        op = Operable([spec1, spec2, spec3, spec4])

        p = phrase(
            op,
            inputs={"scope"},
            outputs={"scope", "from_ctx", "from_row", "from_parser"},
            crud=CrudPattern(table="test_table", lookup={"scope"}),
            result_parser=lambda row: {"from_parser": "parsed"},
            name="test_priorities",
        )

        async def mock_query_fn(table, operation, where, data, ctx):
            return {"from_row": "row_value", "from_ctx": "should_not_use"}

        ctx = MockContext(
            query_fn=mock_query_fn,
            metadata={"from_ctx": "ctx_value"},
        )
        result = await p({"scope": "test"}, ctx)

        assert result.from_ctx == "ctx_value"  # Priority 1: ctx metadata
        assert result.scope == "test"  # Priority 2: pass-through from options
        assert result.from_row == "row_value"  # Priority 3: from row
        assert result.from_parser == "parsed"  # Priority 4: from parser


# =============================================================================
# Tests: Phrase.as_operation
# =============================================================================


class TestPhraseAsOperation:
    """Tests for Phrase.as_operation DAG integration."""

    def test_as_operation_with_options(self):
        """Phrase.as_operation should create Operation from direct options."""
        spec1 = Spec(str, name="name")
        spec2 = Spec(int, name="count")
        op = Operable([spec1, spec2])

        async def handler(options, ctx):
            return {"count": len(options.name)}

        p = Phrase(
            name="count_chars",
            operable=op,
            inputs={"name"},
            outputs={"count"},
            handler=handler,
        )

        operation = p.as_operation(options={"name": "hello"})

        assert operation.parameters == {"name": "hello"}
        assert operation.metadata["name"] == "count_chars"
        assert operation.metadata["phrase"] is True

    def test_as_operation_with_available_data(self):
        """Phrase.as_operation should extract inputs from available_data."""
        spec1 = Spec(str, name="name")
        spec2 = Spec(int, name="age")
        spec3 = Spec(bool, name="valid")
        op = Operable([spec1, spec2, spec3])

        async def handler(options, ctx):
            return {"valid": True}

        p = Phrase(
            name="validate",
            operable=op,
            inputs={"name", "age"},
            outputs={"valid"},
            handler=handler,
        )

        available = {"name": "Alice", "age": 30, "extra": "ignored"}
        operation = p.as_operation(available_data=available)

        assert operation.parameters == {"name": "Alice", "age": 30}

    def test_as_operation_missing_inputs_raises(self):
        """Phrase.as_operation should raise for missing inputs."""
        spec1 = Spec(str, name="name")
        spec2 = Spec(int, name="count")
        op = Operable([spec1, spec2])

        async def handler(options, ctx):
            return {"count": 0}

        p = Phrase(
            name="test",
            operable=op,
            inputs={"name"},
            outputs={"count"},
            handler=handler,
        )

        with pytest.raises(ValueError, match="Missing required inputs"):
            p.as_operation(available_data={"other": "data"})

    def test_as_operation_requires_options_or_available_data(self):
        """Phrase.as_operation should require either options or available_data."""
        spec1 = Spec(str, name="name")
        spec2 = Spec(int, name="count")
        op = Operable([spec1, spec2])

        async def handler(options, ctx):
            return {"count": 0}

        p = Phrase(
            name="test",
            operable=op,
            inputs={"name"},
            outputs={"count"},
            handler=handler,
        )

        with pytest.raises(ValueError, match="Either options or available_data"):
            p.as_operation()

    def test_as_operation_with_metadata(self):
        """Phrase.as_operation should accept additional metadata."""
        spec1 = Spec(str, name="name")
        spec2 = Spec(int, name="count")
        op = Operable([spec1, spec2])

        async def handler(options, ctx):
            return {"count": 0}

        p = Phrase(
            name="test",
            operable=op,
            inputs={"name"},
            outputs={"count"},
            handler=handler,
        )

        operation = p.as_operation(options={"name": "test"}, custom_key="custom_value")

        assert operation.metadata["custom_key"] == "custom_value"
