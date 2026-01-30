# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.specs.phrase - operation templates from Specs."""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from krons.enforcement.context import RequestContext
from krons.errors import ValidationError
from krons.specs import Operable, Spec
from krons.specs.phrase import CrudOperation, CrudPattern, Phrase, _to_pascal, phrase


class TestRequestContextGetattr:
    """Tests for RequestContext.__getattr__ fix (B1)."""

    def test_hasattr_false_for_missing_metadata(self):
        """hasattr must return False for keys not in metadata."""
        ctx = RequestContext(name="test")
        assert not hasattr(ctx, "tenant_id")
        assert not hasattr(ctx, "totally_fake")

    def test_hasattr_true_for_present_metadata(self):
        """hasattr must return True for keys present in metadata."""
        tid = uuid4()
        ctx = RequestContext(name="test", tenant_id=tid)
        assert hasattr(ctx, "tenant_id")
        assert ctx.tenant_id == tid

    def test_getattr_raises_for_missing_metadata(self):
        """Accessing missing metadata key must raise AttributeError."""
        ctx = RequestContext(name="test")
        with pytest.raises(AttributeError):
            _ = ctx.tenant_id

    def test_slots_still_accessible(self):
        """Slot fields remain accessible normally."""
        ctx = RequestContext(name="test_op")
        assert ctx.name == "test_op"
        assert ctx.conn is None
        assert ctx.now is None

    def test_underscore_attrs_raise(self):
        """Private attributes must raise AttributeError."""
        ctx = RequestContext(name="test")
        with pytest.raises(AttributeError):
            _ = ctx._private


class TestPhrase:
    """Tests for Phrase class."""

    def test_phrase_initialization(self):
        """Phrase should initialize with name, operable, inputs, outputs, handler."""
        spec1 = Spec(UUID, name="user_id")
        spec2 = Spec(bool, name="verified")
        op = Operable((spec1, spec2))

        async def handler(options, ctx):
            return {"verified": True}

        phrase = Phrase(
            name="verify_user",
            operable=op,
            inputs={"user_id"},
            outputs={"verified"},
            handler=handler,
        )

        assert phrase.name == "verify_user"
        assert phrase.inputs == ("user_id",)
        assert phrase.outputs == ("verified",)
        assert phrase.handler is handler

    def test_phrase_options_type_generated(self):
        """Phrase.options_type should generate typed options from inputs."""
        spec1 = Spec(UUID, name="user_id")
        spec2 = Spec(int, name="max_age", default=300)
        spec3 = Spec(bool, name="verified")
        op = Operable((spec1, spec2, spec3))

        async def handler(options, ctx):
            return {"verified": True}

        phrase = Phrase(
            name="check_auth",
            operable=op,
            inputs={"user_id", "max_age"},
            outputs={"verified"},
            handler=handler,
        )

        Options = phrase.options_type
        assert Options.__name__ == "CheckAuthOptions"

        # Should be usable
        uid = uuid4()
        opts = Options(user_id=uid)
        assert opts.user_id == uid
        assert opts.max_age == 300  # default

    def test_phrase_result_type_generated(self):
        """Phrase.result_type should generate typed result from outputs."""
        spec1 = Spec(UUID, name="user_id")
        spec2 = Spec(bool, name="verified")
        spec3 = Spec(str, name="reason", nullable=True)
        op = Operable((spec1, spec2, spec3))

        async def handler(options, ctx):
            return {"verified": True, "reason": None}

        phrase = Phrase(
            name="verify_status",
            operable=op,
            inputs={"user_id"},
            outputs={"verified", "reason"},
            handler=handler,
        )

        Result = phrase.result_type
        assert Result.__name__ == "VerifyStatusResult"

    def test_phrase_types_cached(self):
        """Phrase types should be lazily computed and cached."""
        spec1 = Spec(str, name="query")
        spec2 = Spec(bool, name="found")
        op = Operable((spec1, spec2))

        async def handler(options, ctx):
            return {"found": True}

        phrase = Phrase(
            name="search",
            operable=op,
            inputs={"query"},
            outputs={"found"},
            handler=handler,
        )

        # Access twice - should be same object
        t1 = phrase.options_type
        t2 = phrase.options_type
        assert t1 is t2

        r1 = phrase.result_type
        r2 = phrase.result_type
        assert r1 is r2

    @pytest.mark.anyio
    async def test_phrase_call(self):
        """Phrase.__call__ should validate options and invoke handler."""
        spec1 = Spec(str, name="name")
        spec2 = Spec(int, name="count")
        op = Operable((spec1, spec2))

        async def handler(options, ctx):
            return {"count": len(options.name)}

        phrase = Phrase(
            name="count_chars",
            operable=op,
            inputs={"name"},
            outputs={"count"},
            handler=handler,
        )

        Options = phrase.options_type
        result = await phrase(Options(name="hello"), None)
        # Result is now validated and returned as typed result object
        assert result.count == 5


class TestToPascal:
    """Tests for _to_pascal helper."""

    def test_single_word(self):
        assert _to_pascal("check") == "Check"

    def test_snake_case(self):
        assert _to_pascal("verify_consent_token") == "VerifyConsentToken"

    def test_multiple_underscores(self):
        assert _to_pascal("require_monitoring_active") == "RequireMonitoringActive"

    def test_already_capitalized(self):
        assert _to_pascal("Check") == "Check"

    def test_empty_string(self):
        assert _to_pascal("") == ""


class TestPhraseDecorator:
    """Tests for @phrase decorator."""

    @pytest.mark.anyio
    async def test_phrase_decorator_creates_phrase(self):
        """Test @phrase decorator creates a Phrase instance."""
        spec1 = Spec(str, name="user_id")
        spec2 = Spec(bool, name="verified")
        op = Operable((spec1, spec2))

        @phrase(op, inputs={"user_id"}, outputs={"verified"})
        async def verify_user(options, ctx):
            return {"verified": True}

        assert isinstance(verify_user, Phrase)
        assert verify_user.name == "verify_user"
        assert "user_id" in verify_user.inputs
        assert "verified" in verify_user.outputs

    @pytest.mark.anyio
    async def test_phrase_decorator_callable(self):
        """Test decorated phrase is callable."""
        spec1 = Spec(str, name="name")
        spec2 = Spec(int, name="length")
        op = Operable((spec1, spec2))

        @phrase(op, inputs={"name"}, outputs={"length"})
        async def get_length(options, ctx):
            return {"length": len(options.name)}

        Options = get_length.options_type
        result = await get_length(Options(name="hello"), None)
        assert result.length == 5


class TestCrudPattern:
    """Tests for CrudPattern declarative CRUD."""

    def test_crud_pattern_defaults(self):
        """CrudPattern should normalize defaults."""
        crud = CrudPattern(table="users")
        assert crud.table == "users"
        assert crud.operation == CrudOperation.READ
        assert crud.lookup == frozenset()
        assert crud.set_fields == {}
        assert crud.defaults == {}

    def test_crud_pattern_operation_string(self):
        """CrudPattern should accept operation as string."""
        crud = CrudPattern(table="users", operation="insert")
        assert crud.operation == CrudOperation.INSERT

    def test_crud_pattern_lookup_set(self):
        """CrudPattern should normalize lookup to frozenset."""
        crud = CrudPattern(table="users", lookup={"user_id", "scope"})
        assert crud.lookup == frozenset({"user_id", "scope"})

    def test_crud_pattern_frozen(self):
        """CrudPattern should be immutable."""
        crud = CrudPattern(table="users")
        with pytest.raises(AttributeError):
            crud.table = "other"

    def test_crud_pattern_rejects_invalid_table(self):
        """CrudPattern should reject SQL-unsafe table names."""
        with pytest.raises(ValidationError):
            CrudPattern(table="users; DROP TABLE --")
        with pytest.raises(ValidationError):
            CrudPattern(table="")

    def test_crud_pattern_filters_immutable(self):
        """CrudPattern internal dicts should be immutable after construction."""
        crud = CrudPattern(table="users", filters={"status": "active"})
        with pytest.raises(TypeError):
            crud.filters["injected"] = "bad"

    def test_crud_pattern_defaults_immutable(self):
        """CrudPattern defaults should be immutable after construction."""
        crud = CrudPattern(table="users", defaults={"role": "user"})
        with pytest.raises(TypeError):
            crud.defaults["injected"] = "bad"

    def test_crud_pattern_set_fields_immutable(self):
        """CrudPattern set_fields should be immutable after construction."""
        crud = CrudPattern(
            table="users", operation="update", set_fields={"status": "active"}
        )
        with pytest.raises(TypeError):
            crud.set_fields["injected"] = "bad"


class TestCrudPhrase:
    """Tests for Phrase with CrudPattern."""

    def test_phrase_requires_handler_or_crud(self):
        """Phrase should require either handler or crud."""
        spec = Spec(str, name="name")
        op = Operable([spec])

        with pytest.raises(ValueError, match="Either handler or crud"):
            Phrase(name="test", operable=op, inputs={"name"}, outputs={"name"})

    def test_phrase_with_crud_creates_handler(self):
        """Phrase with crud should auto-generate handler."""
        spec1 = Spec(UUID, name="subject_id")
        spec2 = Spec(str, name="scope")
        spec3 = Spec(bool, name="has_consent")
        op = Operable([spec1, spec2, spec3])

        p = Phrase(
            name="verify_consent",
            operable=op,
            inputs={"subject_id", "scope"},
            outputs={"has_consent"},
            crud=CrudPattern(
                table="consent_tokens",
                operation="read",
                lookup={"subject_id", "scope"},
            ),
            result_parser=lambda row: {"has_consent": bool(row)},
        )

        assert p.handler is not None
        assert p.crud is not None
        assert p.crud.table == "consent_tokens"

    def test_phrase_direct_creation_requires_name(self):
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

    def test_phrase_direct_creation(self):
        """phrase() with crud returns Phrase directly."""
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
        assert p.crud.table == "tokens"

    @pytest.mark.anyio
    async def test_crud_read_handler(self):
        """Test auto-generated read handler with real RequestContext."""
        spec1 = Spec(UUID, name="subject_id")
        spec2 = Spec(str, name="scope")
        spec3 = Spec(bool, name="has_consent")
        spec4 = Spec(UUID, name="token_id", nullable=True)
        op = Operable([spec1, spec2, spec3, spec4])

        def result_parser(row):
            if row is None:
                return {"has_consent": False}
            return {"has_consent": row.get("status") == "active"}

        p = phrase(
            op,
            inputs={"subject_id", "scope"},
            outputs={"has_consent", "token_id", "subject_id"},
            crud=CrudPattern(table="consent_tokens", lookup={"subject_id", "scope"}),
            result_parser=result_parser,
            name="verify_consent",
        )

        subject_id = uuid4()
        token_id = uuid4()
        tenant_id = uuid4()

        async def mock_query_fn(table, operation, where, data, ctx):
            assert table == "consent_tokens"
            assert operation == "select_one"
            assert "subject_id" in where
            assert "tenant_id" in where
            assert where["tenant_id"] == tenant_id
            return {"status": "active", "token_id": token_id}

        ctx = RequestContext(
            name="verify_consent",
            query_fn=mock_query_fn,
            tenant_id=tenant_id,
        )
        result = await p({"subject_id": subject_id, "scope": "background"}, ctx)

        assert result.has_consent is True
        assert result.token_id == token_id
        assert result.subject_id == subject_id  # pass-through

    @pytest.mark.anyio
    async def test_crud_read_not_found(self):
        """Test read handler when row not found."""
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

        ctx = RequestContext(
            name="find_token",
            query_fn=mock_query_fn,
            tenant_id=uuid4(),
        )
        result = await p({"scope": "test"}, ctx)
        assert result.found is False
        assert result.scope == "test"  # pass-through

    @pytest.mark.anyio
    async def test_crud_insert_handler(self):
        """Test auto-generated insert handler with real RequestContext."""
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
            assert data["status"] == "active"
            assert data["tenant_id"] == tenant_id
            return {"token_id": token_id, "scope": data["scope"]}

        ctx = RequestContext(
            name="create_token",
            query_fn=mock_query_fn,
            tenant_id=tenant_id,
        )
        result = await p({"scope": "background"}, ctx)
        assert result.token_id == token_id
        assert result.scope == "background"

    @pytest.mark.anyio
    async def test_crud_update_handler(self):
        """Test auto-generated update handler with real RequestContext."""
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

        ctx = RequestContext(
            name="revoke_token",
            query_fn=mock_query_fn,
            tenant_id=tenant_id,
        )
        result = await p({"token_id": token_id}, ctx)
        assert result.updated is True

    @pytest.mark.anyio
    async def test_crud_soft_delete_handler(self):
        """Test auto-generated soft delete handler with real RequestContext."""
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
            assert operation == "update"
            assert where["tenant_id"] == tenant_id
            assert data["is_deleted"] is True
            assert data["deleted_at"] == "2025-01-01T00:00:00Z"
            return {"id": token_id}

        ctx = RequestContext(
            name="delete_token",
            query_fn=mock_query_fn,
            tenant_id=tenant_id,
            now="2025-01-01T00:00:00Z",
        )
        result = await p({"token_id": token_id}, ctx)
        assert result.deleted is True

    @pytest.mark.anyio
    async def test_crud_read_no_tenant(self):
        """Test read handler without tenant_id — should NOT inject tenant_id."""
        spec1 = Spec(str, name="scope")
        spec2 = Spec(bool, name="found")
        op = Operable([spec1, spec2])

        p = phrase(
            op,
            inputs={"scope"},
            outputs={"found"},
            crud=CrudPattern(table="tokens", lookup={"scope"}),
            result_parser=lambda row: {"found": row is not None},
            name="find_token_no_tenant",
        )

        async def mock_query_fn(table, operation, where, data, ctx):
            assert "tenant_id" not in where  # Must NOT inject tenant_id
            return {"scope": "test"}

        ctx = RequestContext(name="find_token", query_fn=mock_query_fn)
        result = await p({"scope": "test"}, ctx)
        assert result.found is True

    @pytest.mark.anyio
    async def test_crud_soft_delete_no_now(self):
        """Test soft delete without now — should NOT inject deleted_at."""
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
            name="delete_token_no_now",
        )

        async def mock_query_fn(table, operation, where, data, ctx):
            assert data["is_deleted"] is True
            assert "deleted_at" not in data  # now is None, should not set
            return {"id": token_id}

        ctx = RequestContext(name="delete_token", query_fn=mock_query_fn)
        result = await p({"token_id": token_id}, ctx)
        assert result.deleted is True

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

        ctx = RequestContext(
            name="test",
            query_fn=mock_query_fn,
            from_ctx="ctx_value",
        )
        result = await p({"scope": "test"}, ctx)
        assert result.from_ctx == "ctx_value"  # Priority 1: ctx metadata
        assert result.scope == "test"  # Priority 2: pass-through from options
        assert result.from_row == "row_value"  # Priority 3: from row
        assert result.from_parser == "parsed"  # Priority 4: from parser
