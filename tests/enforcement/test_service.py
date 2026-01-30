# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for enforcement service components.

Tests RequestContext, ActionMeta, KronConfig, and the @action decorator.
Full KronService tests are in tests/services/ due to abstract method dependencies.
"""

from uuid import uuid4

import pytest

from krons.enforcement import (
    ActionMeta,
    EnforcementLevel,
    KronConfig,
    RequestContext,
    ResolvedPolicy,
    action,
    get_action_meta,
)
from krons.specs import Operable, Spec

# =============================================================================
# Test Fixtures
# =============================================================================

TEST_PROVIDER = "test-provider"
TEST_NAME = "test-service"


def make_config(**kwargs) -> KronConfig:
    """Create a KronConfig with test defaults."""
    defaults = {"provider": TEST_PROVIDER, "name": TEST_NAME}
    defaults.update(kwargs)
    return KronConfig(**defaults)


# =============================================================================
# Test RequestContext
# =============================================================================


class TestRequestContext:
    """Tests for RequestContext dataclass."""

    def test_basic_creation(self):
        """Test basic RequestContext creation."""
        ctx = RequestContext(name="test.action")
        assert ctx.name == "test.action"
        assert ctx.id is not None
        assert ctx.session_id is None
        assert ctx.branch_id is None
        assert ctx.metadata == {}

    def test_with_session_and_branch(self):
        """Test RequestContext with session and branch IDs."""
        session_id = uuid4()
        branch_id = uuid4()
        ctx = RequestContext(
            name="test.action",
            session_id=session_id,
            branch_id=branch_id,
        )
        assert ctx.session_id == session_id
        assert ctx.branch_id == branch_id

    def test_metadata_via_kwargs(self):
        """Test that extra kwargs become metadata."""
        ctx = RequestContext(
            name="test.action",
            actor_id=uuid4(),
            tenant_id=uuid4(),
            custom="value",
        )
        assert "actor_id" in ctx.metadata
        assert "tenant_id" in ctx.metadata
        assert ctx.metadata["custom"] == "value"

    def test_property_accessors(self):
        """Test property accessors for common metadata fields."""
        actor = uuid4()
        subject = uuid4()
        tenant = uuid4()
        ctx = RequestContext(
            name="test",
            actor_id=actor,
            subject_id=subject,
            tenant_id=tenant,
        )
        assert ctx.actor_id == actor
        assert ctx.subject_id == subject
        assert ctx.tenant_id == tenant

    def test_property_accessors_raise_when_missing(self):
        """Test metadata accessors raise AttributeError when not set."""
        ctx = RequestContext(name="test")
        with pytest.raises(AttributeError):
            _ = ctx.actor_id
        with pytest.raises(AttributeError):
            _ = ctx.subject_id
        with pytest.raises(AttributeError):
            _ = ctx.tenant_id
        # hasattr correctly returns False
        assert not hasattr(ctx, "actor_id")
        assert not hasattr(ctx, "subject_id")
        assert not hasattr(ctx, "tenant_id")


# =============================================================================
# Test Action Decorator
# =============================================================================


class TestActionDecorator:
    """Tests for @action method decorator."""

    def test_action_decorator_sets_metadata(self):
        """Test that @action sets the _kron_action attribute."""

        @action(
            name="consent.grant",
            inputs={"permissions", "subject_id"},
            outputs={"consent_id", "granted_at"},
        )
        async def handler(self, options, ctx):
            return {"consent_id": "123", "granted_at": "now"}

        meta = get_action_meta(handler)
        assert meta is not None
        assert meta.name == "consent.grant"
        assert meta.inputs == frozenset({"permissions", "subject_id"})
        assert meta.outputs == frozenset({"consent_id", "granted_at"})

    def test_action_decorator_with_hooks(self):
        """Test @action with pre and post hooks."""

        @action(
            name="test.action",
            pre_hooks=["validate"],
            post_hooks=["audit", "notify"],
        )
        async def handler(self, options, ctx):
            return {}

        meta = get_action_meta(handler)
        assert meta.pre_hooks == ("validate",)
        assert meta.post_hooks == ("audit", "notify")

    def test_action_decorator_defaults(self):
        """Test @action with default values."""

        @action(name="minimal")
        async def handler(self, options, ctx):
            return {}

        meta = get_action_meta(handler)
        assert meta.name == "minimal"
        assert meta.inputs == frozenset()
        assert meta.outputs == frozenset()
        assert meta.pre_hooks == ()
        assert meta.post_hooks == ()


# =============================================================================
# Test KronConfig
# =============================================================================


class TestKronConfig:
    """Tests for KronConfig."""

    def test_default_values(self):
        """Test default config values."""
        config = make_config()
        assert config.operable is None
        assert config.action_timeout is None
        assert config.use_policies is True
        assert config.policy_timeout == 10.0
        assert config.fail_open_on_engine_error is False
        assert config.hooks == {}

    def test_with_operable(self):
        """Test config with operable."""
        operable = Operable(
            [
                Spec(str, name="field1"),
                Spec(int, name="field2"),
            ]
        )
        config = make_config(operable=operable)
        assert config.operable is operable


# =============================================================================
# Test ActionMeta
# =============================================================================


class TestActionMeta:
    """Tests for ActionMeta dataclass."""

    def test_action_meta_creation(self):
        """Test creating ActionMeta."""
        meta = ActionMeta(
            name="test.action",
            inputs=frozenset({"a", "b"}),
            outputs=frozenset({"c"}),
            pre_hooks=("hook1",),
            post_hooks=("hook2", "hook3"),
        )
        assert meta.name == "test.action"
        assert meta.inputs == frozenset({"a", "b"})
        assert meta.outputs == frozenset({"c"})
        assert meta.pre_hooks == ("hook1",)
        assert meta.post_hooks == ("hook2", "hook3")

    def test_action_meta_is_frozen(self):
        """Test that ActionMeta is frozen."""
        meta = ActionMeta(name="test")
        with pytest.raises(AttributeError):
            meta.name = "changed"


# =============================================================================
# Test ResolvedPolicy
# =============================================================================


class TestResolvedPolicy:
    """Tests for ResolvedPolicy."""

    def test_resolved_policy_creation(self):
        """Test creating ResolvedPolicy."""
        policy = ResolvedPolicy(policy_id="test.policy")
        assert policy.policy_id == "test.policy"
        assert policy.enforcement == "hard_mandatory"
        assert policy.metadata == {}

    def test_resolved_policy_with_metadata(self):
        """Test ResolvedPolicy with custom metadata."""
        policy = ResolvedPolicy(
            policy_id="test.policy",
            enforcement="advisory",
            metadata={"source": "charter"},
        )
        assert policy.enforcement == "advisory"
        assert policy.metadata == {"source": "charter"}


# =============================================================================
# Test EnforcementLevel
# =============================================================================


class TestEnforcementLevel:
    """Tests for EnforcementLevel usage in service context."""

    def test_is_blocking_for_results(self):
        """Test is_blocking classmethod."""

        class MockResult:
            enforcement = "hard_mandatory"

        assert EnforcementLevel.is_blocking(MockResult()) is True

        MockResult.enforcement = "soft_mandatory"
        assert EnforcementLevel.is_blocking(MockResult()) is True

        MockResult.enforcement = "advisory"
        assert EnforcementLevel.is_blocking(MockResult()) is False

    def test_is_advisory_for_results(self):
        """Test is_advisory classmethod."""

        class MockResult:
            enforcement = "advisory"

        assert EnforcementLevel.is_advisory(MockResult()) is True

        MockResult.enforcement = "hard_mandatory"
        assert EnforcementLevel.is_advisory(MockResult()) is False


# =============================================================================
# Test KronService
# =============================================================================


class TestKronService:
    """Tests for KronService implementation."""

    def test_kron_service_creation(self):
        """Test creating a KronService subclass."""
        from krons.enforcement.service import KronService
        from krons.services.backend import Calling

        class TestService(KronService):
            @property
            def event_type(self):
                return Calling

            @action(name="test.echo", inputs={"message"}, outputs={"response"})
            async def _handle_echo(self, options, ctx):
                return {"response": options.get("message", "")}

        service = TestService(config=make_config())
        assert "test.echo" in service._action_registry

    def test_kron_service_has_engine_property(self):
        """Test has_engine property."""
        from krons.enforcement.service import KronService
        from krons.services.backend import Calling

        class TestService(KronService):
            @property
            def event_type(self):
                return Calling

        service = TestService(config=make_config())
        assert service.has_engine is False

        # Create a mock engine that satisfies the PolicyEngine protocol
        class MockPolicyEngine:
            async def evaluate(self, policy_id, input_data, **options):
                return {}

            async def evaluate_batch(self, policy_ids, input_data, **options):
                return []

        engine = MockPolicyEngine()
        service = TestService(config=make_config(), policy_engine=engine)
        assert service.has_engine is True

    def test_kron_service_has_resolver_property(self):
        """Test has_resolver property."""
        from krons.enforcement.service import KronService
        from krons.services.backend import Calling

        class TestService(KronService):
            @property
            def event_type(self):
                return Calling

        service = TestService(config=make_config())
        assert service.has_resolver is False

        # Create a mock resolver that satisfies the PolicyResolver protocol
        class MockPolicyResolver:
            def resolve(self, ctx):
                return []

        resolver = MockPolicyResolver()
        service = TestService(config=make_config(), policy_resolver=resolver)
        assert service.has_resolver is True

    def test_fetch_handler_unknown_action(self):
        """Test _fetch_handler raises ValueError for unknown action."""
        from krons.enforcement.service import KronService
        from krons.services.backend import Calling

        class TestService(KronService):
            @property
            def event_type(self):
                return Calling

        service = TestService(config=make_config())
        with pytest.raises(ValueError, match="Unknown action"):
            service._fetch_handler("nonexistent.action")

    @pytest.mark.anyio
    async def test_call_action(self):
        """Test calling an action via KronService.call()."""
        from krons.enforcement.service import KronService
        from krons.services.backend import Calling

        class TestService(KronService):
            @property
            def event_type(self):
                return Calling

            @action(name="math.double", inputs={"value"}, outputs={"result"})
            async def _handle_double(self, options, ctx):
                return {"result": options.get("value", 0) * 2}

        service = TestService(config=make_config(use_policies=False))
        ctx = RequestContext(name="math.double")

        result = await service.call("math.double", {"value": 21}, ctx)
        assert result["result"] == 42

    @pytest.mark.anyio
    async def test_call_action_with_hooks(self):
        """Test calling an action with pre and post hooks."""
        from krons.enforcement.service import KronService
        from krons.services.backend import Calling

        hook_calls = []

        async def pre_hook(service, options, ctx, result=None):
            hook_calls.append(("pre", options))

        async def post_hook(service, options, ctx, result=None):
            hook_calls.append(("post", result))

        class TestService(KronService):
            @property
            def event_type(self):
                return Calling

            @action(
                name="test.hooked",
                pre_hooks=["validate"],
                post_hooks=["audit"],
            )
            async def _handle_hooked(self, options, ctx):
                return {"status": "ok"}

        config = make_config(
            use_policies=False,
            hooks={"validate": pre_hook, "audit": post_hook},
        )
        service = TestService(config=config)
        ctx = RequestContext(name="test.hooked")

        result = await service.call("test.hooked", {"input": "data"}, ctx)

        assert len(hook_calls) == 2
        assert hook_calls[0][0] == "pre"
        assert hook_calls[1][0] == "post"
        assert result["status"] == "ok"

    def test_to_pascal_conversion(self):
        """Test _to_pascal helper function."""
        from krons.enforcement.service import _to_pascal

        assert _to_pascal("consent.grant") == "ConsentGrant"
        assert _to_pascal("consent_grant") == "ConsentGrant"
        assert _to_pascal("simple") == "Simple"
        assert _to_pascal("a.b.c") == "ABC"

    @pytest.mark.xfail(
        reason="Pre-existing bug: service passes frozen=True to compose_structure but "
        "PydanticSpecAdapter doesn't support it"
    )
    def test_build_action_types_with_operable(self):
        """Test that action types are built from operable."""
        from krons.enforcement.service import KronService
        from krons.services.backend import Calling

        operable = Operable(
            [
                Spec(str, name="message"),
                Spec(str, name="response"),
            ]
        )

        class TestService(KronService):
            @property
            def event_type(self):
                return Calling

            @action(name="test.echo", inputs={"message"}, outputs={"response"})
            async def _handle_echo(self, options, ctx):
                return {"response": options.get("message", "")}

        service = TestService(config=make_config(operable=operable))
        _, meta = service._action_registry["test.echo"]

        # Types should be built
        assert meta._options_type is not None
        assert meta._result_type is not None

    @pytest.mark.anyio
    async def test_call_action_with_missing_hook(self):
        """Test that missing hooks log warnings but don't fail."""
        from krons.enforcement.service import KronService
        from krons.services.backend import Calling

        class TestService(KronService):
            @property
            def event_type(self):
                return Calling

            @action(
                name="test.hooked",
                pre_hooks=["missing_hook"],  # This hook is not defined
            )
            async def _handle_hooked(self, options, ctx):
                return {"status": "ok"}

        # Config without the required hook
        config = make_config(use_policies=False, hooks={})
        service = TestService(config=config)
        ctx = RequestContext(name="test.hooked")

        # Should succeed despite missing hook (just logs warning)
        result = await service.call("test.hooked", {}, ctx)
        assert result["status"] == "ok"

    @pytest.mark.anyio
    async def test_call_action_with_failing_hook(self):
        """Test that failing hooks are logged but don't block action."""
        from krons.enforcement.service import KronService
        from krons.services.backend import Calling

        async def failing_hook(service, options, ctx, result=None):
            raise ValueError("Hook failed intentionally")

        class TestService(KronService):
            @property
            def event_type(self):
                return Calling

            @action(
                name="test.hooked",
                pre_hooks=["failing"],
            )
            async def _handle_hooked(self, options, ctx):
                return {"status": "ok"}

        config = make_config(use_policies=False, hooks={"failing": failing_hook})
        service = TestService(config=config)
        ctx = RequestContext(name="test.hooked")

        # Should succeed despite hook error (just logs error)
        result = await service.call("test.hooked", {}, ctx)
        assert result["status"] == "ok"

    def test_build_action_warns_on_invalid_inputs(self, caplog):
        """Test that invalid inputs in action trigger warning."""
        import logging

        from krons.enforcement.service import KronService
        from krons.services.backend import Calling

        operable = Operable(
            [
                Spec(str, name="message"),  # Only 'message' is defined
            ]
        )

        with caplog.at_level(logging.WARNING):
            try:

                class TestService(KronService):
                    @property
                    def event_type(self):
                        return Calling

                    @action(name="test.echo", inputs={"nonexistent"})  # Invalid input
                    async def _handle_echo(self, options, ctx):
                        return {}

                service = TestService(config=make_config(operable=operable))
            except TypeError:
                # May fail due to frozen bug, but warning should still be logged
                pass

        # Should have logged a warning about invalid inputs
        assert any("inputs not in operable" in rec.message for rec in caplog.records)

    def test_build_action_warns_on_invalid_outputs(self, caplog):
        """Test that invalid outputs in action trigger warning."""
        import logging

        from krons.enforcement.service import KronService
        from krons.services.backend import Calling

        operable = Operable(
            [
                Spec(str, name="message"),
            ]
        )

        with caplog.at_level(logging.WARNING):
            try:

                class TestService(KronService):
                    @property
                    def event_type(self):
                        return Calling

                    @action(name="test.echo", outputs={"invalid_output"})  # Invalid output
                    async def _handle_echo(self, options, ctx):
                        return {}

                service = TestService(config=make_config(operable=operable))
            except TypeError:
                # May fail due to frozen bug, but warning should still be logged
                pass

        # Should have logged a warning about invalid outputs
        assert any("outputs not in operable" in rec.message for rec in caplog.records)
