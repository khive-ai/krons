# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.rules.base - Rule, RuleParams, RuleQualifier."""

import pytest

from krons.enforcement.rule import Rule, RuleParams, RuleQualifier, ValidationError


class TestRuleQualifier:
    """Test RuleQualifier enum."""

    def test_qualifier_values(self):
        """RuleQualifier should have FIELD, ANNOTATION, CONDITION values."""
        assert RuleQualifier.FIELD is not None
        assert RuleQualifier.ANNOTATION is not None
        assert RuleQualifier.CONDITION is not None

    def test_from_str_valid_values(self):
        """RuleQualifier.from_str should parse valid strings."""
        assert RuleQualifier.from_str("FIELD") == RuleQualifier.FIELD
        assert RuleQualifier.from_str("field") == RuleQualifier.FIELD
        assert RuleQualifier.from_str("  FIELD  ") == RuleQualifier.FIELD

        assert RuleQualifier.from_str("ANNOTATION") == RuleQualifier.ANNOTATION
        assert RuleQualifier.from_str("annotation") == RuleQualifier.ANNOTATION

        assert RuleQualifier.from_str("CONDITION") == RuleQualifier.CONDITION
        assert RuleQualifier.from_str("condition") == RuleQualifier.CONDITION

    def test_from_str_invalid_value_raises(self):
        """RuleQualifier.from_str should raise for invalid strings."""
        with pytest.raises(ValueError, match="Unknown RuleQualifier"):
            RuleQualifier.from_str("INVALID")

        with pytest.raises(ValueError, match="Unknown RuleQualifier"):
            RuleQualifier.from_str("")


class TestRuleParams:
    """Test RuleParams dataclass."""

    def test_default_values(self):
        """RuleParams should have sensible defaults."""
        params = RuleParams()

        assert params.apply_types == set()
        assert params.apply_fields == set()
        assert params.default_qualifier == RuleQualifier.FIELD
        assert params.auto_fix is False
        assert params.kw == {}

    def test_custom_values(self):
        """RuleParams should accept custom values."""
        params = RuleParams(
            apply_types={str, int},
            apply_fields={"name", "value"},
            default_qualifier=RuleQualifier.ANNOTATION,
            auto_fix=True,
            kw={"extra": "data"},
        )

        assert params.apply_types == {str, int}
        assert params.apply_fields == {"name", "value"}
        assert params.default_qualifier == RuleQualifier.ANNOTATION
        assert params.auto_fix is True
        assert params.kw == {"extra": "data"}

    def test_immutability(self):
        """RuleParams should be frozen (immutable)."""
        params = RuleParams()

        with pytest.raises(AttributeError):
            params.auto_fix = True

    def test_with_updates(self):
        """RuleParams.with_updates should create new instance with changes."""
        params = RuleParams(apply_types={str}, auto_fix=False)
        updated = params.with_updates(auto_fix=True, kw={"new": "data"})

        # Original unchanged
        assert params.auto_fix is False
        assert params.kw == {}

        # New instance has updates
        assert updated.auto_fix is True
        assert updated.kw == {"new": "data"}
        assert updated.apply_types == {str}


class ConcreteRule(Rule):
    """Concrete Rule implementation for testing."""

    def __init__(self, params: RuleParams | None = None, **kw):
        if params is None:
            params = RuleParams(
                apply_types={str},
                apply_fields={"name"},
                default_qualifier=RuleQualifier.FIELD,
                auto_fix=False,
            )
        super().__init__(params, **kw)
        self._validate_calls = []

    async def validate(self, v, t, **kw):
        self._validate_calls.append((v, t))
        if not isinstance(v, str):
            raise ValueError(f"Expected str, got {type(v).__name__}")

    async def perform_fix(self, v, t):
        return str(v)


class TestRuleBase:
    """Test Rule base class."""

    def test_rule_properties(self):
        """Rule should expose params as properties."""
        params = RuleParams(
            apply_types={str},
            apply_fields={"name"},
            default_qualifier=RuleQualifier.ANNOTATION,
            auto_fix=True,
            kw={"extra": "data"},
        )
        rule = ConcreteRule(params)

        assert rule.apply_types == {str}
        assert rule.apply_fields == {"name"}
        assert rule.default_qualifier == RuleQualifier.ANNOTATION
        assert rule.auto_fix is True
        assert rule.validation_kwargs == {"extra": "data"}

    def test_rule_merges_kwargs(self):
        """Rule should merge constructor kwargs with params.kw."""
        params = RuleParams(kw={"a": 1, "b": 2})
        rule = ConcreteRule(params, b=20, c=30)

        # b should be overwritten, c should be added
        assert rule.validation_kwargs == {"a": 1, "b": 20, "c": 30}


class TestRuleApply:
    """Test Rule.apply() qualifier matching."""

    @pytest.mark.anyio
    async def test_apply_field_qualifier(self):
        """Rule should apply by field name."""
        params = RuleParams(apply_fields={"name", "title"})
        rule = ConcreteRule(params)

        assert await rule.apply("name", "test", str) is True
        assert await rule.apply("title", "test", str) is True
        assert await rule.apply("other", "test", str) is False

    @pytest.mark.anyio
    async def test_apply_annotation_qualifier(self):
        """Rule should apply by type annotation."""
        params = RuleParams(
            apply_types={str},
            default_qualifier=RuleQualifier.ANNOTATION,
        )
        rule = ConcreteRule(params)

        assert await rule.apply("field", "test", str) is True
        assert await rule.apply("field", 123, int) is False

    @pytest.mark.anyio
    async def test_apply_qualifier_override(self):
        """Rule should respect qualifier override."""
        params = RuleParams(
            apply_types={str},
            apply_fields={"name"},
            default_qualifier=RuleQualifier.FIELD,
        )
        rule = ConcreteRule(params)

        # Field match (default)
        assert await rule.apply("name", "test", int) is True

        # Force annotation qualifier - should also try field due to precedence
        assert await rule.apply("other", "test", str, qualifier=RuleQualifier.ANNOTATION) is True

    @pytest.mark.anyio
    async def test_apply_condition_not_implemented(self):
        """Rule without rule_condition should skip CONDITION qualifier."""
        params = RuleParams(default_qualifier=RuleQualifier.CONDITION)
        rule = ConcreteRule(params)

        # Should return False because rule_condition raises NotImplementedError
        assert await rule.apply("field", "test", str) is False


class ConditionRule(ConcreteRule):
    """Rule with custom condition."""

    async def rule_condition(self, k, v, t, **kw):
        return k.startswith("special_")


class TestRuleCondition:
    """Test Rule.rule_condition() custom qualifier."""

    @pytest.mark.anyio
    async def test_condition_qualifier(self):
        """Rule with rule_condition should apply by condition."""
        params = RuleParams(default_qualifier=RuleQualifier.CONDITION)
        rule = ConditionRule(params)

        assert await rule.apply("special_field", "test", str) is True
        assert await rule.apply("normal_field", "test", str) is False


class TestRuleValidate:
    """Test Rule.validate() method."""

    @pytest.mark.anyio
    async def test_validate_passes(self):
        """validate() should pass for valid value."""
        rule = ConcreteRule()
        # validate() doesn't raise for valid values
        await rule.validate("valid", str)

    @pytest.mark.anyio
    async def test_validate_fails(self):
        """validate() should raise for invalid value."""
        rule = ConcreteRule()

        with pytest.raises(ValueError, match="Expected str"):
            await rule.validate(123, int)


class TestRuleInvoke:
    """Test Rule.invoke() execution."""

    @pytest.mark.anyio
    async def test_invoke_valid_value(self):
        """invoke() should return valid value unchanged."""
        rule = ConcreteRule()

        result = await rule.invoke("field", "test", str)
        assert result == "test"

    @pytest.mark.anyio
    async def test_invoke_invalid_without_autofix(self):
        """invoke() should raise ValidationError for invalid value without auto_fix."""
        params = RuleParams(apply_types={str}, auto_fix=False)
        rule = ConcreteRule(params)

        with pytest.raises(ValidationError, match="Failed to validate"):
            await rule.invoke("field", 123, str)

    @pytest.mark.anyio
    async def test_invoke_invalid_with_autofix(self):
        """invoke() should auto-fix invalid value when auto_fix=True."""
        params = RuleParams(apply_types={str}, auto_fix=True)
        rule = ConcreteRule(params)

        result = await rule.invoke("field", 123, str)
        assert result == "123"

    @pytest.mark.anyio
    async def test_invoke_autofix_override(self):
        """invoke() should respect auto_fix parameter override."""
        params = RuleParams(apply_types={str}, auto_fix=False)
        rule = ConcreteRule(params)

        # Rule has auto_fix=False, but override with True
        result = await rule.invoke("field", 123, str, auto_fix=True)
        assert result == "123"

    @pytest.mark.anyio
    async def test_invoke_autofix_fail_raises(self):
        """invoke() should raise if auto-fix fails."""

        class UnfixableRule(Rule):
            async def validate(self, v, t, **kw):
                raise ValueError("Invalid")

            async def perform_fix(self, v, t):
                raise ValueError("Cannot fix")

        params = RuleParams(auto_fix=True)
        rule = UnfixableRule(params)

        with pytest.raises(ValidationError, match="Failed to fix"):
            await rule.invoke("field", "bad", str)


class TestRulePerformFix:
    """Test Rule.perform_fix() method."""

    @pytest.mark.anyio
    async def test_perform_fix_default_raises(self):
        """perform_fix() should raise NotImplementedError by default."""

        class NoFixRule(Rule):
            async def validate(self, v, t, **kw):
                raise ValueError("Invalid")

        params = RuleParams()
        rule = NoFixRule(params)

        with pytest.raises(NotImplementedError, match="perform_fix"):
            await rule.perform_fix("value", str)


class TestRuleRepr:
    """Test Rule.__repr__."""

    def test_repr_output(self):
        """__repr__ should show rule configuration."""
        params = RuleParams(
            apply_types={str, int},
            apply_fields={"name"},
            auto_fix=True,
        )
        rule = ConcreteRule(params)

        repr_str = repr(rule)
        assert "ConcreteRule" in repr_str
        assert "str" in repr_str or "int" in repr_str
        assert "auto_fix=True" in repr_str


class TestValidationError:
    """Test ValidationError exception."""

    def test_validation_error_is_exception(self):
        """ValidationError should be an Exception."""
        error = ValidationError("Test error")
        assert isinstance(error, Exception)
        assert str(error) == "Test error"

    def test_validation_error_can_chain(self):
        """ValidationError should support exception chaining."""
        cause = ValueError("Original error")
        error = ValidationError("Wrapped error")
        error.__cause__ = cause

        assert error.__cause__ is cause
