# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.rules.validator - Validator orchestration."""

import pytest

from krons.enforcement.common import NumberRule, StringRule
from krons.enforcement.rule import ValidationError
from krons.enforcement.validator import Validator


class MockRuleRegistry:
    """Simple rule registry for testing."""

    def __init__(self):
        self._type_rules: dict[type, object] = {}
        self._field_rules: dict[str, object] = {}

    def register_type(self, t: type, rule):
        self._type_rules[t] = rule

    def register_field(self, field_name: str, rule):
        self._field_rules[field_name] = rule

    def get_rule(self, *, base_type=None, field_name=None):
        # Field name takes priority
        if field_name and field_name in self._field_rules:
            return self._field_rules[field_name]
        # Then type
        if base_type and base_type in self._type_rules:
            return self._type_rules[base_type]
        return None

    def list_types(self):
        return list(self._type_rules.keys())


class TestValidatorInit:
    """Test Validator initialization."""

    def test_validator_with_custom_registry(self):
        """Validator should accept custom registry."""
        registry = MockRuleRegistry()
        validator = Validator(registry=registry)
        assert validator.registry is registry

    def test_validator_log_max_entries(self):
        """Validator should respect max_log_entries."""
        registry = MockRuleRegistry()
        validator = Validator(registry=registry, max_log_entries=10)
        assert validator.validation_log.maxlen == 10

    def test_validator_unlimited_log(self):
        """Validator with max_log_entries=0 should have unlimited log."""
        registry = MockRuleRegistry()
        validator = Validator(registry=registry, max_log_entries=0)
        assert validator.validation_log.maxlen is None


class TestValidatorLogging:
    """Test Validator logging functionality."""

    def test_log_validation_error(self):
        """Validator should log validation errors."""
        registry = MockRuleRegistry()
        validator = Validator(registry=registry)
        validator.log_validation_error("field", "value", "error message")

        assert len(validator.validation_log) == 1
        entry = validator.validation_log[0]
        assert entry["field"] == "field"
        assert entry["value"] == "value"
        assert entry["error"] == "error message"
        assert "timestamp" in entry

    def test_log_fifo_rotation(self):
        """Validator log should rotate with FIFO when full."""
        registry = MockRuleRegistry()
        validator = Validator(registry=registry, max_log_entries=3)

        validator.log_validation_error("field1", "v1", "e1")
        validator.log_validation_error("field2", "v2", "e2")
        validator.log_validation_error("field3", "v3", "e3")
        validator.log_validation_error("field4", "v4", "e4")  # Should push out first

        assert len(validator.validation_log) == 3
        fields = [e["field"] for e in validator.validation_log]
        assert "field1" not in fields
        assert "field4" in fields

    def test_get_validation_summary(self):
        """get_validation_summary should return summary dict."""
        registry = MockRuleRegistry()
        validator = Validator(registry=registry)
        validator.log_validation_error("field1", "v1", "e1")
        validator.log_validation_error("field2", "v2", "e2")
        validator.log_validation_error("field1", "v3", "e3")  # Duplicate field

        summary = validator.get_validation_summary()

        assert summary["total_errors"] == 3
        assert sorted(summary["fields_with_errors"]) == ["field1", "field2"]
        assert len(summary["error_entries"]) == 3

    def test_clear_log(self):
        """clear_log should empty the validation log."""
        registry = MockRuleRegistry()
        validator = Validator(registry=registry)
        validator.log_validation_error("field", "value", "error")
        assert len(validator.validation_log) == 1

        validator.clear_log()
        assert len(validator.validation_log) == 0


class MockSpec:
    """Mock Spec for testing."""

    def __init__(
        self,
        name: str,
        base_type: type = str,
        is_nullable: bool = False,
        is_listable: bool = False,
        metadata: dict | None = None,
    ):
        self._name = name
        self._base_type = base_type
        self._is_nullable = is_nullable
        self._is_listable = is_listable
        self._metadata = metadata or {}

    @property
    def name(self):
        return self._name

    @property
    def base_type(self):
        return self._base_type

    @property
    def is_nullable(self):
        return self._is_nullable

    @property
    def is_listable(self):
        return self._is_listable

    def get(self, key, default=None):
        return self._metadata.get(key, default)

    async def acreate_default_value(self):
        if "default" in self._metadata:
            return self._metadata["default"]
        raise ValueError("No default value")


class TestValidatorGetRuleForSpec:
    """Test Validator.get_rule_for_spec()."""

    def test_get_rule_by_base_type(self):
        """get_rule_for_spec should find rule by base_type."""
        registry = MockRuleRegistry()
        string_rule = StringRule()
        registry.register_type(str, string_rule)

        validator = Validator(registry=registry)
        spec = MockSpec("field", base_type=str)

        rule = validator.get_rule_for_spec(spec)
        assert rule is string_rule

    def test_get_rule_by_field_name(self):
        """get_rule_for_spec should find rule by field name."""
        registry = MockRuleRegistry()
        custom_rule = StringRule(min_length=5)
        registry.register_field("special_field", custom_rule)

        validator = Validator(registry=registry)
        spec = MockSpec("special_field", base_type=str)

        rule = validator.get_rule_for_spec(spec)
        assert rule is custom_rule

    def test_get_rule_metadata_override(self):
        """get_rule_for_spec should respect metadata rule override."""
        registry = MockRuleRegistry()
        default_rule = StringRule()
        override_rule = StringRule(max_length=10)
        registry.register_type(str, default_rule)

        validator = Validator(registry=registry)
        spec = MockSpec("field", base_type=str, metadata={"rule": override_rule})

        rule = validator.get_rule_for_spec(spec)
        assert rule is override_rule

    def test_get_rule_returns_none_if_not_found(self):
        """get_rule_for_spec should return None if no rule found."""
        registry = MockRuleRegistry()
        validator = Validator(registry=registry)
        spec = MockSpec("field", base_type=str)

        rule = validator.get_rule_for_spec(spec)
        assert rule is None


class TestValidatorValidateSpec:
    """Test Validator.validate_spec()."""

    @pytest.mark.anyio
    async def test_validate_spec_valid(self):
        """validate_spec should return valid value unchanged."""
        registry = MockRuleRegistry()
        registry.register_type(str, StringRule())

        validator = Validator(registry=registry)
        spec = MockSpec("field", base_type=str)

        result = await validator.validate_spec(spec, "hello")
        assert result == "hello"

    @pytest.mark.anyio
    async def test_validate_spec_with_autofix(self):
        """validate_spec should auto-fix invalid value."""
        registry = MockRuleRegistry()
        registry.register_type(str, StringRule())

        validator = Validator(registry=registry)
        spec = MockSpec("field", base_type=str)

        result = await validator.validate_spec(spec, 123, auto_fix=True)
        assert result == "123"

    @pytest.mark.anyio
    async def test_validate_spec_nullable_none(self):
        """validate_spec should return None for nullable spec with None value."""
        registry = MockRuleRegistry()
        validator = Validator(registry=registry)
        spec = MockSpec("field", base_type=str, is_nullable=True)

        result = await validator.validate_spec(spec, None)
        assert result is None

    @pytest.mark.anyio
    async def test_validate_spec_non_nullable_none_with_default(self):
        """validate_spec should use default for non-nullable None value."""
        registry = MockRuleRegistry()
        registry.register_type(str, StringRule())

        validator = Validator(registry=registry)
        spec = MockSpec("field", base_type=str, metadata={"default": "default_value"})

        result = await validator.validate_spec(spec, None)
        assert result == "default_value"

    @pytest.mark.anyio
    async def test_validate_spec_non_nullable_none_raises(self):
        """validate_spec should raise for non-nullable None without default."""
        registry = MockRuleRegistry()
        validator = Validator(registry=registry)
        spec = MockSpec("field", base_type=str)

        with pytest.raises(ValidationError, match="not nullable"):
            await validator.validate_spec(spec, None, strict=True)

    @pytest.mark.anyio
    async def test_validate_spec_listable(self):
        """validate_spec should validate each item in listable spec."""
        registry = MockRuleRegistry()
        registry.register_type(int, NumberRule(ge=0))

        validator = Validator(registry=registry)
        spec = MockSpec("numbers", base_type=int, is_listable=True)

        result = await validator.validate_spec(spec, [1, 2, 3])
        assert result == [1, 2, 3]

    @pytest.mark.anyio
    async def test_validate_spec_listable_wrap_single(self):
        """validate_spec should wrap single value in list for listable spec."""
        registry = MockRuleRegistry()
        registry.register_type(int, NumberRule())

        validator = Validator(registry=registry)
        spec = MockSpec("numbers", base_type=int, is_listable=True)

        result = await validator.validate_spec(spec, 42, auto_fix=True)
        assert result == [42]

    @pytest.mark.anyio
    async def test_validate_spec_logs_errors(self):
        """validate_spec should log validation errors."""
        registry = MockRuleRegistry()
        registry.register_type(str, StringRule(min_length=5))

        validator = Validator(registry=registry)
        spec = MockSpec("field", base_type=str)

        with pytest.raises(ValidationError):
            await validator.validate_spec(spec, "hi", auto_fix=False)

        assert len(validator.validation_log) == 1

    @pytest.mark.anyio
    async def test_validate_spec_no_rule_strict_raises(self):
        """validate_spec with strict=True should raise if no rule found."""
        registry = MockRuleRegistry()
        validator = Validator(registry=registry)
        spec = MockSpec("field", base_type=str)

        with pytest.raises(ValidationError, match="No rule found"):
            await validator.validate_spec(spec, "value", strict=True)

    @pytest.mark.anyio
    async def test_validate_spec_no_rule_non_strict_passes(self):
        """validate_spec with strict=False should pass if no rule found."""
        registry = MockRuleRegistry()
        validator = Validator(registry=registry)
        spec = MockSpec("field", base_type=str)

        result = await validator.validate_spec(spec, "value", strict=False)
        assert result == "value"


class TestValidatorCustomValidators:
    """Test Validator with custom validators in spec metadata."""

    @pytest.mark.anyio
    async def test_custom_sync_validator(self):
        """validate_spec should apply sync custom validators."""
        registry = MockRuleRegistry()
        registry.register_type(str, StringRule())

        def uppercase_validator(v):
            return v.upper()

        validator = Validator(registry=registry)
        spec = MockSpec("field", base_type=str, metadata={"validator": uppercase_validator})

        result = await validator.validate_spec(spec, "hello")
        assert result == "HELLO"

    @pytest.mark.anyio
    async def test_custom_async_validator(self):
        """validate_spec should apply async custom validators."""
        registry = MockRuleRegistry()
        registry.register_type(str, StringRule())

        async def async_validator(v):
            return f"validated_{v}"

        validator = Validator(registry=registry)
        spec = MockSpec("field", base_type=str, metadata={"validator": async_validator})

        result = await validator.validate_spec(spec, "hello")
        assert result == "validated_hello"

    @pytest.mark.anyio
    async def test_custom_validator_list(self):
        """validate_spec should apply list of custom validators in order."""
        registry = MockRuleRegistry()
        registry.register_type(str, StringRule())

        def validator1(v):
            return v.strip()

        def validator2(v):
            return v.upper()

        validator = Validator(registry=registry)
        spec = MockSpec("field", base_type=str, metadata={"validator": [validator1, validator2]})

        result = await validator.validate_spec(spec, "  hello  ")
        assert result == "HELLO"

    @pytest.mark.anyio
    async def test_custom_validator_failure(self):
        """validate_spec should raise if custom validator fails."""
        registry = MockRuleRegistry()
        registry.register_type(str, StringRule())

        def failing_validator(v):
            raise ValueError("Custom validation failed")

        validator = Validator(registry=registry)
        spec = MockSpec("field", base_type=str, metadata={"validator": failing_validator})

        with pytest.raises(ValidationError, match="Custom validator failed"):
            await validator.validate_spec(spec, "hello")


class TestValidatorRepr:
    """Test Validator.__repr__."""

    def test_repr_output(self):
        """__repr__ should identify the Validator class."""
        registry = MockRuleRegistry()
        registry.register_type(str, StringRule())
        registry.register_type(int, NumberRule())

        validator = Validator(registry=registry)
        repr_str = repr(validator)

        assert "Validator" in repr_str
