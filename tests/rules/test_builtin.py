# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.rules built-in rules - String, Number, Boolean, Choice, Mapping."""

import pytest

from krons.enforcement.common import (
    BooleanRule,
    ChoiceRule,
    MappingRule,
    NumberRule,
    StringRule,
)


class TestStringRule:
    """Test StringRule validation."""

    @pytest.mark.anyio
    async def test_string_valid(self):
        """StringRule should accept valid string."""
        rule = StringRule()
        await rule.validate("hello", str)  # Should not raise

    @pytest.mark.anyio
    async def test_string_invalid_type(self):
        """StringRule should reject non-string."""
        rule = StringRule()
        with pytest.raises(ValueError, match="expected str"):
            await rule.validate(123, str)

    @pytest.mark.anyio
    async def test_string_min_length_pass(self):
        """StringRule should accept string above min_length."""
        rule = StringRule(min_length=3)
        await rule.validate("hello", str)

    @pytest.mark.anyio
    async def test_string_min_length_fail(self):
        """StringRule should reject string below min_length."""
        rule = StringRule(min_length=5)
        with pytest.raises(ValueError, match="too short"):
            await rule.validate("hi", str)

    @pytest.mark.anyio
    async def test_string_max_length_pass(self):
        """StringRule should accept string below max_length."""
        rule = StringRule(max_length=10)
        await rule.validate("hello", str)

    @pytest.mark.anyio
    async def test_string_max_length_fail(self):
        """StringRule should reject string above max_length."""
        rule = StringRule(max_length=3)
        with pytest.raises(ValueError, match="too long"):
            await rule.validate("hello", str)

    @pytest.mark.anyio
    async def test_string_pattern_pass(self):
        """StringRule should accept string matching pattern."""
        rule = StringRule(pattern=r"^[A-Za-z]+$")
        await rule.validate("Hello", str)

    @pytest.mark.anyio
    async def test_string_pattern_fail(self):
        """StringRule should reject string not matching pattern."""
        rule = StringRule(pattern=r"^[A-Za-z]+$")
        with pytest.raises(ValueError, match="does not match"):
            await rule.validate("Hello123", str)

    def test_string_redos_pattern_raises(self):
        """StringRule should reject ReDoS-vulnerable patterns."""
        with pytest.raises(ValueError, match="ReDoS"):
            StringRule(pattern=r"(.*)*")

    @pytest.mark.anyio
    async def test_string_pattern_max_input_length(self):
        """StringRule should reject inputs exceeding regex_max_input_length."""
        rule = StringRule(pattern=r".*", regex_max_input_length=10)
        with pytest.raises(ValueError, match="too long for regex"):
            await rule.validate("a" * 20, str)

    @pytest.mark.anyio
    async def test_string_autofix_converts(self):
        """StringRule.perform_fix should convert to string."""
        rule = StringRule()
        result = await rule.perform_fix(123, str)
        assert result == "123"

    @pytest.mark.anyio
    async def test_string_invoke_with_autofix(self):
        """StringRule.invoke should auto-convert non-strings."""
        rule = StringRule()
        result = await rule.invoke("field", 42, str, auto_fix=True)
        assert result == "42"


class TestNumberRule:
    """Test NumberRule validation."""

    @pytest.mark.anyio
    async def test_number_valid_int(self):
        """NumberRule should accept valid int."""
        rule = NumberRule()
        await rule.validate(42, int)

    @pytest.mark.anyio
    async def test_number_valid_float(self):
        """NumberRule should accept valid float."""
        rule = NumberRule()
        await rule.validate(3.14, float)

    @pytest.mark.anyio
    async def test_number_invalid_type(self):
        """NumberRule should reject non-number."""
        rule = NumberRule()
        with pytest.raises(ValueError, match="expected int or float"):
            await rule.validate("42", str)

    @pytest.mark.anyio
    async def test_number_ge_pass(self):
        """NumberRule should accept value >= ge."""
        rule = NumberRule(ge=0)
        await rule.validate(0, int)
        await rule.validate(5, int)

    @pytest.mark.anyio
    async def test_number_ge_fail(self):
        """NumberRule should reject value < ge."""
        rule = NumberRule(ge=0)
        with pytest.raises(ValueError, match="too small"):
            await rule.validate(-1, int)

    @pytest.mark.anyio
    async def test_number_gt_pass(self):
        """NumberRule should accept value > gt."""
        rule = NumberRule(gt=0)
        await rule.validate(1, int)

    @pytest.mark.anyio
    async def test_number_gt_fail(self):
        """NumberRule should reject value <= gt."""
        rule = NumberRule(gt=0)
        with pytest.raises(ValueError, match="too small"):
            await rule.validate(0, int)

    @pytest.mark.anyio
    async def test_number_le_pass(self):
        """NumberRule should accept value <= le."""
        rule = NumberRule(le=10)
        await rule.validate(10, int)
        await rule.validate(5, int)

    @pytest.mark.anyio
    async def test_number_le_fail(self):
        """NumberRule should reject value > le."""
        rule = NumberRule(le=10)
        with pytest.raises(ValueError, match="too large"):
            await rule.validate(15, int)

    @pytest.mark.anyio
    async def test_number_lt_pass(self):
        """NumberRule should accept value < lt."""
        rule = NumberRule(lt=10)
        await rule.validate(9, int)

    @pytest.mark.anyio
    async def test_number_lt_fail(self):
        """NumberRule should reject value >= lt."""
        rule = NumberRule(lt=10)
        with pytest.raises(ValueError, match="too large"):
            await rule.validate(10, int)

    @pytest.mark.anyio
    async def test_number_range(self):
        """NumberRule should enforce combined range constraints."""
        rule = NumberRule(ge=0, le=1)  # Confidence score
        await rule.validate(0.5, float)
        with pytest.raises(ValueError):
            await rule.validate(-0.1, float)
        with pytest.raises(ValueError):
            await rule.validate(1.1, float)

    @pytest.mark.anyio
    async def test_number_autofix_from_string(self):
        """NumberRule.perform_fix should convert string to number."""
        rule = NumberRule()
        result = await rule.perform_fix("42", int)
        assert result == 42
        assert isinstance(result, int)

        result = await rule.perform_fix("3.14", float)
        assert result == 3.14
        assert isinstance(result, float)

    @pytest.mark.anyio
    async def test_number_invoke_with_autofix(self):
        """NumberRule.invoke should auto-convert strings."""
        rule = NumberRule()
        result = await rule.invoke("field", "42", int, auto_fix=True)
        assert result == 42


class TestBooleanRule:
    """Test BooleanRule validation."""

    @pytest.mark.anyio
    async def test_boolean_valid(self):
        """BooleanRule should accept valid boolean."""
        rule = BooleanRule()
        await rule.validate(True, bool)
        await rule.validate(False, bool)

    @pytest.mark.anyio
    async def test_boolean_invalid_type(self):
        """BooleanRule should reject non-boolean."""
        rule = BooleanRule()
        with pytest.raises(ValueError, match="expected bool"):
            await rule.validate("true", bool)

    @pytest.mark.anyio
    async def test_boolean_fix_string_true(self):
        """BooleanRule.perform_fix should convert truthy strings."""
        rule = BooleanRule()
        assert await rule.perform_fix("true", bool) is True
        assert await rule.perform_fix("True", bool) is True
        assert await rule.perform_fix("TRUE", bool) is True
        assert await rule.perform_fix("yes", bool) is True
        assert await rule.perform_fix("YES", bool) is True
        assert await rule.perform_fix("1", bool) is True
        assert await rule.perform_fix("on", bool) is True

    @pytest.mark.anyio
    async def test_boolean_fix_string_false(self):
        """BooleanRule.perform_fix should convert falsy strings."""
        rule = BooleanRule()
        assert await rule.perform_fix("false", bool) is False
        assert await rule.perform_fix("False", bool) is False
        assert await rule.perform_fix("FALSE", bool) is False
        assert await rule.perform_fix("no", bool) is False
        assert await rule.perform_fix("NO", bool) is False
        assert await rule.perform_fix("0", bool) is False
        assert await rule.perform_fix("off", bool) is False

    @pytest.mark.anyio
    async def test_boolean_fix_number(self):
        """BooleanRule.perform_fix should convert numbers."""
        rule = BooleanRule()
        assert await rule.perform_fix(1, bool) is True
        assert await rule.perform_fix(0, bool) is False
        assert await rule.perform_fix(42, bool) is True

    @pytest.mark.anyio
    async def test_boolean_fix_invalid_string_raises(self):
        """BooleanRule.perform_fix should raise for invalid strings."""
        rule = BooleanRule()
        with pytest.raises(ValueError, match="Failed to convert"):
            await rule.perform_fix("maybe", bool)

    @pytest.mark.anyio
    async def test_boolean_invoke_with_autofix(self):
        """BooleanRule.invoke should auto-convert strings."""
        rule = BooleanRule()
        result = await rule.invoke("field", "true", bool, auto_fix=True)
        assert result is True


class TestChoiceRule:
    """Test ChoiceRule validation."""

    @pytest.mark.anyio
    async def test_choice_valid(self):
        """ChoiceRule should accept valid choice."""
        rule = ChoiceRule(choices=["low", "medium", "high"])
        await rule.validate("low", str)
        await rule.validate("medium", str)
        await rule.validate("high", str)

    @pytest.mark.anyio
    async def test_choice_invalid(self):
        """ChoiceRule should reject invalid choice."""
        rule = ChoiceRule(choices=["low", "medium", "high"])
        with pytest.raises(ValueError, match="Invalid choice"):
            await rule.validate("critical", str)

    @pytest.mark.anyio
    async def test_choice_case_sensitive(self):
        """ChoiceRule should respect case_sensitive=True."""
        rule = ChoiceRule(choices=["low", "medium", "high"], case_sensitive=True)
        await rule.validate("low", str)
        with pytest.raises(ValueError):
            await rule.validate("LOW", str)

    @pytest.mark.anyio
    async def test_choice_case_insensitive_fix(self):
        """ChoiceRule with case_sensitive=False should fix case."""
        rule = ChoiceRule(choices=["low", "medium", "high"], case_sensitive=False)
        result = await rule.perform_fix("LOW", str)
        assert result == "low"

    @pytest.mark.anyio
    async def test_choice_with_set(self):
        """ChoiceRule should accept set of choices."""
        rule = ChoiceRule(choices={"a", "b", "c"})
        await rule.validate("a", str)

    @pytest.mark.anyio
    async def test_choice_with_apply_fields(self):
        """ChoiceRule should respect apply_fields."""
        rule = ChoiceRule(choices=["low", "high"], apply_fields={"priority"})
        assert await rule.apply("priority", "low", str) is True
        # Other fields may not match via FIELD qualifier but might via fallback
        # depending on apply_types

    @pytest.mark.anyio
    async def test_choice_invoke_with_autofix(self):
        """ChoiceRule.invoke should auto-fix case."""
        rule = ChoiceRule(choices=["low", "medium", "high"], case_sensitive=False)
        result = await rule.invoke("field", "MEDIUM", str, auto_fix=True)
        assert result == "medium"


class TestMappingRule:
    """Test MappingRule validation."""

    @pytest.mark.anyio
    async def test_mapping_valid(self):
        """MappingRule should accept valid dict."""
        rule = MappingRule()
        await rule.validate({"key": "value"}, dict)

    @pytest.mark.anyio
    async def test_mapping_invalid_type(self):
        """MappingRule should reject non-mapping."""
        rule = MappingRule()
        with pytest.raises(ValueError, match="expected dict/Mapping"):
            await rule.validate("not a dict", dict)

    @pytest.mark.anyio
    async def test_mapping_required_keys_pass(self):
        """MappingRule should accept dict with required keys."""
        rule = MappingRule(required_keys={"name", "value"})
        await rule.validate({"name": "test", "value": 42}, dict)

    @pytest.mark.anyio
    async def test_mapping_required_keys_fail(self):
        """MappingRule should reject dict missing required keys."""
        rule = MappingRule(required_keys={"name", "value"})
        with pytest.raises(ValueError, match="Missing required keys"):
            await rule.validate({"name": "test"}, dict)

    @pytest.mark.anyio
    async def test_mapping_fix_from_json(self):
        """MappingRule.perform_fix should parse JSON string."""
        rule = MappingRule()
        result = await rule.perform_fix('{"key": "value"}', dict)
        assert result == {"key": "value"}

    @pytest.mark.anyio
    async def test_mapping_fix_invalid_json_raises(self):
        """MappingRule.perform_fix should raise for invalid JSON."""
        rule = MappingRule()
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            await rule.perform_fix("not json", dict)

    @pytest.mark.anyio
    async def test_mapping_fuzzy_keys(self):
        """MappingRule with fuzzy_keys should normalize key case."""
        rule = MappingRule(required_keys={"name"}, fuzzy_keys=True)
        result = await rule.perform_fix({"NAME": "test"}, dict)
        assert "name" in result
        assert result["name"] == "test"

    @pytest.mark.anyio
    async def test_mapping_invoke_json_with_autofix(self):
        """MappingRule.invoke should parse JSON string."""
        rule = MappingRule(required_keys={"name"})
        result = await rule.invoke("field", '{"name": "test"}', dict, auto_fix=True)
        assert result == {"name": "test"}


class TestRuleDefaultApplyTypes:
    """Test that built-in rules have correct default apply_types."""

    def test_string_rule_applies_to_str(self):
        """StringRule should apply to str by default."""
        rule = StringRule()
        assert str in rule.apply_types

    def test_number_rule_applies_to_int_float(self):
        """NumberRule should apply to int and float by default."""
        rule = NumberRule()
        assert int in rule.apply_types
        assert float in rule.apply_types

    def test_boolean_rule_applies_to_bool(self):
        """BooleanRule should apply to bool by default."""
        rule = BooleanRule()
        assert bool in rule.apply_types

    def test_mapping_rule_applies_to_dict(self):
        """MappingRule should apply to dict by default."""
        rule = MappingRule()
        assert dict in rule.apply_types


class TestRuleAutoFixDefaults:
    """Test that built-in rules have correct default auto_fix settings."""

    def test_string_rule_autofix_default(self):
        """StringRule should have auto_fix=True by default."""
        rule = StringRule()
        assert rule.auto_fix is True

    def test_number_rule_autofix_default(self):
        """NumberRule should have auto_fix=True by default."""
        rule = NumberRule()
        assert rule.auto_fix is True

    def test_boolean_rule_autofix_default(self):
        """BooleanRule should have auto_fix=True by default."""
        rule = BooleanRule()
        assert rule.auto_fix is True

    def test_mapping_rule_autofix_default(self):
        """MappingRule should have auto_fix=True by default."""
        rule = MappingRule()
        assert rule.auto_fix is True
