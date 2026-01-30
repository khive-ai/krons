# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import pytest

from krons.utils.fuzzy._fuzzy_json import fuzzy_json

# ============================================================================
# Test fuzzy_json main function
# ============================================================================


def test_fuzzy_json_valid():
    """Test fuzzy_json with valid JSON"""
    result = fuzzy_json('{"key": "value"}')
    assert result == {"key": "value"}


def test_fuzzy_json_single_quotes():
    """Test fuzzy_json with single quotes"""
    result = fuzzy_json("{'key': 'value'}")
    assert result == {"key": "value"}


def test_fuzzy_json_unquoted_keys():
    """Test fuzzy_json with unquoted keys"""
    result = fuzzy_json("{key: 'value'}")
    assert result == {"key": "value"}


def test_fuzzy_json_trailing_commas():
    """Test fuzzy_json with trailing commas"""
    result = fuzzy_json('{"key": "value",}')
    assert result == {"key": "value"}


def test_fuzzy_json_missing_closing_bracket():
    """Test fuzzy_json with missing closing bracket"""
    result = fuzzy_json('{"key": "value"')
    assert result == {"key": "value"}


def test_fuzzy_json_invalid():
    """Test fuzzy_json with completely invalid JSON"""
    with pytest.raises(ValueError, match="Invalid JSON string"):
        fuzzy_json("{completely broken")


def test_fuzzy_json_not_string():
    """Test fuzzy_json with non-string input"""
    with pytest.raises(TypeError, match="Input must be a string"):
        fuzzy_json(123)


def test_fuzzy_json_empty():
    """Test fuzzy_json with empty string"""
    with pytest.raises(ValueError, match="Input string is empty"):
        fuzzy_json("")


def test_fuzzy_json_whitespace_only():
    """Test fuzzy_json with whitespace only"""
    with pytest.raises(ValueError, match="Input string is empty"):
        fuzzy_json("   ")


# ============================================================================
# Test with escape sequences
# ============================================================================


def test_fuzzy_json_escaped_backslash():
    """Test fuzzy_json with escaped backslash"""
    json_str = r'{"path": "C:\\Users\\file.txt"}'
    result = fuzzy_json(json_str)
    assert result == {"path": "C:\\Users\\file.txt"}


def test_fuzzy_json_escaped_quote():
    """Test fuzzy_json with escaped quote"""
    json_str = r'{"text": "He said \"hello\""}'
    result = fuzzy_json(json_str)
    assert result == {"text": 'He said "hello"'}


def test_fuzzy_json_with_escapes_comprehensive():
    """Comprehensive test for fuzzy_json with various escape scenarios"""
    json_str = r'{"file": "C:\\Users\\test.txt", "quote": "He said \"hello\""}'
    result = fuzzy_json(json_str)
    assert result["file"] == "C:\\Users\\test.txt"
    assert result["quote"] == 'He said "hello"'


# ============================================================================
# Type Validation Tests
# ============================================================================


def test_fuzzy_json_rejects_primitive_int():
    """Test fuzzy_json rejects primitive int"""
    with pytest.raises(TypeError, match="got primitive type: int"):
        fuzzy_json("42")


def test_fuzzy_json_rejects_primitive_string():
    """Test fuzzy_json rejects primitive string"""
    with pytest.raises(TypeError, match="got primitive type: str"):
        fuzzy_json('"hello"')


def test_fuzzy_json_rejects_primitive_bool():
    """Test fuzzy_json rejects primitive bool"""
    with pytest.raises(TypeError, match="got primitive type: bool"):
        fuzzy_json("true")


def test_fuzzy_json_rejects_primitive_null():
    """Test fuzzy_json rejects primitive null"""
    with pytest.raises(TypeError, match="got primitive type: NoneType"):
        fuzzy_json("null")


def test_fuzzy_json_rejects_primitive_float():
    """Test fuzzy_json rejects primitive float"""
    with pytest.raises(TypeError, match="got primitive type: float"):
        fuzzy_json("3.14")


def test_fuzzy_json_rejects_list_of_primitives():
    """Test fuzzy_json rejects list of primitive values"""
    with pytest.raises(TypeError, match="list with non-dict element at index 0"):
        fuzzy_json("[1, 2, 3]")


def test_fuzzy_json_rejects_list_of_strings():
    """Test fuzzy_json rejects list of strings"""
    with pytest.raises(TypeError, match="list with non-dict element at index 0: str"):
        fuzzy_json('["a", "b", "c"]')


def test_fuzzy_json_rejects_mixed_list():
    """Test fuzzy_json rejects list with mix of dicts and primitives"""
    with pytest.raises(TypeError, match="list with non-dict element at index 1: int"):
        fuzzy_json('[{"key": "value"}, 42, {"other": "data"}]')


def test_fuzzy_json_accepts_empty_list():
    """Test fuzzy_json accepts empty list (vacuously list[dict])"""
    result = fuzzy_json("[]")
    assert result == []


def test_fuzzy_json_accepts_list_of_dicts():
    """Test fuzzy_json accepts list of dicts"""
    result = fuzzy_json('[{"a": 1}, {"b": 2}]')
    assert result == [{"a": 1}, {"b": 2}]


def test_fuzzy_json_accepts_dict():
    """Test fuzzy_json accepts dict"""
    result = fuzzy_json('{"key": "value"}')
    assert result == {"key": "value"}


# ============================================================================
# Security tests - input size limits
# ============================================================================


class TestSecurityLimits:
    """Test security-related input size limits."""

    def test_fuzzy_json_size_limit(self):
        """Test that fuzzy_json rejects inputs exceeding max_size."""
        # Use a small max_size for testing
        large_input = '{"key": "' + "x" * 1000 + '"}'

        with pytest.raises(ValueError, match="exceeds maximum"):
            fuzzy_json(large_input, max_size=100)

    def test_fuzzy_json_within_limit(self):
        """Test that fuzzy_json works for inputs within limit."""
        json_str = '{"key": "value"}'
        result = fuzzy_json(json_str, max_size=1000)
        assert result == {"key": "value"}


# ============================================================================
# String content preservation tests (state machine correctness)
# ============================================================================


class TestStringContentPreservation:
    """Test that fuzzy_json preserves content inside strings.

    The old regex-based approach would corrupt apostrophes and quotes
    inside string values. The state-machine approach should preserve them.
    """

    def test_apostrophe_in_double_quoted_string(self):
        """Test apostrophe inside double-quoted string is preserved."""
        # Valid JSON with apostrophe - should work directly
        result = fuzzy_json('{"text": "it\'s fine"}')
        assert result == {"text": "it's fine"}

    def test_apostrophe_in_single_quoted_string(self):
        """Test apostrophe inside single-quoted string is preserved.

        This was the bug: {'key': "it's fine"} would become {"key": "it"s fine"}
        because the blind replace("'", '"') corrupted the apostrophe.
        """
        # Single-quoted key, double-quoted value with apostrophe
        result = fuzzy_json("{'text': \"it's fine\"}")
        assert result == {"text": "it's fine"}

    def test_double_quote_inside_single_quoted_string(self):
        """Test double quote inside single-quoted string is properly escaped."""
        # Single-quoted string containing a double quote
        result = fuzzy_json("{'text': 'say \"hello\"'}")
        assert result == {"text": 'say "hello"'}

    def test_mixed_quotes_complex(self):
        """Test complex case with mixed quotes."""
        # Single-quoted key and value, value contains apostrophe
        result = fuzzy_json("{'message': 'don\\'t panic'}")
        assert result == {"message": "don't panic"}

    def test_nested_with_apostrophes(self):
        """Test nested structure with apostrophes."""
        result = fuzzy_json("{'outer': {'inner': \"it's nested\", 'also': \"won't break\"}}")
        assert result == {"outer": {"inner": "it's nested", "also": "won't break"}}

    def test_array_with_apostrophes(self):
        """Test array with strings containing apostrophes."""
        result = fuzzy_json('{\'items\': ["it\'s", "that\'s", "what\'s"]}')
        assert result == {"items": ["it's", "that's", "what's"]}

    def test_unquoted_key_with_quoted_value_apostrophe(self):
        """Test unquoted key with value containing apostrophe."""
        result = fuzzy_json('{text: "it\'s fine"}')
        assert result == {"text": "it's fine"}

    def test_trailing_comma_with_apostrophe(self):
        """Test trailing comma removal doesn't affect string content."""
        result = fuzzy_json('{"text": "it\'s fine",}')
        assert result == {"text": "it's fine"}

    def test_escape_sequences_preserved(self):
        """Test that escape sequences in strings are preserved."""
        result = fuzzy_json(r'{"path": "C:\\Users\\file.txt"}')
        assert result == {"path": "C:\\Users\\file.txt"}

    def test_newlines_in_strings_preserved(self):
        """Test that newlines in strings are preserved."""
        result = fuzzy_json('{"text": "line1\\nline2"}')
        assert result == {"text": "line1\nline2"}
