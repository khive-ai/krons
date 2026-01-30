# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Tests for kron.utils._sql_validation - SQL injection prevention."""

import pytest

from krons.errors import ValidationError
from krons.utils.sql._sql_validation import (
    MAX_IDENTIFIER_LENGTH,
    SAFE_IDENTIFIER_PATTERN,
    sanitize_order_by,
    validate_identifier,
)


class TestValidateIdentifier:
    """Test validate_identifier function."""

    def test_valid_identifiers(self):
        """Valid identifiers should pass through unchanged."""
        assert validate_identifier("user_name", "column") == "user_name"
        assert validate_identifier("ID", "column") == "ID"
        assert validate_identifier("_private", "column") == "_private"
        assert validate_identifier("CamelCase", "column") == "CamelCase"
        assert validate_identifier("a123", "column") == "a123"

    def test_empty_identifier_raises(self):
        """Empty identifier should raise ValidationError."""
        with pytest.raises(ValidationError, match="Empty column name"):
            validate_identifier("", "column")

    def test_too_long_identifier_raises(self):
        """Identifier exceeding 63 chars should raise ValidationError."""
        long_name = "a" * 64
        with pytest.raises(ValidationError, match="too long"):
            validate_identifier(long_name, "table")

    def test_max_length_identifier_passes(self):
        """Identifier at exactly 63 chars should pass."""
        max_name = "a" * MAX_IDENTIFIER_LENGTH
        assert validate_identifier(max_name, "table") == max_name

    def test_unsafe_identifiers_raise(self):
        """Identifiers with unsafe characters should raise ValidationError."""
        with pytest.raises(ValidationError, match="Unsafe"):
            validate_identifier("user-name", "column")  # hyphen
        with pytest.raises(ValidationError, match="Unsafe"):
            validate_identifier("user.name", "column")  # dot
        with pytest.raises(ValidationError, match="Unsafe"):
            validate_identifier("user name", "column")  # space
        with pytest.raises(ValidationError, match="Unsafe"):
            validate_identifier("123abc", "column")  # starts with digit
        with pytest.raises(ValidationError, match="Unsafe"):
            validate_identifier("user;drop", "column")  # semicolon (injection)
        with pytest.raises(ValidationError, match="Unsafe"):
            validate_identifier("user'--", "column")  # quote (injection)

    def test_kind_in_error_message(self):
        """Error message should include the kind parameter."""
        with pytest.raises(ValidationError, match="table"):
            validate_identifier("bad;name", "table")
        with pytest.raises(ValidationError, match="schema"):
            validate_identifier("bad;name", "schema")


class TestSanitizeOrderBy:
    """Test sanitize_order_by function."""

    def test_single_column(self):
        """Single column should get quoted with default ASC."""
        assert sanitize_order_by("name") == '"name" ASC'

    def test_column_with_direction(self):
        """Column with direction should be preserved."""
        assert sanitize_order_by("name ASC") == '"name" ASC'
        assert sanitize_order_by("name DESC") == '"name" DESC'
        assert sanitize_order_by("name asc") == '"name" ASC'
        assert sanitize_order_by("name desc") == '"name" DESC'

    def test_multiple_columns(self):
        """Multiple columns should be quoted and joined."""
        result = sanitize_order_by("name, created_at DESC")
        assert result == '"name" ASC, "created_at" DESC'

    def test_invalid_column_raises(self):
        """Invalid column names should raise ValidationError."""
        with pytest.raises(ValidationError, match="Invalid column"):
            sanitize_order_by("user;drop")

    def test_invalid_direction_raises(self):
        """Invalid direction should raise ValidationError."""
        with pytest.raises(ValidationError, match="Invalid direction"):
            sanitize_order_by("name ASCENDING")
        with pytest.raises(ValidationError, match="Invalid direction"):
            sanitize_order_by("name UP")

    def test_empty_clause_raises(self):
        """Empty ORDER BY clause should raise ValidationError."""
        with pytest.raises(ValidationError, match="Empty ORDER BY"):
            sanitize_order_by("")
        with pytest.raises(ValidationError, match="Empty ORDER BY"):
            sanitize_order_by("   ")

    def test_invalid_format_raises(self):
        """Invalid ORDER BY format should raise ValidationError."""
        with pytest.raises(ValidationError, match="Invalid ORDER BY clause"):
            sanitize_order_by("name ASC DESC")  # Too many tokens


class TestRegexPattern:
    """Test SAFE_IDENTIFIER_PATTERN regex."""

    def test_pattern_matches_valid(self):
        """Pattern should match valid SQL identifiers."""
        assert SAFE_IDENTIFIER_PATTERN.match("abc")
        assert SAFE_IDENTIFIER_PATTERN.match("ABC")
        assert SAFE_IDENTIFIER_PATTERN.match("_abc")
        assert SAFE_IDENTIFIER_PATTERN.match("abc123")
        assert SAFE_IDENTIFIER_PATTERN.match("a_b_c")

    def test_pattern_rejects_invalid(self):
        """Pattern should not match invalid identifiers."""
        assert not SAFE_IDENTIFIER_PATTERN.match("123abc")  # starts with digit
        assert not SAFE_IDENTIFIER_PATTERN.match("abc-def")  # hyphen
        assert not SAFE_IDENTIFIER_PATTERN.match("abc.def")  # dot
        assert not SAFE_IDENTIFIER_PATTERN.match("abc def")  # space
        assert not SAFE_IDENTIFIER_PATTERN.match("")  # empty
