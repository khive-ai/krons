# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

"""Error hierarchy tests for kron.

Note: Error paths for individual classes (Pile, Progression, Flow, Graph)
are tested in their respective test files. This file tests the error
hierarchy itself - base class behavior, retryable semantics, serialization.
"""

import pytest

from krons.core import Element
from krons.errors import (
    ConfigurationError,
    ExecutionError,
    ExistsError,
    KronConnectionError,
    KronError,
    KronTimeoutError,
    NotFoundError,
    QueueFullError,
    ValidationError,
)

# =============================================================================
# KronError Base Class (unique - tests error class behavior)
# =============================================================================


class TestKronErrorBase:
    """Tests for KronError base class behavior."""

    def test_default_message(self):
        """Default message is used when none provided."""
        err = KronError()
        assert err.message == "kron error"

    def test_custom_message(self):
        """Custom message overrides default."""
        err = KronError("custom error message")
        assert err.message == "custom error message"

    def test_details_dict(self):
        """Details dict is stored and accessible."""
        details = {"key": "value", "count": 42}
        err = KronError("test", details=details)
        assert err.details == details

    def test_retryable_default(self):
        """Default retryable flag from class attribute."""
        err = KronError()
        assert err.retryable is True

    def test_cause_chaining(self):
        """Cause exception is preserved for traceback."""
        original = ValueError("original error")
        err = KronError("wrapped error", cause=original)
        assert err.__cause__ is original

    def test_to_dict_serialization(self):
        """Error serializes to dict with all fields."""
        err = KronError("test message", details={"key": "value"}, retryable=False)
        data = err.to_dict()
        assert data["error"] == "KronError"
        assert data["message"] == "test message"
        assert data["retryable"] is False
        assert data["details"] == {"key": "value"}


# =============================================================================
# Specialized Errors (unique - tests error hierarchy properties)
# =============================================================================


class TestSpecializedErrorsRetryable:
    """Tests for specialized error subclass retryable defaults."""

    def test_validation_error_not_retryable(self):
        """ValidationError is not retryable by default."""
        err = ValidationError("validation failed")
        assert err.retryable is False

    def test_not_found_error_not_retryable(self):
        """NotFoundError is not retryable by default."""
        err = NotFoundError("item not found")
        assert err.retryable is False

    def test_exists_error_not_retryable(self):
        """ExistsError is not retryable by default."""
        err = ExistsError("item exists")
        assert err.retryable is False

    def test_timeout_error_retryable(self):
        """KronTimeoutError is retryable by default."""
        err = KronTimeoutError("operation timed out")
        assert err.retryable is True

    def test_connection_error_retryable(self):
        """KronConnectionError is retryable by default."""
        err = KronConnectionError("connection lost")
        assert err.retryable is True

    def test_inheritance_hierarchy(self):
        """All specialized errors inherit from KronError."""
        errors = [
            ValidationError(),
            ConfigurationError(),
            ExecutionError(),
            KronConnectionError(),
            KronTimeoutError(),
            NotFoundError(),
            ExistsError(),
            QueueFullError(),
        ]
        for err in errors:
            assert isinstance(err, KronError)


# =============================================================================
# Retryable Consistency (unique - semantic consistency check)
# =============================================================================


class TestRetryableConsistency:
    """Tests for retryable flag consistency across error types."""

    def test_transient_errors_are_retryable(self):
        """Transient errors are retryable."""
        assert KronConnectionError().retryable is True
        assert KronTimeoutError().retryable is True
        assert QueueFullError().retryable is True

    def test_permanent_errors_are_not_retryable(self):
        """Permanent errors are not retryable."""
        assert ValidationError().retryable is False
        assert ConfigurationError().retryable is False
        assert NotFoundError().retryable is False
        assert ExistsError().retryable is False


# =============================================================================
# Element Error Paths (unique - to_dict invalid combinations)
# =============================================================================


class TestElementErrorPaths:
    """Tests for errors in Element operations."""

    def test_to_dict_json_mode_rejects_datetime_format(self):
        """Element.to_dict() with json mode rejects datetime format."""
        elem = Element()
        with pytest.raises(ValueError, match="created_at_format='datetime' not valid"):
            elem.to_dict(mode="json", created_at_format="datetime")
