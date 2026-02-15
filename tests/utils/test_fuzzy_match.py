# Copyright (c) 2025 - 2026, HaiyangLi <quantocean.li at gmail dot com>
# SPDX-License-Identifier: Apache-2.0

import pytest

from krons.utils.fuzzy._fuzzy_match import HandleUnmatched, fuzzy_match_keys
from krons.utils.fuzzy._string_similarity import SimilarityAlgo

# ============================================================================
# Input Validation Tests
# ============================================================================


def test_fuzzy_match_keys_non_dict_input():
    """Test TypeError when first argument is not a dictionary"""
    with pytest.raises(TypeError, match="First argument must be a dictionary"):
        fuzzy_match_keys("not a dict", ["key1"])


def test_fuzzy_match_keys_none_keys():
    """Test TypeError when keys argument is None"""
    with pytest.raises(TypeError, match="Keys argument cannot be None"):
        fuzzy_match_keys({"key1": "value1"}, None)


def test_fuzzy_match_keys_invalid_threshold_below():
    """Test ValueError for similarity_threshold below 0.0"""
    with pytest.raises(ValueError, match=r"similarity_threshold must be between 0.0 and 1.0"):
        fuzzy_match_keys({"key1": "value1"}, ["key1"], similarity_threshold=-0.1)


def test_fuzzy_match_keys_invalid_threshold_above():
    """Test ValueError for similarity_threshold above 1.0"""
    with pytest.raises(ValueError, match=r"similarity_threshold must be between 0.0 and 1.0"):
        fuzzy_match_keys({"key1": "value1"}, ["key1"], similarity_threshold=1.5)


# ============================================================================
# Similarity Algorithm Tests
# ============================================================================


def test_fuzzy_match_keys_similarity_algo_enum():
    """Test with SimilarityAlgo enum"""
    d = {"naem": "John", "age": 30}
    keys = ["name", "age"]
    result = fuzzy_match_keys(
        d,
        keys,
        similarity_algo=SimilarityAlgo.JARO_WINKLER,
        fuzzy_match=True,
    )
    assert "name" in result
    assert result["name"] == "John"
    assert result["age"] == 30


# ============================================================================
# Handle Unmatched Tests
# ============================================================================


def test_fuzzy_match_keys_ignore_unmatched_no_fuzzy_match():
    """Test handle_unmatched=IGNORE with unmatched keys during fuzzy match"""
    d = {"exact_match": "value1", "no_match_xyz": "value2"}
    keys = ["exact_match", "expected_key"]
    result = fuzzy_match_keys(
        d,
        keys,
        fuzzy_match=True,
        handle_unmatched=HandleUnmatched.IGNORE,
        similarity_threshold=0.95,  # High threshold to prevent fuzzy matching
    )
    # exact_match should be in result
    assert result["exact_match"] == "value1"
    # no_match_xyz should also be in result (IGNORE keeps unmatched input keys)
    assert result["no_match_xyz"] == "value2"


def test_fuzzy_match_keys_fill_with_unmatched_input():
    """Test handle_unmatched=FILL keeps unmatched input keys"""
    d = {"key1": "value1", "extra_key": "extra_value"}
    keys = ["key1", "key2"]  # key2 is expected but missing
    result = fuzzy_match_keys(
        d,
        keys,
        handle_unmatched=HandleUnmatched.FILL,
        fill_value="default",
        fuzzy_match=False,  # Disable fuzzy matching for clarity
    )
    # key1 should match exactly
    assert result["key1"] == "value1"
    # key2 should be filled with default
    assert result["key2"] == "default"
    # extra_key should be kept (FILL keeps unmatched input keys)
    assert result["extra_key"] == "extra_value"


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_fuzzy_match_keys_more_input_than_expected():
    """Test break when more input keys than expected during fuzzy match"""
    # More input keys than expected keys - should trigger break
    d = {"key1": "v1", "key2": "v2", "key3": "v3", "key4": "v4"}
    keys = ["key1", "key2"]  # Only 2 expected keys, but 4 input keys
    result = fuzzy_match_keys(
        d,
        keys,
        fuzzy_match=True,
        handle_unmatched=HandleUnmatched.IGNORE,
    )
    # Should match the expected keys
    assert result["key1"] == "v1"
    assert result["key2"] == "v2"
    # Unmatched input keys should be kept (IGNORE mode)
    assert result["key3"] == "v3"
    assert result["key4"] == "v4"


# ============================================================================
# Additional Integration Tests for Completeness
# ============================================================================


def test_fuzzy_match_keys_basic_exact_match():
    """Test basic exact key matching"""
    d = {"name": "John", "age": 30}
    keys = ["name", "age"]
    result = fuzzy_match_keys(d, keys)
    assert result == {"name": "John", "age": 30}


def test_fuzzy_match_keys_basic_fuzzy_match():
    """Test basic fuzzy key matching"""
    d = {"naem": "John", "age": 30}
    keys = ["name", "age"]
    result = fuzzy_match_keys(d, keys, fuzzy_match=True)
    assert result["name"] == "John"
    assert result["age"] == 30


def test_fuzzy_match_keys_empty_keys():
    """Test with empty keys list returns copy of dict"""
    d = {"key1": "value1", "key2": "value2"}
    result = fuzzy_match_keys(d, [])
    assert result == d
    assert result is not d  # Should be a copy


def test_fuzzy_match_keys_handle_unmatched_raise():
    """Test handle_unmatched=RAISE raises ValueError"""
    d = {"key1": "value1", "extra_key": "value2"}
    keys = ["key1"]
    with pytest.raises(ValueError, match="Unmatched keys found"):
        fuzzy_match_keys(d, keys, handle_unmatched=HandleUnmatched.RAISE, fuzzy_match=False)


def test_fuzzy_match_keys_handle_unmatched_force():
    """Test handle_unmatched=FORCE fills missing and removes unmatched"""
    d = {"key1": "value1", "extra_key": "value2"}
    keys = ["key1", "key2"]
    result = fuzzy_match_keys(
        d,
        keys,
        handle_unmatched=HandleUnmatched.FORCE,
        fill_value="default",
        fuzzy_match=False,
    )
    assert result["key1"] == "value1"
    assert result["key2"] == "default"
    assert "extra_key" not in result


def test_fuzzy_match_keys_strict_mode():
    """Test strict mode raises when expected keys missing"""
    d = {"key1": "value1"}
    keys = ["key1", "key2"]
    with pytest.raises(ValueError, match="Missing required keys"):
        fuzzy_match_keys(d, keys, strict=True, fuzzy_match=False)


def test_fuzzy_match_keys_fill_mapping():
    """Test fill_mapping provides custom defaults"""
    d = {"key1": "value1"}
    keys = ["key1", "key2", "key3"]
    result = fuzzy_match_keys(
        d,
        keys,
        handle_unmatched=HandleUnmatched.FILL,
        fill_mapping={"key2": "custom_value"},
        fill_value="default",
        fuzzy_match=False,
    )
    assert result["key1"] == "value1"
    assert result["key2"] == "custom_value"
    assert result["key3"] == "default"


# ============================================================================
# Coverage tests for iterable keys
# ============================================================================


def test_fuzzy_match_keys_with_generator():
    """Test keys as generator (iterable, not list or dict)"""
    d = {"name": "John", "age": 30}

    def key_generator():
        yield "name"
        yield "age"

    result = fuzzy_match_keys(d, key_generator())
    assert result["name"] == "John"
    assert result["age"] == 30


def test_fuzzy_match_keys_with_tuple():
    """Test keys as tuple (iterable, not list)"""
    d = {"name": "John", "age": 30}
    keys = ("name", "age")
    result = fuzzy_match_keys(d, keys)
    assert result["name"] == "John"
    assert result["age"] == 30


def test_fuzzy_match_keys_with_set():
    """Test keys as set (iterable, not list or dict)"""
    d = {"name": "John", "age": 30}
    keys = {"name", "age"}
    result = fuzzy_match_keys(d, keys)
    assert "name" in result
    assert "age" in result
